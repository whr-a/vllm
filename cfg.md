# vLLM CFG 逻辑总览（当前仓库实现）

本文总结当前 `vllm_latest` 里 CFG（Classifier-Free Guidance）在服务入口、调度器、模型执行与输出阶段的完整实现链路。

## 1. 先说结论

这个实现不是“单请求内两路前向”，而是“**主请求 + shadow 请求**”双请求方案：

- 当 `vllm_xargs.cfg > 1` 时，Engine 自动复制出一个 shadow request（无条件分支）。
- Scheduler 把 main/shadow 当成绑定对调度与抢占。
- SpeechLM 仅在 **audio phase** 做 CFG 融合（`main * cfg + shadow * (1 - cfg)`）。
- stream0 和 stream1~7 都会做 CFG 融合，但 stream1~7 只对 main 采样，shadow 复制 main 的结果。

## 2. 参数入口：`vllm_xargs -> SamplingParams.extra_args`

OpenAI 协议层把 `vllm_xargs` 直接塞入 `SamplingParams.extra_args`，后续所有 CFG 逻辑都从这里读：

- `vllm/entrypoints/openai/chat_completion/protocol.py`
- `vllm/entrypoints/openai/completion/protocol.py`
- `vllm/entrypoints/openai/responses/protocol.py`
- `vllm/entrypoints/openai/speech_to_text/protocol.py`

你在请求里传的键（比如 `cfg/audio_temperature/audio_topk/text_temperature/mode`）都会走这条链路。

## 3. Engine Core：自动创建 shadow 请求

在 `vllm/v1/engine/core.py` 里：

1. `add_request` 检查 `request.sampling_params.extra_args['cfg']`。
2. 若 `cfg > 1`，调用 `_create_cfg_shadow(main)` 自动构造 shadow。
3. 建立 main/shadow 双向绑定：
   - `main.cfg_shadow_id = shadow.request_id`
   - `shadow.cfg_main_id = main.request_id`
4. 生成共享 `cfg_group_id`，同时写入二者的 `extra_args`。
5. 调度顺序是 **先 shadow 后 main** 放入 scheduler。

shadow 的关键特征：

- `request_id = f"{main.request_id}-shadow"`
- `prompt_token_ids = [0] * len(main.prompt_token_ids)`
- `extra_args['is_shadow'] = True`

也就是 shadow 天生是“零条件分支”。

## 4. Request 对象的 CFG 字段

`vllm/v1/request.py` 中，Request 额外挂了：

- `cfg_group_id`
- `cfg_shadow_id`（main -> shadow）
- `cfg_main_id`（shadow -> main）

这样 scheduler 能 O(1) 找配对伙伴，不用扫描全局。

## 5. Scheduler：main/shadow 绑定调度与绑定回收

核心文件：`vllm/v1/core/sched/scheduler.py`

### 5.1 RUNNING 队列

- shadow（`cfg_main_id is not None`）不独立调度，直接跳过。
- main 调度成功后会立即尝试给 shadow 分配 KV block。
- 若 shadow 分配失败，会回滚 main 并把这一对一起 preempt。

### 5.2 WAITING 队列

- shadow 不独立出队，等待被 main 捎带。
- main+shadow 需要 2 个 running slot；空位不够则先跳过这对。
- main 入 running 后再捎带 shadow；若 shadow 失败，main 回滚并重排队。

### 5.3 抢占与终止

- 任一方被 preempt，会尝试连带 preempt 对方。
- `finish_requests` 会自动扩展到 partner（abort/finish 都是成对）。
- 有“自动停伙伴”逻辑：一方自然结束时，另一方会被 stop/free，避免孤儿请求卡住 running 槽位。
- `_cfg_groups` 在 add/free 时维护，避免 group 信息泄漏。

## 6. GPU Model Runner：把请求态同步给 SpeechLM + shadow token 覆写

核心文件：`vllm/v1/worker/gpu_model_runner.py`

### 6.1 前向前同步

在 forward 前：

- `SpeechLM._current_batch_req_ids` 会被设置为当前 batch 的请求 ID 顺序。
- 首次见到的请求会把 `sampling_params.extra_args` 拷进 `SpeechLM._per_req_config`。

这一步是 SpeechLM 能做“按请求配置采样/相位控制/CFG 分组”的基础。

### 6.2 采样后 shadow token override

bookkeeping 阶段会按 `cfg_group_id` 找 main/shadow，并强制 shadow 的 token：

- 若 main 在 `audio` phase：shadow token = main token
- 否则（text/transition）：shadow token = `0`（pad）

这保证 shadow 的上下文行为符合“无条件分支”设计。

### 6.3 音频输出只看 main

stream0 codec token 的历史记录和最终解码（base64 wav）会跳过 shadow，只对 main 做。

### 6.4 请求结束清理

请求结束时调用 `SpeechLM.cleanup_request(req_id)`，清理：

- `_per_req_config`
- `_stream_buffer_dict`
- `_stream0_history`
- `_stream17_history`
- `_decoded_audio`

### 6.5 为什么 shadow 不会返回给客户端

`vllm/v1/engine/output_processor.py` 在处理 `EngineCoreOutput` 时，会先用
`request_id` 查 `request_states`；查不到就直接 `continue`。

shadow 是 Engine 内部请求，不会作为外部请求注册到 output processor，所以它的输出会被忽略，只保留 main 的可见输出。

## 7. SpeechLM 内部：CFG 真正发生的地方

核心文件：`vllm/model_executor/models/speechlm.py`

### 7.1 相位机（text -> transition -> audio）

`_update_text_audio_phase` 逻辑：

- 仅在 pure decode（`len(input_ids) == len(batch_req_ids)`）时更新相位。
- main 请求：
  - `text` 相位读到 `eot` -> `transition`
  - `transition` 相位读到 `<|assistant|>` -> `audio`
- shadow 请求：
  - 不靠自己 token 判相位
  - 通过 `cfg_group_id` 从 main 同步相位

### 7.2 stream0 logits 里的 CFG

`compute_logits` 里会：

1. 基于 `cfg_group_id + is_shadow` 构造 batch 内 main/shadow 配对。
2. 仅当 `main.phase == 'audio'` 且 `cfg > 1` 时加入 `cfg_pairs`。
3. 保存 main/shadow 的 raw logits（未 mask 前）。
4. 做 detect/text/audio 模式判定与各自 mask。
5. 最后对 cfg_pair 做融合：
   - `final = main_raw * cfg + shadow_raw * (1 - cfg)`

注意：detect 位置不会做 CFG merge（避免把 `<|audio|>` 这类模式切换 token 抹掉）。

### 7.3 stream1~7 logits 里的 CFG

`_sample_and_buffer_streams` 里：

- 只对 main audio 位置采样 stream1~7。
- 若该 main 有 shadow，则每个 stream 都会额外算 shadow logits 并按同样公式融合。
- 采样完把 main 的 1~7 buffer 复制给 shadow（为下一步 embedding 对齐）。

所以结论是：**stream1~7 也算 CFG logits，不是只有 stream0 算**。

### 7.4 每请求采样参数

`_get_audio_sampling_groups` 会按每个请求的 `(audio_temperature, audio_topk)` 分组采样：

- 优先读 `req_cfg`（也就是 `vllm_xargs` 下发值）
- 无值时退回模型默认值

这就是“优先 `vllm_xargs`”在 stream1~7 上的落地点。

此外，text phase 有一段 temperature compensation，用于把 vLLM 全局 temperature 与 text/audio 两套温度对齐。

## 8. 为什么要把 stream0 和 stream1~7 分开看

设计上它们职责不同：

- stream0：外部自回归主 token（客户端看到的序列），负责文本 token、模式 token、以及音频主码流 token。
- stream1~7：仅在 audio phase 内部生成的附加码流，不直接作为外部主序列输出。

文本生成阶段没有 8 路都采样，只是 stream0 在工作；进入 audio phase 后才会触发 1~7 采样与缓存。

## 9. 文本到音频的“连接点”

连接不是一次跳变，而是 2 步：

1. `text` 相位生成 `eot`，相位变 `transition`
2. `transition` 相位强制输出 `<|assistant|>`，随后相位变 `audio`

进入 `audio` 后，detect 位置允许 `<|text|>/<|audio|>` 模式 token，再进入 codec token 区间，1~7 采样和 CFG merge 才全面生效。

## 10. 为什么 CFG 压测时会越跑越慢（你日志里的现象）

从当前实现看，更像“资源饱和+调度退化”而不是死锁：

- CFG 把一个请求变成 main+shadow，两份 KV、两份 running 槽位压力。
- KV 接近满载时，main/shadow 绑定调度更容易触发回滚/连带 preempt。
- 1~7 stream 的额外 logits/sampling 让单步计算成本上升（尤其 audio 长段）。
- 结果是 `GPU KV cache ~100%` 时，`running` 下降、`waiting` 堆积、吞吐持续下滑。

这和你看到的“KV 常满但 tok/s 越来越低”是吻合的。

## 11. 一页纸数据流（便于排障）

1. Client 传 `vllm_xargs`  
2. OpenAI protocol 写入 `SamplingParams.extra_args`  
3. EngineCore 检测 `cfg>1`，创建 shadow + 绑定 group  
4. Scheduler 成对调度/成对 preempt/成对 finish  
5. ModelRunner 同步 `_per_req_config`，并做 shadow token override  
6. SpeechLM 在 audio phase 做 stream0 + stream1~7 的 CFG merge  
7. 输出阶段只对 main 产出可见结果，shadow 仅内部服务  
8. finish 时清理两侧状态

---

如果你后面要继续做性能定位，优先看这几个观测点是否同步变化：

- `running/waiting` 与 `GPU KV cache usage`
- scheduler 里 CFG rollback/preempt 次数
- audio phase 占比（是否长期停在 audio phase）
- 单步 `compute_logits + _sample_and_buffer_streams` 耗时
