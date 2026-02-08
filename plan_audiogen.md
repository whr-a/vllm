# SpeechLM Audio Output 实现计划 (v2)

> 目标：在现有 SpeechLM vLLM 适配基础上，补全音频输出的端到端 pipeline。
> 模型生成 audio codec tokens → Xcodec 解码 → base64 WAV → 通过 API 返回。
> 输出最多包含一段 text + 一段 audio，分别在不同 key 中返回。
>
> 前置条件：text 生成已验证通过；不考虑 CFG（cfg=1）。

---

## 0. 当前状态回顾

**已实现**（per-step 机制都在 `speechlm.py`）：
- 模态检测：`compute_logits` 中通过 argmax 分类 detect/text/audio
- Stream 0 mask：modality/text/audio 三种 mask 正确切换
- Stream 1-7 内部采样：`_sample_and_buffer_streams` 用 top-k 采样
- Stream buffer：`_stream_buffer [N, 7]` 供下一步 embedding 求和
- Multi-stream embedding：`_apply_stream_embeddings` 把 stream 0-7 的 embedding 求和

**本次计划要做的**：
1. 跨步 stream 1-7 token 的收集（history 积累）
2. Delay de-interleave 后处理
3. Xcodec 解码器集成（HF pretrained，已确认训练时 frozen）
4. Text vs Audio 不同推理参数的支持
5. text_only 模式（enforce 纯文本输出）
6. 多段输出（text + audio）的拆分与解码
7. `SpeechLMGenerator` 封装类，返回 `{"text": ..., "audio": base64}`
8. 端到端测试

---

## 1. 输出格式设计

### 1.1 模型生成的 token 序列

ESPnet 支持多段生成。在 vLLM 中，一次 `generate()` 产出一个连续 token 序列。
根据模型行为，输出有三种模式：

**模式 A：text + audio**
```
<|text|> text_tok_1 ... text_tok_N <|eot|> <|audio|> codec_s0_1 ... codec_s0_M <|eos|>
```
- `<|eot|>(3)` 是段分隔符（text 结束，继续下一段）
- `<|eos|>(2)` 是终止符（全部结束）

**模式 B：text only**
```
<|text|> text_tok_1 ... text_tok_N <|eos|>
```

**模式 C：audio only**（较少见）
```
<|audio|> codec_s0_1 ... codec_s0_M <|eos|>
```

### 1.2 段分隔逻辑

在 ESPnet 中，`<|eot|>` 表示当前段结束但对话继续，`<|eos|>` 表示全部结束。
转换到 vLLM：
- **auto 模式**：`stop_token_ids=[2]`（仅 `<|eos|>` 停止），`<|eot|>` 不停止、模型继续生成
- **text_only 模式**：`stop_token_ids=[2, 3]`（`<|eos|>` 和 `<|eot|>` 都停止）

### 1.3 返回格式

```python
{
    "text": "解码后的文本字符串",      # str | None
    "audio": "base64编码的WAV数据",    # str | None  (base64, 16kHz, PCM16)
    "finish_reason": "stop",           # str
    "token_ids": [7, 500, ..., 2],     # list[int]  原始 token 序列
}
```

**约束**：每次推理最多一段 text + 一段 audio。
- 只有 text → `{"text": "...", "audio": None}`
- 只有 audio → `{"text": None, "audio": "base64..."}`
- text + audio → `{"text": "...", "audio": "base64..."}`

---

## 2. `SpeechLMGenerator` 封装类

### 2.1 为什么需要封装

vLLM 的 `llm.generate()` 返回 `RequestOutput`，只有 `text` 和 `token_ids`。
音频解码需要：
1. 访问模型内部的 `_stream17_history`（stream 1-7 tokens）
2. 运行 Xcodec 解码器
3. 做 base64 编码

这些都不在 vLLM 标准 output pipeline 中。用封装类把这些逻辑收拢，对外提供干净 API。

### 2.2 接口设计

```python
class SpeechLMGenerator:
    """SpeechLM 推理封装：text + audio 输出，base64 WAV 返回。"""

    def __init__(self, model_path: str, **llm_kwargs):
        """
        Args:
            model_path: speechlm-qwen3-8b/ 目录路径
            **llm_kwargs: 传给 vLLM LLM() 的参数
                如 max_model_len, enforce_eager, gpu_memory_utilization 等
        """
        self.llm = LLM(model=model_path, trust_remote_code=True, **llm_kwargs)
        self._model = self._get_model_ref()
        self._tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompt: str,
        audio_data: tuple | None = None,  # (numpy_array, sample_rate)
        text_only: bool = False,
        max_tokens: int = 4096,
        audio_temperature: float = 0.8,
        audio_topk: int = 20,
        text_temperature: float = 0.6,
        text_topk: int = 20,
    ) -> dict:
        """生成文本和/或音频。

        Args:
            prompt: 文本 prompt（如果含 <|audio|> placeholder，需配合 audio_data）
            audio_data: 音频输入 (numpy_array, sample_rate)，可为 None
            text_only: True 则强制纯文本输出
            max_tokens: 最大生成 token 数
            audio/text_temperature/topk: 各模态的采样参数

        Returns:
            {
                "text": str | None,
                "audio": str | None,     # base64 WAV
                "finish_reason": str,
                "token_ids": list[int],
            }
        """

    def _get_model_ref(self):
        """获取底层模型引用（单 GPU offline inference）。"""
        return (self.llm.llm_engine.model_executor
                .driver_worker.model_runner.model)
```

### 2.3 `generate()` 内部流程

```python
def generate(self, prompt, audio_data=None, text_only=False, ...):
    # 1. 设置模型状态
    self._model.reset_audio_collection()
    self._model.set_text_only(text_only)

    # 2. 确定 sampling params
    #    text_only: stop at eos(2) AND eot(3), use text temperature
    #    auto:      stop at eos(2) only, use audio temperature
    #    (因为 auto 模式下模型可能先输出 text 再输出 audio，
    #     text 段用模型自身能力控制质量，audio 段需要高 temperature)
    if text_only:
        sampling_params = SamplingParams(
            temperature=text_temperature,
            top_k=text_topk,
            max_tokens=max_tokens,
            stop_token_ids=[2, 3],
        )
    else:
        sampling_params = SamplingParams(
            temperature=audio_temperature,
            top_k=audio_topk,
            max_tokens=max_tokens,
            stop_token_ids=[2],
        )

    # 3. 构建输入
    inputs = [{"prompt": prompt}]
    if audio_data is not None:
        inputs[0]["multi_modal_data"] = {"audio": audio_data}

    # 4. 调用 vLLM generate
    outputs = self.llm.generate(inputs, sampling_params)
    token_ids = list(outputs[0].outputs[0].token_ids)

    # 5. 后处理：拆分 text 和 audio 段
    result = self._postprocess(token_ids)
    return result
```

### 2.4 `_postprocess()` 后处理逻辑

```python
def _postprocess(self, token_ids: list[int]) -> dict:
    cfg = self._model.config
    EOT, EOS = cfg.eot_token_id, cfg.eos_token_id
    TEXT_ID, AUDIO_ID = cfg.text_token_id, cfg.audio_token_id

    # 找 <|eot|> 分割点
    eot_pos = None
    for i, t in enumerate(token_ids):
        if t == EOT:
            eot_pos = i
            break

    text_result = None
    audio_result = None

    # req_id 从 vLLM output 获取
    req_id = ...  # outputs[i].request_id

    if eot_pos is not None:
        # text + audio 模式
        text_segment = token_ids[:eot_pos]       # 不含 <|eot|>
        audio_segment = token_ids[eot_pos + 1:]  # <|eot|> 之后
        text_result = self._decode_text_segment(text_segment)
        audio_result = self._decode_audio_segment(req_id, audio_segment)
    else:
        # 单段模式：判断是 text 还是 audio
        if len(token_ids) > 0 and token_ids[0] == AUDIO_ID:
            audio_result = self._decode_audio_segment(req_id, token_ids)
        else:
            text_result = self._decode_text_segment(token_ids)

    return {
        "text": text_result,
        "audio": audio_result,
        "finish_reason": "stop",
        "token_ids": token_ids,
    }
```

### 2.5 `_decode_text_segment()`

```python
def _decode_text_segment(self, tokens: list[int]) -> str | None:
    cfg = self._model.config
    # 移除 <|text|> 前缀和 <|eos|> 后缀
    tokens = [t for t in tokens
              if t != cfg.text_token_id and t != cfg.eos_token_id
              and t != cfg.eot_token_id]
    if not tokens:
        return None
    return self._tokenizer.decode(tokens, skip_special_tokens=True)
```

### 2.6 `_decode_audio_segment()`

```python
def _decode_audio_segment(self, req_id: str, tokens: list[int]) -> str | None:
    cfg = self._model.config
    # 移除 <|audio|> 前缀和 <|eos|> 后缀
    codec_tokens = [t for t in tokens
                    if cfg.codec_base_offset <= t < cfg.vocab_size]
    if not codec_tokens:
        return None

    # 调用模型方法做 Xcodec 解码（用 req_id 查找对应的 stream 1-7 history）
    audio_np, sr = self._model.decode_audio_from_tokens(req_id, codec_tokens)

    # 转 base64 WAV
    import io, base64, soundfile
    buf = io.BytesIO()
    soundfile.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")
```

---

## 3. text_only 模式

### 3.1 需求

两种模式：
- **auto**（默认）：模型自行决定输出 text、audio 或 text+audio
- **text_only**：强制全部输出为 text，不生成 audio

### 3.2 实现方式

在模型上设置一个标志位：

```python
# SpeechLMForConditionalGeneration 中
self._text_only = False

def set_text_only(self, value: bool):
    self._text_only = value
```

在 `compute_logits` 的模态检测逻辑中：

```python
# 当 is_detect 为 True 且 _text_only 时，只允许 <|text|>
if is_detect.any() and self._text_only:
    idx = is_detect.nonzero(as_tuple=True)[0]
    # 用一个只允许 text_token_id 的 mask（不允许 audio_token_id）
    stream0_logits[idx] = stream0_logits[idx].masked_fill(
        self.text_only_modality_mask.unsqueeze(0), float("-inf")
    )
else:
    # 正常 modality_mask（允许 text 和 audio）
    stream0_logits[idx] = stream0_logits[idx].masked_fill(
        self.modality_mask.unsqueeze(0), float("-inf")
    )
```

`_build_masks` 中新增：
```python
# text_only modality mask: 只允许 <|text|>=7
text_only_modality_mask = torch.ones(V, dtype=torch.bool)
text_only_modality_mask[config.text_token_id] = False
self.register_buffer("text_only_modality_mask", text_only_modality_mask)
```

### 3.3 配合 stop_token_ids

| 模式 | 模态 mask | stop_token_ids | 效果 |
|------|----------|----------------|------|
| auto | 允许 text+audio | `[2]` (eos) | 模型可生成 text→eot→audio→eos |
| text_only | 只允许 text | `[2, 3]` (eos+eot) | 模型只能生成 text→eos 或 text→eot(停) |

---

## 4. Text vs Audio 推理参数

### 4.1 参数定义

| 参数 | Text 模式 | Audio 模式 |
|------|-----------|-----------|
| temperature | 0.6 | 0.8 |
| top_k | 20 | 20 |

### 4.2 参数传递

**Stream 0**：由 vLLM sampler 控制，参数来自 `SamplingParams`。
**Stream 1-7**：由模型内部 `_top_k_sample` 控制。

在 `SpeechLMConfig` 中增加默认值：
```python
audio_temperature: float = 0.8
audio_topk: int = 20
text_temperature: float = 0.6
text_topk: int = 20
```

模型从 config 读取 stream 1-7 的采样参数：
```python
self._audio_temperature = config.audio_temperature
self._audio_topk = config.audio_topk
```

### 4.3 auto 模式下的 temperature 补偿

**问题**：auto 模式传给 vLLM sampler 的是 `temperature=0.8`（为 audio 准备），但 text 段的 stream 0 理应用 0.6。

**解决**：在 `compute_logits` 返回 logits 之前，根据当前模态做温度补偿：

```
vLLM sampler 会做:  logits / T_vllm → softmax → sample
我们想要:          logits / T_desired → softmax → sample
所以返回前乘:      logits × (T_vllm / T_desired)
vLLM 再除 T_vllm → 等价于 logits / T_desired  ✓
```

```python
# compute_logits 末尾，return stream0_logits 之前：
if is_text.any() and self._vllm_temperature > 0:
    text_idx = is_text.nonzero(as_tuple=True)[0]
    scale = self._vllm_temperature / self._text_temperature  # 0.8/0.6 = 1.333
    stream0_logits[text_idx] = stream0_logits[text_idx] * scale
```

`self._vllm_temperature` 由 `SpeechLMGenerator` 在 generate 前设置到模型上：
```python
self._model._vllm_temperature = audio_temperature  # 0.8
```

这样 text 段和 audio 段的 stream 0 各自用到精确的 temperature。

---

## 5. 跨步 Stream 1-7 Token 收集（per-request）

### 5.1 核心问题

`compute_logits` 收到 `hidden_states: [N, hidden_size]`，N 是 batch 内所有 request 的总和。
模型采样了 stream 1-7，但不知道第 i 行属于哪个 request。
多个 request 同时生成 audio 时，history 会混在一起。

### 5.2 解决方案：用 request_id 做 key

把 `_stream17_history` 从 `list` 改为 `dict[str, list[Tensor]]`，用 request_id 索引。

**让模型知道 request_id**：在 model runner 调 `forward()` 前，把当前 batch 的
request_id 列表设到模型属性上（model runner 本身维护了完整的 `req_id_to_index` 映射）。

#### model runner 改动（1 行）

```python
# vllm/v1/worker/gpu_model_runner.py 中，调 model.forward() 前：
model._current_batch_req_ids = [
    self.input_batch._req_ids[i] for i in range(num_reqs)
]
```

#### 模型侧改动

```python
# __init__ 中：
self._stream17_history: dict[str, list[torch.Tensor]] = {}
self._current_batch_req_ids: list[str] = []

# _sample_and_buffer_streams 末尾：
for i in audio_idx:
    req_id = self._current_batch_req_ids[i]
    self._stream17_history.setdefault(req_id, []).append(
        new_buffer[i].clone()  # [7]
    )

# 新增方法：
def reset_audio_collection(self, req_id: str | None = None):
    """清空指定 request（或全部）的 history。"""
    if req_id is not None:
        self._stream17_history.pop(req_id, None)
    else:
        self._stream17_history.clear()
    self._stream_buffer = None
```

### 5.3 step-by-step 对齐分析

以某个 request 的 text+audio 输出为例：

```
vLLM output token_ids:
  [<|text|>, t1, t2, ..., tN, <|eot|>, <|audio|>, c0, c1, ..., cM, <|eos|>]
   ↑ text segment                       ↑ audio segment

compute_logits 中该 request 位置 is_audio == True 的步骤：
  c0, c1, ..., cM 这 M 步（<|audio|> 和 <|eos|> 步不触发）

_stream17_history[req_id] 收集了 M 条记录，与 c0..cM 一一对应。
```

多个 request 并发时，每个 request 的 history 独立存储，互不干扰。

### 5.4 边界对齐保护

在后处理中，以实际 codec token 数量为准裁剪 history：

```python
def decode_audio_from_tokens(self, req_id: str, stream0_codec_tokens: list[int]):
    history = self._stream17_history.pop(req_id, [])  # pop: 取出并清理
    n_codec = len(stream0_codec_tokens)
    # 裁剪到实际 codec 步数（防御 argmax vs sampled 不一致的边界情况）
    history = history[:n_codec]
    if len(history) < n_codec:
        # 不够则用 pad 补齐（降级，不应出现）
        pad = torch.zeros(7, dtype=torch.long, device=self.stream_emb.weight.device)
        history.extend([pad] * (n_codec - len(history)))
    ...
```

### 5.5 内存管理

- `decode_audio_from_tokens` 用 `pop` 取出 history，解码后自动释放
- 对于异常终止（request 被 cancel）的情况，可定期清理 `_stream17_history` 中过期条目
- 正常流程下不会内存泄漏

---

## 6. Delay De-interleave

直接参考 ESPnet `_apply_delay_deinterleave`：

```python
def _delay_deinterleave(self, codes: torch.Tensor) -> torch.Tensor:
    """[1, T, 8] delay-interleaved → [1, T-7, 8] aligned."""
    _, T, N = codes.size()
    T_original = T - N + 1
    new_codes = []
    for n in range(N):
        new_codes.append(codes[:, n : n + T_original, n])
    return torch.stack(new_codes, dim=-1)
```

---

## 7. Xcodec 解码器集成

### 7.1 加载

训练时 Xcodec 是 frozen 的，直接用 HF pretrained：

```python
# __init__ 中：
from transformers import XcodecModel
self.xcodec_model = XcodecModel.from_pretrained(
    config.xcodec_hf_model_tag  # "hf-audio/xcodec-hubert-general"
).to(device).eval()
```

在 `__init__` 中加载，与主模型同 GPU，额外 ~200MB 显存。

### 7.2 Token → Codebook Index

```python
def _global_to_codebook(self, full_matrix: torch.Tensor) -> torch.Tensor:
    """[1, T, 8] global token IDs → [1, T, 8] codebook indices [0, 1023]."""
    cfg = self.config
    result = full_matrix.clone()
    for s in range(cfg.num_stream):
        offset = cfg.codec_base_offset + s * cfg.codec_layer_size + 1
        result[..., s] = torch.clamp(result[..., s] - offset, min=0, max=1023)
    return result
```

### 7.3 Xcodec 解码

```python
def _xcodec_decode(self, codebook_indices: torch.Tensor) -> torch.Tensor:
    """[1, T, 8] codebook indices → [num_samples] audio numpy."""
    codes = codebook_indices.permute(0, 2, 1)  # [1, 8, T]
    with torch.no_grad():
        audio = self.xcodec_model.decode(codes).audio_values  # [1, 1, samples]
    return audio[0, 0].cpu().numpy()
```

### 7.4 端到端 decode 方法

```python
def decode_audio_from_tokens(
    self, req_id: str, stream0_codec_tokens: list[int]
) -> tuple:
    """从 stream 0 codec tokens + 该 request 的 stream17_history 重建音频。

    Args:
        req_id: 请求 ID，用于查找对应的 stream 1-7 history
        stream0_codec_tokens: vLLM 输出中提取的 stream 0 codec token list

    Returns: (audio_numpy, sample_rate)
    """
    cfg = self.config
    device = self.stream_emb.weight.device
    n = len(stream0_codec_tokens)

    # 1. 取出并清理该 request 的 stream 1-7 history
    history = self._stream17_history.pop(req_id, [])
    history = history[:n]
    if len(history) < n:
        pad = torch.zeros(7, dtype=torch.long, device=device)
        history.extend([pad] * (n - len(history)))

    # 2. 拼接 [N, 8] full token matrix
    s0 = torch.tensor(stream0_codec_tokens, dtype=torch.long, device=device)
    s17 = torch.stack(history, dim=0).to(device)  # [N, 7]
    full = torch.cat([s0.unsqueeze(1), s17], dim=1)  # [N, 8]
    full = full.unsqueeze(0)  # [1, N, 8]

    # 3. Global → codebook index
    codebook = self._global_to_codebook(full)

    # 4. Delay de-interleave
    aligned = self._delay_deinterleave(codebook)  # [1, N-7, 8]

    # 5. Xcodec decode
    audio_np = self._xcodec_decode(aligned)
    return audio_np, cfg.xcodec_sample_rate
```

---

## 8. 具体代码改动清单

### 8.1 `vllm/transformers_utils/configs/speechlm.py`

`SpeechLMConfig` 增加字段：
- `audio_temperature: float = 0.8`
- `audio_topk: int = 20`
- `text_temperature: float = 0.6`
- `text_topk: int = 20`
- `xcodec_hf_model_tag: str = "hf-audio/xcodec-hubert-general"`
- `xcodec_sample_rate: int = 16000`

### 8.2 `speechlm-qwen3-8b/config.json`

增加对应的新字段。

### 8.3 `vllm/v1/worker/gpu_model_runner.py`

在调用 `model.forward()` 前增加 1 行，把当前 batch 的 request_id 列表设到模型上：

```python
model._current_batch_req_ids = [self.input_batch._req_ids[i] for i in range(num_reqs)]
```

### 8.4 `vllm/model_executor/models/speechlm.py`

| 改动类型 | 方法/属性 | 说明 |
|----------|----------|------|
| 新增属性 | `self.xcodec_model` | HF pretrained XcodecModel |
| 新增属性 | `self._stream17_history` | `dict[str, list[Tensor]]`，per-request 收集 |
| 新增属性 | `self._current_batch_req_ids` | `list[str]`，由 model runner 每步设置 |
| 新增属性 | `self._text_only` | bool |
| 新增属性 | `self._audio_temperature/topk` | 从 config 读取 |
| 新增属性 | `self._vllm_temperature` | 由 Generator 设置，用于温度补偿 |
| 新增属性 | `self._text_temperature` | 从 config 读取 |
| 新增 buffer | `self.text_only_modality_mask` | 只允许 `<\|text\|>` 的 mask |
| 修改 | `_build_masks()` | 增加 text_only_modality_mask |
| 修改 | `compute_logits()` | text_only 逻辑 + history 管理 + 温度补偿 |
| 修改 | `_sample_and_buffer_streams()` | 末尾按 req_id 追加 history |
| 修改 | `_top_k_sample()` 调用处 | 使用 config 中的参数 |
| 新增方法 | `set_text_only(bool)` | 设置 text_only 标志 |
| 新增方法 | `reset_audio_collection(req_id)` | 清空指定/全部 request 的 history |
| 新增方法 | `decode_audio_from_tokens(req_id, tokens)` | 端到端音频解码 |
| 新增方法 | `_global_to_codebook()` | token ID → codebook index |
| 新增方法 | `_delay_deinterleave()` | delay pattern 还原 |
| 新增方法 | `_xcodec_decode()` | Xcodec waveform 解码 |

### 8.5 新增 `speechlm_generator.py`（与 speechlm.py 同目录或 scripts/）

`SpeechLMGenerator` 封装类，~150 行：
- `__init__`：创建 LLM，获取 model 引用
- `generate()`：完整推理 + 后处理
- `_postprocess()`：拆分 text/audio 段
- `_decode_text_segment()`：text token → string
- `_decode_audio_segment()`：codec token → base64 WAV

### 8.6 新增测试脚本 `test/test_audio_generation.py`

端到端音频生成 + 验证。

---

## 9. 实现顺序

1. **Config 改动** — `configs/speechlm.py` + `config.json` 增加字段
2. **text_only mask** — `_build_masks` 增加 `text_only_modality_mask`
3. **text_only 逻辑** — `compute_logits` 中增加 `_text_only` 判断
4. **参数化 internal sampling** — `_top_k_sample` 使用 config 参数
5. **model runner 改动** — `gpu_model_runner.py` 增加 1 行传 req_id
6. **Stream 1-7 per-request history 收集** — `_sample_and_buffer_streams` 按 req_id 存储
7. **温度补偿** — `compute_logits` 中按模态调整 logits 缩放
8. **Xcodec 集成** — `__init__` 加载 + `_xcodec_decode` + `_global_to_codebook`
9. **Delay de-interleave** — `_delay_deinterleave` 方法
10. **`decode_audio_from_tokens`** — 端到端 model 方法
11. **`SpeechLMGenerator`** — 封装类
12. **测试脚本**

---

## 10. 已知限制

| 限制 | 原因 | 后续可做 |
|------|------|----------|
| 无 CFG (cfg=1) | 需要双 KV cache | 独立实现 |
| Xcodec 额外 ~200MB 显存 | 始终加载 | 可改懒加载 |
| model runner 需要 1 行改动 | 传 req_id 给模型 | 后续可做成 vLLM 原生 hook |
| 不支持 streaming audio 输出 | 需完整 token 序列才能 Xcodec decode | 可考虑分段解码 |
