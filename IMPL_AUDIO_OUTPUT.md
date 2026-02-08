# SpeechLM Audio Output — 实现规格书

> 给新窗口用的 compact spec。所有需要改的文件、改什么、怎么改，全在这里。
> 详细设计见 `plan_audiogen.md`，现有代码见 `speechlm.py` / `configs/speechlm.py`。
> 原始 ESPnet 参考代码在 `/mnt/home/haoranw4-andr-49167f/work/espnet/espnet2/speechlm/`。
> 推理细节文档在 `/mnt/home/haoranw4-andr-49167f/work/espnet/espnet2/speechlm/claude_qwen_inform.md`。

---

## 背景

SpeechLM 是基于 Qwen3-8B 的多模态语音语言模型，已适配到 vLLM。
**text 生成已验证通过**。现在要补全 **audio 输出**的端到端 pipeline。

模型每步生成 8 个并行 codec stream token。vLLM 只看到 stream 0（由 vLLM sampler 采样），
stream 1-7 在模型内部 `_sample_and_buffer_streams()` 中采样并缓存于 `_stream_buffer`。
当前问题：stream 1-7 每步被覆盖，没有跨步积累；没有 Xcodec 解码；没有 base64 输出。

**不做 CFG**（cfg=1）。

---

## 要改的文件清单（共 6 个文件）

### 1. `vllm/transformers_utils/configs/speechlm.py` — Config 增加字段

在 `SpeechLMConfig` 类中增加：
```python
audio_temperature: float = 0.8
audio_topk: int = 20
text_temperature: float = 0.6
text_topk: int = 20
xcodec_hf_model_tag: str = "hf-audio/xcodec-hubert-general"
xcodec_sample_rate: int = 16000
```

### 2. `speechlm-qwen3-8b/config.json` — 增加对应字段

加入上述 6 个字段的 JSON 表示。

### 3. `vllm/v1/worker/gpu_model_runner.py` — 传 request_id 到模型（加 1 行）

在调用 `model.forward()` 之前，把当前 batch 的 request_id 列表设到模型上：
```python
model._current_batch_req_ids = [self.input_batch._req_ids[i] for i in range(num_reqs)]
```
需要找到正确的位置（在 `execute_model` 或 `_execute_model_run` 方法中，forward 调用前）。

### 4. `vllm/model_executor/models/speechlm.py` — 主要改动（~150 行新增）

#### 4a. `__init__` 增加属性
```python
# 推理参数
self._audio_temperature = config.audio_temperature    # 0.8
self._audio_topk = config.audio_topk                  # 20
self._text_temperature = config.text_temperature      # 0.6
self._vllm_temperature = 0.8  # 由 Generator 在 generate 前设置

# Per-request stream 1-7 history
self._stream17_history: dict[str, list[torch.Tensor]] = {}
self._current_batch_req_ids: list[str] = []

# text_only 模式
self._text_only = False

# Xcodec 解码器（frozen，从 HF pretrained 加载）
from transformers import XcodecModel
self.xcodec_model = XcodecModel.from_pretrained(config.xcodec_hf_model_tag).eval()
# 注意：需要在权重加载后移到正确的 device 上，或在首次 decode 时处理
```

#### 4b. `_build_masks()` 增加 text_only mask
```python
text_only_modality_mask = torch.ones(V, dtype=torch.bool)
text_only_modality_mask[config.text_token_id] = False  # 只允许 <|text|>=7
self.register_buffer("text_only_modality_mask", text_only_modality_mask)
```

#### 4c. `compute_logits()` 修改
三处改动：

**① text_only 模态检测**：当 `is_detect.any()` 时，根据 `self._text_only` 选择 mask：
- `_text_only=True` → 用 `text_only_modality_mask`（只允许 `<|text|>`）
- `_text_only=False` → 用 `modality_mask`（允许 `<|text|>` 和 `<|audio|>`）

**② stream 1-7 history 收集**：在 `_sample_and_buffer_streams` 之后不需要额外逻辑，
收集逻辑放在 `_sample_and_buffer_streams` 内部。

**③ 温度补偿**：return `stream0_logits` 之前：
```python
if is_text.any() and self._vllm_temperature > 0 and self._text_temperature > 0:
    text_idx = is_text.nonzero(as_tuple=True)[0]
    scale = self._vllm_temperature / self._text_temperature
    stream0_logits[text_idx] = stream0_logits[text_idx] * scale
```

#### 4d. `_sample_and_buffer_streams()` 末尾追加 history
```python
# 现有代码末尾 self._stream_buffer = new_buffer 之后：
for i in audio_idx:
    req_id = self._current_batch_req_ids[i]
    self._stream17_history.setdefault(req_id, []).append(
        new_buffer[i].clone()
    )
```

#### 4e. `_top_k_sample()` 调用处改用 config 参数
在 `_sample_and_buffer_streams` 中调用 `_top_k_sample` 时：
```python
sampled = self._top_k_sample(s_logits, temperature=self._audio_temperature, top_k=self._audio_topk)
```

#### 4f. 新增方法：`set_text_only`, `reset_audio_collection`
```python
def set_text_only(self, value: bool):
    self._text_only = value

def reset_audio_collection(self, req_id: str | None = None):
    if req_id is not None:
        self._stream17_history.pop(req_id, None)
    else:
        self._stream17_history.clear()
    self._stream_buffer = None
```

#### 4g. 新增方法：audio 解码 pipeline（4 个方法）

**`_delay_deinterleave(codes)`**：
- 输入 `[1, T, 8]` delay-interleaved → 输出 `[1, T-7, 8]` aligned
- 参考 ESPnet `_apply_delay_deinterleave`：对每个 stream n 取 `codes[:, n:n+T_original, n]`

**`_global_to_codebook(full_matrix)`**：
- 输入 `[1, T, 8]` 全局 token ID → 输出 `[1, T, 8]` codebook index [0,1023]
- 每个 stream s：`offset = codec_base_offset + s * codec_layer_size + 1`，`clamp(token - offset, 0, 1023)`

**`_xcodec_decode(codebook_indices)`**：
- 输入 `[1, T, 8]` codebook indices → permute 为 `[1, 8, T]` → `xcodec_model.decode().audio_values`
- 返回 numpy array

**`decode_audio_from_tokens(req_id, stream0_codec_tokens)`**：
- 用 `pop(req_id)` 取 stream 1-7 history
- 对齐裁剪/补齐
- 拼接 `[N, 8]` → `_global_to_codebook` → `_delay_deinterleave` → `_xcodec_decode`
- 返回 `(audio_numpy, sample_rate)`

### 5. 新增 `scripts/speechlm_generator.py` — 封装类（~150 行）

```python
class SpeechLMGenerator:
    def __init__(self, model_path, **llm_kwargs)
    def generate(self, prompt, audio_data=None, text_only=False, max_tokens=4096,
                 audio_temperature=0.8, audio_topk=20,
                 text_temperature=0.6, text_topk=20) -> dict
    # 返回 {"text": str|None, "audio": str|None (base64 WAV), "finish_reason": str, "token_ids": list}
```

内部流程：
1. `model.reset_audio_collection()` + `model.set_text_only(text_only)`
2. 设置 `model._vllm_temperature = audio_temperature`，`model._text_temperature = text_temperature`
3. 构建 `SamplingParams`：
   - text_only: `temperature=text_temperature, top_k=text_topk, stop_token_ids=[2, 3]`
   - auto: `temperature=audio_temperature, top_k=audio_topk, stop_token_ids=[2]`
4. `llm.generate()` → 拿 token_ids + request_id
5. `_postprocess(req_id, token_ids)`：
   - 找 `<|eot|>(3)` 分割 text/audio 段
   - text 段：过滤 special tokens → `tokenizer.decode()`
   - audio 段：过滤出 codec 范围 tokens → `model.decode_audio_from_tokens(req_id, ...)` → base64 WAV

### 6. 新增 `test/test_audio_generation.py` — 端到端测试

用 `SpeechLMGenerator` 跑一个 audio-in → text+audio-out 的 case，
验证 text 非空、audio base64 能解码成 WAV、WAV 时长合理。

---

## 关键技术点速查

| 问题 | 答案 |
|------|------|
| Xcodec 来源 | HF pretrained `hf-audio/xcodec-hubert-general`，训练时 frozen |
| Xcodec decode 接口 | `model.decode(codes).audio_values`，codes shape `[B, 8, T]` |
| Delay de-interleave | `codes[:, n:n+T_orig, n]` for n in 0..7，T_orig = T - 7 |
| Token → codebook | `clamp(token - (152192 + s*1025 + 1), 0, 1023)` |
| 模态检测 | `compute_logits` 中 argmax 判断：7=text, 8=audio, ≥152192=codec |
| 段分隔 | `<|eot|>(3)` 是段分隔（text 结束继续 audio），`<|eos|>(2)` 是终止 |
| 温度补偿 | text 模式 logits 乘 `T_vllm / T_text` 后返回给 vLLM sampler |
| History 管理 | `dict[req_id → list[Tensor[7]]]`，decode 时 `pop` 取出并清理 |
