# SpeechLM vLLM Usage Guide

SpeechLM is a Qwen3-8B based multimodal speech-language model with 8 parallel codec streams, served via a modified vLLM with OpenAI-compatible API.

## Model

- **Model path**: `speechlm-qwen3-8b/`
- **Architecture**: Qwen3-8B backbone + Whisper audio encoder + Xcodec decoder
- **Vocab**: `[0,256)` special tokens, `[256,152192)` text tokens, `[152192,160392)` audio codec tokens (8 streams x 1025)

## Server Setup

### Single GPU (1x)

```bash
bash scripts/serve_cfg_1.sh
```

Default: port `9000`, TP=1, `max-num-seqs=1024`.

### 4-GPU Tensor Parallel

```bash
bash scripts/serve_cfg.sh
```

Default: port `9001`, TP=4, `max-num-seqs=2048`.

### Custom Options

Both scripts accept extra vLLM arguments:

```bash
bash scripts/serve_cfg.sh --port 8080 --max-model-len 8192
```

### Key Server Parameters

| Parameter | 1-GPU | 4-GPU | Description |
|---|---|---|---|
| `--port` | 9000 | 9001 | API port |
| `--tensor-parallel-size` | 1 | 4 | Number of GPUs |
| `--max-num-seqs` | 1024 | 2048 | Max concurrent sequences |
| `--max-model-len` | 16384 | 16384 | Max sequence length |
| `--gpu-memory-utilization` | 0.90 | 0.90 | GPU memory fraction |
| `--enable-prefix-caching` | yes | yes | KV cache reuse |

No special server flags are needed for CFG. CFG is triggered per-request by the client.

## Generation Modes

### 1. Text-Only (default)

Standard text generation. The model defaults to text-only when no `vllm_xargs` is provided. Use `stop_token_ids: [3]` to stop at `<|eot|>`.

**API payload**:
```json
{
  "model": "speechlm-qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
  ],
  "max_tokens": 4096,
  "temperature": 0.6,
  "top_k": 20,
  "stop_token_ids": [3]
}
```

**Client script**:
```bash
python scripts/client.py --port 9001
python scripts/client.py --input /path/to/input.jsonl --output /path/to/output.jsonl
```

### 2. Text+Audio (no CFG)

Generates text first, then synthesizes audio. The model goes through three phases internally: text -> transition (`<|eot|>` -> `<|assistant|>`) -> audio codec generation until `<|eos|>`.

**API payload**:
```json
{
  "model": "speechlm-qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Read this aloud in a calm voice."}
  ],
  "max_tokens": 12000,
  "temperature": 0.8,
  "top_k": 20,
  "vllm_xargs": {
    "mode": "text_audio",
    "phase": "text",
    "text_temperature": 0.6,
    "audio_temperature": 0.8,
    "audio_topk": 20
  }
}
```

**Response** includes both text and audio:
```json
{
  "choices": [{
    "message": {
      "content": "Generated text...",
      "audio": {"data": "<base64-wav>"}
    },
    "finish_reason": "stop"
  }]
}
```

### 3. Text+Audio with CFG (Classifier-Free Guidance)

Same as text+audio, but with CFG for improved audio quality. The server auto-creates a shadow request internally — the client just passes `"cfg": N` where N > 1.

**API payload** (add `"cfg"` to vllm_xargs):
```json
{
  "model": "speechlm-qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Read this aloud in a calm voice."}
  ],
  "max_tokens": 12000,
  "temperature": 0.8,
  "top_k": 20,
  "vllm_xargs": {
    "mode": "text_audio",
    "phase": "text",
    "text_temperature": 0.6,
    "audio_temperature": 0.8,
    "audio_topk": 20,
    "cfg": 3.0
  }
}
```

**CFG formula**: `logits = main_logits * cfg + shadow_logits * (1 - cfg)`

- `cfg=1.0` is equivalent to no guidance (same as mode 2)
- `cfg=3.0` is a good default for improved audio quality
- Higher values give stronger guidance but may reduce diversity

### 4. Audio Understanding (MMAU)

For audio understanding tasks (e.g., audio QA), send audio as base64-encoded input. The model processes audio via Whisper encoder and generates text responses. No `vllm_xargs` needed.

**API payload with audio input**:
```json
{
  "model": "speechlm-qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are an audio understanding assistant."},
    {"role": "user", "content": [
      {"type": "input_audio", "input_audio": {"data": "<base64-audio>", "format": "wav"}}
    ]},
    {"role": "user", "content": "What sound is in this audio?"}
  ],
  "max_tokens": 4096,
  "temperature": 0.6,
  "stop_token_ids": [3]
}
```

## vllm_xargs Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | str | — | `"text_audio"` for audio generation |
| `phase` | str | — | Always `"text"` (starting phase) |
| `text_temperature` | float | 0.6 | Temperature for text token sampling |
| `audio_temperature` | float | 0.8 | Temperature for audio token sampling |
| `audio_topk` | int | 20 | Top-k for audio token sampling |
| `cfg` | float | 1.0 | CFG guidance strength (>1 enables CFG) |

## Client Scripts

### `client.py` — Text-only / MMAU evaluation

Sends text-only or audio understanding requests. Default input: MMAU dialogue test set.

```bash
# Default (MMAU evaluation on 4-GPU server)
python scripts/client.py

# Custom
python scripts/client.py --port 9000 --input data.jsonl --output results.jsonl --concurrency 128
```

### `client_audiogen_cfg.py` — Audio generation with CFG

Sends text_audio requests with configurable CFG strength. Saves random WAV samples.

```bash
# Default (1000 examples, cfg=3.0)
python scripts/client_audiogen_cfg.py

# No CFG baseline
python scripts/client_audiogen_cfg.py --cfg 1.0 --save-dir test/hard_nocfg

# Custom port
python scripts/client_audiogen_cfg.py --port 9000 --concurrency 128
```

### `client_all.py` — Combined stress test (1-GPU)

Sends text-only + audiogen + audiogen_cfg ALL concurrently. 128 connections per task type (384 total). Port `9000`.

```bash
python scripts/client_all.py
python scripts/client_all.py --concurrency 64 --cfg 2.0
```

### `client_all_4gpu.py` — Combined stress test (4-GPU)

Same as `client_all.py` but for 4-GPU server. 512 connections per task type (1536 total). Port `9001`.

```bash
python scripts/client_all_4gpu.py
python scripts/client_all_4gpu.py --concurrency 256
```

## Input Data Format

Input JSONL files use ESPnet format. Each line is a JSON object:

```json
{
  "example_id": "unique_id",
  "messages": [
    ["system", "text", "System prompt here."],
    ["user", "audio", "/path/to/audio.wav"],
    ["user", "text", "User question here."],
    ["assistant", "text", "Ground truth (skipped for audiogen)."],
    ["assistant", "audio", "/path/to/ground_truth.flac"]
  ]
}
```

- Each message is a tuple: `[role, modality, content]`
- `modality` is `"text"` or `"audio"`
- For audio modality, `content` is a file path to the audio file
- For audiogen clients, `assistant` messages are skipped (ground truth)
- For text-only/MMAU clients, all messages are sent

## Quick Start

```bash
# 1. Start server (4-GPU)
bash scripts/serve_cfg.sh

# 2. Wait for "Uvicorn running" in logs

# 3. Run audio generation with CFG
python scripts/client_audiogen_cfg.py --port 9001

# 4. Check output
ls test/hard/           # random WAV samples
cat test/output/audiogen_cfg3.jsonl | head -5

# 5. Run stress test (all modes combined)
python scripts/client_all_4gpu.py
```

## Troubleshooting

- **Client timeouts**: Default client timeout is 3600s. Long audio sequences (12000 tokens) can take minutes under heavy load.
- **OOM**: Reduce `--max-num-seqs` or `--gpu-memory-utilization`.
- **CFG doubles memory per request**: Each CFG request creates a shadow request internally. Effective batch size is 2x for CFG requests.
- **`MM cache hit rate: 100%`**: This is the multimodal (Whisper) encoder cache. 100% is expected when there are no audio inputs (TTS-only workloads).
