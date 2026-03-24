# Usage

### Model Preparation

Download models from https://huggingface.co/anonymous-release/vLLM_alm

### Environment

Clone https://anonymous.4open.science/r/vllm-3681

```
cd vllm

conda create -n vllm python=3.11

conda activate vllm

pip install uv pip==24.0

VLLM_USE_PRECOMPILED=1 uv pip install --editable .

pip install vllm[audio]
```

