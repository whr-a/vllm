# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.opuslm import (
    OpusLMForConditionalGeneration,
    OpusLMMultiModalProcessor,
)


def _make_opuslm_stub() -> OpusLMForConditionalGeneration:
    model = object.__new__(OpusLMForConditionalGeneration)
    model.config = SimpleNamespace(
        codec_token_start=5256,
        codec_per_stream_size=1024,
        num_codec_streams=8,
    )
    return model


def test_delay_deinterleave_matches_reference_slicing():
    model = _make_opuslm_stub()
    codes = torch.arange(1 * 12 * 9, dtype=torch.long).view(1, 12, 9)

    aligned = model._delay_deinterleave(codes)
    # T_original = 12 - 9 + 1 = 4
    assert aligned.shape == (1, 4, 9)
    for stream_idx in range(9):
        expected = codes[:, stream_idx:stream_idx + 4, stream_idx]
        assert torch.equal(aligned[:, :, stream_idx], expected)


def test_global_to_dac_codebook_offsets_and_clamps():
    model = _make_opuslm_stub()
    cfg = model.config

    # [B=1, T=2, S=8] global DAC ids. Some values are intentionally out-of-range
    # to verify clamping to [0, 1023].
    dac_tokens = torch.tensor(
        [[
            [cfg.codec_token_start + s * cfg.codec_per_stream_size + 7
             for s in range(cfg.num_codec_streams)],
            [cfg.codec_token_start + s * cfg.codec_per_stream_size + 5000
             for s in range(cfg.num_codec_streams)],
        ]],
        dtype=torch.long,
    )

    codebook = model._global_to_dac_codebook(dac_tokens)
    assert codebook.shape == dac_tokens.shape
    assert torch.all(codebook[:, 0, :] == 7)
    assert torch.all(codebook[:, 1, :] == 1023)


def test_resolve_task_token_id_defaults_and_overrides():
    cfg = SimpleNamespace(
        textlm_task_token_id=64,
        codec_ssl_asr_task_token_id=80,
        codec_ssl_tts_task_token_id=81,
        codec_ssl_plain_tts_task_token_id=82,
    )

    assert OpusLMMultiModalProcessor.resolve_task_token_id(
        cfg,
        has_audio_input=True,
    ) == 80
    assert OpusLMMultiModalProcessor.resolve_task_token_id(
        cfg,
        has_audio_input=False,
    ) == 82
    assert OpusLMMultiModalProcessor.resolve_task_token_id(
        cfg,
        has_audio_input=False,
        mode="text_text",
    ) == 64
    assert OpusLMMultiModalProcessor.resolve_task_token_id(
        cfg,
        has_audio_input=False,
        task="tts",
    ) == 81


def test_resolve_task_token_id_rejects_invalid_mode():
    cfg = SimpleNamespace(
        textlm_task_token_id=64,
        codec_ssl_asr_task_token_id=80,
        codec_ssl_tts_task_token_id=81,
        codec_ssl_plain_tts_task_token_id=82,
    )

    with pytest.raises(ValueError):
        OpusLMMultiModalProcessor.resolve_task_token_id(
            cfg,
            has_audio_input=False,
            mode="audio_audio",
        )
