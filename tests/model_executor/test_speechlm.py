from types import SimpleNamespace

import torch

from vllm.model_executor.models.speechlm import (
    SpeechLMForConditionalGeneration,
)


def _make_speechlm_stub() -> SpeechLMForConditionalGeneration:
    model = object.__new__(SpeechLMForConditionalGeneration)
    model.config = SimpleNamespace(
        audio_temperature=0.8,
        audio_topk=20,
        vocab_size=100,
    )
    model._current_batch_req_ids = ["r0", "r1", "r2", "r3", "r4"]
    model._per_req_config = {}
    return model


def test_audio_sampling_groups_prioritize_request_xargs():
    model = _make_speechlm_stub()
    model._per_req_config = {
        "r0": {"audio_temperature": 0.2, "audio_topk": 5},
        "r1": {"audio_temperature": 1.1},
        "r2": {"audio_topk": 999},
        "r3": {"audio_topk": -1},
    }

    groups = model._get_audio_sampling_groups(
        [0, 1, 2, 3, 4], torch.device("cpu")
    )
    grouped_rows = {(temp, top_k): rows.tolist() for temp, top_k, rows in groups}

    assert grouped_rows[(0.2, 5)] == [0]
    assert grouped_rows[(1.1, 20)] == [1]
    # 999 is clamped to vocab_size=100, and -1 means "full vocab".
    assert grouped_rows[(0.8, 100)] == [2, 3]
    assert grouped_rows[(0.8, 20)] == [4]


def test_audio_sampling_param_normalization():
    assert (
        SpeechLMForConditionalGeneration._normalize_audio_temperature(
            "0.5", 0.8
        )
        == 0.5
    )
    assert (
        SpeechLMForConditionalGeneration._normalize_audio_temperature(
            "bad", 0.8
        )
        == 0.8
    )
    assert (
        SpeechLMForConditionalGeneration._normalize_audio_temperature(
            -0.1, 0.8
        )
        == 0.8
    )

    assert (
        SpeechLMForConditionalGeneration._normalize_audio_top_k(
            "7", 20, 100
        )
        == 7
    )
    assert (
        SpeechLMForConditionalGeneration._normalize_audio_top_k(
            "999", 20, 100
        )
        == 100
    )
    assert (
        SpeechLMForConditionalGeneration._normalize_audio_top_k(
            "-1", 20, 100
        )
        == 100
    )
    assert (
        SpeechLMForConditionalGeneration._normalize_audio_top_k(
            "bad", 20, 100
        )
        == 20
    )
