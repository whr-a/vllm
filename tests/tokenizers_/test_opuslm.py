# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import BatchEncoding

from vllm.tokenizers.opuslm import OpusLMTokenizer


class _DummyTokenizer:
    def __init__(self) -> None:
        self.all_special_ids = [99]
        self.all_special_tokens = ["<dummy_special>"]
        self.bos_token_id = 11
        self.eos_token_id = 12
        self.pad_token_id = 0
        self.is_fast = True
        self.vocab_size = 100
        self.max_token_id = 99
        self.max_chars_per_token = 8
        self.truncation_side = "left"

    def num_special_tokens_to_add(self) -> int:
        return 0

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        del text, text_pair, add_special_tokens, truncation, max_length, kwargs
        return BatchEncoding(data={"input_ids": [1, 2, 3]})

    def get_vocab(self) -> dict[str, int]:
        return {"a": 1, "b": 2}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<extra>": 77}

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del text, truncation, max_length, add_special_tokens
        return [10, 20]

    def apply_chat_template(
        self,
        messages,
        tools=None,
        **kwargs,
    ) -> str | list[int]:
        del messages, tools
        if kwargs.get("tokenize", True):
            return [4, 5]
        return "dummy-template"

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            if tokens.startswith("T"):
                return int(tokens[1:])
            return 7
        return [self.convert_tokens_to_ids(t) for t in tokens]  # type: ignore

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return ",".join(str(i) for i in ids)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        del skip_special_tokens
        return [f"T{i}" for i in ids]


def _make_tokenizer() -> OpusLMTokenizer:
    return OpusLMTokenizer(
        tokenizer=_DummyTokenizer(),  # type: ignore[arg-type]
        text_token_offset=100,
        text_token_end=300,
        pad_token_id=0,
        eos_token_id=5,
        codec_ssl_start_end_token_id=34,
        text_bpe_start_end_token_id=35,
    )


def test_opuslm_tokenizer_shifts_encode_and_call():
    tokenizer = _make_tokenizer()
    assert tokenizer.encode("hello") == [110, 120]

    encoded = tokenizer("hello")
    assert encoded["input_ids"] == [101, 102, 103]


def test_opuslm_tokenizer_decode_skips_non_text_specials():
    tokenizer = _make_tokenizer()
    # text ids are in [100, 300), others are Opus special/modality ids.
    decoded = tokenizer.decode([5, 110, 34, 120, 35], skip_special_tokens=True)
    assert decoded == "10,20"

    tokens = tokenizer.convert_ids_to_tokens(
        [5, 110, 120, 34], skip_special_tokens=True
    )
    assert tokens == ["T10", "T20"]


def test_opuslm_tokenizer_decode_without_skip_includes_markers():
    tokenizer = _make_tokenizer()
    decoded = tokenizer.decode([5, 110, 34, 120, 35], skip_special_tokens=False)
    assert decoded == "<sos/eos>10<codec_ssl_start/end>20<text_bpe_start/end>"


def test_opuslm_tokenizer_apply_chat_template_and_special_ids():
    tokenizer = _make_tokenizer()

    tokenized = tokenizer.apply_chat_template([], tokenize=True)
    assert tokenized == [104, 105]
    assert tokenizer.apply_chat_template([], tokenize=False) == "dummy-template"

    assert tokenizer.convert_tokens_to_ids("<codec_ssl_start/end>") == 34
    assert tokenizer.convert_tokens_to_ids("T10") == 110
