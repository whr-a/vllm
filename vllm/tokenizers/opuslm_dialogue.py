# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer wrapper for OpusLM Dialogue global token layout.

OpusLM Dialogue reserves low token IDs for non-text modalities and special markers:
  [0, 13448) are not plain text BPE IDs.

Text BPE IDs from the base SmolLM-1.7B tokenizer are shifted by +text_token_offset
when encoding, and shifted back when decoding.

Dialogue-specific special tokens (IDs 8-11, 88-89) are also handled.
"""

from pathlib import Path
from typing import Any, overload

from transformers import BatchEncoding

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .hf import CachedHfTokenizer
from .protocol import TokenizerLike


class OpusLMDialogueTokenizer(CachedHfTokenizer):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TokenizerLike":
        text_token_offset = int(kwargs.pop("opuslm_dialogue_text_token_offset", 13448))
        text_token_end = int(kwargs.pop("opuslm_dialogue_text_token_end", 62600))
        pad_token_id = int(kwargs.pop("opuslm_dialogue_pad_token_id", 0))
        eos_token_id = int(kwargs.pop("opuslm_dialogue_eos_token_id", 5))
        codec_ssl_start_end_token_id = int(
            kwargs.pop("opuslm_dialogue_codec_ssl_start_end_token_id", 34)
        )
        text_bpe_start_end_token_id = int(
            kwargs.pop("opuslm_dialogue_text_bpe_start_end_token_id", 35)
        )

        tokenizer = super().from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
        return OpusLMDialogueTokenizer(
            tokenizer=tokenizer,
            text_token_offset=text_token_offset,
            text_token_end=text_token_end,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            codec_ssl_start_end_token_id=codec_ssl_start_end_token_id,
            text_bpe_start_end_token_id=text_bpe_start_end_token_id,
        )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        text_token_offset: int,
        text_token_end: int,
        pad_token_id: int,
        eos_token_id: int,
        codec_ssl_start_end_token_id: int,
        text_bpe_start_end_token_id: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.name_or_path = getattr(tokenizer, "name_or_path", "")

        self.text_token_offset = int(text_token_offset)
        self.text_token_end = int(text_token_end)

        self._pad_token_id = int(pad_token_id)
        self._eos_token_id = int(eos_token_id)
        self._bos_token_id = int(eos_token_id)
        self._codec_ssl_start_end_token_id = int(codec_ssl_start_end_token_id)
        self._text_bpe_start_end_token_id = int(text_bpe_start_end_token_id)

        self._special_id_to_token: dict[int, str] = {
            self._pad_token_id: "<pad>",
            self._eos_token_id: "<sos/eos>",
            8: "<system_prompt>",
            9: "<user_input>",
            10: "<assistant_output>",
            11: "<eou>",
            self._codec_ssl_start_end_token_id: "<codec_ssl_start/end>",
            self._text_bpe_start_end_token_id: "<text_bpe_start/end>",
            37: "<spk_start/end>",
            88: "<text_dialogue_task>",
            89: "<audio_dialogue_task>",
        }
        self._special_token_to_id = {v: k for k, v in self._special_id_to_token.items()}

        base_vocab = self.tokenizer.get_vocab()
        self._vocab = {tok: idx + self.text_token_offset for tok, idx in base_vocab.items()}
        for sid, stok in self._special_id_to_token.items():
            self._vocab.setdefault(stok, sid)

        base_added_vocab = self.tokenizer.get_added_vocab()
        self._added_vocab = {
            tok: idx + self.text_token_offset for tok, idx in base_added_vocab.items()
        }

        shifted_base_special_ids = {
            sid + self.text_token_offset for sid in self.tokenizer.all_special_ids
        }
        self._all_special_ids = sorted(
            set(self._special_id_to_token.keys()) | shifted_base_special_ids
        )
        self._all_special_tokens = list(
            dict.fromkeys(
                [*self._special_id_to_token.values(), *self.tokenizer.all_special_tokens]
            )
        )

    def _shift_ids_up(self, input_ids: Any) -> Any:
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [[int(t) + self.text_token_offset for t in row]
                        for row in input_ids]
            return [int(t) + self.text_token_offset for t in input_ids]
        try:
            return input_ids + self.text_token_offset
        except TypeError:
            return input_ids

    def _is_text_global_id(self, token_id: int) -> bool:
        return self.text_token_offset <= token_id < self.text_token_end

    def _to_base_id(self, token_id: int) -> int:
        return int(token_id) - self.text_token_offset

    def _decode_text_chunk(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        if not token_ids:
            return ""
        base_ids = [self._to_base_id(tid) for tid in token_ids]
        return self.tokenizer.decode(base_ids, skip_special_tokens=skip_special_tokens)

    def num_special_tokens_to_add(self) -> int:
        return self.tokenizer.num_special_tokens_to_add()

    @property
    def all_special_tokens(self) -> list[str]:
        return self._all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size + self.text_token_offset

    @property
    def max_token_id(self) -> int:
        return self.tokenizer.max_token_id + self.text_token_offset

    @property
    def max_chars_per_token(self) -> int:
        return self.tokenizer.max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return self.tokenizer.truncation_side

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        encoded = self.tokenizer(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )
        if "input_ids" in encoded:
            encoded["input_ids"] = self._shift_ids_up(encoded["input_ids"])
        return encoded

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()

    def get_added_vocab(self) -> dict[str, int]:
        return self._added_vocab.copy()

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        base_ids = self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return [tid + self.text_token_offset for tid in base_ids]

    def apply_chat_template(
        self,
        conversation: list["ChatCompletionMessageParam"] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        messages = conversation if conversation is not None else kwargs.pop("messages", [])
        out = self.tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
        if isinstance(out, list):
            return [int(tid) + self.text_token_offset for tid in out]
        return out

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            if tokens in self._special_token_to_id:
                return self._special_token_to_id[tokens]
            return int(self.tokenizer.convert_tokens_to_ids(tokens)) + self.text_token_offset

        out: list[int] = []
        for token in tokens:
            if token in self._special_token_to_id:
                out.append(self._special_token_to_id[token])
            else:
                out.append(
                    int(self.tokenizer.convert_tokens_to_ids(token))
                    + self.text_token_offset
                )
        return out

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        if not ids:
            return ""

        if skip_special_tokens:
            text_ids = [tid for tid in ids if self._is_text_global_id(int(tid))]
            return self._decode_text_chunk(text_ids, skip_special_tokens=True)

        parts: list[str] = []
        text_chunk: list[int] = []
        for tid in ids:
            tid = int(tid)
            if self._is_text_global_id(tid):
                text_chunk.append(tid)
                continue

            if text_chunk:
                parts.append(
                    self._decode_text_chunk(text_chunk, skip_special_tokens=False)
                )
                text_chunk.clear()

            parts.append(
                self._special_id_to_token.get(tid, f"<|opus_dialogue_special_{tid}|>")
            )

        if text_chunk:
            parts.append(self._decode_text_chunk(text_chunk, skip_special_tokens=False))

        return "".join(parts)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        out: list[str] = []
        for tid in ids:
            tid = int(tid)
            if self._is_text_global_id(tid):
                out.extend(
                    self.tokenizer.convert_ids_to_tokens(
                        [self._to_base_id(tid)],
                        skip_special_tokens=skip_special_tokens,
                    )
                )
                continue

            if skip_special_tokens:
                continue
            out.append(self._special_id_to_token.get(tid, f"<|opus_dialogue_special_{tid}|>"))
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)
