# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only SpeechLM model (Qwen3-8B based multimodal speech-language
model with 8-stream parallel codec output).

Features:
  - Audio input via Qwen3-Omni audio encoder + linear adaptor
  - Text output (standard autoregressive)
  - Audio output via 8 parallel codec streams with delay interleaving
  - Internal multi-stream sampling (streams 1-7 sampled inside compute_logits)
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen3OmniMoeAudioEncoder,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.speechlm import SpeechLMConfig

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_AUDIO_SAMPLING_RATE = 16000
_NUM_MEL_BINS = 128
_CHUNK_LENGTH = 30  # seconds


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _get_feat_extract_output_lengths(
    input_lengths: torch.Tensor,
    n_window: int = 100,
):
    """Compute audio encoder output lengths after CNN downsampling.

    The audio encoder splits input into chunks of ``n_window * 2`` frames,
    then applies 3 Conv2d layers with stride 2 to each chunk.  This function
    mirrors that logic so we know how many output tokens the encoder will
    produce for a given input length.

    Args:
        input_lengths: Per-audio input frame counts.
        n_window: The audio encoder's ``n_window`` config value.
            Default 100 (SpeechLM).  Qwen3-Omni uses 50.
    """
    chunk_size = n_window * 2  # frames per chunk

    # CNN output length for one full chunk (3 conv layers, stride 2 each)
    full_cnn = chunk_size
    for _ in range(3):
        full_cnn = (full_cnn - 1) // 2 + 1

    # Remainder frames that don't fill a complete chunk
    remainder = input_lengths % chunk_size

    # CNN output length for the remainder
    remainder_cnn = remainder
    for _ in range(3):
        remainder_cnn = (remainder_cnn - 1) // 2 + 1

    # Total = full_chunks * cnn_per_chunk + remainder_cnn
    # When remainder == 0, all chunks are full and remainder_cnn == 0,
    # which is correct because ceil(N/chunk_size) == N//chunk_size when
    # N is exactly divisible.
    output_lengths = (input_lengths // chunk_size) * full_cnn + remainder_cnn
    return output_lengths


# ---------------------------------------------------------------------------
# Minimal processor (no HuggingFace model hub dependency)
# ---------------------------------------------------------------------------
class _SpeechLMProcessor:
    """Lightweight processor for SpeechLM audio input.

    Wraps WhisperFeatureExtractor and provides the audio_token string.
    This avoids requiring a full HuggingFace processor in the model
    directory.
    """

    def __init__(self):
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=_NUM_MEL_BINS,
            sampling_rate=_AUDIO_SAMPLING_RATE,
            chunk_length=_CHUNK_LENGTH,
        )
        self.audio_token = "<|audio|>"


# ---------------------------------------------------------------------------
# Multimodal processing classes
# ---------------------------------------------------------------------------
class SpeechLMProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(SpeechLMConfig)

    def get_hf_processor(self, **kwargs: object) -> _SpeechLMProcessor:
        return _SpeechLMProcessor()

    def get_feature_extractor(
        self, **kwargs: object
    ) -> WhisperFeatureExtractor:
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return SpeechLMMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
        )


class SpeechLMDummyInputsBuilder(
    BaseDummyInputsBuilder[SpeechLMProcessingInfo]
):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        processor = self.info.get_hf_processor()
        return processor.audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        feature_extractor = self.info.get_feature_extractor()
        target_audio_length = (
            min(feature_extractor.chunk_length, 30)
            * feature_extractor.sampling_rate
        )
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


def _speechlm_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    audio_feature_lengths = hf_inputs.get(
        "audio_feature_lengths", torch.empty((0,))
    )
    return dict(
        input_audio_features=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_feature_lengths, dim=1
        ),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
    )


class SpeechLMMultiModalDataParser(MultiModalDataParser):

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={
                    "input_audio_features",
                    "audio_feature_lengths",
                },
                fields_factory=_speechlm_field_config,
            )
        return super()._parse_audio_data(data)


class SpeechLMMultiModalProcessor(
    BaseMultiModalProcessor[SpeechLMProcessingInfo],
):
    """Handles audio preprocessing and prompt token replacement."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process audio through WhisperFeatureExtractor."""
        import numpy as np

        processor = self.info.get_hf_processor()
        feature_extractor = processor.feature_extractor
        tokenizer = self.info.get_tokenizer()

        # Extract audio data
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        # Process audio features
        audio_inputs = {}
        if audios:
            # Pad to hop_length multiples
            hop_length = feature_extractor.hop_length
            padded_audios = []
            for audio in audios:
                if isinstance(audio, np.ndarray):
                    length = audio.shape[-1]
                    if length % hop_length != 0:
                        pad_len = hop_length - (length % hop_length)
                        audio = np.pad(audio, (0, pad_len))
                padded_audios.append(audio)

            audio_features = feature_extractor(
                padded_audios,
                sampling_rate=_AUDIO_SAMPLING_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_features = audio_features["input_features"]
            attention_mask = audio_features["attention_mask"]

            # Reshape from 3D (batch, mel_bins, max_frames) to 2D
            # (mel_bins, total_valid_frames) by flattening batch dimension
            # and removing padding via attention mask.
            # This matches the Qwen2_5OmniAudioFeatureInputs schema
            # which expects shape ('nmb', 'tsl').
            if attention_mask is not None:
                # (batch, mel, frames) -> (batch, frames, mel)
                # -> mask select -> (total_valid, mel)
                # -> transpose -> (mel, total_valid)
                input_features = input_features.permute(0, 2, 1)[
                    attention_mask.bool()
                ].permute(1, 0)
            else:
                # No mask: just flatten batch into time dimension
                # (batch, mel, frames) -> (mel, batch*frames)
                b, m, f = input_features.shape
                input_features = input_features.permute(
                    1, 0, 2
                ).reshape(m, b * f)

            audio_inputs["input_audio_features"] = input_features
            audio_inputs["feature_attention_mask"] = attention_mask
            audio_inputs["audio_feature_lengths"] = attention_mask.sum(-1)

        # Tokenize text
        text_inputs = tokenizer(prompt, return_tensors="pt")

        return BatchFeature(data={**text_inputs, **audio_inputs})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _speechlm_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        audio_token_id = vocab.get(audio_token, 8)  # fallback to ID 8

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")

        hf_config = self.info.get_hf_config()
        n_window = hf_config.audio_config.n_window

        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths, n_window=n_window
            )
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1), n_window=n_window
            )
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} is too short to be represented "
                    "inside the model"
                )
            # ESPnet keeps <|audio|> marker as embed(8) BEFORE audio
            # features: <|user|> embed(<|audio|>) [N features] <|eos|>
            # Position 0 = marker (regular embedding, not multimodal)
            # Positions 1..N = audio features (multimodal, replaced)
            tokens = [audio_token_id] * (num_features + 1)

            def _is_embed(
                _tokenizer: object, _full: object
            ) -> torch.Tensor:
                mask = torch.ones(num_features + 1, dtype=torch.bool)
                mask[0] = False  # keep <|audio|> marker as-is
                return mask

            return PromptUpdateDetails(full=tokens, is_embed=_is_embed)

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_audio,
            ),
        ]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
@MULTIMODAL_REGISTRY.register_processor(
    SpeechLMMultiModalProcessor,
    info=SpeechLMProcessingInfo,
    dummy_inputs=SpeechLMDummyInputsBuilder,
)
class SpeechLMForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """SpeechLM: Qwen3-8B based multimodal speech-language model.

    Supports audio/text input and text/audio output. Audio output uses
    8 parallel codec streams with delay interleaving. From vLLM's
    perspective, the model generates 1 token per step (stream 0);
    streams 1-7 are sampled internally and buffered.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # LLM body
            "model.layers.": "language_model.model.layers.",
            "model.norm.": "language_model.model.norm.",
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "lm_head.": "language_model.lm_head.",
            # Audio encoder
            "multimodal_io_dict.continuous_audio.model.audio_tower.":
                "audio_tower.",
            # Audio adaptor
            "adaptor.continuous_audio.": "audio_adaptor.",
            # Stream embedding (no prefix change)
            "stream_emb.": "stream_emb.",
            # Xcodec decoder
            "multimodal_io_dict.discrete_audio.codec_model.":
                "codec_decoder.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: SpeechLMConfig = vllm_config.model_config.hf_config
        self.config = config

        # --- Audio encoder (Qwen3-Omni, output_dim=2048) ---
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                config.audio_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        # --- Audio adaptor (2048 → 4096) ---
        self.audio_adaptor = nn.Linear(
            config.adaptor_input_dim, config.hidden_size, bias=True
        )

        # --- Language model (Qwen3-8B, vocab=160392) ---
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    config.text_config,
                    architectures=["Qwen3ForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        # --- Multi-stream components ---
        self.stream_emb = nn.Embedding(config.num_stream, config.hidden_size)

        # --- Pipeline parallelism ---
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # --- Internal decode state ---
        # Stream 1-7 token buffer from previous compute_logits call
        self._stream_buffer: torch.Tensor | None = None

        # --- Precompute masks ---
        self._build_masks(config)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------
    def _build_masks(self, config: SpeechLMConfig):
        """Precompute token validity masks for each modality and stream."""
        V = config.vocab_size

        # Modality detection mask: only <|text|>=7 and <|audio|>=8
        modality_mask = torch.ones(V, dtype=torch.bool)
        modality_mask[config.text_token_id] = False
        modality_mask[config.audio_token_id] = False
        self.register_buffer("modality_mask", modality_mask)

        # Text mask (stream 0): allow text range [256, 152192) + eos + eot
        text_mask_s0 = torch.ones(V, dtype=torch.bool)
        text_mask_s0[config.text_token_offset:config.text_token_end] = False
        text_mask_s0[config.eos_token_id] = False
        text_mask_s0[config.eot_token_id] = False
        self.register_buffer("text_mask_s0", text_mask_s0)

        # Audio masks: per-stream codec range masks
        # audio_masks[s] masks out all tokens EXCEPT the valid range for
        # stream s. Stream 0 also allows eos/eot.
        audio_masks = []
        for s in range(config.num_stream):
            mask = torch.ones(V, dtype=torch.bool)
            start = config.codec_base_offset + s * config.codec_layer_size
            end = start + config.codec_layer_size
            mask[start:end] = False  # allow this stream's codec range
            if s == 0:
                mask[config.eos_token_id] = False
                mask[config.eot_token_id] = False
            self.register_buffer(f"audio_mask_s{s}", mask)
            audio_masks.append(mask)
        self._audio_masks = audio_masks

    # ------------------------------------------------------------------
    # Modality detection from token IDs
    # ------------------------------------------------------------------
    def _classify_tokens(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classify each token as modality-detect / text / audio mode.

        Returns three boolean masks of shape [N], where exactly one is
        True per position.
        """
        cfg = self.config

        is_audio_codec = (input_ids >= cfg.codec_base_offset) & (
            input_ids < cfg.vocab_size
        )
        is_audio_start = input_ids == cfg.audio_token_id
        is_audio = is_audio_codec | is_audio_start

        is_text_range = (input_ids >= cfg.text_token_offset) & (
            input_ids < cfg.text_token_end
        )
        is_text_start = input_ids == cfg.text_token_id
        is_text = is_text_range | is_text_start

        # Modality detection: assistant token or anything not text/audio
        is_detect = ~is_audio & ~is_text

        return is_detect, is_text, is_audio

    # ------------------------------------------------------------------
    # Audio input processing
    # ------------------------------------------------------------------
    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Qwen2_5OmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None
        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key == "input_audio_features"
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = (
                    self._parse_and_validate_audio_input(**kwargs)
                )
        return mm_input_by_modality

    def _process_audio_input(
        self, audio_input: Qwen2_5OmniAudioFeatureInputs
    ) -> tuple[torch.Tensor, ...]:
        """Run audio through encoder + adaptor."""
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]
        n_window = self.config.audio_config.n_window
        audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths, n_window=n_window
        )

        audio_features = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        # Apply adaptor: [total_frames, 2048] → [total_frames, 4096]
        audio_features = self.audio_adaptor(audio_features)
        return audio_features.split(audio_output_lengths.tolist())

    # ------------------------------------------------------------------
    # Multimodal embedding interface
    # ------------------------------------------------------------------
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs
        )
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "audio" and multimodal_input is not None:
                audio_embeddings = self._process_audio_input(
                    multimodal_input
                )
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input tokens with multi-stream support for audio decode.

        During prefill: standard text embedding + multimodal merge.
        During decode (audio mode): embed stream 0 token + buffer streams
        1-7, sum across streams.
        """
        # Standard text token embedding
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        # Merge multimodal (audio) embeddings during prefill
        if multimodal_embeddings is not None and len(
            multimodal_embeddings
        ) > 0:
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        # Multi-stream embedding for audio decode tokens
        if self._stream_buffer is not None:
            _, _, is_audio = self._classify_tokens(input_ids)
            if is_audio.any():
                inputs_embeds = self._apply_stream_embeddings(
                    input_ids, inputs_embeds, is_audio
                )

        return inputs_embeds

    def _apply_stream_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        is_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Add stream 1-7 embeddings for audio-mode tokens during decode.

        For audio-mode positions, the embedding becomes:
            sum(embed_tokens(stream_k_token) for k in 0..7)
        where stream 0 is the current input_ids token and streams 1-7
        come from the internal buffer.
        """
        audio_indices = is_audio.nonzero(as_tuple=True)[0]
        if len(audio_indices) == 0:
            return inputs_embeds

        buf = self._stream_buffer
        embed_fn = self.language_model.model.embed_tokens

        # Get stream 1-7 tokens from buffer
        buf_size = min(len(audio_indices), buf.shape[0])
        if buf_size == 0:
            return inputs_embeds

        # Build [num_audio, 7] buffer slice for audio positions
        stream_tokens = buf[:buf_size]  # [buf_size, 7]

        # Embed stream 1-7 tokens
        stream_embeds = embed_fn(stream_tokens)  # [buf_size, 7, hidden]

        # Zero out padding positions (token == 0)
        pad_mask = (stream_tokens == 0).unsqueeze(-1)  # [buf_size, 7, 1]
        stream_embeds = stream_embeds.masked_fill(pad_mask, 0.0)

        # Sum stream 1-7 embeddings and add to inputs_embeds
        stream_sum = stream_embeds.sum(dim=1)  # [buf_size, hidden]
        actual_indices = audio_indices[:buf_size]
        inputs_embeds[actual_indices] = (
            inputs_embeds[actual_indices] + stream_sum
        )

        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    # ------------------------------------------------------------------
    # Multi-stream logits computation
    # ------------------------------------------------------------------
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits with multi-stream support.

        Mode detection uses the argmax of **unmasked** logits rather than
        input token IDs.  This is critical for correctness in batched /
        chunked-prefill inference: ``embed_input_ids`` receives all tokens
        from all requests concatenated, so a per-model-instance
        ``_decode_input_ids`` cannot be reliably aligned with the N
        sampled hidden-state positions that arrive here.

        The argmax approach works because the model is trained with
        disjoint token ranges for each mode:
          - detect: argmax is <|text|>(7) or <|audio|>(8)
          - audio:  argmax is a codec token [152192, 160392)
          - text:   everything else (text range, eos, eot, …)
        """
        cfg = self.config
        lm_head = self.language_model.lm_head
        logits_processor = self.language_model.logits_processor

        # 1. Raw (unmasked) stream-0 logits
        stream0_logits = logits_processor(lm_head, hidden_states)
        if stream0_logits is None:
            return None

        # 2. Determine mode per position from argmax of raw logits
        tentative = stream0_logits.argmax(dim=-1)  # [N]

        is_detect = (
            (tentative == cfg.text_token_id)
            | (tentative == cfg.audio_token_id)
        )
        is_audio = (
            (tentative >= cfg.codec_base_offset)
            & (tentative < cfg.vocab_size)
            & ~is_detect
        )
        is_text = ~is_detect & ~is_audio

        # 3. Apply modality-specific masks to stream-0 logits
        if is_detect.any():
            idx = is_detect.nonzero(as_tuple=True)[0]
            stream0_logits[idx] = stream0_logits[idx].masked_fill(
                self.modality_mask.unsqueeze(0), float("-inf")
            )

        if is_text.any():
            idx = is_text.nonzero(as_tuple=True)[0]
            stream0_logits[idx] = stream0_logits[idx].masked_fill(
                self.text_mask_s0.unsqueeze(0), float("-inf")
            )

        if is_audio.any():
            idx = is_audio.nonzero(as_tuple=True)[0]
            stream0_logits[idx] = stream0_logits[idx].masked_fill(
                self._audio_masks[0].unsqueeze(0), float("-inf")
            )

        # 4. For audio-mode requests: sample streams 1-7 and buffer
        if is_audio.any():
            self._sample_and_buffer_streams(
                hidden_states, is_audio, hidden_states.device
            )
        else:
            self._stream_buffer = None

        return stream0_logits

    def _sample_and_buffer_streams(
        self,
        hidden_states: torch.Tensor,
        is_audio: torch.Tensor,
        device: torch.device,
    ):
        """Sample streams 1-7 for audio-mode requests and update buffer.

        For each audio request:
          stream_s_logits = logits_processor(lm_head, hidden + stream_emb[s])
          Apply audio_mask[s]
          Top-k sample → token for stream s

        Buffer shape: [N, 7] where N is total batch size.
        Non-audio positions get zeros (pad).
        """
        N = hidden_states.shape[0]
        cfg = self.config
        lm_head = self.language_model.lm_head
        logits_processor = self.language_model.logits_processor

        # Initialize buffer
        new_buffer = torch.zeros(N, cfg.num_stream - 1,
                                 dtype=torch.long, device=device)

        audio_idx = is_audio.nonzero(as_tuple=True)[0]
        audio_hidden = hidden_states[audio_idx]  # [num_audio, hidden]

        for s in range(1, cfg.num_stream):
            # Add stream embedding offset
            h_s = audio_hidden + self.stream_emb.weight[s].unsqueeze(0)

            # Compute logits via logits_processor (handles vocab padding
            # and TP correctly, same as stream 0)
            s_logits = logits_processor(lm_head, h_s)

            # Apply stream-specific mask
            s_logits = s_logits.masked_fill(
                self._audio_masks[s].unsqueeze(0), float("-inf")
            )

            # Top-k sample
            sampled = self._top_k_sample(s_logits)
            new_buffer[audio_idx, s - 1] = sampled

        self._stream_buffer = new_buffer

    def _top_k_sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Top-k sampling for a batch of logits.

        Args:
            logits: [batch, vocab_size]
            temperature: sampling temperature
            top_k: number of top candidates

        Returns:
            sampled token IDs: [batch]
        """
        if temperature == 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(topk_values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return topk_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            # Skip Xcodec decoder weights for now (loaded separately if
            # needed for audio output decoding)
            skip_prefixes=["codec_decoder."],
        )
        return loader.load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

    # ------------------------------------------------------------------
    # Multi-model key mapping (for pipeline parallelism)
    # ------------------------------------------------------------------
    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["audio_tower."],
        )
