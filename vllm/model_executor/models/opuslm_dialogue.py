# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only OpusLM Dialogue model (SmolLM2-1.7B / Llama based multimodal
speech-language dialogue model with 9-stream delay-interleaved discrete codec
output).

Features:
  - Multi-turn dialogue with role tokens (system, user, assistant)
  - Text and audio input/output
  - Audio output via 9 streams (1 SSL + 8 DAC) with delay interleaving
  - Internal multi-stream sampling (streams 1-8 sampled inside compute_logits)
  - Audio input support via XEUS + K-means tokenization

Architecture differences from base OpusLM:
  - Base LM: SmolLM2-1.7B / Llama (not OLMo-2-7B) — no q_norm/k_norm
  - hidden_size: 2048 (not 4096)
  - num_layers: 24 (not 32)
  - Vocab: 62670 (not 113870)
  - Dialogue tokens: system=8, user=9, assistant=10, eou=11
  - Task tokens: audio_dialogue=89, text_dialogue=88
"""

import math
import uuid as _uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
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
from vllm.transformers_utils.configs.opuslm_dialogue import OpusLMDialogueConfig

logger = init_logger(__name__)

_AUDIO_SAMPLING_RATE = 16000


# ---------------------------------------------------------------------------
# Audio input tokenizer (shared with base OpusLM — same codec pipeline)
# ---------------------------------------------------------------------------

class _OpusLMDialogueAudioInputProcessor:
    """Server-side audio tokenizer: waveform -> (SSL stream0, DAC streams1-8).

    Same pipeline as base OpusLM:
      1) DAC encoder -> 8 codec streams
      2) XEUS SSL model + KMeans -> SSL stream
      3) Align lengths and map to global token IDs

    Models are loaded on GPU for fast encoding.  The encode_audio_tokens()
    method is called from the model's embed_multimodal() (GPU forward pass),
    NOT from the CPU preprocessor, matching the Bagpiper/Qwen3-Omni pattern.
    """

    def __init__(self, cfg: OpusLMDialogueConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = device
        self._dac_model = None
        self._ssl_model = None
        self._ssl_layer = int(getattr(cfg, "xeus_layer", 18))
        self._kmeans_model = None
        self._km_centroids = None

    def to(self, device: torch.device | str) -> "_OpusLMDialogueAudioInputProcessor":
        self.device = device
        if self._dac_model is not None:
            self._dac_model = self._dac_model.to(device)
        if self._ssl_model is not None:
            self._ssl_model = self._ssl_model.to(device)
        if self._km_centroids is not None:
            self._km_centroids = self._km_centroids.to(device)
        return self

    def _load_dac_model(self):
        if self._dac_model is not None:
            return self._dac_model

        try:
            from espnet2.bin.gan_codec_inference import AudioCoding
        except Exception as e:
            raise RuntimeError(
                "Failed to import ESPnet AudioCoding for DAC encoding. "
                "Please ensure ESPnet runtime dependencies are installed."
            ) from e

        self._dac_model = AudioCoding.from_pretrained(
            self.cfg.dac_hf_model_tag
        ).model.eval().to(self.device)
        logger.info("Loaded DAC encoder on %s", self.device)
        return self._dac_model

    def _resolve_xeus_paths(self) -> tuple[str, str]:
        from huggingface_hub import hf_hub_download

        repo = self.cfg.xeus_hf_model_tag
        ckpt_file = getattr(
            self.cfg, "xeus_checkpoint_filename", "model/xeus_checkpoint_new.pth"
        )
        km_file = self.cfg.km_model_filename

        ckpt_path = hf_hub_download(repo, ckpt_file)
        km_path = hf_hub_download(repo, km_file)
        return ckpt_path, km_path

    def _load_ssl_and_kmeans(self):
        if self._ssl_model is not None and self._kmeans_model is not None:
            return self._ssl_model, self._kmeans_model

        try:
            import joblib
            from espnet2.tasks.ssl import SSLTask
        except Exception as e:
            raise RuntimeError(
                "Failed to import SSL dependencies (joblib/espnet2.tasks.ssl). "
                "Please install required ESPnet dependencies."
            ) from e

        ckpt_path, km_path = self._resolve_xeus_paths()
        self._ssl_model, _ = SSLTask.build_model_from_file(
            None, ckpt_path, device=str(self.device)
        )
        self._ssl_model.eval()
        logger.info("Loaded XEUS SSL model on %s", self.device)
        self._kmeans_model = joblib.load(km_path)
        # Cache K-means centroids as GPU tensor for fast inference
        self._km_centroids = torch.from_numpy(
            self._kmeans_model.cluster_centers_
        ).float().to(self.device)
        logger.info(
            "Cached K-means centroids [%s] on %s",
            self._km_centroids.shape, self.device,
        )
        return self._ssl_model, self._kmeans_model

    def _extract_ssl_labels(self, audio_np: np.ndarray) -> np.ndarray:
        ssl_model, km_model = self._load_ssl_and_kmeans()

        wav = torch.from_numpy(audio_np).float().view(1, -1).to(self.device)
        wav_lens = torch.tensor([wav.shape[1]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            enc_out = ssl_model.encode(wav, wav_lens)

        # New ESPnet API returns Dict with "encoder_output" (list of
        # per-layer tensors).  Old API returned (feats, mask_info, pen).
        if isinstance(enc_out, dict):
            feats = enc_out["encoder_output"]
        else:
            feats = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        if isinstance(feats, (list, tuple)):
            layer = min(max(self._ssl_layer, 0), len(feats) - 1)
            ssl_feats = feats[layer]
        else:
            ssl_feats = feats

        # GPU K-means: use cached centroids tensor for fast argmin
        if hasattr(self, '_km_centroids') and self._km_centroids is not None:
            feats_2d = ssl_feats[0]  # [T, feat_dim], on GPU
            distances = torch.cdist(
                feats_2d.unsqueeze(0),
                self._km_centroids.unsqueeze(0),
            )  # [1, T, n_clusters]
            labels = distances[0].argmin(dim=-1).cpu().numpy().astype(np.int64)
        else:
            ssl_feats_np = ssl_feats[0].detach().cpu().numpy()
            labels = km_model.predict(ssl_feats_np).astype(np.int64)
        max_ssl_id = self.cfg.ssl_token_end - self.cfg.ssl_token_start - 1
        return np.clip(labels, 0, max_ssl_id)

    def _extract_codec_codes(self, audio_np: np.ndarray) -> np.ndarray:
        dac = self._load_dac_model()
        wav = torch.from_numpy(audio_np).float().view(1, 1, -1).to(self.device)
        with torch.inference_mode():
            codes = dac.encode(wav)
            codes = codes.permute(1, 2, 0)
        return codes[0, :, : self.cfg.num_codec_streams].long().cpu().numpy()

    def _to_global_stream18(self, codec_codes: np.ndarray) -> np.ndarray:
        codec_codes = codec_codes.copy()
        for s in range(self.cfg.num_codec_streams):
            offset = self.cfg.codec_token_start + s * self.cfg.codec_per_stream_size
            codec_codes[:, s] = codec_codes[:, s] + offset
        return codec_codes

    def encode_audio_tokens(
        self,
        audio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode one audio item to OpusLM stream tokens.

        Returns:
            stream0_global_ids: [T]
            streams18_global_ids: [T, 8]
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        ssl_labels = self._extract_ssl_labels(audio)
        codec_codes = self._extract_codec_codes(audio)
        codec_global = self._to_global_stream18(codec_codes)

        length = min(len(ssl_labels), len(codec_global))
        if length <= 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 8), dtype=np.int64)

        stream0 = ssl_labels[:length] + self.cfg.ssl_token_start
        streams18 = codec_global[:length]
        return stream0.astype(np.int64), streams18.astype(np.int64)


# ---------------------------------------------------------------------------
# Multimodal processing for dialogue format
# ---------------------------------------------------------------------------

class _OpusLMDialogueProcessor:
    """Minimal processor for OpusLM Dialogue."""

    def __init__(self):
        self.audio_token = "<codec_ssl_start_end>"

    def get_vocab(self) -> dict[str, int]:
        return {self.audio_token: 34}


class OpusLMDialogueProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> OpusLMDialogueConfig:
        return self.ctx.get_hf_config(OpusLMDialogueConfig)

    def get_hf_processor(self, **kwargs: object) -> _OpusLMDialogueProcessor:
        return _OpusLMDialogueProcessor()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        return {"audio": max(1, int(seq_len))}

    def get_data_parser(self) -> MultiModalDataParser:
        return OpusLMDialogueMultiModalDataParser(
            target_sr=_AUDIO_SAMPLING_RATE,
            target_channels=1,
        )


class OpusLMDialogueMultiModalDataParser(MultiModalDataParser):

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        return super()._parse_audio_data(data)


class OpusLMDialogueDummyInputsBuilder(
    BaseDummyInputsBuilder[OpusLMDialogueProcessingInfo]
):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "<codec_ssl_start_end>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        dummy_audio = np.zeros(_AUDIO_SAMPLING_RATE, dtype=np.float32)
        return {
            "audio": [dummy_audio] * num_audios,
        }


class OpusLMDialogueMultiModalProcessor(
    BaseMultiModalProcessor[OpusLMDialogueProcessingInfo],
):
    """Handles server-side OpusLM dialogue tokenization and prompt replacement.

    Converts OpenAI chat messages into the dialogue token sequence format:
        <sos/eos> <audio_dialogue_task>
        <system_prompt> <spk_start/end> [speaker_audio × 500 frames]
        <user_input> <codec_ssl_start/end> [user_audio × T frames]
        <user_input> <text_bpe_start/end> [user_text tokens]
        <assistant_output> <text_bpe_start/end> [asst_text tokens]
        <assistant_output> <codec_ssl_start/end> [GENERATE FROM HERE...]
    """

    @staticmethod
    def resolve_task_token_id(
        cfg: OpusLMDialogueConfig,
        *,
        task: str | int | None = None,
        mode: str | None = None,
        has_audio_input: bool = False,
    ) -> int:
        """Resolve request task to dialogue task token ID."""
        if isinstance(task, int):
            return int(task)

        task_aliases = {
            "audio_dialogue": cfg.audio_dialogue_task_token_id,
            "audio_dialogue_task": cfg.audio_dialogue_task_token_id,
            "text_dialogue": cfg.text_dialogue_task_token_id,
            "text_dialogue_task": cfg.text_dialogue_task_token_id,
            "asr": cfg.codec_ssl_asr_task_token_id,
            "tts": cfg.codec_ssl_tts_task_token_id,
            "plain_tts": cfg.codec_ssl_plain_tts_task_token_id,
            "textlm": cfg.textlm_task_token_id,
        }
        if isinstance(task, str):
            task_norm = task.strip().lower()
            if task_norm in task_aliases:
                return task_aliases[task_norm]
            raise ValueError(
                f"Unsupported task '{task}'. "
                f"Supported: {', '.join(task_aliases.keys())}."
            )

        # Default: audio_dialogue
        return cfg.audio_dialogue_task_token_id

    @staticmethod
    def _set_input_ids(
        text_inputs: BatchFeature,
        input_ids: torch.Tensor,
    ) -> BatchFeature:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return text_inputs

        text_inputs["input_ids"] = input_ids

        attention_mask = text_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            text_inputs["attention_mask"] = torch.ones(
                input_ids.shape,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )

        token_type_ids = text_inputs.get("token_type_ids")
        if isinstance(token_type_ids, torch.Tensor):
            text_inputs["token_type_ids"] = torch.zeros(
                input_ids.shape,
                device=token_type_ids.device,
                dtype=token_type_ids.dtype,
            )

        return text_inputs

    def _get_audio_input_processor(self) -> _OpusLMDialogueAudioInputProcessor:
        cfg = self.info.get_hf_config()
        processor = getattr(self, "_audio_input_processor", None)
        if processor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = _OpusLMDialogueAudioInputProcessor(cfg, device=device)
            self._audio_input_processor = processor
        return processor

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Build dialogue token sequence from prompt and audio inputs.

        The prompt text is used only for text-only messages (the tokenizer
        handles text encoding). For audio messages, the server-side audio
        tokenizer produces SSL + DAC stream tokens.

        The dialogue sequence is built from the messages in mm_kwargs, following
        the ESPnet preprocessor format.
        """
        tokenizer = self.info.get_tokenizer()
        cfg = self.info.get_hf_config()

        # Get task token
        task_obj = mm_kwargs.get("task")
        mode_obj = mm_kwargs.get("mode")
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        has_audio = bool(audios)

        task_token_id = self.resolve_task_token_id(
            cfg,
            task=task_obj if isinstance(task_obj, (str, int)) else None,
            mode=mode_obj if isinstance(mode_obj, str) else None,
            has_audio_input=has_audio,
        )

        # Get dialogue messages from mm_kwargs
        messages = mm_kwargs.get("messages")

        # Support pre-extracted tokens: mm_processor_kwargs can include
        # "pre_tokens" — a list of dicts with keys: role, stream0, streams18.
        # These bypass the SSL encoder entirely and use ARK token data
        # directly. When present, we rebuild the messages to use the
        # "input_tokens" content type.
        pre_tokens = mm_kwargs.get("pre_tokens")
        if pre_tokens and isinstance(pre_tokens, (list, tuple)):
            rebuilt = []
            pt_idx = 0
            if messages:
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        new_parts = []
                        for part in content:
                            if (isinstance(part, dict)
                                    and part.get("type") == "input_audio"
                                    and pt_idx < len(pre_tokens)):
                                pt = pre_tokens[pt_idx]
                                pt_idx += 1
                                new_parts.append({
                                    "type": "input_tokens",
                                    "stream0": pt.get("stream0", []),
                                    "streams18": pt.get("streams18", []),
                                })
                            else:
                                new_parts.append(part)
                        rebuilt.append({"role": role, "content": new_parts})
                    else:
                        rebuilt.append(msg)
            messages = rebuilt if rebuilt else messages

        # If no structured messages, fall back to simple tokenization
        if not messages or not isinstance(messages, (list, tuple)):
            text_inputs = tokenizer(prompt, return_tensors="pt")
            return self._build_simple_sequence(
                text_inputs, cfg, task_token_id, audios
            )

        # Build the full dialogue sequence from messages
        mode = mode_obj if isinstance(mode_obj, str) else None
        return self._build_dialogue_sequence(
            cfg, tokenizer, task_token_id, messages, audios, mode=mode
        )

    def _build_simple_sequence(
        self,
        text_inputs: BatchFeature,
        cfg: OpusLMDialogueConfig,
        task_token_id: int,
        audios: list,
    ) -> BatchFeature:
        """Build a simple (non-dialogue) sequence like TTS."""
        input_ids = text_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return text_inputs

        if input_ids.ndim == 2:
            body_ids = input_ids[0].tolist()
        else:
            body_ids = input_ids.tolist()

        sos = int(cfg.sos_eos_token_id)
        seq = [sos, int(task_token_id)]

        # Add text BPE prefix for text content
        if body_ids:
            seq.append(cfg.text_bpe_start_end_token_id)
            seq.extend(body_ids)

        # Add codec prefix for audio generation
        seq.append(cfg.codec_ssl_start_end_token_id)

        new_input_ids = torch.tensor([seq], dtype=torch.long)
        text_inputs = self._set_input_ids(text_inputs, new_input_ids)

        if not audios:
            return BatchFeature(data={**text_inputs})

        # Compute placeholder frame counts and store raw audio.
        # Actual SSL+DAC encoding happens in embed_multimodal().
        stream0_chunks = []
        stream18_chunks = []
        lengths = []
        raw_audio_chunks = []
        raw_audio_sample_counts = []

        for audio in audios:
            audio_np = (
                audio
                if isinstance(audio, np.ndarray)
                else np.array(audio)
            )
            if audio_np.ndim != 1:
                audio_np = audio_np.reshape(-1)
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            N_samples = len(audio_np)
            T = math.ceil(N_samples / 320) if N_samples > 0 else 0
            lengths.append(T)
            if T > 0:
                stream0_chunks.append(
                    torch.full(
                        (T,),
                        cfg.codec_ssl_start_end_token_id,
                        dtype=torch.long,
                    )
                )
                stream18_chunks.append(
                    torch.zeros(T, cfg.num_codec_streams, dtype=torch.long)
                )
                raw_audio_chunks.append(
                    torch.from_numpy(audio_np).float()
                )
                raw_audio_sample_counts.append(N_samples)

        if stream0_chunks:
            stream0_tensor = torch.cat(stream0_chunks, dim=0).long()
            stream18_tensor = torch.cat(stream18_chunks, dim=0).long()
        else:
            stream0_tensor = torch.empty((0,), dtype=torch.long)
            stream18_tensor = torch.empty(
                (0, cfg.num_codec_streams), dtype=torch.long
            )

        data = {
            **text_inputs,
            "audio_stream0_ids": stream0_tensor,
            "audio_streams18": stream18_tensor,
            "audio_lengths": torch.tensor(lengths, dtype=torch.long),
            "audio_is_system": torch.zeros(
                len(lengths), dtype=torch.bool
            ),
        }

        if raw_audio_chunks:
            data["input_audio_features"] = torch.cat(
                raw_audio_chunks, dim=0
            )
            data["audio_feature_lengths"] = torch.tensor(
                raw_audio_sample_counts, dtype=torch.long
            )

        return BatchFeature(data=data)

    def _build_dialogue_sequence(
        self,
        cfg: OpusLMDialogueConfig,
        tokenizer: Any,
        task_token_id: int,
        messages: list,
        audios: list,
        *,
        mode: str | None = None,
    ) -> BatchFeature:
        """Build multi-turn dialogue token sequence from chat messages.

        Matches the ESPnet inference format (inference_last_segment=true):
          <sos/eos>(5) <task_token>
          <system>(8) <spk>(37) [speaker_audio ×500] [pad ×8]
          <user>(9) <codec_ssl>(34) [user_audio ×T] [pad ×8]
          <user>(9) <text_bpe>(35) [user_text...] [pad ×8]
          <asst>(10) <text_bpe>(35) [asst_text...] [pad ×8]
          <asst>(10) <codec_ssl>(34) [GENERATE FROM HERE...]

        All prefilled segments are non-target: no end tokens (no eou).
        Inter-segment padding (nq-1=8 zero rows) after every non-target
        segment prevents DAC token overlap after delay interleaving.

        Audio segments use a single placeholder token (34) in input_ids.
        The framework's prompt replacement mechanism replaces each [34]
        with the full stream0 tokens + inter-segment padding. For system
        audio, [34] is replaced by just stream0 (spk marker 37 is already
        before the placeholder). For user audio, [34] is replaced by
        [34] + stream0, preserving the modality marker.

        Empty assistant message = generation target (no padding after).
        """
        sos = int(cfg.sos_eos_token_id)
        # ESPnet inter_segment_pad = nq - 1: zero rows between segments
        # to prevent DAC token overlap after delay interleave.
        inter_pad = int(cfg.nq) - 1  # 8
        seq: list[int] = [sos, int(task_token_id)]

        # Check if tokenizer already shifts IDs to global range.
        _needs_text_shift = not hasattr(tokenizer, 'text_token_offset')
        _text_offset = int(cfg.text_token_start) if _needs_text_shift else 0

        stream0_chunks: list[torch.Tensor] = []
        stream18_chunks: list[torch.Tensor] = []
        lengths: list[int] = []
        audio_is_system: list[bool] = []
        raw_audio_chunks: list[torch.Tensor] = []
        raw_audio_sample_counts: list[int] = []

        # Track audio index for matching audios to messages
        audio_idx = 0

        # Role token map
        role_token_map = {
            "system": cfg.system_prompt_token_id,
            "user": cfg.user_input_token_id,
            "assistant": cfg.assistant_output_token_id,
        }

        for msg_idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            is_last = (msg_idx == len(messages) - 1)

            role_token = role_token_map.get(role, cfg.user_input_token_id)

            # Handle different content types
            if isinstance(content, str):
                # Text-only message
                if content == "" and is_last and role == "assistant":
                    # Empty assistant = generation target.
                    # Append [role, modality_marker] prefix to match
                    # ESPnet inference which provides the first 2 tokens
                    # (role + modality) of each target segment as prefill.
                    #
                    # For ASR mode (audio_text), the target is user text
                    # transcription, so use <user>(9) instead of
                    # <assistant>(10) to match ESPnet training format.
                    if mode == "audio_text":
                        gen_role_token = int(cfg.user_input_token_id)
                        gen_modality = int(cfg.text_bpe_start_end_token_id)
                    else:
                        # Determine modality: if a previous assistant
                        # message already has text content, this empty
                        # assistant generates audio (codec_ssl=34).
                        # Otherwise it generates text (text_bpe=35).
                        prev_asst_has_text = any(
                            m.get("role") == "assistant"
                            and isinstance(m.get("content"), str)
                            and m["content"] != ""
                            for m in messages[:msg_idx]
                        )
                        gen_role_token = role_token
                        gen_modality = (
                            int(cfg.codec_ssl_start_end_token_id)
                            if prev_asst_has_text
                            else int(cfg.text_bpe_start_end_token_id)
                        )
                    seq.append(gen_role_token)
                    seq.append(gen_modality)
                    continue

                seq.append(role_token)
                seq.append(cfg.text_bpe_start_end_token_id)
                text_ids = tokenizer.encode(content, add_special_tokens=False)
                seq.extend(tid + _text_offset for tid in text_ids)
                # Non-target segments: no end token (matching ESPnet).
                # Inter-segment padding: nq-1 zero rows.
                seq.extend([0] * inter_pad)

            elif isinstance(content, list):
                # Multi-part content (may contain text + audio)
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    part_type = part.get("type", "")

                    if part_type == "text":
                        text_val = part.get("text", "")
                        if text_val == "" and is_last and role == "assistant":
                            continue

                        seq.append(role_token)
                        seq.append(cfg.text_bpe_start_end_token_id)
                        text_ids = tokenizer.encode(
                            text_val, add_special_tokens=False
                        )
                        text_offset = cfg.text_token_start
                        seq.extend(tid + text_offset for tid in text_ids)
                        # Non-target: no end token. Add inter-segment padding.
                        seq.extend([0] * inter_pad)

                    elif part_type == "input_tokens":
                        # Pre-extracted token arrays (bypass SSL encoder).
                        # Expected format: {"type": "input_tokens",
                        #   "stream0": [int, ...],
                        #   "streams18": [[int,...], ...]}
                        # Values are global token IDs (already with bias).
                        s0_list = part.get("stream0", [])
                        s18_list = part.get("streams18", [])
                        stream0 = np.array(s0_list, dtype=np.int64)
                        if s18_list and isinstance(s18_list[0], list):
                            streams18 = np.array(s18_list, dtype=np.int64)
                        else:
                            streams18 = np.zeros(
                                (len(s0_list), 8), dtype=np.int64
                            )
                        T = len(stream0)
                        if T == 0:
                            continue

                        is_system = (role == "system")
                        seq.append(role_token)

                        if is_system:
                            seq.append(cfg.spk_start_end_token_id)
                            spk_len = cfg.speaker_prompt_length
                            if T >= spk_len:
                                stream0 = stream0[:spk_len]
                                streams18 = streams18[:spk_len]
                            else:
                                # Pad with 0 (pad token) to match ESPnet
                                # training format (NOT stream0[0]).
                                pad_s0 = np.zeros(
                                    spk_len - T, dtype=np.int64
                                )
                                pad_s18 = np.zeros(
                                    (spk_len - T, 8), dtype=np.int64
                                )
                                stream0 = np.concatenate([stream0, pad_s0])
                                streams18 = np.concatenate(
                                    [streams18, pad_s18]
                                )
                            T = spk_len

                        pad_s0 = np.zeros(inter_pad, dtype=np.int64)
                        pad_s18 = np.zeros(
                            (inter_pad, 8), dtype=np.int64
                        )
                        stream0 = np.concatenate([stream0, pad_s0])
                        streams18 = np.concatenate([streams18, pad_s18])
                        T += inter_pad

                        seq.append(int(cfg.codec_ssl_start_end_token_id))

                        lengths.append(T)
                        audio_is_system.append(is_system)
                        stream0_chunks.append(
                            torch.from_numpy(stream0).long()
                        )
                        stream18_chunks.append(
                            torch.from_numpy(streams18).long()
                        )

                    elif part_type == "input_audio":
                        if audio_idx >= len(audios):
                            logger.warning(
                                "Audio index %d exceeds available audios (%d)",
                                audio_idx, len(audios),
                            )
                            continue

                        audio_data = audios[audio_idx]
                        audio_idx += 1

                        # Convert to float32 numpy for model-side encoding
                        audio_np = (
                            audio_data
                            if isinstance(audio_data, np.ndarray)
                            else np.array(audio_data)
                        )
                        if audio_np.ndim != 1:
                            audio_np = audio_np.reshape(-1)
                        if audio_np.dtype != np.float32:
                            audio_np = audio_np.astype(np.float32)
                        N_samples = len(audio_np)
                        T = (
                            math.ceil(N_samples / 320)
                            if N_samples > 0
                            else 0
                        )
                        if T == 0:
                            continue

                        is_system = (role == "system")
                        seq.append(role_token)

                        # Placeholder tokens — actual SSL+DAC encoding
                        # happens in embed_multimodal() on the GPU.
                        stream0 = np.full(
                            T,
                            cfg.codec_ssl_start_end_token_id,
                            dtype=np.int64,
                        )
                        streams18 = np.zeros((T, 8), dtype=np.int64)

                        if is_system:
                            # System audio: spk marker before placeholder
                            seq.append(cfg.spk_start_end_token_id)
                            # Pad/trim to speaker_prompt_length
                            spk_len = cfg.speaker_prompt_length
                            if T >= spk_len:
                                stream0 = stream0[:spk_len]
                                streams18 = streams18[:spk_len]
                            else:
                                pad_s0 = np.zeros(
                                    spk_len - T, dtype=np.int64
                                )
                                pad_s18 = np.zeros(
                                    (spk_len - T, 8), dtype=np.int64
                                )
                                stream0 = np.concatenate([stream0, pad_s0])
                                streams18 = np.concatenate(
                                    [streams18, pad_s18]
                                )
                            T = spk_len

                        # Inter-segment padding: extend audio by nq-1 zero
                        # rows. After per-segment delay interleave, these
                        # positions carry trailing DAC embeddings (matching
                        # ESPnet whole-sequence delay interleave behavior).
                        pad_s0 = np.zeros(inter_pad, dtype=np.int64)
                        pad_s18 = np.zeros((inter_pad, 8), dtype=np.int64)
                        stream0 = np.concatenate([stream0, pad_s0])
                        streams18 = np.concatenate([streams18, pad_s18])
                        T += inter_pad

                        # Use placeholder token [34] in input_ids.
                        # The framework's prompt replacement mechanism
                        # will replace this with the actual stream0
                        # tokens + inter-segment padding.
                        seq.append(int(cfg.codec_ssl_start_end_token_id))

                        lengths.append(T)
                        audio_is_system.append(is_system)
                        stream0_chunks.append(
                            torch.from_numpy(stream0).long()
                        )
                        stream18_chunks.append(
                            torch.from_numpy(streams18).long()
                        )

                        # Store raw audio for encoding in
                        # embed_multimodal()
                        raw_audio_chunks.append(
                            torch.from_numpy(audio_np).float()
                        )
                        raw_audio_sample_counts.append(N_samples)

            else:
                # Fallback: treat as string
                seq.append(role_token)
                seq.append(cfg.text_bpe_start_end_token_id)
                text_ids = tokenizer.encode(str(content), add_special_tokens=False)
                seq.extend(tid + _text_offset for tid in text_ids)
                # Non-target: no end token. Add inter-segment padding.
                seq.extend([0] * inter_pad)

        # Build the final input_ids tensor
        input_ids = torch.tensor([seq], dtype=torch.long)
        text_inputs = BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )

        if not stream0_chunks:
            return text_inputs

        stream0_tensor = torch.cat(stream0_chunks, dim=0).long()
        stream18_tensor = torch.cat(stream18_chunks, dim=0).long()

        data = {
            **text_inputs,
            "audio_stream0_ids": stream0_tensor,
            "audio_streams18": stream18_tensor,
            "audio_lengths": torch.tensor(lengths, dtype=torch.long),
            "audio_is_system": torch.tensor(
                audio_is_system, dtype=torch.bool
            ),
        }

        # Raw audio features for model-side encoding (embed_multimodal).
        # Only present when audio was provided as waveforms (input_audio),
        # NOT for pre-tokenized ARK data (input_tokens).
        if raw_audio_chunks:
            data["input_audio_features"] = torch.cat(
                raw_audio_chunks, dim=0
            )
            data["audio_feature_lengths"] = torch.tensor(
                raw_audio_sample_counts, dtype=torch.long
            )

        return BatchFeature(data=data)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: "MultiModalDataItems",
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Our _call_hf_processor places placeholder [34] tokens in
        # input_ids for each audio segment. The framework finds these
        # targets and replaces them with the full replacement content
        # (stream0 tokens). We return False so the framework applies
        # the replacement itself.
        return False

    def _apply_hf_processor_main(
        self,
        prompt,
        mm_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
        *,
        enable_hf_prompt_update: bool,
    ):
        # Override: always route through _apply_hf_processor_text_mm
        # so our _call_hf_processor is invoked. When the APIServer's
        # chat preprocessing tokenizes the rendered chat template, the
        # prompt arrives as list[int]. The base class would then skip
        # _call_hf_processor entirely. We need _call_hf_processor to
        # build the custom dialogue token sequence from mm_kwargs
        # ["messages"], so we convert list[int] prompts to a dummy
        # string (our _call_hf_processor ignores the prompt text when
        # messages are present).
        if isinstance(prompt, list):
            prompt = "dummy"
        return self._apply_hf_processor_text_mm(
            prompt_text=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    def _cached_apply_hf_processor(
        self,
        prompt: "str | list[int]",
        mm_data_items: "MultiModalDataItems",
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids: Any = None,
    ):
        # Always use uncached path: our _call_hf_processor builds custom
        # dialogue token sequences which can't be cached (audio encoding
        # is dynamic and context-dependent).
        return self._apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _hash_mm_items(
        self,
        mm_items: "MultiModalDataItems",
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids=None,
    ):
        hashes = super()._hash_mm_items(
            mm_items, hf_processor_mm_kwargs, tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        # Use unique hashes for all audio items to prevent encoder
        # cache from returning tensors with mismatched shapes.
        # TODO: investigate why content-based hashing causes shape
        # mismatches and re-enable for cascade cache hits.
        if "audio" in hashes:
            hashes["audio"] = [
                str(_uuid.uuid4()) for _ in hashes["audio"]
            ]
        return hashes

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_lengths = hf_inputs.get("audio_lengths")
        if audio_lengths is None:
            return {}

        assert isinstance(audio_lengths, torch.Tensor)
        fields: dict[str, MultiModalFieldConfig] = {
            "audio_stream0_ids": MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_lengths, dim=0
            ),
            "audio_streams18": MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_lengths, dim=0
            ),
            "audio_lengths": MultiModalFieldConfig.batched("audio"),
            "audio_is_system": MultiModalFieldConfig.batched("audio"),
        }

        # Raw audio features for model-side encoding
        audio_feature_lengths = hf_inputs.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            assert isinstance(audio_feature_lengths, torch.Tensor)
            fields["input_audio_features"] = (
                MultiModalFieldConfig.flat_from_sizes(
                    "audio", audio_feature_lengths, dim=0
                )
            )
            fields["audio_feature_lengths"] = (
                MultiModalFieldConfig.batched("audio")
            )

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        cfg = self.info.get_hf_config()

        audio_token = processor.audio_token

        out_mm_data = out_mm_kwargs.get_data()
        audio_lengths = out_mm_data.get("audio_lengths")
        audio_stream0_ids = out_mm_data.get("audio_stream0_ids")
        audio_is_system = out_mm_data.get("audio_is_system")
        if audio_lengths is None or audio_stream0_ids is None:
            return []

        assert isinstance(audio_lengths, torch.Tensor)
        assert isinstance(audio_stream0_ids, torch.Tensor)

        offsets = [0]
        for l in audio_lengths.tolist():
            offsets.append(offsets[-1] + int(l))

        def get_replacement_audio(item_idx: int):
            start = offsets[item_idx]
            end = offsets[item_idx + 1]
            num_frames = end - start
            if num_frames <= 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} is too short to be represented "
                    "inside the model"
                )

            stream0_ids = audio_stream0_ids[start:end].tolist()

            # System audio: placeholder 34 is replaced entirely by ssl
            # tokens (the spk marker 37 is already in the sequence).
            # User audio: placeholder 34 is replaced by [34, ssl_tokens]
            # so the modality marker stays in the sequence.
            is_sys = (
                audio_is_system is not None
                and isinstance(audio_is_system, torch.Tensor)
                and item_idx < len(audio_is_system)
                and bool(audio_is_system[item_idx])
            )

            if is_sys:
                # System: replace placeholder with just ssl tokens
                tokens = stream0_ids
            else:
                # User/asst: keep modality marker 34 + ssl tokens
                tokens = [int(cfg.codec_ssl_start_end_token_id)] + stream0_ids

            def _is_embed(
                _tokenizer: object,
                _full: object,
            ) -> torch.Tensor:
                mask = torch.ones(len(tokens), dtype=torch.bool)
                if not is_sys and len(tokens) > 0:
                    mask[0] = False  # modality marker 34 is not embed
                return mask

            return PromptUpdateDetails(full=tokens, is_embed=_is_embed)

        # Use integer token ID [34] as target instead of the string
        # "<codec_ssl_start_end>". The string target would fail because
        # _seq2tokens tokenizes it through the SmolLM tokenizer which
        # doesn't have this as a special token.
        placeholder_id = int(cfg.codec_ssl_start_end_token_id)  # 34
        return [
            PromptReplacement(
                modality="audio",
                target=[placeholder_id],
                replacement=get_replacement_audio,
            ),
        ]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
@MULTIMODAL_REGISTRY.register_processor(
    OpusLMDialogueMultiModalProcessor,
    info=OpusLMDialogueProcessingInfo,
    dummy_inputs=OpusLMDialogueDummyInputsBuilder,
)
class OpusLMDialogueForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """OpusLM Dialogue: SmolLM2-1.7B / Llama based multimodal speech-language
    dialogue model.

    Supports multi-turn dialogue with text/audio input and text/audio output.
    Audio output uses 9 streams (1 SSL + 8 DAC) with delay interleaving.
    From vLLM's perspective, the model generates 1 token per step (stream 0);
    streams 1-8 (DAC) are sampled internally and buffered.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<codec_ssl_start_end>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config: OpusLMDialogueConfig = vllm_config.model_config.hf_config
        self.config = config

        # Build a LlamaConfig so we can reuse LlamaModel
        from transformers import LlamaConfig as HFLlamaConfig
        from vllm.model_executor.models.llama import LlamaModel
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        llama_hf_config = HFLlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            tie_word_embeddings=config.tie_word_embeddings,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )

        # Temporarily patch vllm_config to use the LlamaConfig
        llama_vllm_config = vllm_config.with_hf_config(
            llama_hf_config, architectures=["LlamaForCausalLM"]
        )

        # --- Language model (Llama backbone) ---
        with self._mark_language_model(vllm_config):
            self.model = LlamaModel(
                vllm_config=llama_vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        # --- LM head (vocab projection) ---
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        # --- Per-stream head bias embeddings [12, 2048] ---
        self.head_emb = nn.Embedding(12, config.hidden_size)

        # --- Logits processor ---
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # --- Pipeline parallelism ---
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # --- Internal decode state ---
        self._stream_buffer_dict: dict[str, torch.Tensor] = {}
        self._per_req_config: dict[str, dict] = {}
        self._stream18_history: dict[str, list[torch.Tensor]] = {}
        self._stream0_history: dict[str, list[int]] = {}
        self._decoded_audio: dict[str, str] = {}
        self._current_batch_req_ids: list[str] = []

        # --- DAC decoder (lazy-loaded) ---
        self._dac_model = None
        self._dac_sample_rate: int = config.dac_sample_rate
        self._dac_hf_model_tag: str = config.dac_hf_model_tag

        # --- Audio encoder for model-side SSL+DAC encoding (lazy-loaded) ---
        self._audio_input_processor_model: (
            _OpusLMDialogueAudioInputProcessor | None
        ) = None

        # --- Marker for gpu_model_runner detection ---
        self._is_dialogue = True

        # --- Precompute token masks ---
        self._build_masks(config)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------
    def _build_masks(self, config: OpusLMDialogueConfig):
        """Precompute token validity masks for each stream."""
        V = config.vocab_size

        # Stream 0 (SSL): allow [ssl_token_start, ssl_token_end)
        #                      + codec_ssl_start_end (34) + eos (5)
        ssl_mask = torch.ones(V, dtype=torch.bool)
        ssl_mask[config.ssl_token_start:config.ssl_token_end] = False
        ssl_mask[config.codec_ssl_start_end_token_id] = False
        ssl_mask[config.eos_token_id] = False
        self.register_buffer("audio_mask_s0", ssl_mask)

        # Text mask (stream 0 during text phase)
        # Only allow EOS (5) + text BPE tokens [text_token_start, text_token_end).
        # The model generates text content followed by EOS — no special/control
        # tokens (role markers, modality markers, etc.) should appear in the
        # generated text. Allowing [0,256) caused the model to emit token 35
        # (text_bpe_start_end) repeatedly, which the SmolLM tokenizer decodes
        # as the character "3".
        text_mask = torch.ones(V, dtype=torch.bool)
        text_mask[config.eos_token_id] = False
        text_mask[config.text_token_start:config.text_token_end] = False
        self.register_buffer("text_mask_s0", text_mask)

        # Pre-audio mask: forces codec_ssl_start_end (34) output
        pre_audio_mask = torch.ones(V, dtype=torch.bool)
        pre_audio_mask[config.codec_ssl_start_end_token_id] = False
        self.register_buffer("pre_audio_mask", pre_audio_mask)

        # Streams 1-8 (DAC codec)
        audio_masks = [self.audio_mask_s0]
        for k in range(config.num_codec_streams):
            mask = torch.ones(V, dtype=torch.bool)
            start = config.codec_token_start + k * config.codec_per_stream_size
            end = start + config.codec_per_stream_size
            mask[start:end] = False
            mask[0] = False  # allow pad (0)
            self.register_buffer(f"audio_mask_s{k + 1}", mask)
            audio_masks.append(mask)
        self._audio_masks = audio_masks

    # ------------------------------------------------------------------
    # Audio encoder (lazy-loaded, used by embed_multimodal)
    # ------------------------------------------------------------------
    def _get_audio_encoder(self) -> _OpusLMDialogueAudioInputProcessor:
        """Lazy-load the SSL+DAC audio encoder on model device."""
        if self._audio_input_processor_model is None:
            device = next(self.model.parameters()).device
            self._audio_input_processor_model = (
                _OpusLMDialogueAudioInputProcessor(
                    self.config, device=device
                )
            )
            logger.info(
                "Loaded audio encoder (SSL+DAC) on %s", device
            )
        return self._audio_input_processor_model

    @torch.inference_mode()
    def _encode_and_embed_audio(
        self, **kwargs: object
    ) -> tuple[torch.Tensor, ...]:
        """Encode raw audio waveforms via SSL+DAC and produce embeddings.

        This runs the full audio encoding pipeline (XEUS SSL → K-means
        quantization → DAC codec encoding) inside the model forward
        pass, batched by vLLM's ``_execute_mm_encoder()``.

        Each audio item produces T_total embeddings where:
          - System audio: T_total = speaker_prompt_length + inter_pad
          - User audio:   T_total = ceil(N_samples/320) + inter_pad

        The embeddings are delay-interleaved sums of all 9 streams,
        matching the existing ``embed_multimodal()`` logic for
        pre-computed token IDs.
        """
        raw_audio = kwargs["input_audio_features"]
        audio_sample_lengths = kwargs["audio_feature_lengths"]
        audio_is_system = kwargs.get("audio_is_system")
        embed_lengths = kwargs.get("audio_lengths")

        if not isinstance(raw_audio, torch.Tensor):
            raw_audio = torch.as_tensor(raw_audio)
        if not isinstance(audio_sample_lengths, torch.Tensor):
            audio_sample_lengths = torch.as_tensor(audio_sample_lengths)

        device = next(self.model.parameters()).device
        embed_fn = self.model.embed_tokens
        cfg = self.config
        inter_pad = cfg.nq - 1
        V = cfg.vocab_size

        encoder = self._get_audio_encoder()
        mm_embeddings: list[torch.Tensor] = []

        offset = 0
        n_items = len(audio_sample_lengths.tolist()) if audio_sample_lengths.dim() > 0 else 1
        sample_lens = audio_sample_lengths.tolist() if audio_sample_lengths.dim() > 0 else [int(audio_sample_lengths)]
        for i, n_samples in enumerate(sample_lens):
            n_samples = int(n_samples)
            audio_np = raw_audio[offset:offset + n_samples].cpu().numpy()
            offset += n_samples

            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)

            # Encode via SSL + DAC (runs on GPU)
            try:
                stream0, streams18 = encoder.encode_audio_tokens(audio_np)
                torch.cuda.synchronize()
            except Exception as e:
                logger.error(
                    "encode_audio_tokens failed for item %d "
                    "(n_samples=%d): %s", i, n_samples, e,
                )
                raise
            T_enc = len(stream0)

            is_sys = (
                audio_is_system is not None
                and (bool(audio_is_system[i]) if hasattr(audio_is_system, '__getitem__') else bool(audio_is_system))
            )

            if is_sys:
                # Pad/trim to speaker_prompt_length
                spk_len = cfg.speaker_prompt_length
                if T_enc >= spk_len:
                    stream0 = stream0[:spk_len]
                    streams18 = streams18[:spk_len]
                else:
                    pad_s0 = np.zeros(spk_len - T_enc, dtype=np.int64)
                    pad_s18 = np.zeros(
                        (spk_len - T_enc, 8), dtype=np.int64
                    )
                    stream0 = np.concatenate([stream0, pad_s0])
                    streams18 = np.concatenate([streams18, pad_s18])
                T_audio = spk_len
            else:
                T_audio = T_enc

            # Inter-segment padding
            T_total = T_audio + inter_pad
            pad_s0 = np.zeros(inter_pad, dtype=np.int64)
            pad_s18 = np.zeros((inter_pad, 8), dtype=np.int64)
            stream0 = np.concatenate([stream0, pad_s0])
            streams18 = np.concatenate([streams18, pad_s18])

            # Adjust to match expected embed length from preprocessor
            # (ceil may differ from actual encoder output by ±1 frame)
            if embed_lengths is not None:
                if hasattr(embed_lengths, '__getitem__') and (not isinstance(embed_lengths, torch.Tensor) or embed_lengths.dim() > 0):
                    expected = int(embed_lengths[i])
                else:
                    expected = int(embed_lengths)
            else:
                expected = T_total
            if T_total != expected:
                logger.debug(
                    "Audio item %d: T_total=%d != expected=%d, adjusting",
                    i, T_total, expected,
                )
                if T_total > expected:
                    stream0 = stream0[:expected]
                    streams18 = streams18[:expected]
                else:
                    extra = expected - T_total
                    stream0 = np.concatenate(
                        [stream0, np.zeros(extra, dtype=np.int64)]
                    )
                    streams18 = np.concatenate(
                        [streams18,
                         np.zeros((extra, 8), dtype=np.int64)]
                    )
                T_total = expected

            # Bounds-check token IDs
            s0_max = int(stream0.max()) if len(stream0) > 0 else 0
            s18_max = int(streams18.max()) if streams18.size > 0 else 0
            if s0_max >= V or s18_max >= V:
                logger.error(
                    "Token OOB! item=%d s0_max=%d s18_max=%d V=%d "
                    "T_total=%d is_sys=%s n_samples=%d",
                    i, s0_max, s18_max, V, T_total, is_sys, n_samples,
                )

            # Convert to tensors and delay-interleave
            s0 = torch.from_numpy(stream0).long().to(device)
            s18 = torch.from_numpy(streams18).long().to(device)

            # Clamp to valid range to prevent CUDA assert on embed lookup
            s0 = s0.clamp(0, V - 1)
            s18 = s18.clamp(0, V - 1)

            delayed_s18 = torch.zeros_like(s18)  # (T_total, 8)
            for k in range(s18.shape[1]):
                delay = k + 1
                if delay < T_total:
                    delayed_s18[delay:, k] = s18[:T_total - delay, k]

            # Embed and sum all 9 streams
            try:
                s0_embed = embed_fn(s0)
                torch.cuda.synchronize()
                s18_embed = embed_fn(delayed_s18)
                torch.cuda.synchronize()
            except Exception as e:
                logger.error(
                    "embed_fn failed for item %d: s0 range=[%d,%d] "
                    "s18 range=[%d,%d] V=%d shape_s0=%s shape_s18=%s: %s",
                    i, int(s0.min()), int(s0.max()),
                    int(delayed_s18.min()), int(delayed_s18.max()),
                    V, s0.shape, delayed_s18.shape, e,
                )
                raise
            combined = s0_embed + s18_embed.sum(dim=1)
            mm_embeddings.append(combined)
            logger.debug(
                "encode_embed item %d/%d: n_samples=%d T_enc=%d "
                "is_sys=%s expected=%d actual=%d",
                i, len(sample_lens), n_samples, T_enc,
                is_sys, expected, combined.shape[0],
            )

        logger.debug(
            "encode_embed total: %d items, %d total embeddings",
            len(mm_embeddings),
            sum(e.shape[0] for e in mm_embeddings),
        )
        return tuple(mm_embeddings)

    # ------------------------------------------------------------------
    # Multimodal embedding interface
    # ------------------------------------------------------------------
    def _parse_and_validate_audio_stream_inputs(
        self, **kwargs: object
    ) -> dict[str, torch.Tensor] | None:
        stream0_ids = kwargs.get("audio_stream0_ids")
        streams18 = kwargs.get("audio_streams18")
        audio_lengths = kwargs.get("audio_lengths")
        if stream0_ids is None or streams18 is None or audio_lengths is None:
            return None

        if not isinstance(stream0_ids, torch.Tensor):
            stream0_ids = torch.as_tensor(stream0_ids)
        if not isinstance(streams18, torch.Tensor):
            streams18 = torch.as_tensor(streams18)
        if not isinstance(audio_lengths, torch.Tensor):
            audio_lengths = torch.as_tensor(audio_lengths)

        return {
            "audio_stream0_ids": stream0_ids.long(),
            "audio_streams18": streams18.long(),
            "audio_lengths": audio_lengths.long(),
        }

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        """Compute combined (stream0 + streams1-8) embeddings for audio.

        Two paths:
          1. Raw audio (input_audio_features present): encode via SSL+DAC
             on the GPU, then embed. This is the primary path for real
             audio, enabling batched encoding via vLLM's mm encoder
             and encoder cache hits for cascade stages.
          2. Pre-computed token IDs (audio_stream0_ids present): delay
             interleave and embed directly. Used for pre-tokenized ARK
             data (input_tokens path).

        Returns one embedding tensor per audio segment. These are merged
        into the input at positions marked by is_multimodal in embed_input_ids.
        """
        # Path 1: raw audio → encode on GPU → embed
        if kwargs.get("input_audio_features") is not None:
            return self._encode_and_embed_audio(**kwargs)

        # Path 2: pre-computed token IDs → delay interleave → embed
        mm_audio = self._parse_and_validate_audio_stream_inputs(**kwargs)
        if mm_audio is None:
            return []

        stream0_ids = mm_audio["audio_stream0_ids"]
        streams18 = mm_audio["audio_streams18"]
        audio_lengths = mm_audio["audio_lengths"]

        if stream0_ids.numel() == 0 or audio_lengths.numel() == 0:
            return []

        device = next(self.model.parameters()).device
        stream0_ids = stream0_ids.to(device)
        streams18 = streams18.to(device)
        audio_lengths = audio_lengths.to(device)

        embed_fn = self.model.embed_tokens
        mm_embeddings = []
        offset = 0
        for length in audio_lengths.tolist():
            length = int(length)
            if length <= 0:
                mm_embeddings.append(
                    torch.empty((0, self.config.hidden_size), device=device)
                )
                continue

            s0 = stream0_ids[offset:offset + length]
            s18 = streams18[offset:offset + length]
            offset += length

            # Apply delay interleaving to streams 1-8:
            # Stream k+1 (s18[:, k]) has delay (k+1) positions.
            # At position t, delayed stream k+1 sees original frame (t - k - 1).
            # Positions 0..(k) are pad (0).
            delayed_s18 = torch.zeros_like(s18)  # (T, 8)
            for k in range(s18.shape[1]):
                delay = k + 1
                if delay < length:
                    delayed_s18[delay:, k] = s18[:length - delay, k]

            s0_embed = embed_fn(s0)
            s18_embed = embed_fn(delayed_s18)
            # NOTE: Do NOT mask pad(0) embeddings to zero. In ESPnet training,
            # emb(pad=0) is a learned embedding that contributes to the sum
            # at every position. Zeroing it creates a systematic bias.
            mm_embeddings.append(s0_embed + s18_embed.sum(dim=1))


        return tuple(mm_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input tokens with multi-stream support for audio.

        During prefill, combined (s0+s18) multimodal embeddings from
        embed_multimodal() are merged at is_multimodal positions.
        During decode, _stream_buffer_dict provides per-step DAC data.

        ESPnet trains with emb(x).sum(dim=nq), meaning ALL nq stream
        embeddings contribute — including pad(0) for inactive streams.
        To match this:
          - embed_multimodal: returns sum of all 9 stream embeddings
            (including emb(0) for delayed/pad positions)
          - _apply_stream_embeddings: adds stream 1-8 embeddings (including
            emb(0) for pad) during decode
          - This method: adds (nq-1)*emb(0) pad bias to text positions
            where streams 1-8 are implicitly all pad
        """
        N = input_ids.shape[0]

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        # Track which positions have real stream embeddings injected
        has_real_streams = torch.zeros(
            inputs_embeds.shape[0], dtype=torch.bool,
            device=inputs_embeds.device,
        )

        # ----- Prefill: merge multimodal audio embeddings -----
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            if is_multimodal is None:
                raise ValueError(
                    "`embed_input_ids` requires `is_multimodal` when "
                    "multimodal embeddings are provided."
                )

            # Validate shapes before merge
            num_mm_positions = int(is_multimodal.sum().item())
            total_mm_tokens = sum(
                e.shape[0] for e in multimodal_embeddings
            )
            if num_mm_positions != total_mm_tokens:
                logger.error(
                    "SHAPE MISMATCH in embed_input_ids: "
                    "is_multimodal has %d True positions but "
                    "multimodal_embeddings has %d total tokens "
                    "(from %d items: %s). input_ids shape=%s",
                    num_mm_positions,
                    total_mm_tokens,
                    len(multimodal_embeddings),
                    [e.shape for e in multimodal_embeddings],
                    input_ids.shape,
                )

            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            has_real_streams |= is_multimodal

        # ----- Decode: inject stream1-8 from per-request buffers -----
        if self._stream_buffer_dict:
            stream_embed_positions = self._get_stream_embed_positions(input_ids)
            if stream_embed_positions.any():
                inputs_embeds = self._apply_stream_embeddings(
                    input_ids, inputs_embeds, stream_embed_positions
                )
                has_real_streams |= stream_embed_positions

        # ----- Add pad bias to text positions -----
        # ESPnet: emb(x).sum(dim=nq) includes emb(pad=0) for inactive
        # streams. For text positions (streams 1-8 all pad), this adds
        # (nq-1) * emb(0) to the embedding. We match this here.
        text_positions = ~has_real_streams
        if text_positions.any():
            embed_fn = self.model.embed_tokens
            pad_embed = embed_fn(
                torch.zeros(1, dtype=torch.long,
                            device=inputs_embeds.device)
            )  # (1, hidden)
            nq_minus_1 = self.config.nq - 1  # 8
            inputs_embeds = inputs_embeds + (
                text_positions.float().unsqueeze(-1)
                * pad_embed
                * nq_minus_1
            )

        return inputs_embeds

    def _get_stream_embed_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return positions that should include buffered streams 1-8 embeddings."""
        positions = torch.zeros_like(input_ids, dtype=torch.bool)
        batch_rids = self._current_batch_req_ids
        if batch_rids:
            for pos, req_id in enumerate(batch_rids):
                if pos >= positions.numel():
                    break
                if req_id not in self._stream_buffer_dict:
                    continue
                rc = self._per_req_config.get(req_id, {})
                phase = rc.get("phase", "text")
                if phase in ("audio", "audio_flush", "audio_stop"):
                    positions[pos] = True
            return positions

        cfg = self.config
        return (
            (input_ids >= cfg.ssl_token_start)
            & (input_ids < cfg.ssl_token_end)
        )

    def _apply_stream_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        is_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Add streams 1-8 (DAC) embeddings for audio-mode positions."""
        audio_indices = is_audio.nonzero(as_tuple=True)[0]
        if len(audio_indices) == 0:
            return inputs_embeds

        embed_fn = self.model.embed_tokens
        batch_rids = self._current_batch_req_ids

        buf_rows = []
        valid_positions = []
        for pos in audio_indices.tolist():
            if pos < len(batch_rids):
                req_id = batch_rids[pos]
                buf_vec = self._stream_buffer_dict.get(req_id)
                if buf_vec is not None:
                    buf_rows.append(buf_vec)
                    valid_positions.append(pos)

        if not buf_rows:
            return inputs_embeds

        stream_tokens = torch.stack(buf_rows, dim=0)
        stream_embeds = embed_fn(stream_tokens)
        # NOTE: Do NOT mask pad(0) embeddings. ESPnet includes emb(pad)
        # in the sum — it's a learned embedding, not zero.
        stream_sum = stream_embeds.sum(dim=1)
        valid_idx = torch.tensor(
            valid_positions, device=inputs_embeds.device, dtype=torch.long
        )
        inputs_embeds[valid_idx] = inputs_embeds[valid_idx] + stream_sum

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

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    # ------------------------------------------------------------------
    # Logits computation with multi-stream sampling
    # ------------------------------------------------------------------
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute stream-0 logits with phase-aware masking.

        Phases per request (tracked in _per_req_config):
          "text"       -> text mask
          "pre_audio"  -> force codec_ssl_start_end (34) output
          "audio"      -> SSL mask for stream 0; sample DAC streams 1-8
          "audio_flush"-> delay flush (force pad=0 on stream0)
          "audio_stop" -> force EOS (5) to stop the request
        """
        cfg = self.config

        h0 = hidden_states + self.head_emb.weight[0].unsqueeze(0)
        stream0_logits = self.logits_processor(self.lm_head, h0)
        if stream0_logits is None:
            return None

        batch_rids = self._current_batch_req_ids
        N = stream0_logits.shape[0]

        phases = []
        for i in range(N):
            if i < len(batch_rids):
                rc = self._per_req_config.get(batch_rids[i], {})
                phases.append(rc.get("phase", "text"))
            else:
                phases.append("text")

        text_positions = []
        pre_audio_positions = []
        audio_positions = []
        audio_flush_positions = []
        audio_stop_positions = []

        for i, ph in enumerate(phases):
            if ph == "text":
                text_positions.append(i)
            elif ph == "pre_audio":
                pre_audio_positions.append(i)
            elif ph == "audio":
                audio_positions.append(i)
            elif ph == "audio_flush":
                audio_flush_positions.append(i)
            elif ph == "audio_stop":
                audio_stop_positions.append(i)
            else:
                text_positions.append(i)

        dev = stream0_logits.device

        if text_positions:
            idx = torch.tensor(text_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = stream0_logits[idx].masked_fill(
                self.text_mask_s0.unsqueeze(0), float("-inf")
            )

        if audio_positions:
            idx = torch.tensor(audio_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = stream0_logits[idx].masked_fill(
                self.audio_mask_s0.unsqueeze(0), float("-inf")
            )
            for pos in audio_positions:
                req_cfg = self._get_req_config(pos)
                audio_step = int(req_cfg.get("audio_step", 0))
                audio_minlen = int(
                    req_cfg.get("audio_minlen", getattr(cfg, "audio_minlen", 3))
                )
                if audio_step < max(audio_minlen, 0):
                    stream0_logits[pos, cfg.eos_token_id] = float("-inf")

        if pre_audio_positions:
            idx = torch.tensor(pre_audio_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = float("-inf")
            stream0_logits[idx, cfg.codec_ssl_start_end_token_id] = 0.0

        if audio_flush_positions:
            idx = torch.tensor(audio_flush_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = float("-inf")
            stream0_logits[idx, 0] = 0.0

        if audio_stop_positions:
            idx = torch.tensor(audio_stop_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = float("-inf")
            stream0_logits[idx, cfg.eos_token_id] = 0.0

        sample_positions = audio_positions + audio_flush_positions
        if sample_positions:
            self._sample_and_buffer_streams(
                hidden_states,
                sampled_positions=sample_positions,
                device=dev,
            )

        return stream0_logits

    def _sample_and_buffer_streams(
        self,
        hidden_states: torch.Tensor,
        sampled_positions: list[int],
        device: torch.device,
    ):
        """Sample DAC streams 1-8 for audio/flush positions and update buffer."""
        cfg = self.config
        if not sampled_positions:
            return
        audio_idx = torch.tensor(sampled_positions, device=device, dtype=torch.long)

        audio_hidden = hidden_states[audio_idx]
        num_audio = len(audio_idx)
        batch_rids = self._current_batch_req_ids

        row_phase = []
        row_flush_step = []
        row_audio_step = []
        for pos in sampled_positions:
            if pos < len(batch_rids):
                rc = self._per_req_config.get(batch_rids[pos], {})
            else:
                rc = {}
            row_phase.append(rc.get("phase", "audio"))
            row_flush_step.append(int(rc.get("flush_step", 0)))
            row_audio_step.append(int(rc.get("audio_step", 0)))

        new_buffer = torch.zeros(
            num_audio, cfg.num_codec_streams,
            dtype=torch.long, device=device
        )

        for s in range(1, cfg.nq):
            head_bias = self.head_emb.weight[s].unsqueeze(0)
            h_s = audio_hidden + head_bias

            s_logits = self.logits_processor(self.lm_head, h_s)
            if s_logits is None:
                continue

            s_logits = s_logits.masked_fill(
                self._audio_masks[s].unsqueeze(0), float("-inf")
            )

            sampled = torch.zeros(num_audio, dtype=torch.long, device=device)
            for row in range(num_audio):
                if (
                    row_phase[row] == "audio"
                    and row_audio_step[row] < s
                ):
                    sampled[row] = 0
                    continue
                if (
                    row_phase[row] == "audio_flush"
                    and row_flush_step[row] > s
                ):
                    sampled[row] = 0
                    continue

                req_cfg = self._get_req_config(sampled_positions[row])
                temperature = float(
                    req_cfg.get("audio_temperature", cfg.audio_temperature)
                )
                top_k = int(req_cfg.get("audio_topk", cfg.audio_topk))
                if top_k <= 0:
                    top_k = cfg.vocab_size
                top_k = max(1, min(top_k, cfg.vocab_size))
                sampled[row] = self._top_k_sample(
                    s_logits[row:row + 1],
                    temperature=temperature,
                    top_k=top_k,
                )[0]
            new_buffer[:, s - 1] = sampled

        for j, pos in enumerate(sampled_positions):
            if pos < len(batch_rids):
                req_id = batch_rids[pos]
                buf_vec = new_buffer[j]
                self._stream_buffer_dict[req_id] = buf_vec.clone()
                self._stream18_history.setdefault(req_id, []).append(
                    buf_vec.clone()
                )

    def _top_k_sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 30,
    ) -> torch.Tensor:
        """Top-k sampling for a batch of logits."""
        if temperature == 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        topk_values, topk_indices = torch.topk(
            logits, min(top_k, logits.shape[-1]), dim=-1
        )
        probs = torch.softmax(topk_values, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return topk_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------
    # Per-request helpers
    # ------------------------------------------------------------------
    def _get_req_config(self, position_idx: int) -> dict:
        if position_idx < len(self._current_batch_req_ids):
            req_id = self._current_batch_req_ids[position_idx]
            return self._per_req_config.get(req_id, {})
        return {}

    def cleanup_request(self, req_id: str):
        """Remove all per-request state for a finished request."""
        self._per_req_config.pop(req_id, None)
        self._stream18_history.pop(req_id, None)
        self._stream0_history.pop(req_id, None)
        self._decoded_audio.pop(req_id, None)
        self._stream_buffer_dict.pop(req_id, None)

    # ------------------------------------------------------------------
    # Audio decode pipeline
    # ------------------------------------------------------------------
    def _get_dac_model(self):
        """Lazy-load the DAC model on first use."""
        if self._dac_model is None:
            try:
                from espnet2.bin.gan_codec_inference import AudioCoding
                self._dac_model = AudioCoding.from_pretrained(
                    self._dac_hf_model_tag
                ).model.eval()
                device = next(self.model.parameters()).device
                self._dac_model = self._dac_model.to(device)
                logger.info(
                    "Loaded DAC model from %s", self._dac_hf_model_tag
                )
            except Exception as e:
                logger.error("Failed to load DAC model: %s", e)
                raise
        return self._dac_model

    def _delay_deinterleave(self, codes: torch.Tensor) -> torch.Tensor:
        """Remove delay interleaving from 9-stream token tensor.

        Args:
            codes: [B, T, 9] delay-interleaved token tensor

        Returns:
            [B, T-8, 9] de-interleaved (aligned) tensor
        """
        _, T, N = codes.shape
        T_original = T - N + 1
        if T_original <= 0:
            return codes[:, :0, :]

        new_codes = []
        for n in range(N):
            new_codes.append(codes[:, n:n + T_original, n])
        return torch.stack(new_codes, dim=-1)

    def _global_to_dac_codebook(self, dac_tokens: torch.Tensor) -> torch.Tensor:
        """Convert global DAC token IDs to per-stream codebook indices [0, 1023]."""
        cfg = self.config
        result = dac_tokens.clone()
        for k in range(cfg.num_codec_streams):
            offset = cfg.codec_token_start + k * cfg.codec_per_stream_size
            result[..., k] = (result[..., k] - offset).clamp(0, 1023)
        return result

    @torch.inference_mode()
    def _dac_decode(self, codebook_indices: torch.Tensor) -> "Any":
        """Decode DAC codebook indices to audio waveform."""
        dac = self._get_dac_model()
        codes = codebook_indices.permute(2, 0, 1)  # [8, B, T]
        audio = dac.decode(codes)
        return audio.squeeze().cpu().numpy()

    def encode_audio_to_base64_wav(
        self, req_id: str, stream0_tokens: list[int]
    ) -> str | None:
        """Decode stream 0 SSL tokens + stream 1-8 DAC history to base64 WAV."""
        import base64
        import io
        import wave

        if not stream0_tokens:
            return None

        try:
            audio_np, sr = self.decode_audio_from_tokens(
                req_id, stream0_tokens
            )
            if len(audio_np) == 0:
                return None

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            logger.exception(
                "Failed to decode audio for request %s", req_id
            )
            return None

    def decode_audio_from_tokens(
        self,
        req_id: str,
        stream0_ssl_tokens: list[int],
    ) -> tuple["Any", int]:
        """Decode complete audio from stream 0 SSL + stream 1-8 DAC history.

        The stream18 history includes entries from:
          - audio phase (N_ssl steps: SSL tokens generated)
          - EOS step (1 step: model generated EOS on stream 0)
          - audio_flush phase (nq-1=8 steps: pad on stream 0)

        Total stream18 history = N_ssl + 9.  We need ALL of these for
        correct delay de-interleaving — the flush DAC tokens provide the
        trailing frames that stream k (delayed by k) still needs to output.
        """
        history = self._stream18_history.pop(req_id, [])

        N_ssl = len(stream0_ssl_tokens)
        if N_ssl == 0:
            return np.zeros(0, dtype=np.float32), self._dac_sample_rate

        device = next(self.model.parameters()).device
        H = len(history)

        if H == 0:
            # No DAC history at all — just build a minimal matrix
            s0 = torch.tensor(
                stream0_ssl_tokens, dtype=torch.long, device=device
            )
            s18 = torch.zeros(N_ssl, 8, dtype=torch.long, device=device)
            full_matrix = torch.cat(
                [s0.unsqueeze(1), s18], dim=1
            ).unsqueeze(0)
        else:
            # Use the FULL stream18 history (including EOS step + flush).
            # Pad stream0 with zeros to match the total length.
            T = max(N_ssl, H)
            s0_padded = torch.zeros(T, dtype=torch.long, device=device)
            s0_padded[:N_ssl] = torch.tensor(
                stream0_ssl_tokens, dtype=torch.long, device=device
            )
            s18_stack = torch.stack(history, dim=0).to(device)
            if H < T:
                pad = torch.zeros(T - H, 8, dtype=torch.long, device=device)
                s18_stack = torch.cat([s18_stack, pad], dim=0)
            else:
                s18_stack = s18_stack[:T]

            full_matrix = torch.cat(
                [s0_padded.unsqueeze(1), s18_stack], dim=1
            ).unsqueeze(0)  # [1, T, 9]

        aligned = self._delay_deinterleave(full_matrix)

        if aligned.shape[1] == 0:
            return np.zeros(0, dtype=np.float32), self._dac_sample_rate

        # After deinterleaving, we may have more frames than N_ssl
        # (due to EOS/flush entries). Trim to N_ssl real audio frames.
        n_frames = min(aligned.shape[1], N_ssl)
        aligned = aligned[:, :n_frames, :]


        dac_tokens = aligned[:, :, 1:]
        dac_cb = self._global_to_dac_codebook(dac_tokens)
        audio = self._dac_decode(dac_cb)
        return audio, self._dac_sample_rate

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights from safetensors.

        Expected weight names (after opuslm_dialogue_convert.py remapping):
          model.embed_tokens.weight    [62670, 2048]
          model.norm.weight            [2048]
          model.layers.{i}.*           (Llama layer weights)
          lm_head.weight               [62670, 2048]
          head_emb.weight              [12, 2048]
        """
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.startswith("model."):
                if is_pp_missing_parameter(name, self.model):
                    continue
                matched = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    remapped = name.replace(weight_name, param_name)
                    if remapped.endswith(".bias") and remapped not in params_dict:
                        matched = True
                        break
                    if remapped in params_dict:
                        param = params_dict[remapped]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(remapped)
                        matched = True
                        break
                if matched:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

            elif name == "lm_head.weight":
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

            elif name == "head_emb.weight":
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params

    # ------------------------------------------------------------------
    # Multi-model key mapping (for pipeline parallelism)
    # ------------------------------------------------------------------
    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model",
        )
