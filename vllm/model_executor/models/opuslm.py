# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only OpusLM model (OLMo-2-7B based multimodal speech-language
model with 9-stream delay-interleaved discrete codec output).

Features:
  - Text output (standard autoregressive)
  - Audio output via 9 streams (1 SSL + 8 DAC) with delay interleaving
  - Internal multi-stream sampling (streams 1-8 sampled inside compute_logits)
  - Audio input support via XEUS + K-means tokenization (Phase 2)

Architecture differences from SpeechLM (Bagpiper):
  - Base LM: OLMo-2-7B (not Qwen3-8B) — uses q_norm/k_norm, no GQA
  - Audio codec: DAC large (not Xcodec)
  - No CFG support
  - Vocab: 113870 (not 160392)
  - EOS=5, no EOT concept
  - head_emb.weight [12, 4096] per-stream bias embedding
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
from vllm.transformers_utils.configs.opuslm import OpusLMConfig

logger = init_logger(__name__)

_AUDIO_SAMPLING_RATE = 16000


# ---------------------------------------------------------------------------
# Multimodal processing (Phase 2 — audio input)
# ---------------------------------------------------------------------------

class _OpusLMProcessor:
    """Minimal processor for OpusLM: no feature extractor, just a placeholder."""

    def __init__(self):
        self.audio_token = "<codec_ssl_start_end>"

    def get_vocab(self) -> dict[str, int]:
        # The placeholder token id for audio in the prompt
        return {self.audio_token: 34}  # codec_ssl_start_end_token_id


class _OpusLMAudioInputProcessor:
    """Server-side audio tokenizer: waveform -> (SSL stream0, DAC streams1-8).

    This follows the pipeline required by OpusLM:
      1) DAC encoder -> 8 codec streams
      2) XEUS SSL model + KMeans -> SSL stream
      3) Align lengths and map to OpusLM global token IDs
    """

    def __init__(self, cfg: OpusLMConfig):
        self.cfg = cfg
        self._dac_model = None
        self._ssl_model = None
        self._ssl_layer = int(getattr(cfg, "xeus_layer", 18))
        self._kmeans_model = None

    def _load_dac_model(self):
        if self._dac_model is not None:
            return self._dac_model

        try:
            from espnet2.bin.gan_codec_inference import AudioCoding
        except Exception as e:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "Failed to import ESPnet AudioCoding for DAC encoding. "
                "Please ensure ESPnet runtime dependencies are installed."
            ) from e

        self._dac_model = AudioCoding.from_pretrained(
            self.cfg.dac_hf_model_tag
        ).model.eval()
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
        except Exception as e:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "Failed to import SSL dependencies (joblib/espnet2.tasks.ssl). "
                "Please install required ESPnet dependencies (e.g. torch_complex)."
            ) from e

        ckpt_path, km_path = self._resolve_xeus_paths()
        self._ssl_model, _ = SSLTask.build_model_from_file(
            None, ckpt_path, device="cpu"
        )
        self._ssl_model.eval()
        # Disable masking so encode() produces deterministic features
        # (matches ESPnet data prep which uses use_mask=False).
        if hasattr(self._ssl_model, "util_attributes"):
            self._ssl_model.util_attributes.discard("mask")
            self._ssl_model.util_attributes.discard("block_mask")
        self._kmeans_model = joblib.load(km_path)
        return self._ssl_model, self._kmeans_model

    def _extract_ssl_labels(
        self,
        audio_np: np.ndarray,
    ) -> np.ndarray:
        ssl_model, km_model = self._load_ssl_and_kmeans()

        wav = torch.from_numpy(audio_np).float().view(1, -1)
        wav_lens = torch.tensor([wav.shape[1]], dtype=torch.long)
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

        ssl_feats_np = ssl_feats[0].detach().cpu().numpy()
        labels = km_model.predict(ssl_feats_np).astype(np.int64)
        max_ssl_id = self.cfg.ssl_token_end - self.cfg.ssl_token_start - 1
        return np.clip(labels, 0, max_ssl_id)

    def _extract_codec_codes(
        self,
        audio_np: np.ndarray,
    ) -> np.ndarray:
        dac = self._load_dac_model()
        wav = torch.from_numpy(audio_np).float().view(1, 1, -1)
        with torch.inference_mode():
            codes = dac.encode(wav)
            # [n_q, B, T] -> [B, T, n_q]
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


class _OpusLMGPUAudioInputProcessor:
    """Server-side GPU audio tokenizer: waveform -> (SSL stream0, DAC streams1-8).

    Same pipeline as _OpusLMAudioInputProcessor but loads models on GPU for
    fast encoding during the model forward pass.
    """

    def __init__(self, cfg: OpusLMConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = device
        self._dac_model = None
        self._ssl_model = None
        self._ssl_layer = int(getattr(cfg, "xeus_layer", 18))
        self._kmeans_model = None
        self._km_centroids = None

    def to(self, device: torch.device | str) -> "_OpusLMGPUAudioInputProcessor":
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
        import os
        # Support local paths: if xeus_local_checkpoint / km_local_path
        # are set and point to existing files, use them directly.
        local_ckpt = getattr(self.cfg, "xeus_local_checkpoint", None)
        local_km = getattr(self.cfg, "km_local_path", None)
        if local_ckpt and os.path.isfile(local_ckpt):
            ckpt_path = local_ckpt
        else:
            from huggingface_hub import hf_hub_download
            repo = self.cfg.xeus_hf_model_tag
            ckpt_file = getattr(
                self.cfg, "xeus_checkpoint_filename",
                "model/xeus_checkpoint_new.pth",
            )
            ckpt_path = hf_hub_download(repo, ckpt_file)
        if local_km and os.path.isfile(local_km):
            km_path = local_km
        else:
            from huggingface_hub import hf_hub_download
            repo = self.cfg.xeus_hf_model_tag
            km_path = hf_hub_download(repo, self.cfg.km_model_filename)
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
        # Disable masking so encode() produces deterministic features
        # (matches ESPnet data prep which uses use_mask=False).
        if hasattr(self._ssl_model, "util_attributes"):
            self._ssl_model.util_attributes.discard("mask")
            self._ssl_model.util_attributes.discard("block_mask")
        logger.info("Loaded XEUS SSL model on %s", self.device)
        self._kmeans_model = joblib.load(km_path)
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
        if isinstance(enc_out, dict):
            feats = enc_out["encoder_output"]
        else:
            feats = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        if isinstance(feats, (list, tuple)):
            layer = min(max(self._ssl_layer, 0), len(feats) - 1)
            ssl_feats = feats[layer]
        else:
            ssl_feats = feats
        if hasattr(self, '_km_centroids') and self._km_centroids is not None:
            feats_2d = ssl_feats[0]
            distances = torch.cdist(
                feats_2d.unsqueeze(0),
                self._km_centroids.unsqueeze(0),
            )
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


class OpusLMProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> OpusLMConfig:
        return self.ctx.get_hf_config(OpusLMConfig)

    def get_hf_processor(self, **kwargs: object) -> _OpusLMProcessor:
        return _OpusLMProcessor()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # Avoid expensive dummy-audio profiling at engine startup.
        # Audio placeholders in a prompt cannot exceed the sequence budget.
        return {"audio": max(1, int(seq_len))}

    def get_data_parser(self) -> MultiModalDataParser:
        return OpusLMMultiModalDataParser(
            target_sr=_AUDIO_SAMPLING_RATE,
            target_channels=1,
        )


class OpusLMMultiModalDataParser(MultiModalDataParser):

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        return super()._parse_audio_data(data)


class OpusLMDummyInputsBuilder(
    BaseDummyInputsBuilder[OpusLMProcessingInfo]
):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Dummy text — _call_hf_processor ignores this when it has audio.
        # The framework needs placeholder token [34] in the final input_ids
        # to apply prompt replacements; _call_hf_processor inserts [34]
        # directly via _layout_task_sequence for audio-bearing sequences.
        return "dummy"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        import numpy as np
        num_audios = mm_counts.get("audio", 0)
        # 1 second of silence at 16kHz
        dummy_audio = np.zeros(_AUDIO_SAMPLING_RATE, dtype=np.float32)
        return {
            "audio": [dummy_audio] * num_audios,
        }


class OpusLMMultiModalProcessor(
    BaseMultiModalProcessor[OpusLMProcessingInfo],
):
    """Handles server-side OpusLM audio tokenization and prompt replacement."""

    @staticmethod
    def resolve_task_token_id(
        cfg: OpusLMConfig,
        *,
        has_audio_input: bool,
        mode: str | None = None,
        task: str | int | None = None,
    ) -> int:
        """Resolve request task to OpusLM task token ID.

        Supported tasks:
          - ASR     -> <codec_ssl_asr_task>
          - TTS     -> <codec_ssl_tts_task>
          - Plain TTS -> <codec_ssl_plain_tts_task>
          - Text LM -> <textlm_task>
        """
        if isinstance(task, int):
            return int(task)

        task_aliases = {
            "asr": cfg.codec_ssl_asr_task_token_id,
            "codec_ssl_asr": cfg.codec_ssl_asr_task_token_id,
            "codec_ssl_asr_task": cfg.codec_ssl_asr_task_token_id,
            "tts": cfg.codec_ssl_tts_task_token_id,
            "codec_ssl_tts": cfg.codec_ssl_tts_task_token_id,
            "codec_ssl_tts_task": cfg.codec_ssl_tts_task_token_id,
            "plain_tts": cfg.codec_ssl_plain_tts_task_token_id,
            "codec_ssl_plain_tts": cfg.codec_ssl_plain_tts_task_token_id,
            "codec_ssl_plain_tts_task": cfg.codec_ssl_plain_tts_task_token_id,
            "textlm": cfg.textlm_task_token_id,
            "text_lm": cfg.textlm_task_token_id,
            "olmo_textlm": cfg.textlm_task_token_id,
            "textlm_task": cfg.textlm_task_token_id,
            # Dialogue tasks
            "audio_dialogue": cfg.audio_dialogue_task_token_id,
            "audio_dialogue_task": cfg.audio_dialogue_task_token_id,
            "text_dialogue": cfg.text_dialogue_task_token_id,
            "text_dialogue_task": cfg.text_dialogue_task_token_id,
        }
        if isinstance(task, str):
            task_norm = task.strip().lower()
            if task_norm in task_aliases:
                return task_aliases[task_norm]
            raise ValueError(
                f"Unsupported OpusLM task '{task}'. "
                "Supported: asr, tts, plain_tts, textlm, "
                "audio_dialogue, text_dialogue."
            )

        mode_norm = mode.strip().lower() if isinstance(mode, str) else None
        if mode_norm is not None and mode_norm not in (
            "text_audio",
            "audio_text",
            "text_text",
            "audio_dialogue",
            "text_dialogue",
        ):
            raise ValueError(
                f"Unsupported OpusLM mode '{mode}'. "
                "Supported: text_audio, audio_text, text_text, "
                "audio_dialogue, text_dialogue."
            )

        if mode_norm == "audio_dialogue":
            return cfg.audio_dialogue_task_token_id
        if mode_norm == "text_dialogue":
            return cfg.text_dialogue_task_token_id
        if mode_norm == "audio_text":
            return cfg.codec_ssl_asr_task_token_id
        if mode_norm == "text_text":
            return cfg.textlm_task_token_id
        if mode_norm == "text_audio":
            if has_audio_input:
                return cfg.codec_ssl_tts_task_token_id
            return cfg.codec_ssl_plain_tts_task_token_id
        if has_audio_input:
            return cfg.codec_ssl_asr_task_token_id
        return cfg.codec_ssl_plain_tts_task_token_id

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

    @classmethod
    def _layout_task_sequence(
        cls,
        text_inputs: BatchFeature,
        cfg: OpusLMConfig,
        task_token_id: int,
    ) -> BatchFeature:
        """Build task-aligned prompt sequence:

        <sos/eos> <task_token> <conditions...> <target_modality_prefix>
        """
        input_ids = text_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            return text_inputs

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return text_inputs

        sos = int(getattr(cfg, "sos_eos_token_id", cfg.eos_token_id))
        body_ids = input_ids[0].tolist()

        known_task_ids = {
            int(cfg.textlm_task_token_id),
            int(cfg.codec_ssl_asr_task_token_id),
            int(cfg.codec_ssl_tts_task_token_id),
            int(getattr(cfg, "codec_ssl_plain_tts_task_token_id", 82)),
            int(getattr(cfg, "codec_ssl_audiolm_task_token_id", 83)),
        }
        if len(body_ids) >= 2 and body_ids[0] == sos and body_ids[1] in known_task_ids:
            body_ids = body_ids[2:]

        seq = [sos, int(task_token_id)] + body_ids
        tts_task_ids = {
            int(cfg.codec_ssl_tts_task_token_id),
            int(getattr(cfg, "codec_ssl_plain_tts_task_token_id", 82)),
        }

        if task_token_id in tts_task_ids:
            # NOTE: the OpusLMTokenizer already shifts text IDs by
            # +text_token_start (13448) during encode/call, so body_ids
            # arriving here are already in the model's global ID space.
            # Do NOT apply the offset again.

            # Condition segment: <text_bpe_start/end> + text payload.
            if len(body_ids) == 0 or body_ids[0] != cfg.text_bpe_start_end_token_id:
                seq = [sos, int(task_token_id), cfg.text_bpe_start_end_token_id] + body_ids
            # Inter-segment padding after text condition (ESPnet: nq-1 = 8
            # zero frames to prevent DAC overlap after delay interleaving).
            inter_pad = int(getattr(cfg, "nq", 9)) - 1  # 8
            seq.extend([0] * inter_pad)
            # Target segment prefix: <codec_ssl_start/end>.
            if seq[-1] != cfg.codec_ssl_start_end_token_id:
                seq.append(cfg.codec_ssl_start_end_token_id)
        elif task_token_id == int(cfg.codec_ssl_asr_task_token_id):
            # Target text segment prefix.
            if seq[-1] != cfg.text_bpe_start_end_token_id:
                seq.append(cfg.text_bpe_start_end_token_id)

        new_input_ids = torch.tensor(
            [seq],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return cls._set_input_ids(text_inputs, new_input_ids)

    def _get_audio_input_processor(self) -> _OpusLMGPUAudioInputProcessor:
        cfg = self.info.get_hf_config()
        processor = getattr(self, "_audio_input_processor", None)
        if processor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = _OpusLMGPUAudioInputProcessor(cfg, device=device)
            self._audio_input_processor = processor
        return processor

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Tokenize prompt text and server-side audio into stream tokens."""
        tokenizer = self.info.get_tokenizer()

        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        cfg = self.info.get_hf_config()
        mode_obj = mm_kwargs.get("mode")
        task_obj = mm_kwargs.get("task")
        _pre_tokens = mm_kwargs.get("pre_tokens")
        _has_pre = bool(
            _pre_tokens and isinstance(_pre_tokens, (list, tuple))
        )
        task_token_id = self.resolve_task_token_id(
            cfg,
            has_audio_input=bool(audios) or _has_pre,
            mode=mode_obj if isinstance(mode_obj, str) else None,
            task=task_obj if isinstance(task_obj, (str, int)) else None,
        )

        # --- Dialogue path: structured messages ---
        # Only use the dialogue sequence builder for dialogue tasks (which
        # need role tokens like <user>, <assistant>, <system>).  Non-dialogue
        # tasks (ASR, TTS, plain_tts, audiolm) must go through the single-
        # turn path to produce the correct ESPnet sequence without role tokens.
        _dialogue_task_ids = {
            int(getattr(cfg, "audio_dialogue_task_token_id", 89)),
            int(getattr(cfg, "text_dialogue_task_token_id", 88)),
        }
        _is_dialogue_task = int(task_token_id) in _dialogue_task_ids

        messages = mm_kwargs.get("messages")
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

        if _is_dialogue_task and messages and isinstance(messages, (list, tuple)):
            mode = mode_obj if isinstance(mode_obj, str) else None
            return self._build_dialogue_sequence(
                cfg, tokenizer, task_token_id, messages, audios, mode=mode
            )

        # For non-dialogue tasks, extract text and audio from messages
        # so they can be processed through the single-turn path.
        if messages and isinstance(messages, (list, tuple)):
            _text_parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    if content:
                        _text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                _text_parts.append(part.get("text", ""))
                            elif part.get("type") == "input_audio":
                                pass  # audios already extracted above
            prompt = " ".join(_text_parts) if _text_parts else prompt

        # --- Single-turn path (existing) ---
        text_inputs = tokenizer(prompt, return_tensors="pt")
        text_inputs = self._layout_task_sequence(
            text_inputs, cfg, task_token_id
        )

        pre_tokens = mm_kwargs.get("pre_tokens")
        has_pre_tokens = bool(
            pre_tokens and isinstance(pre_tokens, (list, tuple))
        )
        if not audios and not has_pre_tokens:
            return BatchFeature(data={**text_inputs})

        # Build clean OpusLM task sequence: the chat template produces
        # GPT2 token IDs (e.g. for "<|user|><audio>") that are meaningless
        # to the OpusLM model.  Strip them and construct the correct
        # sequence: [sos, task, placeholder(34), text_bpe_start(35)]
        ph = int(cfg.codec_ssl_start_end_token_id)  # 34
        tbs = int(cfg.text_bpe_start_end_token_id)  # 35
        input_ids = text_inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
            ids = input_ids[0].tolist()
            sos = int(getattr(cfg, "sos_eos_token_id", cfg.eos_token_id))
            if ph not in ids:
                # Rebuild: [sos, task, placeholder×N, text_bpe_start]
                n_audio = max(len(audios), len(pre_tokens) if has_pre_tokens else 0)
                ids = [sos, int(task_token_id)]
                for _ in range(n_audio):
                    ids.append(ph)
                ids.append(tbs)
                text_inputs = self._set_input_ids(
                    text_inputs,
                    torch.tensor([ids], dtype=input_ids.dtype,
                                 device=input_ids.device),
                )

        stream0_chunks = list[torch.Tensor]()
        stream18_chunks = list[torch.Tensor]()
        lengths = list[int]()

        # ESPnet inter_segment_pad = nq - 1 = 8: zero rows after each audio
        # segment to prevent DAC token overlap after delay interleaving.
        inter_pad = int(cfg.nq) - 1  # 8

        codec_ssl_end_id = int(cfg.codec_ssl_start_end_token_id)  # 34

        # Pre-tokenized path: use ARK data instead of on-the-fly encoding
        pre_tokens = mm_kwargs.get("pre_tokens")
        if pre_tokens and isinstance(pre_tokens, (list, tuple)):
            token_bias = int(mm_kwargs.get("token_bias", 0))
            for pt in pre_tokens:
                s0 = np.array(pt.get("stream0", []), dtype=np.int64)
                s18_raw = pt.get("streams18", [])
                if s18_raw and isinstance(s18_raw[0], list):
                    s18 = np.array(s18_raw, dtype=np.int64)
                else:
                    s18 = np.zeros((len(s0), 8), dtype=np.int64)
                if token_bias != 0:
                    s0 = s0 + token_bias
                    s18 = s18 + token_bias
                T = len(s0)
                if T > 0:
                    end_s0 = np.array([codec_ssl_end_id], dtype=np.int64)
                    end_s18 = np.zeros(
                        (1, cfg.num_codec_streams), dtype=np.int64
                    )
                    pad_s0 = np.zeros(inter_pad, dtype=np.int64)
                    pad_s18 = np.zeros(
                        (inter_pad, cfg.num_codec_streams), dtype=np.int64
                    )
                    s0 = np.concatenate([s0, end_s0, pad_s0])
                    s18 = np.concatenate([s18, end_s18, pad_s18])
                    lengths.append(T + 1 + inter_pad)
                    stream0_chunks.append(torch.from_numpy(s0))
                    stream18_chunks.append(torch.from_numpy(s18))
                else:
                    lengths.append(0)
        else:
            # Detect whether this is a TTS task (speaker audio needs
            # fixed-length padding, NOT codec_ssl_end terminator).
            is_tts = int(task_token_id) in {
                int(cfg.codec_ssl_tts_task_token_id),
                int(getattr(cfg, "codec_ssl_plain_tts_task_token_id", 82)),
            }
            spk_len = int(getattr(cfg, "speaker_prompt_length", 500))

            audio_processor = self._get_audio_input_processor()
            for audio in audios:
                stream0, streams18 = audio_processor.encode_audio_tokens(audio)
                T = int(len(stream0))
                if T > 0:
                    if is_tts:
                        # TTS speaker reference: pad/clip to exactly
                        # speaker_prompt_length (500) frames, then add
                        # inter-segment padding.  No codec_ssl_end.
                        if T > spk_len:
                            stream0 = stream0[:spk_len]
                            streams18 = streams18[:spk_len]
                        elif T < spk_len:
                            pad_s0 = np.zeros(
                                spk_len - T, dtype=np.int64
                            )
                            pad_s18 = np.zeros(
                                (spk_len - T, cfg.num_codec_streams),
                                dtype=np.int64,
                            )
                            stream0 = np.concatenate([stream0, pad_s0])
                            streams18 = np.concatenate(
                                [streams18, pad_s18]
                            )
                        # Inter-segment padding after speaker segment
                        pad_s0 = np.zeros(inter_pad, dtype=np.int64)
                        pad_s18 = np.zeros(
                            (inter_pad, cfg.num_codec_streams),
                            dtype=np.int64,
                        )
                        stream0 = np.concatenate([stream0, pad_s0])
                        streams18 = np.concatenate(
                            [streams18, pad_s18]
                        )
                        lengths.append(spk_len + inter_pad)
                    else:
                        # ASR / other: audio + codec_ssl_end(34) + pad
                        end_s0 = np.array(
                            [codec_ssl_end_id], dtype=np.int64
                        )
                        end_s18 = np.zeros(
                            (1, cfg.num_codec_streams), dtype=np.int64
                        )
                        pad_s0 = np.zeros(inter_pad, dtype=np.int64)
                        pad_s18 = np.zeros(
                            (inter_pad, cfg.num_codec_streams),
                            dtype=np.int64,
                        )
                        stream0 = np.concatenate(
                            [stream0, end_s0, pad_s0]
                        )
                        streams18 = np.concatenate(
                            [streams18, end_s18, pad_s18]
                        )
                        lengths.append(T + 1 + inter_pad)
                    stream0_chunks.append(torch.from_numpy(stream0))
                    stream18_chunks.append(torch.from_numpy(streams18))
                else:
                    lengths.append(0)

        if stream0_chunks:
            stream0_tensor = torch.cat(stream0_chunks, dim=0).long()
            stream18_tensor = torch.cat(stream18_chunks, dim=0).long()
        else:
            stream0_tensor = torch.empty((0,), dtype=torch.long)
            stream18_tensor = torch.empty(
                (0, cfg.num_codec_streams), dtype=torch.long
            )

        audio_lengths = torch.tensor(lengths, dtype=torch.long)
        return BatchFeature(
            data={
                **text_inputs,
                "audio_stream0_ids": stream0_tensor,
                "audio_streams18": stream18_tensor,
                "audio_lengths": audio_lengths,
            }
        )

    def _build_dialogue_sequence(
        self,
        cfg: OpusLMConfig,
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
          <system>(8) <spk>(37) [speaker_audio x500] [pad x8]
          <user>(9) <codec_ssl>(34) [user_audio xT] [pad x8]
          <user>(9) <text_bpe>(35) [user_text...] [pad x8]
          <asst>(10) <text_bpe>(35) [asst_text...] [pad x8]
          <asst>(10) <codec_ssl>(34) [GENERATE FROM HERE...]
        """
        sos = int(cfg.sos_eos_token_id)
        inter_pad = int(cfg.nq) - 1  # 8
        seq: list[int] = [sos, int(task_token_id)]

        _needs_text_shift = not hasattr(tokenizer, 'text_token_offset')
        _text_offset = int(cfg.text_token_start) if _needs_text_shift else 0

        stream0_chunks: list[torch.Tensor] = []
        stream18_chunks: list[torch.Tensor] = []
        lengths: list[int] = []
        audio_is_system: list[bool] = []
        raw_audio_chunks: list[torch.Tensor] = []
        raw_audio_sample_counts: list[int] = []

        audio_idx = 0

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

            if isinstance(content, str):
                if content == "" and is_last and role == "assistant":
                    if mode == "audio_text":
                        gen_role_token = int(cfg.user_input_token_id)
                        gen_modality = int(cfg.text_bpe_start_end_token_id)
                    else:
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
                seq.extend([0] * inter_pad)

            elif isinstance(content, list):
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
                        seq.extend([0] * inter_pad)

                    elif part_type == "input_tokens":
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

                        stream0 = np.full(
                            T,
                            cfg.codec_ssl_start_end_token_id,
                            dtype=np.int64,
                        )
                        streams18 = np.zeros((T, 8), dtype=np.int64)

                        if is_system:
                            seq.append(cfg.spk_start_end_token_id)
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

                        pad_s0 = np.zeros(inter_pad, dtype=np.int64)
                        pad_s18 = np.zeros((inter_pad, 8), dtype=np.int64)
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

                        raw_audio_chunks.append(
                            torch.from_numpy(audio_np).float()
                        )
                        raw_audio_sample_counts.append(N_samples)

            else:
                seq.append(role_token)
                seq.append(cfg.text_bpe_start_end_token_id)
                text_ids = tokenizer.encode(
                    str(content), add_special_tokens=False
                )
                seq.extend(tid + _text_offset for tid in text_ids)
                seq.extend([0] * inter_pad)

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
        # _call_hf_processor places placeholder [34] in input_ids but does
        # NOT replace it with stream0 tokens.  Returning False tells the
        # framework to do the replacement via _apply_token_matches.
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
        # ["messages"], so we convert list[int] prompts to a dummy string.
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
        # Always use uncached path: dialogue sequences are context-dependent.
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
        }

        # Dialogue fields
        audio_is_system = hf_inputs.get("audio_is_system")
        if audio_is_system is not None:
            fields["audio_is_system"] = MultiModalFieldConfig.batched("audio")

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

        # Dialogue path: audio_is_system is present
        if audio_is_system is not None and isinstance(audio_is_system, torch.Tensor):
            placeholder_id = int(cfg.codec_ssl_start_end_token_id)  # 34

            def get_replacement_dialogue(item_idx: int):
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

                is_sys = (
                    item_idx < len(audio_is_system)
                    and bool(audio_is_system[item_idx])
                )

                if is_sys:
                    tokens = stream0_ids
                else:
                    tokens = [int(cfg.codec_ssl_start_end_token_id)] + stream0_ids

                def _is_embed(
                    _tokenizer: object,
                    _full: object,
                ) -> torch.Tensor:
                    mask = torch.ones(len(tokens), dtype=torch.bool)
                    if not is_sys and len(tokens) > 0:
                        mask[0] = False
                    return mask

                return PromptUpdateDetails(full=tokens, is_embed=_is_embed)

            return [
                PromptReplacement(
                    modality="audio",
                    target=[placeholder_id],
                    replacement=get_replacement_dialogue,
                ),
            ]

        # Single-turn path (existing)
        # Use token ID list (not string) as target so _apply_token_matches
        # can find the placeholder [34] in the prompt directly.
        placeholder_id_st = int(cfg.codec_ssl_start_end_token_id)  # 34
        mode_obj = hf_processor_mm_kwargs.get("mode")
        task_obj = hf_processor_mm_kwargs.get("task")
        task_token_id = self.resolve_task_token_id(
            cfg,
            has_audio_input=True,
            mode=mode_obj if isinstance(mode_obj, str) else None,
            task=task_obj if isinstance(task_obj, (str, int)) else None,
        )
        replacement_boundary_id = cfg.codec_ssl_start_end_token_id
        tts_task_ids = {
            int(cfg.codec_ssl_tts_task_token_id),
            int(getattr(cfg, "codec_ssl_plain_tts_task_token_id", 82)),
        }
        if int(task_token_id) in tts_task_ids:
            replacement_boundary_id = int(
                getattr(cfg, "spk_start_end_token_id", 37)
            )

        # inter_segment_pad appended to each audio chunk
        inter_pad = int(cfg.nq) - 1  # 8

        is_tts = int(task_token_id) in tts_task_ids
        target_modality_id = int(cfg.codec_ssl_start_end_token_id)  # 34

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

            if is_tts:
                # TTS: [spk(37)] + [spk_audio×500 + pad×8] + [34]
                # The stream0_ids already has 500+8=508 items (speaker +
                # inter-segment pad).  Append [34] as target modality
                # marker so the model generates audio after it.
                tokens = (
                    [replacement_boundary_id]
                    + stream0_ids
                    + [target_modality_id]
                )

                def _is_embed(
                    _tokenizer: object,
                    _full: object,
                ) -> torch.Tensor:
                    mask = torch.ones(len(tokens), dtype=torch.bool)
                    mask[0] = False   # [37] spk boundary: text embed
                    mask[-1] = False  # [34] target marker: text embed
                    return mask
            else:
                # ASR / other: [34] + [audio + codec_ssl_end + pad×8]
                tokens = [replacement_boundary_id] + stream0_ids

                def _is_embed(
                    _tokenizer: object,
                    _full: object,
                ) -> torch.Tensor:
                    mask = torch.ones(len(tokens), dtype=torch.bool)
                    mask[0] = False  # boundary token not embedded
                    return mask

            return PromptUpdateDetails(full=tokens, is_embed=_is_embed)

        return [
            PromptReplacement(
                modality="audio",
                target=[placeholder_id_st],
                replacement=get_replacement_audio,
            ),
        ]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
@MULTIMODAL_REGISTRY.register_processor(
    OpusLMMultiModalProcessor,
    info=OpusLMProcessingInfo,
    dummy_inputs=OpusLMDummyInputsBuilder,
)
class OpusLMForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """OpusLM: OLMo-2-7B based multimodal speech-language model.

    Supports text input and text/audio output. Audio output uses
    9 streams (1 SSL + 8 DAC) with delay interleaving. From vLLM's
    perspective, the model generates 1 token per step (stream 0 / SSL);
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
        config: OpusLMConfig = vllm_config.model_config.hf_config
        self.config = config

        # Build a fake Olmo2Config so we can reuse Olmo2Model
        from transformers import Olmo2Config as HFOlmo2Config
        from vllm.model_executor.models.olmo2 import Olmo2Model
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        olmo2_hf_config = HFOlmo2Config(
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

        # Temporarily patch vllm_config to use the Olmo2Config
        olmo2_vllm_config = vllm_config.with_hf_config(
            olmo2_hf_config, architectures=["Olmo2ForCausalLM"]
        )

        # --- Language model (OLMo-2-7B backbone) ---
        with self._mark_language_model(vllm_config):
            self.model = Olmo2Model(
                vllm_config=olmo2_vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        # --- LM head (vocab projection) ---
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        # --- Per-stream head bias embeddings [12, 4096] ---
        # Stream 0 = index 0, streams 1-8 = indices 1-8
        self.head_emb = nn.Embedding(12, config.hidden_size)

        # --- Logits processor ---
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # --- Pipeline parallelism ---
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # --- Internal decode state ---
        # Streams 1-8 token buffer from previous compute_logits call,
        # keyed by request ID. Each entry: Tensor[8] of codec tokens.
        self._stream_buffer_dict: dict[str, torch.Tensor] = {}

        # --- Per-request state ---
        self._per_req_config: dict[str, dict] = {}
        # Stream 1-8 codec token history per request [list of Tensor[8]]
        self._stream18_history: dict[str, list[torch.Tensor]] = {}
        # Stream 0 SSL token history per request [list of int]
        self._stream0_history: dict[str, list[int]] = {}
        # Decoded audio base64 per request
        self._decoded_audio: dict[str, str] = {}
        # Current batch request IDs (set by model runner before forward)
        self._current_batch_req_ids: list[str] = []

        # --- DAC decoder (lazy-loaded on first audio decode) ---
        self._dac_model = None
        self._dac_sample_rate: int = config.dac_sample_rate
        self._dac_hf_model_tag: str = config.dac_hf_model_tag

        # --- GPU audio encoder for dialogue mode (lazy-loaded) ---
        self._audio_input_processor_model: (
            _OpusLMGPUAudioInputProcessor | None
        ) = None

        # --- Dialogue mode flag (default False, overridden per-request) ---
        self._is_dialogue = False

        # --- Precompute token masks ---
        self._build_masks(config)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------
    def _build_masks(self, config: OpusLMConfig):
        """Precompute token validity masks for each stream."""
        V = config.vocab_size

        # Stream 0 (SSL): allow [ssl_token_start, ssl_token_end) + eos (5)
        # ESPnet mask: stream 0 can ONLY output SSL tokens and EOS.
        # Token 34 (codec_ssl_start_end) must NOT be allowed here —
        # if the model samples it during audio phase, it creates a
        # mismatch between stream0_history (N_ssl) and stream18_history,
        # corrupting the de-interleave alignment.
        # EOS is further gated by per-request `audio_minlen` in compute_logits.
        ssl_mask = torch.ones(V, dtype=torch.bool)
        ssl_mask[config.ssl_token_start:config.ssl_token_end] = False
        ssl_mask[config.eos_token_id] = False
        self.register_buffer("audio_mask_s0", ssl_mask)

        # Text mask (stream 0 during text phase):
        # allow [text_token_start, text_token_end) + eos + text_bpe boundary tokens
        text_mask = torch.ones(V, dtype=torch.bool)
        text_mask[config.text_token_start:config.text_token_end] = False
        text_mask[config.eos_token_id] = False
        text_mask[config.text_bpe_start_end_token_id] = False
        self.register_buffer("text_mask_s0", text_mask)

        # Pre-audio mask: forces codec_ssl_start_end (34) output
        # Used when transitioning from text to audio phase.
        pre_audio_mask = torch.ones(V, dtype=torch.bool)
        pre_audio_mask[config.codec_ssl_start_end_token_id] = False
        self.register_buffer("pre_audio_mask", pre_audio_mask)

        # Streams 1-8 (DAC codec): allow [codec_token_start + k*1024,
        #                                  codec_token_start + (k+1)*1024)
        #                               + pad (0)
        audio_masks = [self.audio_mask_s0]  # index 0 = stream 0
        for k in range(config.num_codec_streams):
            mask = torch.ones(V, dtype=torch.bool)
            start = config.codec_token_start + k * config.codec_per_stream_size
            end = start + config.codec_per_stream_size
            mask[start:end] = False
            mask[0] = False  # allow pad (0) for streams 1-8
            self.register_buffer(f"audio_mask_s{k + 1}", mask)
            audio_masks.append(mask)
        self._audio_masks = audio_masks

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

    # ------------------------------------------------------------------
    # Audio encoder (lazy-loaded, used by embed_multimodal for dialogue)
    # ------------------------------------------------------------------
    def _get_audio_encoder(self) -> _OpusLMGPUAudioInputProcessor:
        """Lazy-load the SSL+DAC audio encoder on model device."""
        if self._audio_input_processor_model is None:
            device = next(self.model.parameters()).device
            self._audio_input_processor_model = (
                _OpusLMGPUAudioInputProcessor(
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

        Each audio item produces T_total embeddings where:
          - System audio: T_total = speaker_prompt_length + inter_pad
          - User audio:   T_total = ceil(N_samples/320) + inter_pad
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
        sample_lens = (
            audio_sample_lengths.tolist()
            if audio_sample_lengths.dim() > 0
            else [int(audio_sample_lengths)]
        )
        for i, n_samples in enumerate(sample_lens):
            n_samples = int(n_samples)
            audio_np = raw_audio[offset:offset + n_samples].cpu().numpy()
            offset += n_samples

            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)

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
                and (bool(audio_is_system[i])
                     if hasattr(audio_is_system, '__getitem__')
                     else bool(audio_is_system))
            )

            if is_sys:
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

            T_total = T_audio + inter_pad
            pad_s0 = np.zeros(inter_pad, dtype=np.int64)
            pad_s18 = np.zeros((inter_pad, 8), dtype=np.int64)
            stream0 = np.concatenate([stream0, pad_s0])
            streams18 = np.concatenate([streams18, pad_s18])

            if embed_lengths is not None:
                if (hasattr(embed_lengths, '__getitem__')
                        and (not isinstance(embed_lengths, torch.Tensor)
                             or embed_lengths.dim() > 0)):
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

            s0 = torch.from_numpy(stream0).long().to(device)
            s18 = torch.from_numpy(streams18).long().to(device)

            s0 = s0.clamp(0, V - 1)
            s18 = s18.clamp(0, V - 1)

            delayed_s18 = torch.zeros_like(s18)
            for k in range(s18.shape[1]):
                delay = k + 1
                if delay < T_total:
                    delayed_s18[delay:, k] = s18[:T_total - delay, k]

            try:
                s0_embed = embed_fn(s0)
                torch.cuda.synchronize()
                s18_embed = embed_fn(delayed_s18)
                torch.cuda.synchronize()
            except Exception as e:
                logger.error(
                    "embed_fn failed for item %d: s0 range=[%d,%d] "
                    "s18 range=[%d,%d] V=%d: %s",
                    i, int(s0.min()), int(s0.max()),
                    int(delayed_s18.min()), int(delayed_s18.max()),
                    V, e,
                )
                raise
            combined = s0_embed + s18_embed.sum(dim=1)
            mm_embeddings.append(combined)

        return tuple(mm_embeddings)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        # Path 1: raw audio (GPU-side encoding for dialogue mode)
        if kwargs.get("input_audio_features") is not None:
            return self._encode_and_embed_audio(**kwargs)

        # Path 2: pre-computed token IDs (existing single-turn path)
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

            s0 = stream0_ids[offset:offset + length]            # [L]
            s18 = streams18[offset:offset + length]             # [L, 8]
            offset += length

            # Apply delay interleaving: stream k is delayed by k+1 steps
            delayed_s18 = torch.zeros_like(s18)
            for k in range(s18.shape[1]):
                delay = k + 1
                if delay < length:
                    delayed_s18[delay:, k] = s18[:length - delay, k]

            s0_embed = embed_fn(s0)                             # [L, H]
            s18_embed = embed_fn(delayed_s18)                   # [L, 8, H]
            # Do NOT mask out embed(0) for delayed/pad positions.
            # ESPnet sums all 9 streams including embed(0) for pad.
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
        """Embed input tokens with multi-stream support for audio decode.

        During text prefill: standard token embedding.
        During audio decode: embed stream 0 SSL token + buffer DAC streams 1-8,
        sum across streams.
        """
        # Standard token embedding.
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        # Overwrite placeholder-token embeddings with multimodal embeddings
        # during prefill of audio inputs.
        mm_positions = None
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            if is_multimodal is None:
                raise ValueError(
                    "`embed_input_ids` requires `is_multimodal` when "
                    "multimodal embeddings are provided."
                )
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            mm_positions = is_multimodal

        # Add buffered stream1-8 embeddings for autoregressive audio decode.
        stream_positions = None
        if self._stream_buffer_dict:
            stream_positions = self._get_stream_embed_positions(input_ids)
            if stream_positions.any():
                inputs_embeds = self._apply_stream_embeddings(
                    input_ids, inputs_embeds, stream_positions
                )

        # ESPnet sums all 9 streams: text positions get 8*embed(0) pad bias
        # from streams 1-8 being pad(0). Multimodal and audio-decode positions
        # already have the correct stream embeddings. Add bias to the rest.
        handled = torch.zeros_like(input_ids, dtype=torch.bool)
        if mm_positions is not None:
            handled |= mm_positions
        if stream_positions is not None:
            handled |= stream_positions
        needs_bias = ~handled
        if needs_bias.any():
            nq_minus_1 = int(getattr(self.config, "nq", 9)) - 1  # 8
            pad_bias = self.model.embed_tokens(
                torch.zeros(1, dtype=torch.long, device=input_ids.device)
            )  # [1, H]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[needs_bias] += nq_minus_1 * pad_bias

        return inputs_embeds

    def _get_stream_embed_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return positions that should include buffered streams 1-8 embeddings.

        For OpusLM delay decoding, stream embeddings are needed for active
        audio decode requests regardless of whether stream-0 token is SSL/EOS/PAD.
        """
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

        # Fallback for unexpected contexts.
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
        """Add streams 1-8 (DAC) embeddings for audio-mode SSL tokens.

        For audio-mode positions, the embedding becomes:
            embed(ssl_token) + sum(embed(dac_stream_k_token) for k in 1..8)
        where DAC stream tokens come from the per-request buffer.
        """
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

        stream_tokens = torch.stack(buf_rows, dim=0)  # [N_valid, 8]
        stream_embeds = embed_fn(stream_tokens)       # [N_valid, 8, H]

        # Zero out pad positions (token == 0)
        pad_mask = (stream_tokens == 0).unsqueeze(-1)
        stream_embeds = stream_embeds.masked_fill(pad_mask, 0.0)

        stream_sum = stream_embeds.sum(dim=1)  # [N_valid, H]
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
          "text"       → text mask; EOS or text_bpe boundary allowed
          "pre_audio"  → force codec_ssl_start_end (34) output
          "audio"      → SSL mask for stream 0; also sample DAC streams 1-8
          "audio_flush"→ delay flush (force pad=0 on stream0)
          "audio_stop" → force EOS (5) to stop the request
        """
        cfg = self.config

        # 1. Apply stream-0 head_emb bias (index 0) to get stream-0 logits
        h0 = hidden_states + self.head_emb.weight[0].unsqueeze(0)
        stream0_logits = self.logits_processor(self.lm_head, h0)
        if stream0_logits is None:
            return None

        # 2. Determine per-position phase from _per_req_config
        batch_rids = self._current_batch_req_ids
        N = stream0_logits.shape[0]

        phases = []  # one per logit row
        for i in range(N):
            if i < len(batch_rids):
                rc = self._per_req_config.get(batch_rids[i], {})
                phases.append(rc.get("phase", "text"))
            else:
                phases.append("text")

        # 3. Collect position indices per phase
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

        # 4. Apply phase masks
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
            # Match ARDelay allow_eos=step >= minlen behavior for stream 0.
            for pos in audio_positions:
                req_cfg = self._get_req_config(pos)
                audio_step = int(req_cfg.get("audio_step", 0))
                audio_minlen = int(
                    req_cfg.get("audio_minlen", getattr(cfg, "audio_minlen", 50))
                )
                if audio_step < max(audio_minlen, 0):
                    stream0_logits[pos, cfg.eos_token_id] = float("-inf")

            # ESPnet applies top_k to ALL 9 streams including stream 0
            # inside logits_to_tokens(). In vLLM, streams 1-8 get top_k in
            # _sample_and_buffer_streams, but stream 0 goes to the standard
            # vLLM sampler which uses request-level top_k (often unset/0).
            # Apply audio_topk filtering to stream 0 here so the sampler
            # only sees the top-k candidates.  Temperature is NOT applied
            # here — it comes from the request-level SamplingParams.
            for pos in audio_positions:
                req_cfg = self._get_req_config(pos)
                top_k = int(
                    req_cfg.get("audio_topk", cfg.audio_topk)
                )
                if top_k > 0 and top_k < cfg.vocab_size:
                    row = stream0_logits[pos]
                    topk_vals, topk_idx = torch.topk(row, top_k)
                    stream0_logits[pos] = torch.full_like(
                        row, float("-inf")
                    )
                    stream0_logits[pos].scatter_(0, topk_idx, topk_vals)

        if pre_audio_positions:
            # Force codec_ssl_start_end (34) output
            idx = torch.tensor(pre_audio_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = float("-inf")
            stream0_logits[idx, cfg.codec_ssl_start_end_token_id] = 0.0

        if audio_flush_positions:
            # Force pad (0) output during delay flush.
            idx = torch.tensor(audio_flush_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = float("-inf")
            stream0_logits[idx, 0] = 0.0  # pad token

        if audio_stop_positions:
            # Force EOS output to stop the request
            idx = torch.tensor(audio_stop_positions, device=dev, dtype=torch.long)
            stream0_logits[idx] = float("-inf")
            stream0_logits[idx, cfg.eos_token_id] = 0.0

        # 5. For audio-phase and flush-phase requests:
        # sample DAC streams 1-8 and update per-request buffer.
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

        audio_hidden = hidden_states[audio_idx]  # [N_audio, H]
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

        for s in range(1, cfg.nq):  # streams 1-8
            # Apply per-stream head_emb bias
            head_bias = self.head_emb.weight[s].unsqueeze(0)
            h_s = audio_hidden + head_bias  # [N_audio, H]

            s_logits = self.logits_processor(self.lm_head, h_s)
            if s_logits is None:
                continue

            # Apply stream-specific mask
            s_logits = s_logits.masked_fill(
                self._audio_masks[s].unsqueeze(0), float("-inf")
            )

            sampled = torch.zeros(num_audio, dtype=torch.long, device=device)
            for row in range(num_audio):
                # Delay warmup behavior (ARDelay):
                # step=0 allows only stream0; stream s starts after step >= s.
                if (
                    row_phase[row] == "audio"
                    and row_audio_step[row] < s
                ):
                    sampled[row] = 0
                    continue
                # Flush behavior matching ARDelay semantics:
                # after finish, progressively force stream_n to pad when
                # finish_step > n (stream0 handled by logits mask above).
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

        # Store buffer and history per-request
        for j, pos in enumerate(sampled_positions):
            if pos < len(batch_rids):
                req_id = batch_rids[pos]
                buf_vec = new_buffer[j]
                self._stream_buffer_dict[req_id] = buf_vec.clone()
                self._stream18_history.setdefault(req_id, []).append(
                    buf_vec.clone()
                )

    def _get_audio_sampling_groups(
        self,
        positions: list[int],
        device: torch.device,
    ) -> list[tuple[float, int, torch.Tensor]]:
        """Group request rows by (audio_temperature, audio_topk)."""
        cfg = self.config
        default_temperature = float(cfg.audio_temperature)
        default_top_k = int(cfg.audio_topk)

        grouped_rows: dict[tuple[float, int], list[int]] = {}
        for row_idx, position_idx in enumerate(positions):
            rc = self._get_req_config(position_idx)
            temperature = float(rc.get("audio_temperature", default_temperature))
            top_k = int(rc.get("audio_topk", default_top_k))
            if top_k <= 0:
                top_k = cfg.vocab_size
            top_k = max(1, min(top_k, cfg.vocab_size))
            grouped_rows.setdefault((temperature, top_k), []).append(row_idx)

        return [
            (
                temperature,
                top_k,
                torch.tensor(rows, device=device, dtype=torch.long),
            )
            for (temperature, top_k), rows in grouped_rows.items()
        ]

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
        """Convert global DAC token IDs to per-stream codebook indices [0, 1023].

        Args:
            dac_tokens: [B, T, 8] global DAC token IDs

        Returns:
            [B, T, 8] codebook indices in [0, 1023]
        """
        cfg = self.config
        result = dac_tokens.clone()
        for k in range(cfg.num_codec_streams):
            offset = cfg.codec_token_start + k * cfg.codec_per_stream_size
            result[..., k] = (result[..., k] - offset).clamp(0, 1023)
        return result

    @torch.inference_mode()
    def _dac_decode(self, codebook_indices: torch.Tensor) -> "Any":
        """Decode DAC codebook indices to audio waveform.

        Args:
            codebook_indices: [B, T, 8] indices in [0, 1023]

        Returns:
            numpy array of audio samples
        """
        import numpy as np

        dac = self._get_dac_model()
        # DAC expects [n_q, B, T] or similar — use ESPnet's interface
        # codes shape: [B, T, 8] → [8, B, T] → [8, 1, T] for batch 1
        codes = codebook_indices.permute(2, 0, 1)  # [8, B, T]
        # ESPnet AudioCoding decode
        audio = dac.decode(codes)
        # audio shape: [B, 1, T_wav]
        return audio.squeeze().cpu().numpy()

    def encode_audio_to_base64_wav(
        self, req_id: str, stream0_tokens: list[int]
    ) -> str | None:
        """Decode stream 0 SSL tokens + stream 1-8 DAC history to base64 WAV.

        Args:
            req_id: Request ID (used to retrieve stream 1-8 history)
            stream0_tokens: List of stream 0 SSL token IDs (global IDs,
                values in [ssl_token_start, ssl_token_end))

        Returns:
            Base64-encoded WAV string, or None on failure
        """
        import base64
        import io
        import wave

        import numpy as np

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
                wf.setsampwidth(2)  # 16-bit
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

        ESPnet flow (ar_delay.py):
          1. Generate all steps including EOS step + nq-1 flush steps
          2. inverse_delay_interleave on FULL sequence [total_steps, 9]
          3. Truncate to [:finish_idx - 1] (exclude EOS frame)

        vLLM:
          - stream0_ssl_tokens: only SSL tokens (steps before EOS), length=E
          - stream18_history: entries for ALL audio+flush steps, length=E+1+8=E+9
            (step E = EOS step, steps E+1..E+8 = flush)

        To match ESPnet, we must de-interleave the FULL sequence including
        EOS and flush steps, then truncate to exclude the EOS frame.

        Args:
            req_id: Request ID to retrieve stream 1-8 history
            stream0_ssl_tokens: SSL token IDs (global, in [ssl_token_start, ssl_token_end))
        """
        import numpy as np

        history = self._stream18_history.pop(req_id, [])
        cfg = self.config

        N_ssl = len(stream0_ssl_tokens)
        H = len(history)
        if N_ssl == 0:
            return np.zeros(0, dtype=np.float32), self._dac_sample_rate

        device = next(self.model.parameters()).device

        # The full generated sequence has H steps (stream18 records every
        # audio + flush step).  Stream 0 has N_ssl real SSL tokens (steps
        # 0..E-1), then EOS at step E, then pad(0) for flush steps E+1..E+8.
        # So the full stream0 column should be:
        #   [ssl_0, ssl_1, ..., ssl_{E-1}, eos, 0, 0, ..., 0]  length = H
        T_full = max(H, N_ssl)  # use the longer of the two

        # Build full stream 0 column: SSL tokens + EOS + pad(0)
        s0_full = torch.zeros(T_full, dtype=torch.long, device=device)
        if N_ssl > 0:
            s0_full[:N_ssl] = torch.tensor(
                stream0_ssl_tokens, dtype=torch.long, device=device
            )
        # Position N_ssl = EOS step (ESPnet puts eos in gen_token_seq)
        if N_ssl < T_full:
            s0_full[N_ssl] = cfg.eos_token_id
        # Remaining positions (flush steps) stay 0 (pad)

        # Build full stream 1-8 column
        if H == 0:
            s18_full = torch.zeros(T_full, 8, dtype=torch.long, device=device)
        else:
            s18_stack = torch.stack(history, dim=0)  # [H, 8]
            if H >= T_full:
                s18_full = s18_stack[:T_full]
            else:
                pad = torch.zeros(
                    T_full - H, 8, dtype=torch.long, device=device
                )
                s18_full = torch.cat([s18_stack, pad], dim=0)

        # Full matrix: [T_full, 9]
        full_matrix = torch.cat(
            [s0_full.unsqueeze(1), s18_full], dim=1
        ).unsqueeze(0)  # [1, T_full, 9]

        # De-interleave on FULL sequence: [1, T_full - 8, 9]
        aligned = self._delay_deinterleave(full_matrix)

        if aligned.shape[1] == 0:
            return np.zeros(0, dtype=np.float32), self._dac_sample_rate

        # Truncate to exclude the EOS frame (ESPnet: [:finish_idx - 1]).
        # finish_idx in ESPnet = step after EOS was output as prev_tok.
        # In our case, EOS is at step N_ssl in gen sequence.  After
        # de-interleave, the EOS frame in stream 0 is at position N_ssl.
        # We want [:N_ssl] to exclude EOS (same as ESPnet's [:finish_idx-1]
        # where finish_idx = N_ssl + 1).
        aligned = aligned[:, :N_ssl, :]

        if aligned.shape[1] == 0:
            return np.zeros(0, dtype=np.float32), self._dac_sample_rate

        # Extract DAC codec tokens (cols 1-8): [1, T_aligned, 8]
        dac_tokens = aligned[:, :, 1:]

        # Convert global IDs to codebook indices [0, 1023]
        dac_cb = self._global_to_dac_codebook(dac_tokens)

        # Decode to audio
        audio = self._dac_decode(dac_cb)
        return audio, self._dac_sample_rate

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights from safetensors.

        Expected weight names (after opuslm_convert.py remapping):
          model.embed_tokens.weight    [113870, 4096]
          model.norm.weight            [4096]
          model.layers.{i}.*           (OLMo-2 layer weights)
          lm_head.weight               [113870, 4096]
          head_emb.weight              [12, 4096]
        """
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
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
                # OLMo-2 backbone weights
                if is_pp_missing_parameter(name, self.model):
                    continue
                # Try stacked params first
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

            # else: skip (e.g. criterion.lm_head.weight already filtered)

        return loaded_params

    # ------------------------------------------------------------------
    # Multi-model key mapping (for pipeline parallelism)
    # ------------------------------------------------------------------
    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model",
        )
