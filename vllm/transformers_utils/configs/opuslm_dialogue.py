# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for OpusLM Dialogue (SmolLM2-1.7B / Llama based
multimodal speech-language dialogue model with 9-stream delay-interleaved
discrete codec output).

This model supports multi-turn dialogue with mixed text/audio turns,
using dialogue-specific role tokens (system, user, assistant) and
end-of-utterance markers.

Vocab layout (same as base OpusLM):
    [0,   256)     - Special tokens (pad=0, bos=1, eos=5, ...)
    [256, 5256)    - SSL tokens (XEUS + K-means, 5000 clusters)
    [5256, 13448)  - DAC codec tokens (8 streams x 1024 tokens each)
    [13448, 62600) - Text BPE tokens (SmolLM-1.7B tokenizer)
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OpusLMDialogueConfig(PretrainedConfig):
    """Configuration for OpusLMDialogueForConditionalGeneration.

    OpusLM Dialogue is a SmolLM2-1.7B (Llama) based multimodal speech-language
    dialogue model that uses:
    - 1 SSL stream (XEUS + K-means, 5000 clusters) for stream 0
    - 8 DAC codec streams (1024 codes each) for streams 1-8
    - 9-stream delay interleaving during generation
    - Multi-turn dialogue with role tokens and end-of-utterance markers
    """

    model_type = "opuslm_dialogue"

    def __init__(
        self,
        # SmolLM2-1.7B / Llama backbone arch
        vocab_size: int = 62670,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 130000.0,
        tie_word_embeddings: bool = False,
        # Special token IDs
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 5,
        sos_eos_token_id: int = 5,
        # Token range boundaries
        ssl_token_start: int = 256,
        ssl_token_end: int = 5256,
        codec_token_start: int = 5256,
        codec_token_end: int = 13448,
        text_token_start: int = 13448,
        text_token_end: int = 62600,
        # Modality boundary tokens
        codec_ssl_start_end_token_id: int = 34,   # <codec_ssl_start/end>
        text_bpe_start_end_token_id: int = 35,    # <text_bpe_start/end>
        spk_start_end_token_id: int = 37,         # <spk_start/end>
        # Dialogue-specific tokens
        system_prompt_token_id: int = 8,           # <system_prompt>
        user_input_token_id: int = 9,              # <user_input>
        assistant_output_token_id: int = 10,       # <assistant_output>
        eou_token_id: int = 11,                    # <eou>
        # Task identifiers
        audio_dialogue_task_token_id: int = 89,    # <audio_dialogue_task>
        text_dialogue_task_token_id: int = 88,     # <text_dialogue_task>
        textlm_task_token_id: int = 64,            # <textlm_task>
        codec_ssl_asr_task_token_id: int = 80,     # <codec_ssl_asr_task>
        codec_ssl_tts_task_token_id: int = 81,     # <codec_ssl_tts_task>
        codec_ssl_plain_tts_task_token_id: int = 82,
        codec_ssl_audiolm_task_token_id: int = 83,
        # Multi-stream settings
        nq: int = 9,              # total streams (1 SSL + 8 DAC)
        num_codec_streams: int = 8,
        codec_per_stream_size: int = 1024,
        # Speaker prompt
        speaker_prompt_length: int = 500,
        # Audio generation parameters
        audio_temperature: float = 0.8,
        audio_topk: int = 30,
        audio_minlen: int = 3,
        text_minlen: int = 1,
        # Codec/SSL model tags
        dac_hf_model_tag: str = "ftshijt/espnet_codec_dac_large_v1.4_360epoch",
        xeus_hf_model_tag: str = "espnet/xeus",
        xeus_checkpoint_filename: str = "model/xeus_checkpoint_new.pth",
        km_model_filename: str = "model/km_opus_lm.mdl",
        xeus_layer: int = 18,
        dac_sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # Llama backbone
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        # Token ranges
        self.ssl_token_start = ssl_token_start
        self.ssl_token_end = ssl_token_end
        self.codec_token_start = codec_token_start
        self.codec_token_end = codec_token_end
        self.text_token_start = text_token_start
        self.text_token_end = text_token_end

        # Modality markers
        self.codec_ssl_start_end_token_id = codec_ssl_start_end_token_id
        self.text_bpe_start_end_token_id = text_bpe_start_end_token_id
        self.spk_start_end_token_id = spk_start_end_token_id
        self.sos_eos_token_id = sos_eos_token_id

        # Dialogue tokens
        self.system_prompt_token_id = system_prompt_token_id
        self.user_input_token_id = user_input_token_id
        self.assistant_output_token_id = assistant_output_token_id
        self.eou_token_id = eou_token_id

        # Task identifiers
        self.audio_dialogue_task_token_id = audio_dialogue_task_token_id
        self.text_dialogue_task_token_id = text_dialogue_task_token_id
        self.textlm_task_token_id = textlm_task_token_id
        self.codec_ssl_asr_task_token_id = codec_ssl_asr_task_token_id
        self.codec_ssl_tts_task_token_id = codec_ssl_tts_task_token_id
        self.codec_ssl_plain_tts_task_token_id = codec_ssl_plain_tts_task_token_id
        self.codec_ssl_audiolm_task_token_id = codec_ssl_audiolm_task_token_id

        # Multi-stream
        self.nq = nq
        self.num_codec_streams = num_codec_streams
        self.codec_per_stream_size = codec_per_stream_size

        # Speaker prompt
        self.speaker_prompt_length = speaker_prompt_length

        # Audio generation
        self.audio_temperature = audio_temperature
        self.audio_topk = audio_topk
        self.audio_minlen = audio_minlen
        self.text_minlen = text_minlen

        # External model tags
        self.dac_hf_model_tag = dac_hf_model_tag
        self.xeus_hf_model_tag = xeus_hf_model_tag
        self.xeus_checkpoint_filename = xeus_checkpoint_filename
        self.km_model_filename = km_model_filename
        self.xeus_layer = xeus_layer
        self.dac_sample_rate = dac_sample_rate


__all__ = ["OpusLMDialogueConfig"]
