# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for OpusLM (OLMo-2-7B based multimodal
speech-language model with 9-stream delay-interleaved discrete codec output).

Vocab layout:
    [0,   256)    - Special tokens (pad=0, bos=1, eos=5, ...)
    [256, 5256)   - SSL tokens (XEUS + K-means, 5000 clusters)
    [5256, 13448) - DAC codec tokens (8 streams × 1024 tokens each)
    [13448, ~)    - Text BPE tokens
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OpusLMConfig(PretrainedConfig):
    """Configuration for OpusLMForConditionalGeneration.

    OpusLM is an OLMo-2-7B based multimodal speech-language model that uses:
    - 1 SSL stream (XEUS + K-means, 5000 clusters) for stream 0
    - 8 DAC codec streams (1024 codes each) for streams 1-8
    - 9-stream delay interleaving during generation
    - No continuous audio encoder (audio input is pre-tokenized)
    """

    model_type = "opuslm"

    def __init__(
        self,
        # OLMo-2 backbone arch
        vocab_size: int = 113870,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
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
        text_token_end: int = 113800,
        # Modality boundary tokens
        codec_ssl_start_end_token_id: int = 34,   # <codec_ssl_start/end>
        text_bpe_start_end_token_id: int = 35,    # <text_bpe_start/end>
        spk_start_end_token_id: int = 37,         # <spk_start/end>
        # Task identifiers
        textlm_task_token_id: int = 64,           # <textlm_task>
        codec_ssl_asr_task_token_id: int = 80,    # <codec_ssl_asr_task>
        codec_ssl_tts_task_token_id: int = 81,    # <codec_ssl_tts_task>
        codec_ssl_plain_tts_task_token_id: int = 82,   # <codec_ssl_plain_tts_task>
        codec_ssl_audiolm_task_token_id: int = 83,     # <codec_ssl_audiolm_task>
        # Dialogue tokens (same IDs as dialogue model)
        system_prompt_token_id: int = 8,
        user_input_token_id: int = 9,
        assistant_output_token_id: int = 10,
        eou_token_id: int = 11,
        audio_dialogue_task_token_id: int = 89,
        text_dialogue_task_token_id: int = 88,
        speaker_prompt_length: int = 500,
        # Multi-stream settings
        nq: int = 9,              # total streams (1 SSL + 8 DAC)
        num_codec_streams: int = 8,
        codec_per_stream_size: int = 1024,
        # Audio generation parameters
        audio_temperature: float = 0.7,
        audio_topk: int = 30,
        audio_minlen: int = 50,
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
        # OLMo-2 backbone
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
        self.textlm_task_token_id = textlm_task_token_id
        self.codec_ssl_asr_task_token_id = codec_ssl_asr_task_token_id
        self.codec_ssl_tts_task_token_id = codec_ssl_tts_task_token_id
        self.codec_ssl_plain_tts_task_token_id = codec_ssl_plain_tts_task_token_id
        self.codec_ssl_audiolm_task_token_id = codec_ssl_audiolm_task_token_id

        # Dialogue tokens
        self.system_prompt_token_id = system_prompt_token_id
        self.user_input_token_id = user_input_token_id
        self.assistant_output_token_id = assistant_output_token_id
        self.eou_token_id = eou_token_id
        self.audio_dialogue_task_token_id = audio_dialogue_task_token_id
        self.text_dialogue_task_token_id = text_dialogue_task_token_id
        self.speaker_prompt_length = speaker_prompt_length

        # Multi-stream
        self.nq = nq
        self.num_codec_streams = num_codec_streams
        self.codec_per_stream_size = codec_per_stream_size

        # Audio generation
        self.audio_temperature = audio_temperature
        self.audio_topk = audio_topk
        self.audio_minlen = audio_minlen

        # External model tags
        self.dac_hf_model_tag = dac_hf_model_tag
        self.xeus_hf_model_tag = xeus_hf_model_tag
        self.xeus_checkpoint_filename = xeus_checkpoint_filename
        self.km_model_filename = km_model_filename
        self.xeus_layer = xeus_layer
        self.dac_sample_rate = dac_sample_rate


__all__ = ["OpusLMConfig"]
