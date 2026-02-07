# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration classes for SpeechLM (Qwen3-8B based multimodal
speech-language model with 8-stream parallel codec output)."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SpeechLMAudioEncoderConfig(PretrainedConfig):
    """Configuration for the SpeechLM audio encoder (Qwen3-Omni Audio Tower).

    This encoder converts mel-spectrogram features into continuous audio
    representations. Architecture is identical to Qwen3OmniMoeAudioEncoder
    but with output_dim=2048 (instead of 3584).
    """

    model_type = "speechlm_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=2048,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class SpeechLMTextConfig(PretrainedConfig):
    """Configuration for the SpeechLM text backbone (Qwen3-8B-Base).

    Extended vocabulary: 160392 tokens (256 special + 151936 text + 8*1025
    audio codec tokens).
    """

    model_type = "speechlm_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=160392,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class SpeechLMConfig(PretrainedConfig):
    """Configuration for SpeechLMForConditionalGeneration.

    This is the top-level config that wraps audio_config and text_config.
    It also stores SpeechLM-specific parameters like multi-stream settings,
    special token IDs, and vocab layout information.

    Vocab layout:
        [0, 256)          - Special tokens (pad=0, bos=1, eos=2, eot=3, ...)
        [256, 152192)     - Text tokens (Qwen3 tokenizer IDs + 256 offset)
        [152192, 160392)  - Audio codec tokens (8 streams x 1025 tokens each)
    """

    model_type = "speechlm"
    sub_configs = {
        "audio_config": SpeechLMAudioEncoderConfig,
        "text_config": SpeechLMTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        # Multi-stream settings
        num_stream=8,
        adaptor_input_dim=2048,
        # Vocab layout
        vocab_size=160392,
        hidden_size=4096,
        codec_base_offset=152192,
        codec_layer_size=1025,
        text_token_offset=256,
        text_token_end=152192,
        # Special token IDs
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eot_token_id=3,
        system_token_id=4,
        user_token_id=5,
        assistant_token_id=6,
        text_token_id=7,
        audio_token_id=8,
        # Audio placeholder (same as audio_token_id, used for input audio)
        audio_placeholder_token_id=8,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(audio_config, dict):
            audio_config = SpeechLMAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = SpeechLMAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = SpeechLMTextConfig(**text_config)
        elif text_config is None:
            text_config = SpeechLMTextConfig()
        self.text_config = text_config

        self.num_stream = num_stream
        self.adaptor_input_dim = adaptor_input_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.codec_base_offset = codec_base_offset
        self.codec_layer_size = codec_layer_size
        self.text_token_offset = text_token_offset
        self.text_token_end = text_token_end
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.eot_token_id = eot_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id
        self.text_token_id = text_token_id
        self.audio_token_id = audio_token_id
        self.audio_placeholder_token_id = audio_placeholder_token_id
        self.initializer_range = initializer_range

    def get_text_config(self, decoder=False):
        return self.text_config


__all__ = [
    "SpeechLMConfig",
    "SpeechLMTextConfig",
    "SpeechLMAudioEncoderConfig",
]
