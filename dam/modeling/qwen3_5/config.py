# Copyright 2025 Qwen team, Alibaba Cloud and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merged Qwen 3.5 text model configuration for DAM."""

from transformers.configuration_utils import PretrainedConfig


class MergedQwen3_5Config(PretrainedConfig):
    r"""
    Configuration class for the DAM-merged Qwen 3.5 text model.

    This extends the Qwen3_5TextConfig with DAM-specific parameters for
    differentiable adaptive merging. Qwen 3.5 uses a hybrid architecture
    with both full (softmax) attention and linear (Gated Delta Network)
    attention layers.

    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Vocabulary size of the Qwen 3.5 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for full attention layers.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key_value heads for GQA in full attention layers.
        head_dim (`int`, *optional*, defaults to 256):
            Attention head dimension for full attention layers.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        layer_types (`list[str]`, *optional*):
            Per-layer type specification. Each entry is either "full_attention"
            or "linear_attention". If None, auto-generated from
            full_attention_interval.
        full_attention_interval (`int`, *optional*, defaults to 4):
            Every Nth layer uses full attention when layer_types is auto-generated.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Fraction of head_dim that gets rotary position embeddings.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size for causal conv1d in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Per-head key dimension in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Per-head value dimension in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads in linear attention.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads in linear attention.
        attn_output_gate (`bool`, *optional*, defaults to `True`):
            Whether full attention uses output gating.
        num_merged_models (`int`, *optional*, defaults to 3):
            The number of models being merged.
        init_merger_values (`list[float]`, *optional*, defaults to []):
            Initial values for the merger coefficients.
        use_tanh (`bool`, *optional*, defaults to False):
            Whether to apply tanh non-linearity to merger coefficients.
        dam_embedding_layer (`bool`, *optional*, defaults to True):
            Whether embedding layers use DAM merging.
        dam_layernorms (`bool`, *optional*, defaults to True):
            Whether layer normalization uses DAM merging.
        uses_base_model (`bool`, *optional*, defaults to True):
            Whether the base model is included in the merge.
        is_embedding_coef_trainable (`bool`, *optional*, defaults to False):
            Whether embedding merger coefficients are trainable.
        is_norm_coef_trainable (`bool`, *optional*, defaults to False):
            Whether norm merger coefficients are trainable.
    """

    model_type = "mergedqwen3_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=248044,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # Qwen 3.5 hybrid attention parameters
        layer_types=None,
        full_attention_interval=4,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        attn_output_gate=True,
        # DAM-specific parameters
        num_merged_models=3,
        init_merger_values=[],
        use_tanh=False,
        model_index=None,
        dam_embedding_layer=True,
        dam_layernorms=True,
        uses_base_model=True,
        is_embedding_coef_trainable=False,
        is_norm_coef_trainable=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Qwen 3.5 hybrid attention
        self.full_attention_interval = full_attention_interval
        self.partial_rotary_factor = partial_rotary_factor
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.attn_output_gate = attn_output_gate

        # Auto-generate layer_types if not provided
        if layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        # DAM-specific
        self.num_merged_models = num_merged_models
        self.init_merger_values = init_merger_values
        self.use_tanh = use_tanh
        self.model_index = model_index
        self.dam_embedding_layer = dam_embedding_layer
        self.dam_layernorms = dam_layernorms
        self.uses_base_model = uses_base_model
        self.is_embedding_coef_trainable = is_embedding_coef_trainable
        self.is_norm_coef_trainable = is_norm_coef_trainable

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
