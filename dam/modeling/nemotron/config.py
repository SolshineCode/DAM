# Copyright 2024-2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Merged Nemotron-H model configuration for DAM."""

from transformers.configuration_utils import PretrainedConfig


class MergedNemotronHConfig(PretrainedConfig):
    r"""
    Configuration class for the DAM-merged Nemotron-H model.

    This extends the NemotronHConfig with DAM-specific parameters for
    differentiable adaptive merging. Nemotron-H uses a hybrid architecture
    with three block types: Mamba-2 SSM, standard attention, and MoE.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the Nemotron model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for attention layers.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for GQA.
        head_dim (`int`, *optional*, defaults to 128):
            Attention head dimension.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            Maximum sequence length.
        intermediate_size (`int`, *optional*, defaults to 21504):
            Dimension of the MLP representations.
        mlp_hidden_act (`str`, *optional*, defaults to `"relu2"`):
            Activation function for MLP layers (ReLU squared).
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether MLP layers use bias.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether attention projection layers use bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        sliding_window (`int`, *optional*):
            Sliding window attention size, if applicable.
        layers_block_type (`list[str]`, *optional*):
            Per-layer block type specification. Each entry is one of
            "mamba", "attention", or "moe".
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            Epsilon for layer normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return last key/values attentions.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the RoPE embeddings.
        ssm_state_size (`int`, *optional*, defaults to 128):
            SSM state dimension for Mamba layers.
        mamba_num_heads (`int`, *optional*, defaults to 128):
            Number of Mamba heads.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Mamba head dimension.
        mamba_hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function for Mamba layers.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups in Mamba-2.
        conv_kernel (`int`, *optional*, defaults to 4):
            Kernel size for conv1d in Mamba layers.
        expand (`int`, *optional*, defaults to 2):
            Expansion factor for Mamba layers.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether conv1d uses bias.
        chunk_size (`int`, *optional*, defaults to 128):
            Chunk size for Mamba-2 SSD computation.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether Mamba projection layers use bias.
        n_routed_experts (`int`, *optional*, defaults to 8):
            Number of routed experts in MoE layers.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts in MoE layers.
        moe_intermediate_size (`int`, *optional*, defaults to 7688):
            Intermediate size for MoE expert MLPs.
        moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 7688):
            Intermediate size for shared expert MLPs.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of experts activated per token.
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

    model_type = "mergednemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
        intermediate_size=21504,
        mlp_hidden_act="relu2",
        mlp_bias=False,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        layers_block_type=None,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        # Mamba-2 parameters
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_head_dim=64,
        mamba_hidden_act="silu",
        n_groups=8,
        conv_kernel=4,
        expand=2,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        use_conv_bias=True,
        chunk_size=128,
        mamba_proj_bias=False,
        use_mamba_kernels=True,
        residual_in_fp32=False,
        rescale_prenorm_residual=True,
        # MoE parameters
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=7688,
        moe_shared_expert_intermediate_size=7688,
        moe_latent_size=None,
        moe_shared_expert_overlap=True,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        use_bias=False,
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
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        # Mamba-2
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.mamba_hidden_act = mamba_hidden_act
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.mamba_proj_bias = mamba_proj_bias
        self.use_mamba_kernels = use_mamba_kernels
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual

        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.moe_shared_expert_overlap = moe_shared_expert_overlap
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.use_bias = use_bias

        # Layer types: default to alternating mamba/moe/attention/moe pattern
        if layers_block_type is None:
            default_pattern = ["mamba", "moe", "attention", "moe"]
            self.layers_block_type = [
                default_pattern[i % len(default_pattern)]
                for i in range(num_hidden_layers)
            ]
        else:
            self.layers_block_type = layers_block_type

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
