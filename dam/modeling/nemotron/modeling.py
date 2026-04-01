"""PyTorch Merged Nemotron-H model for DAM (Differentiable Adaptive Merging)."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .config import MergedNemotronHConfig
from ..dam import DAMLinearLayer, DAMEmbeddingLayer, DAMRMSNorm

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MergedNemotronHConfig"


# ---------------------------------------------------------------------------
# Local RMSNorm (used when config.dam_layernorms is False)
# ---------------------------------------------------------------------------
class NemotronHRMSNorm(nn.Module):
    """NemotronHRMSNorm is equivalent to T5LayerNorm."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ---------------------------------------------------------------------------
# Gated RMSNorm (inside Mamba-2 blocks, NOT DAM-merged)
# ---------------------------------------------------------------------------
class NemotronHGatedRMSNorm(nn.Module):
    """Gated RMSNorm used inside Mamba-2 mixer blocks.

    Applies RMSNorm to hidden_states and then gates with silu(gate).
    This is internal to the Mamba computation and is NOT DAM-merged.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        # Apply gating: normed_hidden_states * silu(gate)
        hidden_states = hidden_states * F.silu(gate)
        return hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ---------------------------------------------------------------------------
# Activation: ReLU squared
# ---------------------------------------------------------------------------
def relu_squared(x):
    """ReLU squared activation: relu(x)^2."""
    return F.relu(x).pow(2)


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------
class MergedNemotronHRotaryEmbedding(nn.Module):
    """Rotary position embedding for Nemotron-H attention layers."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Input tensor, used only for dtype and device.
            position_ids: Position indices of shape ``(batch_size, seq_len)``.

        Returns:
            Tuple of ``(cos, sin)`` each of shape ``(batch_size, seq_len, head_dim)``.
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: The query tensor.
        k: The key tensor.
        cos: The cosine part of the rotary embedding.
        sin: The sine part of the rotary embedding.
        position_ids: Deprecated and unused.
        unsqueeze_dim: The dimension along which to unsqueeze cos and sin so they
            can be properly broadcast to the dimensions of q and k.

    Returns:
        Tuple of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# repeat_kv helper
# ---------------------------------------------------------------------------
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for grouped-query attention.

    The hidden states go from ``(batch, num_key_value_heads, seqlen, head_dim)``
    to ``(batch, num_attention_heads, seqlen, head_dim)``.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Expert MLP (non-gated, relu-squared) used within MoE
# ---------------------------------------------------------------------------
class NemotronHExpertMLP(nn.Module):
    """Non-gated MLP used for MoE experts and shared experts.

    Architecture: ``up_proj -> relu_squared -> down_proj``.
    All linear layers use DAMLinearLayer.
    """

    def __init__(self, config: MergedNemotronHConfig, is_shared: bool = False):
        super().__init__()
        intermediate = (
            config.moe_shared_expert_intermediate_size if is_shared else config.moe_intermediate_size
        )
        self.up_proj = DAMLinearLayer(
            config.hidden_size, intermediate, bias=config.mlp_bias, num_models=config.num_merged_models
        )
        self.down_proj = DAMLinearLayer(
            intermediate, config.hidden_size, bias=config.mlp_bias, num_models=config.num_merged_models
        )

    def forward(self, x):
        return self.down_proj(relu_squared(self.up_proj(x)))


# ---------------------------------------------------------------------------
# Attention (eager only, GQA)
# ---------------------------------------------------------------------------
class MergedNemotronHAttention(nn.Module):
    """Multi-headed grouped-query attention for Nemotron-H.

    Uses standard eager (manual) attention. Flash attention and SDPA
    variants are intentionally omitted for the DAM implementation.
    """

    def __init__(self, config: MergedNemotronHConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.q_proj = DAMLinearLayer(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            num_models=config.num_merged_models,
        )
        self.k_proj = DAMLinearLayer(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            num_models=config.num_merged_models,
        )
        self.v_proj = DAMLinearLayer(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            num_models=config.num_merged_models,
        )
        self.o_proj = DAMLinearLayer(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            num_models=config.num_merged_models,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass for attention.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: 4-D causal mask.
            position_ids: Position indices.
            past_key_value: Cached key/value states.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            cache_position: Cache position indices.
            position_embeddings: Pre-computed ``(cos, sin)`` from the rotary embedding.

        Returns:
            Tuple of ``(attn_output, attn_weights, past_key_value)``.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            # Fallback: should not happen when called from the model, but safe default
            raise ValueError("position_embeddings must be provided to MergedNemotronHAttention")

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast to fp32 for numerical stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Mamba-2 Mixer (pure PyTorch recurrent implementation)
# ---------------------------------------------------------------------------
class MergedNemotronHMamba2Mixer(nn.Module):
    """Mamba-2 (SSD) mixer layer for Nemotron-H.

    Implements a pure-PyTorch token-by-token recurrent forward pass so that
    no external ``mamba_ssm`` library is required.

    Key components:
        - ``in_proj``: DAMLinearLayer projecting to ``[gate, hidden_B_C, dt]``
        - ``conv1d``: standard depthwise Conv1d (NOT DAM-merged)
        - ``A_log``, ``dt_bias``, ``D``: standard ``nn.Parameter`` (NOT DAM-merged)
        - ``norm``: NemotronHGatedRMSNorm (NOT DAM-merged)
        - ``out_proj``: DAMLinearLayer
    """

    def __init__(self, config: MergedNemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.mamba_num_heads = config.mamba_num_heads
        self.mamba_head_dim = config.mamba_head_dim
        self.ssm_state_size = config.ssm_state_size
        self.n_groups = config.n_groups
        self.time_step_min = config.time_step_min

        # Derived sizes
        self.intermediate_size = self.mamba_num_heads * self.mamba_head_dim
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # Projection size: gate + conv_input + dt
        self.projection_size = self.intermediate_size + self.conv_dim + self.mamba_num_heads

        # Input projection (DAM-merged)
        self.in_proj = DAMLinearLayer(
            self.hidden_size,
            self.projection_size,
            bias=config.mamba_proj_bias,
            num_models=config.num_merged_models,
        )

        # Depthwise conv1d (standard, NOT DAM-merged)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
            bias=config.use_conv_bias,
        )

        # SSM parameters (standard nn.Parameter, NOT DAM-merged)
        self.A_log = nn.Parameter(torch.zeros(self.mamba_num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.mamba_num_heads))
        self.D = nn.Parameter(torch.zeros(self.mamba_num_heads))

        # Gated RMSNorm (NOT DAM-merged, internal to Mamba computation)
        self.norm = NemotronHGatedRMSNorm(self.intermediate_size, eps=config.layer_norm_epsilon)

        # Output projection (DAM-merged)
        self.out_proj = DAMLinearLayer(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mamba_proj_bias,
            num_models=config.num_merged_models,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for Mamba-2 mixer.

        Uses a pure PyTorch recurrent (token-by-token) computation.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: Padding mask of shape ``(batch, seq_len)`` where 1
                indicates a valid token and 0 indicates padding.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project input
        zxbcdt = self.in_proj(hidden_states)  # (B, L, projection_size)

        # Split into gate (z), conv input (xBC), and dt
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.intermediate_size, self.conv_dim, self.mamba_num_heads],
            dim=-1,
        )

        # Apply depthwise conv1d (causal)
        xBC = xBC.transpose(1, 2)  # (B, conv_dim, L)
        xBC = self.conv1d(xBC)[:, :, :seq_len]  # causal: trim to seq_len
        xBC = F.silu(xBC)
        xBC = xBC.transpose(1, 2)  # (B, L, conv_dim)

        # Split xBC into x, B, C
        x, B, C = torch.split(
            xBC,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # Reshape for SSM computation
        x = x.view(batch_size, seq_len, self.mamba_num_heads, self.mamba_head_dim)
        B = B.view(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.view(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        # Compute A from A_log
        A = -torch.exp(self.A_log.float())  # (num_heads,)

        # Compute dt with bias, softplus, and clamp
        dt = F.softplus(dt + self.dt_bias).clamp(min=self.time_step_min)  # (B, L, num_heads)

        # Expand B and C from groups to heads
        heads_per_group = self.mamba_num_heads // self.n_groups
        B = B.repeat_interleave(heads_per_group, dim=2)  # (B, L, num_heads, ssm_state_size)
        C = C.repeat_interleave(heads_per_group, dim=2)  # (B, L, num_heads, ssm_state_size)

        # Recurrent SSM computation
        state = torch.zeros(
            batch_size,
            self.mamba_num_heads,
            self.mamba_head_dim,
            self.ssm_state_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :, :]    # (B, num_heads, head_dim)
            B_t = B[:, t, :, :]    # (B, num_heads, ssm_state_size)
            C_t = C[:, t, :, :]    # (B, num_heads, ssm_state_size)
            dt_t = dt[:, t, :]     # (B, num_heads)

            # Discretize
            # dA: (B, num_heads, 1, 1)
            dA = torch.exp(dt_t.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(-1).unsqueeze(0))
            # dB: (B, num_heads, head_dim, ssm_state_size)
            dB = dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(2) * x_t.unsqueeze(-1)

            state = state * dA + dB

            # Output: y = (state @ C^T) + D * x
            y_t = (
                torch.einsum("bhds,bhs->bhd", state.to(x_t.dtype), C_t)
                + self.D.unsqueeze(0).unsqueeze(-1) * x_t
            )  # (B, num_heads, head_dim)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, num_heads, head_dim)
        y = y.view(batch_size, seq_len, -1)  # (B, L, intermediate_size)

        # Apply gated RMS norm (gate = z)
        y = self.norm(y, z)

        # Output projection
        y = self.out_proj(y)

        # Mask padding tokens
        if attention_mask is not None:
            y = y * attention_mask.unsqueeze(-1)

        return y


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------
class MergedNemotronHMoE(nn.Module):
    """Mixture of Experts layer for Nemotron-H.

    Uses sigmoid routing with top-k selection and a shared expert that
    is always active. The routing gate is a DAMLinearLayer.
    """

    def __init__(self, config: MergedNemotronHConfig):
        super().__init__()
        self.gate = DAMLinearLayer(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            num_models=config.num_merged_models,
        )
        self.experts = nn.ModuleList(
            [NemotronHExpertMLP(config, is_shared=False) for _ in range(config.n_routed_experts)]
        )
        self.shared_experts = NemotronHExpertMLP(config, is_shared=True)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for MoE.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Routing with sigmoid scores
        router_logits = self.gate(hidden_states_flat)
        routing_weights = torch.sigmoid(router_logits)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        # Normalize selected weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Compute expert outputs
        final_output = torch.zeros_like(hidden_states_flat)
        for expert_idx in range(len(self.experts)):
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                # Gather the weights assigned to this expert for each selected token
                weight_mask = topk_indices == expert_idx
                expert_weights = (topk_weights * weight_mask.float()).sum(dim=-1)
                final_output[expert_mask] += expert_output * expert_weights[expert_mask].unsqueeze(-1)

        # Always add the shared expert output
        shared_output = self.shared_experts(hidden_states_flat)
        final_output = final_output + shared_output

        return final_output.view(batch_size, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Decoder Block (dispatches to mamba / attention / moe)
# ---------------------------------------------------------------------------
class MergedNemotronHBlock(nn.Module):
    """A single Nemotron-H decoder block.

    Each block contains a single sub-module (mixer) which is one of:
        - ``MergedNemotronHMamba2Mixer`` for ``"mamba"`` layers
        - ``MergedNemotronHAttention`` for ``"attention"`` layers
        - ``MergedNemotronHMoE`` for ``"moe"`` layers

    The block applies pre-norm with a residual connection:
    ``output = hidden_states + mixer(norm(hidden_states))``.
    """

    def __init__(self, config: MergedNemotronHConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layers_block_type[layer_idx]

        # Pre-norm (DAM-merged or standard)
        if config.dam_layernorms:
            self.norm = DAMRMSNorm(
                config.hidden_size, eps=config.layer_norm_epsilon, num_models=config.num_merged_models
            )
        else:
            self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Mixer sub-module
        if self.block_type == "mamba":
            self.mixer = MergedNemotronHMamba2Mixer(config, layer_idx)
        elif self.block_type == "attention":
            self.mixer = MergedNemotronHAttention(config, layer_idx)
        elif self.block_type == "moe":
            self.mixer = MergedNemotronHMoE(config)
        else:
            raise ValueError(
                f"Unknown block type '{self.block_type}' at layer {layer_idx}. "
                "Expected one of 'mamba', 'attention', or 'moe'."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass for a single decoder block.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: Causal mask for attention blocks, padding mask for
                mamba blocks, or ``None`` for MoE blocks.
            position_ids: Position indices for attention layers.
            past_key_value: Cached key/value states (attention only).
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            cache_position: Cache position indices.
            position_embeddings: Pre-computed ``(cos, sin)`` for RoPE.

        Returns:
            Tuple starting with the output hidden states. May also contain
            attention weights and cache depending on the block type and flags.
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
            hidden_states = residual + hidden_states
            outputs = (hidden_states,)

        elif self.block_type == "attention":
            hidden_states, self_attn_weights, present_key_value = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)

        else:  # moe
            hidden_states = self.mixer(hidden_states)
            hidden_states = residual + hidden_states
            outputs = (hidden_states,)

        return outputs


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------
NEMOTRONH_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MergedNemotronHConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class MergedNemotronHPreTrainedModel(PreTrainedModel):
    """Base class for all Merged Nemotron-H DAM models."""

    config_class = MergedNemotronHConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MergedNemotronHBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True
    _supports_static_cache = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Input docstring
# ---------------------------------------------------------------------------
NEMOTRONH_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up
            sequential decoding.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# ---------------------------------------------------------------------------
# Model (decoder stack)
# ---------------------------------------------------------------------------
class MergedNemotronHModel(MergedNemotronHPreTrainedModel):
    """Nemotron-H hybrid transformer-mamba-moe decoder stack for DAM.

    Consists of ``config.num_hidden_layers`` blocks. Each block may be a Mamba-2
    SSM layer, a standard GQA attention layer, or a Mixture-of-Experts layer,
    as specified by ``config.layers_block_type``.

    Args:
        config: MergedNemotronHConfig
    """

    def __init__(self, config: MergedNemotronHConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding layer
        if config.dam_embedding_layer:
            self.embed_tokens = DAMEmbeddingLayer(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                num_models=config.num_merged_models,
                padding_idx=self.padding_idx,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder layers
        self.layers = nn.ModuleList(
            [MergedNemotronHBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm
        if config.dam_layernorms:
            self.norm = DAMRMSNorm(
                config.hidden_size, eps=config.layer_norm_epsilon, num_models=config.num_merged_models
            )
        else:
            self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Rotary embeddings (shared, computed once)
        self.rotary_emb = MergedNemotronHRotaryEmbedding(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward pass through the full Nemotron-H decoder stack.

        Args:
            input_ids: Token indices of shape ``(batch_size, sequence_length)``.
            attention_mask: Padding mask of shape ``(batch_size, sequence_length)``.
            position_ids: Position indices.
            past_key_values: Cached key/value states for attention layers.
            inputs_embeds: Pre-computed embeddings, alternative to ``input_ids``.
            use_cache: Whether to return updated caches.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a ``BaseModelOutputWithPast``.
            cache_position: Cache position indices.

        Returns:
            ``BaseModelOutputWithPast`` or tuple of tensors.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and "
                "will be removed in v4.43. Please use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build the 4-D causal mask for attention layers
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        # Padding mask for mamba layers (just the raw attention_mask)
        mamba_mask = attention_mask

        # Pre-compute rotary embeddings for attention layers
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Select the appropriate mask for each block type
            if decoder_layer.block_type == "attention":
                layer_mask = causal_mask
            elif decoder_layer.block_type == "mamba":
                layer_mask = mamba_mask
            else:  # moe
                layer_mask = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache and decoder_layer.block_type == "attention":
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions and decoder_layer.block_type == "attention":
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        use_cache: bool,
        output_attentions: bool,
    ):
        """Build a 4-D causal attention mask for the attention layers.

        This is only used by attention blocks; mamba blocks use the raw padding
        mask and MoE blocks use no mask.

        Args:
            attention_mask: 2-D or 4-D attention mask.
            input_tensor: Input embeddings, used for shape and device info.
            cache_position: Current cache positions.
            past_key_values: Cached states.
            use_cache: Whether caching is enabled.
            output_attentions: Whether attention weights are requested.

        Returns:
            4-D causal mask tensor or ``None``.
        """
        # cache_position must be valid here
        past_seen_tokens = cache_position[0] if past_key_values is not None else 0

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            # Already inverted form
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            exclude_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask *= exclude_mask
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.dim() == 2:
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )

        return causal_mask


# ---------------------------------------------------------------------------
# CausalLM head
# ---------------------------------------------------------------------------
class MergedNemotronHForCausalLM(MergedNemotronHPreTrainedModel):
    """Nemotron-H model with a causal language modeling head for DAM.

    Consists of the ``MergedNemotronHModel`` decoder and a linear ``lm_head``
    that projects hidden states to vocabulary logits.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MergedNemotronHConfig):
        super().__init__(config)
        self.model = MergedNemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.num_merged_models = config.num_merged_models
        self.lm_head = DAMLinearLayer(
            config.hidden_size, config.vocab_size, bias=False, num_models=config.num_merged_models
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def tie_weights(self):
        if isinstance(self.get_input_embeddings(), DAMEmbeddingLayer) and isinstance(self.lm_head, DAMLinearLayer):
            self.lm_head.tie_with_embeddings(self.get_input_embeddings())
        else:
            super().tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""Forward pass for the causal language model.

        Args:
            input_ids: Token indices of shape ``(batch_size, sequence_length)``.
            attention_mask: Padding mask.
            position_ids: Position indices.
            past_key_values: Cached key/value states.
            inputs_embeds: Pre-computed embeddings.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache: Whether to return updated caches.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a ``CausalLMOutputWithPast``.
            cache_position: Cache position indices.

        Returns:
            ``CausalLMOutputWithPast`` or tuple of tensors.

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> model = MergedNemotronHForCausalLM.from_pretrained("nvidia/nemotron-h")
        >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/nemotron-h")
        >>> prompt = "The future of AI is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs: (hidden_states, next_cache, all_hidden_states, all_self_attns)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """Prepare model inputs for auto-regressive generation.

        Handles cache slicing and position_ids creation for efficient
        generation with past key values.

        Args:
            input_ids: Current input token ids.
            past_key_values: Cached key/value states from previous steps.
            attention_mask: Attention/padding mask.
            inputs_embeds: Pre-computed embeddings (used on first step only).
            cache_position: Position indices into the cache.
            position_ids: Position indices for the current step.
            use_cache: Whether to use caching.

        Returns:
            Dictionary of model inputs.
        """
        # If we have cache: slice input_ids to keep only unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (Exception 2 is a no-op)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # Clone to avoid stride issues with torch.compile
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
