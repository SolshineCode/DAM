"""PyTorch Merged Qwen 3.5 text model for DAM (Differentiable Adaptive Merging)."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .config import MergedQwen3_5Config
from ..dam import DAMLinearLayer, DAMEmbeddingLayer, DAMRMSNorm

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MergedQwen3_5Config"


# ---------------------------------------------------------------------------
# RMSNorm variants
# ---------------------------------------------------------------------------

class Qwen3_5RMSNorm(nn.Module):
    """Qwen 3.5 RMS normalization using the (1 + weight) formulation.

    Weight is initialized to zeros so that at init time the norm is an identity
    scaling (the ``1 +`` term). This is the non-DAM variant used when
    ``config.dam_layernorms`` is ``False``.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return ((1.0 + self.weight.float()) * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3_5DAMRMSNorm(DAMRMSNorm):
    """DAM-merged RMS normalization with the Qwen 3.5 ``(1 + weight)`` formulation.

    Inherits from :class:`DAMRMSNorm` and overrides ``forward`` so that the
    merged weight is used as ``(1 + merged_weight) * normed_input`` instead of
    the standard ``merged_weight * normed_input``.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.model_index is not None:
            weight = self.weights[self.model_index].to(hidden_states.device)
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return ((1.0 + weight.float()) * hidden_states).to(input_dtype)
        else:
            weight = self.get_dam_weight().to(hidden_states.device)
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return ((1.0 + weight.float()) * hidden_states).to(input_dtype)


class Qwen3_5GatedRMSNorm(nn.Module):
    """Gated RMS normalization used by the Gated Delta Network linear attention.

    This norm is **not** DAM-merged because it is specific to each layer and
    does not correspond to a weight that needs merging across models.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = (1.0 + self.weight.float()) * hidden_states
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary embeddings (partial)
# ---------------------------------------------------------------------------

class MergedQwen3_5RotaryEmbedding(nn.Module):
    """Rotary position embedding for Qwen 3.5.

    Only ``rotary_dim`` dimensions out of the full ``head_dim`` receive
    rotary encoding (controlled by ``partial_rotary_factor``).
    """

    def __init__(self, rotary_dim, max_position_embeddings=32768, base=10000000.0, device=None):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).float().to(device) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_heads, seq_len, head_dim]
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


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply partial rotary position embeddings to query and key tensors.

    ``cos`` and ``sin`` have shape ``[batch, seq_len, rotary_dim]``. Only the
    first ``rotary_dim`` dimensions of *q* and *k* are rotated; the rest pass
    through unchanged.
    """
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Utility: repeat KV heads
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for grouped-query attention.

    ``hidden_states`` has shape ``(batch, num_kv_heads, seqlen, head_dim)``
    and is expanded to ``(batch, num_kv_heads * n_rep, seqlen, head_dim)``.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MergedQwen3_5MLP(nn.Module):
    """Standard SwiGLU MLP with DAM-merged linear layers."""

    def __init__(self, config: MergedQwen3_5Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = DAMLinearLayer(
            self.hidden_size, self.intermediate_size, bias=False, num_models=config.num_merged_models
        )
        self.up_proj = DAMLinearLayer(
            self.hidden_size, self.intermediate_size, bias=False, num_models=config.num_merged_models
        )
        self.down_proj = DAMLinearLayer(
            self.intermediate_size, self.hidden_size, bias=False, num_models=config.num_merged_models
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# ---------------------------------------------------------------------------
# Full (softmax) attention with gating and QK-norm
# ---------------------------------------------------------------------------

class MergedQwen3_5Attention(nn.Module):
    """Multi-headed full attention with output gating and QK normalization.

    Key features specific to Qwen 3.5:
    * **Output gating**: ``q_proj`` produces 2x the normal output; the extra
      half is a *gate* applied after attention via ``sigmoid(gate) * attn_out``.
    * **QK normalization**: separate per-head RMSNorm on query and key states.
    * **Partial RoPE**: only ``partial_rotary_factor`` fraction of ``head_dim``
      receives rotary embeddings.
    * **Grouped-query attention**: ``num_key_value_heads`` may differ from
      ``num_attention_heads``.
    """

    def __init__(self, config: MergedQwen3_5Config, layer_idx: Optional[int] = None):
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
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.partial_rotary_factor = config.partial_rotary_factor
        self.attn_output_gate = config.attn_output_gate

        # Q outputs 2x for gating when attn_output_gate is True
        q_output_dim = 2 * self.num_heads * self.head_dim if self.attn_output_gate else self.num_heads * self.head_dim
        self.q_proj = DAMLinearLayer(
            self.hidden_size, q_output_dim, bias=config.attention_bias, num_models=config.num_merged_models
        )
        self.k_proj = DAMLinearLayer(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, num_models=config.num_merged_models
        )
        self.v_proj = DAMLinearLayer(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, num_models=config.num_merged_models
        )
        self.o_proj = DAMLinearLayer(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=config.attention_bias, num_models=config.num_merged_models
        )

        # QK norms (per head_dim)
        if config.dam_layernorms:
            self.q_norm = Qwen3_5DAMRMSNorm(
                self.head_dim, eps=config.rms_norm_eps, num_models=config.num_merged_models
            )
            self.k_norm = Qwen3_5DAMRMSNorm(
                self.head_dim, eps=config.rms_norm_eps, num_models=config.num_merged_models
            )
        else:
            self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        # Split Q into query and gate when gating is enabled
        if self.attn_output_gate:
            query_states, gate = torch.chunk(query_states, 2, dim=-1)
        else:
            gate = None

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to heads: (B, L, H, D) -> (B, H, L, D)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if gate is not None:
            gate = gate.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK norms
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply partial RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Expand KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast to fp32 for softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Apply output gate
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Linear attention: Gated Delta Network
# ---------------------------------------------------------------------------

class MergedQwen3_5GatedDeltaNet(nn.Module):
    """Gated Delta Network linear attention layer.

    This implements a recurrent linear attention mechanism using the *delta
    rule* with exponential gating. It does **not** require any external
    libraries (causal_conv1d, fla, etc.) -- everything is pure PyTorch.

    Layers that are **not** DAM-merged:
    * ``conv1d`` -- standard ``nn.Conv1d`` (depthwise causal convolution)
    * ``A_log``, ``dt_bias`` -- ``nn.Parameter``
    * ``norm`` -- ``Qwen3_5GatedRMSNorm`` (gated, not DAM-merged)

    All linear projections are ``DAMLinearLayer``.
    """

    def __init__(self, config: MergedQwen3_5Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.linear_num_key_heads = config.linear_num_key_heads
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim

        self.conv_dim = (
            self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )
        self.value_total_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # Linear projections (DAM-merged)
        self.in_proj_qkv = DAMLinearLayer(
            self.hidden_size, self.conv_dim, bias=False, num_models=config.num_merged_models
        )
        self.in_proj_z = DAMLinearLayer(
            self.hidden_size, self.value_total_dim, bias=False, num_models=config.num_merged_models
        )
        self.in_proj_b = DAMLinearLayer(
            self.hidden_size, self.linear_num_value_heads, bias=False, num_models=config.num_merged_models
        )
        self.in_proj_a = DAMLinearLayer(
            self.hidden_size, self.linear_num_value_heads, bias=False, num_models=config.num_merged_models
        )
        self.out_proj = DAMLinearLayer(
            self.value_total_dim, self.hidden_size, bias=False, num_models=config.num_merged_models
        )

        # Causal depthwise conv1d (standard, NOT DAM)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=config.linear_conv_kernel_dim,
            padding=config.linear_conv_kernel_dim - 1,
            groups=self.conv_dim,
        )

        # Learnable parameters (standard, NOT DAM)
        self.A_log = nn.Parameter(torch.zeros(self.linear_num_value_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.linear_num_value_heads))

        # Gated RMS norm (NOT DAM-merged)
        self.norm = Qwen3_5GatedRMSNorm(self.value_total_dim, eps=config.rms_norm_eps)

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
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs
        qkv = self.in_proj_qkv(hidden_states)  # (B, L, conv_dim)
        z = self.in_proj_z(hidden_states)        # (B, L, value_total_dim) -- gate
        beta = torch.sigmoid(self.in_proj_b(hidden_states))  # (B, L, num_v_heads)
        alpha = self.in_proj_a(hidden_states)    # (B, L, num_v_heads)

        # Causal depthwise conv1d + SiLU
        qkv = qkv.transpose(1, 2)  # (B, conv_dim, L)
        qkv = self.conv1d(qkv)[:, :, :seq_len]  # trim to causal (remove right padding)
        qkv = F.silu(qkv)
        qkv = qkv.transpose(1, 2)  # (B, L, conv_dim)

        # Split into Q, K, V
        q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        v_dim = self.linear_num_value_heads * self.linear_value_head_dim
        q, k, v = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)

        # Reshape to heads
        q = q.view(batch_size, seq_len, self.linear_num_key_heads, self.linear_key_head_dim)
        k = k.view(batch_size, seq_len, self.linear_num_key_heads, self.linear_key_head_dim)
        v = v.view(batch_size, seq_len, self.linear_num_value_heads, self.linear_value_head_dim)

        # L2 normalize Q and K
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Compute decay gate: g = -exp(A_log) * softplus(alpha + dt_bias)
        # alpha is (B, L, num_v_heads), dt_bias is (num_v_heads,)
        g = -torch.exp(self.A_log.float()) * F.softplus(alpha.unsqueeze(-1) + self.dt_bias)
        # g shape: (B, L, num_v_heads, 1)  -- alpha was (B, L, num_v_heads), after unsqueeze(-1) + broadcast
        # Actually alpha is (B, L, num_v_heads), dt_bias is (num_v_heads,), sum is (B, L, num_v_heads)
        # softplus gives (B, L, num_v_heads), multiply gives (B, L, num_v_heads)
        # We need to fix the shape computation:
        g = -torch.exp(self.A_log.float()) * F.softplus(alpha + self.dt_bias)  # (B, L, num_v_heads)
        g = g.transpose(1, 2).unsqueeze(-1)  # (B, num_v_heads, L, 1)

        # If num_v_heads > num_k_heads, repeat-interleave q and k
        num_kv_groups = self.linear_num_value_heads // self.linear_num_key_heads
        if num_kv_groups > 1:
            q = q.repeat_interleave(num_kv_groups, dim=2)
            k = k.repeat_interleave(num_kv_groups, dim=2)

        # Transpose for computation: (B, heads, L, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.unsqueeze(-1).transpose(1, 2)  # (B, num_v_heads, L, 1)

        # Recurrent delta rule computation
        state = torch.zeros(
            batch_size, self.linear_num_value_heads,
            self.linear_key_head_dim, self.linear_value_head_dim,
            dtype=torch.float32, device=hidden_states.device
        )
        outputs = []

        for t in range(seq_len):
            q_t = q[:, :, t:t+1, :]   # (B, H, 1, Dk)
            k_t = k[:, :, t:t+1, :]   # (B, H, 1, Dk)
            v_t = v[:, :, t:t+1, :]   # (B, H, 1, Dv)
            b_t = beta[:, :, t:t+1, :]  # (B, H, 1, 1)
            g_t = g[:, :, t:t+1, :]   # (B, H, 1, 1)

            # Decay
            decay = torch.exp(g_t)  # (B, H, 1, 1)

            # Delta rule:
            # kv_product = k_t @ state -> (B, H, 1, Dk) @ (B, H, Dk, Dv) = (B, H, 1, Dv)
            kv_product = torch.matmul(k_t, state.to(k_t.dtype))
            delta = (v_t - kv_product) * b_t  # (B, H, 1, Dv)

            # Update state: state = state * decay + k_t^T @ delta
            # k_t^T is (B, H, Dk, 1), delta is (B, H, 1, Dv) -> outer product (B, H, Dk, Dv)
            state = state * decay + torch.matmul(k_t.transpose(-1, -2), delta).to(state.dtype)

            # Output: q_t @ state = (B, H, 1, Dk) @ (B, H, Dk, Dv) = (B, H, 1, Dv)
            o_t = torch.matmul(q_t, state.to(q_t.dtype))
            outputs.append(o_t)

        output = torch.cat(outputs, dim=2)  # (B, H, L, Dv)
        output = output.transpose(1, 2).contiguous()  # (B, L, H, Dv)
        output = output.view(batch_size, seq_len, -1)  # (B, L, H*Dv)

        # Apply gated RMS norm
        output = self.norm(output, z)

        # Output projection
        output = self.out_proj(output)

        # Linear attention does not produce per-head attention weights
        return output, None, past_key_value


# ---------------------------------------------------------------------------
# Decoder layer (hybrid: dispatches to full or linear attention)
# ---------------------------------------------------------------------------

class MergedQwen3_5DecoderLayer(nn.Module):
    """Qwen 3.5 decoder layer that dispatches to either full (softmax) or
    linear (Gated Delta Network) attention based on ``config.layer_types``.

    Both branches share the same pre-/post-attention layer norms and MLP.
    """

    def __init__(self, config: MergedQwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        # Attention: pick based on layer type
        if self.layer_type == "linear_attention":
            self.linear_attn = MergedQwen3_5GatedDeltaNet(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = MergedQwen3_5Attention(config=config, layer_idx=layer_idx)

        # MLP
        self.mlp = MergedQwen3_5MLP(config)

        # Layer norms
        NormClass = (
            lambda size, eps: Qwen3_5DAMRMSNorm(size, eps=eps, num_models=config.num_merged_models)
            if config.dam_layernorms
            else lambda size, eps: Qwen3_5RMSNorm(size, eps=eps)
        )
        self.input_layernorm = (
            Qwen3_5DAMRMSNorm(config.hidden_size, eps=config.rms_norm_eps, num_models=config.num_merged_models)
            if config.dam_layernorms
            else Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )
        self.post_attention_layernorm = (
            Qwen3_5DAMRMSNorm(config.hidden_size, eps=config.rms_norm_eps, num_models=config.num_merged_models)
            if config.dam_layernorms
            else Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, 1, query_sequence_length, key_sequence_length)` for
                full attention, or `None` for linear attention.
            position_ids (`torch.LongTensor`, *optional*):
                position ids of shape `(batch_size, sequence_length)`.
            past_key_value (`Cache`, *optional*): cached past key and value projection states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
                Tuple of (cos, sin) for rotary position embeddings.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that inject code
                into the model.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Dispatch to appropriate attention
        if self.layer_type == "linear_attention":
            hidden_states, self_attn_weights, present_key_value = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------

QWEN3_5_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MergedQwen3_5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

QWEN3_5_INPUTS_DOCSTRING = r"""
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

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

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
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

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
# Pre-trained model base
# ---------------------------------------------------------------------------

class MergedQwen3_5PreTrainedModel(PreTrainedModel):
    config_class = MergedQwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MergedQwen3_5DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

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
# Text model (decoder)
# ---------------------------------------------------------------------------

class MergedQwen3_5TextModel(MergedQwen3_5PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a
    [`MergedQwen3_5DecoderLayer`] that uses either full or linear attention depending on the
    layer type specified in `config.layer_types`.

    Args:
        config: MergedQwen3_5Config
    """

    def __init__(self, config: MergedQwen3_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = (
            DAMEmbeddingLayer(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                num_models=config.num_merged_models,
                padding_idx=self.padding_idx,
            )
            if config.dam_embedding_layer
            else nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        )

        self.layers = nn.ModuleList(
            [MergedQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = (
            Qwen3_5DAMRMSNorm(config.hidden_size, eps=config.rms_norm_eps, num_models=config.num_merged_models)
            if config.dam_layernorms
            else Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )

        # Rotary embeddings (partial)
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        self.rotary_emb = MergedQwen3_5RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

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

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        hidden_states = inputs_embeds

        # Compute rotary embeddings once for all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
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
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
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
        """Build a 4-D causal attention mask for full-attention layers.

        Linear attention layers ignore the causal mask (they have their own
        causal structure via the recurrent delta rule), so this mask is only
        consumed by ``MergedQwen3_5Attention``.
        """
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
            # Already a 4-D mask in inverted form
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
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.dim() == 2:
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )

        return causal_mask


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class MergedQwen3_5ForCausalLM(MergedQwen3_5PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MergedQwen3_5Config):
        super().__init__(config)
        self.model = MergedQwen3_5TextModel(config)
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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from dam.modeling.qwen3_5.modeling import MergedQwen3_5ForCausalLM

        >>> model = MergedQwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-7B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-7B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s
                # `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride
                # during the decoding. Here, simply using `.contiguous()` is not sufficient as in the
                # batch size = 1 case, `position_ids` is already contiguous but with varying stride which
                # retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

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
