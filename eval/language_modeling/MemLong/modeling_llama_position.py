# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from .cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .configuration_llama import LlamaConfig
from .utils import ToolkitConfig,MemConfig,MemCache
from typing import Literal
from dataclasses import dataclass
from .toolkit import ToolKit

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa



@dataclass
class BaseModelOutputWithMem(BaseModelOutputWithPast):
    mem_update: Optional[MemCache] = None



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_for_relative_query(q, cos, sin,unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # q: [bs, nh, seq_len, dim]
    return q_embed

def apply_rotary_pos_emb_for_relative_keys(k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

# Based on transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def rotate_one(x, cos, sin, position_ids):
    if len(position_ids.shape) != 2 or x.shape[0] != position_ids.shape[0] or x.shape[-2] != position_ids.shape[1]:
        raise ValueError(f"Position ids shoud have shape [bsz, seq_len] got {position_ids.shape}")
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def rotate_as_if_first(x, rotary_emb):
    # x: [bs, num_attention_heads, seq_len, head_size]
    # apply rotary as if all elements were first in the sequence
    cos, sin = rotary_emb(x, torch.arange(x.shape[-2]).view(1,-1).type_as(x))
    return rotate_one(x, cos, sin, torch.zeros(x.shape[0], x.shape[-2], dtype=torch.long, device=cos.device))

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
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
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class RetGate(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.output_gate = torch.nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ret_score, attn_score):
        # print(self.output_gate[0][0][0])
        # print(self.sigmoid(self.output_gate[0][0][0]))
        return torch.cat((self.sigmoid(self.output_gate) * ret_score,(1 - self.sigmoid(self.output_gate)) * attn_score,),dim=-1,)


class RetrievalCausalAttention(nn.Module):
    
    def __init__(self, config:LlamaConfig, mem_config: MemConfig, layer_idx: Optional[int] = None):
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
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.max_local_cache = config.max_position_embeddings
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # control mem behavior
        self.mem_config = mem_config
        # begin of sentence
        self.clear_memory_on_bos_token_id = getattr(mem_config,"clear_memory_on_bos_token_id",False)
        # end of sentence
        self.clear_memory_on_eos_token_id = getattr(mem_config,"clear_memory_on_eos_token_id",False)
        # None;Zero;Continual
        self.position_type = config.position_type
        self.ret_gate = None
        # Only for Retrieval
        if getattr(mem_config,"use_gate",False):
            self.ret_gate = RetGate(self.num_heads)
        
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
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
        mem_caches: Optional[MemCache] = None,
        output_mem: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("pretraining_tp > 1 not supported for RetrievalCausalAttention")
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Whether to use positional encoding when saving 
        use_positionals = self.mem_config is None or getattr(self.mem_config,"positionals",True)
        # two conditions 
        # 1. use for saving or training 
        mem_no_local_cache = output_mem is True and past_key_value is None and (not use_cache)
        # 2. use for generation
        mem_and_local_cache = output_mem is True and use_cache
        mem_update = None
        if mem_no_local_cache:
            if use_positionals and self.position_type == "Zero":
                rfst_key_states = rotate_as_if_first(key_states, self.rotary_emb)
            else:
                rfst_key_states = key_states
            
            mem_update = MemCache(
                keys=rfst_key_states.detach().to(self.mem_config.cache_dtype),
                values=value_states.detach().to(self.mem_config.cache_dtype),
                masks=attention_mask[...,-1,:,None].detach()
            )   
            
        # Get local_seq_length
        if use_cache and past_key_value is not None:
            past_local_cache_size = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            local_seq_length = past_local_cache_size + hidden_states.shape[-2]
        else:
            local_seq_length = hidden_states.shape[-2]
                
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                merged_key_states = key_states
                merged_value_states = value_states
            else:
                merged_key_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=-2)
                merged_value_states = torch.cat([past_key_value.value_cache[self.layer_idx], value_states], dim=-2)
            # merged_position_ids = torch.cat([past_key_value.position_ids_cache[self.layer_idx], loc_position_ids], dim=-1)
            
            if attention_mask.shape[-1] != merged_key_states.shape[-2] and attention_mask.shape[-2] == query_states.shape[-2]:
                raise ValueError("attention_mask should be provided for all key_states in local context")

            assert local_seq_length == merged_key_states.shape[-2]
            
            if merged_key_states.shape[-2] > self.max_local_cache:
                # We drop half of max_local_cache for Memory    
                num_elems_to_drop = past_local_cache_size // 2
                # key_states,value_states = past_key_value.drop_and_update(drop_keys,drop_values,key_states,value_states,self.layer_idx)
                if mem_and_local_cache:
                    drop_keys = merged_key_states[..., :num_elems_to_drop, :]
                    drop_values = merged_value_states[...,:num_elems_to_drop,:]
                    drop_masks = attention_mask[...,-1,:,None]
                    drop_masks = drop_masks[:,:, :num_elems_to_drop, :]
                    
                    if use_positionals and self.position_type == "Zero":
                        rfst_drop_keys = rotate_as_if_first(drop_keys,self.rotary_emb)
                    else:
                        rfst_drop_keys = drop_keys
                    
                    mem_update = MemCache(
                        keys=rfst_drop_keys.to(self.mem_config.cache_dtype).detach(),
                        values=drop_values.to(self.mem_config.cache_dtype).detach(),
                        masks=drop_masks.to(self.mem_config.cache_dtype).detach(),
                    )            
                key_states, value_states , position_ids = past_key_value.drop_and_update(key_states,value_states,position_ids,num_elems_to_drop,self.layer_idx)
                attention_mask = attention_mask[..., num_elems_to_drop:]
            else:
                key_states, value_states , position_ids = past_key_value.update(key_states, value_states, position_ids , self.layer_idx)

        kv_seq_len = key_states.shape[-2]
        
        # Get mem_caches_length
        if mem_caches is not None:
            mem_caches_length = mem_caches.keys.shape[-2]
        else:
            mem_caches_length = 0
        
        if use_positionals and self.position_type == "Zero":
            loc_position_ids = torch.arange(1,kv_seq_len+1).view(1,-1).type_as(position_ids)
            mem_position_ids = torch.zeros((1,mem_caches_length)).type_as(position_ids)
        elif use_positionals and self.position_type == "Continual":
            loc_position_ids = torch.arange(mem_caches_length,mem_caches_length+kv_seq_len).view(1,-1).type_as(position_ids)
            mem_position_ids = torch.arange(mem_caches_length).view(1,-1).type_as(position_ids)
        else:
            # FIXME
            raise NotImplementedError("For normal generation")
        
        retrieval_upper =  mem_caches is not None and self.layer_idx in self.config.ret_attn_layers

        # We rotate the mem firstly
        if retrieval_upper and self.position_type=="Continual":
            mem_cos , mem_sin = self.rotary_emb(value_states, mem_position_ids)
            mem_caches.keys = apply_rotary_pos_emb_for_relative_keys(mem_caches.keys, mem_cos,mem_sin)
            
        loc_cos, loc_sin = self.rotary_emb(value_states, loc_position_ids)
        query_states = apply_rotary_pos_emb_for_relative_query(query_states, loc_cos[:,-query_states.shape[-2]:], loc_sin[:,-query_states.shape[-2]:])
        key_states = apply_rotary_pos_emb_for_relative_keys(key_states, loc_cos, loc_sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        loc_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        loc_attn_weights = loc_attn_weights + attention_mask

        if retrieval_upper:
            mem_mask = mem_caches.masks.squeeze(-1).unsqueeze(-2)
            # TODO: ret granularity
            mem_attn_weights = torch.matmul(query_states,mem_caches.keys.transpose(2, 3).to(key_states.dtype),) / math.sqrt(self.head_dim)
            
            assert mem_mask.shape[2] == 1
            mem_attn_weights = mem_attn_weights + mem_mask
            if self.ret_gate:
                attn_weights = self.ret_gate(mem_attn_weights, loc_attn_weights)
            else:
                attn_weights = torch.concat((mem_attn_weights,loc_attn_weights),dim=-1)
            combined_value_states = torch.concat([mem_caches.values.to(value_states.dtype),value_states],dim=-2,)
        else:
            attn_weights = loc_attn_weights
            combined_value_states = value_states
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, combined_value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, mem_update

class RetrievalFlashAttention2(RetrievalCausalAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self.ret_gate = None
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        mem_caches: Optional[MemCache] = None,
        output_mem: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        
        if attention_mask == None:
            tgt_seq_len = hidden_states.shape[-2]
            if past_key_value is not None:
                src_seq_len = past_key_value.get_seq_length(self.layer_idx) + tgt_seq_len
            else:
                src_seq_len = tgt_seq_len
            attention_mask = torch.ones(hidden_states.shape[0],tgt_seq_len).type_as(hidden_states)
            rfst_attention_mask = self._gen_causal_mask(attention_mask,cache_position,src_seq_len,tgt_seq_len)
        else:
            rfst_attention_mask = attention_mask

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Whether to use positional encoding when saving 
        use_positionals = self.mem_config is None or getattr(self.mem_config,"positionals",True)
        # two conditions 
        # 1. use for saving or training 
        mem_no_local_cache = output_mem is True and past_key_value is None and (not use_cache)
        # 2. use for generation
        mem_and_local_cache = output_mem is True and use_cache
        mem_update = None
        if mem_no_local_cache:
            if use_positionals and self.position_type == "Zero":
                rfst_key_states = rotate_as_if_first(key_states, self.rotary_emb)
            else:
                rfst_key_states = key_states
                
            mem_update = MemCache(
                keys=rfst_key_states.detach().to(self.mem_config.cache_dtype),
                values=value_states.detach().to(self.mem_config.cache_dtype),
                masks=rfst_attention_mask[...,-1,:,None].detach()
            )
        # Get local_seq_length
        if use_cache and past_key_value is not None:
            past_local_cache_size = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            local_seq_length = past_local_cache_size + hidden_states.shape[-2]
        else:
            local_seq_length = hidden_states.shape[-2]
        
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                merged_key_states = key_states
                merged_value_states = value_states
            else:
                merged_key_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=-2)
                merged_value_states = torch.cat([past_key_value.value_cache[self.layer_idx], value_states], dim=-2)
            if rfst_attention_mask.shape[-1] != merged_key_states.shape[-2] and rfst_attention_mask.shape[-2] == query_states.shape[-2]:
                raise ValueError("attention_mask should be provided for all key_states in local context")

            assert local_seq_length == merged_key_states.shape[-2]
            
            if merged_key_states.shape[-2] > self.max_local_cache:
                # We drop half of max_local_cache for Memory    
                num_elems_to_drop = past_local_cache_size // 2
                # key_states,value_states = past_key_value.drop_and_update(drop_keys,drop_values,key_states,value_states,self.layer_idx)
                if mem_and_local_cache:
                    drop_keys = merged_key_states[..., :num_elems_to_drop, :]
                    drop_values = merged_value_states[...,:num_elems_to_drop,:]
                    drop_masks = rfst_attention_mask[...,-1,:,None]
                    drop_masks = drop_masks[:,:, :num_elems_to_drop, :]
                    
                    if use_positionals and self.position_type == "Zero":
                        rfst_drop_keys = rotate_as_if_first(drop_keys,self.rotary_emb)
                    else:
                        rfst_drop_keys = drop_keys
                    
                    mem_update = MemCache(
                        keys=rfst_drop_keys.to(self.mem_config.cache_dtype).detach(),
                        values=drop_values.to(self.mem_config.cache_dtype).detach(),
                        masks=drop_masks.to(self.mem_config.cache_dtype).detach(),
                    )            
                key_states, value_states , position_ids = past_key_value.drop_and_update(key_states,value_states,position_ids,num_elems_to_drop,self.layer_idx)
                attention_mask = attention_mask[..., num_elems_to_drop:]
            else:
                key_states, value_states , position_ids = past_key_value.update(key_states, value_states, position_ids , self.layer_idx)

        kv_seq_len = key_states.shape[-2]
        
        # Get mem_caches_length
        if mem_caches is not None:
            mem_caches_length = mem_caches.keys.shape[-2]
        else:
            mem_caches_length = 0
        
        if use_positionals and self.position_type == "Zero":
            loc_position_ids = torch.arange(1,kv_seq_len+1).view(1,-1).type_as(position_ids)
            mem_position_ids = torch.zeros((1,mem_caches_length)).type_as(position_ids)
        elif use_positionals and self.position_type == "Continual":
            loc_position_ids = torch.arange(mem_caches_length,mem_caches_length+kv_seq_len).view(1,-1).type_as(position_ids)
            mem_position_ids = torch.arange(mem_caches_length).view(1,-1).type_as(position_ids)
        else:
            # FIXME
            raise NotImplementedError("For normal generation")
        
        retrieval_upper =  mem_caches is not None and self.layer_idx in self.config.ret_attn_layers

        # We rotate the mem firstly
        if retrieval_upper and self.position_type=="Continual":
            mem_cos , mem_sin = self.rotary_emb(value_states, mem_position_ids)
            mem_caches.keys = apply_rotary_pos_emb_for_relative_keys(mem_caches.keys, mem_cos,mem_sin)
            
        loc_cos, loc_sin = self.rotary_emb(value_states, loc_position_ids)
        query_states = apply_rotary_pos_emb_for_relative_query(query_states, loc_cos[:,-query_states.shape[-2]:], loc_sin[:,-query_states.shape[-2]:])
        key_states = apply_rotary_pos_emb_for_relative_keys(key_states, loc_cos, loc_sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
            
        loc_attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate)
        
        if retrieval_upper:
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )
                ret_key_states = mem_caches.keys.to(target_dtype)
                ret_value_states = mem_caches.values.to(target_dtype)
                mem_mask = mem_caches.masks.squeeze(-1).squeeze(-2)
                mem_mask = (mem_mask == 0).to(target_dtype)
            else:
                ret_key_states = mem_caches.keys
                ret_value_states = mem_caches.values
                mem_mask = mem_caches.masks.squeeze(-1).squeeze(-2)
                mem_mask = (mem_mask == 0).to(input_dtype)
            ret_key_states = ret_key_states.transpose(1, 2)
            ret_value_states = ret_value_states.transpose(1, 2)
            ret_attn_output = self._flash_attention_forward(query_states, ret_key_states, ret_value_states, mem_mask, q_len, dropout=dropout_rate)
            attn_output = loc_attn_output + ret_attn_output
        else:
            attn_output = loc_attn_output
        
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, mem_update
    
    def _gen_causal_mask(self,attention_mask,cache_position,sequence_length,target_length):
        dtype, device = attention_mask.dtype, attention_mask.device
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
                    (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(attention_mask.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
        )
        return causal_mask
    
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": RetrievalCausalAttention,
    "flash_attention_2": RetrievalFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int,mem_config: Optional[MemConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mem_config = mem_config
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, mem_config=mem_config,layer_idx=layer_idx)
        self.output_mem = layer_idx == config.mem_layer
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mem_caches=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value , mem_update= self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            mem_caches=mem_caches,
            output_mem=self.output_mem,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (mem_update,)
        
        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

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


LLAMA_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        if config.mem_layer < 0 or config.mem_layer >= config.num_hidden_layers:
            raise ValueError(f"mem_layer should be between 0 and {config.num_hidden_layers - 1}")
        assert isinstance(config.ret_attn_layers, list), "ret_attn_layers should be a list"
        self.mem_config = MemConfig(
            positionals=config.mem_positionals,
            use_gate=config.use_gate,
            cache_dtype=getattr(torch, config.mem_dtype),
        )
        self.ret_attn_layers = config.ret_attn_layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])        
        layers = []
        for layer_id in range(config.num_hidden_layers):
            if layer_id in self.ret_attn_layers or layer_id == config.mem_layer:
                layer = LlamaDecoderLayer(config, layer_id, mem_config=self.mem_config)
            else:
                layer = LlamaDecoderLayer(config, layer_id)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        mem_caches: Optional[Tuple[Optional[MemCache]]] = None,
    ) -> Union[Tuple, BaseModelOutputWithMem]:
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
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        mem_update = None
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
                    mem_caches=mem_caches,
                )

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if layer_outputs[-1] is not None:
                mem_update = layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithMem(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mem_update=mem_update,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

def _split_past_key_values(past_key_values):
    # splits past_key_values to local cache and memory cache
    local_cache_present = False
    mem_caches_present = False
    if past_key_values is not None:
        local_caches = ()
        mem_caches = ()
        for layer in past_key_values:
            if len(layer) != 6:
                raise ValueError(
                    "Expected elements of past_key_values to contain 6 elements."
                    "First 3 describing local cache and last 3 describing memory cache."
                    f"Instead got {len(layer)} elements"
                )
            else:
                lk, lv, li, memk, memv, memm = layer
                if lk.shape[-2] != 0:
                    local_cache_present = True
                    local_caches += ((lk, lv, li),)
                else:
                    local_caches += (None,)

                if memk.shape[-2] != 0:
                    mem_caches_present = True
                    #FIXME
                    mem_caches += (MemCache(keys=memk, values=memv, masks=memm),)
                else:
                    mem_caches += (None,)

    local_caches = local_caches if local_cache_present else None
    mem_caches = mem_caches if mem_caches_present else None

    return local_caches, mem_caches

def _clear_memory(input_ids, toolkit, target:Literal["bos","eos"]=None):
    if target == "bos":
        clear_memory = (input_ids == toolkit.bos_token_id).any(dim=-1)
    else:
        clear_memory = (input_ids == toolkit.eos_token_id).any(dim=-1)
    indices = [i for i, x in enumerate(clear_memory) if x]
    if indices != []:
        print(f"Meet special token. Clear memory at {indices}")
        toolkit.reset_by_batch(indices)
    return indices

def _retrieve(
    toolkit,
    input_ids,
    ret_group_size,
):
    if input_ids.shape[-1] > toolkit.get_max_seq_length():
        input_ids = input_ids[..., -toolkit.get_max_seq_length():]
        
    # Retrieve the memory cache
    queries = toolkit.get_texts(input_ids)
    # [bsz,k*?,num_heads,head_dim] | k (bsz,?,num_heads,head_dim)
    mem_caches = toolkit.retrieve(queries=queries,k=ret_group_size)
    return mem_caches

def _update(
    input_ids,
    mem_update:MemCache,
    context_window_length,
    toolkit,
    clear_memory=None,
    clear_mode: Literal["bos","eos"] = None
) -> None :
    if mem_update == None:
        return

    mem_length = mem_update.keys.shape[-2]
    assert mem_length <= input_ids.shape[-1]
    re_input_ids = input_ids[..., :mem_length]
    spc_token = None
    if clear_mode == "bos":
        spc_token = toolkit.bos_token_id
    elif clear_mode == "eos":
        spc_token = toolkit.eos_token_id    
    for length in range(0,mem_length,context_window_length):
        now_mem_update = MemCache()
        now_mem_update.keys = mem_update.keys[...,length:length+context_window_length,:]
        now_mem_update.values = mem_update.values[...,length:length+context_window_length,:]
        now_mem_update.masks = mem_update.masks[...,length:length+context_window_length,:]
        now_mem_update.texts = toolkit.get_texts(re_input_ids[...,length:length+context_window_length])
        # FIXME: 
        if clear_memory == True:
            positions = (re_input_ids == spc_token).nonzero(as_tuple=True)
            batch_indices,seq_indices = positions
            for bsz_idx in range(re_input_ids.shape[0]):
                if bsz_idx in batch_indices:
                    idxs = (batch_indices == bsz_idx).nonzero(as_tuple=True)[0]
                    last_spc_position = seq_indices[idxs].max().item()
                    if last_spc_position == re_input_ids.shape[1] - 1:
                        textList.append(None)
                        kvList.append(None)
                    else:
                        tokens = re_input_ids[bsz_idx, last_spc_position + 1 :]
                        examples = toolkit.get_texts(
                            [
                                tokens[i : i + toolkit.mem_granularity]
                                for i in range(0, len(tokens), toolkit.mem_granularity)
                            ],
                            skip_special_tokens=True,
                        )
                        textList.append(examples)
                        kvList.append(
                            {
                                "k": mem_update["k"][bsz_idx, last_spc_position + 1 :],
                                "v": mem_update["v"][bsz_idx, last_spc_position + 1 :],
                            }
                        )
                else:
                    tokens = re_input_ids[bsz_idx]
                    examples = toolkit.get_texts(
                        [
                            tokens[i : i + toolkit.mem_granularity]
                            for i in range(0, len(tokens), toolkit.mem_granularity)
                        ],
                        skip_special_tokens=True,
                    )
                    textList.append(examples)
                    kvList.append(
                        {
                            "k": mem_update["k"][bsz_idx],
                            "v": mem_update["v"][bsz_idx],
                        }
                    )
        else:
            toolkit.update(mem_update=now_mem_update)

def _prepare_pos_ids(past_key_values, batch_size, input_length, device):
    if past_key_values is not None:
        # take previous max pos_id + 1
        if past_key_values[0][2].shape[0] != batch_size:
            raise ValueError(
                f"first dimension of past_key_values should match batch size: {batch_size}"
                f"but got {past_key_values[0][2].shape[0]}"
            )
        next_pos = torch.max(past_key_values[0][2].view(batch_size, -1), dim=-1)[0] + 1
        next_pos = next_pos.view(batch_size, 1)
    else:
        next_pos = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

    position_ids = torch.arange(0, input_length, dtype=torch.long, device=device).view(1, input_length)
    position_ids = position_ids + next_pos
    return position_ids

# 如果position_type是Zero，那么position_ids之前就已经定义了
# 如果position_type是Continual，那么position_ids需要在这里定义

def _handle_mem_caches(
    mem_caches_list:Optional[List[List[Optional[MemCache]]]],
):
    if mem_caches_list == None:
        return None
    
    def _concat(mem_caches_list:List[List[Optional[MemCache]]]) -> List[Optional[MemCache]]:
        sigMemCaches = []
        max_length = 0    
        for i , mem_caches in enumerate(mem_caches_list):
            if mem_caches is None:
                sigMemCaches.append(None)
                continue
            mem_caches = [mem_cache[i] for mem_cache in mem_caches if mem_cache is not None]
            if mem_caches == []:
                sigMemCaches.append(None)
                continue
            texts = [mem_cache.text[i] for mem_cache in mem_caches]
            embeddings = [mem_cache.embeddings[i] for mem_cache in mem_caches]
            keys = torch.cat([mem_cache.keys[i] for mem_cache in mem_caches],dim=1)
            values = torch.cat([mem_cache.values[i] for mem_cache in mem_caches],dim=1)
            masks = torch.cat([mem_cache.masks[i] for mem_cache in mem_caches],dim=1)
            assert keys.shape[0] == values.shape[0] == masks.shape[0]
            max_length = max(max_length,keys.shape[0])
            sigMemCaches.append(MemCache(texts=texts,embeddings=embeddings,keys=keys,values=values,masks=masks,length=keys.shape[0]))
        return sigMemCaches , max_length
    
    # mem_caches_list 包含bsz 个列表，每个列表中包含 k 个 mem_caches
    sigMemCaches , max_length = _concat(mem_caches_list=mem_caches_list)
    # to one cache
    mem_caches = MemCache()
    # fill into max_length
    for sigMemCache in sigMemCaches:
        length = sigMemCache.length
        if length < max_length:
            pad_length = max_length - length
            sigMemCache.keys = F.pad(sigMemCache.keys,(0,0,0,pad_length))
            sigMemCache.values = F.pad(sigMemCache.values,(0,0,0,pad_length))
            sigMemCache.masks = F.pad(sigMemCache.masks,(0,0,0,pad_length))
    texts,embeddings, keys,values,masks = [], [] ,[], [] , []
    for sigMemCache in sigMemCaches:
        texts.append(sigMemCache.texts)
        embeddings.append(sigMemCache.embeddings)
        keys.append(sigMemCache.keys)
        values.append(sigMemCache.values)
        masks.append(sigMemCache.masks)
    mem_caches.texts = texts
    mem_caches.embeddings = embeddings
    mem_caches.keys = torch.stack(keys,dim=0)
    mem_caches.values = torch.stack(values,dim=0)
    mem_caches.masks = torch.stack(masks,dim=0)
    
    return mem_caches

def _handle_long_input(
    model,
    toolkit,
    ret_group_size,
    last_context_length,
    max_mem_size,
    context_window_length,
    clear_memories_on_bos_token_id,
    clear_memories_on_eos_token_id,
    input_ids,
    past_input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    use_cache,
    output_attentions,
    output_hidden_states,
    return_dict,
    cache_position,
):
    """
    Handle input that is too long for the model by splitting it in chunks and then concatenating the result.
    """
    
    
    # Split the input into context window and last context
    if output_attentions:
        logger.warning(
            f"Outputing attentions is not supported in MemLong"
            f"Attention of the last window will be returned"
        )

    if past_key_values is not None and use_cache is False:
        raise ValueError("past_key_values it not None should imply use_cache == True")

    if past_key_values is not None:
        initial_past_key_values_length = past_key_values[0][0].shape[-2]
    else:
        initial_past_key_values_length = 0
    
    if input_ids is not None:
        batch_size , input_length = input_ids.shape
    else:
        batch_size , input_length , _ = inputs_embeds.shape 

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = _prepare_pos_ids(past_key_values, batch_size, input_length , device)
        
    if position_ids.shape != (batch_size, input_length):
        raise ValueError(f"Shape of position_ids [{position_ids}] should match [{batch_size, input_length}]")

    if attention_mask is not None:
        attention_mask = attention_mask[..., -(initial_past_key_values_length + input_length) :]
        if attention_mask is not None and (
            attention_mask.shape != (batch_size, initial_past_key_values_length + input_length)
        ):
            raise ValueError(
                "Attention mask should be provided for both the local cache and the input",
                f"Expected shape {(batch_size, initial_past_key_values_length + input_length)},"
                f"got {attention_mask.shape}.",
            )

    if toolkit == None:
        outputs = model(
            input_ids=input_ids if input_ids is not None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds if inputs_embeds is not None else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            mem_caches=None,
        )
        outputs_list = [outputs]
    else:
        MemForwardNum = max(input_length - last_context_length,0)
        MemBankSize = max(MemForwardNum, 0)
        outputs_list = []
        attn_offset = initial_past_key_values_length
        if MemBankSize > 0:
            for i in range(MemForwardNum-MemBankSize , MemForwardNum , context_window_length):
                beg,end = i, min(MemForwardNum, i + context_window_length)    
                if attention_mask is not None:
                    if past_key_values is not None:
                        local_cache_size = past_key_values[0][0].shape[-2]
                    else:
                        local_cache_size = 0
                    attn_length = attention_mask.shape[-1]
                    attn_beg = beg + attn_offset - local_cache_size
                    attn_end = end + attn_offset

                mem_caches = _retrieve(toolkit, input_ids[..., beg:end] if input_ids is not None else None,ret_group_size=ret_group_size)
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids[..., beg:end] if input_ids is not None else None,
                        attention_mask=attention_mask[..., attn_beg:attn_end] if attention_mask is not None else None,
                        position_ids=position_ids[..., beg:end],
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds[..., beg:end, :] if inputs_embeds is not None else None,
                        use_cache=False if past_key_values is None else use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True,
                        cache_position=cache_position[beg:end] if cache_position is not None else None,
                        mem_caches=mem_caches,
                    )

                mem_update = outputs.mem_update  
                outputs.mem_update = None
                past_key_values = outputs.past_key_values  
                outputs.past_key_values = None
                outputs_list.append(outputs)
                clear_memory , clear_mode = False , None
                if clear_memories_on_bos_token_id:
                    _clear_memory(input_ids,token_id="bos",toolkit=toolkit)
                    clear_memory , clear_mode = True , "bos"
                elif clear_memories_on_eos_token_id:
                    _clear_memory(input_ids,token_id="eos",toolkit=toolkit)
                    clear_memory , clear_mode = True , "eos"
                _update(input_ids[...,beg:end],mem_update,context_window_length,toolkit,clear_memory,clear_mode)
        
        remaining_input_length = input_length - MemForwardNum
        beg = MemForwardNum
        attn_length = remaining_input_length
        if past_key_values is not None:
            attn_length += past_key_values[0][0].shape[-2]
        attention_mask = attention_mask[..., -attn_length:] if attention_mask is not None else None
        if past_input_ids is None:
            mem_caches = _retrieve(toolkit, input_ids[..., beg:] if input_ids is not None else None,ret_group_size=ret_group_size)
        else:
            mem_caches = _retrieve(toolkit, input_ids=torch.concat((past_input_ids,input_ids),dim=-1) if input_ids is not None else None,ret_group_size=ret_group_size)
        outputs = model(
            input_ids=input_ids[..., beg:] if input_ids is not None else None,
            attention_mask=attention_mask,
            position_ids=position_ids[..., beg:],
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds[..., beg:, :] if inputs_embeds is not None else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            mem_caches=mem_caches,
        )
        outputs_list.append(outputs) 
        clear_memory , clear_mode = False , None
        if clear_memories_on_bos_token_id:
            _clear_memory(input_ids,token_id="bos",toolkit=toolkit)
            clear_memory , clear_mode = True , "bos"
        elif clear_memories_on_eos_token_id:
            _clear_memory(input_ids,token_id="eos",toolkit=toolkit)
            clear_memory , clear_mode = True , "eos"
        _update(past_input_ids if past_input_ids is not None else input_ids[...,beg:],outputs.mem_update,context_window_length, toolkit,clear_memory,clear_mode)

    if output_hidden_states:
        hidden_states = ()
        for hd in zip(*[x.hidden_states for x in outputs_list]):
            hidden_states += (torch.cat(hd, dim=-2))
    else:
        hidden_states = None
            
    past_key_values = outputs_list[-1].past_key_values
    outputs = BaseModelOutputWithPast(
        last_hidden_state=torch.concat([x.last_hidden_state for x in outputs_list], dim=-2),
        past_key_values=past_key_values,
        hidden_states=hidden_states,
        attentions=outputs_list[-1].attentions,
    )
    if not return_dict:
        outputs = tuple(v for v in [outputs.last_hidden_state, outputs.past_key_values, outputs.hidden_states, outputs.attentions] if v is not None)
    return outputs
    

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, toolkit_config:ToolkitConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.toolkit = ToolKit(model_config=config,toolkit_config=toolkit_config,device=self.model.device)
        self.max_mem_size = config.memory_size 
        # self.mem_group_size = config.mem_group_size
        self.ret_group_size = config.ret_group_size
        self.context_window_length = min(config.max_position_embeddings, self.toolkit.get_max_seq_length(),config.mem_group_size)
        self.clear_memories_on_bos_token_id = config.clear_memories_on_bos_token_id
        self.clear_memories_on_eos_token_id = config.clear_memories_on_eos_token_id
        self.position_type = config.position_type 
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_input_ids: torch.LongTensor = None,
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
        last_context_length: Optional[int] = None,
        use_toolkit: Optional[bool] = True,
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
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if self.toolkit and not use_toolkit:
            print("Warning: You are not using toolkit which may infect the performance of the model.")
            self.toolkit = None
            
        if self.toolkit is not None and self.model.device != self.toolkit.device:
            self.toolkit.to(self.model.device)
            
        last_context_length = (last_context_length if last_context_length is not None else self.config.last_context_length)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = _handle_long_input(
            model=self.model,
            toolkit=self.toolkit,
            # mem_group_size=self.mem_group_size,
            ret_group_size=self.ret_group_size,
            context_window_length=self.context_window_length,
            last_context_length=last_context_length,
            max_mem_size=self.max_mem_size,
            clear_memories_on_bos_token_id=self.clear_memories_on_bos_token_id,
            clear_memories_on_eos_token_id=self.clear_memories_on_eos_token_id,
            input_ids=input_ids,
            past_input_ids=past_input_ids,
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
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
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

    def set_toolkit_tokenizer(self, tokenizer):
        self.toolkit.set_tokenizer(tokenizer)
    
    def reset_memory(self):
        self.toolkit.reset()
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        last_context_length=None,
        use_toolkit=None,
        **kwargs,
    ):        
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                past_input_ids = input_ids[:,:-1]
                input_ids = input_ids[:,-1:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]
        model_inputs.update(
            {
                "past_input_ids": past_input_ids,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "use_toolkit":use_toolkit,
                "last_context_length": last_context_length,
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
