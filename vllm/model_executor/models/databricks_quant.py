# coding=utf-8
# Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ReplicatedLinear)
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.databricks import DatabricksConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.utils import set_weight_attrs

KVCache = Tuple[torch.Tensor, torch.Tensor]


class DatabricksRouter(nn.Module):
    """A Router implementation for Databricks that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: DatabricksConfig,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.d_model = config.d_model
        self.layer = ReplicatedLinear(self.d_model,
                                      self.num_total_experts,
                                      bias=False,
                                      params_dtype=params_dtype,
                                      linear_method=None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.layer(hidden_states)
        return router_logits


class DatabricksExperts(nn.Module):
    """A tensor-parallel MoE implementation for Databricks that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: DatabricksConfig,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k
        self.d_model = config.d_model
        self.intermediate_size = config.ffn_config.ffn_hidden_size

        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = DatabricksRouter(config, self.params_dtype)

        self.w1 = nn.ModuleList([
            ReplicatedLinear(self.d_model,
                             self.intermediate_size,
                             bias=False,
                             linear_method=linear_method)
                if idx in self.expert_indicies else None
                for idx in range(self.num_total_experts)])
        self.v1 = nn.ModuleList([
            ReplicatedLinear(self.d_model,
                             self.intermediate_size,
                             bias=False,
                             linear_method=linear_method)
                if idx in self.expert_indicies else None
                for idx in range(self.num_total_experts)])
        self.w2 = nn.ModuleList([
            ReplicatedLinear(self.intermediate_size,
                             self.d_model,
                             bias=False,
                             linear_method=linear_method)
                if idx in self.expert_indicies else None
                for idx in range(self.num_total_experts)])

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)
            w1_out, _ = self.w1[expert_idx](hidden_states)
            w1_out = self.act_fn(w1_out)
            v1_out, _ = self.v1[expert_idx](hidden_states)
            current_hidden_states = w1_out * v1_out
            current_hidden_states, _ = self.w2[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states.mul_(
                expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            batch_size, sequence_length, hidden_size)

class DatabricksAttention(nn.Module):

    def __init__(
        self,
        config: DatabricksConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.total_num_kv_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        self.rope_theta = config.attn_config.rope_theta
        self.max_position = config.max_seq_len

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_world_size
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=2)
        q, k = self.rotary_emb(position_ids, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        hidden_states, _ = self.out_proj(attn_output)
        return hidden_states


class DatabricksFusedNormAttention(nn.Module):

    def __init__(
        self,
        config: DatabricksConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DatabricksAttention(config, linear_method)
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        x = self.attn(position_ids=position_ids,
                      hidden_states=hidden_states,
                      kv_cache=kv_cache,
                      input_metadata=input_metadata)
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        return hidden_states, residual


class DatabricksBlock(nn.Module):

    def __init__(
        self,
        config: DatabricksConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.norm_attn_norm = DatabricksFusedNormAttention(
            config, linear_method)
        self.ffn = DatabricksExperts(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states, residual = self.norm_attn_norm(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class DatabricksModel(nn.Module):

    def __init__(
        self,
        config: DatabricksConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.blocks = nn.ModuleList([
            DatabricksBlock(config, linear_method)
            for _ in range(config.n_layers)
        ])
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5)
        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias,
                                                      nn.Parameter):
                # Remove the bias term in Linear and LayerNorm.
                module.register_parameter("bias", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class DatabricksForCausalLM(nn.Module):

    def __init__(
        self,
        config: DatabricksConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DatabricksModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.d_model,
                                      org_num_embeddings=config.vocab_size,
                                      padding_size=DEFAULT_VOCAB_PADDING_SIZE)
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        expert_params_mapping = [
            ("ws" if weight_name in ["w1", "v1"] else "w2s",
             f"experts.mlp.{weight_name}")
            for weight_name in ["w1", "v1", "w2"]
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "ffn.experts.mlp" in name:
                name = name.replace("experts.mlp.w1_", "w1.")
                name = name.replace("experts.mlp.v1_", "v1.")
                name = name.replace("experts.mlp.w2_", "w2.")
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
