# coding=utf-8
# Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from typing import Callable

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ReplicatedLinear,
                                               UnquantizedLinearMethod, MergedColumnParallelLinear)
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
from vllm.transformers_utils.configs.dbrx import DbrxConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.utils import set_weight_attrs

KVCache = Tuple[torch.Tensor, torch.Tensor]


class DbrxRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: DbrxConfig,
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

class DbrxExpertMLP(torch.nn.Module):

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation_fn = activation_fn

        self.up_proj = ReplicatedLinear(
            in_features=hidden_size,
            out_features=ffn_hidden_size,
            bias=False,
        )
        self.down_proj = ReplicatedLinear(
            in_features=ffn_hidden_size,
            out_features=hidden_size,
            bias=False,
        )
        self.gate_proj = ReplicatedLinear(
            in_features=hidden_size,
            out_features=ffn_hidden_size,
            bias=False,
        )

    def forward(self, x):
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = self.down_proj(x1)
        return x1

class DbrxMoE(nn.Module):

    def __init__(self, hidden_size: int, ffn_hidden_size: int,
                 moe_num_experts: int, activation_fn: Callable[[torch.Tensor],
                                                               torch.Tensor]):
        super().__init__()
        self.moe_num_experts = moe_num_experts
        self.activation_fn = activation_fn
        self.mlp = nn.ModuleList([
            DbrxExpertMLP(hidden_size=hidden_size,
                                ffn_hidden_size=ffn_hidden_size,
                                activation_fn=self.activation_fn)
            for i in range(self.moe_num_experts)
        ])

    def forward(self, x, weights, top_weights, top_experts):
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(
            top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = self.mlp[expert_idx](expert_tokens) * top_weights[
                token_list, topk_list, None]

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out

class DbrxExperts(nn.Module):
    """A tensor-parallel MoE implementation for DBRX that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k
        self.d_model = config.d_model
        self.intermediate_size = config.ffn_config.ffn_hidden_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.linear_method = linear_method

        self.router = DbrxRouter(config, self.params_dtype)
        if self.linear_method and not isinstance(
                self.linear_method, UnquantizedLinearMethod
        ) and self.linear_method.quant_config.support_fused_moe():
            self.ws = MergedColumnParallelLinear(self.d_model,
                                                    [self.intermediate_size] * 2,
                                                    bias=False,
                                                    linear_method=linear_method,
                                                    num_experts=self.num_total_experts)
            self.w2s = RowParallelLinear(self.intermediate_size,
                                            self.d_model,
                                            bias=False,
                                            linear_method=linear_method,
                                            num_experts=self.num_total_experts)
        else:
            self.ws = nn.Parameter(
                torch.empty(self.num_total_experts,
                            2 * self.intermediate_size,
                            self.d_model,
                            device="cuda",
                            dtype=self.params_dtype))
            self.w2s = nn.Parameter(
                torch.empty(self.num_total_experts,
                            self.d_model,
                            self.intermediate_size,
                            device="cuda",
                            dtype=self.params_dtype))
            print('DbrxExperts: using fused_moe')
            set_weight_attrs(self.ws, {
                "weight_loader": self.weight_loader,
            })
            set_weight_attrs(self.w2s, {
                "weight_loader": self.weight_loader,
            })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model])
            param_data[:, 0:shard_size, :] = loaded_weight[:, shard, :]
        if weight_name.endswith("v1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model])
            param_data[:,
                    shard_size:2 * shard_size, :] = loaded_weight[:,
                                                                    shard, :]
        if weight_name.endswith("w2"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model
                ]).transpose(1, 2)
            param_data[:] = loaded_weight[:, :, shard]


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)

        if self.linear_method and not isinstance(
                self.linear_method, UnquantizedLinearMethod
        ) and self.linear_method.quant_config.support_fused_moe():
            final_hidden_states = self.linear_method.apply_moe_weights(
                self.ws.linear_weights,
                self.w2s.linear_weights,
                self.d_model,
                router_logits,
                self.top_k,
                renormalize=True,
            )
        else:
            final_hidden_states = fused_moe(hidden_states,
                                self.ws,
                                self.w2s,
                                router_logits,
                                self.top_k,
                                renormalize=True,
                                inplace=True)

            if self.tp_size > 1:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length,
                                        hidden_size)


class DbrxAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
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


class DbrxFusedNormAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(config, linear_method)
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


class DbrxBlock(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.norm_attn_norm = DbrxFusedNormAttention(
            config, linear_method)
        self.ffn = DbrxExperts(config, linear_method)

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


class DbrxModel(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.blocks = nn.ModuleList([
            DbrxBlock(config, linear_method)
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


class DbrxForCausalLM(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DbrxModel(config, linear_method)
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

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        if self.linear_method and not isinstance(
                self.linear_method, UnquantizedLinearMethod
        ) and self.linear_method.quant_config.support_fused_moe():
            expert_params_mapping = [
            # (param_name, weight_name, shard_id, expert_id)
            ("ws" if weight_name in ["up_proj", "gate_proj"] else "w2s",
             f"experts.mlp.{expert_id}.{weight_name}", shard_id, expert_id)
            for expert_id in range(self.config.ffn_config.moe_num_experts)
            for weight_name, shard_id in [("up_proj", 0), ("gate_proj", 1), ("down_proj", None)]
            ]

            for name, loaded_weight in hf_model_weights_iterator(
                    model_name_or_path, cache_dir, load_format, revision):
                for (param_name, weight_name, shard_id,
                     expert_id) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if shard_id is None:
                        weight_loader(param,
                                      loaded_weight,
                                      expert_id=expert_id)
                    else:
                        weight_loader(param,
                                      loaded_weight,
                                      shard_id,
                                      expert_id=expert_id)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
        else:
            expert_params_mapping = [
                ("ws" if weight_name in ["w1", "v1"] else "w2s",
                f"experts.mlp.{weight_name}")
                for weight_name in ["w1", "v1", "w2"]
            ]
            print(f'{expert_params_mapping=}')
            print(f'{params_dict.keys()=}')
            for name, loaded_weight in hf_model_weights_iterator(
                    model_name_or_path, cache_dir, load_format, revision):
                print(f'{name=}')
                for param_name, weight_name in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    print(f'{param_name=}, {weight_name=}, {name=}')
                    name = name.replace(weight_name, param_name)
                    print(f'{param_name=}, {weight_name=}, {name=}')

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, weight_name)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
