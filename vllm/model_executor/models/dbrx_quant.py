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
from vllm.model_executor.models.dbrx import DbrxRouter, DbrxFusedNormAttention

KVCache = Tuple[torch.Tensor, torch.Tensor]


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

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.linear_method = linear_method

        self.router = DbrxRouter(config, self.params_dtype)

        assert self.linear_method and not isinstance(
                self.linear_method, UnquantizedLinearMethod
        ) and self.linear_method.quant_config.support_fused_moe()
        self.intermediate_size = config.ffn_config.ffn_hidden_size

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

        assert self.linear_method and not isinstance(
                self.linear_method, UnquantizedLinearMethod
        ) and self.linear_method.quant_config.support_fused_moe()

        final_hidden_states = self.linear_method.apply_moe_weights(
            self.ws.linear_weights,
            self.w2s.linear_weights,
            hidden_states,
            router_logits,
            self.top_k,
            renormalize=True,
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length,
                                        hidden_size)

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

        assert self.linear_method and not isinstance(
                self.linear_method, UnquantizedLinearMethod
        ) and self.linear_method.quant_config.support_fused_moe()
        expert_params_mapping = [
            ("ws" if weight_name in ["w1", "v1"] else "w2s",
            f"experts.mlp.{weight_name}", shard_id)
            for weight_name, shard_id in [("w1", 0), ("v1", 1), ("w2", None)]
            ]
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False):
            for (param_name, weight_name, shard_id) in expert_params_mapping:
                if weight_name not in name:
                    continue
                original_name = name
                expert_id = int(name.split(".")[-2].split("_")[1])
                name = name.replace(weight_name + f"_{expert_id}", param_name)
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("ffn.experts.mlp." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
