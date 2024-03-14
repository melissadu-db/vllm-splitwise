# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and Databricks team.
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
""" Databricks configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional, Any


logger = logging.get_logger(__name__)

DATABRICKS_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DatabricksAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DatabricksAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        clip_qkv (`float`, *optional*, defualts to None):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        kv_n_heads (Optional[int]): For grouped_query_attention only, allow user to specify number of kv heads.
        rope_theta (float): The base frequency for rope.
    """

    def __init__(
        self,
        attn_pdrop: float = 0,
        clip_qkv: Optional[float] = None,
        kv_n_heads: int = 8,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads=kv_n_heads
        self.rope_theta=rope_theta

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "databricks":
            config_dict = config_dict["attn_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class DatabricksFFNConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DatabricksFFN`] class. It is used to instantiate
    feedforward layers according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:

    """

    def __init__(
        self,
        ffn_act_fn: str = 'silu',
        ffn_hidden_size: int = 10752,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        moe_jitter_eps: Optional[float] = None,
        moe_loss_weight: float = 0.05,
        moe_normalize_expert_weights: Optional[float] = 1,
        uniform_expert_assignment: bool = False,
    ):
        super().__init__()
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "databricks":
            config_dict = config_dict["ffn_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class DatabricksConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DatabricksModel`]. It is used to instantiate a Databricks model
    according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 6144):
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        max_seq_len (`int`, *optional*, defaults to 32768):
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Databricks model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DatabricksModel`].
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        attn_config (`dict`, *optional*):
            A dictionary used to configure the model's attention module.
        ffn_config (`dict`, *optional*):
            A dictionary used to configure the model's FFN module.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*): # TODO
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1): # TODO
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2): # TODO
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Example:

    ```python
    >>> from transformers import DatabricksConfig, DatabricksModel

    >>> # Initializing a Databricks configuration
    >>> configuration = DatabricksConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DatabricksModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "databricks"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "max_position_embeddings": "max_seq_len"
    }

    def __init__(
        self,
        d_model: int = 6144,
        n_heads: int = 48,
        n_layers: int = 40,
        max_seq_len: int = 32768,
        vocab_size: int = 100352,
        attn_config: Optional[DatabricksAttentionConfig] = None,
        ffn_config: Optional[DatabricksFFNConfig] = None,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ):
        if attn_config is None:
            self.attn_config = DatabricksAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = DatabricksAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        if ffn_config is None:
            self.ffn_config = DatabricksFFNConfig()
        elif isinstance(ffn_config, dict):
            self.ffn_config = DatabricksFFNConfig(**ffn_config)
        else:
            self.ffn_config = ffn_config

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.kv_n_heads = self.attn_config.kv_n_heads
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
