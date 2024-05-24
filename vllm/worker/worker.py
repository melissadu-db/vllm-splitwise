"""A GPU worker class."""
import gc
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized, get_stage_parallel_group)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import WorkerType
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker_base import WorkerBase


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        mscclpp_init_method: str = None,
        worker_type: WorkerType = WorkerType.MIXED,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.mscclpp_init_method = mscclpp_init_method
        self.worker_type = worker_type
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        ModelRunnerClass = (EmbeddingModelRunner if
                            self.model_config.embedding_mode else ModelRunner)
        self.model_runner = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[torch.tensor]] = None

        self.kvcache_comm_manager = None

    def is_prompt_worker(self) -> bool:
        return self.worker_type == WorkerType.PROMPT

    def is_token_worker(self) -> bool:
        return self.worker_type == WorkerType.TOKEN

    def is_mixed_worker(self) -> bool:
        return self.worker_type == WorkerType.MIXED

    def init_model(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method)
        self.init_kvcache_comm(self.mscclpp_init_method)
        if not self.parallel_config.disable_custom_all_reduce:
            init_custom_ar()

        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()
        if self.parallel_config.sep_prompt_token:
            # Populate Sampler with dst_rank as driver worker's rank.
            self.model_runner.model.sampler.set_dst_rank(
                self.model_runner.driver_rank)

    def init_kvcache_comm(self,
                          mscclpp_init_method: Optional[str] = None) -> None:
        if mscclpp_init_method is not None:
            from vllm.worker.comm_utils import KVCacheCommManager
            self.kvcache_comm_manager = KVCacheCommManager(
                self.rank, self.parallel_config.world_size,
                self.parallel_config.num_prompt_workers, mscclpp_init_method)

            self.worker_type = (WorkerType.PROMPT if self.rank <
                                self.parallel_config.num_prompt_workers else
                                WorkerType.TOKEN)

            # Set the driver worker rank for prompt and token workers.
            self.model_runner.driver_rank = (
                self.rank // self.parallel_config.num_prompt_workers
            ) * self.parallel_config.num_prompt_workers
            if self.rank == self.model_runner.driver_rank:
                self.is_driver_worker = True
                self.model_runner.is_driver_worker = True

    def setup_kvcache_comm(self) -> None:
        # Setup the communication for the KV cache.
        if self.kvcache_comm_manager is not None:
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            self.kvcache_comm_manager.setup_comm(num_layers, self.gpu_cache)

            # Populate the attention modules with the KV cache communicator.
            self.set_comm_for_attention_modules()

    def destroy_kvcache_comm(self) -> None:
        if self.kvcache_comm_manager is not None:
            self.kvcache_comm_manager.destroy_comm()
            self.unset_comm_for_attention_modules()

    def set_comm_for_attention_modules(self) -> None:
        attention_modules = list(
            filter(
                lambda module: "PagedAttention" in module.__class__.__name__,
                self.model_runner.model.modules()))
        for i, attention_module in enumerate(attention_modules):
            attention_module.set_kvcache_comm_manager(
                self.kvcache_comm_manager)
            attention_module.layer_id = i

    def unset_comm_for_attention_modules(self) -> None:
        attention_modules = list(
            filter(
                lambda module: "PagedAttention" in module.__class__.__name__,
                self.model_runner.model.modules()))
        for attention_module in attention_modules:
            del attention_module.kvcache_comm_manager

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: torch.Tensor,
        blocks_to_swap_out: torch.Tensor,
        blocks_to_copy: torch.Tensor,
    ) -> None:
        # Issue cache operations.
        if blocks_to_swap_in.numel() > 0:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out.numel() > 0:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy.numel() > 0:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        blocks_to_nw: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        is_prompt = False
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            is_prompt = seq_group_metadata_list[0].is_prompt
        if self.is_driver_worker and self.should_execute(is_prompt):
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            assert blocks_to_nw is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
                "blocks_to_nw": blocks_to_nw,
                "is_prompt": is_prompt,
            }
            broadcast_tensor_dict(data,
                                  src=self.model_runner.driver_rank,
                                  group=get_stage_parallel_group())
        else:
            data = broadcast_tensor_dict(src=self.model_runner.driver_rank,
                                         group=get_stage_parallel_group())
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]
            blocks_to_nw = data["blocks_to_nw"]
            is_prompt = data["is_prompt"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        if len(blocks_to_nw) and self.is_token_worker() and not is_prompt:
            for sem_id in blocks_to_nw:
                self.kvcache_comm_manager.wait(sem_id)

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache, blocks_to_nw)

        if len(blocks_to_nw) and self.is_prompt_worker() and is_prompt:
            for sem_id in blocks_to_nw:
                self.kvcache_comm_manager.signal_and_flush(sem_id)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def should_execute(self, is_prompt: bool) -> bool:
        return self.is_mixed_worker() or (
            self.is_prompt_worker() and is_prompt) or (self.is_token_worker()
                                                       and not is_prompt)

    def set_gpucache(self):
        from vllm.worker.comm_utils import HEAD_TYPES
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        for layer_id in range(num_layers):
            for head_type in HEAD_TYPES:
                self.gpu_cache[layer_id][head_type][:] = self.rank * (
                    num_layers *
                    len(HEAD_TYPES)) + layer_id * len(HEAD_TYPES) + head_type
        torch.cuda.synchronize()

    def send_recv_kvcache_all(self):
        if self.kvcache_comm_manager is not None:
            num_gpu_blocks = self.cache_config.num_gpu_blocks
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            if self.rank < self.parallel_config.num_prompt_workers:
                for layer_id in range(num_layers):
                    self.kvcache_comm_manager.put(0, layer_id, 0,
                                                  num_gpu_blocks)
                self.kvcache_comm_manager.signal_and_flush(0)
            else:
                self.kvcache_comm_manager.wait(0)
            torch.cuda.synchronize()

    def check_gpucache(self):
        if self.kvcache_comm_manager is not None:
            from vllm.worker.comm_utils import HEAD_TYPES
            num_prompt_workers = self.parallel_config.num_prompt_workers
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            expected_worker_id = self.rank if self.rank < num_prompt_workers else self.rank - num_prompt_workers
            for layer_id in range(num_layers):
                for head_type in HEAD_TYPES:
                    expected_scalar = expected_worker_id * (num_layers * len(
                        HEAD_TYPES)) + layer_id * len(HEAD_TYPES) + head_type
                    expected_tensor = torch.ones_like(
                        self.gpu_cache[layer_id][head_type]) * expected_scalar
                    assert torch.allclose(self.gpu_cache[layer_id][head_type],
                                          expected_tensor)


def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size,
                                      parallel_config.sep_prompt_token)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
