from typing import Optional, List, Tuple, TYPE_CHECKING
import pickle
from vllm.config import ParallelConfig, ModelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip, set_cuda_visible_devices, get_ip

logger = init_logger(__name__)

try:
    import ray

    class RayWorkerVllm:
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, init_cached_hf_modules=False) -> None:
            if init_cached_hf_modules:
                from transformers.dynamic_module_utils import init_hf_modules
                init_hf_modules()
            self.worker = None
            # Since the compiled DAG runs a main execution
            # in a different thread that calls cuda.set_device.
            # The flag indicates is set_device is called on
            # that thread.
            self.compiled_dag_cuda_device_set = False

        def init_worker(self, worker_init_fn):
            self.worker = worker_init_fn()

        def __getattr__(self, name):
            return getattr(self.worker, name)

        def execute_method(self, method, *args, **kwargs):
            executor = getattr(self, method)
            return executor(*args, **kwargs)

        def get_node_ip(self) -> str:
            return get_ip()

        def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
            node_id = ray.get_runtime_context().get_node_id()
            gpu_ids = ray.get_gpu_ids()
            return node_id, gpu_ids

        def set_cuda_visible_devices(self, device_ids) -> None:
            set_cuda_visible_devices(device_ids)

        def execute_model_compiled_dag_remote(self, ignored):
            """Used only when compiled DAG is enabled."""
            import torch
            if not self.compiled_dag_cuda_device_set:
                torch.cuda.set_device(self.worker.device)
                self.compiled_dag_cuda_device_set = True

            output = self.worker.execute_model()
            output = pickle.dumps(output)
            return output

except ImportError as e:
    logger.warning(f"Failed to import Ray with {e!r}. "
                   "For distributed inference, please install Ray with "
                   "`pip install ray`.")
    ray = None
    RayWorkerVllm = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
) -> Optional["PlacementGroup"]:
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address."""

    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    # Connect to a ray cluster.
    if is_hip():
        ray.init(address=ray_address,
                 ignore_reinit_error=True,
                 num_gpus=parallel_config.world_size)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True)
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    if parallel_config.placement_group:
        # Placement group is already set.
        return parallel_config.placement_group

    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get("GPU", 0)
            if bundle_gpus > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 GPU.")
            if bundle_gpus:
                gpu_bundles += 1
        if parallel_config.world_size > gpu_bundles:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the placement group.")
    else:
        num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
        if parallel_config.world_size > num_gpus_in_cluster:
            raise ValueError(
                "The number of required GPUs exceeds the total number of "
                "available GPUs in the cluster.")
        # Create a new placement group
        placement_group_specs = ([{"GPU": 1}] * parallel_config.world_size)
        current_placement_group = ray.util.placement_group(
            placement_group_specs)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        ray.get(current_placement_group.ready(), timeout=1800)

    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group
    return current_placement_group

def initialize_placement_disagg(
    parallel_config: ParallelConfig,
    model_config: ModelConfig,
    ray_address: Optional[str] = None,
) -> Optional["PlacementGroup"]:
        """
        Create placement groups for all engines and all workers
        
        Currently we force the same layer of the context & decoding stage to be executed
        on the same node (we call this "aligned"). This simplifies k/v cache migration.
        """
        prefill_tp = parallel_config.prefill_tp
        decode_tp = parallel_config.decode_tp
        prefill_pp = 1
        decode_pp = 1
        
        # Each placement group is responsible for `layer_per_placement_group` layers
        # Pipeline parallelism isn't supported yet, so number of layers is the same
        layer_per_placement_group = model_config.get_num_layers(parallel_config)
        
        # Each placement group contains `workers_per_placement_group` workers
        workers_per_placement_group = \
            layer_per_placement_group // layer_per_placement_group * prefill_tp \
            + layer_per_placement_group // layer_per_placement_group * decode_tp
        
        # There should be `num_placement_groups` placement groups in total
        # Without pipeline parallelism, there should only be one placement group
        num_placement_groups = 1
        assert num_placement_groups * workers_per_placement_group == \
            prefill_pp * prefill_tp + decode_pp * decode_tp, \
            f"Expected {workers_per_placement_group}, got {prefill_pp * prefill_tp + decode_pp * decode_tp}"
        
        # Create placement groups
        placement_group = ray.util.placement_group(
            [ { "GPU": 1 }] * workers_per_placement_group,
            strategy="STRICT_PACK",
        )
        ray.get(placement_group.ready(), timeout=1000)

        # No pipeline parallelism means we don't support multiple placement groups by layer
        # placement_groups = []
        # for i in range(num_placement_groups):
        #     placement_group = ray.util.placement_group(
        #         [ { "GPU": 1 }] * workers_per_placement_group,
        #         strategy="STRICT_PACK",
        #     )
        #     ray.get(placement_group.ready(), timeout=1000)
        #     placement_groups.append(placement_group)
        
        parallel_config.placement_group = placement_group
        return placement_group
