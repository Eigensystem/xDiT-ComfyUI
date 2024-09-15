from dataclasses import dataclass
from itertools import islice, repeat
from typing import Any, Dict, List, Optional, Tuple
import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy

from xdit_comfyui.logger import init_logger
from xdit_comfyui.worker.worker_wrapper import RayWorkerWrapper
from ..base_executor import BaseExecutor
from .utils import initialize_ray_cluster, get_distributed_init_method, get_ip, get_open_port

logger = init_logger(__name__)

class NodeMetadata:
    def __init__(self, bundle_ids: List[int], gpu_ids: List[List[int]]):
        self.bundle_ids = bundle_ids
        self.gpu_ids = gpu_ids

class RayGPUExecutor(BaseExecutor):
    def _init_executor(self,):
        self._init_ray_workers()

    def _init_ray_workers(self):
        initialize_ray_cluster(self.engine_config.parallel_config)
        self.workers = []
        
        placement_group = self.engine_config.parallel_config.placement_group

        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy,
            )(RayWorkerWrapper).remote("xdit_comfyui.worker.worker", "Worker")

            self.workers.append(worker)


        self.node_metadata: Dict[str, NodeMetadata] = {}

        node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids")
        for bundle_id, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
            metadata = self.node_metadata.get(node_id, None)
            if metadata is None:
                self.node_metadata[node_id] = NodeMetadata(
                    bundle_ids=[bundle_id],
                    gpu_ids=[gpu_ids],
                )
            else:
                metadata.bundle_ids.append(bundle_id)
                metadata.gpu_ids.append(gpu_ids)
                self.node_metadata[node_id] = metadata

        all_kwargs = self._get_worker_env_args()
        self._run_workers("init_worker", all_kwargs=all_kwargs)


    def _get_worker_env_args(self):

        master_addr = get_ip()
        master_port = get_open_port()
        if len(self.node_metadata) == 1:
            master_addr = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(master_addr, master_port)

        num_workers = len(self.workers)
        all_kwargs = [None] * num_workers
        for _, metadata in self.node_metadata.items():
            for local_id, bundle_id in enumerate(metadata.bundle_ids):
                all_kwargs[bundle_id] = {
                    "engine_config": self.engine_config,
                    "rank": bundle_id,
                    "local_rank": local_id,
                    "distributed_init_method": distributed_init_method,
                }
        return all_kwargs


    def _run_workers(
        self,
        method: str,
        *args,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        num_workers = len(self.workers)
        all_worker_args = repeat(args, num_workers) if all_args is None \
            else all_args 
        all_worker_kwargs = repeat(kwargs, num_workers) if all_kwargs is None \
            else all_kwargs

        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(self.workers, all_worker_args, all_worker_kwargs)
        ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs
