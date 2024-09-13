import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy


from xdit_comfyui.logger import init_logger
from ..base_executor import BaseExecutor
from xdit_comfyui.worker.worker_wrapper import RayWorkerWrapper
from .utils import initialize_ray_cluster
from xdit_comfyui.worker.worker import Worker

logger = init_logger(__name__)

class RayGPUExecutor(BaseExecutor):
    def _init_executor(self,):
        self._init_ray_workers()

    def _init_ray_workers(self):
        initialize_ray_cluster(self.engine_config.parallel_config)
        self.workers = []
        self.node_metadata = {}
        
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

        node_and_gpu_ids = self.execute_method("get_node_and_gpu_ids")
        for rank, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
            print(f"{rank=}, {gpu_ids=}, {node_id=}")


            


    def execute_method(self, method: str, *args, **kwargs):
        workers_output = []
        for worker in self.workers:
            executing_method = getattr(worker, method, None) or getattr(worker.worker, method, None)
            if executing_method is None:
                raise ValueError(f"Method {method} not found in worker")
            workers_output.append(executing_method.remote(*args, **kwargs))

        return ray.get(workers_output)

        

