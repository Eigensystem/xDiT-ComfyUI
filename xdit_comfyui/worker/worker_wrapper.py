import ray
import importlib
from typing import Any, List, Tuple

class WorkerWrapper:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self.worker = None

    #lazy import 
    def init_worker(self, *args, **kwargs):
        module = importlib.import_module(self.module_name) 
        worker_class = getattr(module, self.class_name)
        self.worker = worker_class(*args, **kwargs)

    def init_device(self,):
        pass

    def execute_method(self, method: str, *args, **kwargs) -> Any:
        method = getattr(self, method, None) or getattr(self.worker, method, None)
        if not method:
            raise(AttributeError(f"Method {method} not found in Worker class"))
        return method(*args, **kwargs)

class RayWorkerWrapper(WorkerWrapper):
    def get_node_and_gpu_ids(self,) -> Tuple[str, List[int]]:
        gpu_ids = ray.get_gpu_ids()
        node_id = ray.get_runtime_context().get_node_id()
        return node_id, gpu_ids
