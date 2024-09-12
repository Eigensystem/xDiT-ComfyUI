import ray

from xdit_comfyui.logger import init_logger
from ..base_executor import BaseExecutor
from .utils import initialize_ray_cluster

logger = init_logger(__name__)

class RayGPUExecutor(BaseExecutor):
    def _init_executor(self,):
        self._init_ray_workers()

    def _init_ray_workers(self):
        initialize_ray_cluster(self.engine_config.parallel_config)
        

        
        
