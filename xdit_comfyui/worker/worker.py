
from xdit_comfyui.config.config import EngineConfig, ParallelConfig


class Worker:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self._init_distributed_enviroment(engine_config.parallel_config)


    def _init_distributed_enviroment(self, parallel_config: ParallelConfig):
        pass

    def load_model(self):
        pass

    def run_model(self):
        pass