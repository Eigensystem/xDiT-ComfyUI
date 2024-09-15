import torch
import torch.distributed as dist
from xdit_comfyui import envs

from xdit_comfyui.config.config import EngineConfig, ParallelConfig


class Worker:
    def __init__(self, engine_config: EngineConfig, rank: int, local_rank: int, distributed_init_method: str = 'env://'):
        self.engine_config = engine_config
        self.rank = rank
        self.local_rank = local_rank

        self._init_distributed_enviroment(engine_config.parallel_config, rank, local_rank, distributed_init_method)

    def _init_distributed_enviroment(self, parallel_config: ParallelConfig, rank: int, local_rank: int, distributed_init_method: str):
        dist.init_process_group(
            backend='nccl',
            init_method=distributed_init_method,
            world_size=parallel_config.world_size,
            rank=rank,
        )
        torch.cuda.set_device("cuda:0")

    def load_model(self):
        pass

    def run_model(self):
        pass
