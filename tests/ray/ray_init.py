import pytest

from xdit_comfyui.executor.ray_executors.gpu_executor import RayGPUExecutor
from xdit_comfyui.config import xFuserEngineArgs


# def test_ray_gpu_executor_init():
engine_config = xFuserEngineArgs(
    model="test", 
    ulysses_degree=4, 
    ring_degree=2
).create_config()
RayGPUExecutor(engine_config=engine_config)
