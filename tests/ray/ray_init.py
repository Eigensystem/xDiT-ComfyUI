from xdit_comfyui.executor.ray_executors.gpu_executor import RayGPUExecutor
from xdit_comfyui.config import xFuserEngineArgs

engine_config = xFuserEngineArgs(model="test", ulysses_degree=4, ring_degree=4).create_config()
RayGPUExecutor(engine_config=engine_config)