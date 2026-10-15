from mmengine.config import read_base

with read_base():
    from .fakecontextbench_gen import fakecontextbench_datasets  # noqa: F401
