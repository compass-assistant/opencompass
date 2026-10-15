from mmengine.config import read_base

with read_base():
    from .over_editing_gen import over_editing_datasets  # noqa: F401
