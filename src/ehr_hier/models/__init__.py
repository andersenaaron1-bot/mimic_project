try:
    from .hier_transformer import HierModel

    __all__ = ["HierModel"]
except ModuleNotFoundError:
    __all__ = []
