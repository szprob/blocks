"""推理后端集合。"""

from .base import VCBackend
from .mock import MockBackend
from .rvc import RVCBackend, pick_device

BACKENDS = {
    "mock": MockBackend,
    "rvc": RVCBackend,
}


def build_backend(name: str) -> VCBackend:
    """按名称构造后端实例。未知名称抛 ValueError。"""
    if name not in BACKENDS:
        raise ValueError(f"未知后端 {name!r}，可选: {list(BACKENDS)}")
    return BACKENDS[name]()
