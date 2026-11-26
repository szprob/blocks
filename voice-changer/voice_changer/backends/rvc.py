"""RVC 后端：加载 RVC 模型做真正的神经网络音色转换。

跨平台设计：
- 本模块不 import 任何 RVC 代码，直到 load() 被调用。
- RVC 推理核心通过一个「适配器（adapter）」接入。默认提供一个基于
  标准 RVC 仓库 API 的适配器骨架；使用者可以设置环境变量
  RVC_ADAPTER=<python模块路径> 提供自己的适配器，从而适配任何 RVC 实现
  （w-okada、ailia、自建封装等）而不改本项目代码。

适配器契约（模块级函数）：
    create_adapter() -> 对象，需实现：
        load(model_path: str, index_path: str | None, device: str) -> None
        infer(audio: np.ndarray, sample_rate: int, f0_up_key: int,
              index_rate: float, f0_method: str) -> np.ndarray
        device_name() -> str
        close() -> None

设备选择：优先 CUDA，其次 MPS（Apple Silicon），最后 CPU。
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from ..protocol import Config
from .base import VCBackend


def pick_device(preference: str = "auto") -> tuple[str, str]:
    """返回 (torch设备字符串, 后端名)。preference: auto/cuda/mps/cpu。"""
    try:
        import torch
    except ImportError:
        if preference == "auto":
            return "cpu", "cpu (torch 未安装)"
        raise RuntimeError(f"指定 {preference} 但需要 torch，当前未安装")

    has_cuda = torch.cuda.is_available()
    has_mps = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    if preference == "cuda":
        if not has_cuda:
            raise RuntimeError("指定 cuda 但当前环境不可用（没有 NVIDIA GPU 或 torch 无 CUDA 支持）")
        return "cuda:0", "cuda:0"
    if preference == "mps":
        if not has_mps:
            raise RuntimeError("指定 mps 但当前环境不可用（非 Apple Silicon 或 torch 无 MPS 支持）")
        return "mps", "mps"
    if preference == "cpu":
        return "cpu", "cpu"

    # auto
    if has_cuda:
        return "cuda:0", "cuda:0"
    if has_mps:
        return "mps", "mps"
    return "cpu", "cpu"


class RVCBackend(VCBackend):
    name = "rvc"

    def __init__(self) -> None:
        super().__init__()
        self._adapter: Any | None = None
        self._device = "cpu"
        self._device_label = "cpu"

    def _load_adapter(self) -> Any:
        """按 RVC_ADAPTER 环境变量加载适配器；未设置时用内置骨架。"""
        adapter_path = os.environ.get("RVC_ADAPTER", "").strip()
        if adapter_path:
            module = importlib.import_module(adapter_path)
        else:
            module = importlib.import_module("voice_changer.backends.rvc_builtin")
        if not hasattr(module, "create_adapter"):
            raise RuntimeError(
                f"适配器模块 {adapter_path or 'rvc_builtin'} 缺少 create_adapter() 函数"
            )
        return module.create_adapter()

    def load(self, config: Config, models_dir: str) -> None:
        self._config = config
        device, label = pick_device(config.backend)
        self._device = device
        self._device_label = label

        if not config.model:
            raise ValueError("rvc 后端必须提供 model（模型文件名或路径）")

        model_path = config.model
        if not os.path.isabs(model_path):
            model_path = os.path.join(models_dir, config.model)

        index_path: str | None = None
        index_candidate = os.path.splitext(model_path)[0] + ".index"
        if os.path.exists(index_candidate):
            index_path = index_candidate

        self._adapter = self._load_adapter()
        self._adapter.load(model_path, index_path, self._device)

    def infer(self, audio: np.ndarray) -> np.ndarray:
        if self._adapter is None:
            raise RuntimeError("rvc 后端尚未 load()")
        cfg = self.config
        audio = np.asarray(audio, dtype=np.float32)
        out = self._adapter.infer(
            audio,
            sample_rate=cfg.sample_rate,
            f0_up_key=cfg.transpose,
            index_rate=cfg.index_rate,
            f0_method=cfg.f0_method,
        )
        return np.asarray(out, dtype=np.float32)

    def device_name(self) -> str:
        return self._device_label

    def close(self) -> None:
        if self._adapter is not None and hasattr(self._adapter, "close"):
            self._adapter.close()
        self._adapter = None
