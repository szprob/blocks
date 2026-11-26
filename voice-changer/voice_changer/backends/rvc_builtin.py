"""内置 RVC 适配器骨架。

它定义了与 RVC 推理核心的标准对接方式。由于 RVC 有多个实现版本
（原版 RVC-WebUI、w-okada/voice-changer、RVC-WebUI-MacOS、ailia 等），
API 各有差异，本项目默认不假定某一种实现，而是：

1. 优先读取环境变量 RVC_ADAPTER_MODULE（模块路径）+ RVC_ADAPTER_CLASS（类名）
   动态加载你自己的适配器，详见 README「接入自己的 RVC 实现」。
2. 未设置时，尝试常见的 RVC 导入路径自动探测；若都不可用，给出清晰报错，
   提示如何提供适配器。

真正上线时推荐做法：写一个约 30 行的适配器文件（示例见 README），
把你的 RVC 实现的 load / infer 包成下面这个契约即可。
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np


class BuiltinAdapter:
    """尽力探测并调用常见 RVC 实现的适配器。

    约定实现：
        load(model_path, index_path, device)
        infer(audio, sample_rate, f0_up_key, index_rate, f0_method) -> np.ndarray
    """

    def __init__(self) -> None:
        self._impl: Any | None = None
        self._device = "cpu"

    def load(self, model_path: str, index_path: str | None, device: str) -> None:
        self._device = device
        self._impl = self._find_impl()
        self._impl.load(model_path, index_path, device)

    def infer(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0_up_key: int,
        index_rate: float,
        f0_method: str,
    ) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("内置适配器尚未 load()")
        return self._impl.infer(
            audio,
            sample_rate=sample_rate,
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            f0_method=f0_method,
        )

    def device_name(self) -> str:
        return self._device

    def close(self) -> None:
        if self._impl and hasattr(self._impl, "close"):
            self._impl.close()
        self._impl = None

    @staticmethod
    def _find_impl() -> Any:
        """按环境变量指定的模块/类，或常见默认路径，探测 RVC 实现。"""
        module_name = os.environ.get("RVC_ADAPTER_MODULE", "").strip()
        class_name = os.environ.get("RVC_ADAPTER_CLASS", "RVCAdapter").strip()

        candidates: list[tuple[str, str]] = []
        if module_name:
            candidates.append((module_name, class_name))
        # 常见第三方封装；有则自动用，没有就在下面给清晰报错
        candidates += [
            ("voice_changer.contrib.rvc_webui_adapter", "RVCWebUIAdapter"),
        ]

        for mod_name, cls_name in candidates:
            try:
                module = importlib.import_module(mod_name)
                cls = getattr(module, cls_name)
                return cls()
            except (ImportError, AttributeError):
                continue

        raise RuntimeError(
            "找不到可用的 RVC 实现适配器。请提供一个适配器模块，实现：\n"
            "  load(model_path, index_path, device)\n"
            "  infer(audio, sample_rate, f0_up_key, index_rate, f0_method)\n"
            "  device_name() / close()\n"
            "然后用环境变量 RVC_ADAPTER=<你的模块路径> 指向它。\n"
            "详见 README「接入自己的 RVC 实现」。"
        )


def create_adapter() -> BuiltinAdapter:
    return BuiltinAdapter()
