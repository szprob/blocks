"""推理后端抽象层。

所有后端都遵守同一接口：输入一段 float32 音频（[-1, 1]），输出等长的
float32 音频。这样服务端完全不关心底层是 mock DSP、RVC、还是别的模型。

新增后端只需继承 VCBackend 并实现 load / infer / device_name。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..protocol import Config


class VCBackend(ABC):
    """语音转换后端的统一接口。"""

    name = "base"

    def __init__(self) -> None:
        self._config: Config | None = None

    @property
    def config(self) -> Config:
        if self._config is None:
            raise RuntimeError("后端尚未 load()")
        return self._config

    @abstractmethod
    def load(self, config: Config, models_dir: str) -> None:
        """加载模型/初始化资源。models_dir 是服务端模型根目录。"""

    @abstractmethod
    def infer(self, audio: np.ndarray) -> np.ndarray:
        """输入 float32 [-1,1] 单声道，输出等长 float32 [-1,1] 单声道。"""

    @abstractmethod
    def device_name(self) -> str:
        """返回实际使用的计算设备描述，如 'cuda:0' / 'mps' / 'cpu'。"""

    def close(self) -> None:
        """释放资源；默认无操作。"""

    def update_params(self, **kwargs: Any) -> None:
        """热更新参数；默认直接改 config 字段。子类可覆盖。"""
        if self._config is None:
            return
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        self._config.validate()
