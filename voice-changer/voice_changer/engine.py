"""变声引擎：把后端封装成线程安全、带统计的推理器。

职责：
- 根据配置选择并加载后端（mock / rvc）。
- int16 PCM ⇄ float32 转换。
- 记录逐帧耗时，供服务端回报统计。
- 热更新参数（变调、index_rate 等）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .backends import build_backend
from .backends.base import VCBackend
from .protocol import Config


@dataclass
class EngineStats:
    frames: int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.frames if self.frames else 0.0


class VoiceEngine:
    """线程安全的变声引擎。"""

    def __init__(self) -> None:
        self._backend: VCBackend | None = None
        self._backend_name = ""
        self._lock = threading.RLock()
        self._stats = EngineStats()

    def load(self, config: Config, models_dir: str = "models") -> str:
        """加载后端。返回实际设备名。"""
        config.validate()
        with self._lock:
            name = "mock" if config.backend == "auto" and not config.model else config.backend
            if name == "auto":
                name = "rvc"  # 有 model 时 auto → rvc
            self._backend = build_backend(name)
            self._backend_name = name
            self._backend.load(config, models_dir)
            self._stats = EngineStats()
            return self._backend.device_name()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def device(self) -> str:
        if self._backend is None:
            return "未加载"
        return self._backend.device_name()

    def stats(self) -> tuple[int, float, float]:
        with self._lock:
            return self._stats.frames, self._stats.avg_ms, self._stats.total_ms

    def update_params(self, **kwargs) -> None:
        with self._lock:
            if self._backend is None:
                raise RuntimeError("引擎尚未 load()")
            self._backend.update_params(**kwargs)

    def process(self, pcm_int16: np.ndarray) -> np.ndarray:
        """输入 int16 PCM，输出等长 int16 PCM。"""
        if self._backend is None:
            raise RuntimeError("引擎尚未 load()")
        pcm = np.asarray(pcm_int16, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        start = time.perf_counter()
        out = self._backend.infer(audio)
        elapsed = (time.perf_counter() - start) * 1000.0
        out = np.asarray(out, dtype=np.float32)
        if len(out) != len(pcm):
            # 后端_len 不匹配时按长度对齐，避免客户端播放缓冲错乱
            if len(out) < len(pcm):
                out = np.pad(out, (0, len(pcm) - len(out)))
            else:
                out = out[: len(pcm)]
        result = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
        with self._lock:
            self._stats.frames += 1
            self._stats.total_ms += elapsed
        return result

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
