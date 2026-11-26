"""mock 后端：纯 DSP 演示，不依赖任何模型或 GPU。

用途：在没有 RVC 模型 / GPU / 预训练权重时，验证「采集 → 传输 → 变声 → 播放」
整条链路是否打通。它不是神经网络音色转换，音质无法和 RVC 比；
真正的男声→温柔女声请换 rvc 后端。

实现：
- 变调：对整帧做重采样（比例 2^(transpose/12)），再截/补回原长度。
- 共振峰：对频谱做轻微上移，模拟女声更靠上的共振峰。
- 加淡入淡出，减少帧边界的咔哒声。
"""

from __future__ import annotations

import numpy as np

from ..protocol import Config
from .base import VCBackend


class MockBackend(VCBackend):
    name = "mock"

    def load(self, config: Config, models_dir: str) -> None:
        self._config = config

    def device_name(self) -> str:
        return "cpu (dsp mock)"

    def infer(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n == 0:
            return audio

        # 1) 变调：重采样改变基频，再恢复到原长度
        ratio = 2.0 ** (cfg.transpose / 12.0)
        if abs(ratio - 1.0) > 1e-6:
            idx = np.arange(n, dtype=np.float32) / ratio
            idx = np.clip(idx, 0, n - 1)
            shifted = np.interp(idx, np.arange(n), audio).astype(np.float32)
        else:
            shifted = audio.copy()

        # 2) 轻量共振峰上移：把频谱包络往高频搬，模拟女声更靠上的共振峰
        window = np.hanning(n).astype(np.float32)
        spec = np.fft.rfft(shifted * window)
        freqs = np.fft.rfftfreq(n, d=1.0 / cfg.sample_rate)
        shift_hz = max(0, cfg.transpose) * 30.0  # 半音数 × 30Hz，温柔女声约 +360Hz
        # output[f] 取 input[f - shift] → 频谱内容向高频移动
        mag = np.interp(freqs, freqs - shift_hz, np.abs(spec), left=0.0)
        spec_out = mag * np.exp(1j * np.angle(spec))
        out = np.fft.irfft(spec_out, n=n).astype(np.float32)

        # 3) 帧边界淡入淡出，减少咔哒声
        fade = max(1, n // 16)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]

        # 4) 峰值归一化到输入音量，避免频谱搬移带来的放大/削波
        in_peak = float(np.max(np.abs(audio))) if np.max(np.abs(audio)) > 1e-8 else 1.0
        out_peak = float(np.max(np.abs(out))) if np.max(np.abs(out)) > 1e-8 else 1.0
        out = out * (in_peak / out_peak)
        return np.clip(out, -1.0, 1.0).astype(np.float32)
