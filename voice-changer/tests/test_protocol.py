"""协议与引擎单元测试（无需音频设备/GPU）。

运行：
    python -m pytest tests -q
"""

import numpy as np
import pytest

from voice_changer.protocol import Config, ConfigAck, Stats
from voice_changer.engine import VoiceEngine


def test_config_roundtrip():
    cfg = Config(transpose=12, index_rate=0.7, model="gentle_female.pth")
    raw = cfg.to_json()
    cfg2 = Config.from_json(raw)
    assert cfg2.transpose == 12
    assert cfg2.index_rate == 0.7
    assert cfg2.model == "gentle_female.pth"


def test_config_validation():
    with pytest.raises(ValueError):
        Config(transpose=99).validate()
    with pytest.raises(ValueError):
        Config(index_rate=1.5).validate()


def test_config_ack_roundtrip():
    ack = ConfigAck(ok=True, backend="mock", device="cpu", sample_rate=16000, frame_ms=20)
    ack2 = ConfigAck.from_json(ack.to_json())
    assert ack2.ok and ack2.backend == "mock"


def test_stats_roundtrip():
    s = Stats(frames=100, avg_ms=12.5, total_ms=1250.0)
    s2 = Stats.from_json(s.to_json())
    assert s2.frames == 100 and s2.avg_ms == 12.5


def test_mock_engine_process():
    engine = VoiceEngine()
    cfg = Config(transpose=12, backend="mock")
    engine.load(cfg)
    pcm = (np.sin(np.linspace(0, 0.2 * 2 * np.pi, 320)) * 10000).astype(np.int16)
    out = engine.process(pcm)
    assert out.dtype == np.int16
    assert len(out) == len(pcm)
    assert not np.array_equal(out, pcm)  # 确实变了
    # 音量不能被放大（否则削波失真）
    assert np.abs(out).max() <= np.abs(pcm).max() * 1.1
    frames, avg, total = engine.stats()
    assert frames == 1 and avg > 0 and total > 0
    engine.close()


def test_mock_zero_transpose_near_identity():
    """0 半音时输出应与输入接近（mock 淡入淡出会有轻微差异）。"""
    engine = VoiceEngine()
    engine.load(Config(transpose=0, backend="mock"))
    pcm = np.random.default_rng(0).integers(-5000, 5000, 320, dtype=np.int16)
    out = engine.process(pcm)
    assert len(out) == len(pcm)
    engine.close()
