"""客户端与服务端共享的协议定义。

设计要点：
- 控制消息走 JSON 文本帧（UTF-8）。
- 音频帧走二进制 WebSocket 帧：固定 20ms、16kHz、单声道、int16。
- 双方必须就 sample_rate / frame_ms 达成一致；握手时服务端可回传确认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 20
# 20ms @ 16kHz = 320 samples = 640 bytes int16
DEFAULT_FRAME_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_MS // 1000


class MsgType:
    CONFIG = "config"          # 客户端 → 服务端：配置模型/参数
    CONFIG_ACK = "config_ack"  # 服务端 → 客户端：确认配置
    STATS = "stats"            # 服务端 → 客户端：周期统计
    ERROR = "error"            # 双向：错误
    PING = "ping"
    PONG = "pong"


@dataclass
class Config:
    """一次变声会话的可调参数。

    transpose: 变调半音数。男声→女声常用 +12（高一个八度）；范围建议 6~14。
    index_rate: RVC 检索特征占比，0~1。越高越像目标音色，过高会有电音/咬字损失。
    f0_method: F0 提取算法，rmvpe 效果最好；mock 模式下忽略。
    model: 模型文件名（位于服务端 models 目录），mock 模式下忽略。
    backend: 期望的推理后端；"auto" 时服务端按硬件自动选。
    """

    transpose: int = 12
    index_rate: float = 0.7
    f0_method: str = "rmvpe"
    model: str = ""
    backend: str = "auto"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_ms: int = DEFAULT_FRAME_MS

    def validate(self) -> None:
        if not -24 <= self.transpose <= 24:
            raise ValueError(f"transpose 超出范围 [-24, 24]: {self.transpose}")
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError(f"index_rate 超出范围 [0, 1]: {self.index_rate}")
        if self.f0_method not in ("rmvpe", "crepe", "dio", "harvest"):
            raise ValueError(f"未知 f0_method: {self.f0_method}")
        if self.sample_rate <= 0 or self.frame_ms <= 0:
            raise ValueError("sample_rate / frame_ms 必须为正数")

    def to_json(self) -> str:
        self.validate()
        data = asdict(self)
        data["type"] = MsgType.CONFIG
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Config":
        data = json.loads(raw)
        data.pop("type", None)
        return cls(**data)


@dataclass
class ConfigAck:
    ok: bool
    backend: str
    device: str
    sample_rate: int
    frame_ms: int
    message: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        data["type"] = MsgType.CONFIG_ACK
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ConfigAck":
        data = json.loads(raw)
        data.pop("type", None)
        return cls(**data)


@dataclass
class Stats:
    frames: int
    avg_ms: float
    total_ms: float

    def to_json(self) -> str:
        data = asdict(self)
        data["type"] = MsgType.STATS
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Stats":
        data = json.loads(raw)
        data.pop("type", None)
        return cls(**data)


def error_msg(message: str) -> str:
    return json.dumps({"type": MsgType.ERROR, "message": message}, ensure_ascii=False)


def parse_message(raw: str) -> Dict[str, Any]:
    """解析 JSON 控制消息，返回原始 dict（含 type）。"""
    data = json.loads(raw)
    if not isinstance(data, dict) or "type" not in data:
        raise ValueError(f"非法控制消息: {raw[:200]}")
    return data
