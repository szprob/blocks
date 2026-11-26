"""跨平台变声客户端：麦克风采集 → WebSocket → 变声 → 播放/虚拟声卡。

不绑定 Win/Mac：
- 音频设备由 sounddevice 枚举，用户在界面里选（或命令行指定索引）。
- 输出到虚拟声卡（Windows VB-CABLE / Mac BlackHole）由用户自行配置，
  本程序只负责把变声后的音频写到选中的输出设备。

两种用法：
1) 命令行直连（无界面）：
   python -m voice_changer.client --server ws://127.0.0.1:8765 --mock --list-devices
   python -m voice_changer.client --server ws://127.0.0.1:8765 --mock --in 1 --out 2
2) 图形界面（Tkinter，无需额外依赖）：
   python -m voice_changer.client --gui
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import queue
from typing import Optional

import numpy as np

from .protocol import Config, ConfigAck, MsgType, Stats, error_msg


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


async def voice_loop(
    server: str,
    config: Config,
    in_device: Optional[int],
    out_device: Optional[int],
    on_stats=None,
    on_error=None,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    import sounddevice as sd
    import websockets

    frame_size = config.sample_rate * config.frame_ms // 1000
    q_in: asyncio.Queue = asyncio.Queue(maxsize=8)

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"[client] 音频输入状态: {status}", file=sys.stderr)
        try:
            data = indata[:, 0].astype(np.int16).tobytes()
            q_in.put_nowait(data)
        except queue.Full:
            pass  # 丢帧好过阻塞采集线程

    stream_in = sd.InputStream(
        samplerate=config.sample_rate,
        channels=1,
        dtype="int16",
        device=in_device,
        blocksize=frame_size,
        callback=audio_callback,
    )
    stream_out = sd.OutputStream(
        samplerate=config.sample_rate,
        channels=1,
        dtype="int16",
        device=out_device,
        blocksize=frame_size,
    )

    print(f"[client] 连接 {server} ...", flush=True)
    async with websockets.connect(server, max_size=2**20) as ws:
        await ws.send(config.to_json())
        ack_raw = await ws.recv()
        try:
            ack = ConfigAck.from_json(ack_raw)
        except Exception:
            ack = None
        if ack and not ack.ok:
            msg = f"服务端拒绝配置: {ack.message}"
            if on_error:
                on_error(msg)
            print(f"[client] {msg}", file=sys.stderr)
            return
        print(
            f"[client] 已连接 | 后端={getattr(ack, 'backend', '?')} "
            f"设备={getattr(ack, 'device', '?')} | 开始变声，Ctrl+C 停止",
            flush=True,
        )

        with stream_in, stream_out:
            while stop_event is None or not stop_event.is_set():
                try:
                    pcm = await asyncio.wait_for(q_in.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await ws.send(pcm)
                reply = await ws.recv()
                if isinstance(reply, str):
                    # 文本帧：统计 / 错误 / pong
                    try:
                        parsed = json.loads(reply)
                        mtype = parsed.get("type")
                        if mtype == MsgType.ERROR and on_error:
                            on_error(parsed.get("message", "未知错误"))
                        elif mtype == MsgType.STATS and on_stats:
                            on_stats(Stats.from_json(reply))
                    except Exception:
                        pass
                    continue  # 文本帧不是音频，等下一帧
                stream_out.write(np.frombuffer(reply, dtype="<i2"))


def run_cli(args: argparse.Namespace) -> None:
    if args.list_devices:
        list_devices()
        return

    config = Config(
        transpose=args.transpose,
        index_rate=args.index_rate,
        backend="mock" if args.mock else "auto",
        model=args.model or "",
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
    )
    try:
        asyncio.run(
            voice_loop(args.server, config, args.in_device, args.out_device)
        )
    except KeyboardInterrupt:
        print("\n[client] 已停止", file=sys.stderr)
    except Exception as exc:
        print(f"[client] 错误: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="变声客户端（麦克风 → 服务 → 虚拟声卡）")
    parser.add_argument("--server", default="ws://127.0.0.1:8765", help="服务端 WebSocket 地址")
    parser.add_argument("--gui", action="store_true", help="启动 Tkinter 图形界面")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    parser.add_argument("--in", dest="in_device", type=int, default=None, help="输入设备索引")
    parser.add_argument("--out", dest="out_device", type=int, default=None, help="输出设备索引")
    parser.add_argument("--mock", action="store_true", help="用 mock 后端（无需模型/GPU）")
    parser.add_argument("--model", default="", help="RVC 模型文件名")
    parser.add_argument("--transpose", type=int, default=12, help="变调半音数（默认 12）")
    parser.add_argument("--index-rate", type=float, default=0.7, help="检索特征占比（默认 0.7）")
    parser.add_argument("--sample-rate", type=int, default=16000, help="采样率（默认 16000）")
    parser.add_argument("--frame-ms", type=int, default=20, help="帧长毫秒（默认 20）")
    args = parser.parse_args()

    if args.gui:
        from .gui import run_gui

        run_gui()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
