"""WebSocket 变声推理服务端。

协议：
- 客户端连上后先发一条 JSON config 文本帧（见 protocol.Config）。
- 服务端回一条 config_ack。
- 之后客户端持续发二进制 PCM 帧，服务端逐帧回变声后的二进制 PCM 帧。
- 客户端可随时再发 config 文本帧热更新参数；服务端回 config_ack。
- 每 100 帧服务端附带一条 stats 文本帧（统计平均推理耗时）。

启动：
    python -m voice_changer.server --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

import websockets
from websockets.asyncio.server import serve

from .engine import VoiceEngine
from .protocol import (
    Config,
    ConfigAck,
    MsgType,
    Stats,
    error_msg,
    parse_message,
)


async def handle_client(websocket, engine: VoiceEngine, models_dir: str) -> None:
    peer = websocket.remote_address
    print(f"[server] 客户端连接: {peer}", flush=True)
    local_engine = VoiceEngine()  # 每个连接一个独立引擎实例，互不干扰
    try:
        async for message in websocket:
            if isinstance(message, str):
                # ---- 控制消息 ----
                try:
                    data = parse_message(message)
                except Exception as exc:
                    await websocket.send(error_msg(f"控制消息解析失败: {exc}"))
                    continue

                mtype = data.get("type")
                if mtype == MsgType.PING:
                    await websocket.send('{"type": "pong"}')
                    continue

                if mtype == MsgType.CONFIG:
                    try:
                        cfg = Config.from_json(message)
                        device = local_engine.load(cfg, models_dir)
                        ack = ConfigAck(
                            ok=True,
                            backend=local_engine.backend_name,
                            device=device,
                            sample_rate=cfg.sample_rate,
                            frame_ms=cfg.frame_ms,
                        )
                        print(
                            f"[server] {peer} 配置完成: backend={local_engine.backend_name} "
                            f"device={device} model={cfg.model or '(mock)'}",
                            flush=True,
                        )
                    except Exception as exc:
                        ack = ConfigAck(
                            ok=False,
                            backend="",
                            device="",
                            sample_rate=0,
                            frame_ms=0,
                            message=str(exc),
                        )
                        print(f"[server] {peer} 配置失败: {exc}", flush=True)
                    await websocket.send(ack.to_json())
                    continue

                # 未知控制消息：热更新参数（把 type 之外的字段当参数）
                params = {k: v for k, v in data.items() if k != "type"}
                try:
                    local_engine.update_params(**params)
                    await websocket.send(
                        ConfigAck(
                            ok=True,
                            backend=local_engine.backend_name,
                            device=local_engine.device,
                            sample_rate=0,
                            frame_ms=0,
                            message="参数已更新",
                        ).to_json()
                    )
                except Exception as exc:
                    await websocket.send(error_msg(f"参数更新失败: {exc}"))
                continue

            # ---- 二进制音频帧 ----
            try:
                import numpy as np

                pcm = np.frombuffer(message, dtype="<i2")
                out = local_engine.process(pcm)
                await websocket.send(out.tobytes())

                frames, avg, total = local_engine.stats()
                if frames % 100 == 0:
                    await websocket.send(
                        Stats(frames=frames, avg_ms=avg, total_ms=total).to_json()
                    )
            except RuntimeError as exc:
                # 未配置就发音频：提示一次后忽略，避免刷屏
                await websocket.send(error_msg(str(exc)))
                await asyncio.sleep(0.5)
            except Exception as exc:
                await websocket.send(error_msg(f"推理失败: {exc}"))
    except websockets.ConnectionClosed:
        print(f"[server] 客户端断开: {peer}", flush=True)
    finally:
        local_engine.close()


async def run_server(host: str, port: int, models_dir: str) -> None:
    async with serve(
        lambda ws: handle_client(ws, VoiceEngine(), models_dir),
        host,
        port,
        max_size=2**20,
    ) as server:
        print(f"[server] 变声服务已启动: ws://{host}:{port}", flush=True)
        print(f"[server] 模型目录: {models_dir}", flush=True)
        await asyncio.Future()  # 永久运行


def main() -> None:
    parser = argparse.ArgumentParser(description="变声 WebSocket 服务端")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--models-dir", default="models", help="RVC 模型目录")
    args = parser.parse_args()

    try:
        asyncio.run(run_server(args.host, args.port, args.models_dir))
    except KeyboardInterrupt:
        print("\n[server] 已退出", file=sys.stderr)


if __name__ == "__main__":
    main()
