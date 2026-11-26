"""端到端集成测试：起真实 WebSocket 服务端，发 PCM，收变声后 PCM。

运行：
    python -m pytest tests -q
"""

import asyncio
import os
import sys

import numpy as np
import pytest
import websockets

from voice_changer.protocol import Config


@pytest.mark.asyncio
async def test_websocket_roundtrip_mock():
    from voice_changer.server import run_server

    port = 18765
    server_task = asyncio.create_task(run_server("127.0.0.1", port, "models"))
    await asyncio.sleep(0.4)

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            cfg = Config(transpose=12, backend="mock")
            await ws.send(cfg.to_json())
            ack = await ws.recv()
            assert '"ok": true' in ack or '"ok":true' in ack

            pcm = (np.sin(np.linspace(0, 0.2 * 2 * np.pi, 320)) * 8000).astype("<i2")
            await ws.send(pcm.tobytes())
            reply = await ws.recv()
            assert isinstance(reply, bytes)
            out = np.frombuffer(reply, dtype="<i2")
            assert len(out) == len(pcm)
            assert not np.array_equal(out, pcm)
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
