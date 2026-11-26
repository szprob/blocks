"""Tkinter 图形界面（标准库，无额外依赖）。

界面元素：
- 服务端地址
- 麦克风 / 输出设备下拉框（输出选虚拟声卡）
- 后端：mock / rvc，模型名（rvc 时填）
- 变调滑块、Index Rate 滑块
- 开始 / 停止，状态与延迟显示

音频循环跑在独立 asyncio 线程里，避免卡住界面。
"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from tkinter import ttk

from .protocol import Config, Stats


def _device_names() -> tuple[list[str], dict[str, int], dict[str, int]]:
    """返回 (显示名列表, 显示名→输入索引, 显示名→输出索引)。"""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        return ["(无法枚举设备)"], {}, {}

    ins: dict[str, int] = {}
    outs: dict[str, int] = {}
    names: list[str] = []
    for idx, dev in enumerate(devices):
        name = f"{idx}: {dev['name']}"
        names.append(name)
        if dev.get("max_input_channels", 0) > 0:
            ins[name] = idx
        if dev.get("max_output_channels", 0) > 0:
            outs[name] = idx
    return names, ins, outs


def run_gui() -> None:
    root = tk.Tk()
    root.title("变声客户端 · 男声 → 温柔女声")
    root.geometry("560x420")

    names, ins, outs = _device_names()
    in_names = [n for n in names if n in ins] or ["(默认)"]
    out_names = [n for n in names if n in outs] or ["(默认)"]

    server_var = tk.StringVar(value="ws://127.0.0.1:8765")
    in_var = tk.StringVar(value=in_names[0])
    out_var = tk.StringVar(value=out_names[0])
    backend_var = tk.StringVar(value="mock")
    model_var = tk.StringVar(value="")
    transpose_var = tk.IntVar(value=12)
    index_var = tk.DoubleVar(value=0.7)
    status_var = tk.StringVar(value="未连接")
    stats_var = tk.StringVar(value="")

    row = 0
    ttk.Label(root, text="服务端地址:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(root, textvariable=server_var, width=40).grid(row=row, column=1, columnspan=2, sticky="w")
    row += 1

    ttk.Label(root, text="麦克风输入:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    ttk.Combobox(root, textvariable=in_var, values=in_names, width=40).grid(row=row, column=1, columnspan=2, sticky="w")
    row += 1

    ttk.Label(root, text="输出设备(选虚拟声卡):").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    ttk.Combobox(root, textvariable=out_var, values=out_names, width=40).grid(row=row, column=1, columnspan=2, sticky="w")
    row += 1

    ttk.Label(root, text="后端:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    ttk.Combobox(root, textvariable=backend_var, values=["mock", "rvc"], width=10, state="readonly").grid(row=row, column=1, sticky="w")
    ttk.Label(root, text="模型名(rvc 时):").grid(row=row, column=2, sticky="w")
    ttk.Entry(root, textvariable=model_var, width=20).grid(row=row, column=3, sticky="w")
    row += 1

    ttk.Label(root, text="变调(半音):").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    ttk.Scale(root, from_=0, to=24, orient="horizontal", variable=transpose_var, length=180).grid(row=row, column=1, sticky="w")
    ttk.Label(root, textvariable=transpose_var).grid(row=row, column=2, sticky="w")
    row += 1

    ttk.Label(root, text="Index Rate:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", variable=index_var, length=180).grid(row=row, column=1, sticky="w")
    ttk.Label(root, textvariable=index_var).grid(row=row, column=2, sticky="w")
    row += 1

    status_lbl = ttk.Label(root, textvariable=status_var, foreground="blue")
    status_lbl.grid(row=row, column=0, columnspan=4, sticky="w", padx=8, pady=6)
    row += 1
    ttk.Label(root, textvariable=stats_var).grid(row=row, column=0, columnspan=4, sticky="w", padx=8)
    row += 1

    # 运行控制
    loop: dict = {"asyncio_loop": None, "thread": None, "stop": None}

    def on_stats(s: Stats):
        stats_var.set(f"已处理 {s.frames} 帧 · 平均推理 {s.avg_ms:.1f} ms")

    def on_error(msg: str):
        status_var.set(f"错误: {msg}")
        status_lbl.configure(foreground="red")

    def worker(cfg: Config, in_idx, out_idx, stop_evt):
        new_loop = asyncio.new_event_loop()
        loop["asyncio_loop"] = new_loop
        asyncio.set_event_loop(new_loop)
        from .client import voice_loop

        try:
            new_loop.run_until_complete(
                voice_loop(server_var.get(), cfg, in_idx, out_idx, on_stats, on_error, stop_evt)
            )
        except Exception as exc:
            root.after(0, lambda: on_error(str(exc)))
        finally:
            root.after(0, lambda: (status_var.set("已停止"), status_lbl.configure(foreground="blue")))
            new_loop.close()

    def start():
        if loop["thread"] and loop["thread"].is_alive():
            return
        backend = backend_var.get()
        cfg = Config(
            transpose=int(transpose_var.get()),
            index_rate=float(index_var.get()),
            backend=backend,
            model=model_var.get().strip() if backend == "rvc" else "",
        )
        in_idx = ins.get(in_var.get())
        out_idx = outs.get(out_var.get())
        stop_evt = asyncio.Event()
        loop["stop"] = stop_evt
        status_var.set("连接中…")
        status_lbl.configure(foreground="blue")
        t = threading.Thread(target=worker, args=(cfg, in_idx, out_idx, stop_evt), daemon=True)
        loop["thread"] = t
        t.start()

    def stop():
        if loop["stop"]:
            # asyncio.Event 必须在它所属的 loop 里 set
            async def _set():
                loop["stop"].set()

            if loop["asyncio_loop"]:
                loop["asyncio_loop"].call_soon_threadsafe(lambda: asyncio.create_task(_set()))

    ttk.Button(root, text="开始变声", command=start).grid(row=row, column=0, columnspan=2, padx=8, pady=10)
    ttk.Button(root, text="停止", command=stop).grid(row=row, column=2, columnspan=2, padx=8, pady=10)

    root.protocol("WM_DELETE_WINDOW", lambda: (stop(), root.destroy()))
    root.mainloop()
