"""voice-changer: 跨平台实时变声（男声 → 温柔女声）。

设计原则：
- 代码本身不绑定任何操作系统；音频 I/O 用 sounddevice（PortAudio），
  推理后端按环境自动探测（CUDA → MPS → CPU），虚拟声卡由使用者自行配置。
- 服务端 / 客户端通过 WebSocket 传 PCM 帧，二者可同机或分机运行。

可用模式：
- mock  : 不加载任何模型，用 DSP（变调 + 共振峰）演示整条链路是否打通。
         用于没有 RVC 模型、没有 GPU、或只想验证端到端音频链路时。
- rvc   : 加载 RVC 模型做真正的神经网络音色转换（需要自备 .pth/.index）。
"""

__version__ = "0.1.0"
