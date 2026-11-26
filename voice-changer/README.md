# voice-changer · 跨平台实时变声

男声 → 温柔女声的实时变声软件骨架。**代码本身不绑定 Windows 或 macOS**：

- 音频采集/播放用 `sounddevice`（PortAudio，跨平台，wheel 自带驱动）。
- 推理后端按硬件自动探测：CUDA → MPS（Apple Silicon）→ CPU。
- 虚拟声卡由使用者在系统里配置（Windows VB-CABLE / macOS BlackHole），本程序只负责读写选中的音频设备。
- 服务端与客户端走 WebSocket，可同机运行，也可分机（Mac 做瘦客户端、Windows/云 GPU 跑推理）。

## 目录结构

```
voice-changer/
├── voice_changer/
│   ├── __init__.py
│   ├── protocol.py          # 共享协议：Config / ConfigAck / Stats
│   ├── engine.py            # 变声引擎：后端选择、int16⇄float32、统计
│   ├── server.py            # WebSocket 推理服务端
│   ├── client.py            # 命令行客户端
│   ├── gui.py               # Tkinter 图形界面（标准库）
│   └── backends/
│       ├── base.py          # 后端抽象接口
│       ├── mock.py          # 纯 DSP 演示后端（无需模型/GPU）
│       ├── rvc.py           # RVC 后端（设备探测 + 适配器加载）
│       └── rvc_builtin.py   # 内置适配器骨架
├── tests/                   # 单元测试 + 端到端测试
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd voice-changer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Python 3.10+。GUI 用 Tkinter（标准库自带，无需安装）。

### 2. 先验证链路（mock 模式，不需要模型/GPU）

终端 A 启动服务端：

```bash
python -m voice_changer.server --host 127.0.0.1 --port 8765
```

终端 B 先列设备，再启动客户端：

```bash
python -m voice_changer.client --list-devices          # 看麦克风/扬声器索引
python -m voice_changer.client --mock --in 1 --out 2   # --out 选虚拟声卡
# 或启动图形界面：
python -m voice_changer.client --gui
```

> 戴耳机监听，避免扬声器声音被麦克风二次采集造成啸叫。

`--mock` 是纯 DSP 演示（重采样变调 + 共振峰搬移），**音质不能和 RVC 比**，只用来确认「采集 → 传输 → 变声 → 播放」整条链路打通。

### 3. 切换到真正的 RVC 变声

1. 自行准备 RVC 模型（`.pth` + 同名 `.index`），放到服务端 `models/` 目录。
2. 提供一个 RVC 适配器（见下节），告诉本项目如何调用你的 RVC 实现。
3. 客户端启动时去掉 `--mock`，指定模型：

```bash
python -m voice_changer.client --model gentle_female.pth --transpose 12 --index-rate 0.7
```

男声→女声建议 `transpose=12`（高一个八度），可调范围 6~14。`index_rate` 越高越像目标音色（0.6~0.8），过高会有电音、咬字损失。

## 接入自己的 RVC 实现

RVC 有多个实现（原版 RVC-WebUI、w-okada/voice-changer、RVC-WebUI-MacOS、ailia 等），API 各不相同。本项目不绑定任何一种，通过**适配器**对接。写一个 Python 模块，实现如下契约：

```python
# my_rvc_adapter.py —— 示例骨架（按你的实际 RVC API 填充）
import numpy as np

class RVCAdapter:
    def load(self, model_path, index_path, device):
        # 加载你的 RVC 模型到 device（"cuda:0" / "mps" / "cpu"）
        self.model = YourRVCInfer(model_path, index_path, device)

    def infer(self, audio, sample_rate, f0_up_key, index_rate, f0_method):
        # audio: float32 [-1,1] 单声道
        # f0_up_key: 变调半音数（如 12）
        # 返回等长 float32 音频
        return self.model.infer(audio, f0_up_key, index_rate, f0_method)

    def device_name(self):
        return self.model.device

    def close(self):
        ...

def create_adapter():
    return RVCAdapter()
```

然后用环境变量指向它（服务端进程）：

```bash
# 方式一：直接给模块路径（rvc.py 会调用其中的 create_adapter()）
RVC_ADAPTER=my_rvc_adapter python -m voice_changer.server --port 8765

# 方式二：给模块 + 类名（内置骨架用）
RVC_ADAPTER_MODULE=my_rvc_adapter RVC_ADAPTER_CLASS=RVCAdapter \
    python -m voice_changer.server --port 8765
```

> 服务端在 `auto` 模式下优先用 CUDA，其次 MPS，最后 CPU；也可在客户端配置里指定 `backend: cuda / mps / cpu`。

## 协议

- 控制消息：JSON 文本帧，见 `voice_changer/protocol.py`（`Config` / `ConfigAck` / `Stats`）。
- 音频帧：二进制 WebSocket 帧，固定 20ms、16kHz、单声道、int16（320 样本 = 640 字节）。
- 客户端连上后先发一条 `Config`，收到 `ConfigAck` 后开始发音频帧；服务端逐帧回变声后音频。
- 可随时再发 `Config` 热更新参数；每 100 帧服务端附带一条 `Stats`。

## 虚拟声卡配置（关键，决定会议软件能否听到变声）

- **Windows**：安装 VB-CABLE / Voicemeeter。客户端输出设备选 `CABLE Input`，会议软件（Zoom/Discord/OBS）麦克风选 `CABLE Output`。
- **macOS**：`brew install blackhole-2ch`。客户端输出选 BlackHole，会议软件输入选 BlackHole。

配置后务必戴耳机监听，防止啸叫/回声。

## 自测

```bash
python -m pytest tests -q
```

覆盖：协议序列化/校验、mock 引擎处理、真实 WebSocket 端到端音频往返（无需音频设备/GPU）。

## 设计说明

- **后端无关**：`VCBackend` 抽象接口让 mock / RVC / 未来其他模型（Seed-VC 等）可插拔。
- **每连接独立引擎**：服务端为每个 WebSocket 客户端建独立 `VoiceEngine`，支持多人同时使用、各自调参。
- **线程安全**：引擎内部用 `RLock` 保护推理与统计。
- **无平台专用代码**：设备探测全靠 torch/sounddevice 的运行时能力判断；本仓库不含任何 OS 分支。

## 许可与合规

本项目仅为技术骨架。变声涉及声音权与肖像权，请只用于自己或已获授权的声音；使用他人音色模型前确认模型授权范围。
