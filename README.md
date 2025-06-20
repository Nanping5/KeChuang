#  机械臂智能抓取系统

本项目是一个基于YOLO目标检测、Gradio前端、Flask视频流、机械臂控制与大模型交互的智能抓取系统。支持高分辨率相机实时检测、机械臂精准抓取、语音/文本指令控制和大模型智能对话。

---

## 目录结构与主要文件说明

```
/
│
├── FrontEnd/
│   ├── gradiotest.py         # Gradio前端主程序，UI与事件绑定
│   ├── last_result.txt       # 语音识别结果缓存
│   └── recorded.wav          # 语音输入缓存
│
├── biaoding.txt              # 标定点数据
├── biaodingXML/              # 相机-机械臂标定XML文件
│   ├── 250616.xml
│   ├── 250617.xml
│   └── 250620.xml
│
├── coco_cat.txt              # COCO类别中英文对照表
├── command_parser.py         # 语音/文本运动指令解析与执行
├── connect.py                # 机械臂连接与通信
├── control.py                # 机械臂方向/旋转等基础控制
├── dobot_api.py              # Dobot机械臂底层API
├── grab_api.py               # 目标抓取主逻辑（YOLO检测结果+机械臂动作）
├── handEyeCali.py            # 手眼标定与像素-机械臂坐标变换
├── llm_api.py                # 大模型（LLM）对话与智能指令处理
├── requirements.txt          # pip依赖包列表
├── video_stream_server.py    # Flask视频流与YOLO检测后端服务
├── Whispertest.py            # Whisper语音识别
├── yolo_pth/
│   ├── yolov8m.pt            # YOLOv8模型权重
│   └── yolov8n.pt
├── yoloDemo.py                   # 本地YOLO+机械臂抓取测试脚本
├── yolo_stream.py            # YOLOStream类，摄像头采集与YOLO推理线程

```

---

## 主要模块功能

- **FrontEnd/gradiotest.py**  
  Gradio前端UI，负责用户交互、视频流展示、语音/文本输入、机械臂控制按钮等。所有功能通过API调用后端。

- **video_stream_server.py**  
  Flask服务，负责摄像头采集、YOLO目标检测、检测结果API、视频流API（原始/缩略图）。

- **grab_api.py**  
  目标抓取主逻辑。根据YOLO检测结果和标定参数，控制机械臂完成抓取动作。

- **llm_api.py**  
  大模型（如ChatGPT/DeepSeek）对话接口，结合检测结果智能理解用户意图并驱动抓取。

- **Whispertest.py**  
  Whisper语音识别，支持中英文语音转文本。

- **control.py / command_parser.py**  
  机械臂基础运动控制、语音/文本运动指令解析。

- **handEyeCali.py**  
  相机-机械臂手眼标定与像素坐标到机械臂坐标的转换。

- **yolo_stream.py**  
  YOLOStream类，负责摄像头采集线程和YOLO推理。

- **connect.py / dobot_api.py**  
  机械臂底层连接与通信。

---

## 环境依赖（conda环境推荐）

建议使用Anaconda/Miniconda创建独立环境，以下为推荐的 `environment.yml`：

```yaml
name: kechaung
channels:
  - defaults
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - pip
  - numpy=1.23.*
  - opencv=4.6.*
  - flask=2.2.*
  - requests=2.28.*
  - gradio=3.50.*
  - pytorch=2.0.*        # 或根据你的CUDA版本选择
  - torchvision=0.15.*
  - torchaudio=2.0.*
  - pyyaml
  - pip:
      - ultralytics==8.0.197
      - openai==1.3.5
      - python-dotenv
      - whisper @ git+https://github.com/openai/whisper.git
      - deepseek-openai
```

> **注意：**  
> - YOLOv8/ultralytics、Whisper、openai、deepseek-openai等需通过pip安装。
> - 若用GPU，需提前安装好对应CUDA驱动和PyTorch版本。
> - 机械臂相关依赖（如Dobot SDK）请根据实际硬件环境补充。

---

## 快速启动

1. **创建环境并安装依赖**
   ```bash
   conda env create -f environment.yml
   conda activate kechaung
   ```

2. **启动后端视频流与检测服务**
   ```bash
   python video_stream_server.py
   ```

3. **启动前端Gradio界面**
   ```bash
   cd FrontEnd
   python gradiotest.py
   ```

4. **访问前端**
   - 默认在 [http://localhost:7860](http://localhost:7860) 打开Gradio界面
   - 视频流/检测API端口为5001

---

## 其它说明

- **标定文件**：请将相机-机械臂标定XML放在 `biaodingXML/` 下，并在相关py文件中指定路径。
- **YOLO模型权重**：请将yolov8模型权重放在 `yolo_pth/` 下，路径需与代码一致。
- **API密钥**：如用OpenAI/DeepSeek等大模型服务，请在 `.env` 文件中配置 `OPENAI_API_KEY`。

---

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Gradio](https://gradio.app/)
- [Flask](https://flask.palletsprojects.com/)
- 以及所有开源社区！

---

如有问题欢迎提issue或联系作者。 