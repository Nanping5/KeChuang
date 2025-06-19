import gradio as gr
from connect import ConnectRobot
from control import move_robot
from Whispertest import transcribe_full, transcribe_keywords
from command_parser import execute_motion
from llm_api import llm_chat
from grab_api import grab_by_name, coco_categories
from yolo_stream import YOLOStream

# 初始化全局YOLOStream实例
yolo_stream = YOLOStream()

# 机械臂连接
try:
    dashboard, move, feed = ConnectRobot()
except Exception as e:
    dashboard = move = feed = None
    print(f"机械臂连接失败: {e}")

# COCO类别中英文列表
COCO_NAMES = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", "船", "交通灯", "消防栓", "停车标志", "停车计费器", "长凳", "鸟", "猫", "狗", "马", "绵羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板", "雪橇", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 获取当前YOLO检测到的所有英文类别名
def get_yolo_detected_names(yolo_stream):
    frame = yolo_stream.get_frame(width=2592, height=1944)
    if frame is None:
        return []
    from ultralytics import YOLO
    import torch
    model = YOLO(r'F:\KeChaung\yolo_pth\yolov8m.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model(frame, device=device, verbose=False)
    detected = set()
    for res in results:
        for box in res.boxes:
            class_id = int(box.cls[0])
            class_name_en = res.names[class_id] if hasattr(res, 'names') else str(class_id)
            class_name_cn = coco_categories.get(class_id, class_name_en)
            detected.add(class_name_en.strip().lower())
            detected.add(str(class_name_cn).strip().lower())
    return list(detected)

# 智能抓取prompt
SMART_PROMPT = (
    "你是一个机械臂助手。当前画面可抓取的物体有：{detected_str}。"
    "用户如果说'帮我拿一下xxx'、'抓取xxx'等指令时，请优先从可抓取物体中选择最合适的一个，并只回复物体名称（如'banana'），不要回复其它内容。"
    "如果没有可抓取的物体或用户请求的物体不在可抓取列表中，请回复'当前画面没有可抓取的目标'。"
)

def is_coco_name(text):
    return text.strip() in COCO_NAMES

# 语音对话主逻辑（大模型对话，完整文本）
def voice_to_llm(audio, history):
    user_text = transcribe_full(audio)
    if not user_text or user_text == "请录音":
        return history, "未识别到语音"
    detected_names = get_yolo_detected_names(yolo_stream)
    detected_str = ", ".join(detected_names) if detected_names else "无"
    system_prompt = SMART_PROMPT.format(detected_str=detected_str)
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
    messages.append({"role": "user", "content": user_text})
    bot_reply = llm_chat(messages, system_prompt=system_prompt)
    reply_norm = bot_reply.strip().lower()
    if reply_norm in detected_names:
        grab_result = grab_by_name(bot_reply, yolo_stream=yolo_stream)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": f"[抓取]{bot_reply}\n{grab_result}"})
        return history, grab_result
    elif reply_norm == "当前画面没有可抓取的目标":
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_reply})
        return history, "当前画面没有可抓取的目标"
    else:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_reply})
        return history, ""

# 打字输入对话逻辑
def text_to_llm(user_text, history):
    if not user_text:
        return history, "请输入内容"
    detected_names = get_yolo_detected_names(yolo_stream)
    detected_str = ", ".join(detected_names) if detected_names else "无"
    system_prompt = SMART_PROMPT.format(detected_str=detected_str)
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
    messages.append({"role": "user", "content": user_text})
    bot_reply = llm_chat(messages, system_prompt=system_prompt)
    reply_norm = bot_reply.strip().lower()
    if reply_norm in detected_names:
        grab_result = grab_by_name(bot_reply, yolo_stream=yolo_stream)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": f"[抓取]{bot_reply}\n{grab_result}"})
        return history, grab_result
    elif reply_norm == "当前画面没有可抓取的目标":
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_reply})
        return history, "当前画面没有可抓取的目标"
    else:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": bot_reply})
        return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# GUI")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 方向/旋转控制")
            with gr.Row():
                up_btn = gr.Button("↑")
            with gr.Row():
                left_btn = gr.Button("←")
                right_btn = gr.Button("→")
            with gr.Row():
                down_btn = gr.Button("↓")
            with gr.Row():
                rplus_btn = gr.Button("R+")
                rminus_btn = gr.Button("R-")
            gr.Markdown("## 语音转文字（Whisper）")
            audio_in = gr.Audio(type="filepath", label="请录音或上传音频")
            recog_btn = gr.Button("识别音频")
            recog_out = gr.Textbox(label="筛选后指令", interactive=True)
            exec_btn = gr.Button("执行指令")
            exec_out = gr.Textbox(label="执行结果")
        with gr.Column():
            status_box = gr.Textbox(label="机械臂状态", value="等待操作...", interactive=False)
            gr.Markdown("## YOLO目标检测实时画面")
            gr.HTML('<img src="http://localhost:5001/video_feed" width="640" height="480" />')
            gr.Markdown("## 大模型多轮对话")
            chatbot = gr.Chatbot(label="对话历史", type="messages")
            with gr.Row():
                audio_chat = gr.Audio(type="filepath", label="语音输入")
                talk_btn = gr.Button("语音对话")
            with gr.Row():
                text_input = gr.Textbox(label="打字输入")
                send_btn = gr.Button("发送")
            chat_status = gr.Textbox(label="对话状态", interactive=False)
            talk_btn.click(voice_to_llm, inputs=[audio_chat, chatbot], outputs=[chatbot, chat_status])
            send_btn.click(text_to_llm, inputs=[text_input, chatbot], outputs=[chatbot, chat_status])

    up_btn.click(lambda: move_robot('up', dashboard, move), outputs=status_box)
    down_btn.click(lambda: move_robot('down', dashboard, move), outputs=status_box)
    left_btn.click(lambda: move_robot('left', dashboard, move), outputs=status_box)
    right_btn.click(lambda: move_robot('right', dashboard, move), outputs=status_box)
    rplus_btn.click(lambda: move_robot('r+', dashboard, move), outputs=status_box)
    rminus_btn.click(lambda: move_robot('r-', dashboard, move), outputs=status_box)

    recog_btn.click(transcribe_keywords, inputs=audio_in, outputs=recog_out)
    exec_btn.click(lambda filtered: execute_motion(filtered, dashboard, move), inputs=recog_out, outputs=exec_out)

demo.launch()
