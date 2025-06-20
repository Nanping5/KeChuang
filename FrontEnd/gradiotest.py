import gradio as gr
from connect import ConnectRobot
from control import move_robot
from Whispertest import transcribe_full, transcribe_keywords
from command_parser import execute_motion
from llm_api import llm_chat
from grab_api import grab_by_name, coco_categories
import requests
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from handEyeCali import useAffineMat, handEyeCalibration
import time
aff_m = handEyeCalibration(r'F:\KeChaung\biaodingXML\250620.xml')
# 获取最新视频帧（从video_stream_server）
def get_latest_frame_from_stream(url="http://localhost:5001/video_feed"):
    try:
        stream = requests.get(url, stream=True, timeout=2)
        bytes_data = b""
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                # bytes_data = bytes_data[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return img
        return None
    except Exception as e:
        print(f"拉取视频帧失败: {e}")
        return None

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
def get_yolo_detected_names(dummy=None):
    detection = get_latest_detection()
    if not detection or not detection.get("boxes"):
        return []
    detected = set()
    for box in detection["boxes"]:
        detected.add(box["class_name"].strip().lower())
        # 可加中英文映射
    return list(detected)

# 智能抓取prompt
SMART_PROMPT = (
    "你是一个机械臂助手。当前画面可抓取的物体有：{detected_str}。"
    "用户如果说'帮我拿一下xxx'、'抓取xxx'等指令时，请优先从可抓取物体中选择最合适的一个，并只回复物体名称（如'banana'），不要回复其它内容。"
    "如果没有可抓取的物体或用户请求的物体不在可抓取列表中，请回复'当前画面没有可抓取的目标'。"
)

def is_coco_name(text):
    return text.strip() in COCO_NAMES

# 获取检测结果
def get_latest_detection(url="http://localhost:5001/detect_result"):
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        print(f"拉取检测结果失败: {e}")
        return None

# 用检测结果做抓取
def grab_by_name_from_detection(target_name):
    detection = get_latest_detection()
    if not detection or not detection.get("boxes"):
        return {"result": "没有检测到目标"}
    for box in detection["boxes"]:
        class_name = box["class_name"]
        class_id = box["class_id"]
        class_name_cn = coco_categories.get(class_id, class_name)
        if target_name == class_name or target_name == class_name_cn:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            robot_x, robot_y = useAffineMat(aff_m, center_x, center_y)
            robot_z = -130
            robot_r = 0
            try:
                dashboard, move, feed = ConnectRobot()
                result = move.MovJ(robot_x, robot_y, robot_z, robot_r)
                move.Sync()
                dashboard.DOExecute(1, 1)
                time.sleep(1)
                move.MovJ(197, -250, 58, 0)
                move.Sync()
                dashboard.DOExecute(1, 0)
                dashboard.DOExecute(2, 1)
                time.sleep(0.5)
                dashboard.DOExecute(2, 0)
                return {
                    "result": f"成功抓取: {class_name_cn} ({class_name})",
                    "class_name_cn": class_name_cn,
                    "class_name_en": class_name,
                    "center_x": center_x,
                    "center_y": center_y,
                    "robot_x": robot_x,
                    "robot_y": robot_y
                }
            except Exception as e:
                return {"result": f"机械臂抓取失败: {e}"}
    return {"result": f"未检测到目标: {target_name}"}

# 语音对话主逻辑（大模型对话，完整文本）
def voice_to_llm(audio, history):
    user_text = transcribe_full(audio)
    if not user_text or user_text == "请录音":
        return history, "未识别到语音"
    detected_names = get_yolo_detected_names()
    detected_str = ", ".join(detected_names) if detected_names else "无"
    system_prompt = SMART_PROMPT.format(detected_str=detected_str)
    print("detected_names:", detected_names)
    print("system_prompt:", system_prompt)
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
    messages.append({"role": "user", "content": user_text})
    bot_reply = llm_chat(messages, system_prompt=system_prompt)
    print("bot_reply:", bot_reply)
    reply_norm = bot_reply.strip().lower()
    matched = any(reply_norm in name or name in reply_norm for name in detected_names)
    if matched:
        grab_result = grab_by_name_from_detection(bot_reply)
        if isinstance(grab_result, dict):
            print(f"检测到目标: {grab_result.get('class_name_cn')} ({grab_result.get('class_name_en')})，像素中心: ({grab_result.get('center_x')}, {grab_result.get('center_y')})，机械臂坐标: ({grab_result.get('robot_x')}, {grab_result.get('robot_y')})")
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": f"[抓取]{bot_reply}\n{grab_result.get('result')}"})
            return history, grab_result.get('result')
        else:
            print(grab_result)
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
    detected_names = get_yolo_detected_names()
    detected_str = ", ".join(detected_names) if detected_names else "无"
    system_prompt = SMART_PROMPT.format(detected_str=detected_str)
    print("detected_names:", detected_names)
    print("system_prompt:", system_prompt)
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
    messages.append({"role": "user", "content": user_text})
    bot_reply = llm_chat(messages, system_prompt=system_prompt)
    print("bot_reply:", bot_reply)
    reply_norm = bot_reply.strip().lower()
    matched = any(reply_norm in name or name in reply_norm for name in detected_names)
    if matched:
        grab_result = grab_by_name_from_detection(bot_reply)
        if isinstance(grab_result, dict):
            print(f"检测到目标: {grab_result.get('class_name_cn')} ({grab_result.get('class_name_en')})，像素中心: ({grab_result.get('center_x')}, {grab_result.get('center_y')})，机械臂坐标: ({grab_result.get('robot_x')}, {grab_result.get('robot_y')})")
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": f"[抓取]{bot_reply}\n{grab_result.get('result')}"})
            return history, grab_result.get('result')
        else:
            print(grab_result)
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

with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background: #f7f9fa;} .gr-button {margin: 2px 4px;} .gr-textbox {margin-bottom: 8px;}") as demo:
    gr.Markdown("""
    <h1 style='text-align:center; color:#2d3a4b;'>GUI</h1>
    <hr style='margin-bottom: 10px;'>
    """)
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("<h3 style='color:#3b5998;'>机械臂方向/旋转控制</h3>")
            with gr.Row():
                up_btn = gr.Button("↑", elem_id="up-btn")
            with gr.Row():
                left_btn = gr.Button("←", elem_id="left-btn")
                right_btn = gr.Button("→", elem_id="right-btn")
            with gr.Row():
                down_btn = gr.Button("↓", elem_id="down-btn")
            with gr.Row():
                rplus_btn = gr.Button("R+", elem_id="rplus-btn")
                rminus_btn = gr.Button("R-", elem_id="rminus-btn")
            gr.Markdown("<hr style='margin:10px 0;'>")
            gr.Markdown("<h3 style='color:#3b5998;'>语音转文字（Whisper）</h3>")
            audio_in = gr.Audio(type="filepath", label="请录音或上传音频", elem_id="audio-in")
            with gr.Row():
                recog_btn = gr.Button("识别音频", elem_id="recog-btn")
                exec_btn = gr.Button("执行指令", elem_id="exec-btn")
            recog_out = gr.Textbox(label="筛选后指令", interactive=True, elem_id="recog-out")
            exec_out = gr.Textbox(label="执行结果", elem_id="exec-out")
            gr.Markdown("<hr style='margin:10px 0;'>")
            status_box = gr.Textbox(label="机械臂状态", value="等待操作...", interactive=False, elem_id="status-box")
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("<h3 style='color:#3b5998;'>实时画面</h3>")
            gr.HTML('<div style="display:flex; justify-content:center;"><img src="http://localhost:5001/video_feed_thumb" width="640" height="480" style="border-radius:8px; box-shadow:0 2px 8px #ccc;" /></div>')
            gr.Markdown("<hr style='margin:10px 0;'>")
            gr.Markdown("<h3 style='color:#3b5998;'>大模型对话</h3>")
            chatbot = gr.Chatbot(label="对话历史", type="messages", elem_id="chatbot")
            with gr.Row():
                audio_chat = gr.Audio(type="filepath", label="语音输入", elem_id="audio-chat")
                talk_btn = gr.Button("语音识别", elem_id="talk-btn")
            with gr.Row():
                text_input = gr.Textbox(label="输入", elem_id="text-input")
                send_btn = gr.Button("发送", elem_id="send-btn")
            chat_status = gr.Textbox(label="对话状态", interactive=False, elem_id="chat-status")
            gr.Markdown("<hr style='margin:10px 0;'>")
    # 按钮事件绑定
    up_btn.click(lambda: move_robot('up', dashboard, move), outputs=status_box)
    down_btn.click(lambda: move_robot('down', dashboard, move), outputs=status_box)
    left_btn.click(lambda: move_robot('left', dashboard, move), outputs=status_box)
    right_btn.click(lambda: move_robot('right', dashboard, move), outputs=status_box)
    rplus_btn.click(lambda: move_robot('r+', dashboard, move), outputs=status_box)
    rminus_btn.click(lambda: move_robot('r-', dashboard, move), outputs=status_box)
    recog_btn.click(transcribe_keywords, inputs=audio_in, outputs=recog_out)
    exec_btn.click(lambda filtered: execute_motion(filtered, dashboard, move), inputs=recog_out, outputs=exec_out)
    talk_btn.click(voice_to_llm, inputs=[audio_chat, chatbot], outputs=[chatbot, chat_status])
    send_btn.click(text_to_llm, inputs=[text_input, chatbot], outputs=[chatbot, chat_status])

demo.launch()
