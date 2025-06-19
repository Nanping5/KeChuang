import gradio as gr
import whisper
import sys
import time
import threading
from connect import ConnectRobot
import re

try:
    import msvcrt  # Windows下用于捕获键盘输入
except ImportError:
    print("本脚本仅支持Windows平台。")
    sys.exit(1)

# 加载本地模型
model = whisper.load_model(r"F:\KeChaung\whisper\small.pt")

# 机械臂连接
try:
    dashboard, move, feed = ConnectRobot()
except Exception as e:
    dashboard = move = feed = None
    print(f"机械臂连接失败: {e}")

# 机械臂限位
X_MIN, X_MAX = 200, 400
Y_MIN, Y_MAX = -200, 90
STEP_XY = 10
STEP_R = 5

# 获取当前位姿
def get_pose():
    pose_str = dashboard.GetPose()
    try:
        match = re.search(r"\{([^}]*)\}", pose_str)
        if not match:
            raise ValueError("未找到内容")
        pose_data = match.group(1)
        x, y, z, r = [float(v) for v in pose_data.split(',')[:4]]
        return x, y, z, r
    except Exception as e:
        return None

# 机械臂控制函数

def move_robot(direction):
    if move is None or dashboard is None:
        return "机械臂未连接"
    pose = get_pose()
    if pose is None:
        return "获取位姿失败"
    x, y, z, r = pose
    # 限位判断
    if not (X_MIN < x < X_MAX and Y_MIN < y < Y_MAX):
        # 只允许回安全区方向移动
        if direction == 'up' and y < Y_MIN:
            move.RelMovL(0, STEP_XY, 0, 0)
            return "上 (y+10)"
        elif direction == 'down' and y > Y_MAX:
            move.RelMovL(0, -STEP_XY, 0, 0)
            return "下 (y-10)"
        elif direction == 'left' and x > X_MAX:
            move.RelMovL(-STEP_XY, 0, 0, 0)
            return "左 (x-10)"
        elif direction == 'right' and x < X_MIN:
            move.RelMovL(STEP_XY, 0, 0, 0)
            return "右 (x+10)"
        else:
            return "当前位置已超限，只允许回安全区方向移动！"
    # 正常限位内
    if direction == 'up':
        if Y_MIN < y + STEP_XY < Y_MAX:
            move.RelMovL(0, STEP_XY, 0, 0)
            return "上 (y+10)"
        else:
            return "超出限位，禁止移动！"
    elif direction == 'down':
        if Y_MIN < y - STEP_XY < Y_MAX:
            move.RelMovL(0, -STEP_XY, 0, 0)
            return "下 (y-10)"
        else:
            return "超出限位，禁止移动！"
    elif direction == 'left':
        if X_MIN < x - STEP_XY < X_MAX:
            move.RelMovL(-STEP_XY, 0, 0, 0)
            return "左 (x-10)"
        else:
            return "超出限位，禁止移动！"
    elif direction == 'right':
        if X_MIN < x + STEP_XY < X_MAX:
            move.RelMovL(STEP_XY, 0, 0, 0)
            return "右 (x+10)"
        else:
            return "超出限位，禁止移动！"
    elif direction == 'r+':
        move.RelMovJ(0, 0, 0, STEP_R)
        return "r+1"
    elif direction == 'r-':
        move.RelMovJ(0, 0, 0, -STEP_R)
        return "r-1"
    else:
        return "未知指令"

# 中文数字转阿拉伯数字
def chinese_to_arabic(text):
    zh2ar = {
        "零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
    }
    # 只处理十以内的单个数字
    for zh, ar in zh2ar.items():
        text = re.sub(zh, ar, text)
    return text

# 语音识别（仅筛选并显示指令，不执行）
def transcribe(audio):
    if audio is None:
        return "请录音"
    result = model.transcribe(audio)
    text = result["text"]
    text = chinese_to_arabic(text)
    # 支持"向上3步""往左2步"等表达
    directions = ["上", "下", "左", "右", "R\+", "R-", "r\+", "r-"]
    units = ["步", "度", "格", "step", "degree"]
    # 新增前缀"向""往"可选
    pattern = r"((?:向|往)?(?P<dir>" + "|".join(directions) + r")(?:\s*(?P<num>\d+))?(?:\s*(?P<unit>" + "|".join(units) + r"))?)"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    filtered = " ".join([m.group(0) for m in matches if m.group(0)])
    if not filtered:
        pattern2 = r"((?:向|往)?(?P<dir>" + "|".join(directions) + r")(?:\s*(?P<num>\d+))?)"
        matches = list(re.finditer(pattern2, text, flags=re.IGNORECASE))
        filtered = " ".join([m.group(0) for m in matches if m.group(0)])
    with open("last_result.txt", "w", encoding="utf-8") as f:
        f.write(filtered)
    return filtered if filtered else "未检测到运动指令"

# 新增：解析并执行运动指令（从文本框内容）
def execute_motion(filtered):
    import re
    filtered = chinese_to_arabic(filtered)
    directions = ["上", "下", "左", "右", "R\+", "R-", "r\+", "r-"]
    units = ["步", "度", "格", "step", "degree"]
    pattern = r"((?:向|往)?(?P<dir>" + "|".join(directions) + r")(?:\s*(?P<num>\d+))?(?:\s*(?P<unit>" + "|".join(units) + r"))?)"
    matches = list(re.finditer(pattern, filtered, flags=re.IGNORECASE))
    if not matches:
        pattern2 = r"((?:向|往)?(?P<dir>" + "|".join(directions) + r")(?:\s*(?P<num>\d+))?)"
        matches = list(re.finditer(pattern2, filtered, flags=re.IGNORECASE))
    if matches:
        feedbacks = []
        for m in matches:
            cmd = m.group(0)
            dir_word = m.group('dir')
            num = m.group('num')
            if re.search(r"上", dir_word, re.IGNORECASE):
                direction = 'up'
            elif re.search(r"下", dir_word, re.IGNORECASE):
                direction = 'down'
            elif re.search(r"左", dir_word, re.IGNORECASE):
                direction = 'left'
            elif re.search(r"右", dir_word, re.IGNORECASE):
                direction = 'right'
            elif re.search(r"R\+|r\+", dir_word, re.IGNORECASE):
                direction = 'r+'
            elif re.search(r"R-|r-", dir_word, re.IGNORECASE):
                direction = 'r-'
            else:
                direction = None
            count = int(num) if num else 1
            step = 10
            if direction:
                global STEP_XY, STEP_R
                old_step_xy, old_step_r = STEP_XY, STEP_R
                if direction in ['up', 'down', 'left', 'right']:
                    STEP_XY = step
                else:
                    STEP_R = step
                for i in range(count):
                    result = move_robot(direction)
                    feedbacks.append(f"{cmd} 第{i+1}次: {result}")
                    if "超出限位" in result or "只允许回安全区方向移动" in result or "未连接" in result or "获取位姿失败" in result:
                        break
                STEP_XY, STEP_R = old_step_xy, old_step_r
            else:
                feedbacks.append(f"识别到: {cmd}，但未能解析方向")
        return "\n".join(feedbacks)
    return "未检测到运动指令"

# Gradio界面
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

    # 机械臂控制按钮事件
    up_btn.click(lambda: move_robot('up'), outputs=status_box)
    down_btn.click(lambda: move_robot('down'), outputs=status_box)
    left_btn.click(lambda: move_robot('left'), outputs=status_box)
    right_btn.click(lambda: move_robot('right'), outputs=status_box)
    rplus_btn.click(lambda: move_robot('r+'), outputs=status_box)
    rminus_btn.click(lambda: move_robot('r-'), outputs=status_box)

    # 语音识别按钮事件（只识别并筛选，不执行）
    recog_btn.click(transcribe, inputs=audio_in, outputs=recog_out)
    # 执行按钮事件（从文本框内容执行）
    exec_btn.click(execute_motion, inputs=recog_out, outputs=exec_out)

demo.launch()
