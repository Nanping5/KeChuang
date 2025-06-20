from openai import OpenAI
import os
from dotenv import load_dotenv
from grab_api import get_yolo_detected_names, SMART_PROMPT, grab_by_name_from_detection
from Whispertest import transcribe_full

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("请先设置环境变量 OPENAI_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def llm_chat(messages, system_prompt=None):
    # messages: list of {"role": "user"/"assistant", "content": ...}
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content 

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