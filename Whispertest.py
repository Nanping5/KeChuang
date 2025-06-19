import whisper
import re

model = whisper.load_model(r"F:\KeChaung\whisper\small.pt")

def chinese_to_arabic(text):
    zh2ar = {
        "零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
    }
    for zh, ar in zh2ar.items():
        text = re.sub(zh, ar, text)
    return text

def transcribe_full(audio):
    if audio is None:
        return "请录音"
    result = model.transcribe(audio)
    text = result["text"]
    text = chinese_to_arabic(text)
    return text

def transcribe_keywords(audio):
    if audio is None:
        return "请录音"
    result = model.transcribe(audio)
    text = result["text"]
    text = chinese_to_arabic(text)
    directions = ["上", "下", "左", "右", "R\\+", "R-", "r\\+", "r-"]
    units = ["步", "度", "格", "step", "degree"]
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