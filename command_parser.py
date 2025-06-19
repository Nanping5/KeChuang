import re
from control import move_robot
from Whispertest import chinese_to_arabic

def execute_motion(filtered, dashboard, move):
    filtered = chinese_to_arabic(filtered)
    directions = ["上", "下", "左", "右", "R+", "R-", "r+", "r-"]
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
            for i in range(count):
                result = move_robot(direction, dashboard, move)
                feedbacks.append(f"{cmd} 第{i+1}次: {result}")
                if "超出限位" in result or "只允许回安全区方向移动" in result or "未连接" in result or "获取位姿失败" in result:
                    break
        return "\n".join(feedbacks)
    return "未检测到运动指令" 