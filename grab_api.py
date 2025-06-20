import torch
from ultralytics import YOLO
from handEyeCali import handEyeCalibration, useAffineMat
from connect import ConnectRobot
import time

# 集成COCO类别中英文对照字典
coco_categories = {
    0: "人", 1: "自行车", 2: "汽车", 3: "摩托车", 4: "飞机", 5: "公共汽车", 6: "火车", 7: "卡车", 8: "船", 9: "交通灯",
    10: "消防栓", 11: "停车标志", 12: "停车计费器", 13: "长凳", 14: "鸟", 15: "猫", 16: "狗", 17: "马", 18: "绵羊", 19: "牛",
    20: "大象", 21: "熊", 22: "斑马", 23: "长颈鹿", 24: "背包", 25: "雨伞", 26: "手提包", 27: "领带", 28: "行李箱", 29: "飞盘",
    30: "滑雪板", 31: "雪橇", 32: "运动球", 33: "风筝", 34: "棒球棒", 35: "棒球手套", 36: "滑板", 37: "冲浪板", 38: "网球拍", 39: "瓶子",
    40: "酒杯", 41: "杯子", 42: "叉子", 43: "刀", 44: "勺子", 45: "碗", 46: "香蕉", 47: "苹果", 48: "三明治", 49: "橙子",
    50: "西兰花", 51: "胡萝卜", 52: "热狗", 53: "披萨", 54: "甜甜圈", 55: "蛋糕", 56: "椅子", 57: "沙发", 58: "盆栽植物", 59: "床",
    60: "餐桌", 61: "马桶", 62: "电视", 63: "笔记本电脑", 64: "鼠标", 65: "遥控器", 66: "键盘", 67: "手机", 68: "微波炉", 69: "烤箱",
    70: "烤面包机", 71: "水槽", 72: "冰箱", 73: "书", 74: "时钟", 75: "花瓶", 76: "剪刀", 77: "泰迪熊", 78: "吹风机", 79: "牙刷"
}

# 初始化
model = YOLO(r'F:\KeChaung\yolo_pth\yolov8m.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
aff_m = handEyeCalibration(r'F:\KeChaung\biaodingXML\250620.xml')

# COCO类别中英文列表（如需）
COCO_NAMES = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", "船", "交通灯", "消防栓", "停车标志", "停车计费器", "长凳", "鸟", "猫", "狗", "马", "绵羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板", "雪橇", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

SMART_PROMPT = (
    "你是一个机械臂助手。当前画面可抓取的物体有：{detected_str}。"
    "用户如果说'帮我拿一下xxx'、'抓取xxx'等指令时，请优先从可抓取物体中选择最合适的一个，并只回复物体名称（如'banana'），不要回复其它内容。"
    "如果没有可抓取的物体或用户请求的物体不在可抓取列表中，请回复'当前画面没有可抓取的目标'。"
)

def is_coco_name(text):
    return text.strip() in COCO_NAMES

def get_latest_detection(url="http://localhost:5001/detect_result"):
    import requests
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        print(f"拉取检测结果失败: {e}")
        return None

def get_yolo_detected_names(dummy=None):
    detection = get_latest_detection()
    if not detection or not detection.get("boxes"):
        return []
    detected = set()
    for box in detection["boxes"]:
        detected.add(box["class_name"].strip().lower())
    return list(detected)

def grab_by_name_from_detection(target_name):
    from handEyeCali import useAffineMat, handEyeCalibration
    from connect import ConnectRobot
    import time
    aff_m = handEyeCalibration(r'F:\KeChaung\biaodingXML\250620.xml')
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
                move.MovJ(robot_x, robot_y, 58, 0)
                move.Sync()
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

def grab_by_name(target_name: str, yolo_stream=None, frame=None):
    """
    传入目标名称（中英文均可），检测yolo_stream最新画面或frame，若有匹配则抓取。
    yolo_stream: 可选YOLOStream实例。
    frame: 可选OpenCV图像帧。
    返回: dict, 包含抓取结果、类别名、像素中心、机械臂坐标等
    """
    if frame is not None:
        input_frame = frame
    elif yolo_stream is not None:
        input_frame = yolo_stream.get_frame(width=2592, height=1944)
        if input_frame is None:
            return {"result": "没有可用画面"}
    else:
        return {"result": "未传入YOLOStream实例或frame，无法获取摄像头画面"}
    results = model(input_frame, device=device, verbose=False)
    found = False
    for res in results:
        for box in res.boxes:
            class_id = int(box.cls[0])
            class_name_en = res.names[class_id] if hasattr(res, 'names') else str(class_id)
            class_name_cn = coco_categories.get(class_id, class_name_en)
            if target_name == class_name_en or target_name == class_name_cn:
                found = True
                xyxy = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
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
                    move.MovJ(robot_x, robot_y, 58, 0)
                    move.Sync()
                    move.MovJ(197, -250, -50, 0)
                    move.Sync()
                    dashboard.DOExecute(1, 0)
                    dashboard.DOExecute(2, 1)
                    time.sleep(0.5)
                    dashboard.DOExecute(2, 0)
                    return {
                        "result": f"成功抓取: {class_name_cn} ({class_name_en})",
                        "class_name_cn": class_name_cn,
                        "class_name_en": class_name_en,
                        "center_x": center_x,
                        "center_y": center_y,
                        "robot_x": robot_x,
                        "robot_y": robot_y
                    }
                except Exception as e:
                    return {"result": f"机械臂抓取失败: {e}"}
    if not found:
        return {"result": f"未检测到目标: {target_name}"} 