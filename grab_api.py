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
aff_m = handEyeCalibration(r'F:\KeChaung\biaodingXML\250617.xml')

def grab_by_name(target_name: str, yolo_stream=None):
    """
    传入目标名称（中英文均可），检测yolo_stream最新画面，若有匹配则抓取。
    yolo_stream: 必须传入YOLOStream实例。
    返回: str, 抓取结果描述
    """
    if yolo_stream is None:
        return "未传入YOLOStream实例，无法获取摄像头画面"
    # 获取原始分辨率帧
    frame = yolo_stream.get_frame(width=2592, height=1944)
    if frame is None:
        return "没有可用画面"
    # 检测
    results = model(frame, device=device, verbose=False)
    found = False
    for res in results:
        for box in res.boxes:
            class_id = int(box.cls[0])
            class_name_en = res.names[class_id] if hasattr(res, 'names') else str(class_id)
            class_name_cn = coco_categories.get(class_id, class_name_en)
            # 中英文全等匹配
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
                    # 1. 移动到目标
                    result = move.MovJ(robot_x, robot_y, robot_z, robot_r)
                    move.Sync()
                    # 2. 吸气
                    dashboard.DOExecute(1, 1)
                    time.sleep(1)
                    # 3. 移动到复位
                    move.MovJ(197, -250, 58, 0)
                    move.Sync()
                    # 4. 释放
                    dashboard.DOExecute(1, 0)
                    dashboard.DOExecute(2, 1)
                    time.sleep(0.5)
                    dashboard.DOExecute(2, 0)
                    return f"成功抓取: {class_name_cn} ({class_name_en})"
                except Exception as e:
                    return f"机械臂抓取失败: {e}"
    if not found:
        return f"未检测到目标: {target_name}" 