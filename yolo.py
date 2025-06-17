import cv2
from ultralytics import YOLO
from handEyeCali import handEyeCalibration, useAffineMat
from Basic01_dobot import ConnectRobot

# 初始化仿射矩阵
aff_m = handEyeCalibration(r'biaodingXML\\250616.xml')  # 路径按实际修改

# 连接机械臂
try:
    dashboard, move, feed = ConnectRobot()
except Exception as e:
    print("机械臂连接失败：", e)
    dashboard = move = feed = None

model = YOLO(r'F:\\KeChaung\\yolo_pth\\yolov8m.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

move_pending = None  # 存储待移动的目标中心点

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头未采集到画面")
        break

    print("frame sum before YOLO:", frame.sum())
    if frame.sum() == 0:
        print("frame全黑，实际分辨率：", frame.shape)
    else:
        print("frame正常，实际分辨率：", frame.shape)

    results = model(frame)
    move_pending = None
    for res in results:
        for box in res.boxes:
            xyxy = box.xyxy.squeeze().tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # 获取类别名
            class_id = int(box.cls[0])
            class_name = res.names[class_id] if hasattr(res, 'names') else str(class_id)
            cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            # 只取第一个目标
            if move_pending is None:
                move_pending = (x1, y1, x2, y2)
    print("frame sum after YOLO:", frame.sum())

    display_frame = cv2.resize(frame, (680, 480))
    cv2.imshow('YOLO Camera', display_frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key == 13 and move_pending is not None and move is not None:  # Enter键
        x1, y1, x2, y2 = move_pending
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"检测到目标中心像素: ({center_x}, {center_y})，frame.shape={frame.shape}")
        robot_x, robot_y = useAffineMat(aff_m, center_x, center_y)
        robot_z = -140
        robot_r = 0
        print(f"变换后机械臂坐标: ({robot_x}, {robot_y}, {robot_z}, {robot_r})")
        try:
            result = move.MovJ(robot_x, robot_y, robot_z, robot_r)
            print("机械臂回复：", result)
        except Exception as e:
            print("机械臂移动失败：", e)

cap.release()
cv2.destroyAllWindows()
