import time
import cv2
import torch
import threading
from queue import Queue
from collections import deque
from ultralytics import YOLO
from handEyeCali import handEyeCalibration, useAffineMat
from connect import ConnectRobot
import datetime

def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

class FPS:
    def __init__(self, avg_frames=30):
        self.timestamps = deque(maxlen=avg_frames)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])

# 检查是否有可用的GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    log(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    log("未检测到GPU，使用CPU")

# 初始化
aff_m = handEyeCalibration(r'biaodingXML\\250617.xml')

# 连接机械臂
try:
    dashboard, move, feed = ConnectRobot()
except Exception as e:
    log(f"机械臂连接失败：{e}")
    dashboard = move = feed = None

# 加载模型到GPU
model = YOLO(r'F:\\KeChaung\\yolo_pth\\yolov8m.pt')
model.to(device)

# 创建队列用于线程间通信
frame_queue = Queue(maxsize=1)  # 减小队列大小，只保留最新帧
result_queue = Queue(maxsize=1)

# 创建FPS计数器
capture_fps = FPS()
process_fps = FPS()
display_fps = FPS()

# 全局检测开关
detecting = threading.Event()
detecting.set()  # 初始允许检测

center_history = deque(maxlen=5)  # 保存最近5帧中心点

# 创建线程控制标志
exit_flag = threading.Event()
busy = False
grab_queue = Queue(maxsize=1)  # 用于传递抓取任务

def capture_frames(cap):
    """摄像头捕获线程"""
    while not exit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            log("摄像头未采集到画面")
            break
        
        # 更新旧帧
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame)
        capture_fps.update()
    cap.release()

def process_frames():
    """YOLO处理线程"""
    last_detect_time = 0
    detect_interval = 0.3  # 每0.3秒检测一次
    global center_history
    while not exit_flag.is_set():
        if frame_queue.empty():
            time.sleep(0.001)
            continue
        frame = frame_queue.get()
        
        # 即使不进行检测，也要传递原始帧以保持显示
        if not detecting.is_set():
            processed_frame = frame.copy()
            fps_info = f"Capture: {capture_fps.get_fps():.1f} Process: {process_fps.get_fps():.1f} Display: {display_fps.get_fps():.1f}"
            cv2.putText(processed_frame, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, "Grabbing...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except:
                    pass
            result_queue.put((processed_frame, None))
            process_fps.update()
            continue
            
        # 检查检测开关
        if not detecting.is_set():
            # 只显示原始画面，不做检测
            processed_frame = frame.copy()
            fps_info = f"Capture: {capture_fps.get_fps():.1f} Process: {process_fps.get_fps():.1f} Display: {display_fps.get_fps():.1f}"
            cv2.putText(processed_frame, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except:
                    pass
            result_queue.put((processed_frame, None))
            process_fps.update()
            continue
        now = time.time()
        if now - last_detect_time < detect_interval:
            # 只显示画面，不做检测
            processed_frame = frame.copy()
            fps_info = f"Capture: {capture_fps.get_fps():.1f} Process: {process_fps.get_fps():.1f} Display: {display_fps.get_fps():.1f}"
            cv2.putText(processed_frame, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except:
                    pass
            result_queue.put((processed_frame, None))
            process_fps.update()
            continue
        last_detect_time = now
        # 在GPU上运行检测
        results = model(frame, device=device, verbose=False)
        move_pending = None
        processed_frame = frame.copy()
        center_point = None
        for res in results:
            for box in res.boxes:
                xyxy = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # 获取类别名
                class_id = int(box.cls[0])
                class_name = res.names[class_id] if hasattr(res, 'names') else str(class_id)
                cv2.putText(processed_frame, f'{class_name} {box.conf[0]:.2f}', 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # 只取第一个目标
                if move_pending is None:
                    move_pending = (x1, y1, x2, y2)
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # 物体静止检测
        static_ready = False
        if center_point is not None:
            center_history.append(center_point)
            if len(center_history) == center_history.maxlen:
                dx = max(abs(center_history[i][0] - center_history[i-1][0]) for i in range(1, len(center_history)))
                dy = max(abs(center_history[i][1] - center_history[i-1][1]) for i in range(1, len(center_history)))
                if dx < 5 and dy < 5:
                    static_ready = True
        else:
            center_history.clear()
        
        # 添加FPS和静止状态到画面
        fps_info = f"Capture: {capture_fps.get_fps():.1f} Process: {process_fps.get_fps():.1f} Display: {display_fps.get_fps():.1f}"
        cv2.putText(processed_frame, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if static_ready:
            cv2.putText(processed_frame, "Ready to Grab", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 更新旧结果
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except:
                pass
        # 只有静止时才允许自动抓取
        result_queue.put((processed_frame, move_pending if static_ready else None))
        process_fps.update()

def safe_shutdown():
    """安全关闭所有设备和进程"""
    log("开始关闭程序...")
    if move is not None:
        try:
            # 停止机械臂当前动作
            move.Sync()
            # 关闭所有IO设备
            dashboard.DOExecute(1, 0)  # 关闭吸气
            dashboard.DOExecute(2, 0)  # 关闭吹气
            # 机械臂回到安全位置
            move.MovJ(180, 0, 30, 0)
            move.Sync()
            log("机械臂已停止")
        except Exception as e:
            log(f"机械臂关闭过程出错：{e}")

def display_loop():
    """显示线程"""
    while not exit_flag.is_set():
        if result_queue.empty():
            time.sleep(0.001)
            continue
            
        processed_frame, move_pending = result_queue.get()
        display_frame = cv2.resize(processed_frame, (680, 480))
        cv2.imshow('YOLO Camera', display_frame)
        display_fps.update()
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            log("检测到退出指令")
            exit_flag.set()
            safe_shutdown()
            break
        
        handle_keyboard_events(key, move_pending)

def handle_keyboard_events(key, move_pending):
    """处理键盘事件"""
    # 自动抓取流程
    if move_pending is not None and not busy and grab_queue.empty():
        grab_queue.put(move_pending)  # 将抓取任务加入队列

def grab_worker():
    """抓取线程的工作函数"""
    global busy
    while not exit_flag.is_set():
        try:
            if grab_queue.empty():
                time.sleep(0.001)
                continue
                
            move_pending = grab_queue.get()
            busy = True
            detecting.clear()  # 只暂停检测，不影响显示
            
            x1, y1, x2, y2 = move_pending
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            log(f"检测到目标中心像素: ({center_x}, {center_y})")
            robot_x, robot_y = useAffineMat(aff_m, center_x, center_y)
            robot_z = -130
            robot_r = 0
            log(f"变换后机械臂坐标: ({robot_x}, {robot_y}, {robot_z}, {robot_r})")
            try:
                # 1. 移动到目标
                result = move.MovJ(robot_x, robot_y, robot_z, robot_r)
                log(f"机械臂回复：{result}")
                move.Sync()
                # 2. 吸气
                result_io = dashboard.DOExecute(1, 1)
                log(f"吸气IO回复：{result_io}")
                time.sleep(1)
                # 3. 移动到复位
                result = move.MovJ(197, -250, 58, 0)
                log(f"机械臂复位回复：{result}")
                move.Sync()
                # 4. 释放
                result_io = dashboard.DOExecute(1, 0)
                log(f"关闭吸气IO回复：{result_io}")
                result_io = dashboard.DOExecute(2, 1)
                log(f"吹气IO回复：{result_io}")
                time.sleep(0.5)
                result_io = dashboard.DOExecute(2, 0)
                log(f"关闭吹气IO回复：{result_io}")
            except Exception as e:
                log(f"机械臂抓取或复位失败：{e}")
            finally:
                busy = False
                detecting.set()  # 恢复检测
        except Exception as e:
            log(f"抓取线程发生错误: {e}")
            busy = False
            detecting.set()

# 主程序
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

# 优化摄像头设置
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置期望的帧率
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 使用MJPG编码

# 启动线程
capture_thread = threading.Thread(target=capture_frames, args=(cap,))
process_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_loop)
grab_thread = threading.Thread(target=grab_worker)  # 新增抓取线程

capture_thread.start()
process_thread.start()
display_thread.start()
grab_thread.start()  # 启动抓取线程

try:
    # 等待线程结束
    capture_thread.join()
    process_thread.join()
    display_thread.join()
    grab_thread.join()  # 等待抓取线程结束
finally:
    # 清理资源
    exit_flag.set()
    safe_shutdown()
    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()
    log("程序已安全退出")
