import cv2
import threading
from ultralytics import YOLO
import torch

class YOLOStream:
    def __init__(self, camera_index=0, width=2592, height=1944):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"摄像头 {camera_index} 打开失败！")
        else:
            print(f"摄像头 {camera_index} 打开成功！")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.model = YOLO(r'F:\KeChaung\yolo_pth\yolov8m.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("未获取到摄像头画面")
                continue
            results = self.model(frame, device=self.device, verbose=False)
            annotated = frame.copy()
            for res in results:
                for box in res.boxes:
                    xyxy = box.xyxy.squeeze().tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    class_id = int(box.cls[0])
                    class_name = res.names[class_id] if hasattr(res, 'names') else str(class_id)
                    cv2.putText(annotated, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.frame = annotated

    def get_frame(self, width=640, height=480):
        if self.frame is None:
            return None
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (int(width), int(height)))
        return rgb

    def release(self):
        self.running = False
        self.cap.release() 