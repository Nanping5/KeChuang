from flask import Flask, Response, jsonify
import cv2
from yolo_stream import YOLOStream
import threading
import time

app = Flask(__name__)
yolo_stream = YOLOStream()

# 全局检测结果
latest_detection = {
    "boxes": [],
    "frame_shape": None,
    "timestamp": None
}
detection_lock = threading.Lock()

def yolo_detect_loop():
    global latest_detection
    while True:
        frame = yolo_stream.get_frame(2592, 1944)
        if frame is None:
            time.sleep(0.05)
            continue
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_stream.model(frame, device=yolo_stream.device, verbose=False)
        boxes = []
        for res in results:
            for box in res.boxes:
                xyxy = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = res.names[class_id] if hasattr(res, 'names') else str(class_id)
                boxes.append({
                    "xyxy": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": class_name,
                    "conf": conf
                })
        print("后端检测到类别：", [b["class_name"] for b in boxes])
        with detection_lock:
            latest_detection = {
                "boxes": boxes,
                "frame_shape": frame.shape,
                "timestamp": time.time()
            }
        time.sleep(0.05)  # 控制检测频率

# 启动独立检测线程
detect_thread = threading.Thread(target=yolo_detect_loop, daemon=True)
detect_thread.start()

def gen():
    while True:
        frame = yolo_stream.get_frame(2592, 1944)
        if frame is None:
            continue
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 画框（用latest_detection）
        with detection_lock:
            boxes = latest_detection["boxes"]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_bgr, f'{box["class_name"]} {box["conf"]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        thumb = cv2.resize(frame_bgr, (640, 480))
        ret, jpeg = cv2.imencode('.jpg', thumb)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_result')
def detect_result():
    with detection_lock:
        return jsonify(latest_detection)

@app.route('/video_feed_thumb')
def video_feed_thumb():
    def gen_thumb():
        while True:
            frame = yolo_stream.get_frame(2592, 1944)
            if frame is None:
                continue
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            thumb = cv2.resize(frame_bgr, (640, 480))
            # 画框（用latest_detection）
            with detection_lock:
                boxes = latest_detection["boxes"]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box["xyxy"])
                cv2.rectangle(thumb, (x1*640//2592, y1*480//1944), (x2*640//2592, y2*480//1944), (0, 255, 255), 2)
                cv2.putText(thumb, f'{box["class_name"]} {box["conf"]:.2f}', (x1*640//2592, y1*480//1944 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', thumb)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(gen_thumb(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True) 