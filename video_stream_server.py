from flask import Flask, Response
import cv2
from yolo_stream import YOLOStream

app = Flask(__name__)
yolo_stream = YOLOStream()

def gen():
    while True:
        frame = yolo_stream.get_frame(640, 480)
        if frame is None:
            continue
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame_bgr)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True) 