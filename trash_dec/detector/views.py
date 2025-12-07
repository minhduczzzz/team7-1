from django.http import StreamingHttpResponse
import cv2
from detector.yolo_camera import YOLOCamera  # import class YOLOCamera đã tối ưu

# Khởi tạo YOLOCamera (chỉ khởi tạo 1 lần)
yolo_cam = YOLOCamera(model_path="detector/models/best.pt", src=0, conf=0.45, imgsz=640)


def gen_frames():
    while True:
        frame = yolo_cam.get_annotated_frame(show_conf=True)
        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def camera(request):
    """
    Django view trả về video stream trực tiếp với YOLO detection.
    """
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
