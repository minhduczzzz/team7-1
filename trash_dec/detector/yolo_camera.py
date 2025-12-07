import cv2
from ultralytics import YOLO

class YOLOCamera:
    def __init__(self, model_path="detector/models/best.pt", src=0, conf=0.75, imgsz=640):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(src)
        self.conf = conf
        self.imgsz = imgsz

        # Tracking mặc định: ByteTrack
        self.tracker = self.model.track(
            source=0,
            stream=True,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            verbose=False
        )

    def get_annotated_frame(self, show_conf=True):
        """
        Lấy frame đã detect + track.
        """
        try:
            result = next(self.tracker)  # lấy frame tiếp theo
        except StopIteration:
            return None

        if result is None or result.orig_img is None:
            return None

        # Vẽ box + ID + conf
        frame = result.plot(conf=show_conf)
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
