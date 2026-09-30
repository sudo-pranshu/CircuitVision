"""Component detection with YOLO (ultralytics).

Train (Colab GPU or Mac MPS):
  yolo detect train data=data/yolo/dataset.yaml model=yolov8s.pt imgsz=1024 epochs=100 device=mps
"""
from .classes import CLASSES
from .types import Detection


class Detector:
    def __init__(self, weights: str, conf: float = 0.35):
        from ultralytics import YOLO  # lazy: heavy import
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, img) -> list[Detection]:
        r = self.model.predict(img, conf=self.conf, verbose=False)[0]
        names = r.names
        out = []
        for box, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            cls = names[int(c)]
            if cls in CLASSES:
                out.append(Detection(cls, tuple(int(v) for v in box), float(p)))
        return out
