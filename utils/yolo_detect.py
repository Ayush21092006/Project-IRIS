from ultralytics import YOLO
import os

def load_yolo_model():
    custom_model = "iris_best.pt"
    coco_model = "yolov8s.pt"

    if os.path.exists(custom_model):
        return YOLO(custom_model), "custom"
    elif os.path.exists(coco_model):
        return YOLO(coco_model), "coco"
    else:
        return None, None

def detect_objects(image, model, conf=0.5):
    if model is None:
        return None, []
    results = model(image, conf=conf)
    result = results[0]
    labels = [result.names[int(box.cls[0])] for box in result.boxes]
    return result, labels
