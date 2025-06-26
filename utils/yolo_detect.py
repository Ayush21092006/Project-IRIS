from ultralytics import YOLO
import os
import cv2

def load_yolo_models():
    models = {}
    paths = {
        "custom": "models/custom/best.pt",
        "coco": "models/coco/yolov8s.pt",
        "pothole": "models/pothole_yolov8n.pt"
    }

    for key, path in paths.items():
        if os.path.exists(path):
            model = YOLO(path)
            models[key] = model
            print(f"✅ Loaded {key} model with classes:", model.names)

    return models.get("custom"), models.get("coco"), models.get("pothole")


def detect_objects(image, custom_model=None, coco_model=None, pothole_model=None, conf=0.5):
    results_combined = []
    labels_combined = []

    for model in [custom_model, coco_model, pothole_model]:
        if not model:
            continue
        results = model(image, conf=conf)[0]
        if results.boxes:
            class_names = model.names
            labels = [class_names[int(cls)] for cls in results.boxes.cls]
            results_combined.append(results)
            labels_combined.extend(labels)

    return results_combined, labels_combined


def draw_and_alert(image, results_list, violation_labels, alert_callback, show_warning=False):
    alerted = False
    for result in results_list:
        class_names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0)

            if label == "pothole":
                color = (0, 0, 255)
                if not alerted and show_warning:
                    st.warning("⚠️ Pothole detected, be careful!")
                    alert_callback()
                    alerted = True
            elif label in violation_labels:
                color = (0, 0, 255)
                alert_callback()

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf_score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

