import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained outfit classifier
def load_outfit_classifier():
    return load_model("models/outfit_classifier.h5")

# Predict outfit from cropped person image
def predict_outfit(image, model):
    if image is None or image.size == 0:
        return "Unknown"
    
    try:
        image = cv2.resize(image, (128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = model.predict(image)[0]
        class_index = np.argmax(prediction)

        labels = ["Casual", "Formal", "Sportswear", "Traditional"]  # Your class labels
        return labels[class_index] if class_index < len(labels) else "Unknown"
    except Exception as e:
        print("Outfit prediction error:", e)
        return "Unknown"
