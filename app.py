import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import pyttsx3
import pygame
import time
import os
import easyocr

from utils.yolo_detect import load_yolo_model, detect_objects
from utils.captioning import load_caption_model, generate_caption
from utils.history import save_to_history, load_history

# ======================
# INIT
# ======================
pygame.mixer.init()
engine = pyttsx3.init()
ocr_reader = easyocr.Reader(['en'])
ALERT_SOUND = "alert.wav"
confidence = 0.5
alert_mode = "Both"

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        pass

def play_alert():
    if os.path.exists(ALERT_SOUND):
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()

def stop_alert():
    pygame.mixer.music.stop()
    engine.stop()

def extract_text_easyocr(image_np):
    results = ocr_reader.readtext(image_np)
    return " ".join([text for _, text, _ in results])

# ======================
# STREAMLIT APP CONFIG
# ======================
st.set_page_config(page_title="IRIS: A Smart Vision", layout="wide", page_icon="🧠")
st.markdown(
    "<h1 style='text-align: center; font-size: 60px; color:#4B8BBE;'>🧠 IRIS — A Smart Vision System</h1>",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_models():
    yolo_model, model_type = load_yolo_model()
    caption_processor, caption_model = load_caption_model()
    return yolo_model, model_type, caption_processor, caption_model

yolo_model, model_type, caption_processor, caption_model = load_models()

if yolo_model is None:
    st.warning("⚠ No YOLO model found. Place iris_best.pt or yolov8n.pt in the root folder.")
else:
    st.success(f"✅ Using {'Custom IRIS Model' if model_type == 'custom' else 'COCO Pretrained Model'}")

tab1, tab2, tab3 = st.tabs(["📷 Image", "🎥 Video", "🔴 Live Camera"])

# -----------------------
# IMAGE TAB
# -----------------------
with tab1:
    st.header("Image Analysis")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        np_img = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)

        with st.spinner("Analyzing..."):
            result, labels = detect_objects(np_img, yolo_model, conf=confidence)
            img_with_boxes = np_img.copy()

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, f"{label} {conf_score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            caption = generate_caption(image, caption_processor, caption_model)
            detected_text = extract_text_easyocr(np_img)

        with col2:
            st.image(img_with_boxes, caption="Detections", use_column_width=True)

        if caption:
            st.markdown(f"<h3 style='color:#006400;'>📝 Scene Caption:</h3><p style='font-size:20px'>{caption}</p>", unsafe_allow_html=True)

        if detected_text:
            st.markdown(f"<h3 style='color:#8B0000;'>🔡 OCR Text Detected:</h3><p style='font-size:20px'>{detected_text}</p>", unsafe_allow_html=True)

        if st.button("🔊 Speak Caption and Text", key="speak_caption_image"):
            speak(caption)
            if detected_text:
                speak("Detected text: " + detected_text)

        save_to_history(objects=labels, img=np_img, caption_text=caption)

# -----------------------
# VIDEO TAB
# -----------------------
with tab2:
    st.header("Video Analysis")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file and yolo_model:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        cap = cv2.VideoCapture(temp_file.name)
        st_frame = st.empty()
        stop_btn = st.button("⏹ Stop", key="stop_video")

        last_frame = None

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, labels = detect_objects(frame_rgb, yolo_model, conf=confidence)

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st_frame.image(frame, channels="BGR", use_column_width=True)

        cap.release()

        if last_frame is not None:
            st.success("✅ Video Analysis Complete")
            caption = generate_caption(Image.fromarray(last_frame), caption_processor, caption_model)
            detected_text = extract_text_easyocr(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))

            st.markdown(f"<h3 style='color:#006400;'>📝 Final Summary:</h3><p style='font-size:20px'>{caption}</p>", unsafe_allow_html=True)

            if detected_text:
                st.markdown(f"<h3 style='color:#8B0000;'>🔡 OCR Text Detected:</h3><p style='font-size:20px'>{detected_text}</p>", unsafe_allow_html=True)

            if st.button("🔊 Speak Final Summary", key="speak_summary_video"):
                speak(caption)
                if detected_text:
                    speak("Detected text: " + detected_text)

            save_to_history(objects=labels, img=last_frame, caption_text=caption)

# -----------------------
# LIVE CAMERA TAB
# -----------------------
with tab3:
    st.header("Live Camera")
    run = st.checkbox("Start Camera")
    cam_window = st.empty()

    if run and yolo_model:
        cam = cv2.VideoCapture(0)
        last_alert_time = 0

        while run:
            ret, frame = cam.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, labels = detect_objects(frame_rgb, yolo_model, conf=confidence)

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cam_window.image(frame, channels="BGR", use_column_width=True)

            if st.button("⏹ Stop", key="stop_live"):
                run = False

        cam.release()

# -----------------------
# HISTORY
# -----------------------
st.markdown("---")
st.subheader("🕒 Detection History (Last 5)")
history = load_history()
if history:
    for entry in history[-5:][::-1]:
        st.image(entry['img'], width=400)
        st.markdown(f"🕓 *{entry['time']}*")
        st.success(f"Caption: {entry['caption']}")
        st.markdown("---")
else:
    st.info("No history yet.")
