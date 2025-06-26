import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import torch
import time
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr
import pyttsx3
import json

# ---- PATHS ----
CUSTOM_MODEL_PATH = "models/pothole_yolov8n.pt"
COCO_MODEL_PATH = "models/coco/yolov8s.pt"
HISTORY_FILE = "detection_history.json"

# ---- LOAD MODELS ----
if not os.path.exists(CUSTOM_MODEL_PATH):
    st.error(f"Custom model not found at: {CUSTOM_MODEL_PATH}")
    st.stop()
if not os.path.exists(COCO_MODEL_PATH):
    st.error(f"COCO model not found at: {COCO_MODEL_PATH}")
    st.stop()

custom_model = YOLO(CUSTOM_MODEL_PATH)
coco_model = YOLO(COCO_MODEL_PATH)

# ---- OCR ----
reader = easyocr.Reader(['en'], gpu=False)

# ---- BLIP ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ---- SPEECH ----
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        st.warning("🗣️ Voice engine busy. Please wait or restart app.")

# ---- CAPTION ----
def get_caption(pil_image):
    inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---- VQA ----
def answer_question(pil_image, question):
    inputs = blip_processor(pil_image, question, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# ---- PAGE UI ----
st.set_page_config(page_title="IRIS: A Smart Vision Platform", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f8fafc;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        padding: 12px 24px;
        background-color: #eff6ff;
        color: #1a1a1a;
        border-radius: 8px 8px 0 0;
    }
    .stButton>button {
        background: linear-gradient(to right, #3b82f6, #06b6d4);
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
    }
    .alert {
        background-color: #e02424;
        color: white;
        padding: 1rem;
        font-weight: bold;
        font-size: 18px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 IRIS: Visual AI for Smart Interaction System ")
st.markdown("<div style='font-size:30px';>📸 SEE. SENSE. UNDERSTAND.Going beyond detection to interaction </div>", unsafe_allow_html=True)

# ---- HISTORY ----
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-5:], f, indent=2)

if "history" not in st.session_state:
    st.session_state.history = load_history()

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Live Camera"])

# ---- IMAGE TAB ----
with tab1:
    uploaded_file = st.file_uploader("📷 Upload an Image for Analysis", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        results_custom = custom_model(image_np)[0]
        results_coco = coco_model(image_np)[0]

        annotated = image_np.copy()
        labels = []

        for r in [results_custom, results_coco]:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]
                if label not in labels:
                    labels.append(label)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                cv2.putText(annotated, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        col1, col2 = st.columns(2)
        col1.image(image_np, caption="Original Image", use_container_width=True)
        col2.image(annotated, caption="Detected Image", use_container_width=True)

        if "pothole" in labels:
            st.markdown('<div class="alert">🚨 ALERT: POTHOLE DETECTED! BE CAREFUL!</div>', unsafe_allow_html=True)

        with st.expander("🔍 OCR Result"):
            ocr_result = reader.readtext(image_np)
            text = "\n".join([item[1] for item in ocr_result]) or "No text detected."
            st.markdown(f'<div style="background-color:#fef3c7;padding:1rem;border-radius:10px;color:black;">{text}</div>', unsafe_allow_html=True)

        with st.expander("🧠 Image Caption"):
            caption = get_caption(image)
            st.markdown(f'<div style="background-color:#fef3c7;padding:1rem;border-radius:10px;color:black;">{caption}</div>', unsafe_allow_html=True)
            if st.button("🔊 Speak Caption"):
                speak_text(caption)

        with st.expander("❓ Ask a Question About the Image"):
            question = st.text_input("Your Question")
            if question:
                answer = answer_question(image, question)
                st.success(f"Answer: {answer}")
                if st.button("🔊 Speak Answer"):
                    speak_text(answer)

        st.session_state.history.append({
            "Mode": "Image",
            "Detected": ", ".join(labels),
            "Caption": caption,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history(st.session_state.history)

        with st.expander("📜 Detection History"):
            st.dataframe(st.session_state.history[-5:])

# ---- VIDEO TAB ----
with tab2:
    st.header("🎞️ Upload a Video for Analysis")
    video_file = st.file_uploader("📼 Choose a video file...", type=["mp4", "mov", "avi"], key="video_uploader")
    if video_file:
        if st.button("▶ Start Video"):
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / fps if fps > 0 else 1 / 24

            stframe = st.empty()
            stop_video = st.button("⛔ Stop Video")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_video:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_custom = custom_model(frame_rgb)[0]
                results_coco = coco_model(frame_rgb)[0]

                for r in [results_custom, results_coco]:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = r.names[cls]
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame_rgb, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                        cv2.putText(frame_rgb, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(delay)
            cap.release()

# ---- LIVE CAMERA TAB ----
with tab3:
    st.header("📷 Live Camera Detection")
    start_button = st.button("▶ Start Live Camera")
    stop_button = st.button("⛔ Stop Camera")

    if start_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_custom = custom_model(frame_rgb)[0]
            results_coco = coco_model(frame_rgb)[0]

            for r in [results_custom, results_coco]:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = r.names[cls]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame_rgb, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                    cv2.putText(frame_rgb, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
        cap.release()
