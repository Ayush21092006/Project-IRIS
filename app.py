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
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
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

# For captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# For VQA
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

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
    inputs = caption_processor(images=pil_image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

# ---- VQA ----
def answer_question(pil_image, question):
    inputs = vqa_processor(images=pil_image, text=question, return_tensors="pt").to(device)
    output_ids = vqa_model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"])
    answer = vqa_processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


# ---- PAGE UI ----
# Load background image using HTML + CSS
import streamlit as st

st.set_page_config(page_title="IRIS: Visual AI", layout="wide")

# Custom CSS Styling
st.markdown("""
<style>
/* Background with gradient overlay */
body {
    background: linear-gradient(to right, rgba(15,23,42,0.8), rgba(30,41,59,0.9)),
            url('download.jpeg');

    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}

/* Global font and spacing */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: white !important;
    font-size: 20px;
    font-weight: 600;
    border-radius: 12px 12px 0 0 !important;
    margin-right: 6px;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #1e293b !important;
    color: #60a5fa !important;
    transform: scale(1.03);
}

/* File uploader */
.stFileUploader {
    background-color: rgba(0, 0, 0, 0.5) !important;
    padding: 1.5rem;
    border-radius: 16px;
    border: 2px dashed #60a5fa;
    box-shadow: 0 0 12px rgba(96, 165, 250, 0.4);
}
.stFileUploader:hover {
    background-color: rgba(30, 41, 59, 0.85) !important;
    border: 2px solid #3b82f6;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #3b82f6, #06b6d4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    transition: 0.3s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(to right, #2563eb, #0891b2);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Beautiful Title Block
st.markdown("""
<div style='text-align:center; padding: 50px 0;'>
    <h1 style='font-size: 52px; font-weight: bold; color: #ffffff; text-shadow: 2px 2px #000;'>🚦 IRIS: Visual AI for Smart Interaction System</h1>
    <p style='font-size: 28px; color: #e2e8f0; font-weight: 500;'>
        📸 <b>SEE. SENSE. UNDERSTAND.</b> <br>Going beyond detection to interaction.
    </p>
</div>
""", unsafe_allow_html=True)


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
                with st.spinner("🤖 Thinking..."):
                    try:
                        answer = answer_question(image, question)
                        st.success(f"🗨️ Answer: {answer}")
                        if st.button("🔊 Speak Answer"):
                            speak_text(answer)
                    except Exception as e:
                        st.error(f"⚠️ Failed to generate answer: {e}")

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
            stframe = st.empty()
            stop_video = st.button("⛔ Stop Video")

            frame_count = 0
            skip_rate = 20

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_video:
                    break

                frame_count += 1
                if frame_count % skip_rate != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (360, 240))   # 👈 Reduce resolution
                results_custom = custom_model.predict(frame_rgb, imgsz=256, conf=0.3)[0]
                results_coco = coco_model.predict(frame_rgb, imgsz=256, conf=0.3)[0]


                for r in [results_custom, results_coco]:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = r.names[cls]
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame_rgb, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                        cv2.putText(frame_rgb, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                stframe.image(frame_rgb, channels="RGB", width=800)

                # time.sleep(0.0001)  # Or completely remove it
            cap.release()
            st.success("✅ Video processing complete or stopped.")

# ---- LIVE CAMERA TAB ----
with tab3:
    st.header("📷 Live Camera Detection")
    start_button = st.button("▶ Start Live Camera")
    stop_button = st.button("⛔ Stop Camera")

    if start_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        frame_count = 0
        skip_rate = 10
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame_count += 1
            if frame_count % skip_rate != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (360, 240)) # 👈 Reduce resolution
            results_custom = custom_model.predict(frame_rgb, imgsz=256, conf=0.3)[0]
            results_coco = coco_model.predict(frame_rgb, imgsz=256, conf=0.3)[0]


            for r in [results_custom, results_coco]:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = r.names[cls]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame_rgb, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                    cv2.putText(frame_rgb, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame_rgb, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            stframe.image(frame_rgb, channels="RGB", width=800)
            # time.sleep(0.001)

        cap.release()
        st.success("✅ Camera stopped")
