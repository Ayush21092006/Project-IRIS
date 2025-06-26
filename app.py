import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr
import pyttsx3
from datetime import datetime

# ---- MODEL PATHS ----
CUSTOM_MODEL_PATH = "models/pothole_yolov8n.pt"
COCO_MODEL_PATH = "models/coco/yolov8s.pt"

# ---- CHECK & LOAD MODELS ----
if not os.path.exists(CUSTOM_MODEL_PATH):
    st.error(f"Custom model not found at: {CUSTOM_MODEL_PATH}")
    st.stop()
if not os.path.exists(COCO_MODEL_PATH):
    st.error(f"COCO model not found at: {COCO_MODEL_PATH}")
    st.stop()

custom_model = YOLO(CUSTOM_MODEL_PATH)
coco_model = YOLO(COCO_MODEL_PATH)

# ---- INIT OCR & BLIP ----
reader = easyocr.Reader(['en'], gpu=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ---- SPEAK ----
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
    return blip_processor.decode(out[0], skip_special_tokens=True)

# ---- VQA ----
def answer_question(pil_image, question):
    inputs = blip_processor(pil_image, question, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# ---- STYLING ----
st.set_page_config(page_title="IRIS: A Smart Vision Platform", layout="wide")
st.markdown("""<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f4f6f9;
}
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    padding: 12px 24px;
    background-color: #e6f0fb;
    color: #1a1a1a;
    border-radius: 8px 8px 0 0;
}
.stButton>button {
    background: linear-gradient(to right, #3b82f6, #06b6d4);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
}
.alert {
    animation: blink 1s infinite;
    color: white;
    background-color: #e02424;
    font-size: 20px;
    font-weight: bold;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 10px rgba(224,36,36,0.6);
}
.caption-box, .ocr-box {
    background-color: #fef3c7;
    padding: 1rem;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 500;
    color: #111827;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.4; }
    100% { opacity: 1; }
}
</style>""", unsafe_allow_html=True)

st.title("🚦 IRIS: A Smart Vision Platform")
st.markdown("📸 Real-Time Detection, Captioning, OCR, and VQA")

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

        st.markdown("### 🧾 Image Comparison")
        col1, col2 = st.columns(2)
        col1.image(image_np, caption="📤 Uploaded Image", use_container_width=True)
        col2.image(annotated, caption="🧠 Detected Image", use_container_width=True)

        if "pothole" in labels:
            st.markdown('<div class="alert">🚨 ALERT: POTHOLE DETECTED! BE CAREFUL!</div>', unsafe_allow_html=True)

        with st.expander("🔍 OCR Result"):
            ocr_result = reader.readtext(image_np)
            text = "\n".join([item[1] for item in ocr_result]) or "No text detected."
            st.markdown(f'<div class="ocr-box">{text}</div>', unsafe_allow_html=True)

        with st.expander("🧠 Image Caption"):
            caption = get_caption(image)
            st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)
            if st.button("🔊 Speak Caption"):
                speak_text(caption)


        with st.expander("❓ Ask a Question About the Image"):
            question = st.text_input("Your Question")
            if question:
                answer = answer_question(image, question)
                st.success(f"Answer: {answer}")
                if st.button("🔊 Speak Answer"):
                    speak_text(answer)

        # ---- HISTORY: Last 5 only
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Mode": "Image",
            "Detected": ", ".join(labels),
            "Caption": caption,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.history = st.session_state.history[-5:]

        with st.expander("📜 Detection History (Last 5 Images)"):
            st.dataframe(st.session_state.history)

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
            cap.release()

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
    st.header("📷 Start Live Camera Detection")
    if st.button("🎥 Start Camera"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_live = st.button("⛔ Stop Camera")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_live:
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
