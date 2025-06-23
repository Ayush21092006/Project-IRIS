import streamlit as st
from datetime import datetime
import numpy as np

def save_to_history(objects, img, caption_text):
    if "history" not in st.session_state:
        st.session_state["history"] = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "time": timestamp,
        "objects": objects,
        "img": img,
        "caption": caption_text
    }

    st.session_state["history"].append(entry)

def load_history():
    return st.session_state.get("history", [])
