import av
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase

class VideoCaptureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img
