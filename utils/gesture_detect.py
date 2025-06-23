
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Very basic gesture recognition
def detect_gesture(image):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    gesture_label = "No gesture"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_tips = [hand_landmarks.landmark[i].y for i in [8, 12, 16, 20]]
            wrist = hand_landmarks.landmark[0].y
            if all(tip < wrist for tip in finger_tips):
                gesture_label = "Open Palm"
            elif all(tip > wrist for tip in finger_tips):
                gesture_label = "Fist"
            else:
                gesture_label = "Unknown"

    return gesture_label, image
