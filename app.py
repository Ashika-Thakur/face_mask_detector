"""
Face-Mask Detection — Streamlit app (FIXED)

Key fixes:
1. IMG_SIZE = 224 (matches training)
2. Face padding added (important for mask detection)
3. RGB consistency maintained
4. Correct class mapping
5. Optional test-time augmentation
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# Face detector
# -------------------------------
USE_MEDIAPIPE = True
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4
    )
except Exception:
    USE_MEDIAPIPE = False
    HAAR = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_faces(rgb_img):
    """Return list of (x, y, w, h) bounding boxes."""
    h, w = rgb_img.shape[:2]
    boxes = []

    if USE_MEDIAPIPE:
        res = mp_face.process(rgb_img)
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x = int(bb.xmin * w)
                y = int(bb.ymin * h)
                bw = int(bb.width * w)
                bh = int(bb.height * h)
                boxes.append((x, y, bw, bh))
    else:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        dets = HAAR.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        boxes = [tuple(map(int, b)) for b in dets]

    return boxes


def pad_box(x, y, w, h, img_w, img_h, pad_ratio=0.30):
    """Expand bounding box for better mask visibility."""
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return x1, y1, x2, y2


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Face Mask Detection", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #eaeaea; }
.stButton>button {
    background-color:#ff4b4b;
    color:#fff;
    border:0;
    padding:8px 20px;
    border-radius:6px;
}
</style>
""", unsafe_allow_html=True)

st.title("😷 Face Mask Detection")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector.model")

model = load_model()

IMG_SIZE = 224
CLASS_NAMES = ["Mask", "No Mask"]

# Sidebar controls
THRESHOLD = st.sidebar.slider(
    "Decision threshold for 'No Mask'",
    0.10, 0.90, 0.50, 0.05
)

TTA = st.sidebar.checkbox(
    "Test-time augmentation (flip)",
    value=True
)

# -------------------------------
# Prediction
# -------------------------------
def preprocess(face_rgb):
    face = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32")
    face = preprocess_input(face)
    return np.expand_dims(face, axis=0)


def predict(face_rgb):
    x = preprocess(face_rgb)
    prob = float(model.predict(x, verbose=0)[0][0])

    if TTA:
        x_flip = preprocess(cv2.flip(face_rgb, 1))
        prob = 0.5 * (prob + float(model.predict(x_flip, verbose=0)[0][0]))

    if prob > THRESHOLD:
        return "Mask", prob
    else:
        return "No Mask", 1.0 - prob


def annotate(img_rgb, boxes):
    h, w = img_rgb.shape[:2]

    for (bx, by, bw, bh) in boxes:
        x1, y1, x2, y2 = pad_box(bx, by, bw, bh, w, h)
        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        label, conf = predict(crop)

        color = (0, 200, 0) if label == "Mask" else (220, 40, 40)

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img_rgb, (x1, max(y1 - 25, 0)), (x2, y1), color, cv2.FILLED)
        cv2.putText(
            img_rgb,
            f"{label} {conf*100:.1f}%",
            (x1 + 5, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return img_rgb


# -------------------------------
# UI Tabs
# -------------------------------
tab1, tab2 = st.tabs(["📁 Upload", "📷 Webcam"])

# Upload tab
with tab1:
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if file and st.button("Detect"):
        img = np.array(Image.open(file).convert("RGB"))
        boxes = detect_faces(img)

        if not boxes:
            st.warning("No face detected")
            st.image(img)
        else:
            result = annotate(img.copy(), boxes)
            st.image(result)
            st.success(f"Faces detected: {len(boxes)}")

# Webcam tab
with tab2:
    run = st.checkbox("Start Camera")

    FRAME = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detect_faces(rgb)
            out = annotate(rgb, boxes)

            FRAME.image(out)

        cap.release()