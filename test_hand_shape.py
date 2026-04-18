import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = "models/hand_shape_model.tflite"
LABELS_PATH = "models/hand_shape_labels.txt"
LANDMARKER_PATH = "models/hand_landmarker.task"

# 1. Load Labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 2. Initialize TFLite Interpreter (Correct Way)
# We use the standard tf.lite runtime to avoid the mp.tasks.core error
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Initialize MediaPipe Hand Landmarker
# Access BaseOptions directly from mp.tasks as in your collector script
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)


def extract_features(landmarks):
    # --- MIRROR COLLECTOR LOGIC ---
    wrist = landmarks[0]
    m_knuckle = landmarks[9]
    hand_scale = ((m_knuckle.x - wrist.x) ** 2 + (m_knuckle.y - wrist.y) ** 2) ** 0.5
    hand_scale = max(hand_scale, 1e-6)

    data = []
    # 2. Normalized Coordinates (42)
    for lm in landmarks:
        data.append((lm.x - wrist.x) / hand_scale)
        data.append((lm.y - wrist.y) / hand_scale)

    # 3. Extension Ratios (5)
    tips = [4, 8, 12, 16, 20]
    knuckles = [2, 5, 9, 13, 17]
    for t_idx, k_idx in zip(tips, knuckles):
        t, k = landmarks[t_idx], landmarks[k_idx]
        dist_tip = ((t.x - wrist.x) ** 2 + (t.y - wrist.y) ** 2) ** 0.5
        dist_knuckle = ((k.x - wrist.x) ** 2 + (k.y - wrist.y) ** 2) ** 0.5
        data.append(dist_tip / max(dist_knuckle, 1e-6))

    # 4. Inter-tip Distances (4)
    tip_pairs = [(8, 12), (12, 16), (16, 20), (4, 8)]
    for p1, p2 in tip_pairs:
        t1, t2 = landmarks[p1], landmarks[p2]
        dist = ((t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2) ** 0.5
        data.append(dist / hand_scale)

    return np.array([data], dtype=np.float32)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        lms = results.hand_landmarks[0]

        # Draw Bounding Box and Web
        xs, ys = [lm.x for lm in lms], [lm.y for lm in lms]
        cv2.rectangle(
            frame,
            (int(min(xs) * w) - 15, int(min(ys) * h) - 15),
            (int(max(xs) * w) + 15, int(max(ys) * h) + 15),
            (0, 255, 0),
            2,
        )

        # 1. Extract Features
        features = extract_features(lms)

        # 2. Run Inference
        interpreter.set_tensor(input_details[0]["index"], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])[0]

        # 3. Get Result
        max_idx = np.argmax(prediction)
        confidence = prediction[max_idx]
        label = labels[max_idx] if confidence > 0.7 else "UNRECOGNIZED"

        # UI Overlay
        color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
        cv2.putText(
            frame,
            f"{label} ({confidence*100:.1f}%)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    cv2.imshow("Jutsu Tester", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
