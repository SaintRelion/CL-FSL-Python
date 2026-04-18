import cv2
import mediapipe as mp
import csv
import time
import os
import numpy as np
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
DATA_FILE = "data/hand_vocabulary.csv"
SAMPLES_PER_SIGN = 30
MODEL_PATH = "models/hand_landmarker.task"

if not os.path.exists("data"):
    os.makedirs("data")

# --- INITIALIZE MEDIAPIPE TASK ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)


def save_snapshot(label, landmarks):
    # --- 1. SCALE ANCHOR ---
    wrist = landmarks[0]
    m_knuckle = landmarks[9]  # Middle finger base
    hand_scale = ((m_knuckle.x - wrist.x) ** 2 + (m_knuckle.y - wrist.y) ** 2) ** 0.5
    hand_scale = max(hand_scale, 1e-6)

    data = []

    # --- 2. NORMALIZED COORDINATES (42 values) ---
    for lm in landmarks:
        data.append((lm.x - wrist.x) / hand_scale)
        data.append((lm.y - wrist.y) / hand_scale)

    # --- 3. FINGER EXTENSION RATIOS (5 values) ---
    tips = [4, 8, 12, 16, 20]
    knuckles = [2, 5, 9, 13, 17]
    for t_idx, k_idx in zip(tips, knuckles):
        t, k = landmarks[t_idx], landmarks[k_idx]
        dist_tip = ((t.x - wrist.x) ** 2 + (t.y - wrist.y) ** 2) ** 0.5
        dist_knuckle = ((k.x - wrist.x) ** 2 + (k.y - wrist.y) ** 2) ** 0.5
        data.append(dist_tip / max(dist_knuckle, 1e-6))

    # --- 4. INTER-TIP DISTANCES (4 values) ---
    # Index-Middle, Middle-Ring, Ring-Pinky, Thumb-Index
    tip_pairs = [(8, 12), (12, 16), (16, 20), (4, 8)]
    for p1, p2 in tip_pairs:
        t1, t2 = landmarks[p1], landmarks[p2]
        dist = ((t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2) ** 0.5
        # Normalize by hand_scale so "pinching" is distance-independent
        data.append(dist / hand_scale)

    with open(DATA_FILE, "a", newline="") as f:
        csv.writer(f).writerow([label] + data)


cap = cv2.VideoCapture(0)
current_label = ""
is_collecting = False
sample_count = 0
last_snapshot_time = 0

print("--- HAND SHAPE COLLECTOR READY ---")
print("1. Click the CAMERA WINDOW to focus it.")
print("2. Type the name of your Jutsu.")
print("3. Press 'Enter' or 'Space' to begin.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Task API requires mp.Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    # --- UI HUD ---
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"SIGN: {current_label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    status_text = f"READY (Press ENTER)"
    status_color = (255, 255, 0)

    if is_collecting:
        status_text = f"RECORDING: {sample_count}/{SAMPLES_PER_SIGN}"
        status_color = (0, 0, 255)

    cv2.putText(
        frame, status_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1
    )

    # --- LANDMARK LOGIC ---
    if results.hand_landmarks:
        lms = results.hand_landmarks[0]

        # 1. Draw Bounding Box
        xs, ys = [lm.x for lm in lms], [lm.y for lm in lms]
        xmin, ymin = int(min(xs) * w) - 15, int(min(ys) * h) - 15
        xmax, ymax = int(max(xs) * w) + 15, int(max(ys) * h) + 15
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 2. Draw Neural Web Points (Wrist + Tips)
        # These match the indices used in your PredictorHandler feature logic
        web_indices = [0, 4, 8, 12, 16, 20]
        for idx in web_indices:
            point = lms[idx]
            px, py = int(point.x * w), int(point.y * h)

            # Draw Wrist in a different color (Yellow)
            color = (0, 255, 255) if idx == 0 else (255, 0, 255)
            cv2.circle(frame, (px, py), 5, color, -1)

            # Optional: Draw lines from wrist to tips to see the "Web"
            if idx != 0:
                wrist = lms[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.line(frame, (wx, wy), (px, py), (255, 255, 255), 1)

        # 3. Collection Logic
        if is_collecting:
            if time.time() - last_snapshot_time > 0.2:
                save_snapshot(current_label, lms)
                sample_count += 1
                last_snapshot_time = time.time()
                if sample_count >= SAMPLES_PER_SIGN:
                    is_collecting = False
                    print(f"Captured {current_label}!")

    cv2.imshow("Jutsu Collector", frame)

    # --- ROBUST KEY LISTENER ---
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to quit
        break

    elif key in [13, 10]:  # ENTER
        if current_label.strip() != "":
            is_collecting = True
            sample_count = 0
            last_snapshot_time = time.time()

    elif key == 8:  # BACKSPACE
        current_label = current_label[:-1]

    # ONLY accept printable characters (Space is 32, Z is 122)
    # This prevents the string from clearing itself when 'no key' (255) is detected
    elif 32 <= key <= 126:
        char = chr(key).upper()
        current_label += char

cap.release()
cv2.destroyAllWindows()
