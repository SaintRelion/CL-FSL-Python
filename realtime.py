# python
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import vision
import os
import time
import json

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"
PILLAR_MODEL_PATH = "models/pillar_path_model.tflite"
ANATOMY_PATH = "data/gesture_anatomy.json"
SHAPE_LABELS_PATH = "models/hand_shape_labels.txt"

# 1. Load All Possible Gesture Labels
GESTURE_LABELS = sorted(
    [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
)

# 2. Load Anatomy Map for Culling
with open(ANATOMY_PATH, "r") as f:
    ANATOMY_MAP = json.load(f)

# 3. Load Shape Labels
with open(SHAPE_LABELS_PATH, "r") as f:
    SHAPE_LABELS = [line.strip() for line in f.readlines()]

# --- STATE ---
sequence_buffer = []
is_collecting = False
start_time = 0
COLLECTION_DURATION = 1.7
TARGET_FRAMES = 30
active_hand_shapes = [-1, -1]
last_prediction = ""
prediction_expiry = 0


# --- INITIALIZATION ---
def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


shape_interp, shape_in, shape_out = load_tflite(SHAPE_MODEL_PATH)
path_interp, path_in, path_out = load_tflite(PILLAR_MODEL_PATH)
NUM_SHAPE_CLASSES = shape_out[0]["shape"][-1]

BaseOptions = mp.tasks.BaseOptions
h_mod = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2,
    )
)
p_mod = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task")
    )
)
f_mod = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task")
    )
)


def get_shape_id(lms):
    wrist, m_knk = lms[0], lms[9]
    scale = max(((m_knk.x - wrist.x) ** 2 + (m_knk.y - wrist.y) ** 2) ** 0.5, 1e-6)
    feats = []
    for lm in lms:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
    for t, k in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        feats.append(
            (((lms[t].x - wrist.x) ** 2 + (lms[t].y - wrist.y) ** 2) ** 0.5)
            / max(
                (((lms[k].x - wrist.x) ** 2 + (lms[k].y - wrist.y) ** 2) ** 0.5), 1e-6
            )
        )
    for p1, p2 in [(8, 12), (12, 16), (16, 20), (4, 8)]:
        feats.append(
            (((lms[p1].x - lms[p2].x) ** 2 + (lms[p1].y - lms[p2].y) ** 2) ** 0.5)
            / scale
        )
    shape_interp.set_tensor(shape_in[0]["index"], np.array([feats], dtype=np.float32))
    shape_interp.invoke()
    return np.argmax(shape_interp.get_tensor(shape_out[0]["index"])[0])


# --- DYNAMIC GRID UI ---
def draw_label_grid(frame, labels, active_h_ids):
    h, w, _ = frame.shape
    start_x, start_y = 30, 80
    col_width = 220
    row_height = 25
    rows_per_col = (h - 150) // row_height

    legal_labels = []

    # Draw Semi-Transparent Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 60), (w - 10, h - 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, label in enumerate(labels):
        col = i // rows_per_col
        row = i % rows_per_col

        x = start_x + (col * col_width)
        y = start_y + (row * row_height)

        req = ANATOMY_MAP.get(label, {"h1_required": -1, "h2_required": -1})
        is_legal = (
            req["h1_required"] == active_h_ids[0]
            and req["h2_required"] == active_h_ids[1]
        )

        color = (
            (0, 255, 0) if is_legal else (100, 100, 100)
        )  # Green if active, Grey if culled
        prefix = ">> " if is_legal else "   "

        cv2.putText(
            frame, f"{prefix}{label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
        if is_legal:
            legal_labels.append(label)

    return legal_labels


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    h_res, p_res, f_res = (
        h_mod.detect(mp_img),
        p_mod.detect(mp_img),
        f_mod.detect(mp_img),
    )

    f_scale = 0.2
    cur_pillars = [
        [0.5, 0.4],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.7, 0.7],
        [0.5, 0.7],
        [0.4, 0.3],
        [0.6, 0.3],
    ]
    if f_res.face_landmarks:
        fl = f_res.face_landmarks[0]
        f_scale = max(abs(fl[10].y - fl[152].y), 1e-6)
        cur_pillars[0], cur_pillars[1] = [fl[1].x, fl[1].y], [
            (fl[13].x + fl[14].x) / 2,
            (fl[13].y + fl[14].y) / 2,
        ]
        cur_pillars[5], cur_pillars[6] = [fl[234].x, fl[234].y], [fl[454].x, fl[454].y]

    if p_res.pose_landmarks:
        pl = p_res.pose_landmarks[0]
        cur_pillars[2], cur_pillars[3] = [pl[11].x, pl[11].y], [pl[12].x, pl[12].y]
        cur_pillars[4] = [(pl[11].x + pl[12].x) / 2, (pl[11].y + pl[12].y) / 2]

    # Hand Logic
    hand_detected = h_res.hand_landmarks is not None and len(h_res.hand_landmarks) > 0
    current_feat_vector = []
    active_hand_shapes = [-1, -1]

    if hand_detected:
        if not is_collecting:
            is_collecting, start_time, sequence_buffer = True, time.time(), []

        for i in range(2):
            if i < len(h_res.hand_landmarks):
                lms = h_res.hand_landmarks[i]
                active_hand_shapes[i] = get_shape_id(lms)
                wrist = np.array([lms[0].x, lms[0].y])
                tips = np.mean(
                    [[lms[t].x, lms[t].y] for t in [4, 8, 12, 16, 20]], axis=0
                )
                w_dists = [
                    np.linalg.norm(wrist - np.array(p)) / f_scale for p in cur_pillars
                ]
                t_dists = [
                    np.linalg.norm(tips - np.array(p)) / f_scale for p in cur_pillars
                ]
                current_feat_vector += w_dists + t_dists + [1.0]
                for lm in lms:
                    cv2.circle(
                        frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1
                    )
            else:
                current_feat_vector += [0.0] * 15  # 7 Wrist + 7 Tip + 1 Presence
        sequence_buffer.append(current_feat_vector)
    else:
        is_collecting = False

    # LAYER 1: Dynamic Grid UI
    legal_labels = draw_label_grid(frame, GESTURE_LABELS, active_hand_shapes)

    # Header Info
    cv2.putText(
        frame,
        f"ACTIVE SHAPE: {SHAPE_LABELS[active_hand_shapes[0]] if active_hand_shapes[0] != -1 else 'NONE'}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # LAYER 2: Trigger Prediction
    if is_collecting:
        elapsed = time.time() - start_time
        cv2.rectangle(
            frame,
            (0, h - 10),
            (int(w * (elapsed / COLLECTION_DURATION)), h),
            (0, 255, 255),
            -1,
        )

        if elapsed >= COLLECTION_DURATION:
            from fsl_helper import resample_sequence

            final_seq = resample_sequence(sequence_buffer, TARGET_FRAMES)
            path_interp.set_tensor(
                path_in[0]["index"], np.array([final_seq], dtype=np.float32)
            )
            path_interp.invoke()
            raw_probs = path_interp.get_tensor(path_out[0]["index"])[0]

            # THE CULL
            best_idx = -1
            max_p = -1
            for i, label in enumerate(GESTURE_LABELS):
                if label in legal_labels:
                    if raw_probs[i] > max_p:
                        max_p = raw_probs[i]
                        best_idx = i

            if best_idx != -1 and max_p > 0.4:
                last_prediction = GESTURE_LABELS[best_idx]
                prediction_expiry = time.time() + 3.0

            is_collecting = False

    if time.time() < prediction_expiry:
        cv2.putText(
            frame,
            f"DETECTED: {last_prediction}",
            (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,
        )

    cv2.imshow("Jutsu Master - Full Grid Diagnostic", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
