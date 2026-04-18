# python
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import vision
import os
import time

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"
PILLAR_MODEL_PATH = "models/pillar_path_model.tflite"
SHAPE_LABELS_PATH = "models/hand_shape_labels.txt"

GESTURE_LABELS = sorted(
    [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
)

# --- STATE VARIABLES ---
sequence_buffer = []
is_collecting = False
start_time = 0
COLLECTION_DURATION = 1.7  # Your requested 1.7 seconds
TARGET_FRAMES = 30
last_prediction = ""
prediction_display_until = 0


# --- INITIALIZATION (Same as before) ---
def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


shape_interp, shape_in, shape_out = load_tflite(SHAPE_MODEL_PATH)
path_interp, path_in, path_out = load_tflite(PILLAR_MODEL_PATH)
NUM_SHAPE_CLASSES = shape_out[0]["shape"][-1]

with open(SHAPE_LABELS_PATH, "r") as f:
    SHAPE_LABELS = [line.strip() for line in f.readlines()]

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


def get_shape_probs(lms):
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
    return shape_interp.get_tensor(shape_out[0]["index"])[0]


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

    # 1. Update Pillar Coords
    f_scale = 0.2
    cur_pillars = [[0.5, 0.4], [0.5, 0.5], [0.3, 0.7], [0.7, 0.7], [0.5, 0.7]]
    if f_res.face_landmarks:
        fl = f_res.face_landmarks[0]
        cur_pillars[0], cur_pillars[1] = [fl[1].x, fl[1].y], [
            (fl[13].x + fl[14].x) / 2,
            (fl[13].y + fl[14].y) / 2,
        ]
        f_scale = max(abs(fl[10].y - fl[152].y), 1e-6)
    if p_res.pose_landmarks:
        pl = p_res.pose_landmarks[0]
        cur_pillars[2], cur_pillars[3], cur_pillars[4] = (
            [pl[11].x, pl[11].y],
            [pl[12].x, pl[12].y],
            [(pl[11].x + pl[12].x) / 2, (pl[11].y + pl[12].y) / 2],
        )

    # 2. Hand Detection & Triggering
    hand_detected = h_res.hand_landmarks is not None and len(h_res.hand_landmarks) > 0
    full_vector = []
    active_shapes = ["None", "None"]

    if hand_detected:
        # Start collection if not already
        if not is_collecting:
            is_collecting = True
            start_time = time.time()
            sequence_buffer = []  # Clear old data
            print("Hand detected! Starting 1.7s collection...")

        for i in range(2):
            if i < len(h_res.hand_landmarks):
                lms = h_res.hand_landmarks[i]
                probs = get_shape_probs(lms)
                active_shapes[i] = SHAPE_LABELS[np.argmax(probs)]
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
                full_vector += list(probs) + w_dists + t_dists + [1.0]
                # Draw skeleton
                for lm in lms:
                    cv2.circle(
                        frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1
                    )
            else:
                full_vector += [0.0] * NUM_SHAPE_CLASSES + [0.0] * 10 + [0.0]

        if is_collecting:
            sequence_buffer.append(full_vector)
    else:
        # If hand lost during collection, we reset (Optional: could also just wait)
        if is_collecting and (time.time() - start_time) > 0.3:  # 0.3s grace period
            is_collecting = False
            sequence_buffer = []

    # 3. Time-Based Prediction
    if is_collecting:
        elapsed = time.time() - start_time
        # Progress Bar
        bar_w = int(w * (elapsed / COLLECTION_DURATION))
        cv2.rectangle(frame, (0, h - 15), (bar_w, h), (0, 255, 255), -1)

        if elapsed >= COLLECTION_DURATION:
            # Standardize buffer to 30 frames (exactly what model expects)
            from fsl_helper import resample_sequence

            final_seq = resample_sequence(sequence_buffer, TARGET_FRAMES)

            input_data = np.array([final_seq], dtype=np.float32)
            path_interp.set_tensor(path_in[0]["index"], input_data)
            path_interp.invoke()
            prediction = path_interp.get_tensor(path_out[0]["index"])[0]

            idx = np.argmax(prediction)
            if prediction[idx] > 0.7:
                last_prediction = f"{GESTURE_LABELS[idx]} ({prediction[idx]*100:.0f}%)"
                prediction_display_until = time.time() + 2.0  # Show for 2 seconds

            # Reset
            is_collecting = False
            sequence_buffer = []

    # 4. Display Results
    if time.time() < prediction_display_until:
        cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"JUTSU: {last_prediction}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

    cv2.imshow("Timed Jutsu Master", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
