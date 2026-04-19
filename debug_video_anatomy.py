import cv2
import mediapipe as mp
import numpy as np
import json
import os

try:
    import ai_edge_litert.interpreter as litert
except ImportError:
    import tensorflow.lite as litert

from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
PILLAR_SIG_PATH = "data/pillar_signatures.json"
SHAPE_SIG_PATH = "data/hand_shape_signatures.json"
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"
SHAPE_LABELS_PATH = "models/hand_shape_labels.txt"
POSE_TASK = "models/pose_landmarker_full.task"
HAND_TASK = "models/hand_landmarker.task"
ROOT_DIR = "common_clips"

# 1. Load All Signatures
with open(PILLAR_SIG_PATH, "r") as f:
    PILLAR_SIGS = json.load(f)
with open(SHAPE_SIG_PATH, "r") as f:
    SHAPE_SIGS = json.load(f)
with open(SHAPE_LABELS_PATH, "r") as f:
    SHAPE_LABELS = [line.strip() for line in f.readlines()]


def load_interp(path):
    interp = litert.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


s_interp, s_in, s_out = load_interp(SHAPE_MODEL_PATH)


def create_landmarkers():
    BaseOptions = mp.tasks.BaseOptions
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK)
    )
    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK), num_hands=1
    )
    return vision.PoseLandmarker.create_from_options(
        p_opt
    ), vision.HandLandmarker.create_from_options(h_opt)


p_landmarker, h_landmarker = create_landmarkers()
all_labels = sorted(PILLAR_SIGS.keys())


def extract_shape_features(lms):
    wrist, m_knk = lms[0], lms[9]
    scale = max(((m_knk.x - wrist.x) ** 2 + (m_knk.y - wrist.y) ** 2) ** 0.5, 1e-6)
    feats = []
    for lm in lms:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
    for t, k in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        dist_t = ((lms[t].x - wrist.x) ** 2 + (lms[t].y - wrist.y) ** 2) ** 0.5
        dist_k = max(
            ((lms[k].x - wrist.x) ** 2 + (lms[k].y - wrist.y) ** 2) ** 0.5, 1e-6
        )
        feats.append(dist_t / dist_k)
    for p1, p2 in [(8, 12), (12, 16), (16, 20), (4, 8)]:
        dist_p = ((lms[p1].x - lms[p2].x) ** 2 + (lms[p1].y - lms[p2].y) ** 2) ** 0.5
        feats.append(dist_p / scale)
    return np.array([feats], dtype=np.float32)


# --- NAVIGATION ---
label_idx = 0
while label_idx < len(all_labels):
    current_label = all_labels[label_idx]
    v_path = os.path.join(ROOT_DIR, current_label)
    if not os.path.exists(v_path):
        label_idx += 1
        continue

    v_list = sorted([v for v in os.listdir(v_path) if v.endswith((".mp4", ".avi"))])

    for v_name in v_list:
        cap = cv2.VideoCapture(os.path.join(v_path, v_name))
        print(f"Auditing: [{current_label}] | Video: {v_name}")

        # Buffer to track unique hand shapes detected in THIS video session
        seen_shapes_buffer = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video - stay on frame or wait for Enter?
            # If the video ends naturally, we will loop back to the start until Enter is pressed
            h, w, _ = frame.shape
            ui_overlay = np.zeros_like(frame)

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            p_res = p_landmarker.detect(mp_img)
            h_res = h_landmarker.detect(mp_img)

            detected_shape = "NONE"
            if p_res.pose_landmarks and h_res.hand_landmarks:
                p_lms = p_res.pose_landmarks[0]
                h_lms = h_res.hand_landmarks[0]

                # 1. SHAPE GATE
                feats = extract_shape_features(h_lms)
                s_interp.set_tensor(s_in[0]["index"], feats)
                s_interp.invoke()
                probs = s_interp.get_tensor(s_out[0]["index"])[0]
                detected_shape = SHAPE_LABELS[np.argmax(probs)]
                if probs[np.argmax(probs)] > 0.7:
                    seen_shapes_buffer.add(detected_shape)

                # 2. PILLAR SETUP
                shl_dist = np.sqrt(
                    (p_lms[11].x - p_lms[12].x) ** 2 + (p_lms[11].y - p_lms[12].y) ** 2
                )
                scale = max(shl_dist, 0.05)
                tip = (h_lms[8].x, h_lms[8].y)
                nose = p_lms[0]
                head_size = abs(p_lms[11].y - nose.y)
                live_pillars = {
                    "nose": (nose.x, nose.y),
                    "forehead": (nose.x, nose.y - (head_size * 0.3)),
                    "chin": (nose.x, nose.y + (head_size * 0.25)),
                    "chest": (
                        (p_lms[11].x + p_lms[12].x) / 2,
                        (p_lms[11].y + p_lms[12].y) / 2,
                    ),
                }

                # 3. DYNAMIC AUDIT GRID
                cols = 4
                rows_per_col = (len(all_labels) + cols - 1) // cols
                col_w = w // cols
                for idx, sign in enumerate(all_labels):
                    c, r = idx // rows_per_col, idx % rows_per_col
                    x, y = 10 + (c * col_w), 30 + (r * 22)

                    required_shapes = set(SHAPE_SIGS.get(sign, []))
                    intersection = required_shapes.intersection(seen_shapes_buffer)

                    # GATE LOGIC
                    is_gate_open = len(intersection) >= 2 or (
                        len(required_shapes) < 2 and len(intersection) > 0
                    )

                    best_p = 0.0
                    for p_name, stats in PILLAR_SIGS[sign].items():
                        if p_name not in live_pillars:
                            continue
                        p_pos = live_pillars[p_name]
                        dist = (
                            np.sqrt((tip[0] - p_pos[0]) ** 2 + (tip[1] - p_pos[1]) ** 2)
                            / scale
                        )
                        t, l = stats["target_touch"], stats["strict_limit"]
                        act = (
                            1.0
                            if dist <= t
                            else np.clip(1.0 - ((dist - t) / (l - t)), 0.0, 1.0)
                        )
                        if act > best_p:
                            best_p = act

                    # COLOR LOGIC
                    if is_gate_open:
                        if best_p > 0.8:
                            color = (0, 255, 0)  # GREEN: Full Match
                        else:
                            color = (
                                0,
                                165,
                                255,
                            )  # ORANGE: Shapes found, but Pillar out of range
                        text = f"{sign}: {int(best_p*100)}%"
                    else:
                        color = (40, 40, 40)  # GRAY: Blocked by Shape Gate
                        text = f"{sign} [{len(intersection)}/{len(required_shapes)}]"

                    cv2.putText(
                        ui_overlay,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.33,
                        color,
                        1,
                    )

            cv2.putText(
                frame,
                f"VIDEO: {v_name} | SHAPE: {detected_shape}",
                (20, h - 30),
                0,
                0.7,
                (0, 255, 0),
                2,
            )
            frame = cv2.addWeighted(frame, 0.4, ui_overlay, 0.8, 0)
            cv2.imshow("FSL Precision Audit", frame)

            key = cv2.waitKey(40) & 0xFF
            if key == 13:  # ENTER - Moves to next video
                break
            if key == 84:  # DOWN - Moves to next Label
                cap.release()
                v_list = []  # Exit current video list
                break
            if key == 27:
                exit()

        # Keep current frame displayed and wait for ENTER if the video ends
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
            if key == 27:
                exit()
            if key == 84:
                v_list = []
                break

        cap.release()
    label_idx += 1
