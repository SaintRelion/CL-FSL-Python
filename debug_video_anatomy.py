import cv2
import mediapipe as mp
import numpy as np
import json
import os

# New LiteRT Import (Fallback to tflite if package not installed yet)
try:
    import ai_edge_litert.interpreter as litert
except ImportError:
    import tensorflow.lite as litert

from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SIGNATURES_PATH = "data/pillar_signatures.json"
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"
SHAPE_LABELS_PATH = "models/hand_shape_labels.txt"
POSE_TASK = "models/pose_landmarker_full.task"
HAND_TASK = "models/hand_landmarker.task"
ROOT_DIR = "common_clips"

# 1. Load Labels and Signatures
with open(SIGNATURES_PATH, "r") as f:
    GESTURE_SIGS = json.load(f)
with open(SHAPE_LABELS_PATH, "r") as f:
    SHAPE_LABELS = [line.strip() for line in f.readlines()]


# 2. Initialize LiteRT / TFLite
def load_interp(path):
    interp = litert.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


s_interp, s_in, s_out = load_interp(SHAPE_MODEL_PATH)


def create_landmarkers():
    BaseOptions = mp.tasks.BaseOptions
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK),
        running_mode=vision.RunningMode.IMAGE,
    )
    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return vision.PoseLandmarker.create_from_options(
        p_opt
    ), vision.HandLandmarker.create_from_options(h_opt)


p_landmarker, h_landmarker = create_landmarkers()
all_labels = sorted(GESTURE_SIGS.keys())


def extract_shape_features(lms):
    """Your proven feature engineering logic."""
    wrist, m_knk = lms[0], lms[9]
    scale = max(((m_knk.x - wrist.x) ** 2 + (m_knk.y - wrist.y) ** 2) ** 0.5, 1e-6)
    feats = []
    # Relative coordinates
    for lm in lms:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
    # Finger extension ratios
    for t, k in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        dist_t = ((lms[t].x - wrist.x) ** 2 + (lms[t].y - wrist.y) ** 2) ** 0.5
        dist_k = max(
            ((lms[k].x - wrist.x) ** 2 + (lms[k].y - wrist.y) ** 2) ** 0.5, 1e-6
        )
        feats.append(dist_t / dist_k)
    # Finger spread ratios
    for p1, p2 in [(8, 12), (12, 16), (16, 20), (4, 8)]:
        dist_p = ((lms[p1].x - lms[p2].x) ** 2 + (lms[p1].y - lms[p2].y) ** 2) ** 0.5
        feats.append(dist_p / scale)
    return np.array([feats], dtype=np.float32)


# --- MAIN DEBUG LOOP ---
while True:
    for current_label in all_labels:
        v_path = os.path.join(ROOT_DIR, current_label)
        v_list = sorted([v for v in os.listdir(v_path) if v.endswith((".mp4", ".avi"))])

        for v_name in v_list:
            cap = cv2.VideoCapture(os.path.join(v_path, v_name))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                ui_overlay = np.zeros_like(frame)

                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                p_res = p_landmarker.detect(mp_img)
                h_res = h_landmarker.detect(mp_img)

                if p_res.pose_landmarks and h_res.hand_landmarks:
                    p_lms = p_res.pose_landmarks[0]
                    h_lms = h_res.hand_landmarks[0]

                    # 1. SHAPE PREDICTION
                    feats = extract_shape_features(h_lms)
                    s_interp.set_tensor(s_in[0]["index"], feats)
                    s_interp.invoke()
                    probs = s_interp.get_tensor(s_out[0]["index"])[0]
                    shape_idx = np.argmax(probs)
                    detected_shape = SHAPE_LABELS[shape_idx]
                    shape_conf = probs[shape_idx]

                    # 2. PILLAR ACTIVATION (Body Scaled)
                    shl_dist = np.sqrt(
                        (p_lms[11].x - p_lms[12].x) ** 2
                        + (p_lms[11].y - p_lms[12].y) ** 2
                    )
                    scale = max(shl_dist, 0.05)
                    tip = (h_lms[8].x, h_lms[8].y)  # Index Tip

                    nose = p_lms[0]
                    h_size = abs(p_lms[11].y - nose.y)
                    live_pillars = {
                        "nose": (nose.x, nose.y),
                        "left_ear": (p_lms[7].x, p_lms[7].y),
                        "right_ear": (p_lms[8].x, p_lms[8].y),
                        "left_shoulder": (p_lms[11].x, p_lms[11].y),
                        "right_shoulder": (p_lms[12].x, p_lms[12].y),
                        "forehead": (nose.x, nose.y - (h_size * 0.3)),
                        "chin": (nose.x, nose.y + (h_size * 0.25)),
                        "chest": (
                            (p_lms[11].x + p_lms[12].x) / 2,
                            (p_lms[11].y + p_lms[12].y) / 2,
                        ),
                    }

                    # 3. GRID RENDERING
                    cols = 4
                    rows_per_col = (len(all_labels) + cols - 1) // cols
                    col_w = w // cols
                    for idx, sign in enumerate(all_labels):
                        c, r = idx // rows_per_col, idx % rows_per_col
                        x, y = 10 + (c * col_w), 30 + (r * 22)

                        best_act = 0.0
                        for p_name, stats in GESTURE_SIGS[sign].items():
                            p_pos = live_pillars[p_name]
                            dist = (
                                np.sqrt(
                                    (tip[0] - p_pos[0]) ** 2 + (tip[1] - p_pos[1]) ** 2
                                )
                                / scale
                            )

                            limit, target = stats["strict_limit"], stats["target_touch"]
                            act = (
                                1.0
                                if dist <= target
                                else np.clip(
                                    1.0 - ((dist - target) / (limit - target)), 0.0, 1.0
                                )
                            )
                            if act > best_act:
                                best_act = act

                        # Stricter Color Logic
                        color = (
                            (0, 255, 0)
                            if best_act > 0.9
                            else (0, 255, 255) if best_act > 0.1 else (50, 50, 50)
                        )
                        cv2.putText(
                            ui_overlay,
                            f"{sign}: {int(best_act*100)}%",
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1 if best_act < 0.9 else 2,
                        )

                    # HUD Overlay
                    cv2.putText(
                        frame,
                        f"SHAPE: {detected_shape} ({shape_conf:.2f})",
                        (20, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                frame = cv2.addWeighted(frame, 0.5, ui_overlay, 0.8, 0)
                cv2.imshow("FSL Master Debug (LiteRT Standard)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    exit()
            cap.release()

            break
