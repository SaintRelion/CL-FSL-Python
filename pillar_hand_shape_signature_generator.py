import os
import cv2
import numpy as np
import mediapipe as mp
import json
import tensorflow as tf
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
POSE_MODEL = "models/pose_landmarker_full.task"
HAND_MODEL = "models/hand_landmarker.task"
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"
SHAPE_LABELS_PATH = "models/hand_shape_labels.txt"

PILLAR_LABELS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "forehead",
    "chin",
    "chest",
]

# Load Hand Shape Labels
with open(SHAPE_LABELS_PATH, "r") as f:
    SHAPE_LABELS = [line.strip() for line in f.readlines()]


def create_models():
    BaseOptions = mp.tasks.BaseOptions
    gpu = BaseOptions.Delegate.GPU
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL, delegate=gpu),
        running_mode=vision.RunningMode.IMAGE,
    )
    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL, delegate=gpu),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return vision.PoseLandmarker.create_from_options(
        p_opt
    ), vision.HandLandmarker.create_from_options(h_opt)


def extract_hand_shape_features(lms):
    """Your proven scaling and distance-based feature extraction."""
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


def generate_signatures():
    p_landmarker, h_landmarker = create_models()

    # Load TFLite for Hand Shape
    s_interp = tf.lite.Interpreter(model_path=SHAPE_MODEL_PATH)
    s_interp.allocate_tensors()
    s_in, s_out = s_interp.get_input_details(), s_interp.get_output_details()

    gesture_pillars = {}
    gesture_shapes = {}

    labels = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )

    for label in labels:
        print(f"Processing: {label}")
        shape_counts = {}
        all_frames_data = []

        path = os.path.join(ROOT_DIR, label)
        videos = sorted([v for v in os.listdir(path) if v.endswith((".mp4", ".avi"))])

        # PASS 1: COLLECT DATA & PREDICT SHAPES
        for v in videos:
            cap = cv2.VideoCapture(os.path.join(path, v))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                p_res = p_landmarker.detect(mp_img)
                h_res = h_landmarker.detect(mp_img)

                if p_res.pose_landmarks and h_res.hand_landmarks:
                    p_lms = p_res.pose_landmarks[0]
                    h_lms = h_res.hand_landmarks[0]

                    # Detect Shape
                    feats = extract_hand_shape_features(h_lms)
                    s_interp.set_tensor(s_in[0]["index"], feats)
                    s_interp.invoke()
                    probs = s_interp.get_tensor(s_out[0]["index"])[0]
                    shape_name = SHAPE_LABELS[np.argmax(probs)]

                    # Only count confident shapes
                    if probs[np.argmax(probs)] > 0.7:
                        shape_counts[shape_name] = shape_counts.get(shape_name, 0) + 1

                    # Store Temporal Data
                    l_shl, r_shl = p_lms[11], p_lms[12]
                    scale = max(
                        np.sqrt((l_shl.x - r_shl.x) ** 2 + (l_shl.y - r_shl.y) ** 2),
                        0.05,
                    )
                    nose = p_lms[0]
                    h_size = abs(l_shl.y - nose.y)

                    pillars = {
                        "nose": (nose.x, nose.y),
                        "left_ear": (p_lms[7].x, p_lms[7].y),
                        "right_ear": (p_lms[8].x, p_lms[8].y),
                        "left_shoulder": (l_shl.x, l_shl.y),
                        "right_shoulder": (r_shl.x, r_shl.y),
                        "forehead": (nose.x, nose.y - (h_size * 0.3)),
                        "chin": (nose.x, nose.y + (h_size * 0.25)),
                        "chest": ((l_shl.x + r_shl.x) / 2, (l_shl.y + r_shl.y) / 2),
                    }
                    all_frames_data.append(
                        {
                            "pillars": pillars,
                            "tip": (h_lms[8].x, h_lms[8].y),
                            "scale": scale,
                            "shape": shape_name,
                        }
                    )
            cap.release()

        # PASS 2: FILTER PILLAR STATS BY ALLOWED SHAPES
        total_valid_frames = sum(shape_counts.values())
        # A shape must exist in at least 20% of the video to be a "key"
        allowed_shapes = [
            s
            for s, count in shape_counts.items()
            if (count / total_valid_frames) > 0.05
        ]
        gesture_shapes[label] = allowed_shapes

        # Record distances ONLY when a valid hand shape is active
        video_min_dists = {p: [] for p in PILLAR_LABELS}
        current_video_mins = {p: 99.0 for p in PILLAR_LABELS}

        for frame_data in all_frames_data:
            if frame_data["shape"] in allowed_shapes:
                tip = frame_data["tip"]
                for p_name in PILLAR_LABELS:
                    p_pos = frame_data["pillars"][p_name]
                    dist = (
                        np.sqrt((tip[0] - p_pos[0]) ** 2 + (tip[1] - p_pos[1]) ** 2)
                        / frame_data["scale"]
                    )
                    if dist < current_video_mins[p_name]:
                        current_video_mins[p_name] = dist

        # Build strict signature
        signature = {}
        for p_name in PILLAR_LABELS:
            target = (
                current_video_mins[p_name] if current_video_mins[p_name] < 99 else 0.5
            )
            signature[p_name] = {
                "target_touch": round(float(target), 4),
                "strict_limit": round(
                    float(target * 1.25), 4
                ),  # Very strict 25% buffer
            }
        gesture_pillars[label] = signature

    # Export
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "pillar_signatures.json"), "w") as f:
        json.dump(gesture_pillars, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "hand_shape_signatures.json"), "w") as f:
        json.dump(gesture_shapes, f, indent=4)

    print("\nGeneration Complete. Pillar stats are now shape-gated.")


if __name__ == "__main__":
    generate_signatures()
