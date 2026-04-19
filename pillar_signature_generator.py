import os
import cv2
import numpy as np
import mediapipe as mp
import json
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
POSE_MODEL = "models/pose_landmarker_full.task"
HAND_MODEL = "models/hand_landmarker.task"

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


def create_models():
    """Initializes Pose and Hand landmarkers with GPU acceleration."""
    BaseOptions = mp.tasks.BaseOptions
    gpu = BaseOptions.Delegate.GPU

    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL, delegate=gpu),
        running_mode=vision.RunningMode.IMAGE,
    )
    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL, delegate=gpu),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,  # Track dominant hand
    )
    return (
        vision.PoseLandmarker.create_from_options(p_opt),
        vision.HandLandmarker.create_from_options(h_opt),
    )


def extract_normalized_data(frame, p_landmarker, h_landmarker):
    """Extracts fingertip and pillars, normalized by shoulder-to-shoulder width."""
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    p_res = p_landmarker.detect(mp_img)
    h_res = h_landmarker.detect(mp_img)

    # We need both body and hand to create a valid signature
    if not p_res.pose_landmarks or not h_res.hand_landmarks:
        return None

    p_lms = p_res.pose_landmarks[0]
    h_lms = h_res.hand_landmarks[0]

    # 1. SCALE: Shoulder-to-Shoulder distance (The stable "Unit" of 1.0)
    l_shl, r_shl = p_lms[11], p_lms[12]
    shoulder_dist = np.sqrt((l_shl.x - r_shl.x) ** 2 + (l_shl.y - r_shl.y) ** 2)
    scale = max(shoulder_dist, 0.05)

    # 2. TARGET: Index Finger Tip (Landmark 8)
    tip = h_lms[8]

    # 3. PILLARS (Exactly matching debug_video_anatomy logic)
    nose = p_lms[0]
    head_size = abs(l_shl.y - nose.y)
    pillars = {
        "nose": (nose.x, nose.y),
        "left_ear": (p_lms[7].x, p_lms[7].y),
        "right_ear": (p_lms[8].x, p_lms[8].y),
        "left_shoulder": (l_shl.x, l_shl.y),
        "right_shoulder": (r_shl.x, r_shl.y),
        "forehead": (nose.x, nose.y - (head_size * 0.3)),
        "chin": (nose.x, nose.y + (head_size * 0.25)),
        "chest": ((l_shl.x + r_shl.x) / 2, (l_shl.y + r_shl.y) / 2),
    }

    return {"pillars": pillars, "tip": (tip.x, tip.y), "scale": scale}


def generate_signatures():
    p_landmarker, h_landmarker = create_models()
    gesture_stats = {}

    # Sort labels A-Z
    labels = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )

    for label in labels:
        print(f"Analyzing Gesture: {label}...")
        all_distances = {p: [] for p in PILLAR_LABELS}
        path = os.path.join(ROOT_DIR, label)
        videos = sorted(
            [v for v in os.listdir(path) if v.endswith((".mp4", ".mov", ".avi"))]
        )

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(path, v))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                data = extract_normalized_data(frame, p_landmarker, h_landmarker)
                if data:
                    tip = data["tip"]
                    for p_name in PILLAR_LABELS:
                        p = data["pillars"][p_name]
                        # Calculate Euclidean distance and DIVIDE by scale
                        raw_dist = np.sqrt((tip[0] - p[0]) ** 2 + (tip[1] - p[1]) ** 2)
                        norm_dist = raw_dist / data["scale"]
                        all_distances[p_name].append(norm_dist)
            cap.release()

        # Generate Signature (Mean + Standard Deviation)
        signature = {}
        for p_name in PILLAR_LABELS:
            dists = all_distances[p_name]
            if dists:
                m = np.mean(dists)
                s = np.std(dists)
                signature[p_name] = {
                    "mean": round(float(m), 4),
                    "std": round(float(s), 4),
                    "threshold_2sigma": round(float(m + (s * 2)), 4),
                }

        gesture_stats[label] = signature

    # Export to JSON
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "pillar_signatures.json"), "w") as f:
        json.dump(gesture_stats, f, indent=4)

    print(
        f"\nProcessing Complete. Signatures saved to {OUTPUT_DIR}/pillar_signatures.json"
    )


if __name__ == "__main__":
    generate_signatures()
