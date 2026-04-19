import os
import cv2
import numpy as np
import mediapipe as mp
import json
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
MODEL_PATH = "models/pose_landmarker_full.task"

# Derived Pillars
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


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    # GPU Acceleration for fast processing (Match old video_collector style)
    gpu = BaseOptions.Delegate.GPU

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=gpu),
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_frame_data(frame, pose_landmarker):
    """Extracts raw coordinates for pillars and identifies dominant hand."""
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    lms = result.pose_landmarks[0]

    # 1. Base Landmarks
    nose = lms[0]
    l_ear = lms[7]
    r_ear = lms[8]
    l_shl = lms[11]
    r_shl = lms[12]
    r_wrist = lms[16]
    l_wrist = lms[15]

    # 2. Derive Pillars (Generator Logic)
    head_size = abs(l_shl.y - nose.y)
    pillars = {
        "nose": (nose.x, nose.y),
        "left_ear": (l_ear.x, l_ear.y),
        "right_ear": (r_ear.x, r_ear.y),
        "left_shoulder": (l_shl.x, l_shl.y),
        "right_shoulder": (r_shl.x, r_shl.y),
        "forehead": (nose.x, nose.y - (head_size * 0.25)),
        "chin": (nose.x, nose.y + (head_size * 0.2)),
        "chest": ((l_shl.x + r_shl.x) / 2, (l_shl.y + r_shl.y) / 2),
    }

    # Use the hand with higher visibility (Dominant Hand Identification)
    hand = (
        (r_wrist.x, r_wrist.y)
        if r_wrist.visibility > l_wrist.visibility
        else (l_wrist.x, l_wrist.y)
    )

    return {"pillars": pillars, "hand": hand}


def generate_signatures():
    landmarker = create_landmarker()
    gesture_stats = {}

    # ALPHABETICAL ORDER: Ensure labels are processed A-Z
    labels = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )

    for label in labels:
        print(f"Processing Gesture: {label}...")
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

                data = extract_frame_data(frame, landmarker)
                if data:
                    h = data["hand"]
                    for p_name in PILLAR_LABELS:
                        p = data["pillars"][p_name]
                        # Euclidean distance calculation
                        dist = np.sqrt((h[0] - p[0]) ** 2 + (h[1] - p[1]) ** 2)
                        all_distances[p_name].append(dist)
            cap.release()

        # Build Statistical Signature using the 2-Sigma Rule
        signature = {}
        for p_name in PILLAR_LABELS:
            dists = all_distances[p_name]
            if dists:
                mean_val = np.mean(dists)
                std_val = np.std(dists)
                signature[p_name] = {
                    "mean": round(float(mean_val), 4),
                    "std": round(float(std_val), 4),
                    "threshold_2sigma": round(float(mean_val + (std_val * 2)), 4),
                }

        gesture_stats[label] = signature

    # Export to JSON
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, "pillar_signatures.json")
    with open(output_path, "w") as f:
        json.dump(gesture_stats, f, indent=4)

    print(f"\nSuccess! Alphabetical signatures saved to {output_path}")


if __name__ == "__main__":
    generate_signatures()
