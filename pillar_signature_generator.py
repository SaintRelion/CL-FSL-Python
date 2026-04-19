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

# Derived Pillars: nose(0), l_ear(7), r_ear(8), l_shoulder(11), r_shoulder(12)
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
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_frame_data(frame, pose_landmarker):
    """Extracts raw coordinates for pillars and the dominant hand."""
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    lms = result.pose_landmarks[0]

    # 1. Base MediaPipe Landmarks
    nose = lms[0]
    l_ear = lms[7]
    r_ear = lms[8]
    l_shl = lms[11]
    r_shl = lms[12]
    # Dominant Hand (Wrist) - using Pose landmarks for simplicity in pillar logic
    # Right wrist is 16, Left wrist is 15
    r_wrist = lms[16]
    l_wrist = lms[15]

    # 2. Derive Custom Pillars
    # Forehead: Offset above nose relative to head size
    head_size = abs(l_shl.y - nose.y)
    forehead_y = nose.y - (head_size * 0.25)
    # Chin: Offset below nose
    chin_y = nose.y + (head_size * 0.2)
    # Chest: Midpoint of shoulders
    chest_x = (l_shl.x + r_shl.x) / 2
    chest_y = (l_shl.y + r_shl.y) / 2

    pillars = {
        "nose": (nose.x, nose.y),
        "left_ear": (l_ear.x, l_ear.y),
        "right_ear": (r_ear.x, r_ear.y),
        "left_shoulder": (l_shl.x, l_shl.y),
        "right_shoulder": (r_shl.x, r_shl.y),
        "forehead": (nose.x, forehead_y),
        "chin": (nose.x, chin_y),
        "chest": (chest_x, chest_y),
    }

    # Identify dominant hand (the one closest to the nose/active area)
    hand = (
        (r_wrist.x, r_wrist.y)
        if r_wrist.visibility > l_wrist.visibility
        else (l_wrist.x, l_wrist.y)
    )

    return {"pillars": pillars, "hand": hand}


def generate_signatures():
    landmarker = create_landmarker()
    gesture_stats = {}

    labels = [
        d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))
    ]

    for label in labels:
        print(f"Processing: {label}...")
        all_distances = {p: [] for p in PILLAR_LABELS}
        path = os.path.join(ROOT_DIR, label)
        videos = [v for v in os.listdir(path) if v.endswith((".mp4", ".mov", ".avi"))]

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
                        # Calculate Euclidean distance
                        dist = np.sqrt((h[0] - p[0]) ** 2 + (h[1] - p[1]) ** 2)
                        all_distances[p_name].append(dist)
            cap.release()

        # Generate the Statistical Signature for this Label
        signature = {}
        for p_name in PILLAR_LABELS:
            dists = all_distances[p_name]
            if dists:
                # We calculate Mean and Standard Deviation (Sigma)
                signature[p_name] = {
                    "mean": round(float(np.mean(dists)), 4),
                    "std": round(float(np.std(dists)), 4),
                    "threshold_2sigma": round(
                        float(np.mean(dists) + (np.std(dists) * 2)), 4
                    ),
                }

        gesture_stats[label] = signature

    # Export to JSON
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "pillar_signatures.json"), "w") as f:
        json.dump(gesture_stats, f, indent=4)

    print("\nGeneration Complete. File saved to data/pillar_signatures.json")


if __name__ == "__main__":
    generate_signatures()
