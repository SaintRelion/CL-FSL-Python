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

# Selected Pillars
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


def extract_frame_data(frame, p_landmarker, h_landmarker):
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    p_res = p_landmarker.detect(mp_img)
    h_res = h_landmarker.detect(mp_img)

    if not p_res.pose_landmarks or not h_res.hand_landmarks:
        return None

    p_lms = p_res.pose_landmarks[0]
    h_lms = h_res.hand_landmarks[0]

    # 1. Body Scale (Shoulder Width)
    l_shl, r_shl = p_lms[11], p_lms[12]
    scale = np.sqrt((l_shl.x - r_shl.x) ** 2 + (l_shl.y - r_shl.y) ** 2)
    scale = max(scale, 0.05)

    # 2. Index Tip (Target)
    tip = (h_lms[8].x, h_lms[8].y)

    # 3. Derived Pillars
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

    return {"pillars": pillars, "tip": tip, "scale": scale}


def generate_signatures():
    p_landmarker, h_landmarker = create_models()
    gesture_signatures = {}

    labels = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )

    for label in labels:
        print(f"Extracting Strict Proximity for: {label}")
        # We track the 'Minimum Distance' per video to find the "Touch Point"
        video_min_dists = {p: [] for p in PILLAR_LABELS}

        path = os.path.join(ROOT_DIR, label)
        videos = sorted(
            [v for v in os.listdir(path) if v.endswith((".mp4", ".mov", ".avi"))]
        )

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(path, v))
            current_video_mins = {p: 99.0 for p in PILLAR_LABELS}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                data = extract_frame_data(frame, p_landmarker, h_landmarker)
                if data:
                    tip = data["tip"]
                    for p_name in PILLAR_LABELS:
                        p = data["pillars"][p_name]
                        dist = (
                            np.sqrt((tip[0] - p[0]) ** 2 + (tip[1] - p[1]) ** 2)
                            / data["scale"]
                        )
                        # Track the absolute closest point in this video
                        if dist < current_video_mins[p_name]:
                            current_video_mins[p_name] = dist
            cap.release()

            # Record the "Peak" of this video
            for p_name in PILLAR_LABELS:
                if current_video_mins[p_name] < 99.0:
                    video_min_dists[p_name].append(current_video_mins[p_name])

        # Create Signature based on 'Peak Activation'
        signature = {}
        for p_name in PILLAR_LABELS:
            peaks = video_min_dists[p_name]
            if peaks:
                # The 'Average Best Touch'
                target = np.mean(peaks)

                # We drop Standard Deviation and use a hard Strictness Buffer
                # This ensures the bubble is only as large as your closest average touch
                signature[p_name] = {
                    "target_touch": round(float(target), 4),
                    "strict_limit": round(float(target * 1.3), 4),  # 30% tolerance only
                }

        gesture_signatures[label] = signature

    with open(os.path.join(OUTPUT_DIR, "pillar_signatures.json"), "w") as f:
        json.dump(gesture_signatures, f, indent=4)
    print("\nStrict Signatures Generated.")


if __name__ == "__main__":
    generate_signatures()
