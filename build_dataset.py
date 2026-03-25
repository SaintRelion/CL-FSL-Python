# build_dataset.py
import os
import cv2
import numpy as np
import json
from fsl_helper import resample_sequence

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

import numpy as np
import os
import re


def sanitize_label(label):
    label = label.replace("'", "_").replace("’", "_").replace(" ", "_")
    label = re.sub(r"_+", "_", label)
    return label.upper()


# ------------------- Paths -------------------
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
TARGET_FRAMES = 30
# Hand1(5 ext + 2 ori) + Hand2(5 ext + 2 ori) + Proximity(4 bools) = 18
FEATURE_COUNT = 18

# Aliases for Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarker = vision.HandLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarker = vision.PoseLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
VisionRunningMode = vision.RunningMode


# ------------------- MediaPipe Options -------------------
def create_hand_model(
    MODEL_PATH="models/hand_landmarker.task", NUM_HANDS=2, MIN_CONFIDENCE=0.5
):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


def create_pose_model(
    MODEL_PATH="models/pose_landmarker_full.task", MIN_CONFIDENCE=0.5
):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        min_pose_detection_confidence=MIN_CONFIDENCE,
    )
    return PoseLandmarker.create_from_options(options)


def create_face_model(MODEL_PATH="models/face_landmarker.task", MIN_CONFIDENCE=0.5):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        min_face_detection_confidence=MIN_CONFIDENCE,
    )
    return FaceLandmarker.create_from_options(options)


# ------------------- Feature Extraction -------------------
def extract_frame_features(frame, hand_model, pose_model, face_model, prev):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_model.detect(mp_image)
    face_result = face_model.detect(mp_image)
    # Pose kept for shoulder anchors if needed, but face-width is our primary scale
    pose_result = pose_model.detect(mp_image)

    # 1. Setup Anchors & Scale
    face_width = 0.1  # Default small value
    mouth_pos = np.array([0.5, 0.5])
    forehead_pos = np.array([0.5, 0.2])

    if face_result.face_landmarks:
        fl = face_result.face_landmarks[0]
        # Eye-to-eye distance for scale (Landmarks 33 and 263)
        p1 = np.array([fl[33].x, fl[33].y])
        p2 = np.array([fl[263].x, fl[263].y])
        face_width = max(np.linalg.norm(p1 - p2), 1e-6)

        # Mouth Center (Average of upper/lower lip)
        mouth_pos = (
            np.array([fl[13].x, fl[13].y]) + np.array([fl[14].x, fl[14].y])
        ) / 2
        # Forehead (Landmark 10)
        forehead_pos = np.array([fl[10].x, fl[10].y])

    # 2. Process Hands (Extensions & Orientation)
    hand_feat = []
    wrist_positions = []

    # MediaPipe doesn't guarantee LEFT then RIGHT, but for training,
    # we just take the first two detected slots.
    for i in range(2):
        if hand_result.hand_landmarks and i < len(hand_result.hand_landmarks):
            h = hand_result.hand_landmarks[i]
            wrist = np.array([h[0].x, h[0].y])
            wrist_positions.append(wrist)

            # Use Wrist-to-Middle-Knuckle (0 to 9) as local hand scale
            palm_len = np.linalg.norm(wrist - np.array([h[9].x, h[9].y]))
            palm_len = max(palm_len, 1e-6)

            # 5 Extensions: Wrist-to-Tip normalized by Palm
            for tip_idx in [4, 8, 12, 16, 20]:
                tip = np.array([h[tip_idx].x, h[tip_idx].y])
                hand_feat.append(np.linalg.norm(tip - wrist) / palm_len)

            # Palm Orientation Vector (normalized)
            ori_vec = (np.array([h[9].x, h[9].y]) - wrist) / palm_len
            hand_feat.extend([ori_vec[0], ori_vec[1]])
        else:
            hand_feat.extend([0] * 7)  # 5 extensions + 2 orientation
            wrist_positions.append(None)

    # 3. Proximity Triggers (The "State Machine")
    # Threshold: approx 1.2x face width is usually a good "touching" zone
    zone_threshold = face_width * 1.2
    proximity_feat = []

    for w in wrist_positions:
        if w is not None:
            near_mouth = 1 if np.linalg.norm(w - mouth_pos) < zone_threshold else 0
            near_forehead = (
                1 if np.linalg.norm(w - forehead_pos) < zone_threshold else 0
            )
            proximity_feat.extend([near_mouth, near_forehead])
        else:
            proximity_feat.extend([0, 0])

    # 4. Final Assemble (14 Hand features + 4 Bool features)
    feat = hand_feat + proximity_feat

    # Check if we detected anything at all
    if all(v == 0 for v in feat):
        if prev is not None:
            return prev, prev
        return None, None

    return feat, feat


def extract_sequence_features(
    video_path, hand_model, pose_model, face_model, show_progress=False
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_iter = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_iter.append(frame)
    cap.release()

    features = []
    prev = None
    for frame in frames_iter:
        feat, prev = extract_frame_features(
            frame, hand_model, pose_model, face_model, prev
        )
        if feat is not None:
            features.append(feat)

    # Return None if the sequence is too garbage (less than 5 frames of data)
    return features if len(features) >= 5 else None


def build_dataset(root_dir=ROOT_DIR, max_labels=None, max_videos_per_label=None):
    X, y = [], []

    # Use your defined alphabetical list to ensure the map is consistent
    common_labels = [
        "BOY",
        "COLD",
        "CORRECT",
        "DONT_KNOW",
        "DONT_UNDERSTAND",
        "FAST",
        "GIRL",
        "GOOD_AFTERNOON",
        "GOOD_EVENING",
        "GOOD_MORNING",
        "HELLO",
        "HOT",
        "HOW_ARE_YOU",
        "IM_FINE",
        "KNOW",
        "MAN",
        "NICE_TO_MEET_YOU",
        "NO",
        "PARENTS",
        "SEE_YOU_TOMORROW",
        "SLOW",
        "THANK_YOU",
        "TODAY",
        "TOMORROW",
        "UNDERSTAND",
        "WOMAN",
        "WRONG",
        "YES",
        "YESTERDAY",
        "YOURE_WELCOME",
    ]
    label_map = {label: i for i, label in enumerate(common_labels)}

    hand_model = create_hand_model()
    pose_model = create_pose_model()
    face_model = create_face_model()

    # Iterate through the common_labels to keep directory processing in order
    for label in common_labels:
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            print(f"Skipping {label}: Directory not found")
            continue

        video_list = sorted([f for f in os.listdir(label_path) if f.endswith(".mp4")])
        print(f"Processing {label} ({len(video_list)} videos)...")

        for vid_idx, vid in enumerate(video_list):
            if max_videos_per_label and vid_idx >= max_videos_per_label:
                break

            raw = extract_sequence_features(
                os.path.join(label_path, vid), hand_model, pose_model, face_model
            )

            if raw:
                # Use your existing resample_sequence to hit TARGET_FRAMES (30)
                feats = resample_sequence(raw, TARGET_FRAMES)
                if feats is not None:
                    assert len(feats) == TARGET_FRAMES
                    assert len(feats[0]) == FEATURE_COUNT
                    X.append(feats)
                    y.append(label_map[label])

    return np.array(X, dtype=np.float32), np.array(y), label_map


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y, label_map = build_dataset()

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSuccess! Dataset built.")
    print(f"X Shape: {X.shape} (Videos, Frames, Features)")
    print(f"y Shape: {y.shape}")
