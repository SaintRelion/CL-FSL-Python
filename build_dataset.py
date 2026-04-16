# build_dataset.py
import os
import cv2
import numpy as np
import json
import re
import mediapipe as mp
from mediapipe.tasks.python import vision
from fsl_helper import resample_sequence

# ------------------- Configuration -------------------
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
TARGET_FRAMES = 30
# Hand1(7) + Hand2(7) + 10 Pillar Distances = 24
FEATURE_COUNT = 24

# Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
PoseLandmarker = vision.PoseLandmarker
FaceLandmarker = vision.FaceLandmarker


# ------------------- Model Creation -------------------
def create_models():
    # Using the standardized model paths from your setup
    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
        min_pose_detection_confidence=0.5,
    )
    f_opt = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        min_face_detection_confidence=0.5,
    )

    return (
        HandLandmarker.create_from_options(h_opt),
        PoseLandmarker.create_from_options(p_opt),
        FaceLandmarker.create_from_options(f_opt),
    )


# ------------------- Visualization Helper -------------------
def get_heatmap_color(dist, threshold=0.15):
    """Converts distance into a BGR color (Blue for cold, Red for hot)."""
    # Normalize distance: 0.0 is 'on top of pillar', 1.0 is 'far away'
    score = np.clip(dist / threshold, 0, 1)

    # Simple Linear Interpolation: Red (0,0,255) to Cyan (255,255,0)
    # Blue component increases as you get farther
    # Red component increases as you get closer
    r = int(255 * (1 - score))
    g = int(255 * score)
    b = int(255 * score)
    return (b, g, r), score


def draw_debug_ui(frame, pillars, hand_results, pillar_distances):
    h, w, _ = frame.shape

    # 1. Draw "Heatmap" Pillars
    for i, (name, pos) in enumerate(pillars.items()):
        center = (int(pos[0] * w), int(pos[1] * h))

        # Get color based on the actual distance calculated in extract_frame_features
        dist = pillar_distances[i]
        color, score = get_heatmap_color(dist)

        # Draw an outer "Glow" or "Heat Ring"
        # Radius expands slightly when 'Hot'
        radius = int(10 + (20 * (1 - score)))
        thickness = int(1 + (5 * (1 - score)))

        cv2.circle(frame, center, radius, color, thickness)
        cv2.circle(frame, center, 4, color, -1)  # Core point

        # Label with "Heat %"
        heat_percent = int((1 - score) * 100)
        cv2.putText(
            frame,
            f"{name} {heat_percent}%",
            (center[0] + 15, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    # 2. Draw Hand Landmarks
    if hand_results.hand_landmarks:
        for hl in hand_results.hand_landmarks:
            for lm in hl:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

    return frame


# ------------------- Feature Extraction -------------------
def extract_frame_features(frame, hand_model, pose_model, face_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    h_res = hand_model.detect(mp_image)
    f_res = face_model.detect(mp_image)
    p_res = pose_model.detect(mp_image)

    # 1. Define Pillar Anchors (Relative Coordinates)
    pillars = {
        "forehead": [0.5, 0.2],
        "nose": [0.5, 0.4],
        "mouth": [0.5, 0.5],
        "chin": [0.5, 0.6],
        "l_ear": [0.4, 0.4],
        "r_ear": [0.6, 0.4],
        "l_shoulder": [0.3, 0.8],
        "r_shoulder": [0.7, 0.8],
        "chest": [0.5, 0.8],
        "neutral": [0.9, 0.9],
    }

    if f_res.face_landmarks:
        fl = f_res.face_landmarks[0]
        pillars["forehead"] = [fl[10].x, fl[10].y]
        pillars["nose"] = [fl[1].x, fl[1].y]
        pillars["mouth"] = [(fl[13].x + fl[14].x) / 2, (fl[13].y + fl[14].y) / 2]
        pillars["chin"] = [fl[152].x, fl[152].y]
        pillars["l_ear"] = [fl[234].x, fl[234].y]
        pillars["r_ear"] = [fl[454].x, fl[454].y]

    if p_res.pose_landmarks:
        pl = p_res.pose_landmarks[0]
        pillars["l_shoulder"] = [pl[11].x, pl[11].y]
        pillars["r_shoulder"] = [pl[12].x, pl[12].y]
        pillars["chest"] = [(pl[11].x + pl[12].x) / 2, (pl[11].y + pl[12].y) / 2]

    # 2. Extract Hand Landmarks (Local Geometry)
    hand_feats = []
    primary_wrist = None

    for i in range(2):
        if h_res.hand_landmarks and i < len(h_res.hand_landmarks):
            h = h_res.hand_landmarks[i]
            wrist = np.array([h[0].x, h[0].y])
            if i == 0:
                primary_wrist = wrist

            # Use Wrist-to-Middle-Knuckle as local scale
            palm_len = max(np.linalg.norm(wrist - np.array([h[9].x, h[9].y])), 1e-6)

            # 5 finger extensions (distance from wrist to tip)
            for tip in [4, 8, 12, 16, 20]:
                tip_pos = np.array([h[tip].x, h[tip].y])
                hand_feats.append(np.linalg.norm(tip_pos - wrist) / palm_len)

            # 2D Direction Vector (Orientation)
            ori = (np.array([h[9].x, h[9].y]) - wrist) / palm_len
            hand_feats.extend([ori[0], ori[1]])
        else:
            hand_feats.extend([0] * 7)

    # 3. Calculate Pillar Distances (The "Heatmap" Activation)
    pillar_feats = []
    for name, pos in pillars.items():
        if primary_wrist is not None:
            # Linear distance from wrist to anchor point
            dist = np.linalg.norm(primary_wrist - np.array(pos))
            pillar_feats.append(dist)
        else:
            pillar_feats.append(1.0)  # Far distance if no hand detected

    debug_frame = draw_debug_ui(frame.copy(), pillars, h_res, pillar_feats)
    return hand_feats + pillar_feats, debug_frame


def extract_sequence_features(video_path, models):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        feat, debug_frame = extract_frame_features(frame, *models)

        # # Visualize during processing so you can see the Pillars "hitting"
        # cv2.imshow("FSL LINK - Build Dataset Debug", debug_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # Valid frame if hand data exists (not all zeros/ones)
        if any(v != 0 and v != 1.0 for v in feat[:14]):
            features.append(feat)

    cap.release()
    cv2.destroyAllWindows()
    return features if len(features) >= 5 else None


# ------------------- Dataset Builder -------------------
def build_dataset():
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
    models = create_models()
    X, y = [], []

    for label in common_labels:
        label_path = os.path.join(ROOT_DIR, label)
        if not os.path.isdir(label_path):
            print(f"Skipping {label}: Directory not found")
            continue

        videos = sorted([f for f in os.listdir(label_path) if f.endswith(".mp4")])
        print(f"Processing {label} ({len(videos)} videos)...")

        temp = 0
        for vid in videos:
            raw = extract_sequence_features(os.path.join(label_path, vid), models)
            if raw:
                # Resample to 30 frames for LSTM/GRU input
                feats = resample_sequence(raw, TARGET_FRAMES)
                if feats is not None:
                    X.append(feats)
                    y.append(label_map[label])

            temp += 1
            if temp >= 2:
                break

    return np.array(X, dtype=np.float32), np.array(y), label_map


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y, label_map = build_dataset()

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSuccess! X Shape: {X.shape} (Videos, Frames, Features)")
