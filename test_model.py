# python
import os
import json
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from build_dataset import extract_frame_features, resample_sequence

# ---------------- Config ----------------
MODEL_FILE = "models/gesture_model.keras"
LABEL_MAP_FILE = "data/label_map.json"
TARGET_FRAMES = 30
EXPECTED_DIM = 18
TOP_K = 3

# ---------------- Stress Test Settings ----------------
TEST_FLIP = True  # Test Left vs Right Hand
TEST_RESIZE = True  # Test Distance/Scale variance
TEST_BRIGHTNESS = True  # Test Lighting robustness

# ---------------- Load model + labels ----------------
model = tf.keras.models.load_model(MODEL_FILE)
with open(LABEL_MAP_FILE) as f:
    label_map = json.load(f)
inv_label_map = {int(v): k for k, v in label_map.items()}

# ---------------- MediaPipe Setup ----------------
BaseOptions = mp.tasks.BaseOptions
hand_model = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2,
    )
)
pose_model = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task")
    )
)
face_model = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task")
    )
)


# ---------------- Augmentation Logic ----------------
def apply_augmentations(frame):
    aug_frame = frame.copy()
    h, w = aug_frame.shape[:2]

    # --- NEW: Random Translation (Moving the Person) ---
    # Shift up to 15% of the width/height in any direction
    max_shift_x = int(w * 0.15)
    max_shift_y = int(h * 0.15)

    tx = np.random.randint(-max_shift_x, max_shift_x)
    ty = np.random.randint(-max_shift_y, max_shift_y)

    # Translation Matrix: [[1, 0, tx], [0, 1, ty]]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    aug_frame = cv2.warpAffine(aug_frame, translation_matrix, (w, h))

    # 1. Flip (The "Agnostic" Test)
    if TEST_FLIP:
        aug_frame = cv2.flip(aug_frame, 1)

    # 2. Resize/Scale (The "Depth" Test)
    if TEST_RESIZE:
        scale = np.random.uniform(0.7, 1.3)
        new_w, new_h = int(w * scale), int(h * scale)
        temp = cv2.resize(aug_frame, (new_w, new_h))

        # Letterbox/Crop back to original size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        y_off = max(0, (h - new_h) // 2)
        x_off = max(0, (w - new_w) // 2)
        sh, sw = min(new_h, h), min(new_w, w)
        canvas[y_off : y_off + sh, x_off : x_off + sw] = temp[:sh, :sw]
        aug_frame = canvas

    # 3. Brightness (The "Sensor" Test)
    if TEST_BRIGHTNESS:
        gamma = np.random.uniform(0.5, 1.5)
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        aug_frame = cv2.LUT(aug_frame, table)

    return aug_frame


# ---------------- Main Test Loop ----------------
def test_augmented_model(root_dir, max_vids=3):
    print("\n" + "=" * 50)
    print(
        f"STRESS TEST: Flip={TEST_FLIP}, Scale={TEST_RESIZE}, Light={TEST_BRIGHTNESS}"
    )
    print("=" * 50 + "\n")

    labels = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )

    for label in labels:
        label_path = os.path.join(root_dir, label)
        print(f"🔹 LABEL: {label}")

        videos = sorted([v for v in os.listdir(label_path) if v.endswith(".mp4")])[
            :max_vids
        ]

        for vid in videos:
            cap = cv2.VideoCapture(os.path.join(label_path, vid))
            raw_feats = []
            prev = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # AUGMENT BEFORE EXTRACTION
                processed_frame = apply_augmentations(frame)

                feat, prev = extract_frame_features(
                    processed_frame, hand_model, pose_model, face_model, prev
                )
                if feat:
                    raw_feats.append(feat)
            cap.release()

            # Resample and Predict
            X_seq = resample_sequence(raw_feats, TARGET_FRAMES)
            if X_seq:
                X_input = np.expand_dims(np.array(X_seq, dtype=np.float32), axis=0)
                probs = model.predict(X_input, verbose=0)[0]
                top_idx = np.argsort(probs)[::-1][:TOP_K]

                print(f"  → {vid}:")
                for i in top_idx:
                    pred_name = inv_label_map.get(i, "???")
                    star = "⭐" if pred_name == label else "  "
                    print(f"     {star} {pred_name:<20} {probs[i]:.4f}")
            else:
                print(f"  → {vid}: ⚠️ EXTRACTION FAILED")


if __name__ == "__main__":
    test_augmented_model("common_clips")
