import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from fsl_helper import resample_sequence

# ------------------- Configuration -------------------
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
TARGET_FRAMES = 30
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"

# Ensure these models are in your AWS instance models/ folder
HAND_TASK = "models/hand_landmarker.task"
POSE_TASK = "models/pose_landmarker_full.task"
FACE_TASK = "models/face_landmarker.task"

# ------------------- Initialization -------------------


def create_models():
    BaseOptions = mp.tasks.BaseOptions
    # Setting delegate=BaseOptions.Delegate.GPU for AWS L4 acceleration
    gpu_delegate = BaseOptions.Delegate.GPU

    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK, delegate=gpu_delegate),
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK, delegate=gpu_delegate),
        min_pose_detection_confidence=0.5,
    )
    f_opt = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_TASK, delegate=gpu_delegate),
        min_face_detection_confidence=0.5,
    )

    # Model A (Shape) Interpreter - Also GPU if library supports it
    interpreter = tf.lite.Interpreter(model_path=SHAPE_MODEL_PATH)
    interpreter.allocate_tensors()

    return (
        interpreter,
        vision.HandLandmarker.create_from_options(h_opt),
        vision.PoseLandmarker.create_from_options(p_opt),
        vision.FaceLandmarker.create_from_options(f_opt),
    )


def get_shape_probabilities(interpreter, landmarks):
    """Calculates Model A probabilities for a single hand."""
    wrist = landmarks[0]
    m_knuckle = landmarks[9]
    scale = max(
        ((m_knuckle.x - wrist.x) ** 2 + (m_knuckle.y - wrist.y) ** 2) ** 0.5, 1e-6
    )

    feats = []
    # 42 Coords
    for lm in landmarks:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
    # 5 Extensions
    tips, knks = [4, 8, 12, 16, 20], [2, 5, 9, 13, 17]
    for t, k in zip(tips, knks):
        d_t = ((landmarks[t].x - wrist.x) ** 2 + (landmarks[t].y - wrist.y) ** 2) ** 0.5
        d_k = ((landmarks[k].x - wrist.x) ** 2 + (landmarks[k].y - wrist.y) ** 2) ** 0.5
        feats.append(d_t / max(d_k, 1e-6))
    # 4 Spreads
    for p1, p2 in [(8, 12), (12, 16), (16, 20), (4, 8)]:
        d = (
            (landmarks[p1].x - landmarks[p2].x) ** 2
            + (landmarks[p1].y - landmarks[p2].y) ** 2
        ) ** 0.5
        feats.append(d / scale)

    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(in_idx, np.array([feats], dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0]


def extract_frame_features(frame, shape_int, h_mod, p_mod, f_mod):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect on GPU
    h_res = h_mod.detect(mp_img)
    p_res = p_mod.detect(mp_img)
    f_res = f_mod.detect(mp_img)

    # 1. Body Pillars (N, M, LS, RS, C)
    pillars = [[0.5, 0.4], [0.5, 0.5], [0.3, 0.7], [0.7, 0.7], [0.5, 0.7]]
    f_scale = 0.2

    if f_res.face_landmarks:
        fl = f_res.face_landmarks[0]
        pillars[0] = [fl[1].x, fl[1].y]  # Nose
        pillars[1] = [(fl[13].x + fl[14].x) / 2, (fl[13].y + fl[14].y) / 2]  # Mouth
        f_scale = max(abs(fl[10].y - fl[152].y), 1e-6)

    if p_res.pose_landmarks:
        pl = p_res.pose_landmarks[0]
        pillars[2] = [pl[11].x, pl[11].y]  # L_Shoulder
        pillars[3] = [pl[12].x, pl[12].y]  # R_Shoulder
        pillars[4] = [(pl[11].x + pl[12].x) / 2, (pl[11].y + pl[12].y) / 2]  # Chest

    num_shapes = shape_int.get_output_details()[0]["shape"][-1]
    full_vector = []

    # 2. Process Two Hand Slots
    for i in range(2):
        if h_res.hand_landmarks and i < len(h_res.hand_landmarks):
            lms = h_res.hand_landmarks[i]
            probs = get_shape_probabilities(shape_int, lms)

            wrist = np.array([lms[0].x, lms[0].y])
            tips_mean = np.mean(
                [[lms[t].x, lms[t].y] for t in [4, 8, 12, 16, 20]], axis=0
            )

            # Distances to Pillars
            w_dists = [np.linalg.norm(wrist - np.array(p)) / f_scale for p in pillars]
            t_dists = [
                np.linalg.norm(tips_mean - np.array(p)) / f_scale for p in pillars
            ]

            full_vector += list(probs) + w_dists + t_dists + [1.0]  # + Presence
        else:
            full_vector += [0.0] * num_shapes + [0.0] * 5 + [0.0] * 5 + [0.0]

    return full_vector


# ------------------- Dataset Builder -------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = create_models()

    # Identify labels from subdirectories
    labels = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )
    label_map = {l: i for i, l in enumerate(labels)}

    X, y = [], []
    for label in labels:
        label_path = os.path.join(ROOT_DIR, label)
        videos = [f for f in os.listdir(label_path) if f.endswith(".mp4")]
        print(f"AWS Processing: {label} ({len(videos)} videos)")

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(label_path, v))
            raw_seq = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_seq.append(extract_frame_features(frame, *models))
            cap.release()

            if len(raw_seq) > 5:
                X.append(resample_sequence(raw_seq, TARGET_FRAMES))
                y.append(label_map[label])

    np.save(f"{OUTPUT_DIR}/X.npy", np.array(X, dtype=np.float32))
    np.save(f"{OUTPUT_DIR}/y.npy", np.array(y))
    print(f"Finished. Final X Shape: {np.array(X).shape}")
