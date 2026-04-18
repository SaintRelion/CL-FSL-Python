import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from mediapipe.tasks.python import vision
from fsl_helper import resample_sequence

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
OUTPUT_DIR = "data"
TARGET_FRAMES = 30
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"

# Pillars: Nose, Mouth, L_Shoulder, R_Shoulder, Chest, L_Ear, R_Ear
PILLAR_COUNT = 7


def create_models():
    BaseOptions = mp.tasks.BaseOptions
    gpu = BaseOptions.Delegate.GPU  # Acceleration for AWS L4

    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/hand_landmarker.task", delegate=gpu
        ),
        num_hands=2,
    )
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/pose_landmarker_full.task", delegate=gpu
        )
    )
    f_opt = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/face_landmarker.task", delegate=gpu
        )
    )

    # Model A Interpreter (To attribute anatomy during collection)
    interpreter = tf.lite.Interpreter(model_path=SHAPE_MODEL_PATH)
    interpreter.allocate_tensors()

    return (
        interpreter,
        vision.HandLandmarker.create_from_options(h_opt),
        vision.PoseLandmarker.create_from_options(p_opt),
        vision.FaceLandmarker.create_from_options(f_opt),
    )


def get_shape_id(interpreter, landmarks):
    """Returns the most likely shape index from Model A."""
    wrist = landmarks[0]
    m_knk = landmarks[9]
    scale = max(((m_knk.x - wrist.x) ** 2 + (m_knk.y - wrist.y) ** 2) ** 0.5, 1e-6)

    feats = []
    for lm in landmarks:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
    # Extensions and Spread logic (simplified for speed)
    for t, k in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        feats.append(
            (((landmarks[t].x - wrist.x) ** 2 + (landmarks[t].y - wrist.y) ** 2) ** 0.5)
            / max(
                (
                    ((landmarks[k].x - wrist.x) ** 2 + (landmarks[k].y - wrist.y) ** 2)
                    ** 0.5
                ),
                1e-6,
            )
        )
    for p1, p2 in [(8, 12), (12, 16), (16, 20), (4, 8)]:
        feats.append(
            (
                (
                    (landmarks[p1].x - landmarks[p2].x) ** 2
                    + (landmarks[p1].y - landmarks[p2].y) ** 2
                )
                ** 0.5
            )
            / scale
        )

    interpreter.set_tensor(
        interpreter.get_input_details()[0]["index"], np.array([feats], dtype=np.float32)
    )
    interpreter.invoke()
    return np.argmax(
        interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
    )


def extract_frame_features(frame, shape_int, h_mod, p_mod, f_mod):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    h_res, p_res, f_res = (
        h_mod.detect(mp_img),
        p_mod.detect(mp_img),
        f_mod.detect(mp_img),
    )

    # 1. 7-Point Pillar Grid
    pillars = [
        [0.5, 0.4],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.7, 0.7],
        [0.5, 0.7],
        [0.4, 0.3],
        [0.6, 0.3],
    ]
    f_scale = 0.2

    if f_res.face_landmarks:
        fl = f_res.face_landmarks[0]
        pillars[0], pillars[1] = [fl[1].x, fl[1].y], [
            (fl[13].x + fl[14].x) / 2,
            (fl[13].y + fl[14].y) / 2,
        ]
        pillars[5], pillars[6] = [fl[234].x, fl[234].y], [fl[454].x, fl[454].y]
        f_scale = max(abs(fl[10].y - fl[152].y), 1e-6)
    if p_res.pose_landmarks:
        pl = p_res.pose_landmarks[0]
        pillars[2], pillars[3] = [pl[11].x, pl[11].y], [pl[12].x, pl[12].y]
        pillars[4] = [(pl[11].x + pl[12].x) / 2, (pl[11].y + pl[12].y) / 2]

    full_vector = []
    shapes_in_frame = [-1, -1]  # Track shape IDs for this frame

    for i in range(2):
        if h_res.hand_landmarks and i < len(h_res.hand_landmarks):
            lms = h_res.hand_landmarks[i]
            shapes_in_frame[i] = get_shape_id(shape_int, lms)

            wrist = np.array([lms[0].x, lms[0].y])
            tips = np.mean([[lms[t].x, lms[t].y] for t in [4, 8, 12, 16, 20]], axis=0)

            # 7 Dists for Wrist + 7 Dists for Tips = 14 spatial features
            w_dists = [np.linalg.norm(wrist - np.array(p)) / f_scale for p in pillars]
            t_dists = [np.linalg.norm(tips - np.array(p)) / f_scale for p in pillars]
            full_vector += w_dists + t_dists + [1.0]
        else:
            full_vector += [0.0] * (PILLAR_COUNT * 2) + [0.0]

    return full_vector, shapes_in_frame


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = create_models()

    labels = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )
    anatomy_data = {}
    X, y = [], []

    for label in labels:
        path = os.path.join(ROOT_DIR, label)
        videos = [f for f in os.listdir(path) if f.endswith(".mp4")]

        # Track dominant shapes for this gesture
        all_h1_shapes, all_h2_shapes = [], []

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(path, v))
            raw_seq, shape_seq = [], []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                feat, shapes = extract_frame_features(frame, *models)
                raw_seq.append(feat)
                shape_seq.append(shapes)
            cap.release()

            if len(raw_seq) > 5:
                X.append(resample_sequence(raw_seq, TARGET_FRAMES))
                y.append(labels.index(label))
                # Add shapes to the anatomy tracker
                all_h1_shapes.extend([s[0] for s in shape_seq if s[0] != -1])
                all_h2_shapes.extend([s[1] for s in shape_seq if s[1] != -1])

        # Attribute dominant shapes to the label
        dom_h1 = (
            max(set(all_h1_shapes), key=all_h1_shapes.count) if all_h1_shapes else -1
        )
        dom_h2 = (
            max(set(all_h2_shapes), key=all_h2_shapes.count) if all_h2_shapes else -1
        )
        anatomy_data[label] = {"h1_shape": int(dom_h1), "h2_shape": int(dom_h2)}

    np.save(f"{OUTPUT_DIR}/X.npy", np.array(X, dtype=np.float32))
    np.save(f"{OUTPUT_DIR}/y.npy", np.array(y))
    with open(f"{OUTPUT_DIR}/gesture_anatomy.json", "w") as f:
        json.dump(anatomy_data, f, indent=2)
    print(f"Dataset Built. Feature Count: {len(X[0][0])}. Anatomy Mapped.")
