# python
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import vision
import os

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
SHAPE_MODEL_PATH = "models/hand_shape_model.tflite"
SHAPE_LABELS_PATH = "models/hand_shape_labels.txt"
HAND_TASK = "models/hand_landmarker.task"
POSE_TASK = "models/pose_landmarker_full.task"
FACE_TASK = "models/face_landmarker.task"

# 1. Load Data Structure
LABELS = sorted(
    [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
)
with open(SHAPE_LABELS_PATH, "r") as f:
    SHAPE_LABELS = [line.strip() for line in f.readlines()]


# 2. Initialize Models
def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


s_interp, s_in, s_out = load_tflite(SHAPE_MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
h_mod = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK), num_hands=2
    )
)
p_mod = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=POSE_TASK))
)
f_mod = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=FACE_TASK))
)


def get_shape_prediction(lms):
    wrist, m_knk = lms[0], lms[9]
    scale = max(((m_knk.x - wrist.x) ** 2 + (m_knk.y - wrist.y) ** 2) ** 0.5, 1e-6)
    feats = []
    for lm in lms:
        feats.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
    for t, k in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
        feats.append(
            (((lms[t].x - wrist.x) ** 2 + (lms[t].y - wrist.y) ** 2) ** 0.5)
            / max(
                (((lms[k].x - wrist.x) ** 2 + (lms[k].y - wrist.y) ** 2) ** 0.5), 1e-6
            )
        )
    for p1, p2 in [(8, 12), (12, 16), (16, 20), (4, 8)]:
        feats.append(
            (((lms[p1].x - lms[p2].x) ** 2 + (lms[p1].y - lms[p2].y) ** 2) ** 0.5)
            / scale
        )

    s_interp.set_tensor(s_in[0]["index"], np.array([feats], dtype=np.float32))
    s_interp.invoke()
    probs = s_interp.get_tensor(s_out[0]["index"])[0]
    idx = np.argmax(probs)
    return SHAPE_LABELS[idx], probs[idx]


# --- NAVIGATOR STATE ---
L_IDX = 0  # Label Index
V_IDX = 0  # Video Index

while True:
    label_name = LABELS[L_IDX]
    video_list = sorted(
        [
            f
            for f in os.listdir(os.path.join(ROOT_DIR, label_name))
            if f.endswith(".mp4")
        ]
    )

    if not video_list:
        L_IDX = (L_IDX + 1) % len(LABELS)
        continue

    v_name = video_list[V_IDX % len(video_list)]
    cap = cv2.VideoCapture(os.path.join(ROOT_DIR, label_name, v_name))

    print(f"DEBUGGING: {label_name} -> {v_name}")

    should_skip_video = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        h_res, p_res, f_res = (
            h_mod.detect(mp_img),
            p_mod.detect(mp_img),
            f_mod.detect(mp_img),
        )

        # 1. Draw Body Pillars
        pillars = [
            [0.5, 0.4],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.7, 0.7],
            [0.5, 0.7],
            [0.4, 0.3],
            [0.6, 0.3],
        ]
        if f_res.face_landmarks:
            fl = f_res.face_landmarks[0]
            pillars[0], pillars[1] = [fl[1].x, fl[1].y], [
                (fl[13].x + fl[14].x) / 2,
                (fl[13].y + fl[14].y) / 2,
            ]
            pillars[5], pillars[6] = [fl[234].x, fl[234].y], [fl[454].x, fl[454].y]
        if p_res.pose_landmarks:
            pl = p_res.pose_landmarks[0]
            pillars[2], pillars[3] = [pl[11].x, pl[11].y], [pl[12].x, pl[12].y]
            pillars[4] = [(pl[11].x + pl[12].x) / 2, (pl[11].y + pl[12].y) / 2]
        for p in pillars:
            cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 5, (255, 0, 0), -1)

        # 2. Draw Hand Shape & Detection
        if h_res.hand_landmarks:
            for i, lms in enumerate(h_res.hand_landmarks):
                shape, conf = get_shape_prediction(lms)
                for lm in lms:
                    cv2.circle(
                        frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1
                    )
                cv2.putText(
                    frame,
                    f"H{i}: {shape} ({conf*100:.0f}%)",
                    (int(lms[0].x * w), int(lms[0].y * h) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "HAND LOST",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        # UI HUD
        cv2.putText(
            frame,
            f"LABEL: {label_name} ({L_IDX+1}/{len(LABELS)})",
            (20, 30),
            0,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"VIDEO: {v_name} ({V_IDX+1}/{len(video_list)})",
            (20, 60),
            0,
            0.6,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "Keys: [RIGHT] Next Video | [DOWN] Next Label | [ESC] Exit",
            (20, h - 20),
            0,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.imshow("Anatomy Debug Suite", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == 83 or key == ord("d"):  # Right Arrow or D
            V_IDX += 1
            should_skip_video = True
            break
        elif key == 81 or key == ord("a"):  # Left Arrow or A
            V_IDX = max(0, V_IDX - 1)
            should_skip_video = True
            break
        elif key == 84 or key == ord("s"):  # Down Arrow or S
            L_IDX = (L_IDX + 1) % len(LABELS)
            V_IDX = 0
            should_skip_video = True
            break
        elif key == 82 or key == ord("w"):  # Up Arrow or W
            L_IDX = (L_IDX - 1) % len(LABELS)
            V_IDX = 0
            should_skip_video = True
            break

    cap.release()
    if not should_skip_video:
        V_IDX = (V_IDX + 1) % len(video_list)
