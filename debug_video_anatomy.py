import cv2
import mediapipe as mp
import numpy as np
import json
import os
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SIGNATURES_PATH = "data/pillar_signatures.json"
POSE_TASK = "models/pose_landmarker_full.task"
HAND_TASK = "models/hand_landmarker.task"
ROOT_DIR = "common_clips"

with open(SIGNATURES_PATH, "r") as f:
    GESTURE_SIGS = json.load(f)

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
    p_opt = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK),
        running_mode=vision.RunningMode.IMAGE,
    )
    h_opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return vision.PoseLandmarker.create_from_options(
        p_opt
    ), vision.HandLandmarker.create_from_options(h_opt)


# Initialize
p_landmarker, h_landmarker = create_models()
all_labels = sorted(GESTURE_SIGS.keys())

while True:
    for current_label in all_labels:
        video_path = os.path.join(ROOT_DIR, current_label)
        video_list = sorted(
            [v for v in os.listdir(video_path) if v.endswith((".mp4", ".avi"))]
        )

        for v_name in video_list:
            cap = cv2.VideoCapture(os.path.join(video_path, v_name))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                ui_overlay = np.zeros_like(frame)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                p_result = p_landmarker.detect(mp_image)
                h_result = h_landmarker.detect(mp_image)

                if p_result.pose_landmarks and h_result.hand_landmarks:
                    p_lms = p_result.pose_landmarks[0]
                    h_lms = h_result.hand_landmarks[0]

                    # 1. DYNAMIC SCALE (Shoulder to Shoulder)
                    scale = np.sqrt(
                        (p_lms[11].x - p_lms[12].x) ** 2
                        + (p_lms[11].y - p_lms[12].y) ** 2
                    )
                    scale = max(scale, 0.05)

                    # 2. TARGET POINT (Index Finger Tip - Landmark 8)
                    tip = (h_lms[8].x, h_lms[8].y)

                    # 3. LIVE PILLARS
                    nose = p_lms[0]
                    head_size = abs(p_lms[11].y - nose.y)
                    live_pillars = {
                        "nose": (nose.x, nose.y),
                        "left_ear": (p_lms[7].x, p_lms[7].y),
                        "right_ear": (p_lms[8].x, p_lms[8].y),
                        "left_shoulder": (p_lms[11].x, p_lms[11].y),
                        "right_shoulder": (p_lms[12].x, p_lms[12].y),
                        "forehead": (nose.x, nose.y - (head_size * 0.3)),
                        "chin": (nose.x, nose.y + (head_size * 0.25)),
                        "chest": (
                            (p_lms[11].x + p_lms[12].x) / 2,
                            (p_lms[11].y + p_lms[12].y) / 2,
                        ),
                    }

                    # --- GRID UI (4 COLUMNS) ---
                    cols = 4
                    rows_per_col = (len(all_labels) + cols - 1) // cols
                    col_w = w // cols

                    for idx, sign in enumerate(all_labels):
                        c, r = idx // rows_per_col, idx % rows_per_col
                        x, y = 10 + (c * col_w), 30 + (r * 20)

                        sig_data = GESTURE_SIGS[sign]
                        best_alpha = 0.0

                        for p_name, stats in sig_data.items():
                            p_pos = live_pillars[p_name]

                            # Normalized Distance
                            raw_dist = np.sqrt(
                                (tip[0] - p_pos[0]) ** 2 + (tip[1] - p_pos[1]) ** 2
                            )
                            dist = raw_dist / scale

                            # Calculate Z-score
                            # We add a small 'stiffness' constant to the denominator to make it even stricter
                            stiffness = 0.8
                            z = (dist - stats["mean"]) / (stats["std"] * stiffness)

                            # GAUSSIAN ALPHA: Exponentially stricter than linear
                            # Perfect match (z=0) = 1.0
                            # 1 StdDev away (z=1) = 0.60
                            # 2 StdDev away (z=2) = 0.13 (Almost invisible)
                            current_alpha = np.exp(-0.5 * (z**2))

                            if current_alpha > best_alpha:
                                best_alpha = current_alpha

                        # --- NEW STRICT COLOR LOGIC ---
                        if best_alpha > 0.85:  # Hard Green only for center-hits
                            color = (0, 255, 0)
                            thick = 2
                        elif best_alpha > 0.4:  # Yellow for "Close enough"
                            color = (0, 255, 255)
                            thick = 1
                        else:  # Gray out everything else
                            color = (45, 45, 45)
                            thick = 1

                        cv2.putText(
                            ui_overlay,
                            f"{sign}: {best_alpha:.2f}",
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1 if best_alpha < 0.7 else 2,
                        )

                    # Draw visual markers on main frame
                    cv2.circle(
                        frame, (int(tip[0] * w), int(tip[1] * h)), 6, (255, 0, 255), -1
                    )  # Fingertip

                frame = cv2.addWeighted(frame, 0.5, ui_overlay, 0.8, 0)
                cv2.imshow("FSL Precision Grid", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    exit()
            cap.release()
