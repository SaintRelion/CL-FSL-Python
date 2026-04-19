import cv2
import mediapipe as mp
import numpy as np
import json
import os
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
SIGNATURES_PATH = "data/pillar_signatures.json"
POSE_TASK = "models/pose_landmarker_full.task"
ROOT_DIR = "common_clips"

with open(SIGNATURES_PATH, "r") as f:
    GESTURE_SIGS = json.load(f)


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_TASK),
            running_mode=vision.RunningMode.IMAGE,
        )
    )


landmarker = create_landmarker()
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
                result = landmarker.detect(mp_image)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]

                    # --- REFINED SCALING ---
                    # Use the vertical distance from Nose(0) to Mid-Shoulder as the unit
                    mid_shoulder_y = (lms[11].y + lms[12].y) / 2
                    scale = max(abs(mid_shoulder_y - lms[0].y), 0.05)

                    head_size = scale  # Use this for derived pillars
                    live_pillars = {
                        "nose": (lms[0].x, lms[0].y),
                        "left_ear": (lms[7].x, lms[7].y),
                        "right_ear": (lms[8].x, lms[8].y),
                        "left_shoulder": (lms[11].x, lms[11].y),
                        "right_shoulder": (lms[12].x, lms[12].y),
                        "forehead": (
                            lms[0].x,
                            lms[0].y - (head_size * 0.8),
                        ),  # Adjusted
                        "chin": (lms[0].x, lms[0].y + (head_size * 0.4)),
                        "chest": (
                            (lms[11].x + lms[12].x) / 2,
                            (lms[11].y + lms[12].y) / 2,
                        ),
                    }
                    hand = (lms[16].x, lms[16].y)  # Right wrist

                    # --- GRID UI ---
                    cols = 4
                    rows_per_col = (len(all_labels) + cols - 1) // cols
                    col_w = w // cols

                    for idx, sign in enumerate(all_labels):
                        c, r = idx // rows_per_col, idx % rows_per_col
                        x, y = 10 + (c * col_w), 30 + (r * 20)

                        sig_data = GESTURE_SIGS[sign]
                        best_alpha = 0.0

                        for p_name, stats in sig_data.items():
                            if p_name not in live_pillars:
                                continue
                            p_pos = live_pillars[p_name]

                            # Normalized Distance
                            raw_dist = np.sqrt(
                                (hand[0] - p_pos[0]) ** 2 + (hand[1] - p_pos[1]) ** 2
                            )
                            dist = raw_dist / scale

                            # Z-score clipping: If std is too high, it makes everything light up.
                            # We cap the impact of high variance.
                            std = max(stats["std"], 0.01)
                            z = (dist - stats["mean"]) / std

                            # ALPHA CALCULATION
                            # If hand is exactly at mean, alpha = 1.0.
                            # If hand is 2 standard deviations away, alpha = 0.0.
                            current_alpha = np.clip(1.0 - (abs(z) / 2.0), 0.0, 1.0)
                            if current_alpha > best_alpha:
                                best_alpha = current_alpha

                        # --- COLOR LOGIC ---
                        if best_alpha > 0.7:
                            color = (0, 255, 0)  # Green
                            thickness = 2
                        elif best_alpha > 0.1:
                            color = (0, 255, 255)  # Yellow
                            thickness = 1
                        else:
                            color = (60, 60, 60)  # Dark Gray
                            thickness = 1

                        cv2.putText(
                            ui_overlay,
                            f"{sign}: {best_alpha:.2f}",
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            thickness,
                        )

                frame = cv2.addWeighted(frame, 0.5, ui_overlay, 0.8, 0)
                cv2.imshow("FSL Precision Grid", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    exit()
            cap.release()
