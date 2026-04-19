import cv2
import mediapipe as mp
import numpy as np
import json
import os
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
ROOT_DIR = "common_clips"
SIGNATURES_PATH = "data/pillar_signatures.json"
POSE_TASK = "models/pose_landmarker_full.task"

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


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_TASK),
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def get_live_pillars(lms):
    nose, l_ear, r_ear, l_shl, r_shl = lms[0], lms[7], lms[8], lms[11], lms[12]
    h_size = abs(l_shl.y - nose.y)
    return {
        "nose": (nose.x, nose.y),
        "left_ear": (l_ear.x, l_ear.y),
        "right_ear": (r_ear.x, r_ear.y),
        "left_shoulder": (l_shl.x, l_shl.y),
        "right_shoulder": (r_shl.x, r_shl.y),
        "forehead": (nose.x, nose.y - (h_size * 0.25)),
        "chin": (nose.x, nose.y + (h_size * 0.2)),
        "chest": ((l_shl.x + r_shl.x) / 2, (l_shl.y + r_shl.y) / 2),
    }


landmarker = create_landmarker()
# Sort labels alphabetically for the grid
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

                # Dark UI Overlay for visibility
                ui_overlay = np.zeros_like(frame)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect(mp_image)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    live_pillars = get_live_pillars(lms)
                    # Right wrist (16) is the target
                    hand = (lms[16].x, lms[16].y)

                    # Draw Pillars on the original frame
                    for p_name, pos in live_pillars.items():
                        cv2.circle(
                            frame,
                            (int(pos[0] * w), int(pos[1] * h)),
                            4,
                            (0, 255, 0),
                            -1,
                        )

                    # --- MULTI-COLUMN GRID LAYOUT ---
                    cols = 4  # Adjust columns based on your screen width
                    rows_per_col = (len(all_labels) // cols) + 1
                    col_width = w // cols
                    font_scale = 0.45
                    line_height = 22

                    for idx, sign in enumerate(all_labels):
                        c = idx // rows_per_col
                        r = idx % rows_per_col

                        x = 15 + (c * col_width)
                        y = 40 + (r * line_height)

                        # Calculate Activation Logic
                        sig_data = GESTURE_SIGS[sign]
                        best_z = 99.0

                        for p_name, stats in sig_data.items():
                            p_pos = live_pillars[p_name]
                            dist = np.sqrt(
                                (hand[0] - p_pos[0]) ** 2 + (hand[1] - p_pos[1]) ** 2
                            )

                            # Z-Score: dist from mean / std
                            z = (
                                (dist - stats["mean"]) / stats["std"]
                                if stats["std"] > 0
                                else 99
                            )
                            if z < best_z:
                                best_z = z

                        # Activation UI Logic
                        # 0.0 Z-score = 1.0 alpha (Green)
                        # 2.0 Z-score = 0.2 alpha (Gray)
                        alpha = np.clip(1.0 - (best_z / 2.0), 0.1, 1.0)

                        if alpha > 0.6:
                            color = (0, 255, 255)  # Bright Yellow for active
                            thickness = 2
                        elif alpha > 0.2:
                            color = (0, 150, 150)  # Dimmer cyan
                            thickness = 1
                        else:
                            color = (80, 80, 80)  # Gray for inactive
                            thickness = 1

                        text = f"{sign}: {alpha:.2f}"
                        cv2.putText(
                            ui_overlay,
                            text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            thickness,
                        )

                        # Background highlight for active signs
                        if alpha > 0.8:
                            cv2.rectangle(
                                ui_overlay,
                                (x - 5, y - 15),
                                (x + col_width - 20, y + 5),
                                (0, 255, 0),
                                1,
                            )

                # Blend UI with frame
                frame = cv2.addWeighted(frame, 0.4, ui_overlay, 0.6, 0)

                cv2.imshow("FSL Statistical Culling Dashboard", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    exit()
            cap.release()
