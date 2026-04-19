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

# Load Statistical Signatures
with open(SIGNATURES_PATH, "r") as f:
    GESTURE_SIGS = json.load(f)

# Pillars extracted by the signature generator
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
    """Derived pillars exactly matching the generator logic."""
    nose = lms[0]
    l_ear = lms[7]
    r_ear = lms[8]
    l_shl = lms[11]
    r_shl = lms[12]

    head_size = abs(l_shl.y - nose.y)
    pillars = {
        "nose": (nose.x, nose.y),
        "left_ear": (l_ear.x, l_ear.y),
        "right_ear": (r_ear.x, r_ear.y),
        "left_shoulder": (l_shl.x, l_shl.y),
        "right_shoulder": (r_shl.x, r_shl.y),
        "forehead": (nose.x, nose.y - (head_size * 0.25)),
        "chin": (nose.x, nose.y + (head_size * 0.2)),
        "chest": ((l_shl.x + r_shl.x) / 2, (l_shl.y + r_shl.y) / 2),
    }
    return pillars


# --- MAIN DEBUG LOOP ---
landmarker = create_landmarker()
labels = sorted(GESTURE_SIGS.keys())
L_IDX = 0

while True:
    current_label = labels[L_IDX]
    video_list = [v for v in os.listdir(os.path.join(ROOT_DIR, current_label))]

    for v_name in video_list:
        cap = cv2.VideoCapture(os.path.join(ROOT_DIR, current_label, v_name))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect(mp_image)

            active_culling_list = []

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                live_pillars = get_live_pillars(lms)
                # Use Right Wrist (16) as dominant hand
                hand = (lms[16].x, lms[16].y)

                # PILLAR ACTIVATION LOGIC
                for sign_name, sig_data in GESTURE_SIGS.items():
                    # Check every pillar defined for this sign
                    is_sign_active = False
                    for p_name, stats in sig_data.items():
                        p_pos = live_pillars[p_name]
                        dist = np.sqrt(
                            (hand[0] - p_pos[0]) ** 2 + (hand[1] - p_pos[1]) ** 2
                        )

                        # TRIGGER: Is hand within the 2-Sigma window?
                        if dist <= stats["threshold_2sigma"]:
                            is_sign_active = True
                            break  # One pillar match triggers culling inclusion

                    if is_sign_active:
                        active_culling_list.append(sign_name)

                # --- VISUALS ---
                # Draw Pillars
                for p_name, pos in live_pillars.items():
                    cv2.circle(
                        frame, (int(pos[0] * w), int(pos[1] * h)), 8, (0, 255, 0), -1
                    )

                # Draw Active Culling List
                cv2.putText(
                    frame,
                    f"ACTIVE SIGNS: {', '.join(active_culling_list[:5])}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Pillar Activation Debug", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                exit()

        cap.release()
