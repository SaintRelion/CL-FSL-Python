import os
import json
import time
import threading
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
from fsl_helper import forward_fill, shoulder_anchor
from build_dataset import extract_sequence_features, resample_sequence

# ---------------- Config ----------------
MODEL_FILE = "models/gru_gesture_model.keras"
LABEL_MAP_FILE = "data/label_map.json"

TARGET_FRAMES = 30
EXPECTED_DIM = 76
TOP_K = 3

FPS = 30
CAPTURE_SECONDS = 3
COOLDOWN_SECONDS = 1.0

# ---------------- Load model + labels ----------------
model = tf.keras.models.load_model(MODEL_FILE)

with open(LABEL_MAP_FILE) as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ---------------- MediaPipe Tasks ----------------
BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision

hand_model = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
    )
)

pose_model = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
        running_mode=vision.RunningMode.IMAGE,
    )
)

face_model = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
    )
)

# SAME helpers as training
from build_dataset import  resample_sequence

def enhance_brightness_contrast(frame, gamma=1.2, alpha=1.2, beta=10):
    # Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)
    
    # Alpha/Beta adjust (contrast/brightness)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return frame

def enhance_edges(frame):
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    enhanced = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    return enhanced

def draw_pose_and_hands(frame_bgr, pose_result=None, hand_result=None):
    annotated = frame_bgr.copy()

    h, w, _ = annotated.shape

    # ---------------- Draw POSE ----------------
    if pose_result and pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:

            # Draw full pose lightly (optional)
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 200, 0), thickness=1
                )
            )

            # Emphasize shoulders + wrists
            for idx in [11, 12, 15, 16]:  # L/R shoulder, L/R wrist
                lm = pose_landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (x, y), 6, (255, 0, 255), -1)
                
    # ---------------- Draw HANDS ----------------
    if hand_result and hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=hand_landmarks,
                connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
            )

    return annotated

def extract_frame_features_image(
    pose_result,
    hand_result,
    face_result,
    prev
):
    pose_feat = []
    hand_feat = []
    face_feat = []

    scale = 1.0
    wrist_positions = []

    # ---------------- Pose ----------------
    if pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]
        _, scale = shoulder_anchor(pose_result)
        scale = max(scale, 1e-6)

        for side in ["LEFT", "RIGHT"]:
            w_idx = 15 if side == "LEFT" else 16
            s_idx = 11 if side == "LEFT" else 12

            if w_idx < len(pose_landmarks) and s_idx < len(pose_landmarks):
                wrist = pose_landmarks[w_idx]
                shoulder = pose_landmarks[s_idx]
                vec = (
                    np.array([shoulder.x, shoulder.y, shoulder.z]) -
                    np.array([wrist.x, wrist.y, wrist.z])
                ) / scale
                pose_feat.extend([*vec, 1])
            else:
                pose_feat.extend([0, 0, 0, 0])
    else:
        pose_feat = [0] * 8

    # ---------------- Hands ----------------
    hand_data = [None, None]
    if hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks[:2]):
            hand_data[i] = hand

    for h in hand_data:
        if h:
            wrist = np.array([h[0].x, h[0].y, h[0].z])
            wrist_positions.append(wrist)
            hand_feat.extend([*wrist, 1])

            for tip in [4, 8, 12, 16, 20]:
                if tip < len(h):
                    lm = h[tip]
                    vec = (np.array([lm.x, lm.y, lm.z]) - wrist) / scale
                    hand_feat.extend([*vec, 1])
                else:
                    hand_feat.extend([0, 0, 0, 0])
        else:
            hand_feat.extend([0] * 24)
            wrist_positions.append(None)

    # Wrist ↔ Wrist
    if wrist_positions[0] is not None and wrist_positions[1] is not None:
        vec = (wrist_positions[1] - wrist_positions[0]) / scale
        hand_feat.extend([*vec, 1])
    else:
        hand_feat.extend([0, 0, 0, 0])

    # ---------------- Face ----------------
    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]

        mouth = (
            np.array([face_landmarks[13].x, face_landmarks[13].y, face_landmarks[13].z]) +
            np.array([face_landmarks[14].x, face_landmarks[14].y, face_landmarks[14].z])
        ) / 2

        eye = (
            np.array([face_landmarks[33].x, face_landmarks[33].y, face_landmarks[33].z]) +
            np.array([face_landmarks[263].x, face_landmarks[263].y, face_landmarks[263].z])
        ) / 2

        for w in wrist_positions:
            if w is not None:
                face_feat.extend([*(mouth - w) / scale, 1, *(eye - w) / scale, 1])
            else:
                face_feat.extend([0] * 8)
    else:
        face_feat.extend([0] * 16)

    feat = pose_feat + hand_feat + face_feat
    if len(feat) != 76:
        return None, prev

    feat = forward_fill(feat, prev)
    return feat, feat

# ---------------- Async processing ----------------
def async_process_frame_buffer(frames):
    raw = extract_sequence_features(
        frames=frames,
        hand_model=hand_model,
        pose_model=pose_model,
        face_model=face_model,
        show_progress=False
    )

    if raw is None or len(raw) < 5:
        print("⚠️ Not enough valid frames")
        return

    frames_feat = resample_sequence(raw, TARGET_FRAMES)
    X = np.array(frames_feat, dtype=np.float32)

    if X.shape != (TARGET_FRAMES, EXPECTED_DIM):
        print("⚠️ Shape mismatch:", X.shape)
        return

    probs = model.predict(X[None, ...], verbose=0)[0]
    topk = np.argsort(probs)[::-1][:TOP_K]

    print("\n🎯 Async Top predictions:")
    for i in topk:
        print(f"   {inv_label_map[i]:<20} {probs[i]:.4f}")
    print()

# -------------------------------
# Main loop
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    buffering = False
    frame_buffer = []
    start_time = None
    cooldown_until = 0.0

    print(f"\n🎥 Realtime GRU Gesture Recognition ({CAPTURE_SECONDS}s window)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Detect hand ONLY ---
        # frame = cv2.resize(frame, (640, 480))
        # frame = enhance_brightness_contrast(frame, gamma=1.3, alpha=1.3, beta=15)
        # frame = enhance_edges(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # --- Hand detection only for trigger ---
        if not buffering and time.time() >= cooldown_until:
            hand_result = hand_model.detect(mp_image)
            hand_detected = bool(hand_result.hand_landmarks)

        # Start capture
        if hand_detected and not buffering:
            buffering = True
            frame_buffer = []
            start_time = time.time()
            print("🟢 Gesture capture started")

        # Buffer frames (no detection)
        if buffering:
            frame_buffer.append(frame.copy())

            if time.time() - start_time >= CAPTURE_SECONDS:
                buffering = False
                print(f"🔵 Captured {len(frame_buffer)} frames")
                threading.Thread(target=async_process_frame_buffer, args=(frame_buffer.copy(),)).start()
                frame_buffer = []
                cooldown_until = time.time() + COOLDOWN_SECONDS

                hand_detected = False

        # UI
        status_text = "CAPTURING..." if buffering else "WAITING FOR HAND" if time.time()>=cooldown_until else "COOLDOWN..."
        status_color = (0,255,0) if buffering else (0,0,255) if time.time()>=cooldown_until else (0,255,255)
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.imshow("Realtime GRU Gesture", frame)

        if cv2.waitKey(1) & 0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
