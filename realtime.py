import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

BaseOptions = mp.tasks.BaseOptions

hand_model = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,  # IMAGE mode for more control
        min_hand_detection_confidence=0.3,
        
    )
)

pose_model = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
)

def draw_pose_and_hands(frame_bgr, pose_result=None, hand_result=None):
    annotated = frame_bgr.copy()

    h, w, _ = annotated.shape

    # # ---------------- Draw POSE ----------------
    # if pose_result and pose_result.pose_landmarks:
    #     for pose_landmarks in pose_result.pose_landmarks:

    #         # Draw full pose lightly (optional)
    #         drawing_utils.draw_landmarks(
    #             image=annotated,
    #             landmark_list=pose_landmarks,
    #             connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
    #             landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
    #             connection_drawing_spec=drawing_utils.DrawingSpec(
    #                 color=(0, 200, 0), thickness=1
    #             )
    #         )

    #         # Emphasize shoulders + wrists
    #         for idx in [11, 12, 15, 16]:  # L/R shoulder, L/R wrist
    #             lm = pose_landmarks[idx]
    #             x, y = int(lm.x * w), int(lm.y * h)
    #             cv2.circle(annotated, (x, y), 6, (255, 0, 255), -1)
                
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

def enhance_frame_for_hand_detection(frame):
    """
    Enhance frame to improve hand detection.
    Steps:
    1. Slight brightness & contrast adjustment
    2. Gamma correction
    3. Edge enhancement / slight sharpening
    """

    # ---------------- Contrast & Brightness ----------------
    alpha = 1.3  # contrast: 1.0 = original, >1 = stronger
    beta = 15    # brightness: 0 = original, >0 = brighter
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # ---------------- Gamma correction ----------------
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)

    # ---------------- Slight edge enhancement ----------------
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    return frame

# ---------------- Main Loop ----------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance frame to improve hand detection
        # frame_proc = enhance_frame(frame, gamma=1.3, alpha=1.3, beta=15)
        frame = cv2.resize(frame, (640, 480))
        frame = enhance_frame_for_hand_detection(frame)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection (image mode)
        hand_result = hand_model.detect(mp_image)
        pose_result = pose_model.detect(mp_image)

        # Draw
        visual_frame = draw_pose_and_hands(frame, pose_result, hand_result)

        # Show
        cv2.imshow("Hand Detection", visual_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()