import cv2
import numpy as np
import random
import os
from build_dataset import create_models, extract_frame_features, ROOT_DIR


# ------------------- Augmentation Profiles -------------------
def get_random_augmentation_profile():
    """Generates a static profile to be used for the entire video duration."""
    return {
        "zoom_factor": random.uniform(0.8, 1.2),
        "brightness": random.uniform(0.8, 1.2),
        "contrast": random.randint(-20, 20),
        "apply_noise": random.random() > 0.5,
        "crop_offset_x": random.uniform(-0.1, 0.1),  # % offset
        "crop_offset_y": random.uniform(-0.1, 0.1),
    }


def apply_profile_aug(frame, profile):
    """Applies the pre-selected profile to a single frame."""
    h, w = frame.shape[:2]
    zf = profile["zoom_factor"]

    # 1. Zoom and Crop Logic
    if zf > 1.0:
        new_h, new_w = int(h / zf), int(w / zf)
        # Apply the static offset from the profile
        center_y = int((h - new_h) * (0.5 + profile["crop_offset_y"]))
        center_x = int((w - new_w) * (0.5 + profile["crop_offset_x"]))
        start_y = np.clip(center_y, 0, h - new_h)
        start_x = np.clip(center_x, 0, w - new_w)
        frame = frame[start_y : start_y + new_h, start_x : start_x + new_w]
        frame = cv2.resize(frame, (w, h))
    else:
        new_h, new_w = int(h * zf), int(w * zf)
        rescaled = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        y_off, x_off = (h - new_h) // 2, (w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = rescaled
        frame = canvas

    # 2. Lighting and Noise
    frame = cv2.convertScaleAbs(
        frame, alpha=profile["brightness"], beta=profile["contrast"]
    )
    # if profile["apply_noise"]:
    #     noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
    #     frame = cv2.add(frame, noise)

    return frame


# ------------------- Main Test Loop -------------------
def run_stable_augmentation_test():
    hand_model, pose_model, face_model = create_models()  #
    models = (hand_model, pose_model, face_model)

    common_labels = [
        "BOY",
        "COLD",
        "CORRECT",
        "DONT_KNOW",
        "DONT_UNDERSTAND",
        "FAST",
        "GIRL",
        "GOOD_AFTERNOON",
        "GOOD_EVENING",
        "GOOD_MORNING",
        "HELLO",
        "HOT",
        "HOW_ARE_YOU",
        "IM_FINE",
        "KNOW",
        "MAN",
        "NICE_TO_MEET_YOU",
        "NO",
        "PARENTS",
        "SEE_YOU_TOMORROW",
        "SLOW",
        "THANK_YOU",
        "TODAY",
        "TOMORROW",
        "UNDERSTAND",
        "WOMAN",
        "WRONG",
        "YES",
        "YESTERDAY",
        "YOURE_WELCOME",
    ]
    use_aug = True

    for label in common_labels:
        label_path = os.path.join(ROOT_DIR, label)
        if not os.path.isdir(label_path):
            continue
        videos = sorted([f for f in os.listdir(label_path) if f.endswith(".mp4")])

        for vid in videos:
            cap = cv2.VideoCapture(os.path.join(label_path, vid))
            # PICK ONLY ONCE: Select augmentation profile for this specific video
            current_profile = get_random_augmentation_profile()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Go to next video

                # Apply the static profile
                test_frame = (
                    apply_profile_aug(frame.copy(), current_profile)
                    if use_aug
                    else frame.copy()
                )

                # Inference and Heatmap Visualization
                feat, debug_frame = extract_frame_features(test_frame, *models)

                if debug_frame is not None:
                    cv2.putText(
                        debug_frame,
                        f"LABEL: {label} | VIDEO: {vid}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.imshow("FSL LINK - Stable Augmentation Test", debug_frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    return
                if key == ord("a"):
                    use_aug = not use_aug
                if key == ord("n"):
                    break  # Skip to next video

            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_stable_augmentation_test()
