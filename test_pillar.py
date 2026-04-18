import os
import cv2
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from handsign.pillar_collector import create_models, extract_frame_features
from fsl_helper import resample_sequence

# ------------------- Configuration -------------------
ROOT_DIR = "common_clips"
MODEL_PATH = "models/gesture_model.keras"
LABEL_MAP_PATH = "data/label_map.json"
TARGET_FRAMES = 30
# NEW FEATURE COUNT: 10 (Extensions) + 50 (Web Distances) = 60
FEATURE_COUNT = 60


def apply_fixed_zoom(frame, zoom_factor):
    """Safe zoom with padding logic for zoom-out and cropping for zoom-in."""
    h, w = frame.shape[:2]
    if zoom_factor >= 1.0:
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        y1, x1 = (h - new_h) // 2, (w - new_w) // 2
        crop = frame[y1 : y1 + new_h, x1 : x1 + new_w]
        return cv2.resize(crop, (w, h))
    else:
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros_like(frame)
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        canvas[pad_h : pad_h + resized.shape[0], pad_w : pad_w + resized.shape[1]] = (
            resized
        )
        return canvas


def plot_diagnostics(
    snapshot, feats_30, prediction, inv_label_map, label_name, pred_label
):
    """
    Displays an updated dual-pane diagnostic:
    Left: Actual spatial path based on the 'Web' distances.
    Right: Probability bar chart for top predictions.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- PANE 1: NEURAL WEB SPATIAL TRUTH ---
    ax1.imshow(cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB))
    feats_arr = np.array(feats_30)  # Convert list to array for slicing

    # We plot the 'Neural Web' activity (Features 10-60)
    # Every 10 features represent one finger's relationship to the 10 pillars
    for t in range(0, TARGET_FRAMES, 3):
        prog = t / (TARGET_FRAMES - 1)
        dot_color = (1.0, 1.0 - prog, 1.0)  # Yellow to Magenta

        # Check which finger was 'closest' to any pillar at this time
        # Indices 10 to 60 contain the 50 web distances
        web_slice = feats_arr[t, 10:60]
        if np.min(web_slice) < 0.2:  # Only plot if there was a close interaction
            ax1.scatter([], [], color=dot_color, s=50, edgecolors="white")

    ax1.set_title(f"Neural Web Path: {label_name}")
    ax1.axis("off")

    # --- PANE 2: PROBABILITY BAR CHART ---
    top_indices = np.argsort(prediction)[-5:]
    top_scores = prediction[top_indices]
    top_labels = [inv_label_map.get(str(i), "Unknown") for i in top_indices]

    colors = ["gray"] * 4 + ["green" if pred_label == label_name else "red"]
    ax2.barh(top_labels, top_scores, color=colors)
    ax2.set_xlim(0, 1.1)
    ax2.set_title("Softmax Probabilities")
    for i, v in enumerate(top_scores):
        ax2.text(v + 0.02, i, f"{v*100:.1f}%", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(4)
    plt.close()


def run_diagnostic_test():
    print(f"🔄 Loading model with {FEATURE_COUNT} inputs...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_MAP_PATH, "r") as f:
        # Standardize loading regardless of int/str keys in JSON
        raw_map = json.load(f)
        inv_map = {str(v): k for k, v in raw_map.items()}

    models = create_models()  #

    for label_name in sorted(os.listdir(ROOT_DIR)):
        label_path = os.path.join(ROOT_DIR, label_name)
        if not os.path.isdir(label_path):
            continue

        skip_label = False
        for video_file in [f for f in os.listdir(label_path) if f.endswith(".mp4")]:
            if skip_label:
                break

            # Random Fixed Zoom test (0.8x to 1.5x)
            zoom = np.random.uniform(0.8, 1.5)
            print(f"🔬 Testing: {label_name}/{video_file} (Zoom: {zoom:.2f}x)")

            cap = cv2.VideoCapture(os.path.join(label_path, video_file))
            raw_features, snapshot = [], None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                test_frame = apply_fixed_zoom(frame, zoom)
                # This uses your updated 60-feature extractor
                feat, debug = extract_frame_features(test_frame, *models)

                # Hand detected if any of the 10 finger extensions > 0
                if any(v > 0 for v in feat[:10]):
                    raw_features.append(feat)
                    if snapshot is None:
                        snapshot = debug.copy()

                cv2.imshow("FSL Diagnostic Window", debug)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    skip_label = True
                    break
                if key == ord("q"):
                    return

            cap.release()
            if skip_label or len(raw_features) < 10:
                continue

            # Resample and Inference
            feats_30 = resample_sequence(raw_features, TARGET_FRAMES)
            input_data = np.expand_dims(np.array(feats_30), axis=0)

            prediction = model.predict(input_data, verbose=0)[0]
            pred_idx = np.argmax(prediction)
            pred_label = inv_map.get(str(pred_idx), "Unknown")

            # Dual-Pane Spatial + Probability Logic
            plot_diagnostics(
                snapshot, feats_30, prediction, inv_map, label_name, pred_label
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_diagnostic_test()
