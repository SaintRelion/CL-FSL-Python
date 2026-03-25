import pandas as pd

df = pd.read_csv("train.csv")

groups = {
    label: rows.reset_index(drop=True)
    for label, rows in df.groupby("label")
}

labels = list(groups.keys())
num_labels = len(labels)

print(f"Found {num_labels} labels")

max_len = max(len(g) for g in groups.values())

round_robin = []

for i in range(max_len):
    for label in labels:
        if i < len(groups[label]):
            round_robin.append(groups[label].iloc[i])


import cv2
import os

def mark_clip(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = None
    end_frame = None

    frame_idx = 0
    paused = False

    def get_frame(idx):
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        return ret, frame

    ret, frame = get_frame(frame_idx)

    while ret:
        display = frame.copy()
        cv2.putText(
            display,
            f"Frame: {frame_idx}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if start_frame is not None:
            cv2.putText(display, f"START: {start_frame}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if end_frame is not None:
            cv2.putText(display, f"END: {end_frame}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(
            "Controls: space=play/pause | a=back | d=forward | s=start | e=end | q=quit",
            display
        )

        key = cv2.waitKey(0 if paused else int(1000 / fps)) & 0xFF

        if key == ord(' '):  # pause / play
            paused = not paused

        elif key == ord('a'):  # reverse one frame
            frame_idx -= 10
            ret, frame = get_frame(frame_idx)
            paused = True

        elif key == ord('d'):  # forward one frame
            frame_idx += 10
            ret, frame = get_frame(frame_idx)
            paused = True

        elif key == ord('s'):
            start_frame = frame_idx
            print("Start marked:", start_frame)

        elif key == ord('e'):
            end_frame = frame_idx
            print("End marked:", end_frame)
            break

        elif key == ord('q'):
            break

        if not paused:
            frame_idx += 1
            ret, frame = get_frame(frame_idx)

    cap.release()
    cv2.destroyAllWindows()

    if start_frame is not None and end_frame is not None:
        return start_frame, end_frame

    return None


def export_clip(video_path, start, end, out_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start <= frame_idx <= end:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

from pathlib import Path

output_root = Path("clean_clips")

for idx, row in enumerate(round_robin):
    video_path = row["vid_path"]
    label = row["label"]

    print(f"\nProcessing [{idx+1}/{len(round_robin)}]: {video_path}")

    result = mark_clip(video_path)
    if result is None:
        print("Skipped.")
        continue

    start, end = result

    label_dir = output_root / label
    existing = len(list(label_dir.glob("*.mp4")))
    out_path = label_dir / f"video_{existing+1}.mp4"

    export_clip(video_path, start, end, str(out_path))

    print(f"Saved → {out_path}")
