# fsl_helper.py
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

# Aliases for Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarker = vision.HandLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarker = vision.PoseLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
VisionRunningMode = vision.RunningMode

TARGET_FRAMES = 30

# ------------------- Helpers -------------------
def resample_sequence(seq, target_len):
    """Uniformly resample list of frames to target length"""
    if len(seq) == target_len:
        return seq
    idxs = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return [seq[i] for i in idxs]

def forward_fill(current_frame, prev_frame):
    if prev_frame is None:
        return current_frame
    # Make sure we only iterate over the overlap
    min_len = min(len(current_frame), len(prev_frame))
    for i in range(min_len):
        if current_frame[i] == 0 and prev_frame[i] != 0:
            current_frame[i] = prev_frame[i]
    return current_frame

def shoulder_anchor(detection_result):
    """
    Returns anchor and scale from detection_result (Tasks API)
    """
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]  # take first detected pose

    # LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12
    ls = pose_landmarks[11]
    rs = pose_landmarks[12]

    anchor = np.array([(ls.x + rs.x)/2, (ls.y + rs.y)/2, (ls.z + rs.z)/2])
    scale = np.linalg.norm(np.array([ls.x, ls.y, ls.z]) - np.array([rs.x, rs.y, rs.z])) + 1e-6
    return anchor, scale
