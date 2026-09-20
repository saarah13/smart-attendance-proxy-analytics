"""
aligner/align.py
Aligns a detected face to a canonical 112×112 crop
using the 5 facial landmark affine transform (ArcFace standard).
"""
import cv2
import numpy as np

# ArcFace canonical 5-landmark positions for 112×112 output
_ARCFACE_DEST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame: np.ndarray, landmarks: np.ndarray,
               out_size: int = 112) -> np.ndarray | None:
    """
    Warp a face region to 112×112 using 5-landmark similarity transform.

    Args:
        frame:     BGR image (H×W×3)
        landmarks: (5, 2) float32 array from detector
        out_size:  output square size (default 112 for ArcFace)

    Returns:
        Aligned BGR face crop (112×112×3), or None if transform fails.
    """
    if landmarks is None or len(landmarks) < 5:
        return None

    src = landmarks.astype(np.float32)
    dst = _ARCFACE_DEST * (out_size / 112.0)

    transform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    if transform is None:
        return None

    aligned = cv2.warpAffine(frame, transform, (out_size, out_size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return aligned


def preprocess_for_arcface(aligned_face: np.ndarray) -> np.ndarray:
    """
    Convert 112×112 BGR crop to float32 NCHW tensor normalised to [-1, 1].
    Returns shape (1, 3, 112, 112).
    """
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0              # ArcFace standard normalisation
    chw = np.transpose(rgb, (2, 0, 1))       # HWC → CHW
    return np.expand_dims(chw, axis=0)       # → NCHW (1, 3, 112, 112)
