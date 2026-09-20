"""
liveness/anti_spoof.py
MiniFASNet-based anti-spoofing using ONNX Runtime.
Classifies a face crop as real (live) or fake (photo / screen replay).

Model: MiniFASNetV2  (downloaded separately — see setup.bat)
Input:  80×80 BGR image, normalised
Output: 2-class softmax [fake_prob, real_prob]
"""
import os
import cv2
import numpy as np
import config

_session = None
_INPUT_SIZE = (80, 80)


def _load_session():
    global _session
    if _session is None:
        model_path = config.ANTI_SPOOF_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Anti-spoof model not found: {model_path}\n"
                "Run setup.bat to download it."
            )
        import onnxruntime as ort
        _session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
    return _session


def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Resize to 80×80 and normalise to float32 in [-1, 1]."""
    resized = cv2.resize(face_bgr, _INPUT_SIZE)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    # HWC → NCHW
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def is_real_face(face_bgr: np.ndarray,
                 threshold: float = None) -> tuple[bool, float]:
    """
    Classify a face crop as real or spoof.

    Args:
        face_bgr:  BGR face crop (any size — resized internally)
        threshold: minimum real_prob to accept as real (default from config)

    Returns:
        (is_real: bool, real_score: float)
    """
    if threshold is None:
        threshold = config.ANTI_SPOOF_THRESHOLD

    try:
        sess = _load_session()
    except FileNotFoundError:
        # If model not present, skip anti-spoof (graceful degradation)
        return True, 1.0

    inp = _preprocess(face_bgr)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: inp})
    probs = outputs[0][0]          # shape (2,)

    # Softmax if not already
    exp = np.exp(probs - probs.max())
    probs = exp / exp.sum()

    real_score = float(probs[1])   # index 1 = real class
    return real_score >= threshold, real_score
