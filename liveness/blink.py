"""
liveness/blink.py
Eye Aspect Ratio (EAR) based blink detection using MediaPipe Face Mesh.
Detects at least one genuine blink within a time window to confirm liveness.
"""
import time
import numpy as np
import mediapipe as mp

# MediaPipe Face Mesh landmark indices for left & right eye
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

_face_mesh = None


def _get_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _face_mesh


def _eye_aspect_ratio(landmarks, indices: list, w: int, h: int) -> float:
    """Compute EAR for given eye landmark indices."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0.0


class BlinkDetector:
    """
    Stateful blink detector — call update() per frame.
    Returns True from update() when a blink is confirmed.
    """

    def __init__(self, threshold: float = 0.25, consec_frames: int = 2):
        self.threshold = threshold
        self.consec    = consec_frames
        self._counter  = 0
        self.blink_count = 0

    def update(self, frame) -> bool:
        """
        Process one BGR frame.
        Returns True if a blink was detected in this frame sequence.
        """
        import cv2
        mesh = _get_mesh()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._counter = 0
            return False

        lm = results.multi_face_landmarks[0].landmark
        ear = (_eye_aspect_ratio(lm, _LEFT_EYE,  w, h) +
               _eye_aspect_ratio(lm, _RIGHT_EYE, w, h)) / 2.0

        if ear < self.threshold:
            self._counter += 1
        else:
            if self._counter >= self.consec:
                self.blink_count += 1
                self._counter = 0
                return True
            self._counter = 0
        return False

    def reset(self):
        self._counter = 0
        self.blink_count = 0


def wait_for_blink(stream, timeout: float = 5.0,
                   ear_threshold: float = 0.25) -> bool:
    """
    Block until a blink is detected from the webcam stream or timeout.

    Args:
        stream:        WebcamStream instance
        timeout:       seconds to wait
        ear_threshold: EAR value below which eye is considered closed

    Returns True if blink detected within timeout.
    """
    detector = BlinkDetector(threshold=ear_threshold)
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = stream.read()
        if frame is None:
            time.sleep(0.03)
            continue
        if detector.update(frame):
            return True
    return False
