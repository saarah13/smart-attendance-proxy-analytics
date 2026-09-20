"""
liveness/head_pose.py
Head pose estimation using MediaPipe Face Mesh.
Detects left/right head turn to confirm liveness alongside blink detection.
"""
import time
import numpy as np
import mediapipe as mp
import cv2

_face_mesh = None

# Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

_MP_INDICES = [1, 152, 263, 33, 287, 57]   # matching landmark IDs


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


def get_head_yaw(frame) -> float | None:
    """
    Returns yaw angle (horizontal rotation) in degrees, or None if no face.
    Negative = turned left, Positive = turned right.
    """
    mesh = _get_mesh()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    image_points = np.array(
        [(lm[i].x * w, lm[i].y * h) for i in _MP_INDICES], dtype=np.float64
    )

    focal = w
    cam_matrix = np.array([
        [focal,  0,     w / 2],
        [0,      focal, h / 2],
        [0,      0,     1    ],
    ], dtype=np.float64)
    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, cam_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x5(rmat) if hasattr(cv2, "RQDecomp3x5") else (None,)
    # Simpler: extract yaw from rotation matrix
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
    return float(yaw)


def wait_for_head_turn(stream, turn_deg: float = 15.0,
                        timeout: float = 6.0) -> bool:
    """
    Block until user turns head by at least `turn_deg` in either direction.
    Returns True if turn detected within timeout.
    """
    deadline = time.time() + timeout
    baseline = None

    while time.time() < deadline:
        frame = stream.read()
        if frame is None:
            time.sleep(0.03)
            continue

        yaw = get_head_yaw(frame)
        if yaw is None:
            time.sleep(0.03)
            continue

        if baseline is None:
            baseline = yaw
            continue

        if abs(yaw - baseline) >= turn_deg:
            return True

    return False
