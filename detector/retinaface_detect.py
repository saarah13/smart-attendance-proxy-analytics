"""
detector/retinaface_detect.py
Face detection using MediaPipe Face Detection (no insightface needed).
Returns bounding boxes + approximate 5-point landmarks for alignment.
"""
import cv2
import numpy as np
import mediapipe as mp

_face_detection = None


def _get_detector():
    global _face_detection
    if _face_detection is None:
        _face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,             # 1 = full-range model (up to 5m)
            min_detection_confidence=0.5,
        )
    return _face_detection


def detect_faces(frame: np.ndarray) -> list[dict]:
    """
    Detect all faces in a BGR frame using MediaPipe.

    Returns list of dicts:
    {
        'bbox':      [x1, y1, x2, y2]  (int),
        'landmarks': np.ndarray shape (5, 2)  — approx 5-pt for alignment,
        'confidence': float
    }
    """
    det = _get_detector()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = det.process(rgb)

    faces = []
    if not results.detections:
        return faces

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width)  * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        # MediaPipe Face Detection keypoints (6 total):
        # 0=right_eye, 1=left_eye, 2=nose_tip, 3=mouth_center,
        # 4=right_ear_tragion, 5=left_ear_tragion
        kps = detection.location_data.relative_keypoints
        # Map to ArcFace 5-pt order:
        # left_eye, right_eye, nose, mouth_left, mouth_right
        landmarks = np.array([
            [kps[1].x * w, kps[1].y * h],   # left eye
            [kps[0].x * w, kps[0].y * h],   # right eye
            [kps[2].x * w, kps[2].y * h],   # nose tip
            # approximate mouth corners from center ± offset
            [kps[3].x * w - (x2 - x1) * 0.12, kps[3].y * h],
            [kps[3].x * w + (x2 - x1) * 0.12, kps[3].y * h],
        ], dtype=np.float32)

        faces.append({
            "bbox":       [x1, y1, x2, y2],
            "landmarks":  landmarks,
            "confidence": float(detection.score[0]),
        })

    return faces


def draw_detections(frame: np.ndarray, faces: list[dict],
                    labels: list[str] = None) -> np.ndarray:
    """Draw bounding boxes + name labels onto a BGR frame copy."""
    out = frame.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]
        label = labels[i] if labels and i < len(labels) else ""
        color = (0, 220, 100) if (label and label != "Unknown") else (0, 80, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    return out
