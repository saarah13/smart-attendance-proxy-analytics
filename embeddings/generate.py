"""
embeddings/generate.py
ArcFace face embedding using onnxruntime-gpu directly.
No insightface Python package needed — just the ONNX model file.

Model: w600k_r50.onnx  (ArcFace R50, from buffalo_l model pack)
       Downloaded by setup.bat to models/buffalo_l/w600k_r50.onnx
Input:  (1, 3, 112, 112)  float32  normalised to [-1, 1]
Output: (1, 512)          float32  face embedding
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
import config

_session:      ort.InferenceSession | None = None
_INPUT_NAME:   str = ""
_MODEL_PATHS = [
    os.path.join(config.MODELS_DIR, "buffalo_l", "w600k_r50.onnx"),
    os.path.join(config.MODELS_DIR, "w600k_r50.onnx"),
]


def _load_session() -> ort.InferenceSession:
    global _session, _INPUT_NAME
    if _session is not None:
        return _session

    model_path = None
    for p in _MODEL_PATHS:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(
            "ArcFace ONNX model not found.\n"
            f"Expected at: {_MODEL_PATHS[0]}\n"
            "Run setup.bat to download it."
        )

    providers = ["CPUExecutionProvider"]
    # Uncomment below to use GPU if onnxruntime-gpu is installed:
    # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session    = ort.InferenceSession(model_path, providers=providers)
    _INPUT_NAME = _session.get_inputs()[0].name
    print(f"[ArcFace] Loaded from {model_path}  providers={providers}")
    return _session


def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Resize to 112×112, normalise to [-1, 1], convert to NCHW."""
    face = cv2.resize(face_bgr, (112, 112))
    rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb  = (rgb - 127.5) / 128.0
    chw  = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)   # (1, 3, 112, 112)


def get_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Generate a 512-D L2-normalised ArcFace embedding from a face crop.

    Args:
        face_bgr: BGR face image (any size — resized internally to 112×112)

    Returns:
        np.ndarray shape (512,) float32, or None on error.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    try:
        sess = _load_session()
        inp  = _preprocess(face_bgr)
        out  = sess.run(None, {_INPUT_NAME: inp})[0]   # (1, 512)
        emb  = out[0].astype(np.float32)
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 1e-6 else emb
    except FileNotFoundError as e:
        print(f"[ArcFace] {e}")
        return None
    except Exception as e:
        print(f"[ArcFace] Inference error: {e}")
        return None


def get_embeddings_batch(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Generate embeddings for a list of full webcam frames (registration use).
    Internally detects + crops the face before embedding.
    Returns list of valid (512,) embeddings.
    """
    from detector.retinaface_detect import detect_faces

    embeddings = []
    for frame in frames:
        faces = detect_faces(frame)
        if not faces:
            continue
        # Use the highest-confidence face
        best  = max(faces, key=lambda f: f["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emb = get_embedding(crop)
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average multiple embeddings and re-normalise (for registration)."""
    if not embeddings:
        raise ValueError("No valid embeddings to average")
    stacked = np.stack(embeddings, axis=0)
    mean    = stacked.mean(axis=0)
    norm    = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 1e-6 else mean


def save_embedding(usn: str, embedding: np.ndarray):
    """Save embedding as .npy backup file."""
    path = os.path.join(config.DATASET_DIR, f"{usn}.npy")
    np.save(path, embedding)


def load_embedding(usn: str) -> np.ndarray | None:
    """Load .npy embedding backup. Returns None if not found."""
    path = os.path.join(config.DATASET_DIR, f"{usn}.npy")
    return np.load(path).astype(np.float32) if os.path.exists(path) else None
