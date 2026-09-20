"""
matcher/faiss_matcher.py
FAISS-based cosine similarity search against the student embedding database.
Rebuilt from MongoDB on startup or when a new student is registered.
"""
import numpy as np
import faiss
from pymongo import MongoClient
import config

_index:     faiss.Index | None  = None
_usn_list:  list[str]           = []   # parallel list mapping FAISS row → USN
_name_list: list[str]           = []   # parallel list mapping FAISS row → name


# ─── Index Management ─────────────────────────────────────────────────────────

def build_index():
    """
    Load all student embeddings from MongoDB and build an in-memory FAISS index.
    Must be called once at startup and after any new registration.
    """
    global _index, _usn_list, _name_list

    client = MongoClient(config.MONGO_URI)
    db     = client[config.DB_NAME]
    students = list(db.students.find(
        {"embedding": {"$exists": True}},
        {"usn": 1, "name": 1, "embedding": 1}
    ))
    client.close()

    if not students:
        _index     = None
        _usn_list  = []
        _name_list = []
        return

    dim = 512
    _index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on L2-normalised vecs

    embeddings = []
    usns       = []
    names      = []

    for s in students:
        emb = np.array(s["embedding"], dtype=np.float32)
        if emb.shape == (512,):
            embeddings.append(emb)
            usns.append(s["usn"])
            names.append(s["name"])

    if embeddings:
        matrix = np.stack(embeddings, axis=0)           # (N, 512)
        # Ensure L2-normalised
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-6, None)
        _index.add(matrix)
        _usn_list  = usns
        _name_list = names


def _ensure_index():
    if _index is None:
        build_index()


# ─── Search ───────────────────────────────────────────────────────────────────

def search(embedding: np.ndarray, top_k: int = 1,
           threshold: float = None) -> list[dict]:
    """
    Find closest students to a query embedding.

    Args:
        embedding:  (512,) float32 L2-normalised query vector
        top_k:      number of results to return
        threshold:  cosine similarity cutoff (default from config)

    Returns list of dicts:
        [{"usn": str, "name": str, "similarity": float}]
    Only includes matches above threshold.  Empty list = Unknown.
    """
    _ensure_index()
    if _index is None or _index.ntotal == 0:
        return []

    if threshold is None:
        threshold = config.RECOGNITION_THRESHOLD

    # Normalise query
    norm = np.linalg.norm(embedding)
    q = (embedding / norm).astype(np.float32) if norm > 0 else embedding
    q = q.reshape(1, -1)

    k = min(top_k, _index.ntotal)
    distances, indices = _index.search(q, k)   # distances = cosine similarities

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        similarity = float(dist)
        if similarity >= threshold:
            results.append({
                "usn":        _usn_list[idx],
                "name":       _name_list[idx],
                "similarity": round(similarity, 4),
            })
    return results


def total_registered() -> int:
    """Return number of enrolled students in the index."""
    _ensure_index()
    return _index.ntotal if _index else 0
