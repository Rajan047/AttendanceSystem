"""
face_embedding.py — turn an uploaded photo into a 512-d embedding for the
admin "add employee" flow, in the SAME format your entry_cameras.py already
consumes.

Your entry_cameras.py loads embeddings.pkl as:
    { name: [emb1, emb2, ...] }   # each emb is a 512-float list/array
and averages them. So when the admin adds an employee, we:
    1. detect + embed the uploaded face (InsightFace buffalo_l — SAME model
       your cameras use, so the vector space matches exactly)
    2. append it under that employee's name in embeddings.pkl
    3. (optionally) also store it in Postgres face_embeddings for backup

Because we reuse buffalo_l, a face added here is immediately recognizable by
your live cameras after they reload embeddings.pkl (restart the camera
service, or add a hot-reload — see rebuild note at the bottom).
"""
import os
import pickle
import threading

import numpy as np

import config

_EMB_LOCK = threading.Lock()

# Lazy singleton so the model loads once, and ONLY if the admin actually adds
# a photo (the web server process may not otherwise need InsightFace).
_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        # NOTE: on the Render web server there's no GPU — fall back to CPU.
        # On the DGX/local server you can force CUDA. We try CUDA then CPU.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(320, 320))
        _face_app = app
    return _face_app


def generate_embedding(image_path):
    """
    Returns a 512-d numpy float32 vector for the largest face in the image,
    or None if no face is detected.
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = _get_face_app().get(img)
    if not faces:
        return None
    # pick the largest detected face (most likely the subject)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    emb = faces[0].embedding.astype(np.float32)
    return emb


def add_to_pickle(name, embedding, embeddings_file=None):
    """
    Append `embedding` under `name` in embeddings.pkl, matching the
    { name: [emb, emb, ...] } structure entry_cameras.py loads. Creates the
    file if missing. Thread-safe.
    """
    embeddings_file = embeddings_file or config.EMBEDDINGS_FILE
    with _EMB_LOCK:
        data = {}
        if os.path.exists(embeddings_file):
            with open(embeddings_file, "rb") as f:
                data = pickle.load(f)
        data.setdefault(name, [])
        data[name].append(np.asarray(embedding, dtype=np.float32))
        os.makedirs(os.path.dirname(embeddings_file) or ".", exist_ok=True)
        with open(embeddings_file, "wb") as f:
            pickle.dump(data, f)
    return len(data[name])


def remove_from_pickle(name, embeddings_file=None):
    """Drop an employee entirely from embeddings.pkl (used on hard-delete)."""
    embeddings_file = embeddings_file or config.EMBEDDINGS_FILE
    with _EMB_LOCK:
        if not os.path.exists(embeddings_file):
            return
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)
        if name in data:
            del data[name]
            with open(embeddings_file, "wb") as f:
                pickle.dump(data, f)


def embedding_to_bytes(embedding):
    """Serialize a vector for Postgres BYTEA storage (face_embeddings.embedding)."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# Making new employees live on the cameras WITHOUT a manual restart:
#
# entry_cameras.py loads embeddings.pkl once at startup into _known_matrix.
# Two options to pick up admin-added faces:
#   (a) simplest: `sudo systemctl restart entry-cameras.service` after adds
#       (a nightly restart cron also works if adds aren't urgent), or
#   (b) add a lightweight file-watcher in entry_cameras.py that re-reads
#       embeddings.pkl + rebuilds _known_matrix when the file's mtime changes.
# For (b), the rebuild is just the same 6 lines already at the top of
# entry_cameras.py wrapped in a function and called when mtime changes.
# ---------------------------------------------------------------------------
