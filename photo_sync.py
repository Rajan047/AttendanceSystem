"""
photo_sync.py — Jetson pe chalne wala poller.
Har 30s mein Supabase Storage se nayi employee photos check karta hai,
download karta hai, aur face embedding generate karke embeddings.pkl update karta hai.
"""

import os
import time
import pickle
import requests
import threading
import numpy as np
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL    = "https://bjgiirtpqfjaaxnftptf.supabase.co"
SUPABASE_KEY    = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
BUCKET          = "employee-photos"
POLL_INTERVAL   = 30
EMBEDDINGS_FILE = "/home/cyamsys/Documents/HR_attendance/embeddings/embeddings.pkl"
PHOTOS_DIR      = "/home/cyamsys/Documents/HR_attendance/photos"
PROCESSED_FILE  = "/home/cyamsys/Documents/HR_attendance/embeddings/synced_photos.txt"

HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

_EMB_LOCK  = threading.Lock()
_face_app  = None
_face_lock = threading.Lock()

# ── Supabase Storage — recursive file listing ─────────────────────────────────
def list_files_in_folder(prefix=""):
    """
    Supabase Storage mein ek folder ke andar saari files list karo.
    Agar koi entry folder hai (id=None) toh recursively uske andar bhi jao.
    Returns list of full storage paths (e.g. 'Rajan/profile_abc.jpg')
    """
    url  = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
    body = {"prefix": prefix, "limit": 1000, "offset": 0}
    resp = requests.post(url, headers=HEADERS, json=body, timeout=15)
    if resp.status_code != 200:
        print(f"[sync] list failed ({prefix}): {resp.status_code} {resp.text[:100]}")
        return []

    entries    = resp.json()
    file_paths = []

    for entry in entries:
        name = entry.get("name", "")
        eid  = entry.get("id")

        if not name:
            continue

        full_path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"

        if eid is None:
            # Yeh ek folder hai — recursively list karo
            sub_files = list_files_in_folder(full_path)
            file_paths.extend(sub_files)
        else:
            # Yeh ek actual file hai
            file_paths.append(full_path)

    return file_paths


def list_all_storage_files():
    """Bucket ki saari files recursively list karo."""
    return list_files_in_folder("")


# ── Download ──────────────────────────────────────────────────────────────────
def download_file(storage_path, local_path):
    url  = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"[sync] download failed: {storage_path} → {resp.status_code}")
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return True


# ── Processed tracker ─────────────────────────────────────────────────────────
def load_processed():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_processed(storage_path):
    with open(PROCESSED_FILE, "a") as f:
        f.write(storage_path + "\n")


# ── Face model ────────────────────────────────────────────────────────────────
def get_face_app():
    global _face_app
    if _face_app is None:
        with _face_lock:
            if _face_app is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    allowed_modules=["detection", "recognition"]
                )
                app.prepare(ctx_id=0, det_size=(320, 320))
                _face_app = app
                print("[sync] ✅ InsightFace model loaded")
    return _face_app


def generate_embedding(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = get_face_app().get(img)
    if not faces:
        return None
    faces.sort(
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    return faces[0].embedding.astype(np.float32)


def add_to_pickle(name, embedding):
    with _EMB_LOCK:
        data = {}
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                data = pickle.load(f)
        data.setdefault(name, [])
        data[name].append(np.asarray(embedding, dtype=np.float32))
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(data, f)
    # mtime touch — entry_cameras.py watcher detect karega
    os.utime(EMBEDDINGS_FILE, None)
    print(f"[sync] ✅ {name} ki embedding embeddings.pkl mein add ho gayi")


# ── Employee name from storage path ───────────────────────────────────────────
def extract_name(storage_path):
    """
    'Rajan/profile_abc.jpg' → 'Rajan'
    'Prashant_Kumar/face_1_xyz.jpg' → 'Prashant_Kumar'
    Sirf profile_ aur face_ prefix wali files embed karo (attendance captures skip)
    """
    parts = storage_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None, False

    folder   = parts[0]
    filename = parts[-1].lower()

    # Sirf profile photos embed karo — attendance date folders skip karo
    # e.g. Rajan/2026-08-07/Entry-1_xxx.jpg → skip
    # e.g. Rajan/profile_abc.jpg → embed
    is_profile = filename.startswith("profile_") or filename.startswith("face_")
    return folder, is_profile


# ── Main sync ─────────────────────────────────────────────────────────────────
def sync_once():
    processed = load_processed()
    all_files = list_all_storage_files()

    if not all_files:
        print("[sync] Koi file nahi mili storage mein")
        return

    new_count = 0
    for storage_path in all_files:
        if storage_path in processed:
            continue

        emp_name, is_profile = extract_name(storage_path)
        if not emp_name:
            mark_processed(storage_path)
            continue

        if not is_profile:
            # Attendance capture — download karo (dashboard pe dikhne ke liye)
            # but embedding mat banao
            local_path = os.path.join(PHOTOS_DIR, storage_path.replace("/", os.sep))
            if not os.path.exists(local_path):
                download_file(storage_path, local_path)
            mark_processed(storage_path)
            continue

        print(f"[sync] Nayi profile photo: {storage_path} → {emp_name}")

        # Download karo
        local_path = os.path.join(PHOTOS_DIR, storage_path.replace("/", os.sep))
        if not download_file(storage_path, local_path):
            continue

        # Embedding generate karo
        emb = generate_embedding(local_path)
        if emb is None:
            print(f"[sync] ⚠️  {storage_path}: face detect nahi hua — skip")
            mark_processed(storage_path)
            continue

        # embeddings.pkl update
        add_to_pickle(emp_name, emb)
        mark_processed(storage_path)
        new_count += 1

    if new_count > 0:
        print(f"[sync] {new_count} nayi embedding(s) add ho gayi ✅")


def main():
    print(f"[sync] Photo sync watcher started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[sync] Supabase bucket: {BUCKET}")
    print(f"[sync] Poll interval: {POLL_INTERVAL}s")
    print(f"[sync] Embeddings file: {EMBEDDINGS_FILE}")

    try:
        sync_once()
    except Exception as e:
        print(f"[sync] startup error: {e}")

    while True:
        time.sleep(POLL_INTERVAL)
        try:
            sync_once()
        except Exception as e:
            print(f"[sync] sync error: {e}")


if __name__ == "__main__":
    main()