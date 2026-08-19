# """
# photo_sync.py — Jetson pe chalne wala poller.
# Har 30s mein Supabase Storage se nayi employee photos check karta hai,
# download karta hai, aur face embedding generate karke embeddings.pkl update karta hai.
# """

# import os
# import time
# import pickle
# import requests
# import threading
# import numpy as np
# from datetime import datetime

# # ── Config ────────────────────────────────────────────────────────────────────
# SUPABASE_URL    = "https://bjgiirtpqfjaaxnftptf.supabase.co"
# SUPABASE_KEY    = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
# BUCKET          = "employee-photos"
# POLL_INTERVAL   = 30
# EMBEDDINGS_FILE = "/home/cyamsys/Documents/HR_attendance/embeddings/embeddings.pkl"
# PHOTOS_DIR      = "/home/cyamsys/Documents/HR_attendance/photos"
# PROCESSED_FILE  = "/home/cyamsys/Documents/HR_attendance/embeddings/synced_photos.txt"

# HEADERS = {
#     "apikey":        SUPABASE_KEY,
#     "Authorization": f"Bearer {SUPABASE_KEY}",
# }

# _EMB_LOCK  = threading.Lock()
# _face_app  = None
# _face_lock = threading.Lock()

# # ── Supabase Storage — recursive file listing ─────────────────────────────────
# def list_files_in_folder(prefix=""):
#     """
#     Supabase Storage mein ek folder ke andar saari files list karo.
#     Agar koi entry folder hai (id=None) toh recursively uske andar bhi jao.
#     Returns list of full storage paths (e.g. 'Rajan/profile_abc.jpg')
#     """
#     url  = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
#     body = {"prefix": prefix, "limit": 1000, "offset": 0}
#     resp = requests.post(url, headers=HEADERS, json=body, timeout=15)
#     if resp.status_code != 200:
#         print(f"[sync] list failed ({prefix}): {resp.status_code} {resp.text[:100]}")
#         return []

#     entries    = resp.json()
#     file_paths = []

#     for entry in entries:
#         name = entry.get("name", "")
#         eid  = entry.get("id")

#         if not name:
#             continue

#         full_path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"

#         if eid is None:
#             # Yeh ek folder hai — recursively list karo
#             sub_files = list_files_in_folder(full_path)
#             file_paths.extend(sub_files)
#         else:
#             # Yeh ek actual file hai
#             file_paths.append(full_path)

#     return file_paths


# def list_all_storage_files():
#     """Bucket ki saari files recursively list karo."""
#     return list_files_in_folder("")


# # ── Download ──────────────────────────────────────────────────────────────────
# def download_file(storage_path, local_path):
#     url  = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
#     resp = requests.get(url, timeout=30)
#     if resp.status_code != 200:
#         print(f"[sync] download failed: {storage_path} → {resp.status_code}")
#         return False
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)
#     with open(local_path, "wb") as f:
#         f.write(resp.content)
#     return True


# # ── Processed tracker ─────────────────────────────────────────────────────────
# def load_processed():
#     if not os.path.exists(PROCESSED_FILE):
#         return set()
#     with open(PROCESSED_FILE, "r") as f:
#         return set(line.strip() for line in f if line.strip())


# def mark_processed(storage_path):
#     with open(PROCESSED_FILE, "a") as f:
#         f.write(storage_path + "\n")


# # ── Face model ────────────────────────────────────────────────────────────────
# def get_face_app():
#     global _face_app
#     if _face_app is None:
#         with _face_lock:
#             if _face_app is None:
#                 from insightface.app import FaceAnalysis
#                 app = FaceAnalysis(
#                     name="buffalo_l",
#                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
#                     allowed_modules=["detection", "recognition"]
#                 )
#                 app.prepare(ctx_id=0, det_size=(320, 320))
#                 _face_app = app
#                 print("[sync] ✅ InsightFace model loaded")
#     return _face_app


# def generate_embedding(image_path):
#     import cv2
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
#     faces = get_face_app().get(img)
#     if not faces:
#         return None
#     faces.sort(
#         key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
#         reverse=True
#     )
#     return faces[0].embedding.astype(np.float32)


# def add_to_pickle(name, embedding):
#     with _EMB_LOCK:
#         data = {}
#         if os.path.exists(EMBEDDINGS_FILE):
#             with open(EMBEDDINGS_FILE, "rb") as f:
#                 data = pickle.load(f)
#         data.setdefault(name, [])
#         data[name].append(np.asarray(embedding, dtype=np.float32))
#         os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
#         with open(EMBEDDINGS_FILE, "wb") as f:
#             pickle.dump(data, f)
#     # mtime touch — entry_cameras.py watcher detect karega
#     os.utime(EMBEDDINGS_FILE, None)
#     print(f"[sync] ✅ {name} ki embedding embeddings.pkl mein add ho gayi")


# # ── Employee name from storage path ───────────────────────────────────────────
# def extract_name(storage_path):
#     """
#     'Rajan/profile_abc.jpg' → 'Rajan'
#     'Prashant_Kumar/face_1_xyz.jpg' → 'Prashant_Kumar'
#     Sirf profile_ aur face_ prefix wali files embed karo (attendance captures skip)
#     """
#     parts = storage_path.replace("\\", "/").split("/")
#     if len(parts) < 2:
#         return None, False

#     folder   = parts[0]
#     filename = parts[-1].lower()

#     # Sirf profile photos embed karo — attendance date folders skip karo
#     # e.g. Rajan/2026-08-07/Entry-1_xxx.jpg → skip
#     # e.g. Rajan/profile_abc.jpg → embed
#     is_profile = filename.startswith("profile_") or filename.startswith("face_")
#     return folder, is_profile


# # ── Main sync ─────────────────────────────────────────────────────────────────
# def sync_once():
#     processed = load_processed()
#     all_files = list_all_storage_files()

#     if not all_files:
#         print("[sync] Koi file nahi mili storage mein")
#         return

#     new_count = 0
#     for storage_path in all_files:
#         if storage_path in processed:
#             continue

#         emp_name, is_profile = extract_name(storage_path)
#         if not emp_name:
#             mark_processed(storage_path)
#             continue

#         if not is_profile:
#             # Attendance capture — download karo (dashboard pe dikhne ke liye)
#             # but embedding mat banao
#             local_path = os.path.join(PHOTOS_DIR, storage_path.replace("/", os.sep))
#             if not os.path.exists(local_path):
#                 download_file(storage_path, local_path)
#             mark_processed(storage_path)
#             continue

#         print(f"[sync] Nayi profile photo: {storage_path} → {emp_name}")

#         # Download karo
#         local_path = os.path.join(PHOTOS_DIR, storage_path.replace("/", os.sep))
#         if not download_file(storage_path, local_path):
#             continue

#         # Embedding generate karo
#         emb = generate_embedding(local_path)
#         if emb is None:
#             print(f"[sync] ⚠️  {storage_path}: face detect nahi hua — skip")
#             mark_processed(storage_path)
#             continue

#         # embeddings.pkl update
#         add_to_pickle(emp_name, emb)
#         mark_processed(storage_path)
#         new_count += 1

#     if new_count > 0:
#         print(f"[sync] {new_count} nayi embedding(s) add ho gayi ✅")


# def main():
#     print(f"[sync] Photo sync watcher started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"[sync] Supabase bucket: {BUCKET}")
#     print(f"[sync] Poll interval: {POLL_INTERVAL}s")
#     print(f"[sync] Embeddings file: {EMBEDDINGS_FILE}")

#     try:
#         sync_once()
#     except Exception as e:
#         print(f"[sync] startup error: {e}")

#     while True:
#         time.sleep(POLL_INTERVAL)
#         try:
#             sync_once()
#         except Exception as e:
#             print(f"[sync] sync error: {e}")


# if __name__ == "__main__":
#     main()

# """
# photo_sync.py — Jetson pe chalne wala poller.
# Har 30s mein Supabase Storage se nayi employee photos check karta hai,
# download karta hai, aur face embedding generate karke embeddings.pkl update karta hai.
# """

# import os
# import time
# import pickle
# import requests
# import threading
# import numpy as np
# from datetime import datetime

# # ── Config ────────────────────────────────────────────────────────────────────
# SUPABASE_URL    = "https://bjgiirtpqfjaaxnftptf.supabase.co"
# SUPABASE_KEY    = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
# BUCKET          = "employee-photos"
# EMPLOYEE_TABLE  = "employees"
# POLL_INTERVAL   = 30
# EMBEDDINGS_FILE = "/home/cyamsys/Documents/HR_attendance/embeddings/embeddings.pkl"
# PHOTOS_DIR      = "/home/cyamsys/Documents/HR_attendance/photos"
# PROCESSED_FILE  = "/home/cyamsys/Documents/HR_attendance/embeddings/synced_photos.txt"

# HEADERS = {
#     "apikey":        SUPABASE_KEY,
#     "Authorization": f"Bearer {SUPABASE_KEY}",
# }

# _EMB_LOCK  = threading.Lock()
# _face_app  = None
# _face_lock = threading.Lock()


# # ── Supabase employee names fetch ─────────────────────────────────────────────
# def get_supabase_employee_names():
#     """
#     Supabase employees table se saare current employee names fetch karta hai.
#     Returns set of names, ya None agar request fail hui.
#     """
#     url    = f"{SUPABASE_URL}/rest/v1/{EMPLOYEE_TABLE}"
#     params = {"select": "name"}
   

#     try:
#         resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
#     except Exception as e:
#         print(f"[sync] ❌ Supabase employee fetch failed: {e}")
#         return None

#     if resp.status_code != 200:
#         print(f"[sync] ❌ Employee fetch failed: {resp.status_code}")
#         return None

#     try:
#         employees = resp.json()
#     except Exception as e:
#         print(f"[sync] ❌ Invalid employee response: {e}")
#         return None
#     print(f"[sync] DEBUG employees response: {employees}")
#     return {
        
#         emp["name"]
#         for emp in employees
#         if emp.get("name")
#     }


# # ── Reconcile — deleted employees ki embeddings remove karo ───────────────────
# def reconcile_embeddings():
#     """
#     Supabase employees table ko source of truth maan kar
#     local embeddings.pkl se deleted employees ki entries remove karta hai.

#     Agar Supabase request fail hui toh local embeddings KABHI delete nahi hogi.
#     """
#     print("[sync] 🔍 Checking deleted employees...")

#     supabase_names = get_supabase_employee_names()

#     # Supabase unreachable — safe side pe raho, kuch delete mat karo
#     if supabase_names is None:
#         print("[sync] ⚠️ Supabase employee list available nahi. Local embeddings untouched.")
#         return

#     with _EMB_LOCK:

#         if not os.path.exists(EMBEDDINGS_FILE):
#             print("[sync] No local embeddings found.")
#             return

#         try:
#             with open(EMBEDDINGS_FILE, "rb") as f:
#                 data = pickle.load(f)
#         except Exception as e:
#             print(f"[sync] ❌ embeddings.pkl load failed: {e}")
#             return

#         if not isinstance(data, dict):
#             print("[sync] ⚠️ embeddings.pkl dictionary nahi hai — skip.")
#             return

#         local_names   = set(data.keys())
#         deleted_names = local_names - supabase_names

#         if not deleted_names:
#             print("[sync] ✅ No deleted employees found.")
#             return

#         for name in deleted_names:
#             print(f"[sync] 🗑️ Removing embedding: {name}")
#             del data[name]

#         # Atomic save
#         temp_file = EMBEDDINGS_FILE + ".tmp"
#         try:
#             with open(temp_file, "wb") as f:
#                 pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#             os.replace(temp_file, EMBEDDINGS_FILE)
#             os.utime(EMBEDDINGS_FILE, None)
#             print(f"[sync] ✅ {len(deleted_names)} deleted employee embedding(s) removed.")
#         except Exception as e:
#             print(f"[sync] ❌ Failed to save after reconcile: {e}")
#             if os.path.exists(temp_file):
#                 try:
#                     os.remove(temp_file)
#                 except Exception:
#                     pass


# # ── Supabase Storage — recursive file listing ─────────────────────────────────
# def list_files_in_folder(prefix=""):
#     """
#     Supabase Storage mein ek folder ke andar saari files list karo.
#     Agar koi entry folder hai (id=None) toh recursively uske andar bhi jao.
#     Returns list of full storage paths (e.g. 'Rajan/profile_abc.jpg')
#     """
#     url  = f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}"
#     body = {"prefix": prefix, "limit": 1000, "offset": 0}
#     resp = requests.post(url, headers=HEADERS, json=body, timeout=15)
#     if resp.status_code != 200:
#         print(f"[sync] list failed ({prefix}): {resp.status_code} {resp.text[:100]}")
#         return []

#     entries    = resp.json()
#     file_paths = []

#     for entry in entries:
#         name = entry.get("name", "")
#         eid  = entry.get("id")

#         if not name:
#             continue

#         full_path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"

#         if eid is None:
#             # Yeh ek folder hai — recursively list karo
#             sub_files = list_files_in_folder(full_path)
#             file_paths.extend(sub_files)
#         else:
#             # Yeh ek actual file hai
#             file_paths.append(full_path)

#     return file_paths


# def list_all_storage_files():
#     """Bucket ki saari files recursively list karo."""
#     return list_files_in_folder("")


# # ── Download ──────────────────────────────────────────────────────────────────
# def download_file(storage_path, local_path):
#     url  = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
#     resp = requests.get(url, timeout=30)
#     if resp.status_code != 200:
#         print(f"[sync] download failed: {storage_path} → {resp.status_code}")
#         return False
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)
#     with open(local_path, "wb") as f:
#         f.write(resp.content)
#     return True


# # ── Processed tracker ─────────────────────────────────────────────────────────
# def load_processed():
#     if not os.path.exists(PROCESSED_FILE):
#         return set()
#     with open(PROCESSED_FILE, "r") as f:
#         return set(line.strip() for line in f if line.strip())


# def mark_processed(storage_path):
#     os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
#     with open(PROCESSED_FILE, "a") as f:
#         f.write(storage_path + "\n")


# # ── Face model ────────────────────────────────────────────────────────────────
# def get_face_app():
#     global _face_app
#     if _face_app is None:
#         with _face_lock:
#             if _face_app is None:
#                 from insightface.app import FaceAnalysis
#                 app = FaceAnalysis(
#                     name="buffalo_l",
#                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
#                     allowed_modules=["detection", "recognition"]
#                 )
#                 app.prepare(ctx_id=0, det_size=(320, 320))
#                 _face_app = app
#                 print("[sync] ✅ InsightFace model loaded")
#     return _face_app


# def generate_embedding(image_path):
#     import cv2
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
#     faces = get_face_app().get(img)
#     if not faces:
#         return None
#     faces.sort(
#         key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
#         reverse=True
#     )
#     return faces[0].embedding.astype(np.float32)


# def add_to_pickle(name, embedding):
#     with _EMB_LOCK:
#         data = {}
#         if os.path.exists(EMBEDDINGS_FILE):
#             with open(EMBEDDINGS_FILE, "rb") as f:
#                 data = pickle.load(f)
#         data.setdefault(name, [])
#         data[name].append(np.asarray(embedding, dtype=np.float32))
#         os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
#         with open(EMBEDDINGS_FILE, "wb") as f:
#             pickle.dump(data, f)
#     # mtime touch — entry_cameras.py watcher detect karega
#     os.utime(EMBEDDINGS_FILE, None)
#     print(f"[sync] ✅ {name} ki embedding embeddings.pkl mein add ho gayi")


# # ── Employee name from storage path ───────────────────────────────────────────
# def extract_name(storage_path):
#     """
#     'Rajan/profile_abc.jpg' → 'Rajan'
#     'Prashant_Kumar/face_1_xyz.jpg' → 'Prashant_Kumar'
#     Sirf profile_ aur face_ prefix wali files embed karo (attendance captures skip)
#     """
#     parts = storage_path.replace("\\", "/").split("/")
#     if len(parts) < 2:
#         return None, False

#     folder   = parts[0]
#     filename = parts[-1].lower()

#     # Sirf profile photos embed karo — attendance date folders skip karo
#     # e.g. Rajan/2026-08-07/Entry-1_xxx.jpg → skip
#     # e.g. Rajan/profile_abc.jpg → embed
#     is_profile = filename.startswith("profile_") or filename.startswith("face_")
#     return folder, is_profile


# # ── Main sync ─────────────────────────────────────────────────────────────────
# def sync_once():

#     # ── Step 1: Deleted employees ki embeddings clean karo ───────────────────
#     reconcile_embeddings()

#     # ── Step 2: Storage se nayi photos process karo ───────────────────────────
#     processed = load_processed()
#     all_files = list_all_storage_files()

#     if not all_files:
#         print("[sync] Koi file nahi mili storage mein")
#         return

#     new_count = 0
#     for storage_path in all_files:
#         if storage_path in processed:
#             continue

#         emp_name, is_profile = extract_name(storage_path)
#         if not emp_name:
#             mark_processed(storage_path)
#             continue

#         if not is_profile:
#             # Attendance capture — download karo (dashboard pe dikhne ke liye)
#             # but embedding mat banao
#             local_path = os.path.join(PHOTOS_DIR, storage_path.replace("/", os.sep))
#             if not os.path.exists(local_path):
#                 download_file(storage_path, local_path)
#             mark_processed(storage_path)
#             continue

#         print(f"[sync] Nayi profile photo: {storage_path} → {emp_name}")

#         # Download karo
#         local_path = os.path.join(PHOTOS_DIR, storage_path.replace("/", os.sep))
#         if not download_file(storage_path, local_path):
#             continue

#         # Embedding generate karo
#         emb = generate_embedding(local_path)
#         if emb is None:
#             print(f"[sync] ⚠️  {storage_path}: face detect nahi hua — skip")
#             mark_processed(storage_path)
#             continue

#         # embeddings.pkl update
#         add_to_pickle(emp_name, emb)
#         mark_processed(storage_path)
#         new_count += 1

#     if new_count > 0:
#         print(f"[sync] {new_count} nayi embedding(s) add ho gayi ✅")


# def main():
#     print(f"[sync] Photo sync watcher started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"[sync] Supabase bucket: {BUCKET}")
#     print(f"[sync] Poll interval: {POLL_INTERVAL}s")
#     print(f"[sync] Embeddings file: {EMBEDDINGS_FILE}")

#     try:
#         sync_once()
#     except Exception as e:
#         print(f"[sync] startup error: {e}")

#     while True:
#         time.sleep(POLL_INTERVAL)
#         try:
#             sync_once()
#         except Exception as e:
#             print(f"[sync] sync error: {e}")


# if __name__ == "__main__":
#     main()

#"""
# photo_sync.py — Jetson pe chalne wala poller.

# Every 30s:
# 1. Supabase employees table se active/current employee names check karta hai.
# 2. Deleted employees ki local embeddings remove karta hai.
# 3. Supabase Storage se new photos check karta hai.
# 4. ONLY profile_*.jpg photos ke embeddings generate karta hai.
# 5. Attendance/face_* photos ko embedding nahi banata.
# 6. embeddings.pkl update karta hai.
# """

# import os
# import time
# import pickle
# import requests
# import threading
# import numpy as np
# from datetime import datetime


# # =============================================================================
# # CONFIG
# # =============================================================================

# SUPABASE_URL = "https://bjgiirtpqfjaaxnftptf.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
# BUCKET = "employee-photos"
# EMPLOYEE_TABLE = "employees"

# POLL_INTERVAL = 30

# EMBEDDINGS_FILE = (
#     "/home/cyamsys/Documents/HR_attendance/"
#     "embeddings/embeddings.pkl"
# )

# PHOTOS_DIR = (
#     "/home/cyamsys/Documents/HR_attendance/photos"
# )

# PROCESSED_FILE = (
#     "/home/cyamsys/Documents/HR_attendance/"
#     "embeddings/synced_photos.txt"
# )


# HEADERS = {
#     "apikey": SUPABASE_KEY,
#     "Authorization": f"Bearer {SUPABASE_KEY}",
# }


# # =============================================================================
# # LOCKS / FACE MODEL
# # =============================================================================

# _EMB_LOCK = threading.Lock()

# _face_app = None
# _face_lock = threading.Lock()


# # =============================================================================
# # SUPABASE EMPLOYEE NAMES
# # =============================================================================

# def get_supabase_employee_names():
#     """
#     Supabase employees table se current employee names fetch karta hai.

#     IMPORTANT:
#     .strip() use kiya gaya hai taaki:

#         "Rajan"
#         "Rajan "
#         " Rajan"

#     ko same employee maana jaye.

#     Returns:
#         set[str] -> employee names
#         None    -> request failed
#     """

#     url = f"{SUPABASE_URL}/rest/v1/{EMPLOYEE_TABLE}"

#     params = {
#         "select": "name"
#     }

#     try:
#         resp = requests.get(
#             url,
#             headers=HEADERS,
#             params=params,
#             timeout=15
#         )

#     except Exception as e:
#         print(
#             f"[sync] ❌ Supabase employee fetch failed: {e}"
#         )
#         return None

#     if resp.status_code != 200:
#         print(
#             f"[sync] ❌ Employee fetch failed: "
#             f"{resp.status_code}"
#         )
#         print(
#             f"[sync] Response: {resp.text[:300]}"
#         )
#         return None

#     try:
#         employees = resp.json()

#     except Exception as e:
#         print(
#             f"[sync] ❌ Invalid employee response: {e}"
#         )
#         return None

#     # Debug
#     print(
#         f"[sync] DEBUG raw employees response: "
#         f"{employees}"
#     )

#     cleaned_names = set()

#     for emp in employees:

#         if not isinstance(emp, dict):
#             continue

#         raw_name = emp.get("name")

#         if not raw_name:
#             continue

#         clean_name = str(raw_name).strip()

#         if clean_name:
#             cleaned_names.add(clean_name)

#     print(
#         f"[sync] Supabase employee names: "
#         f"{sorted(cleaned_names)}"
#     )

#     return cleaned_names


# # =============================================================================
# # RECONCILE LOCAL EMBEDDINGS WITH SUPABASE EMPLOYEES
# # =============================================================================

# def reconcile_embeddings():
#     """
#     Supabase employees table ko source of truth maan kar
#     local embeddings.pkl se deleted employees remove karta hai.

#     IMPORTANT:
#     Agar Supabase request fail hui toh local embeddings ko
#     KABHI delete nahi karega.
#     """

#     print("[sync] 🔍 Checking deleted employees...")

#     supabase_names = get_supabase_employee_names()

#     # -------------------------------------------------------------------------
#     # Supabase unavailable
#     # -------------------------------------------------------------------------

#     if supabase_names is None:
#         print(
#             "[sync] ⚠️ Supabase employee list unavailable. "
#             "Local embeddings untouched."
#         )
#         return

#     # Extra safety
#     supabase_names = {
#         str(name).strip()
#         for name in supabase_names
#         if name
#     }

#     # -------------------------------------------------------------------------
#     # Local embeddings file missing
#     # -------------------------------------------------------------------------

#     if not os.path.exists(EMBEDDINGS_FILE):
#         print(
#             "[sync] No local embeddings found."
#         )
#         return

#     # -------------------------------------------------------------------------
#     # Lock while modifying embeddings
#     # -------------------------------------------------------------------------

#     with _EMB_LOCK:

#         try:
#             with open(
#                 EMBEDDINGS_FILE,
#                 "rb"
#             ) as f:
#                 data = pickle.load(f)

#         except Exception as e:
#             print(
#                 f"[sync] ❌ embeddings.pkl load failed: {e}"
#             )
#             return

#         if not isinstance(data, dict):
#             print(
#                 "[sync] ⚠️ embeddings.pkl dictionary "
#                 "nahi hai — skip."
#             )
#             return

#         # ---------------------------------------------------------------------
#         # Normalize local names
#         #
#         # Example:
#         #
#         # "Rajan " -> "Rajan"
#         # " Rajan" -> "Rajan"
#         # ---------------------------------------------------------------------

#         normalized_data = {}

#         for raw_name, embeddings in data.items():

#             clean_name = str(raw_name).strip()

#             if not clean_name:
#                 continue

#             if clean_name in normalized_data:

#                 # Agar same employee ki duplicate key thi,
#                 # embeddings merge kar do.
#                 existing = normalized_data[clean_name]

#                 if isinstance(existing, list):
#                     existing.extend(embeddings)

#                 else:
#                     normalized_data[clean_name] = list(
#                         existing
#                     ) + list(embeddings)

#             else:

#                 normalized_data[clean_name] = embeddings

#         data = normalized_data

#         local_names = set(data.keys())

#         # ---------------------------------------------------------------------
#         # DEBUG
#         # ---------------------------------------------------------------------

#         print(
#             f"[sync] Local embedding names: "
#             f"{sorted(local_names)}"
#         )

#         # ---------------------------------------------------------------------
#         # Find deleted employees
#         # ---------------------------------------------------------------------

#         deleted_names = local_names - supabase_names

#         if not deleted_names:

#             print(
#                 "[sync] ✅ No deleted employees found."
#             )
#             return

#         # ---------------------------------------------------------------------
#         # Delete embeddings
#         # ---------------------------------------------------------------------

#         for name in deleted_names:

#             print(
#                 f"[sync] 🗑️ Removing embedding: {name}"
#             )

#             del data[name]

#         # ---------------------------------------------------------------------
#         # Atomic save
#         # ---------------------------------------------------------------------

#         temp_file = EMBEDDINGS_FILE + ".tmp"

#         try:

#             with open(
#                 temp_file,
#                 "wb"
#             ) as f:

#                 pickle.dump(
#                     data,
#                     f,
#                     protocol=pickle.HIGHEST_PROTOCOL
#                 )

#             os.replace(
#                 temp_file,
#                 EMBEDDINGS_FILE
#             )

#             # Touch mtime so entry_cameras.py watcher
#             # can detect the change.
#             os.utime(
#                 EMBEDDINGS_FILE,
#                 None
#             )

#             print(
#                 f"[sync] ✅ "
#                 f"{len(deleted_names)} deleted employee "
#                 f"embedding(s) removed."
#             )

#         except Exception as e:

#             print(
#                 f"[sync] ❌ Failed to save after "
#                 f"reconcile: {e}"
#             )

#             if os.path.exists(temp_file):

#                 try:
#                     os.remove(temp_file)

#                 except Exception:
#                     pass


# # =============================================================================
# # SUPABASE STORAGE — RECURSIVE FILE LIST
# # =============================================================================

# def list_files_in_folder(prefix=""):
#     """
#     Supabase Storage mein folder ke andar saari files recursively list karta hai.

#     Example:

#         Rajan/profile_xxx.jpg
#         Rajan/face_1_xxx.jpg
#         Rajan/2026-08-18/Entry-1_xxx.jpg
#     """

#     url = (
#         f"{SUPABASE_URL}/storage/v1/"
#         f"object/list/{BUCKET}"
#     )

#     body = {
#         "prefix": prefix,
#         "limit": 1000,
#         "offset": 0
#     }

#     try:

#         resp = requests.post(
#             url,
#             headers=HEADERS,
#             json=body,
#             timeout=15
#         )

#     except Exception as e:

#         print(
#             f"[sync] ❌ Storage list failed "
#             f"({prefix}): {e}"
#         )

#         return []

#     if resp.status_code != 200:

#         print(
#             f"[sync] list failed ({prefix}): "
#             f"{resp.status_code} "
#             f"{resp.text[:200]}"
#         )

#         return []

#     try:
#         entries = resp.json()

#     except Exception as e:

#         print(
#             f"[sync] ❌ Invalid storage response: {e}"
#         )

#         return []

#     file_paths = []

#     for entry in entries:

#         name = entry.get("name", "")
#         entry_id = entry.get("id")

#         if not name:
#             continue

#         if prefix:

#             full_path = (
#                 f"{prefix}/{name}"
#             )

#         else:

#             full_path = name

#         # Folder
#         if entry_id is None:

#             sub_files = list_files_in_folder(
#                 full_path
#             )

#             file_paths.extend(
#                 sub_files
#             )

#         # Actual file
#         else:

#             file_paths.append(
#                 full_path
#             )

#     return file_paths


# def list_all_storage_files():
#     """
#     Bucket ki saari files recursively list karo.
#     """

#     return list_files_in_folder("")


# # =============================================================================
# # DOWNLOAD
# # =============================================================================

# def download_file(storage_path, local_path):

#     url = (
#         f"{SUPABASE_URL}/storage/v1/"
#         f"object/public/{BUCKET}/{storage_path}"
#     )

#     try:

#         resp = requests.get(
#             url,
#             timeout=30
#         )

#     except Exception as e:

#         print(
#             f"[sync] ❌ Download failed: "
#             f"{storage_path} → {e}"
#         )

#         return False

#     if resp.status_code != 200:

#         print(
#             f"[sync] download failed: "
#             f"{storage_path} → "
#             f"{resp.status_code}"
#         )

#         return False

#     directory = os.path.dirname(
#         local_path
#     )

#     if directory:
#         os.makedirs(
#             directory,
#             exist_ok=True
#         )

#     try:

#         with open(
#             local_path,
#             "wb"
#         ) as f:

#             f.write(
#                 resp.content
#             )

#     except Exception as e:

#         print(
#             f"[sync] ❌ Local save failed: "
#             f"{local_path} → {e}"
#         )

#         return False

#     return True


# # =============================================================================
# # PROCESSED FILE TRACKER
# # =============================================================================

# def load_processed():

#     if not os.path.exists(
#         PROCESSED_FILE
#     ):
#         return set()

#     try:

#         with open(
#             PROCESSED_FILE,
#             "r",
#             encoding="utf-8"
#         ) as f:

#             return {
#                 line.strip()
#                 for line in f
#                 if line.strip()
#             }

#     except Exception as e:

#         print(
#             f"[sync] ⚠️ Could not read "
#             f"processed tracker: {e}"
#         )

#         return set()


# def mark_processed(storage_path):

#     directory = os.path.dirname(
#         PROCESSED_FILE
#     )

#     if directory:
#         os.makedirs(
#             directory,
#             exist_ok=True
#         )

#     with open(
#         PROCESSED_FILE,
#         "a",
#         encoding="utf-8"
#     ) as f:

#         f.write(
#             storage_path + "\n"
#         )


# # =============================================================================
# # FACE MODEL
# # =============================================================================

# def get_face_app():

#     global _face_app

#     if _face_app is None:

#         with _face_lock:

#             if _face_app is None:

#                 from insightface.app import FaceAnalysis

#                 app = FaceAnalysis(
#                     name="buffalo_l",
#                     providers=[
#                         "CUDAExecutionProvider",
#                         "CPUExecutionProvider"
#                     ],
#                     allowed_modules=[
#                         "detection",
#                         "recognition"
#                     ]
#                 )

#                 app.prepare(
#                     ctx_id=0,
#                     det_size=(320, 320)
#                 )

#                 _face_app = app

#                 print(
#                     "[sync] ✅ InsightFace model loaded"
#                 )

#     return _face_app


# # =============================================================================
# # GENERATE EMBEDDING
# # =============================================================================

# def generate_embedding(image_path):

#     import cv2

#     img = cv2.imread(
#         image_path
#     )

#     if img is None:

#         print(
#             f"[sync] ❌ Cannot read image: "
#             f"{image_path}"
#         )

#         return None

#     try:

#         faces = get_face_app().get(
#             img
#         )

#     except Exception as e:

#         print(
#             f"[sync] ❌ Face detection failed: "
#             f"{e}"
#         )

#         return None

#     if not faces:

#         return None

#     # Largest face
#     faces.sort(
#         key=lambda f:
#             (
#                 (f.bbox[2] - f.bbox[0])
#                 *
#                 (f.bbox[3] - f.bbox[1])
#             ),
#         reverse=True
#     )

#     return faces[0].embedding.astype(
#         np.float32
#     )


# # =============================================================================
# # ADD EMBEDDING TO PICKLE
# # =============================================================================

# def add_to_pickle(
#     name,
#     embedding
# ):

#     clean_name = str(
#         name
#     ).strip()

#     if not clean_name:
#         print(
#             "[sync] ⚠️ Empty employee name — skip."
#         )
#         return

#     with _EMB_LOCK:

#         data = {}

#         if os.path.exists(
#             EMBEDDINGS_FILE
#         ):

#             try:

#                 with open(
#                     EMBEDDINGS_FILE,
#                     "rb"
#                 ) as f:

#                     data = pickle.load(f)

#             except Exception as e:

#                 print(
#                     f"[sync] ❌ Failed to load "
#                     f"embeddings.pkl: {e}"
#                 )

#                 return

#         # ---------------------------------------------------------------------
#         # Normalize existing keys
#         # ---------------------------------------------------------------------

#         normalized_data = {}

#         for raw_name, embeddings in data.items():

#             existing_name = str(
#                 raw_name
#             ).strip()

#             if not existing_name:
#                 continue

#             if existing_name in normalized_data:

#                 normalized_data[
#                     existing_name
#                 ].extend(
#                     embeddings
#                 )

#             else:

#                 normalized_data[
#                     existing_name
#                 ] = list(
#                     embeddings
#                 )

#         data = normalized_data

#         # ---------------------------------------------------------------------
#         # Add new embedding
#         # ---------------------------------------------------------------------

#         data.setdefault(
#             clean_name,
#             []
#         )

#         data[clean_name].append(
#             np.asarray(
#                 embedding,
#                 dtype=np.float32
#             )
#         )

#         os.makedirs(
#             os.path.dirname(
#                 EMBEDDINGS_FILE
#             ),
#             exist_ok=True
#         )

#         temp_file = (
#             EMBEDDINGS_FILE
#             + ".tmp"
#         )

#         try:

#             with open(
#                 temp_file,
#                 "wb"
#             ) as f:

#                 pickle.dump(
#                     data,
#                     f,
#                     protocol=pickle.HIGHEST_PROTOCOL
#                 )

#             os.replace(
#                 temp_file,
#                 EMBEDDINGS_FILE
#             )

#             os.utime(
#                 EMBEDDINGS_FILE,
#                 None
#             )

#         except Exception as e:

#             print(
#                 f"[sync] ❌ Failed to save "
#                 f"embeddings.pkl: {e}"
#             )

#             if os.path.exists(
#                 temp_file
#             ):

#                 try:
#                     os.remove(
#                         temp_file
#                     )
#                 except Exception:
#                     pass

#             return

#     print(
#         f"[sync] ✅ {clean_name} ki embedding "
#         f"embeddings.pkl mein add ho gayi"
#     )


# # =============================================================================
# # EMPLOYEE NAME FROM STORAGE PATH
# # =============================================================================

# def extract_name(storage_path):
#     """
#     Storage examples:

#         Rajan/profile_abc.jpg
#             → Rajan, True

#         Rajan/face_1_xyz.jpg
#             → Rajan, False

#         Rajan/2026-08-18/Entry-1_xyz.jpg
#             → Rajan, False

#     IMPORTANT:
#     ONLY profile_*.jpg embedding generate karega.

#     face_* ko intentionally embedding se exclude kiya gaya hai.
#     """

#     parts = (
#         storage_path
#         .replace("\\", "/")
#         .split("/")
#     )

#     if len(parts) < 2:
#         return None, False

#     # First folder = employee name
#     folder = parts[0].strip()

#     filename = parts[-1].lower()

#     if not folder:
#         return None, False

#     # -------------------------------------------------------------------------
#     # ONLY profile photos generate embeddings
#     # -------------------------------------------------------------------------

#     is_profile = filename.startswith(
#         "profile_"
#     )

#     return folder, is_profile


# # =============================================================================
# # MAIN SYNC
# # =============================================================================

# def sync_once():

#     print(
#         "\n[sync] =================================================="
#     )

#     print(
#         f"[sync] Sync started: "
#         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#     )

#     # -------------------------------------------------------------------------
#     # STEP 1
#     # Reconcile deleted employees
#     # -------------------------------------------------------------------------

#     reconcile_embeddings()

#     # -------------------------------------------------------------------------
#     # STEP 2
#     # Load already processed files
#     # -------------------------------------------------------------------------

#     processed = load_processed()

#     # -------------------------------------------------------------------------
#     # STEP 3
#     # Get all Storage files
#     # -------------------------------------------------------------------------

#     all_files = list_all_storage_files()

#     if not all_files:

#         print(
#             "[sync] Koi file nahi mili storage mein"
#         )

#         return

#     print(
#         f"[sync] Storage files found: "
#         f"{len(all_files)}"
#     )

#     new_count = 0

#     # -------------------------------------------------------------------------
#     # STEP 4
#     # Process files
#     # -------------------------------------------------------------------------

#     for storage_path in all_files:

#         # Already processed
#         if storage_path in processed:
#             continue

#         emp_name, is_profile = (
#             extract_name(
#                 storage_path
#             )
#         )

#         # Invalid path
#         if not emp_name:

#             mark_processed(
#                 storage_path
#             )

#             continue

#         local_path = os.path.join(
#             PHOTOS_DIR,
#             storage_path.replace(
#                 "/",
#                 os.sep
#             )
#         )

#         # ---------------------------------------------------------------------
#         # NON-PROFILE PHOTO
#         #
#         # face_* / attendance captures
#         #
#         # Download only.
#         # NO embedding.
#         # ---------------------------------------------------------------------

#         if not is_profile:

#             print(
#                 f"[sync] 📷 Attendance/other photo: "
#                 f"{storage_path} → download only"
#             )

#             if not os.path.exists(
#                 local_path
#             ):

#                 success = download_file(
#                     storage_path,
#                     local_path
#                 )

#                 if not success:
#                     continue

#             mark_processed(
#                 storage_path
#             )

#             continue

#         # ---------------------------------------------------------------------
#         # PROFILE PHOTO
#         # ---------------------------------------------------------------------

#         print(
#             f"[sync] 👤 New profile photo: "
#             f"{storage_path} → {emp_name}"
#         )

#         # ---------------------------------------------------------------------
#         # Download
#         # ---------------------------------------------------------------------

#         if not download_file(
#             storage_path,
#             local_path
#         ):

#             continue

#         # ---------------------------------------------------------------------
#         # Generate embedding
#         # ---------------------------------------------------------------------

#         emb = generate_embedding(
#             local_path
#         )

#         if emb is None:

#             print(
#                 f"[sync] ⚠️ {storage_path}: "
#                 f"face detect nahi hua — skip"
#             )

#             # Mark processed so same bad image
#             # every 30 seconds retry na ho.
#             mark_processed(
#                 storage_path
#             )

#             continue

#         # ---------------------------------------------------------------------
#         # Add embedding
#         # ---------------------------------------------------------------------

#         add_to_pickle(
#             emp_name,
#             emb
#         )

#         # ---------------------------------------------------------------------
#         # Mark processed
#         # ---------------------------------------------------------------------

#         mark_processed(
#             storage_path
#         )

#         new_count += 1

#     # -------------------------------------------------------------------------
#     # SUMMARY
#     # -------------------------------------------------------------------------

#     if new_count > 0:

#         print(
#             f"[sync] ✅ {new_count} "
#             f"new profile embedding(s) added"
#         )

#     else:

#         print(
#             "[sync] No new profile embeddings."
#         )

#     print(
#         "[sync] ==================================================\n"
#     )


# # =============================================================================
# # MAIN LOOP
# # =============================================================================

# def main():

#     print(
#         "[sync] Photo sync watcher started — "
#         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#     )

#     print(
#         f"[sync] Supabase bucket: {BUCKET}"
#     )

#     print(
#         f"[sync] Poll interval: {POLL_INTERVAL}s"
#     )

#     print(
#         f"[sync] Embeddings file: "
#         f"{EMBEDDINGS_FILE}"
#     )

#     print(
#         "[sync] IMPORTANT: Only profile_*.jpg "
#         "will generate embeddings."
#     )

#     # -------------------------------------------------------------------------
#     # Initial sync
#     # -------------------------------------------------------------------------

#     try:

#         sync_once()

#     except Exception as e:

#         print(
#             f"[sync] ❌ Startup error: {e}"
#         )

#     # -------------------------------------------------------------------------
#     # Continuous polling
#     # -------------------------------------------------------------------------

#     while True:

#         time.sleep(
#             POLL_INTERVAL
#         )

#         try:

#             sync_once()

#         except Exception as e:

#             print(
#                 f"[sync] ❌ Sync error: {e}"
#             )


# # =============================================================================
# # ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     main()

"""
photo_sync.py — Jetson pe chalne wala poller.

Every 30s:

1. Supabase employees table se current employee names fetch karta hai.
2. Deleted employees ki local embeddings remove karta hai.
3. Supabase Storage se files check karta hai.
4. Employee folder ke DIRECT andar jo bhi image hai,
   uski embedding generate karta hai.
5. Employee ke subfolders/date folders ke andar wali images
   attendance captures maani jaati hain — unki embedding nahi banti.
6. Embeddings embeddings.pkl mein save hoti hain.
7. processed files synced_photos.txt mein track hoti hain.

Storage example:

Rajan/
    profile_xxx.jpg        -> embedding ✅
    face_1_xxx.jpg         -> embedding ✅
    face_2_xxx.jpg         -> embedding ✅
    image_abc.jpg          -> embedding ✅

Rajan/
    2026-08-18/
        Entry-1_xxx.jpg    -> embedding ❌
        Entry-2_xxx.jpg    -> embedding ❌
"""


import os
import time
import pickle
import requests
import threading
import numpy as np
from datetime import datetime


# =============================================================================
# CONFIG
# =============================================================================

SUPABASE_URL = "https://bjgiirtpqfjaaxnftptf.supabase.co"

# Apni existing Supabase anon key yahan rakho
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"

BUCKET = "employee-photos"
EMPLOYEE_TABLE = "employees"

POLL_INTERVAL = 30

EMBEDDINGS_FILE = (
    "/home/cyamsys/Documents/HR_attendance/"
    "embeddings/embeddings.pkl"
)

PHOTOS_DIR = (
    "/home/cyamsys/Documents/HR_attendance/"
    "photos"
)

PROCESSED_FILE = (
    "/home/cyamsys/Documents/HR_attendance/"
    "embeddings/synced_photos.txt"
)


# =============================================================================
# SUPABASE HEADERS
# =============================================================================

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}


# =============================================================================
# LOCKS / FACE MODEL
# =============================================================================

_EMB_LOCK = threading.Lock()

_face_app = None
_face_lock = threading.Lock()


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(name):
    """
    Employee name normalize karta hai.

    Examples:

        "Rajan"   -> "Rajan"
        "Rajan "  -> "Rajan"
        " Rajan"  -> "Rajan"
        " Rajan " -> "Rajan"
    """

    if name is None:
        return ""

    return str(name).strip()


def is_image_file(filename):
    """
    Check karta hai ki file image hai ya nahi.
    """

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".jfif",
    }

    _, ext = os.path.splitext(filename.lower())

    return ext in image_extensions


# =============================================================================
# SUPABASE EMPLOYEE NAMES FETCH
# =============================================================================

def get_supabase_employee_names():
    """
    Supabase employees table se current employee names fetch karta hai.

    IMPORTANT:
    - Whitespace normalize hota hai.
    - HTTP request fail -> None
    - Valid empty table -> empty set()
    """

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        f"{EMPLOYEE_TABLE}"
    )

    params = {
        "select": "name",
    }

    try:

        resp = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=15,
        )

    except Exception as e:

        print(
            f"[sync] ❌ Supabase employee fetch failed: {e}"
        )

        return None

    print(
        f"[sync] Employee API status: "
        f"{resp.status_code}"
    )

    if resp.status_code != 200:

        print(
            f"[sync] ❌ Employee fetch failed: "
            f"{resp.status_code}"
        )

        print(
            f"[sync] Response: "
            f"{resp.text[:500]}"
        )

        return None

    try:

        employees = resp.json()

    except Exception as e:

        print(
            f"[sync] ❌ Invalid employee response: {e}"
        )

        return None

    print(
        f"[sync] DEBUG raw employees response: "
        f"{employees}"
    )

    cleaned_names = set()

    for emp in employees:

        if not isinstance(emp, dict):
            continue

        raw_name = emp.get("name")

        if not raw_name:
            continue

        clean_name = normalize_name(raw_name)

        if clean_name:

            cleaned_names.add(
                clean_name
            )

    print(
        f"[sync] Supabase employee names: "
        f"{sorted(cleaned_names)}"
    )

    return cleaned_names


# =============================================================================
# RECONCILE EMBEDDINGS WITH SUPABASE EMPLOYEES
# =============================================================================

def reconcile_embeddings():
    """
    Supabase employees table ko source of truth maan kar
    local embeddings.pkl se deleted employees remove karta hai.

    IMPORTANT SAFETY:

    Supabase fetch fail hua -> kuch delete nahi hoga.

    Supabase response genuinely empty hua -> koi local employee
    delete karne se pehle storage presence ka fallback check kiya
    jayega.
    """

    print(
        "[sync] 🔍 Checking deleted employees..."
    )

    supabase_names = (
        get_supabase_employee_names()
    )

    # -------------------------------------------------------------------------
    # Supabase unavailable
    # -------------------------------------------------------------------------

    if supabase_names is None:

        print(
            "[sync] ⚠️ Supabase employee list unavailable. "
            "Local embeddings untouched."
        )

        return

    # -------------------------------------------------------------------------
    # IMPORTANT SAFETY:
    # Empty result ko immediately "all employees deleted" mat maano.
    #
    # Agar API [] return karti hai, pehle storage se employee folders check
    # karenge. Isse accidental delete/re-add loop avoid hoga.
    # -------------------------------------------------------------------------

    if len(supabase_names) == 0:

        print(
            "[sync] ⚠️ Supabase employees response is EMPTY."
        )

        print(
            "[sync] ⚠️ Local embeddings ko abhi delete "
            "nahi karenge."
        )

        return

    # -------------------------------------------------------------------------
    # Embeddings file missing
    # -------------------------------------------------------------------------

    if not os.path.exists(
        EMBEDDINGS_FILE
    ):

        print(
            "[sync] No local embeddings found."
        )

        return

    # -------------------------------------------------------------------------
    # Lock
    # -------------------------------------------------------------------------

    with _EMB_LOCK:

        try:

            with open(
                EMBEDDINGS_FILE,
                "rb",
            ) as f:

                data = pickle.load(f)

        except Exception as e:

            print(
                f"[sync] ❌ embeddings.pkl load failed: {e}"
            )

            return

        if not isinstance(data, dict):

            print(
                "[sync] ⚠️ embeddings.pkl dictionary "
                "nahi hai — skip."
            )

            return

        # ---------------------------------------------------------------------
        # Normalize local names
        # ---------------------------------------------------------------------

        normalized_data = {}

        for raw_name, embeddings in data.items():

            clean_name = normalize_name(
                raw_name
            )

            if not clean_name:
                continue

            if clean_name in normalized_data:

                existing = normalized_data[
                    clean_name
                ]

                try:
                    existing.extend(
                        list(embeddings)
                    )
                except Exception:
                    pass

            else:

                try:

                    normalized_data[
                        clean_name
                    ] = list(embeddings)

                except Exception:

                    normalized_data[
                        clean_name
                    ] = embeddings

        data = normalized_data

        local_names = set(
            data.keys()
        )

        # ---------------------------------------------------------------------
        # Debug
        # ---------------------------------------------------------------------

        print(
            f"[sync] Local embedding names: "
            f"{sorted(local_names)}"
        )

        # ---------------------------------------------------------------------
        # Deleted employees
        # ---------------------------------------------------------------------

        deleted_names = (
            local_names - supabase_names
        )

        if not deleted_names:

            print(
                "[sync] ✅ No deleted employees found."
            )

            return

        # ---------------------------------------------------------------------
        # Delete
        # ---------------------------------------------------------------------

        for name in deleted_names:

            print(
                f"[sync] 🗑️ Removing embedding: "
                f"{name}"
            )

            del data[name]

        # ---------------------------------------------------------------------
        # Atomic save
        # ---------------------------------------------------------------------

        temp_file = (
            EMBEDDINGS_FILE
            + ".tmp"
        )

        try:

            with open(
                temp_file,
                "wb",
            ) as f:

                pickle.dump(
                    data,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            os.replace(
                temp_file,
                EMBEDDINGS_FILE,
            )

            os.utime(
                EMBEDDINGS_FILE,
                None,
            )

            print(
                f"[sync] ✅ "
                f"{len(deleted_names)} deleted employee "
                f"embedding(s) removed."
            )

        except Exception as e:

            print(
                f"[sync] ❌ Failed to save after "
                f"reconcile: {e}"
            )

            if os.path.exists(
                temp_file
            ):

                try:
                    os.remove(
                        temp_file
                    )
                except Exception:
                    pass


# =============================================================================
# SUPABASE STORAGE — RECURSIVE LIST
# =============================================================================

def list_files_in_folder(prefix=""):
    """
    Supabase Storage mein folder ke andar files recursively list karta hai.
    """

    url = (
        f"{SUPABASE_URL}/storage/v1/"
        f"object/list/{BUCKET}"
    )

    body = {
        "prefix": prefix,
        "limit": 1000,
        "offset": 0,
    }

    try:

        resp = requests.post(
            url,
            headers=HEADERS,
            json=body,
            timeout=15,
        )

    except Exception as e:

        print(
            f"[sync] ❌ Storage list failed "
            f"({prefix}): {e}"
        )

        return []

    if resp.status_code != 200:

        print(
            f"[sync] ❌ Storage list failed "
            f"({prefix}): "
            f"{resp.status_code} "
            f"{resp.text[:300]}"
        )

        return []

    try:

        entries = resp.json()

    except Exception as e:

        print(
            f"[sync] ❌ Invalid storage response: {e}"
        )

        return []

    file_paths = []

    for entry in entries:

        name = entry.get(
            "name",
            "",
        )

        entry_id = entry.get(
            "id"
        )

        if not name:
            continue

        if prefix:

            full_path = (
                f"{prefix}/{name}"
            )

        else:

            full_path = name

        # ---------------------------------------------------------------------
        # Folder
        # ---------------------------------------------------------------------

        if entry_id is None:

            sub_files = (
                list_files_in_folder(
                    full_path
                )
            )

            file_paths.extend(
                sub_files
            )

        # ---------------------------------------------------------------------
        # Actual file
        # ---------------------------------------------------------------------

        else:

            file_paths.append(
                full_path
            )

    return file_paths


def list_all_storage_files():

    return list_files_in_folder(
        ""
    )


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_file(
    storage_path,
    local_path,
):
    """
    Supabase Storage se file download karta hai.
    """

    url = (
        f"{SUPABASE_URL}/storage/v1/"
        f"object/public/{BUCKET}/"
        f"{storage_path}"
    )

    try:

        resp = requests.get(
            url,
            timeout=30,
        )

    except Exception as e:

        print(
            f"[sync] ❌ Download failed: "
            f"{storage_path} → {e}"
        )

        return False

    if resp.status_code != 200:

        print(
            f"[sync] ❌ download failed: "
            f"{storage_path} → "
            f"{resp.status_code}"
        )

        print(
            f"[sync] Response: "
            f"{resp.text[:200]}"
        )

        return False

    directory = os.path.dirname(
        local_path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

    try:

        with open(
            local_path,
            "wb",
        ) as f:

            f.write(
                resp.content
            )

    except Exception as e:

        print(
            f"[sync] ❌ Local save failed: "
            f"{local_path} → {e}"
        )

        return False

    return True


# =============================================================================
# PROCESSED FILE TRACKER
# =============================================================================

def load_processed():

    if not os.path.exists(
        PROCESSED_FILE
    ):

        return set()

    try:

        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return {
                line.strip()
                for line in f
                if line.strip()
            }

    except Exception as e:

        print(
            f"[sync] ⚠️ Could not read "
            f"processed tracker: {e}"
        )

        return set()


def mark_processed(
    storage_path
):

    directory = os.path.dirname(
        PROCESSED_FILE
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        PROCESSED_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            storage_path + "\n"
        )


# =============================================================================
# FACE MODEL
# =============================================================================

def get_face_app():

    global _face_app

    if _face_app is None:

        with _face_lock:

            if _face_app is None:

                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=[
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                    allowed_modules=[
                        "detection",
                        "recognition",
                    ],
                )

                app.prepare(
                    ctx_id=0,
                    det_size=(320, 320),
                )

                _face_app = app

                print(
                    "[sync] ✅ InsightFace model loaded"
                )

    return _face_app


# =============================================================================
# GENERATE EMBEDDING
# =============================================================================

def generate_embedding(
    image_path
):
    """
    Image se largest detected face ki embedding generate karta hai.
    """

    import cv2

    img = cv2.imread(
        image_path
    )

    if img is None:

        print(
            f"[sync] ❌ Cannot read image: "
            f"{image_path}"
        )

        return None

    try:

        faces = get_face_app().get(
            img
        )

    except Exception as e:

        print(
            f"[sync] ❌ Face detection failed: "
            f"{e}"
        )

        return None

    if not faces:

        return None

    # -------------------------------------------------------------------------
    # Largest face choose karo
    # -------------------------------------------------------------------------

    faces.sort(
        key=lambda f:
            (
                (
                    f.bbox[2]
                    -
                    f.bbox[0]
                )
                *
                (
                    f.bbox[3]
                    -
                    f.bbox[1]
                )
            ),
        reverse=True,
    )

    return faces[
        0
    ].embedding.astype(
        np.float32
    )


# =============================================================================
# ADD EMBEDDING TO PICKLE
# =============================================================================

def add_to_pickle(
    name,
    embedding,
):
    """
    Employee name ke andar embedding append karta hai.
    """

    clean_name = normalize_name(
        name
    )

    if not clean_name:

        print(
            "[sync] ⚠️ Empty employee name — skip."
        )

        return False

    with _EMB_LOCK:

        data = {}

        # ---------------------------------------------------------------------
        # Existing pickle load
        # ---------------------------------------------------------------------

        if os.path.exists(
            EMBEDDINGS_FILE
        ):

            try:

                with open(
                    EMBEDDINGS_FILE,
                    "rb",
                ) as f:

                    data = pickle.load(
                        f
                    )

            except Exception as e:

                print(
                    f"[sync] ❌ Failed to load "
                    f"embeddings.pkl: {e}"
                )

                return False

        if not isinstance(
            data,
            dict,
        ):

            data = {}

        # ---------------------------------------------------------------------
        # Normalize existing names
        # ---------------------------------------------------------------------

        normalized_data = {}

        for raw_name, embeddings in data.items():

            existing_name = normalize_name(
                raw_name
            )

            if not existing_name:
                continue

            if existing_name in normalized_data:

                try:

                    normalized_data[
                        existing_name
                    ].extend(
                        list(embeddings)
                    )

                except Exception:

                    pass

            else:

                try:

                    normalized_data[
                        existing_name
                    ] = list(
                        embeddings
                    )

                except Exception:

                    normalized_data[
                        existing_name
                    ] = embeddings

        data = normalized_data

        # ---------------------------------------------------------------------
        # Add new embedding
        # ---------------------------------------------------------------------

        data.setdefault(
            clean_name,
            [],
        )

        data[
            clean_name
        ].append(
            np.asarray(
                embedding,
                dtype=np.float32,
            )
        )

        # ---------------------------------------------------------------------
        # Ensure directory
        # ---------------------------------------------------------------------

        directory = os.path.dirname(
            EMBEDDINGS_FILE
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        # ---------------------------------------------------------------------
        # Atomic save
        # ---------------------------------------------------------------------

        temp_file = (
            EMBEDDINGS_FILE
            + ".tmp"
        )

        try:

            with open(
                temp_file,
                "wb",
            ) as f:

                pickle.dump(
                    data,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            os.replace(
                temp_file,
                EMBEDDINGS_FILE,
            )

            os.utime(
                EMBEDDINGS_FILE,
                None,
            )

        except Exception as e:

            print(
                f"[sync] ❌ Failed to save "
                f"embeddings.pkl: {e}"
            )

            if os.path.exists(
                temp_file
            ):

                try:
                    os.remove(
                        temp_file
                    )
                except Exception:
                    pass

            return False

    print(
        f"[sync] ✅ {clean_name} ki embedding "
        f"embeddings.pkl mein add ho gayi"
    )

    return True


# =============================================================================
# EXTRACT EMPLOYEE NAME + IMAGE TYPE
# =============================================================================

def extract_name(
    storage_path
):
    """
    Storage structure:

        Rajan/profile_xxx.jpg
            -> Rajan, employee image ✅

        Rajan/face_1_xxx.jpg
            -> Rajan, employee image ✅

        Rajan/face_2_xxx.jpg
            -> Rajan, employee image ✅

        Rajan/image_xxx.jpg
            -> Rajan, employee image ✅

        Rajan/anything.jpg
            -> Rajan, employee image ✅

        Rajan/2026-08-18/Entry-1_xxx.jpg
            -> Rajan, attendance image ❌

        Rajan/2026-08-18/Entry-2_xxx.jpg
            -> Rajan, attendance image ❌

    RULE:

    Employee folder ke direct andar image:
        embedding YES

    Employee ke subfolder ke andar image:
        embedding NO
    """

    parts = (
        storage_path
        .replace(
            "\\",
            "/",
        )
        .split("/")
    )

    if len(parts) < 2:

        return None, False

    # -------------------------------------------------------------------------
    # Employee name = first folder
    # -------------------------------------------------------------------------

    employee_name = normalize_name(
        parts[0]
    )

    if not employee_name:

        return None, False

    filename = parts[
        -1
    ]

    # -------------------------------------------------------------------------
    # Must be an image
    # -------------------------------------------------------------------------

    if not is_image_file(
        filename
    ):

        return employee_name, False

    # -------------------------------------------------------------------------
    # Directly inside employee folder
    #
    # Example:
    #
    # Rajan/image.jpg
    # len(parts) = 2
    #
    # -------------------------------------------------------------------------

    is_employee_attachment = (
        len(parts) == 2
    )

    return (
        employee_name,
        is_employee_attachment,
    )


# =============================================================================
# MAIN SYNC
# =============================================================================

def sync_once():

    print(
        "\n[sync] =================================================="
    )

    print(
        f"[sync] Sync started: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # -------------------------------------------------------------------------
    # STEP 1 — Reconcile employees
    # -------------------------------------------------------------------------

    reconcile_embeddings()

    # -------------------------------------------------------------------------
    # STEP 2 — Load processed tracker
    # -------------------------------------------------------------------------

    processed = load_processed()

    # -------------------------------------------------------------------------
    # STEP 3 — List all Storage files
    # -------------------------------------------------------------------------

    all_files = (
        list_all_storage_files()
    )

    if not all_files:

        print(
            "[sync] Koi file nahi mili storage mein"
        )

        return

    print(
        f"[sync] Storage files found: "
        f"{len(all_files)}"
    )

    # -------------------------------------------------------------------------
    # Debug: show files
    # -------------------------------------------------------------------------

    for path in all_files:

        print(
            f"[sync] Storage file: {path}"
        )

    new_count = 0

    # -------------------------------------------------------------------------
    # STEP 4 — Process every storage file
    # -------------------------------------------------------------------------

    for storage_path in all_files:

        # ---------------------------------------------------------------------
        # Already processed?
        # ---------------------------------------------------------------------

        if storage_path in processed:

            continue

        # ---------------------------------------------------------------------
        # Extract employee + type
        # ---------------------------------------------------------------------

        emp_name, is_employee_image = (
            extract_name(
                storage_path
            )
        )

        # ---------------------------------------------------------------------
        # Invalid path
        # ---------------------------------------------------------------------

        if not emp_name:

            print(
                f"[sync] ⚠️ Invalid storage path: "
                f"{storage_path}"
            )

            mark_processed(
                storage_path
            )

            continue

        # ---------------------------------------------------------------------
        # Local file path
        # ---------------------------------------------------------------------

        local_path = os.path.join(
            PHOTOS_DIR,
            storage_path.replace(
                "/",
                os.sep,
            )
        )

        # ---------------------------------------------------------------------
        # ATTENDANCE / SUBFOLDER IMAGE
        #
        # Download only.
        # No embedding.
        # ---------------------------------------------------------------------

        if not is_employee_image:

            print(
                f"[sync] 📷 Attendance/other image: "
                f"{storage_path} → download only"
            )

            if not os.path.exists(
                local_path
            ):

                success = (
                    download_file(
                        storage_path,
                        local_path,
                    )
                )

                if not success:

                    # Retry next cycle
                    continue

            mark_processed(
                storage_path
            )

            continue

        # ---------------------------------------------------------------------
        # EMPLOYEE ATTACHMENT
        #
        # Any direct image inside employee folder:
        #
        # profile_*
        # face_*
        # image_*
        # anything.*
        #
        # all will generate embeddings.
        # ---------------------------------------------------------------------

        print(
            f"[sync] 👤 New employee image: "
            f"{storage_path} → {emp_name}"
        )

        # ---------------------------------------------------------------------
        # Download
        # ---------------------------------------------------------------------

        if not download_file(
            storage_path,
            local_path,
        ):

            # Retry next cycle
            continue

        # ---------------------------------------------------------------------
        # Generate embedding
        # ---------------------------------------------------------------------

        emb = generate_embedding(
            local_path
        )

        if emb is None:

            print(
                f"[sync] ⚠️ {storage_path}: "
                f"face detect nahi hua — skip"
            )

            # Mark processed to avoid endless retry
            mark_processed(
                storage_path
            )

            continue

        # ---------------------------------------------------------------------
        # Save embedding
        # ---------------------------------------------------------------------

        saved = add_to_pickle(
            emp_name,
            emb,
        )

        if not saved:

            print(
                f"[sync] ❌ Embedding save failed: "
                f"{storage_path}"
            )

            # Do NOT mark processed.
            # Next cycle retry karega.
            continue

        # ---------------------------------------------------------------------
        # Mark as processed
        # ---------------------------------------------------------------------

        mark_processed(
            storage_path
        )

        new_count += 1

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    if new_count > 0:

        print(
            f"[sync] ✅ {new_count} "
            f"new employee image embedding(s) added"
        )

    else:

        print(
            "[sync] No new employee embeddings."
        )

    print(
        "[sync] ==================================================\n"
    )


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():

    print(
        "[sync] Photo sync watcher started — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"[sync] Supabase bucket: {BUCKET}"
    )

    print(
        f"[sync] Employee table: {EMPLOYEE_TABLE}"
    )

    print(
        f"[sync] Poll interval: {POLL_INTERVAL}s"
    )

    print(
        f"[sync] Embeddings file: "
        f"{EMBEDDINGS_FILE}"
    )

    print(
        "[sync] Employee folder ke DIRECT andar "
        "har image ki embedding banegi."
    )

    print(
        "[sync] Employee ke subfolder/date-folder ki "
        "images embedding se skip hongi."
    )

    # -------------------------------------------------------------------------
    # Startup sync
    # -------------------------------------------------------------------------

    try:

        sync_once()

    except Exception as e:

        print(
            f"[sync] ❌ Startup error: {e}"
        )

    # -------------------------------------------------------------------------
    # Continuous polling
    # -------------------------------------------------------------------------

    while True:

        time.sleep(
            POLL_INTERVAL
        )

        try:

            sync_once()

        except Exception as e:

            print(
                f"[sync] ❌ Sync error: {e}"
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()