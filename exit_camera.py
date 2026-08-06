# import cv2
# import os
# import time
# import pickle
# import threading
# import numpy as np
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from urllib.parse import quote

# import config
# import db

# # =============================================================================
# #  CONFIG  —  edit this section only
# # =============================================================================

# EMBEDDINGS_FILE = config.EMBEDDINGS_FILE
# PHOTOS_DIR      = config.PHOTOS_DIR

# RTSP_HOST  = "192.168.1.103:554"
# RTSP_PATH  = "/video/live?channel=1&subtype=0"
# RTSP_USER  = "admin"
# RTSP_PASS  = "Cyamsys@123"

# EXIT_CAMERA_ID = "EXIT MAIN"   # stored as camera_id in the attendance table

# FRAME_WIDTH  = 1280
# FRAME_HEIGHT = 720

# THRESHOLD             = 0.50
# DETECTION_INTERVAL    = 5
# TRACK_HOLD_FRAMES     = 10
# MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
# STATUS_CACHE_TTL      = config.STATUS_CACHE_TTL
# TARGET_FPS            = 15

# EXIT_REGION = (0, 252, 1200, 718)

# # =============================================================================
# #  PRE-BUILT NUMPY MATRIX for vectorized recognition
# # =============================================================================
# _known_matrix = None
# _known_names  = []

# # =============================================================================
# #  STATUS CACHE — { name: "Present" | "Exit" | None }, refreshed from DB
# #  at most once every STATUS_CACHE_TTL seconds instead of a query per face.
# #
# #  DATE-ROLLOVER FIX: this service runs 24x7 under systemd (Restart=always),
# #  so without tracking the current date, an "Exit" status from yesterday
# #  would silently stay cached across midnight. _cache_date tracks the day
# #  this cache currently reflects; any refresh that notices the date changed
# #  does a full reset instead of a normal TTL-based refresh.
# # =============================================================================
# _status_cache      = {}
# _status_cache_lock = threading.Lock()
# _status_cache_ts   = 0.0
# _cache_date        = None   # 'YYYY-MM-DD' — day the cache currently reflects


# def _refresh_status_cache_if_stale():
#     """DB se aaj ke saare statuses ek baar padho aur cache update karo.
#        Agar date badal gayi hai (raat 12 baj gayi), cache ko poora reset
#        karo — warna kal ka 'Exit' status aaj bhi atka reh jaayega."""
#     global _status_cache_ts, _cache_date
#     now       = time.time()
#     today_str = datetime.now().strftime("%Y-%m-%d")
#     day_changed = (_cache_date is not None and _cache_date != today_str)

#     if not day_changed and now - _status_cache_ts < STATUS_CACHE_TTL:
#         return
#     _status_cache_ts = now
#     _cache_date       = today_str

#     today_rows = db.read_day_rows(today_str)
#     tmp = {}
#     # rows are newest-first per query; keep the first (latest) status per name
#     for r in today_rows:
#         tmp.setdefault(r["name"], r["status"])

#     with _status_cache_lock:
#         _status_cache.clear()
#         _status_cache.update(tmp)

#     if day_changed:
#         with exit_cache_lock:
#             exit_cache.clear()
#             for name, status in tmp.items():
#                 if status == "Exit":
#                     exit_cache.add(name)
#         print(f"[cache] Naya din shuru — {today_str} ke liye status cache reset ho gaya.")


# def _cached_status(name: str):
#     with _status_cache_lock:
#         return _status_cache.get(name, None)


# def _set_cached_status(name: str, status: str):
#     with _status_cache_lock:
#         _status_cache[name] = status


# # =============================================================================
# #  EXIT SUMMARY CACHE — names who exited today, for the on-screen overlay
# # =============================================================================
# exit_cache      = set()
# exit_cache_lock = threading.Lock()


# def _rebuild_caches_from_db():
#     """Startup pe DB se aaj ka status ek baar padho, dono caches seed karo."""
#     global _cache_date
#     _cache_date = datetime.now().strftime("%Y-%m-%d")
#     today_rows  = db.read_day_rows(_cache_date)
#     last_status = {}
#     for r in today_rows:
#         last_status.setdefault(r["name"], r["status"])
#     with exit_cache_lock:
#         exit_cache.clear()
#         for name, status in last_status.items():
#             if status == "Exit":
#                 exit_cache.add(name)
#     with _status_cache_lock:
#         _status_cache.update(last_status)


# # =============================================================================
# #  PHOTO CAPTURE  — same crop/pad/sharpen logic as entry_cameras.py, so
# #  exit photos look consistent with entry photos.
# # =============================================================================
# def _safe(name: str) -> str:
#     return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# def _save_capture(frame, bbox, name, cam_id):
#     """Bbox ke around padding le kar face + shoulders wala crop banao,
#        fixed-size photo (480x600) me PAD karke (crop nahi) resize karo,
#        aur high-quality JPEG save karo. Returns path RELATIVE to PHOTOS_DIR."""
#     try:
#         h, w = frame.shape[:2]
#         x1, y1, x2, y2 = bbox
#         bw, bh = (x2 - x1), (y2 - y1)

#         pad_x        = int(bw * 0.9)
#         pad_y_top    = int(bh * 0.35)
#         pad_y_bottom = int(bh * 1.8)

#         x1 = max(0, x1 - pad_x)
#         y1 = max(0, y1 - pad_y_top)
#         x2 = min(w, x2 + pad_x)
#         y2 = min(h, y2 + pad_y_bottom)

#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             return None

#         TARGET_W, TARGET_H = 480, 600
#         ch, cw = crop.shape[:2]
#         target_ratio = TARGET_W / TARGET_H
#         crop_ratio    = cw / ch

#         if crop_ratio > target_ratio:
#             need_h  = int(cw / target_ratio)
#             extra   = max(0, need_h - ch)
#             top_pad = extra // 2
#             bot_pad = extra - top_pad
#             crop = cv2.copyMakeBorder(crop, top_pad, bot_pad, 0, 0,
#                                        cv2.BORDER_REPLICATE)
#         else:
#             need_w    = int(ch * target_ratio)
#             extra     = max(0, need_w - cw)
#             left_pad  = extra // 2
#             right_pad = extra - left_pad
#             crop = cv2.copyMakeBorder(crop, 0, 0, left_pad, right_pad,
#                                        cv2.BORDER_REPLICATE)

#         crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

#         sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         crop = cv2.filter2D(crop, -1, sharpen_kernel)

#         day_dir = datetime.now().strftime("%Y-%m-%d")
#         rel_dir = os.path.join(_safe(name), day_dir)
#         abs_dir = os.path.join(PHOTOS_DIR, rel_dir)
#         os.makedirs(abs_dir, exist_ok=True)

#         fname    = f"{cam_id}_{datetime.now().strftime('%H%M%S')}.jpg"
#         rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
#         abs_path = os.path.join(abs_dir, fname)

#         cv2.imwrite(abs_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         return rel_path
#     except Exception as e:
#         print(f"[photo] save failed for {name}: {e}")
#         return None


# # =============================================================================
# #  LOAD EMBEDDINGS
# # =============================================================================
# print("Loading embeddings …")
# with open(EMBEDDINGS_FILE, "rb") as f:
#     known_embeddings = pickle.load(f)

# known_face_db = {}
# for name, emb_list in known_embeddings.items():
#     emb_array = np.array(emb_list, dtype=np.float32)
#     mean_emb  = np.mean(emb_array, axis=0)
#     mean_emb  = mean_emb / np.linalg.norm(mean_emb)
#     known_face_db[name] = mean_emb

# _known_names  = list(known_face_db.keys())
# _known_matrix = np.stack(list(known_face_db.values()), axis=0)  # (N, 512)

# print(f"Total employees loaded: {len(known_face_db)}\n")

# # =============================================================================
# #  LOAD FACE MODEL
# # =============================================================================
# app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CUDAExecutionProvider"],
# )
# app.prepare(ctx_id=0, det_size=(320, 320))


# def recognize_face(live_embedding: np.ndarray):
#     emb    = live_embedding / np.linalg.norm(live_embedding)
#     scores = _known_matrix @ emb
#     idx    = int(np.argmax(scores))
#     return _known_names[idx], float(scores[idx])


# def is_in_exit_region(cx, cy):
#     x1, y1, x2, y2 = EXIT_REGION
#     return x1 <= cx <= x2 and y1 <= cy <= y2


# # =============================================================================
# #  EXIT MARKING  — Postgres-backed, with duplicate prevention + photo save.
# #  Every recognition attempt is now logged via db.log_detection(), regardless
# #  of outcome, so you have an exact independent timestamp for EVERY event —
# #  including "already_exited"/"not_inside" hits that never reach
# #  db.mark_exit().
# # =============================================================================
# def mark_exit(name, frame, bbox, confidence):
#     """Mirrors entry_cameras.py's mark_attendance(), but for exits.
#        Returns 'marked' | 'already_exited' | 'not_inside'."""
#     last_status = _cached_status(name)

#     if last_status is None:
#         db.log_detection(name, EXIT_CAMERA_ID, confidence, "not_inside")
#         return "not_inside"
#     if last_status == "Exit":
#         db.log_detection(name, EXIT_CAMERA_ID, confidence, "already_exited")
#         return "already_exited"

#     photo_path = _save_capture(frame, bbox, name, EXIT_CAMERA_ID)

#     result = db.mark_exit(
#         name, EXIT_CAMERA_ID, confidence=confidence, photo_path=photo_path
#     )

#     db.log_detection(name, EXIT_CAMERA_ID, confidence, result)

#     if result == "marked":
#         _set_cached_status(name, "Exit")
#         with exit_cache_lock:
#             exit_cache.add(name)
#         print(f"[EXIT MARKED] {name} at {datetime.now().strftime('%H:%M:%S')}")
#         return "marked"

#     return result


# # =============================================================================
# #  FFMPEG / GSTREAMER CAPTURE HELPERS
# # =============================================================================
# def _build_ffmpeg_url(host, path, user, pw):
#     safe_pw = quote(pw, safe="")
#     return f"rtsp://{user}:{safe_pw}@{host}{path}"


# def _open_ffmpeg_cap(label, ffmpeg_url):
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#         "rtsp_transport;tcp|stimeout;5000000"
#     )
#     print(f"[{label}] Trying FFmpeg: {ffmpeg_url}")
#     cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
#     if cap.isOpened():
#         print(f"[{label}] ✅ FFmpeg stream opened")
#     else:
#         print(f"[{label}] ❌ FFmpeg failed to open stream")
#     return cap


# def build_gst_pipeline(host, path, user, password, width, height):
#     return (
#         f'rtspsrc location="rtsp://{host}{path}" '
#         f'user-id="{user}" user-pw="{password}" '
#         f'latency=100 protocols=tcp ! '
#         f'decodebin ! '
#         f'videoconvert ! '
#         f'videoscale ! '
#         f'video/x-raw,width={width},height={height},format=BGR ! '
#         f'appsink drop=true max-buffers=1 sync=false'
#     )


# def _open_capture():
#     gst_pipeline = build_gst_pipeline(
#         RTSP_HOST, RTSP_PATH, RTSP_USER, RTSP_PASS, FRAME_WIDTH, FRAME_HEIGHT
#     )
#     print(f"Trying GStreamer pipeline:\n  {gst_pipeline}")
#     cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

#     if cap.isOpened():
#         deadline = time.time() + 5.0
#         while time.time() < deadline:
#             ret, frame = cap.read()
#             if ret and frame is not None:
#                 print("✅ GStreamer pipeline confirmed (decodebin)")
#                 return cap
#             time.sleep(0.05)
#         cap.release()
#         print("⚠️  GStreamer opened but produced no frames — falling back to FFmpeg")
#     else:
#         print("⚠️  GStreamer failed to open — falling back to FFmpeg")

#     ffmpeg_url = _build_ffmpeg_url(RTSP_HOST, RTSP_PATH, RTSP_USER, RTSP_PASS)
#     return _open_ffmpeg_cap("Exit-Gate", ffmpeg_url)


# # =============================================================================
# #  GLOBAL SHARED FRAME
# # =============================================================================
# latest_frame = None
# frame_lock   = threading.Lock()
# frame_event  = threading.Event()
# running      = True


# def capture_frames():
#     global latest_frame, running

#     cap = _open_capture()
#     if not cap.isOpened():
#         print("ERROR: Could not open stream.")
#         running = False
#         return

#     while running:
#         ret, frame = cap.read()
#         if not ret:
#             print("Stream lost — reconnecting …")
#             cap.release()
#             time.sleep(2)
#             ffmpeg_url = _build_ffmpeg_url(RTSP_HOST, RTSP_PATH, RTSP_USER, RTSP_PASS)
#             cap = _open_ffmpeg_cap("Exit-Gate", ffmpeg_url)
#             continue

#         with frame_lock:
#             latest_frame = frame
#         frame_event.set()

#     cap.release()


# # =============================================================================
# #  MAIN
# # =============================================================================
# def main():
#     global running

#     db.init_db()
#     _rebuild_caches_from_db()

#     capture_thread = threading.Thread(target=capture_frames, daemon=True)
#     capture_thread.start()

#     if not frame_event.wait(timeout=8.0):
#         print("No frames received — exiting.")
#         running = False
#         return

#     frame_count            = 0
#     frame_time              = 1.0 / TARGET_FPS
#     last_mark_attempt_time  = {}
#     tracked_faces           = []
#     track_lock              = threading.Lock()

#     exit_overlay = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
#     rx1, ry1, rx2, ry2 = EXIT_REGION
#     cv2.rectangle(exit_overlay, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
#     cv2.putText(exit_overlay, "EXIT ZONE",
#                 (rx1 + 6, ry1 + 22),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

#     while running:
#         loop_start = time.time()

#         frame_event.wait(timeout=frame_time)
#         frame_event.clear()

#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame = latest_frame.copy()

#         frame_count  += 1
#         display_frame = frame

#         np.maximum(display_frame, exit_overlay, out=display_frame)

#         _refresh_status_cache_if_stale()

#         if frame_count % DETECTION_INTERVAL == 0:
#             faces       = app.get(frame)
#             new_tracked = []

#             for face in faces:
#                 x1, y1, x2, y2 = map(int, face.bbox)
#                 best_name, best_score = recognize_face(face.embedding)

#                 cx        = (x1 + x2) // 2
#                 cy        = (y1 + y2) // 2
#                 in_region = is_in_exit_region(cx, cy)

#                 label = ""
#                 color = (0, 0, 255)

#                 if best_score >= THRESHOLD:
#                     last_status = _cached_status(best_name)

#                     if last_status == "Exit":
#                         label = f"{best_name} - Exit Already Marked ({best_score:.2f})"
#                         color = (0, 255, 255)

#                     elif last_status != "Present":
#                         label = f"{best_name} - No Entry Found ({best_score:.2f})"
#                         color = (0, 165, 255)

#                     elif in_region:
#                         now      = time.time()
#                         last_try = last_mark_attempt_time.get(best_name, 0)
#                         if now - last_try > MARK_COOLDOWN_SECONDS:
#                             result = mark_exit(best_name, frame, (x1, y1, x2, y2), best_score)
#                             last_mark_attempt_time[best_name] = now
#                             if result == "marked":
#                                 label = f"{best_name} - EXIT MARKED ({best_score:.2f})"
#                             elif result == "already_exited":
#                                 label = f"{best_name} - Exit Already Marked ({best_score:.2f})"
#                             else:
#                                 label = f"{best_name} - No Entry Found ({best_score:.2f})"
#                         else:
#                             label = f"{best_name} - Cooldown ({best_score:.2f})"
#                         color = (0, 255, 0)

#                     else:
#                         label = f"{best_name} ({best_score:.2f}) - not in region"
#                         color = (255, 200, 0)

#                 else:
#                     label = f"Unknown ({best_score:.2f})"
#                     color = (0, 0, 255)

#                 new_tracked.append({
#                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                     "label": label, "color": color,
#                     "hold": TRACK_HOLD_FRAMES,
#                 })

#             with track_lock:
#                 tracked_faces = new_tracked

#         with track_lock:
#             current_faces = list(tracked_faces)
#             tracked_faces = [f for f in tracked_faces if f["hold"] > 0]
#             for f in tracked_faces:
#                 f["hold"] -= 1

#         for f in current_faces:
#             x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
#             color, label    = f["color"], f["label"]

#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#             label_y     = max(30, y1 - 10)
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#             cv2.rectangle(display_frame,
#                           (x1, label_y - th - 6),
#                           (x1 + tw + 6, label_y + 2),
#                           color, -1)
#             cv2.putText(display_frame, label,
#                         (x1 + 3, label_y - 2),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

#         with exit_cache_lock:
#             exit_snapshot = set(exit_cache)
#         summary = "Exited Today: " + (", ".join(sorted(exit_snapshot)) if exit_snapshot else "None")
#         cv2.putText(display_frame, summary,
#                     (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

#         # cv2.imshow("HR Face Attendance - EXIT GATE", display_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             running = False
#             break

#         elapsed    = time.time() - loop_start
#         sleep_time = frame_time - elapsed
#         if sleep_time > 0:
#             time.sleep(sleep_time)

#     cv2.destroyAllWindows()
#     time.sleep(1)
#     os._exit(0)


# if __name__ == "__main__":
#     main()


#------------------------------------------------NEW CODE---------------------------------------------------------
import cv2
import os
import time
import pickle
import threading
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from urllib.parse import quote

import config
import db

# =============================================================================
#  CONFIG  —  edit this section only
# =============================================================================
EMBEDDINGS_FILE = config.EMBEDDINGS_FILE
PHOTOS_DIR      = config.PHOTOS_DIR

RTSP_HOST  = "192.168.1.103:554"
RTSP_PATH  = "/video/live?channel=1&subtype=0"
RTSP_USER  = "admin"
RTSP_PASS  = "Cyamsys@123"

EXIT_CAMERA_ID = "EXIT MAIN"

FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720
THRESHOLD             = 0.50
DETECTION_INTERVAL    = 5
TRACK_HOLD_FRAMES     = 10
MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
TARGET_FPS            = 15

EXIT_REGION = (0, 252, 1200, 718)

# =============================================================================
#  VECTORIZED RECOGNITION MATRIX
# =============================================================================
_known_matrix = None
_known_names  = []

# =============================================================================
#  STATUS CACHE  —  db.py ke in-memory cache ka thin wrapper
#
#  SQLite-first architecture mein:
#    - _cached_status()     → db._get_effective_status() delegate karta hai
#    - _set_cached_status() → db._mem_set() delegate karta hai
#    - Date rollover        → db._mem_reset_if_new_day() handle karta hai
#    - Zero network call    → in-memory RAM se answer milta hai
# =============================================================================
def _refresh_status_cache_if_stale():
    """SQLite-first mein sirf date rollover check — no DB/network call."""
    db._mem_reset_if_new_day()


def _cached_status(name: str):
    """In-memory cache → SQLite fallback — zero network."""
    return db._get_effective_status(name)


def _set_cached_status(name: str, status: str):
    db._mem_set(name, status)


# =============================================================================
#  EXIT SUMMARY CACHE  —  "Exited Today" overlay ke liye
#  db.py ke present_cache ki tarah, exit_cache locally track karta hai
# =============================================================================
exit_cache      = set()
exit_cache_lock = threading.Lock()


def _rebuild_caches_from_db():
    """
    Startup pe SQLite se exit_cache seed karo.
    db.init_db() andar _mem_seed_from_local() pehle se call hoti hai,
    isliye yahan sirf exit_cache sync karna hai.
    """
    db._mem_reset_if_new_day()
    today = datetime.now().strftime("%Y-%m-%d")
    rows  = db._local_day_rows(today)
    seen  = {}
    for r in rows:   # newest-first — pehla = latest status per name
        seen.setdefault(r["name"], r["status"])
    with exit_cache_lock:
        exit_cache.clear()
        for name, status in seen.items():
            if status == "Exit":
                exit_cache.add(name)
    print(f"[cache] Exit cache seed: {len(exit_cache)} employee(s) already exited today.")


# =============================================================================
#  PHOTO CAPTURE
# =============================================================================
def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _save_capture(frame, bbox, name, cam_id):
    """Face + shoulders crop, 480x600 pad, JPEG save. Returns relative path."""
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = (x2 - x1), (y2 - y1)

        pad_x        = int(bw * 0.9)
        pad_y_top    = int(bh * 0.35)
        pad_y_bottom = int(bh * 1.8)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y_top)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y_bottom)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        TARGET_W, TARGET_H = 480, 600
        ch, cw = crop.shape[:2]
        target_ratio = TARGET_W / TARGET_H
        crop_ratio   = cw / ch

        if crop_ratio > target_ratio:
            need_h  = int(cw / target_ratio)
            extra   = max(0, need_h - ch)
            top_pad = extra // 2
            bot_pad = extra - top_pad
            crop = cv2.copyMakeBorder(crop, top_pad, bot_pad, 0, 0,
                                      cv2.BORDER_REPLICATE)
        else:
            need_w    = int(ch * target_ratio)
            extra     = max(0, need_w - cw)
            left_pad  = extra // 2
            right_pad = extra - left_pad
            crop = cv2.copyMakeBorder(crop, 0, 0, left_pad, right_pad,
                                      cv2.BORDER_REPLICATE)

        crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        crop = cv2.filter2D(crop, -1, sharpen_kernel)

        day_dir  = datetime.now().strftime("%Y-%m-%d")
        rel_dir  = os.path.join(_safe(name), day_dir)
        abs_dir  = os.path.join(PHOTOS_DIR, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        fname    = f"{cam_id}_{datetime.now().strftime('%H%M%S')}.jpg"
        rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
        abs_path = os.path.join(abs_dir, fname)

        cv2.imwrite(abs_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return rel_path
    except Exception as e:
        print(f"[photo] save failed for {name}: {e}")
        return None


# =============================================================================
#  LOAD EMBEDDINGS
# =============================================================================
print("Loading embeddings …")
with open(EMBEDDINGS_FILE, "rb") as f:
    known_embeddings = pickle.load(f)

known_face_db = {}
for name, emb_list in known_embeddings.items():
    emb_array = np.array(emb_list, dtype=np.float32)
    mean_emb  = np.mean(emb_array, axis=0)
    mean_emb  = mean_emb / np.linalg.norm(mean_emb)
    known_face_db[name] = mean_emb

_known_names  = list(known_face_db.keys())
_known_matrix = np.stack(list(known_face_db.values()), axis=0)  # (N, 512)
print(f"Total employees loaded: {len(known_face_db)}\n")

# =============================================================================
#  LOAD FACE MODEL
# =============================================================================
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(320, 320))


def recognize_face(live_embedding: np.ndarray):
    emb    = live_embedding / np.linalg.norm(live_embedding)
    scores = _known_matrix @ emb
    idx    = int(np.argmax(scores))
    return _known_names[idx], float(scores[idx])


def is_in_exit_region(cx, cy):
    x1, y1, x2, y2 = EXIT_REGION
    return x1 <= cx <= x2 and y1 <= cy <= y2


# =============================================================================
#  EXIT MARKING  —  SQLite-FIRST
#
#  FLOW:
#    1. Timestamp capture (network se pehle — hamesha)
#    2. In-memory cache check:
#         None    → "not_inside" (entry nahi tha aaj)
#         "Exit"  → "already_exited" (already exit ho gaya)
#         "Present" → aage badho
#    3. Photo save (disk pe — fast, no network)
#    4. db.mark_exit() → SQLite insert + in-memory cache update
#    5. exit_cache update → overlay pe turant dikhega
#    6. Background worker 5 min mein Supabase push karega
# =============================================================================
def mark_exit(name, frame, bbox, confidence):
    """Returns: 'marked' | 'already_exited' | 'not_inside' """
    last_status = _cached_status(name)

    if last_status is None:
        db.log_detection(name, EXIT_CAMERA_ID, confidence, "not_inside")
        return "not_inside"

    if last_status == "Exit":
        db.log_detection(name, EXIT_CAMERA_ID, confidence, "already_exited")
        return "already_exited"

    # last_status == "Present" → exit mark karo
    photo_path = _save_capture(frame, bbox, name, EXIT_CAMERA_ID)

    # db.mark_exit() → SQLite insert + in-memory cache update (Exit)
    result = db.mark_exit(
        name, EXIT_CAMERA_ID, confidence=confidence, photo_path=photo_path
    )

    db.log_detection(name, EXIT_CAMERA_ID, confidence, result)

    if result == "marked":
        # exit_cache update — overlay pe turant dikhega
        with exit_cache_lock:
            exit_cache.add(name)
        print(
            f"[EXIT MARKED] {name} at {datetime.now().strftime('%H:%M:%S')} "
            f"— SQLite saved, Supabase sync 5 min mein"
        )

    return result


# =============================================================================
#  FFMPEG / GSTREAMER HELPERS
# =============================================================================
def _build_ffmpeg_url(host, path, user, pw):
    safe_pw = quote(pw, safe="")
    return f"rtsp://{user}:{safe_pw}@{host}{path}"


def _open_ffmpeg_cap(label, ffmpeg_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000"
    )
    print(f"[{label}] Trying FFmpeg: {ffmpeg_url}")
    cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if cap.isOpened():
        print(f"[{label}] ✅ FFmpeg stream opened")
    else:
        print(f"[{label}] ❌ FFmpeg failed to open stream")
    return cap


def build_gst_pipeline(host, path, user, password, width, height):
    return (
        f'rtspsrc location="rtsp://{host}{path}" '
        f'user-id="{user}" user-pw="{password}" '
        f'latency=100 protocols=tcp ! '
        f'decodebin ! videoconvert ! videoscale ! '
        f'video/x-raw,width={width},height={height},format=BGR ! '
        f'appsink drop=true max-buffers=1 sync=false'
    )


def _open_capture():
    gst_pipeline = build_gst_pipeline(
        RTSP_HOST, RTSP_PATH, RTSP_USER, RTSP_PASS, FRAME_WIDTH, FRAME_HEIGHT
    )
    print(f"Trying GStreamer:\n  {gst_pipeline}")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ GStreamer confirmed (decodebin)")
                return cap
            time.sleep(0.05)
        cap.release()
        print("⚠️  GStreamer no frames — FFmpeg fallback")
    else:
        print("⚠️  GStreamer failed — FFmpeg fallback")

    ffmpeg_url = _build_ffmpeg_url(RTSP_HOST, RTSP_PATH, RTSP_USER, RTSP_PASS)
    return _open_ffmpeg_cap("Exit-Gate", ffmpeg_url)


# =============================================================================
#  GLOBAL SHARED FRAME
# =============================================================================
latest_frame = None
frame_lock   = threading.Lock()
frame_event  = threading.Event()
running      = True


def capture_frames():
    global latest_frame, running

    cap = _open_capture()
    if not cap.isOpened():
        print("ERROR: Could not open exit camera stream.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost — reconnecting …")
            cap.release()
            time.sleep(2)
            ffmpeg_url = _build_ffmpeg_url(RTSP_HOST, RTSP_PATH, RTSP_USER, RTSP_PASS)
            cap = _open_ffmpeg_cap("Exit-Gate", ffmpeg_url)
            continue
        with frame_lock:
            latest_frame = frame
        frame_event.set()

    cap.release()


# =============================================================================
#  MAIN
# =============================================================================
def main():
    global running

    db.init_db()              # SQLite init + in-memory cache seed + sync worker
    _rebuild_caches_from_db() # exit_cache seed from SQLite

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    if not frame_event.wait(timeout=8.0):
        print("No frames received — exiting.")
        running = False
        return

    frame_count            = 0
    frame_time             = 1.0 / TARGET_FPS
    last_mark_attempt_time = {}
    tracked_faces          = []
    track_lock             = threading.Lock()

    # Exit zone overlay — ek baar banao, har frame pe blend karo
    exit_overlay = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    rx1, ry1, rx2, ry2 = EXIT_REGION
    cv2.rectangle(exit_overlay, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
    cv2.putText(exit_overlay, "EXIT ZONE",
                (rx1 + 6, ry1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    while running:
        loop_start = time.time()

        frame_event.wait(timeout=frame_time)
        frame_event.clear()

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_count   += 1
        display_frame  = frame

        # Exit zone overlay blend
        np.maximum(display_frame, exit_overlay, out=display_frame)

        # Date rollover check
        _refresh_status_cache_if_stale()

        # ── Periodic face detection ───────────────────────────────────────────
        if frame_count % DETECTION_INTERVAL == 0:
            faces       = face_app.get(frame)
            new_tracked = []

            for face in faces:
                x1, y1, x2, y2       = map(int, face.bbox)
                best_name, best_score = recognize_face(face.embedding)

                cx        = (x1 + x2) // 2
                cy        = (y1 + y2) // 2
                in_region = is_in_exit_region(cx, cy)

                label = ""
                color = (0, 0, 255)

                if best_score >= THRESHOLD:
                    last_status = _cached_status(best_name)

                    if last_status == "Exit":
                        label = f"{best_name} - Exit Already Marked ({best_score:.2f})"
                        color = (0, 255, 255)    # yellow

                    elif last_status is None:
                        label = f"{best_name} - No Entry Found ({best_score:.2f})"
                        color = (0, 165, 255)    # orange

                    elif in_region:
                        # last_status == "Present" + in exit zone → mark exit
                        now      = time.time()
                        last_try = last_mark_attempt_time.get(best_name, 0)

                        if now - last_try > MARK_COOLDOWN_SECONDS:
                            result = mark_exit(
                                best_name, frame, (x1, y1, x2, y2), best_score
                            )
                            last_mark_attempt_time[best_name] = now

                            if result == "marked":
                                label = (
                                    f"{best_name} - EXIT MARKED "
                                    f"({best_score:.2f}) [SQLite✓]"
                                )
                                color = (0, 255, 0)    # green
                            elif result == "already_exited":
                                label = f"{best_name} - Exit Already Marked ({best_score:.2f})"
                                color = (0, 255, 255)
                            else:
                                label = f"{best_name} - No Entry Found ({best_score:.2f})"
                                color = (0, 165, 255)
                        else:
                            label = f"{best_name} - Cooldown ({best_score:.2f})"
                            color = (255, 200, 0)   # amber

                    else:
                        # Present hai but exit zone mein nahi
                        label = f"{best_name} ({best_score:.2f}) - not in exit zone"
                        color = (255, 200, 0)

                else:
                    label = f"Unknown ({best_score:.2f})"
                    color = (0, 0, 255)

                new_tracked.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label, "color": color,
                    "hold": TRACK_HOLD_FRAMES,
                })

            with track_lock:
                tracked_faces = new_tracked

        # ── Draw tracked boxes ────────────────────────────────────────────────
        with track_lock:
            current_faces = list(tracked_faces)
            tracked_faces = [f for f in tracked_faces if f["hold"] > 0]
            for f in tracked_faces:
                f["hold"] -= 1

        for f in current_faces:
            x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
            color, label    = f["color"], f["label"]

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label_y     = max(30, y1 - 10)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(
                display_frame,
                (x1, label_y - th - 6),
                (x1 + tw + 6, label_y + 2),
                color, -1,
            )
            cv2.putText(
                display_frame, label,
                (x1 + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
            )

        # ── Summary overlays ──────────────────────────────────────────────────
        # Exited Today
        with exit_cache_lock:
            exit_snapshot = set(exit_cache)
        summary = "Exited Today: " + (
            ", ".join(sorted(exit_snapshot)) if exit_snapshot else "None"
        )
        cv2.putText(
            display_frame, summary,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2,
        )

        # Pending sync count — SQLite mein kitne rows Supabase ka wait kar rahe hain
        pending_n = db.pending_sync_count()
        if pending_n > 0:
            cv2.putText(
                display_frame,
                f"Pending sync: {pending_n}",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2,
            )

        # Camera ID top-right
        cv2.putText(
            display_frame, EXIT_CAMERA_ID,
            (FRAME_WIDTH - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )

        # Headless (systemd service) mode ke liye commented out — no display
        # wahan attached hota. Manual debug run ke liye uncomment karo.
        # cv2.imshow("HR Face Attendance - EXIT GATE", display_frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     running = False
        #     break

        elapsed    = time.time() - loop_start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # cv2.destroyAllWindows()
    time.sleep(1)
    os._exit(0)


if __name__ == "__main__":
    main()