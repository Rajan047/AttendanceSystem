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

# RTSP_CAMERAS = [
#     {
#         "id":      "Entry-1",
#         "url":     "rtsp://192.168.1.101:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-2",
#         "url":     "rtsp://192.168.1.104:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-3",
#         "url":     "rtsp://192.168.1.107:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-4",
#         "url":     "rtsp://192.168.1.106:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
# ]

# FRAME_WIDTH           = 1280
# FRAME_HEIGHT          = 720
# THRESHOLD             = 0.50
# DETECTION_INTERVAL    = 10    # detect every 10 frames (cuts inference cost in half)
# TRACK_HOLD_FRAMES     = 20    # hold boxes longer to match new interval
# MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
# TARGET_FPS            = 10    # 10 fps is smooth enough for a door camera

# # =============================================================================
# #  PRE-BUILT NUMPY MATRIX for vectorized recognition (replaces Python for-loop)
# # =============================================================================
# _known_matrix = None   # shape (N, 512), float32, each row already L2-normalised
# _known_names  = []     # index → name, parallel to _known_matrix rows

# # =============================================================================
# #  STATUS CACHE — replaces a MySQL read on every detected face every N frames.
# #  { name: "Present" | "Exit" | None }  updated only when a write happens
# #  or once per STATUS_CACHE_TTL seconds (cheap periodic refresh).
# #
# #  DATE-ROLLOVER FIX: this service runs 24x7 under systemd (Restart=always),
# #  so without tracking the current date, a "Present" status from yesterday
# #  would silently stay cached across midnight and block today's first entry
# #  ("already_present" false-positive). _cache_date tracks the day this cache
# #  was last built for; any refresh that notices the date changed does a full
# #  reset instead of a normal TTL-based refresh.
# # =============================================================================
# STATUS_CACHE_TTL   = config.STATUS_CACHE_TTL
# _status_cache      = {}
# _status_cache_lock = threading.Lock()
# _status_cache_ts   = 0.0
# _cache_date        = None   # 'YYYY-MM-DD' — day the cache currently reflects

# def _refresh_status_cache_if_stale():
#     """DB se aaj ke saare statuses ek baar padho aur cache update karo.
#        Agar date badal gayi hai (raat 12 baj gayi), cache ko poora reset
#        karo — warna kal ka 'Present' status aaj bhi atka reh jaayega."""
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
#     # rows already newest-first per query; keep the first (latest) status per name
#     for r in today_rows:
#         tmp.setdefault(r["name"], r["status"])

#     with _status_cache_lock:
#         _status_cache.clear()
#         _status_cache.update(tmp)

#     if day_changed:
#         with present_cache_lock:
#             present_cache.clear()
#             for name, status in tmp.items():
#                 if status == "Present":
#                     present_cache.add(name)
#         print(f"[cache] Naya din shuru — {today_str} ke liye status cache reset ho gaya.")

# def _cached_status(name: str):
#     with _status_cache_lock:
#         return _status_cache.get(name, None)

# def _set_cached_status(name: str, status: str):
#     with _status_cache_lock:
#         _status_cache[name] = status

# # =============================================================================
# #  PRESENT CACHE  — avoids a DB read every frame for the summary overlay
# # =============================================================================
# present_cache      = set()
# present_cache_lock = threading.Lock()

# def _rebuild_cache_from_db():
#     """Startup pe DB se aaj ka status ek baar padho, dono caches seed karo."""
#     global _cache_date
#     _cache_date = datetime.now().strftime("%Y-%m-%d")
#     today_rows  = db.read_day_rows(_cache_date)
#     last_status = {}
#     for r in today_rows:
#         last_status.setdefault(r["name"], r["status"])
#     with present_cache_lock:
#         present_cache.clear()
#         for name, status in last_status.items():
#             if status == "Present":
#                 present_cache.add(name)
#     with _status_cache_lock:
#         _status_cache.update(last_status)

# # =============================================================================
# #  PHOTO CAPTURE  — cropped face, saved to disk, path returned for DB insert
# # =============================================================================
# def _safe(name: str) -> str:
#     """Filesystem-safe folder name."""
#     return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# def _save_capture(frame, bbox, name, cam_id):
#     """Bbox ke around padding le kar face + shoulders wala crop banao,
#        fixed-size photo (480x600) me PAD karke (crop nahi, taaki content
#        kata na jaaye) resize karo, aur high-quality JPEG save karo.
#        Returns path RELATIVE to PHOTOS_DIR (yahi DB me store hota hai)."""
#     try:
#         h, w = frame.shape[:2]
#         x1, y1, x2, y2 = bbox
#         bw, bh = (x2 - x1), (y2 - y1)

#         pad_x        = int(bw * 0.9)
#         pad_y_top    = int(bh * 0.35)   # sirf thoda sa upar — poora forehead+hair
#         pad_y_bottom = int(bh * 1.8)    # neeche bahut zyada — chin, neck, shoulders

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
#             need_w   = int(ch * target_ratio)
#             extra    = max(0, need_w - cw)
#             left_pad = extra // 2
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
# #  FFMPEG URL HELPER
# # =============================================================================

# def _build_ffmpeg_url(url: str, user_id: str, user_pw: str) -> str:
#     prefix = "rtsp://"
#     body   = url[len(prefix):]
#     if "@" in body.split("/")[0]:
#         return url
#     safe_pw = quote(user_pw, safe="")
#     return f"{prefix}{user_id}:{safe_pw}@{body}"


# def _open_ffmpeg_cap(cam_id: str, ffmpeg_url: str) -> cv2.VideoCapture:
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#         "rtsp_transport;t
# |stimeout;5000000"
#     )
#     print(f"[{cam_id}] Trying FFmpeg: {ffmpeg_url}")
#     cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
#     if cap.isOpened():
#         print(f"[{cam_id}] ✅ FFmpeg stream opened")
#     else:
#         print(f"[{cam_id}] ❌ FFmpeg also failed to open stream")
#     return cap

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
#     print(f"  {name}: {len(emb_list)} embeddings")

# print(f"Total employees loaded: {len(known_face_db)}\n")

# _known_names  = list(known_face_db.keys())
# _known_matrix = np.stack(list(known_face_db.values()), axis=0)  # (N, 512)

# # =============================================================================
# #  LOAD FACE MODEL
# # =============================================================================
# face_app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CUDAExecutionProvider"],
# )
# face_app.prepare(ctx_id=0, det_size=(320, 320))

# # =============================================================================
# #  VECTORIZED RECOGNITION  — single matmul instead of Python for-loop
# # =============================================================================
# def recognize_face(live_embedding: np.ndarray):
#     emb    = live_embedding / np.linalg.norm(live_embedding)
#     scores = _known_matrix @ emb
#     idx    = int(np.argmax(scores))
#     return _known_names[idx], float(scores[idx])

# # =============================================================================
# #  ATTENDANCE MARKING  — MySQL-backed, with duplicate prevention + photo save
# #  Every recognition attempt is now logged via db.log_detection(), regardless
# #  of outcome, so you have an exact independent timestamp for EVERY event —
# #  including "already_present" hits that never reach db.mark_attendance().
# # =============================================================================
# def mark_attendance(name, camera_id, frame, bbox, confidence):
#     last_status = _cached_status(name)

#     if last_status == "Present":
#         db.log_detection(name, camera_id, confidence, "already_present")
#         return "already_present"

#     photo_path = _save_capture(frame, bbox, name, camera_id)

#     result = db.mark_attendance(
#         name, camera_id, confidence=confidence, photo_path=photo_path
#     )

#     db.log_detection(name, camera_id, confidence, result)

#     if result == "marked":
#         _set_cached_status(name, "Present")
#         with present_cache_lock:
#             present_cache.add(name)
#         action = "RE-ENTRY" if last_status == "Exit" else "FIRST ENTRY"
#         print(f"[{action}] {name} via {camera_id} at "
#               f"{datetime.now().strftime('%H:%M:%S')}")

#     return result

# # =============================================================================
# #  GSTREAMER PIPELINE BUILDER
# # =============================================================================
# def build_gst_pipeline(url, user_id, user_pw, width, height):
#     return (
#         f'rtspsrc location="{url}" '
#         f'user-id="{user_id}" user-pw="{user_pw}" '
#         f'latency=100 protocols=tcp ! '
#         f'decodebin ! '
#         f'videoconvert ! '
#         f'videoscale ! '
#         f'video/x-raw,width={width},height={height},format=BGR ! '
#         f'appsink drop=true max-buffers=1 sync=false'
#     )

# # =============================================================================
# #  OPEN CAPTURE — must always be called from the MAIN THREAD
# # =============================================================================
# def open_capture_main_thread(cam_id, url, user_id, user_pw):
#     gst_pipeline = build_gst_pipeline(url, user_id, user_pw, FRAME_WIDTH, FRAME_HEIGHT)
#     print(f"[{cam_id}] Trying GStreamer pipeline:\n  {gst_pipeline}")
#     cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

#     if cap.isOpened():
#         deadline = time.time() + 5.0
#         while time.time() < deadline:
#             ret, frame = cap.read()
#             if ret and frame is not None:
#                 print(f"[{cam_id}] ✅ GStreamer pipeline confirmed (GPU decoding via decodebin)")
#                 return cap, frame
#             time.sleep(0.05)
#         cap.release()
#         print(f"[{cam_id}] ⚠️  GStreamer opened but produced no frames — falling back to FFmpeg")
#     else:
#         print(f"[{cam_id}] ⚠️  GStreamer failed to open — falling back to FFmpeg")

#     ffmpeg_url = _build_ffmpeg_url(url, user_id, user_pw)
#     cap = _open_ffmpeg_cap(cam_id, ffmpeg_url)
#     return cap, None

# # =============================================================================
# #  PER-CAMERA WORKER
# # =============================================================================
# class CameraWorker:

#     def __init__(self, camera_cfg, shared_running_flag):
#         self.cam_id  = camera_cfg["id"]
#         self.url     = camera_cfg["url"]
#         self.user_id = camera_cfg["user_id"]
#         self.user_pw = camera_cfg["user_pw"]
#         self.running = shared_running_flag

#         self.latest_frame = None
#         self.frame_lock   = threading.Lock()
#         self.frame_event  = threading.Event()

#         self.last_mark_attempt = {}
#         self.tracked_faces     = []
#         self.track_lock        = threading.Lock()

#         self._cap      = None
#         self._cap_lock = threading.Lock()

#     def open(self):
#         cap, first_frame = open_capture_main_thread(
#             self.cam_id, self.url, self.user_id, self.user_pw
#         )
#         self._cap = cap
#         if first_frame is not None:
#             with self.frame_lock:
#                 self.latest_frame = first_frame.copy()
#             self.frame_event.set()

#     def _reopen(self):
#         print(f"[{self.cam_id}] Reconnecting via FFmpeg …")
#         ffmpeg_url = _build_ffmpeg_url(self.url, self.user_id, self.user_pw)
#         return _open_ffmpeg_cap(self.cam_id, ffmpeg_url)

#     # ── capture thread ────────────────────────────────────────────────────────
#     def _capture_loop(self):
#         cap = self._cap

#         if not cap.isOpened():
#             print(f"[{self.cam_id}] ERROR: Could not open stream — check IP, credentials, and network.")
#             self.running["value"] = False
#             return

#         while self.running["value"]:
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[{self.cam_id}] Stream lost — reconnecting …")
#                 cap.release()
#                 time.sleep(2)
#                 cap = self._reopen()
#                 continue
#             with self.frame_lock:
#                 self.latest_frame = frame
#             self.frame_event.set()

#         cap.release()

#     # ── main processing loop ──────────────────────────────────────────────────
#     def run(self):
#         t = threading.Thread(target=self._capture_loop, daemon=True)
#         t.start()

#         frame_count = 0
#         window_name = f"Entry Gate — {self.cam_id}"
#         frame_time  = 1.0 / TARGET_FPS

#         if not self.frame_event.wait(timeout=8.0):
#             print(f"[{self.cam_id}] No frames received — skipping display loop.")
#             self.running["value"] = False
#             return

#         while self.running["value"]:
#             loop_start = time.time()

#             self.frame_event.wait(timeout=frame_time)
#             self.frame_event.clear()

#             with self.frame_lock:
#                 if self.latest_frame is None:
#                     continue
#                 frame = self.latest_frame.copy()

#             frame_count  += 1
#             display_frame = frame

#             _refresh_status_cache_if_stale()

#             # ── periodic face detection ──────────────────────────────────────
#             if frame_count % DETECTION_INTERVAL == 0:
#                 faces       = face_app.get(frame)
#                 new_tracked = []

#                 for face in faces:
#                     x1, y1, x2, y2       = map(int, face.bbox)
#                     best_name, best_score = recognize_face(face.embedding)

#                     if best_score >= THRESHOLD:
#                         now      = time.time()
#                         last_try = self.last_mark_attempt.get(best_name, 0)

#                         if now - last_try > MARK_COOLDOWN_SECONDS:
#                             result = mark_attendance(
#                                 best_name, self.cam_id, frame,
#                                 (x1, y1, x2, y2), best_score,
#                             )
#                             self.last_mark_attempt[best_name] = now
#                         else:
#                             result = "cooldown"

#                         if result == "marked":
#                             rows_today = db.count_today_rows(best_name)
#                             label = f"{best_name} - {'Re-Entry' if rows_today > 1 else 'Present'} Marked ({best_score:.2f})"
#                             color = (0, 255, 0)
#                         elif result == "already_present":
#                             label = f"{best_name} - Already Present ({best_score:.2f})"
#                             color = (0, 255, 255)
#                         else:
#                             label = f"{best_name} ({best_score:.2f})"
#                             color = (255, 200, 0)
#                     else:
#                         label = f"Unknown ({best_score:.2f})"
#                         color = (0, 0, 255)

#                     new_tracked.append({
#                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                         "label": label, "color": color,
#                         "hold": TRACK_HOLD_FRAMES,
#                     })

#                 with self.track_lock:
#                     self.tracked_faces = new_tracked

#             # ── draw tracked boxes ───────────────────────────────────────────
#             with self.track_lock:
#                 current_faces      = list(self.tracked_faces)
#                 self.tracked_faces = [f for f in self.tracked_faces if f["hold"] > 0]
#                 for f in self.tracked_faces:
#                     f["hold"] -= 1

#             for f in current_faces:
#                 x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
#                 color, label    = f["color"], f["label"]

#                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#                 label_y     = max(30, y1 - 10)
#                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#                 cv2.rectangle(display_frame,
#                               (x1, label_y - th - 6),
#                               (x1 + tw + 6, label_y + 2),
#                               color, -1)
#                 cv2.putText(display_frame, label,
#                             (x1 + 3, label_y - 2),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

#             # ── summary overlay — reads from cache, zero DB I/O ──────────────
#             with present_cache_lock:
#                 present_snapshot = set(present_cache)
#             summary = "Inside Now: " + (", ".join(sorted(present_snapshot)) if present_snapshot else "None")
#             cv2.putText(display_frame, summary,
#                         (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

#             cv2.putText(display_frame, self.cam_id,
#                         (FRAME_WIDTH - 140, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

#             cv2.imshow(window_name, display_frame)

#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 self.running["value"] = False
#                 break

#             elapsed    = time.time() - loop_start
#             sleep_time = frame_time - elapsed
#             if sleep_time > 0:
#                 time.sleep(sleep_time)

#         cv2.destroyWindow(window_name)

# # =============================================================================
# #  MAIN
# # =============================================================================
# def main():
#     db.init_db()
#     db.sync_roster(_known_names)   # embeddings.pkl ke saare naam roster me ensure karo
#     n = db.sync_employee_photos_from_dir()
#     if n:
#         print(f"[photos] {n} employee profile photo(s) linked from {PHOTOS_DIR}")
#     _rebuild_cache_from_db()

#     running  = {"value": True}
#     workers  = [CameraWorker(cfg, running) for cfg in RTSP_CAMERAS]

#     for worker in workers:
#         worker.open()

#     threads = []
#     for worker in workers[:-1]:
#         t = threading.Thread(target=worker.run, daemon=True)
#         t.start()
#         threads.append(t)

#     workers[-1].run()

#     for t in threads:
#         t.join()

#     cv2.destroyAllWindows()
#     print("All cameras stopped.")

#     time.sleep(1)
#     os._exit(0)


# if __name__ == "__main__":
#     main()


#-----------------------------------------------------------NEW CODE---------------------------------------------------------------------
# import cv2
# import os
# import time
# import pickle
# import threading
# import numpy as np
# from datetime import datetime
# from urllib.parse import quote

# import config
# import db

# # =============================================================================
# #  CONFIG
# # =============================================================================
# EMBEDDINGS_FILE = config.EMBEDDINGS_FILE
# PHOTOS_DIR      = config.PHOTOS_DIR

# RTSP_CAMERAS = [
#     {
#         "id":      "Entry-1",
#         "url":     "rtsp://192.168.1.101:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-2",
#         "url":     "rtsp://192.168.1.104:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-3",
#         "url":     "rtsp://192.168.1.107:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-4",
#         "url":     "rtsp://192.168.1.106:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
# ]

# FRAME_WIDTH           = 1280
# FRAME_HEIGHT          = 720
# THRESHOLD             = 0.50
# DETECTION_INTERVAL    = 10      # har 10 frames pe face detect
# TRACK_HOLD_FRAMES     = 20      # box itne frames tak dikhta rahe
# MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
# TARGET_FPS            = 10

# # =============================================================================
# #  VECTORIZED RECOGNITION MATRIX
# # =============================================================================
# _known_matrix = None   # (N, 512) float32, L2-normalized
# _known_names  = []

# # =============================================================================
# #  STATUS CACHE  —  db.py ke in-memory cache ka thin wrapper
# #  (SQLite-first architecture mein yeh sirf db._mem_cache ko delegate karta hai)
# #
# #  DATE-ROLLOVER: db.py ke andar _mem_reset_if_new_day() handle karta hai.
# # =============================================================================
# def _refresh_status_cache_if_stale():
#     """
#     SQLite-first mein yeh function sirf date rollover check karta hai.
#     Actual status hamesha db._get_effective_status() se aata hai jo
#     in-memory cache → SQLite fallback karta hai — zero network call.
#     """
#     db._mem_reset_if_new_day()


# def _cached_status(name: str):
#     """In-memory cache se status lo — zero DB/network call."""
#     return db._get_effective_status(name)


# def _set_cached_status(name: str, status: str):
#     """In-memory cache update karo."""
#     db._mem_set(name, status)


# # present_cache directly db se use karo
# present_cache      = db.present_cache
# present_cache_lock = db.present_cache_lock


# def _rebuild_cache_from_db():
#     """
#     Startup pe SQLite se cache seed karo.
#     db.init_db() ke andar _mem_seed_from_local() already call hoti hai,
#     isliye yeh function sirf present_cache sync karta hai.
#     """
#     db._sync_present_cache_from_mem()


# # =============================================================================
# #  PHOTO CAPTURE
# # =============================================================================
# def _safe(name: str) -> str:
#     return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# def _save_capture(frame, bbox, name, cam_id):
#     """
#     Face + shoulders ka crop banao, 480x600 mein pad karke save karo.
#     Returns path relative to PHOTOS_DIR.
#     """
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
#         crop_ratio   = cw / ch

#         if crop_ratio > target_ratio:
#             need_h  = int(cw / target_ratio)
#             extra   = max(0, need_h - ch)
#             top_pad = extra // 2
#             bot_pad = extra - top_pad
#             crop = cv2.copyMakeBorder(crop, top_pad, bot_pad, 0, 0,
#                                       cv2.BORDER_REPLICATE)
#         else:
#             need_w    = int(ch * target_ratio)
#             extra     = max(0, need_w - cw)
#             left_pad  = extra // 2
#             right_pad = extra - left_pad
#             crop = cv2.copyMakeBorder(crop, 0, 0, left_pad, right_pad,
#                                       cv2.BORDER_REPLICATE)

#         crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

#         sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         crop = cv2.filter2D(crop, -1, sharpen_kernel)

#         day_dir  = datetime.now().strftime("%Y-%m-%d")
#         rel_dir  = os.path.join(_safe(name), day_dir)
#         abs_dir  = os.path.join(PHOTOS_DIR, rel_dir)
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
# #  FFMPEG / GSTREAMER HELPERS
# # =============================================================================
# def _build_ffmpeg_url(url: str, user_id: str, user_pw: str) -> str:
#     prefix = "rtsp://"
#     body   = url[len(prefix):]
#     if "@" in body.split("/")[0]:
#         return url
#     safe_pw = quote(user_pw, safe="")
#     return f"{prefix}{user_id}:{safe_pw}@{body}"


# def _open_ffmpeg_cap(cam_id: str, ffmpeg_url: str) -> cv2.VideoCapture:
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#         "rtsp_transport;tcp|stimeout;5000000"
#     )
#     print(f"[{cam_id}] Trying FFmpeg: {ffmpeg_url}")
#     cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
#     if cap.isOpened():
#         print(f"[{cam_id}] ✅ FFmpeg stream opened")
#     else:
#         print(f"[{cam_id}] ❌ FFmpeg failed to open stream")
#     return cap


# def build_gst_pipeline(url, user_id, user_pw, width, height):
#     return (
#         f'rtspsrc location="{url}" '
#         f'user-id="{user_id}" user-pw="{user_pw}" '
#         f'latency=100 protocols=tcp ! '
#         f'decodebin ! videoconvert ! videoscale ! '
#         f'video/x-raw,width={width},height={height},format=BGR ! '
#         f'appsink drop=true max-buffers=1 sync=false'
#     )


# def open_capture_main_thread(cam_id, url, user_id, user_pw):
#     gst_pipeline = build_gst_pipeline(url, user_id, user_pw, FRAME_WIDTH, FRAME_HEIGHT)
#     print(f"[{cam_id}] Trying GStreamer:\n  {gst_pipeline}")
#     cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

#     if cap.isOpened():
#         deadline = time.time() + 5.0
#         while time.time() < deadline:
#             ret, frame = cap.read()
#             if ret and frame is not None:
#                 print(f"[{cam_id}] ✅ GStreamer confirmed (GPU decoding)")
#                 return cap, frame
#             time.sleep(0.05)
#         cap.release()
#         print(f"[{cam_id}] ⚠️  GStreamer opened but no frames — FFmpeg fallback")
#     else:
#         print(f"[{cam_id}] ⚠️  GStreamer failed — FFmpeg fallback")

#     ffmpeg_url = _build_ffmpeg_url(url, user_id, user_pw)
#     cap = _open_ffmpeg_cap(cam_id, ffmpeg_url)
#     return cap, None


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
#     print(f"  {name}: {len(emb_list)} embeddings")

# print(f"Total employees loaded: {len(known_face_db)}\n")

# _known_names  = list(known_face_db.keys())
# _known_matrix = np.stack(list(known_face_db.values()), axis=0)   # (N, 512)

# # =============================================================================
# #  LOAD FACE MODEL
# # =============================================================================
# from insightface.app import FaceAnalysis
# face_app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CUDAExecutionProvider"],
# )
# face_app.prepare(ctx_id=0, det_size=(320, 320))


# # =============================================================================
# #  VECTORIZED RECOGNITION  —  single matmul, no Python loop
# # =============================================================================
# def recognize_face(live_embedding: np.ndarray):
#     emb    = live_embedding / np.linalg.norm(live_embedding)
#     scores = _known_matrix @ emb
#     idx    = int(np.argmax(scores))
#     return _known_names[idx], float(scores[idx])


# # =============================================================================
# #  ATTENDANCE MARKING
# #
# #  SQLite-first flow:
# #    1. In-memory cache check → already Present? return immediately
# #    2. Photo save karo (disk pe — fast, no network)
# #    3. db.mark_attendance() → SQLite insert + cache update
# #    4. Camera pe turant "Present Marked" dikhega ✅
# #    5. db.py ka background worker 5 min mein Supabase push karega
# # =============================================================================
# def mark_attendance(name, camera_id, frame, bbox, confidence):
#     last_status = _cached_status(name)

#     if last_status == "Present":
#         db.log_detection(name, camera_id, confidence, "already_present")
#         return "already_present"

#     # Photo save (fast — disk only, no network)
#     photo_path = _save_capture(frame, bbox, name, camera_id)

#     # db.mark_attendance() → SQLite + in-memory cache update
#     result = db.mark_attendance(
#         name, camera_id, confidence=confidence, photo_path=photo_path
#     )

#     db.log_detection(name, camera_id, confidence, result)

#     if result == "marked":
#         action = "RE-ENTRY" if last_status == "Exit" else "FIRST ENTRY"
#         print(
#             f"[{action}] {name} via {camera_id} at "
#             f"{datetime.now().strftime('%H:%M:%S')} "
#             f"— SQLite saved, Supabase sync 5 min mein"
#         )

#     return result


# # =============================================================================
# #  PER-CAMERA WORKER
# # =============================================================================
# class CameraWorker:

#     def __init__(self, camera_cfg, shared_running_flag):
#         self.cam_id  = camera_cfg["id"]
#         self.url     = camera_cfg["url"]
#         self.user_id = camera_cfg["user_id"]
#         self.user_pw = camera_cfg["user_pw"]
#         self.running = shared_running_flag

#         self.latest_frame = None
#         self.frame_lock   = threading.Lock()
#         self.frame_event  = threading.Event()

#         self.last_mark_attempt = {}   # { name: timestamp }
#         self.tracked_faces     = []
#         self.track_lock        = threading.Lock()

#         self._cap = None

#     def open(self):
#         cap, first_frame = open_capture_main_thread(
#             self.cam_id, self.url, self.user_id, self.user_pw
#         )
#         self._cap = cap
#         if first_frame is not None:
#             with self.frame_lock:
#                 self.latest_frame = first_frame.copy()
#             self.frame_event.set()

#     def _reopen(self):
#         print(f"[{self.cam_id}] Reconnecting via FFmpeg …")
#         ffmpeg_url = _build_ffmpeg_url(self.url, self.user_id, self.user_pw)
#         return _open_ffmpeg_cap(self.cam_id, ffmpeg_url)

#     # ── Capture thread ────────────────────────────────────────────────────────
#     def _capture_loop(self):
#         cap = self._cap

#         if not cap.isOpened():
#             print(
#                 f"[{self.cam_id}] ERROR: stream open nahi hua — "
#                 f"IP/credentials/network check karo."
#             )
#             self.running["value"] = False
#             return

#         while self.running["value"]:
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[{self.cam_id}] Stream lost — reconnecting …")
#                 cap.release()
#                 time.sleep(2)
#                 cap = self._reopen()
#                 continue
#             with self.frame_lock:
#                 self.latest_frame = frame
#             self.frame_event.set()

#         cap.release()

#     # ── Main processing loop ──────────────────────────────────────────────────
#     def run(self):
#         t = threading.Thread(target=self._capture_loop, daemon=True)
#         t.start()

#         frame_count = 0
#         window_name = f"Entry Gate — {self.cam_id}"
#         frame_time  = 1.0 / TARGET_FPS

#         if not self.frame_event.wait(timeout=8.0):
#             print(f"[{self.cam_id}] No frames received — skipping.")
#             self.running["value"] = False
#             return

#         while self.running["value"]:
#             loop_start = time.time()

#             self.frame_event.wait(timeout=frame_time)
#             self.frame_event.clear()

#             with self.frame_lock:
#                 if self.latest_frame is None:
#                     continue
#                 frame = self.latest_frame.copy()

#             frame_count   += 1
#             display_frame  = frame

#             # Date rollover check (midnight ke baad cache reset)
#             _refresh_status_cache_if_stale()

#             # ── Periodic face detection ───────────────────────────────────────
#             if frame_count % DETECTION_INTERVAL == 0:
#                 faces       = face_app.get(frame)
#                 new_tracked = []

#                 for face in faces:
#                     x1, y1, x2, y2       = map(int, face.bbox)
#                     best_name, best_score = recognize_face(face.embedding)

#                     if best_score >= THRESHOLD:
#                         now      = time.time()
#                         last_try = self.last_mark_attempt.get(best_name, 0)

#                         if now - last_try > MARK_COOLDOWN_SECONDS:
#                             result = mark_attendance(
#                                 best_name, self.cam_id, frame,
#                                 (x1, y1, x2, y2), best_score,
#                             )
#                             self.last_mark_attempt[best_name] = now
#                         else:
#                             result = "cooldown"

#                         # ── Display label logic ───────────────────────────────
#                         if result == "marked":
#                             rows_today = db.count_today_rows(best_name)
#                             label = (
#                                 f"{best_name} - "
#                                 f"{'Re-Entry' if rows_today > 1 else 'Present'} "
#                                 f"({best_score:.2f}) [SQLite✓]"
#                             )
#                             color = (0, 255, 0)       # green

#                         elif result == "already_present":
#                             label = f"{best_name} - Already Present ({best_score:.2f})"
#                             color = (0, 255, 255)     # yellow

#                         elif result == "cooldown":
#                             label = f"{best_name} ({best_score:.2f}) [cooldown]"
#                             color = (255, 200, 0)     # orange

#                         else:
#                             label = f"{best_name} ({best_score:.2f})"
#                             color = (255, 200, 0)

#                     else:
#                         label = f"Unknown ({best_score:.2f})"
#                         color = (0, 0, 255)           # red

#                     new_tracked.append({
#                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                         "label": label, "color": color,
#                         "hold": TRACK_HOLD_FRAMES,
#                     })

#                 with self.track_lock:
#                     self.tracked_faces = new_tracked

#             # ── Draw tracked boxes ────────────────────────────────────────────
#             with self.track_lock:
#                 current_faces      = list(self.tracked_faces)
#                 self.tracked_faces = [f for f in self.tracked_faces if f["hold"] > 0]
#                 for f in self.tracked_faces:
#                     f["hold"] -= 1

#             for f in current_faces:
#                 x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
#                 color, label    = f["color"], f["label"]

#                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#                 label_y     = max(30, y1 - 10)
#                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#                 cv2.rectangle(
#                     display_frame,
#                     (x1, label_y - th - 6),
#                     (x1 + tw + 6, label_y + 2),
#                     color, -1,
#                 )
#                 cv2.putText(
#                     display_frame, label,
#                     (x1 + 3, label_y - 2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
#                 )

#             # ── Summary overlay (in-memory cache — zero DB/network) ───────────
#             with present_cache_lock:
#                 present_snapshot = set(present_cache)

#             summary = "Inside Now: " + (
#                 ", ".join(sorted(present_snapshot)) if present_snapshot else "None"
#             )
#             cv2.putText(
#                 display_frame, summary,
#                 (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2,
#             )

#             # Pending sync count overlay — SQLite me kitne baaki hain
#             pending_n = db.pending_sync_count()
#             if pending_n > 0:
#                 sync_text = f"Pending sync: {pending_n}"
#                 cv2.putText(
#                     display_frame, sync_text,
#                     (20, 58),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2,
#                 )

#             cv2.putText(
#                 display_frame, self.cam_id,
#                 (FRAME_WIDTH - 140, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
#             )

#             # cv2.imshow(window_name, display_frame)

#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 self.running["value"] = False
#                 break

#             elapsed    = time.time() - loop_start
#             sleep_time = frame_time - elapsed
#             if sleep_time > 0:
#                 time.sleep(sleep_time)

#         cv2.destroyWindow(window_name)


# # =============================================================================
# #  MAIN
# # =============================================================================
# def main():
#     db.init_db()                        # SQLite init + cache seed + worker start
#     db.sync_roster(_known_names)        # employees table sync
#     n = db.sync_employee_photos_from_dir()
#     if n:
#         print(f"[photos] {n} employee profile photo(s) linked from {PHOTOS_DIR}")
#     _rebuild_cache_from_db()            # present_cache sync from in-memory

#     running = {"value": True}
#     workers = [CameraWorker(cfg, running) for cfg in RTSP_CAMERAS]

#     for worker in workers:
#         worker.open()

#     threads = []
#     for worker in workers[:-1]:
#         t = threading.Thread(target=worker.run, daemon=True)
#         t.start()
#         threads.append(t)

#     workers[-1].run()   # last worker main thread pe

#     for t in threads:
#         t.join()

#     cv2.destroyAllWindows()
#     print("All cameras stopped.")
#     time.sleep(1)
#     os._exit(0)


# if __name__ == "__main__":
#     main()


# #----------------------------------------------update---------------------------------------------
# import cv2
# import os
# import time
# import pickle
# import threading
# import numpy as np
# from datetime import datetime
# from urllib.parse import quote

# import config
# import db

# # =============================================================================
# #  CONFIG
# # =============================================================================
# EMBEDDINGS_FILE = config.EMBEDDINGS_FILE
# PHOTOS_DIR      = config.PHOTOS_DIR

# RTSP_CAMERAS = [
#     {
#         "id":      "Entry-1",
#         "url":     "rtsp://192.168.1.101:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     {
#         "id":      "Entry-2",
#         "url":     "rtsp://192.168.1.104:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
#     # {
#     #     "id":      "Entry-3",
#     #     "url":     "rtsp://192.168.1.107:554/video/live?channel=1&subtype=1",
#     #     "user_id": "admin",
#     #     "user_pw": "Tts@110092",
#     # },
#     {
#         "id":      "Entry-4",
#         "url":     "rtsp://192.168.1.106:554/video/live?channel=1&subtype=1",
#         "user_id": "admin",
#         "user_pw": "Tts@110092",
#     },
# ]

# FRAME_WIDTH           = 1280
# FRAME_HEIGHT          = 720
# THRESHOLD             = 0.50
# DETECTION_INTERVAL    = 10      # har 10 frames pe face detect
# TRACK_HOLD_FRAMES     = 20      # box itne frames tak dikhta rahe
# MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
# TARGET_FPS            = 10

# # Live reload — embeddings.pkl file-watcher
# RELOAD_CHECK_INTERVAL = 10   # seconds

# # =============================================================================
# #  VECTORIZED RECOGNITION MATRIX  +  live-reload lock
# # =============================================================================
# _known_matrix     = None   # (N, 512) float32, L2-normalized
# _known_names      = []
# _embeddings_lock  = threading.Lock()   # recognize_face + reload dono protect
# _embeddings_mtime = 0.0

# # =============================================================================
# #  STATUS CACHE  —  db.py ke in-memory cache ka thin wrapper
# #  (SQLite-first architecture mein yeh sirf db._mem_cache ko delegate karta hai)
# #
# #  DATE-ROLLOVER: db.py ke andar _mem_reset_if_new_day() handle karta hai.
# # =============================================================================
# def _refresh_status_cache_if_stale():
#     """
#     SQLite-first mein yeh function sirf date rollover check karta hai.
#     Actual status hamesha db._get_effective_status() se aata hai jo
#     in-memory cache → SQLite fallback karta hai — zero network call.
#     """
#     db._mem_reset_if_new_day()


# def _cached_status(name: str):
#     """In-memory cache se status lo — zero DB/network call."""
#     return db._get_effective_status(name)


# def _set_cached_status(name: str, status: str):
#     """In-memory cache update karo."""
#     db._mem_set(name, status)


# # present_cache directly db se use karo
# present_cache      = db.present_cache
# present_cache_lock = db.present_cache_lock


# def _rebuild_cache_from_db():
#     """
#     Startup pe SQLite se cache seed karo.
#     db.init_db() ke andar _mem_seed_from_local() already call hoti hai,
#     isliye yeh function sirf present_cache sync karta hai.
#     """
#     db._sync_present_cache_from_mem()


# # =============================================================================
# #  PHOTO CAPTURE
# # =============================================================================
# def _safe(name: str) -> str:
#     return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# def _save_capture(frame, bbox, name, cam_id):
#     """
#     Face + shoulders ka crop banao, 480x600 mein pad karke save karo.
#     Returns path relative to PHOTOS_DIR.
#     """
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
#         crop_ratio   = cw / ch

#         if crop_ratio > target_ratio:
#             need_h  = int(cw / target_ratio)
#             extra   = max(0, need_h - ch)
#             top_pad = extra // 2
#             bot_pad = extra - top_pad
#             crop = cv2.copyMakeBorder(crop, top_pad, bot_pad, 0, 0,
#                                       cv2.BORDER_REPLICATE)
#         else:
#             need_w    = int(ch * target_ratio)
#             extra     = max(0, need_w - cw)
#             left_pad  = extra // 2
#             right_pad = extra - left_pad
#             crop = cv2.copyMakeBorder(crop, 0, 0, left_pad, right_pad,
#                                       cv2.BORDER_REPLICATE)

#         crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

#         sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         crop = cv2.filter2D(crop, -1, sharpen_kernel)

#         day_dir  = datetime.now().strftime("%Y-%m-%d")
#         rel_dir  = os.path.join(_safe(name), day_dir)
#         abs_dir  = os.path.join(PHOTOS_DIR, rel_dir)
#         os.makedirs(abs_dir, exist_ok=True)

#         fname    = f"{cam_id}_{datetime.now().strftime('%H%M%S')}.jpg"
#         rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
#         abs_path = os.path.join(abs_dir, fname)

#         cv2.imwrite(abs_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

#         def _upload_to_supabase(local_path, storage_path):
#     try:
#         import requests
#         SUPABASE_URL = "https://bjgiirtpqfjaaxnftptf.supabase.co"
#         SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
#         headers = {
#             "apikey": SUPABASE_KEY,
#             "Authorization": f"Bearer {SUPABASE_KEY}",
#             "Content-Type": "image/jpeg",
#             "x-upsert": "true"
#         }
#         url = f"{SUPABASE_URL}/storage/v1/object/employee-photos/{storage_path}"
#         with open(local_path, "rb") as f:
#             requests.put(url, headers=headers, data=f, timeout=10)
#     except Exception as e:
#         print(f"[photo] Supabase upload failed (non-fatal): {e}")

# threading.Thread(
#     target=_upload_to_supabase,
#     args=(abs_path, rel_path),
#     daemon=True
# ).start()

# return rel_path
#         return rel_path
#     except Exception as e:
#         print(f"[photo] save failed for {name}: {e}")
#         return None


# # =============================================================================
# #  FFMPEG / GSTREAMER HELPERS
# # =============================================================================
# def _build_ffmpeg_url(url: str, user_id: str, user_pw: str) -> str:
#     prefix = "rtsp://"
#     body   = url[len(prefix):]
#     if "@" in body.split("/")[0]:
#         return url
#     safe_pw = quote(user_pw, safe="")
#     return f"{prefix}{user_id}:{safe_pw}@{body}"


# def _open_ffmpeg_cap(cam_id: str, ffmpeg_url: str) -> cv2.VideoCapture:
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#         "rtsp_transport;tcp|stimeout;5000000"
#     )
#     print(f"[{cam_id}] Trying FFmpeg: {ffmpeg_url}")
#     cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
#     if cap.isOpened():
#         print(f"[{cam_id}] ✅ FFmpeg stream opened")
#     else:
#         print(f"[{cam_id}] ❌ FFmpeg failed to open stream")
#     return cap


# def build_gst_pipeline(url, user_id, user_pw, width, height):
#     return (
#         f'rtspsrc location="{url}" '
#         f'user-id="{user_id}" user-pw="{user_pw}" '
#         f'latency=100 protocols=tcp ! '
#         f'decodebin ! videoconvert ! videoscale ! '
#         f'video/x-raw,width={width},height={height},format=BGR ! '
#         f'appsink drop=true max-buffers=1 sync=false'
#     )


# def open_capture_main_thread(cam_id, url, user_id, user_pw):
#     gst_pipeline = build_gst_pipeline(url, user_id, user_pw, FRAME_WIDTH, FRAME_HEIGHT)
#     print(f"[{cam_id}] Trying GStreamer:\n  {gst_pipeline}")
#     cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

#     if cap.isOpened():
#         deadline = time.time() + 5.0
#         while time.time() < deadline:
#             ret, frame = cap.read()
#             if ret and frame is not None:
#                 print(f"[{cam_id}] ✅ GStreamer confirmed (GPU decoding)")
#                 return cap, frame
#             time.sleep(0.05)
#         cap.release()
#         print(f"[{cam_id}] ⚠️  GStreamer opened but no frames — FFmpeg fallback")
#     else:
#         print(f"[{cam_id}] ⚠️  GStreamer failed — FFmpeg fallback")

#     ffmpeg_url = _build_ffmpeg_url(url, user_id, user_pw)
#     cap = _open_ffmpeg_cap(cam_id, ffmpeg_url)
#     return cap, None


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
#     print(f"  {name}: {len(emb_list)} embeddings")

# print(f"Total employees loaded: {len(known_face_db)}\n")

# _known_names  = list(known_face_db.keys())
# _known_matrix = np.stack(list(known_face_db.values()), axis=0)   # (N, 512)

# # =============================================================================
# #  LOAD FACE MODEL
# # =============================================================================
# from insightface.app import FaceAnalysis
# face_app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
# )
# face_app.prepare(ctx_id=0, det_size=(320, 320))


# # =============================================================================
# #  VECTORIZED RECOGNITION  —  single matmul, no Python loop
# # =============================================================================
# def recognize_face(live_embedding: np.ndarray):
#     """
#     Thread-safe recognition — _embeddings_lock se protect karo kyunki
#     _watch_embeddings() background thread _known_matrix live update kar
#     sakta hai kisi bhi waqt.
#     """
#     emb = live_embedding / np.linalg.norm(live_embedding)
#     with _embeddings_lock:
#         scores = _known_matrix @ emb
#         idx    = int(np.argmax(scores))
#         name   = _known_names[idx]
#         score  = float(scores[idx])
#     return name, score


# # =============================================================================
# #  ATTENDANCE MARKING
# #
# #  SQLite-first flow:
# #    1. In-memory cache check → already Present? return immediately
# #    2. Photo save karo (disk pe — fast, no network)
# #    3. db.mark_attendance() → SQLite insert + cache update
# #    4. Camera pe turant "Present Marked" dikhega ✅
# #    5. db.py ka background worker 5 min mein Supabase push karega
# # =============================================================================
# def mark_attendance(name, camera_id, frame, bbox, confidence):
#     last_status = _cached_status(name)

#     if last_status == "Present":
#         db.log_detection(name, camera_id, confidence, "already_present")
#         return "already_present"

#     # Photo save (fast — disk only, no network)
#     photo_path = _save_capture(frame, bbox, name, camera_id)

#     # db.mark_attendance() → SQLite + in-memory cache update
#     result = db.mark_attendance(
#         name, camera_id, confidence=confidence, photo_path=photo_path
#     )

#     db.log_detection(name, camera_id, confidence, result)

#     if result == "marked":
#         action = "RE-ENTRY" if last_status == "Exit" else "FIRST ENTRY"
#         print(
#             f"[{action}] {name} via {camera_id} at "
#             f"{datetime.now().strftime('%H:%M:%S')} "
#             f"— SQLite saved, Supabase sync 5 min mein"
#         )

#     return result


# # =============================================================================
# #  PER-CAMERA WORKER
# # =============================================================================
# class CameraWorker:

#     def __init__(self, camera_cfg, shared_running_flag):
#         self.cam_id  = camera_cfg["id"]
#         self.url     = camera_cfg["url"]
#         self.user_id = camera_cfg["user_id"]
#         self.user_pw = camera_cfg["user_pw"]
#         self.running = shared_running_flag

#         self.latest_frame = None
#         self.frame_lock   = threading.Lock()
#         self.frame_event  = threading.Event()

#         self.last_mark_attempt = {}
#         self.tracked_faces     = []
#         self.track_lock        = threading.Lock()

#         # Display buffer — GUI (imshow) sirf main thread pe hoga
#         self.window_name   = f"Entry Gate \u2014 {self.cam_id}"
#         self.display_frame = None
#         self.display_lock  = threading.Lock()

#         self._cap = None

#     def open(self):
#         cap, first_frame = open_capture_main_thread(
#             self.cam_id, self.url, self.user_id, self.user_pw
#         )
#         self._cap = cap
#         if first_frame is not None:
#             with self.frame_lock:
#                 self.latest_frame = first_frame.copy()
#             self.frame_event.set()

#     def _reopen(self):
#         print(f"[{self.cam_id}] Reconnecting via FFmpeg …")
#         ffmpeg_url = _build_ffmpeg_url(self.url, self.user_id, self.user_pw)
#         return _open_ffmpeg_cap(self.cam_id, ffmpeg_url)

#     # ── Capture thread ────────────────────────────────────────────────────────
#     def _capture_loop(self):
#         cap = self._cap

#         if not cap.isOpened():
#             print(f"[{self.cam_id}] ERROR: stream open nahi hua — IP/credentials/network check karo.")
#             self.running["value"] = False
#             return

#         while self.running["value"]:
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[{self.cam_id}] Stream lost — reconnecting …")
#                 cap.release()
#                 time.sleep(2)
#                 cap = self._reopen()
#                 continue
#             with self.frame_lock:
#                 self.latest_frame = frame
#             self.frame_event.set()

#         cap.release()

#     # ── start() — capture + process dono background threads mein ─────────────
#     def start(self):
#         threading.Thread(target=self._capture_loop, daemon=True,
#                          name=f"cap-{self.cam_id}").start()
#         threading.Thread(target=self._process_loop, daemon=True,
#                          name=f"proc-{self.cam_id}").start()

#     # ── show() — MAIN THREAD ONLY. Annotated frame window mein dikhao.
#     #    Headless (systemd service) mode ke liye commented out — no display
#     #    wahan attached hota. Manual debug run ke liye uncomment karo.
#     def show(self):
#         with self.display_lock:
#             frame = self.display_frame
#         # if frame is not None:
#         #     cv2.imshow(self.window_name, frame)

#     # ── Processing loop (background thread — NO imshow/waitKey here) ─────────
#     def _process_loop(self):
#         frame_count = 0
#         frame_time  = 1.0 / TARGET_FPS

#         if not self.frame_event.wait(timeout=10.0):
#             print(f"[{self.cam_id}] No frames received — skipping.")
#             return

#         while self.running["value"]:
#             loop_start = time.time()

#             self.frame_event.wait(timeout=frame_time)
#             self.frame_event.clear()

#             with self.frame_lock:
#                 if self.latest_frame is None:
#                     continue
#                 frame = self.latest_frame.copy()

#             frame_count   += 1
#             display_frame  = frame

#             # Date rollover check (midnight ke baad cache reset)
#             _refresh_status_cache_if_stale()

#             # ── Periodic face detection ───────────────────────────────────────
#             if frame_count % DETECTION_INTERVAL == 0:
#                 faces       = face_app.get(frame)
#                 new_tracked = []
#                 print(f"[{self.cam_id}] frame={frame_count} faces={len(faces)}", flush=True)
#                 for face in faces:
#                     x1, y1, x2, y2       = map(int, face.bbox)
#                     best_name, best_score = recognize_face(face.embedding)
#                     print(f"[DEBUG] {best_name} score={best_score:.3f}", flush=True)

#                     print(f"[DEBUG] {best_name} score={best_score:.3f}", flush=True)
#                     if best_score >= THRESHOLD:
#                         now      = time.time()
#                         last_try = self.last_mark_attempt.get(best_name, 0)

#                         if now - last_try > MARK_COOLDOWN_SECONDS:
#                             result = mark_attendance(
#                                 best_name, self.cam_id, frame,
#                                 (x1, y1, x2, y2), best_score,
#                             )
#                             self.last_mark_attempt[best_name] = now
#                         else:
#                             result = "cooldown"

#                         # ── Display label logic ───────────────────────────────
#                         if result == "marked":
#                             rows_today = db.count_today_rows(best_name)
#                             label = (
#                                 f"{best_name} - "
#                                 f"{'Re-Entry' if rows_today > 1 else 'Present'} "
#                                 f"({best_score:.2f}) [SQLite✓]"
#                             )
#                             color = (0, 255, 0)       # green

#                         elif result == "already_present":
#                             label = f"{best_name} - Already Present ({best_score:.2f})"
#                             color = (0, 255, 255)     # yellow

#                         elif result == "cooldown":
#                             label = f"{best_name} ({best_score:.2f}) [cooldown]"
#                             color = (255, 200, 0)     # orange

#                         else:
#                             label = f"{best_name} ({best_score:.2f})"
#                             color = (255, 200, 0)

#                     else:
#                         label = f"Unknown ({best_score:.2f})"
#                         color = (0, 0, 255)           # red

#                     new_tracked.append({
#                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                         "label": label, "color": color,
#                         "hold": TRACK_HOLD_FRAMES,
#                     })

#                 with self.track_lock:
#                     self.tracked_faces = new_tracked

#             # ── Draw tracked boxes ────────────────────────────────────────────
#             with self.track_lock:
#                 current_faces      = list(self.tracked_faces)
#                 self.tracked_faces = [f for f in self.tracked_faces if f["hold"] > 0]
#                 for f in self.tracked_faces:
#                     f["hold"] -= 1

#             for f in current_faces:
#                 x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
#                 color, label    = f["color"], f["label"]

#                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#                 label_y     = max(30, y1 - 10)
#                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#                 cv2.rectangle(
#                     display_frame,
#                     (x1, label_y - th - 6),
#                     (x1 + tw + 6, label_y + 2),
#                     color, -1,
#                 )
#                 cv2.putText(
#                     display_frame, label,
#                     (x1 + 3, label_y - 2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
#                 )

#             # ── Summary overlay (in-memory cache — zero DB/network) ───────────
#             with present_cache_lock:
#                 present_snapshot = set(present_cache)

#             summary = "Inside Now: " + (
#                 ", ".join(sorted(present_snapshot)) if present_snapshot else "None"
#             )
#             cv2.putText(
#                 display_frame, summary,
#                 (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2,
#             )

#             # Pending sync count overlay — SQLite me kitne baaki hain
#             pending_n = db.pending_sync_count()
#             if pending_n > 0:
#                 sync_text = f"Pending sync: {pending_n}"
#                 cv2.putText(
#                     display_frame, sync_text,
#                     (20, 58),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2,
#                 )

#             cv2.putText(
#                 display_frame, self.cam_id,
#                 (FRAME_WIDTH - 140, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
#             )

#             # Frame main thread ke liye save karo — imshow wahan hoga
#             with self.display_lock:
#                 self.display_frame = display_frame

#             elapsed    = time.time() - loop_start
#             sleep_time = frame_time - elapsed
#             if sleep_time > 0:
#                 time.sleep(sleep_time)


# # =============================================================================
# #  MAIN
# # =============================================================================
# def _reload_embeddings():
#     """
#     embeddings.pkl se fresh embeddings load karo aur
#     _known_matrix + _known_names live update karo.
#     Camera processing thread read karta hai inhe — lock se protect karo.
#     """
#     global _known_matrix, _known_names, _embeddings_mtime

#     try:
#         with open(EMBEDDINGS_FILE, "rb") as f:
#             import pickle as _pickle
#             data = _pickle.load(f)
#     except Exception as e:
#         print(f"[reload] embeddings.pkl read failed: {e}")
#         return

#     new_names  = []
#     new_vecs   = []
#     for name, emb_list in data.items():
#         arr      = np.array(emb_list, dtype=np.float32)
#         mean_emb = np.mean(arr, axis=0)
#         mean_emb = mean_emb / np.linalg.norm(mean_emb)
#         new_names.append(name)
#         new_vecs.append(mean_emb)

#     if not new_vecs:
#         print("[reload] embeddings.pkl empty — skip.")
#         return

#     new_matrix = np.stack(new_vecs, axis=0)

#     with _embeddings_lock:
#         _known_names  = new_names
#         _known_matrix = new_matrix
#         _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)

#     print(f"[reload] ✅ embeddings.pkl reloaded — {len(new_names)} employee(s) active.")


# def _watch_embeddings():
#     """Background thread — embeddings.pkl ka mtime check karo."""
#     global _embeddings_mtime

#     # Startup mtime set karo
#     try:
#         _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)
#     except Exception:
#         pass

#     while True:
#         time.sleep(RELOAD_CHECK_INTERVAL)
#         try:
#             current_mtime = os.path.getmtime(EMBEDDINGS_FILE)
#             if current_mtime != _embeddings_mtime:
#                 print(f"[reload] embeddings.pkl changed — reloading ...")
#                 _reload_embeddings()
#         except Exception as e:
#             print(f"[reload] watcher error (non-fatal): {e}")


# # =============================================================================
# #  SUPABASE EMBEDDINGS POLL
# #
# #  Admin panel Jetson pe chale ya kahin bhi (Render, ya koi doosra PC) — har
# #  photo ka embedding auth_db.add_employee_embedding() se Supabase ke
# #  face_embeddings table mein turant chala jaata hai (uska add/update route
# #  hamesha network-reachable Supabase ko hi likhta hai). Lekin local
# #  embeddings.pkl file sirf usi machine pe update hoti hai jahan request
# #  process hui — agar admin panel Render (ya kisi doosre PC) se use ho raha
# #  ho, wahan ka PHOTOS_DIR/embeddings.pkl is Jetson se bilkul alag hai, aur
# #  file-watcher (_watch_embeddings, upar) ko kabhi pata hi nahi chalega.
# #
# #  Yeh periodic poll us gap ko band karta hai: Supabase se seedha padhta
# #  hai, local pickle ke saath MERGE karta hai (dono taraf ke embeddings ek
# #  saath), aur naye Supabase-only vectors ko local pickle mein bhi backfill
# #  kar deta hai — taaki offline restart ke baad bhi last-known-good state
# #  local cache se mil jaaye.
# # =============================================================================
# SUPABASE_EMBEDDINGS_POLL_INTERVAL = 30   # seconds


# def _reload_embeddings_merged():
#     """Local embeddings.pkl + Supabase face_embeddings dono merge karke
#        _known_matrix/_known_names rebuild karo. Naye Supabase-only vectors
#        local pickle mein bhi backfill ho jaate hain (offline cache)."""
#     global _known_matrix, _known_names, _embeddings_mtime

#     # 1) Local pickle padho (agar hai) — baseline
#     local_data = {}
#     try:
#         with open(EMBEDDINGS_FILE, "rb") as f:
#             local_data = pickle.load(f)
#     except Exception:
#         pass   # pehli baar ho sakta hai ki file hi na ho

#     # 2) Supabase se sabhi ACTIVE employees ke embeddings padho
#     try:
#         sb_raw = db.get_all_face_embeddings()   # { name: [raw_bytes, ...] }
#     except Exception as e:
#         print(f"[reload] Supabase embeddings fetch failed (non-fatal): {e}")
#         sb_raw = {}

#     if not local_data and not sb_raw:
#         return

#     # 3) Merge — Supabase ke naye vectors (jo local mein nahi hain) backfill karo
#     changed = False
#     for name, raw_list in sb_raw.items():
#         existing = local_data.setdefault(name, [])
#         existing_set = {np.asarray(e, dtype=np.float32).tobytes() for e in existing}
#         for raw in raw_list:
#             vec = np.frombuffer(raw, dtype=np.float32)
#             vb  = vec.tobytes()
#             if vb not in existing_set:
#                 existing.append(vec)
#                 existing_set.add(vb)
#                 changed = True

#     # 4) Kuch naya mila to local pickle mein backfill save karo
#     if changed:
#         try:
#             os.makedirs(os.path.dirname(EMBEDDINGS_FILE) or ".", exist_ok=True)
#             with open(EMBEDDINGS_FILE, "wb") as f:
#                 pickle.dump(local_data, f)
#         except Exception as e:
#             print(f"[reload] local pickle backfill save failed (non-fatal): {e}")

#     # 5) _known_matrix/_known_names rebuild — sirf tab jab kuch naya mila ho
#     if not changed:
#         return

#     new_names, new_vecs = [], []
#     for name, emb_list in local_data.items():
#         if not emb_list:
#             continue
#         arr      = np.array(emb_list, dtype=np.float32)
#         mean_emb = np.mean(arr, axis=0)
#         mean_emb = mean_emb / np.linalg.norm(mean_emb)
#         new_names.append(name)
#         new_vecs.append(mean_emb)

#     if not new_vecs:
#         return

#     new_matrix = np.stack(new_vecs, axis=0)
#     with _embeddings_lock:
#         _known_names  = new_names
#         _known_matrix = new_matrix
#         try:
#             _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)
#         except Exception:
#             pass

#     print(f"[reload] ✅ Supabase se naye embeddings mile (remote add?) — "
#           f"{len(new_names)} employee(s) active (merged).")


# def _watch_embeddings_supabase():
#     """Background thread — Supabase face_embeddings table ko periodically
#        poll karo, taaki kisi bhi machine se add kiya employee is Jetson ke
#        cameras pe recognizable ho jaaye."""
#     while True:
#         time.sleep(SUPABASE_EMBEDDINGS_POLL_INTERVAL)
#         try:
#             _reload_embeddings_merged()
#         except Exception as e:
#             print(f"[reload] Supabase watcher error (non-fatal): {e}")


# def main():
#     db.init_db()                        # SQLite init + cache seed + worker start
#     db.sync_roster(_known_names)        # employees table sync
#     n = db.sync_employee_photos_from_dir()
#     if n:
#         print(f"[photos] {n} employee profile photo(s) linked from {PHOTOS_DIR}")
#     _rebuild_cache_from_db()            # present_cache sync from in-memory

#     # Embeddings file-watcher start karo — live reload bina restart ke
#     watcher_t = threading.Thread(target=_watch_embeddings, daemon=True,
#                                  name="embeddings-watcher")
#     watcher_t.start()
#     print(f"[reload] Embeddings watcher started (check every {RELOAD_CHECK_INTERVAL}s)")

#     # Supabase embeddings poll start karo — REMOTE machine se add kiya
#     # employee bhi is Jetson ke cameras pe recognizable ho jaaye
#     sb_watcher_t = threading.Thread(target=_watch_embeddings_supabase, daemon=True,
#                                     name="embeddings-supabase-watcher")
#     sb_watcher_t.start()
#     print(f"[reload] Supabase embeddings poll started (check every {SUPABASE_EMBEDDINGS_POLL_INTERVAL}s)")

#     running = {"value": True}
#     workers = [CameraWorker(cfg, running) for cfg in RTSP_CAMERAS]

#     # Pehle sabki streams open karo (main thread pe — GStreamer needs this)
#     for worker in workers:
#         worker.open()

#     # Windows pehle create karo MAIN THREAD pe — Qt needs this before imshow
#     # Headless (systemd service) mode ke liye commented out — no display.
#     # for worker in workers:
#     #     cv2.namedWindow(worker.window_name, cv2.WINDOW_NORMAL)
#     #     cv2.resizeWindow(worker.window_name, 640, 360)

#     # Sab workers ke capture + processing threads start karo (background)
#     for worker in workers:
#         worker.start()

#     print("[main] Headless mode — cameras running in background, no display")

#     # ── Idle loop — GUI disabled for headless service mode. Capture +
#     #    detection already run fully in the background threads above; this
#     #    just keeps the process alive until a worker signals a failure.
#     while running["value"]:
#         # for worker in workers:
#         #     worker.show()
#         # key = cv2.waitKey(1) & 0xFF
#         # if key == ord("q"):
#         #     running["value"] = False
#         #     break
#         time.sleep(0.5)

#     # cv2.destroyAllWindows()
#     print("All cameras stopped.")
#     time.sleep(1)
#     os._exit(0)


# if __name__ == "__main__":
#     main()



#----------------------------------------------update---------------------------------------------
import cv2
import os
import time
import pickle
import threading
import numpy as np
from datetime import datetime
from urllib.parse import quote

import config
import db

# =============================================================================
#  CONFIG
# =============================================================================
EMBEDDINGS_FILE = config.EMBEDDINGS_FILE
PHOTOS_DIR      = config.PHOTOS_DIR

RTSP_CAMERAS = [
    {
        "id":      "Entry-1",
        "url":     "rtsp://192.168.1.101:554/video/live?channel=1&subtype=1",
        "user_id": "admin",
        "user_pw": "Tts@110092",
    },
    {
        "id":      "Entry-2",
        "url":     "rtsp://192.168.1.104:554/video/live?channel=1&subtype=1",
        "user_id": "admin",
        "user_pw": "Tts@110092",
    },
    # {
    #     "id":      "Entry-3",
    #     "url":     "rtsp://192.168.1.107:554/video/live?channel=1&subtype=1",
    #     "user_id": "admin",
    #     "user_pw": "Tts@110092",
    # },
    {
        "id":      "Entry-4",
        "url":     "rtsp://192.168.1.106:554/video/live?channel=1&subtype=1",
        "user_id": "admin",
        "user_pw": "Tts@110092",
    },
]

FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720
THRESHOLD             = 0.50
DETECTION_INTERVAL    = 10      # har 10 frames pe face detect
TRACK_HOLD_FRAMES     = 20      # box itne frames tak dikhta rahe
MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
TARGET_FPS            = 10

# Live reload — embeddings.pkl file-watcher
RELOAD_CHECK_INTERVAL = 10   # seconds

# =============================================================================
#  VECTORIZED RECOGNITION MATRIX  +  live-reload lock
# =============================================================================
_known_matrix     = None   # (N, 512) float32, L2-normalized
_known_names      = []
_embeddings_lock  = threading.Lock()   # recognize_face + reload dono protect
_embeddings_mtime = 0.0

# =============================================================================
#  STATUS CACHE  —  db.py ke in-memory cache ka thin wrapper
#  (SQLite-first architecture mein yeh sirf db._mem_cache ko delegate karta hai)
#
#  DATE-ROLLOVER: db.py ke andar _mem_reset_if_new_day() handle karta hai.
# =============================================================================
def _refresh_status_cache_if_stale():
    """
    SQLite-first mein yeh function sirf date rollover check karta hai.
    Actual status hamesha db._get_effective_status() se aata hai jo
    in-memory cache → SQLite fallback karta hai — zero network call.
    """
    db._mem_reset_if_new_day()


def _cached_status(name: str):
    """In-memory cache se status lo — zero DB/network call."""
    return db._get_effective_status(name)


def _set_cached_status(name: str, status: str):
    """In-memory cache update karo."""
    db._mem_set(name, status)


# present_cache directly db se use karo
present_cache      = db.present_cache
present_cache_lock = db.present_cache_lock


def _rebuild_cache_from_db():
    """
    Startup pe SQLite se cache seed karo.
    db.init_db() ke andar _mem_seed_from_local() already call hoti hai,
    isliye yeh function sirf present_cache sync karta hai.
    """
    db._sync_present_cache_from_mem()


# =============================================================================
#  PHOTO CAPTURE
# =============================================================================
def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _save_capture(frame, bbox, name, cam_id):
    """
    Face + shoulders ka crop banao, 480x600 mein pad karke save karo.
    Returns path relative to PHOTOS_DIR.
    """
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

        def _upload_to_supabase(local_path, storage_path):
            try:
                import requests
                SUPABASE_URL = "https://bjgiirtpqfjaaxnftptf.supabase.co"
                SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "image/jpeg",
                    "x-upsert": "true"
                }
                url = f"{SUPABASE_URL}/storage/v1/object/employee-photos/{storage_path}"
                with open(local_path, "rb") as f:
                    requests.put(url, headers=headers, data=f, timeout=10)
            except Exception as e:
                print(f"[photo] Supabase upload failed (non-fatal): {e}")
    
        threading.Thread(
            target=_upload_to_supabase,
            args=(abs_path, rel_path),
            daemon=True
        ).start()

        return rel_path
    except Exception as e:
        print(f"[photo] save failed for {name}: {e}")
        return None


# =============================================================================
#  FFMPEG / GSTREAMER HELPERS
# =============================================================================
def _build_ffmpeg_url(url: str, user_id: str, user_pw: str) -> str:
    prefix = "rtsp://"
    body   = url[len(prefix):]
    if "@" in body.split("/")[0]:
        return url
    safe_pw = quote(user_pw, safe="")
    return f"{prefix}{user_id}:{safe_pw}@{body}"


def _open_ffmpeg_cap(cam_id: str, ffmpeg_url: str) -> cv2.VideoCapture:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000"
    )
    print(f"[{cam_id}] Trying FFmpeg: {ffmpeg_url}")
    cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if cap.isOpened():
        print(f"[{cam_id}] ✅ FFmpeg stream opened")
    else:
        print(f"[{cam_id}] ❌ FFmpeg failed to open stream")
    return cap


def build_gst_pipeline(url, user_id, user_pw, width, height):
    return (
        f'rtspsrc location="{url}" '
        f'user-id="{user_id}" user-pw="{user_pw}" '
        f'latency=100 protocols=tcp ! '
        f'decodebin ! videoconvert ! videoscale ! '
        f'video/x-raw,width={width},height={height},format=BGR ! '
        f'appsink drop=true max-buffers=1 sync=false'
    )


def open_capture_main_thread(cam_id, url, user_id, user_pw):
    gst_pipeline = build_gst_pipeline(url, user_id, user_pw, FRAME_WIDTH, FRAME_HEIGHT)
    print(f"[{cam_id}] Trying GStreamer:\n  {gst_pipeline}")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[{cam_id}] ✅ GStreamer confirmed (GPU decoding)")
                return cap, frame
            time.sleep(0.05)
        cap.release()
        print(f"[{cam_id}] ⚠️  GStreamer opened but no frames — FFmpeg fallback")
    else:
        print(f"[{cam_id}] ⚠️  GStreamer failed — FFmpeg fallback")

    ffmpeg_url = _build_ffmpeg_url(url, user_id, user_pw)
    cap = _open_ffmpeg_cap(cam_id, ffmpeg_url)
    return cap, None


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
    print(f"  {name}: {len(emb_list)} embeddings")

print(f"Total employees loaded: {len(known_face_db)}\n")

_known_names  = list(known_face_db.keys())
_known_matrix = np.stack(list(known_face_db.values()), axis=0)   # (N, 512)

# =============================================================================
#  LOAD FACE MODEL
# =============================================================================
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(320, 320))


# =============================================================================
#  VECTORIZED RECOGNITION  —  single matmul, no Python loop
# =============================================================================
def recognize_face(live_embedding: np.ndarray):
    """
    Thread-safe recognition — _embeddings_lock se protect karo kyunki
    _watch_embeddings() background thread _known_matrix live update kar
    sakta hai kisi bhi waqt.
    """
    emb = live_embedding / np.linalg.norm(live_embedding)
    with _embeddings_lock:
        scores = _known_matrix @ emb
        idx    = int(np.argmax(scores))
        name   = _known_names[idx]
        score  = float(scores[idx])
    return name, score


# =============================================================================
#  ATTENDANCE MARKING
#
#  SQLite-first flow:
#    1. In-memory cache check → already Present? return immediately
#    2. Photo save karo (disk pe — fast, no network)
#    3. db.mark_attendance() → SQLite insert + cache update
#    4. Camera pe turant "Present Marked" dikhega ✅
#    5. db.py ka background worker 5 min mein Supabase push karega
# =============================================================================
def mark_attendance(name, camera_id, frame, bbox, confidence):
    last_status = _cached_status(name)

    if last_status == "Present":
        db.log_detection(name, camera_id, confidence, "already_present")
        return "already_present"

    # Photo save (fast — disk only, no network)
    photo_path = _save_capture(frame, bbox, name, camera_id)

    # db.mark_attendance() → SQLite + in-memory cache update
    result = db.mark_attendance(
        name, camera_id, confidence=confidence, photo_path=photo_path
    )

    db.log_detection(name, camera_id, confidence, result)

    if result == "marked":
        action = "RE-ENTRY" if last_status == "Exit" else "FIRST ENTRY"
        print(
            f"[{action}] {name} via {camera_id} at "
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"— SQLite saved, Supabase sync 5 min mein"
        )

    return result


# =============================================================================
#  PER-CAMERA WORKER
# =============================================================================
class CameraWorker:

    def __init__(self, camera_cfg, shared_running_flag):
        self.cam_id  = camera_cfg["id"]
        self.url     = camera_cfg["url"]
        self.user_id = camera_cfg["user_id"]
        self.user_pw = camera_cfg["user_pw"]
        self.running = shared_running_flag

        self.latest_frame = None
        self.frame_lock   = threading.Lock()
        self.frame_event  = threading.Event()

        self.last_mark_attempt = {}
        self.tracked_faces     = []
        self.track_lock        = threading.Lock()

        # Display buffer — GUI (imshow) sirf main thread pe hoga
        self.window_name   = f"Entry Gate \u2014 {self.cam_id}"
        self.display_frame = None
        self.display_lock  = threading.Lock()

        self._cap = None

    def open(self):
        cap, first_frame = open_capture_main_thread(
            self.cam_id, self.url, self.user_id, self.user_pw
        )
        self._cap = cap
        if first_frame is not None:
            with self.frame_lock:
                self.latest_frame = first_frame.copy()
            self.frame_event.set()

    def _reopen(self):
        print(f"[{self.cam_id}] Reconnecting via FFmpeg …")
        ffmpeg_url = _build_ffmpeg_url(self.url, self.user_id, self.user_pw)
        return _open_ffmpeg_cap(self.cam_id, ffmpeg_url)

    # ── Capture thread ────────────────────────────────────────────────────────
    def _capture_loop(self):
        cap = self._cap

        if not cap.isOpened():
            print(f"[{self.cam_id}] ERROR: stream open nahi hua — IP/credentials/network check karo.")
            self.running["value"] = False
            return

        while self.running["value"]:
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.cam_id}] Stream lost — reconnecting …")
                cap.release()
                time.sleep(2)
                cap = self._reopen()
                continue
            with self.frame_lock:
                self.latest_frame = frame
            self.frame_event.set()

        cap.release()

    # ── start() — capture + process dono background threads mein ─────────────
    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True,
                         name=f"cap-{self.cam_id}").start()
        threading.Thread(target=self._process_loop, daemon=True,
                         name=f"proc-{self.cam_id}").start()

    # ── show() — MAIN THREAD ONLY. Annotated frame window mein dikhao.
    #    Headless (systemd service) mode ke liye commented out — no display
    #    wahan attached hota. Manual debug run ke liye uncomment karo.
    def show(self):
        with self.display_lock:
            frame = self.display_frame
        # if frame is not None:
        #     cv2.imshow(self.window_name, frame)

    # ── Processing loop (background thread — NO imshow/waitKey here) ─────────
    def _process_loop(self):
        frame_count = 0
        frame_time  = 1.0 / TARGET_FPS

        if not self.frame_event.wait(timeout=10.0):
            print(f"[{self.cam_id}] No frames received — skipping.")
            return

        while self.running["value"]:
            loop_start = time.time()

            self.frame_event.wait(timeout=frame_time)
            self.frame_event.clear()

            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()

            frame_count   += 1
            display_frame  = frame

            # Date rollover check (midnight ke baad cache reset)
            _refresh_status_cache_if_stale()

            # ── Periodic face detection ───────────────────────────────────────
            if frame_count % DETECTION_INTERVAL == 0:
                faces       = face_app.get(frame)
                new_tracked = []
                print(f"[{self.cam_id}] frame={frame_count} faces={len(faces)}", flush=True)
                for face in faces:
                    x1, y1, x2, y2       = map(int, face.bbox)
                    best_name, best_score = recognize_face(face.embedding)
                    print(f"[DEBUG] {best_name} score={best_score:.3f}", flush=True)

                    print(f"[DEBUG] {best_name} score={best_score:.3f}", flush=True)
                    if best_score >= THRESHOLD:
                        now      = time.time()
                        last_try = self.last_mark_attempt.get(best_name, 0)

                        if now - last_try > MARK_COOLDOWN_SECONDS:
                            result = mark_attendance(
                                best_name, self.cam_id, frame,
                                (x1, y1, x2, y2), best_score,
                            )
                            self.last_mark_attempt[best_name] = now
                        else:
                            result = "cooldown"

                        # ── Display label logic ───────────────────────────────
                        if result == "marked":
                            rows_today = db.count_today_rows(best_name)
                            label = (
                                f"{best_name} - "
                                f"{'Re-Entry' if rows_today > 1 else 'Present'} "
                                f"({best_score:.2f}) [SQLite✓]"
                            )
                            color = (0, 255, 0)       # green

                        elif result == "already_present":
                            label = f"{best_name} - Already Present ({best_score:.2f})"
                            color = (0, 255, 255)     # yellow

                        elif result == "cooldown":
                            label = f"{best_name} ({best_score:.2f}) [cooldown]"
                            color = (255, 200, 0)     # orange

                        else:
                            label = f"{best_name} ({best_score:.2f})"
                            color = (255, 200, 0)

                    else:
                        label = f"Unknown ({best_score:.2f})"
                        color = (0, 0, 255)           # red

                    new_tracked.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "label": label, "color": color,
                        "hold": TRACK_HOLD_FRAMES,
                    })

                with self.track_lock:
                    self.tracked_faces = new_tracked

            # ── Draw tracked boxes ────────────────────────────────────────────
            with self.track_lock:
                current_faces      = list(self.tracked_faces)
                self.tracked_faces = [f for f in self.tracked_faces if f["hold"] > 0]
                for f in self.tracked_faces:
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

            # ── Summary overlay (in-memory cache — zero DB/network) ───────────
            with present_cache_lock:
                present_snapshot = set(present_cache)

            summary = "Inside Now: " + (
                ", ".join(sorted(present_snapshot)) if present_snapshot else "None"
            )
            cv2.putText(
                display_frame, summary,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2,
            )

            # Pending sync count overlay — SQLite me kitne baaki hain
            pending_n = db.pending_sync_count()
            if pending_n > 0:
                sync_text = f"Pending sync: {pending_n}"
                cv2.putText(
                    display_frame, sync_text,
                    (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2,
                )

            cv2.putText(
                display_frame, self.cam_id,
                (FRAME_WIDTH - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
            )

            # Frame main thread ke liye save karo — imshow wahan hoga
            with self.display_lock:
                self.display_frame = display_frame

            elapsed    = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# =============================================================================
#  MAIN
# =============================================================================
def _reload_embeddings():
    """
    embeddings.pkl se fresh embeddings load karo aur
    _known_matrix + _known_names live update karo.
    Camera processing thread read karta hai inhe — lock se protect karo.
    """
    global _known_matrix, _known_names, _embeddings_mtime

    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            import pickle as _pickle
            data = _pickle.load(f)
    except Exception as e:
        print(f"[reload] embeddings.pkl read failed: {e}")
        return

    new_names  = []
    new_vecs   = []
    for name, emb_list in data.items():
        arr      = np.array(emb_list, dtype=np.float32)
        mean_emb = np.mean(arr, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        new_names.append(name)
        new_vecs.append(mean_emb)

    if not new_vecs:
        print("[reload] embeddings.pkl empty — skip.")
        return

    new_matrix = np.stack(new_vecs, axis=0)

    with _embeddings_lock:
        _known_names  = new_names
        _known_matrix = new_matrix
        _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)

    print(f"[reload] ✅ embeddings.pkl reloaded — {len(new_names)} employee(s) active.")


def _watch_embeddings():
    """Background thread — embeddings.pkl ka mtime check karo."""
    global _embeddings_mtime

    # Startup mtime set karo
    try:
        _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)
    except Exception:
        pass

    while True:
        time.sleep(RELOAD_CHECK_INTERVAL)
        try:
            current_mtime = os.path.getmtime(EMBEDDINGS_FILE)
            if current_mtime != _embeddings_mtime:
                print(f"[reload] embeddings.pkl changed — reloading ...")
                _reload_embeddings()
        except Exception as e:
            print(f"[reload] watcher error (non-fatal): {e}")


# =============================================================================
#  SUPABASE EMBEDDINGS POLL
#
#  Admin panel Jetson pe chale ya kahin bhi (Render, ya koi doosra PC) — har
#  photo ka embedding auth_db.add_employee_embedding() se Supabase ke
#  face_embeddings table mein turant chala jaata hai (uska add/update route
#  hamesha network-reachable Supabase ko hi likhta hai). Lekin local
#  embeddings.pkl file sirf usi machine pe update hoti hai jahan request
#  process hui — agar admin panel Render (ya kisi doosre PC) se use ho raha
#  ho, wahan ka PHOTOS_DIR/embeddings.pkl is Jetson se bilkul alag hai, aur
#  file-watcher (_watch_embeddings, upar) ko kabhi pata hi nahi chalega.
#
#  Yeh periodic poll us gap ko band karta hai: Supabase se seedha padhta
#  hai, local pickle ke saath MERGE karta hai (dono taraf ke embeddings ek
#  saath), aur naye Supabase-only vectors ko local pickle mein bhi backfill
#  kar deta hai — taaki offline restart ke baad bhi last-known-good state
#  local cache se mil jaaye.
# =============================================================================
SUPABASE_EMBEDDINGS_POLL_INTERVAL = 30   # seconds


def _reload_embeddings_merged():
    """Local embeddings.pkl + Supabase face_embeddings dono merge karke
       _known_matrix/_known_names rebuild karo. Naye Supabase-only vectors
       local pickle mein bhi backfill ho jaate hain (offline cache)."""
    global _known_matrix, _known_names, _embeddings_mtime

    # 1) Local pickle padho (agar hai) — baseline
    local_data = {}
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            local_data = pickle.load(f)
    except Exception:
        pass   # pehli baar ho sakta hai ki file hi na ho

    # 2) Supabase se sabhi ACTIVE employees ke embeddings padho
    try:
        sb_raw = db.get_all_face_embeddings()   # { name: [raw_bytes, ...] }
    except Exception as e:
        print(f"[reload] Supabase embeddings fetch failed (non-fatal): {e}")
        sb_raw = {}

    if not local_data and not sb_raw:
        return

    # 3) Merge — Supabase ke naye vectors (jo local mein nahi hain) backfill karo
    changed = False
    for name, raw_list in sb_raw.items():
        existing = local_data.setdefault(name, [])
        existing_set = {np.asarray(e, dtype=np.float32).tobytes() for e in existing}
        for raw in raw_list:
            vec = np.frombuffer(raw, dtype=np.float32)
            vb  = vec.tobytes()
            if vb not in existing_set:
                existing.append(vec)
                existing_set.add(vb)
                changed = True

    # 4) Kuch naya mila to local pickle mein backfill save karo
    if changed:
        try:
            os.makedirs(os.path.dirname(EMBEDDINGS_FILE) or ".", exist_ok=True)
            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump(local_data, f)
        except Exception as e:
            print(f"[reload] local pickle backfill save failed (non-fatal): {e}")

    # 5) _known_matrix/_known_names rebuild — sirf tab jab kuch naya mila ho
    if not changed:
        return

    new_names, new_vecs = [], []
    for name, emb_list in local_data.items():
        if not emb_list:
            continue
        arr      = np.array(emb_list, dtype=np.float32)
        mean_emb = np.mean(arr, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        new_names.append(name)
        new_vecs.append(mean_emb)

    if not new_vecs:
        return

    new_matrix = np.stack(new_vecs, axis=0)
    with _embeddings_lock:
        _known_names  = new_names
        _known_matrix = new_matrix
        try:
            _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)
        except Exception:
            pass

    print(f"[reload] ✅ Supabase se naye embeddings mile (remote add?) — "
          f"{len(new_names)} employee(s) active (merged).")


def _watch_embeddings_supabase():
    """Background thread — Supabase face_embeddings table ko periodically
       poll karo, taaki kisi bhi machine se add kiya employee is Jetson ke
       cameras pe recognizable ho jaaye."""
    while True:
        time.sleep(SUPABASE_EMBEDDINGS_POLL_INTERVAL)
        try:
            _reload_embeddings_merged()
        except Exception as e:
            print(f"[reload] Supabase watcher error (non-fatal): {e}")


def main():
    db.init_db()                        # SQLite init + cache seed + worker start
    db.sync_roster(_known_names)        # employees table sync
    n = db.sync_employee_photos_from_dir()
    if n:
        print(f"[photos] {n} employee profile photo(s) linked from {PHOTOS_DIR}")
    _rebuild_cache_from_db()            # present_cache sync from in-memory

    # Embeddings file-watcher start karo — live reload bina restart ke
    watcher_t = threading.Thread(target=_watch_embeddings, daemon=True,
                                 name="embeddings-watcher")
    watcher_t.start()
    print(f"[reload] Embeddings watcher started (check every {RELOAD_CHECK_INTERVAL}s)")

    # Supabase embeddings poll start karo — REMOTE machine se add kiya
    # employee bhi is Jetson ke cameras pe recognizable ho jaaye
    sb_watcher_t = threading.Thread(target=_watch_embeddings_supabase, daemon=True,
                                    name="embeddings-supabase-watcher")
    sb_watcher_t.start()
    print(f"[reload] Supabase embeddings poll started (check every {SUPABASE_EMBEDDINGS_POLL_INTERVAL}s)")

    running = {"value": True}
    workers = [CameraWorker(cfg, running) for cfg in RTSP_CAMERAS]

    # Pehle sabki streams open karo (main thread pe — GStreamer needs this)
    for worker in workers:
        worker.open()

    # Windows pehle create karo MAIN THREAD pe — Qt needs this before imshow
    # Headless (systemd service) mode ke liye commented out — no display.
    # for worker in workers:
    #     cv2.namedWindow(worker.window_name, cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow(worker.window_name, 640, 360)

    # Sab workers ke capture + processing threads start karo (background)
    for worker in workers:
        worker.start()

    print("[main] Headless mode — cameras running in background, no display")

    # ── Idle loop — GUI disabled for headless service mode. Capture +
    #    detection already run fully in the background threads above; this
    #    just keeps the process alive until a worker signals a failure.
    while running["value"]:
        # for worker in workers:
        #     worker.show()
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     running["value"] = False
        #     break
        time.sleep(0.5)

    # cv2.destroyAllWindows()
    print("All cameras stopped.")
    time.sleep(1)
    os._exit(0)


if __name__ == "__main__":
    main()