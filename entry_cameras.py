

#----------------------------------------------update---------------------------------------------
import cv2
import os
import time
import pickle
import threading
import csv
import numpy as np
from datetime import datetime, timedelta
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
        "id":      "IT-Room",
        "url":     "rtsp://192.168.1.101:554/video/live?channel=1&subtype=1",
        "user_id": "admin",
        "user_pw": "Tts@110092",
    },
    {
        "id":      "R&D-Room",
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
        "id":      "R&D-Hall",
        "url":     "rtsp://192.168.1.106:554/video/live?channel=1&subtype=1",
        "user_id": "admin",
        "user_pw": "Tts@110092",
    },
]

FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720
THRESHOLD             = 0.50
LOW_CONF_LOG_MIN      = 0.35    # isse kam score wali detection CSV mein log nahi hogi (pure noise)
DETECTION_INTERVAL    = 10      # har 10 frames pe face detect
TRACK_HOLD_FRAMES     = 20      # box itne frames tak dikhta rahe
MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
TARGET_FPS            = 10

# Live reload — embeddings.pkl file-watcher
RELOAD_CHECK_INTERVAL = 10   # seconds

# =============================================================================
#  PRESENCE TIMEOUT — exit camera bharosemand nahi hai, isliye safety net:
#  jab bhi koi entry camera pe recognize hota hai, uska timer restart hota
#  hai. PRESENCE_TIMEOUT_MINUTES tak dobara na dikhe to automatically Exit
#  mark ho jaata hai.
# =============================================================================
PRESENCE_TIMEOUT_MINUTES = 15   # itne min entry camera pe na dikhe → outside

_last_seen: dict = {}           # { name: time.time() }
_last_seen_lock  = threading.Lock()

# =============================================================================
#  LOG BOOK — CSV entry/exit log with 2-day auto-cleanup
# =============================================================================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "camera_log.csv")
_LOG_LOCK = threading.Lock()

def _log_event_to_csv(name, camera_id, confidence, result, event_type):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _LOG_LOCK:
            write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "name", "camera_id", "event_type", "confidence", "result"])
                writer.writerow([now, name, camera_id, event_type, f"{confidence:.2f}", result])
    except Exception as e:
        print(f"[log] CSV log failed: {e}")

def _cleanup_old_logs():
    try:
        # Existing CSV cleanup
        if os.path.exists(LOG_FILE):
            cutoff = datetime.now() - timedelta(days=2)
            rows = []

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.strptime(
                            row["timestamp"],
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if ts >= cutoff:
                            rows.append(row)
                    except Exception:
                        continue

            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "name",
                        "camera_id",
                        "event_type",
                        "confidence",
                        "result"
                    ]
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"[log] CSV cleanup done — {len(rows)} rows remaining")

        # entry_camera.log cleanup
        ENTRY_LOG_FILE = os.path.join(
            os.path.dirname(__file__),
            "entry_camera.log"
        )

        if os.path.exists(ENTRY_LOG_FILE):
            file_time = datetime.fromtimestamp(
                os.path.getmtime(ENTRY_LOG_FILE)
            )

            cutoff = datetime.now() - timedelta(days=2)

            if file_time < cutoff:
                os.remove(ENTRY_LOG_FILE)
                print("[log] entry_camera.log deleted — older than 2 days")

    except Exception as e:
        print(f"[log] Cleanup failed: {e}")
def _log_cleanup_loop():
    while True:
        time.sleep(3600)
        _cleanup_old_logs()


# =============================================================================
#  PRESENCE TIMEOUT WORKER
# =============================================================================
def _presence_timeout_loop():
    """
    Har 60 sec pe check karo — jo employee PRESENCE_TIMEOUT_MINUTES se
    zyada time se entry camera pe nahi dikhey → automatically Exit mark karo.
    """
    while True:
        time.sleep(60)
        try:
            now         = time.time()
            timeout_sec = PRESENCE_TIMEOUT_MINUTES * 60

            with _last_seen_lock:
                timed_out = [
                    name for name, ts in _last_seen.items()
                    if now - ts > timeout_sec
                ]

            for name in timed_out:
                current_status = db._get_effective_status(name)
                if current_status == "Present":
                    result = db.mark_exit(
                        name,
                        camera_id="TIMEOUT",
                        confidence=None,
                        photo_path=None,
                    )
                    if result == "marked":
                        print(
                            f"[timeout] ✅ {name} — {PRESENCE_TIMEOUT_MINUTES} min "
                            f"se nahi dikha → Exit marked "
                            f"({datetime.now().strftime('%H:%M:%S')})"
                        )
                with _last_seen_lock:
                    _last_seen.pop(name, None)

        except Exception as e:
            print(f"[timeout] error (non-fatal): {e}")


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
#  NAME CANONICALIZATION
#
#  portal.py photos ko sanitized folder name (_safe_name — spaces→underscore)
#  ke andar save karta hai, aur photo_sync.py wahi sanitized folder name
#  embeddings.pkl ki key bana deta hai. Yeh key kabhi Supabase employees.name
#  (canonical, jaisa manual attendance use karta hai) se match nahi karta —
#  isi wajah se camera recognition manual attendance se alag "person" ban
#  jaata tha, aur sync_roster() neeche stray/mismatched naam ko naya ghost
#  employee bana deta tha. Yahan har local pickle key ko canonical roster
#  naam se map karo, aur jo kisi active employee se match hi nahi karta use
#  drop kar do — kabhi bhi naya employee yahan se create nahi hoga.
# =============================================================================
def _canonicalize_embedding_names(data):
    """Convert stale/local embedding keys such as Rishu_Kumar to the
       canonical employee name stored in Supabase, e.g. Rishu Kumar."""
    if not data:
        return {}

    try:
        roster, _ = db.get_roster()
        canonical_by_safe = {_safe(name): name for name in roster}
    except Exception as e:
        print(f"[names] Supabase roster unavailable; keeping local names: {e}")
        return data

    if not canonical_by_safe:
        return data

    out = {}
    for raw_name, emb_list in data.items():
        canonical = canonical_by_safe.get(_safe(str(raw_name)))
        if canonical is not None:
            if canonical != raw_name:
                print(f"[names] Canonicalized: {raw_name!r} -> {canonical!r}")
            out.setdefault(canonical, []).extend(emb_list)
        else:
            print(f"[names] Ignoring unknown local embedding name: {raw_name!r}")

    return out


def _save_canonical_embeddings(data):
    try:
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE) or ".", exist_ok=True)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[names] Could not save canonical embeddings cache: {e}")


# =============================================================================
#  LOAD EMBEDDINGS
# =============================================================================
print("Loading embeddings …")
try:
    with open(EMBEDDINGS_FILE, "rb") as f:
        known_embeddings = pickle.load(f)
except Exception as e:
    print(f"[reload] embeddings.pkl read failed: {e}")
    known_embeddings = {}

known_embeddings = _canonicalize_embedding_names(known_embeddings)
_save_canonical_embeddings(known_embeddings)

known_face_db = {}
for name, emb_list in known_embeddings.items():
    if not emb_list:
        continue
    emb_array = np.array(emb_list, dtype=np.float32)
    mean_emb  = np.mean(emb_array, axis=0)
    norm      = np.linalg.norm(mean_emb)
    if norm == 0:
        continue
    mean_emb  = mean_emb / norm
    known_face_db[name] = mean_emb
    print(f"  {name}: {len(emb_list)} embeddings")

print(f"Total employees loaded: {len(known_face_db)}\n")

_known_names  = list(known_face_db.keys())
_known_matrix = (
    np.stack(list(known_face_db.values()), axis=0)   # (N, 512)
    if known_face_db else None
)

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
        _log_event_to_csv(name, camera_id, confidence, "already_present", "entry")
        return "already_present"

    # Photo save (fast — disk only, no network)
    photo_path = _save_capture(frame, bbox, name, camera_id)

    # db.mark_attendance() → SQLite + in-memory cache update
    result = db.mark_attendance(
        name, camera_id, confidence=confidence, photo_path=photo_path
    )

    db.log_detection(name, camera_id, confidence, result)
    _log_event_to_csv(name, camera_id, confidence, result, "entry")

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

                    if best_score >= THRESHOLD:
                        # Presence timer — chahe cooldown ho ya already_present,
                        # last_seen hamesha update (safety-net exit timeout)
                        with _last_seen_lock:
                            _last_seen[best_name] = time.time()

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
                        # 0.35 se kam score = pure noise, log mein spam nahi karna
                        if best_score > LOW_CONF_LOG_MIN:
                            _log_event_to_csv(
                                best_name, self.cam_id, best_score,
                                "low_confidence", "entry"
                            )

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

    data = _canonicalize_embedding_names(data)
    _save_canonical_embeddings(data)

    new_names  = []
    new_vecs   = []
    for name, emb_list in data.items():
        if not emb_list:
            continue
        arr      = np.array(emb_list, dtype=np.float32)
        mean_emb = np.mean(arr, axis=0)
        norm     = np.linalg.norm(mean_emb)
        if norm == 0:
            continue
        mean_emb = mean_emb / norm
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
       local pickle mein bhi backfill ho jaate hain (offline cache). Dono
       taraf ke naam canonical employee-name mein normalize kiye jaate hain
       — photo_sync.py sanitized folder-names (Rishu_Kumar) use karta hai,
       jo yahan asli Supabase naam (Rishu Kumar) se map/merge ho jaate hain."""
    global _known_matrix, _known_names, _embeddings_mtime

    # 1) Local pickle padho (agar hai) — baseline
    local_data = {}
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            local_data = pickle.load(f)
    except Exception:
        pass   # pehli baar ho sakta hai ki file hi na ho

    try:
        roster, _ = db.get_roster()
        canonical_by_safe = {_safe(name): name for name in roster}
    except Exception as e:
        print(f"[reload] Supabase roster fetch failed (non-fatal): {e}")
        canonical_by_safe = {}

    # Local keys ko canonical naam mein normalize karo (drop unknowns)
    canonical_local = {}
    for raw_name, emb_list in local_data.items():
        canonical = (
            canonical_by_safe.get(_safe(str(raw_name)))
            if canonical_by_safe else raw_name
        )
        if canonical is None:
            print(f"[reload] Ignoring stale local embedding: {raw_name!r}")
            continue
        canonical_local.setdefault(canonical, []).extend(emb_list)
    local_data = canonical_local

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
    for raw_name, raw_list in sb_raw.items():
        canonical = (
            canonical_by_safe.get(_safe(str(raw_name)), raw_name)
            if canonical_by_safe else raw_name
        )
        existing = local_data.setdefault(canonical, [])
        existing_set = {np.asarray(e, dtype=np.float32).tobytes() for e in existing}
        for raw in raw_list:
            vec = np.frombuffer(raw, dtype=np.float32)
            vb  = vec.tobytes()
            if vb not in existing_set:
                existing.append(vec)
                existing_set.add(vb)
                changed = True

    # 4) Local pickle ab canonical + merged hai — disk pe save karo agar
    #    naam ya vectors kuch bhi badla ho
    try:
        old_serialized     = pickle.dumps(local_data)
        current_serialized = None
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                current_serialized = f.read()
        except Exception:
            pass
        if current_serialized != old_serialized:
            os.makedirs(os.path.dirname(EMBEDDINGS_FILE) or ".", exist_ok=True)
            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump(local_data, f)
            changed = True
    except Exception as e:
        print(f"[reload] local pickle canonical save failed (non-fatal): {e}")

    if not local_data:
        return

    new_names, new_vecs = [], []
    for name, emb_list in local_data.items():
        if not emb_list:
            continue
        arr      = np.array(emb_list, dtype=np.float32)
        mean_emb = np.mean(arr, axis=0)
        norm     = np.linalg.norm(mean_emb)
        if norm == 0:
            continue
        mean_emb = mean_emb / norm
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

    if changed:
        print(f"[reload] ✅ Embeddings rebuilt with canonical employee names — "
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

    _reload_embeddings_merged()         # canonicalize names BEFORE roster sync
    db.sync_roster(_known_names)        # employees table sync — now safe, no ghosts

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

    _cleanup_old_logs()
    cleanup_t = threading.Thread(target=_log_cleanup_loop, daemon=True, name="log-cleanup")
    cleanup_t.start()
    print("[log] Log book cleanup started (every 1h, keeps 2 days)")

    # Presence timeout thread start karo — exit camera miss ho jaaye to bhi
    # PRESENCE_TIMEOUT_MINUTES ke baad automatically Exit mark ho jaayega
    timeout_t = threading.Thread(target=_presence_timeout_loop, daemon=True,
                                 name="presence-timeout")
    timeout_t.start()
    print(f"[timeout] Presence timeout thread started — {PRESENCE_TIMEOUT_MINUTES} min")

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
# ---------------------------------------------update---------------------------------------------
