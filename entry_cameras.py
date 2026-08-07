import cv2
import os
import time
import pickle
import uuid
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
THRESHOLD             = 0.45   # per-embedding max score
DETECTION_INTERVAL    = 10
TRACK_HOLD_FRAMES     = 20
MARK_COOLDOWN_SECONDS = config.MARK_COOLDOWN_SECONDS
TARGET_FPS            = 10
RELOAD_CHECK_INTERVAL = 10

# =============================================================================
#  RECOGNITION — all embeddings stored per person (not just mean)
# =============================================================================
# _known_all: { name: np.ndarray of shape (N, 512), L2-normalized }
_known_all       = {}
_embeddings_lock = threading.Lock()
_embeddings_mtime = 0.0


def _load_embeddings_from_file(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    result = {}
    for name, emb_list in raw.items():
        if name == "abc":          # dummy entry skip
            continue
        arr = np.array(emb_list, dtype=np.float32)
        # L2-normalize each row
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms
        result[name] = arr
    return result


def recognize_face(live_embedding: np.ndarray):
    """
    Compare live embedding against ALL stored embeddings for each person.
    Return (best_name, best_score) where score = max cosine similarity.
    Thread-safe via _embeddings_lock.
    """
    emb = live_embedding / (np.linalg.norm(live_embedding) + 1e-8)
    best_name  = "Unknown"
    best_score = 0.0
    with _embeddings_lock:
        for name, mat in _known_all.items():
            scores = mat @ emb          # shape (N,)
            s      = float(scores.max())
            if s > best_score:
                best_score = s
                best_name  = name
    return best_name, best_score


# =============================================================================
#  EMBEDDING WATCHER — live reload without restart
# =============================================================================
def _reload_embeddings():
    global _known_all, _embeddings_mtime
    try:
        new_data = _load_embeddings_from_file(EMBEDDINGS_FILE)
    except Exception as e:
        print(f"[reload] read failed: {e}")
        return
    with _embeddings_lock:
        _known_all        = new_data
        _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)
    print(f"[reload] ✅ {len(new_data)} employee(s) reloaded.")


def _watch_embeddings():
    global _embeddings_mtime
    try:
        _embeddings_mtime = os.path.getmtime(EMBEDDINGS_FILE)
    except Exception:
        pass
    while True:
        time.sleep(RELOAD_CHECK_INTERVAL)
        try:
            mt = os.path.getmtime(EMBEDDINGS_FILE)
            if mt != _embeddings_mtime:
                print("[reload] embeddings.pkl changed — reloading ...")
                _reload_embeddings()
        except Exception as e:
            print(f"[reload] watcher error: {e}")


# =============================================================================
#  PENDING PHOTOS  —  photos uploaded from a machine without a face model
#  (e.g. the Render admin panel) land in Postgres as raw bytes; this Jetson
#  polls for them, embeds them with the model it already has loaded, and
#  saves a local copy so the dashboard can serve it too.
# =============================================================================
PENDING_PHOTOS_POLL_INTERVAL = 30   # seconds


def _process_pending_photo(row):
    """One row from db.fetch_pending_photos(): decode, save locally, embed
       with the already-loaded face_app (GPU), append to embeddings.pkl +
       Supabase backup. Always marks the row processed when done — a bad/
       corrupt photo shouldn't retry forever."""
    name = row["emp_name"]
    try:
        arr = np.frombuffer(bytes(row["photo_bytes"]), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[pending-photo] decode failed for {name} (id={row['id']})")
            return

        # Local disk pe bhi save karo — dashboard/photo route ise serve kar sake
        ext      = os.path.splitext(row["filename"] or "")[1].lower() or ".jpg"
        safe     = _safe(name)
        abs_dir  = os.path.join(PHOTOS_DIR, safe)
        os.makedirs(abs_dir, exist_ok=True)
        prefix   = "profile" if row["is_profile"] else "remote"
        fname    = f"{prefix}_{uuid.uuid4().hex}{ext}"
        abs_path = os.path.join(abs_dir, fname)
        rel_path = f"{safe}/{fname}"
        cv2.imwrite(abs_path, img)

        faces = face_app.get(img)
        if not faces:
            print(f"[pending-photo] no face detected for {name} (id={row['id']})")
            return
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                   reverse=True)
        emb = faces[0].embedding.astype(np.float32)

        # embeddings.pkl mein append karo — same lock jo _reload_embeddings() use karta hai
        with _embeddings_lock:
            data = {}
            if os.path.exists(EMBEDDINGS_FILE):
                with open(EMBEDDINGS_FILE, "rb") as f:
                    data = pickle.load(f)
            data.setdefault(name, [])
            data[name].append(emb)
            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump(data, f)

        _reload_embeddings()   # in-memory _known_all turant update — 10s watcher ka wait mat karo

        # Supabase backup + (agar profile photo hai) photo_path update
        try:
            import auth_db
            auth_db.add_employee_embedding(row["employee_id"], emb.tobytes())
            if row["is_profile"]:
                auth_db.update_employee(row["employee_id"], photo_path=rel_path)
        except Exception as e:
            print(f"[pending-photo] Supabase backup failed (non-fatal): {e}")

        print(f"[pending-photo] ✅ Embedded {name} from remote upload (id={row['id']})")
    except Exception as e:
        print(f"[pending-photo] processing error for {name} (id={row['id']}): {e}")
    finally:
        db.mark_photo_processed(row["id"])


def _watch_pending_photos():
    while True:
        time.sleep(PENDING_PHOTOS_POLL_INTERVAL)
        try:
            rows = db.fetch_pending_photos()
            for row in rows:
                _process_pending_photo(row)
        except Exception as e:
            print(f"[pending-photo] watcher error: {e}")


# =============================================================================
#  STATUS CACHE
# =============================================================================
present_cache      = db.present_cache
present_cache_lock = db.present_cache_lock


def _refresh_status_cache_if_stale():
    db._mem_reset_if_new_day()


def _cached_status(name):
    return db._get_effective_status(name)


def _rebuild_cache_from_db():
    db._sync_present_cache_from_mem()


# =============================================================================
#  PHOTO CAPTURE
# =============================================================================
def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _save_capture(frame, bbox, name, cam_id):
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = (x2 - x1), (y2 - y1)
        pad_x        = int(bw * 0.9)
        pad_y_top    = int(bh * 0.35)
        pad_y_bottom = int(bh * 1.8)
        x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y_top)
        x2 = min(w, x2 + pad_x);  y2 = min(h, y2 + pad_y_bottom)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        TARGET_W, TARGET_H = 480, 600
        ch, cw = crop.shape[:2]
        if (cw / ch) > (TARGET_W / TARGET_H):
            need_h = int(cw / (TARGET_W / TARGET_H))
            extra  = max(0, need_h - ch)
            crop   = cv2.copyMakeBorder(crop, extra//2, extra-extra//2, 0, 0, cv2.BORDER_REPLICATE)
        else:
            need_w = int(ch * (TARGET_W / TARGET_H))
            extra  = max(0, need_w - cw)
            crop   = cv2.copyMakeBorder(crop, 0, 0, extra//2, extra-extra//2, cv2.BORDER_REPLICATE)
        crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
        crop = cv2.filter2D(crop, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        day_dir  = datetime.now().strftime("%Y-%m-%d")
        rel_dir  = os.path.join(_safe(name), day_dir)
        abs_dir  = os.path.join(PHOTOS_DIR, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)
        fname    = f"{cam_id}_{datetime.now().strftime('%H%M%S')}.jpg"
        rel_path = os.path.join(rel_dir, fname).replace("\\", "/")
        cv2.imwrite(os.path.join(abs_dir, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return rel_path
    except Exception as e:
        print(f"[photo] save failed for {name}: {e}")
        return None


# =============================================================================
#  FFMPEG / GSTREAMER
# =============================================================================
def _build_ffmpeg_url(url, user_id, user_pw):
    prefix = "rtsp://"
    body   = url[len(prefix):]
    if "@" in body.split("/")[0]:
        return url
    return f"{prefix}{user_id}:{quote(user_pw, safe='')}@{body}"


def _open_ffmpeg_cap(cam_id, ffmpeg_url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    print(f"[{cam_id}] Trying FFmpeg: {ffmpeg_url}")
    cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    print(f"[{cam_id}] {'✅ FFmpeg stream opened' if cap.isOpened() else '❌ FFmpeg failed'}")
    return cap


def open_capture_main_thread(cam_id, url, user_id, user_pw):
    gst = (
        f'rtspsrc location="{url}" user-id="{user_id}" user-pw="{user_pw}" '
        f'latency=100 protocols=tcp ! decodebin ! videoconvert ! videoscale ! '
        f'video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT},format=BGR ! '
        f'appsink drop=true max-buffers=1 sync=false'
    )
    print(f"[{cam_id}] Trying GStreamer:\n  {gst}")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[{cam_id}] ✅ GStreamer confirmed")
                return cap, frame
            time.sleep(0.05)
        cap.release()
        print(f"[{cam_id}] ⚠️  GStreamer opened but no frames — FFmpeg fallback")
    else:
        print(f"[{cam_id}] ⚠️  GStreamer failed — FFmpeg fallback")
    return _open_ffmpeg_cap(cam_id, _build_ffmpeg_url(url, user_id, user_pw)), None


# =============================================================================
#  ATTENDANCE MARKING
# =============================================================================
def mark_attendance(name, camera_id, frame, bbox, confidence):
    last_status = _cached_status(name)
    if last_status == "Present":
        db.log_detection(name, camera_id, confidence, "already_present")
        return "already_present"
    photo_path = _save_capture(frame, bbox, name, camera_id)
    result     = db.mark_attendance(name, camera_id, confidence=confidence, photo_path=photo_path)
    db.log_detection(name, camera_id, confidence, result)
    if result == "marked":
        action = "RE-ENTRY" if last_status == "Exit" else "FIRST ENTRY"
        print(f"[{action}] {name} via {camera_id} at {datetime.now().strftime('%H:%M:%S')} — SQLite saved")
    return result


# =============================================================================
#  CAMERA WORKER
# =============================================================================
class CameraWorker:

    def __init__(self, cfg, running):
        self.cam_id  = cfg["id"]
        self.url     = cfg["url"]
        self.user_id = cfg["user_id"]
        self.user_pw = cfg["user_pw"]
        self.running = running

        self.latest_frame = None
        self.frame_lock   = threading.Lock()
        self.frame_event  = threading.Event()

        self.last_mark_attempt = {}
        self.tracked_faces     = []
        self.track_lock        = threading.Lock()
        self._cap              = None

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
        return _open_ffmpeg_cap(self.cam_id, _build_ffmpeg_url(self.url, self.user_id, self.user_pw))

    def _capture_loop(self):
        cap = self._cap
        if not cap.isOpened():
            print(f"[{self.cam_id}] ERROR: stream open nahi hua.")
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

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True, name=f"cap-{self.cam_id}").start()
        threading.Thread(target=self._process_loop, daemon=True, name=f"proc-{self.cam_id}").start()

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

            frame_count += 1
            _refresh_status_cache_if_stale()

            if frame_count % DETECTION_INTERVAL == 0:
                from insightface.app import FaceAnalysis as _FA
                faces       = face_app.get(frame)
                new_tracked = []
                print(f"[{self.cam_id}] frame={frame_count} faces={len(faces)}", flush=True)

                for face in faces:
                    x1, y1, x2, y2       = map(int, face.bbox)
                    best_name, best_score = recognize_face(face.embedding)
                    print(f"[{self.cam_id}] {best_name} score={best_score:.3f}", flush=True)

                    if best_score >= THRESHOLD:
                        now      = time.time()
                        last_try = self.last_mark_attempt.get(best_name, 0)
                        if now - last_try > MARK_COOLDOWN_SECONDS:
                            result = mark_attendance(best_name, self.cam_id, frame,
                                                     (x1, y1, x2, y2), best_score)
                            self.last_mark_attempt[best_name] = now
                        else:
                            result = "cooldown"

                        if result == "marked":
                            rows_today = db.count_today_rows(best_name)
                            label = f"{best_name} - {'Re-Entry' if rows_today > 1 else 'Present'} ({best_score:.2f})"
                            color = (0, 255, 0)
                        elif result == "already_present":
                            label = f"{best_name} - Already Present ({best_score:.2f})"
                            color = (0, 255, 255)
                        elif result == "cooldown":
                            label = f"{best_name} ({best_score:.2f}) [cooldown]"
                            color = (255, 200, 0)
                        else:
                            label = f"{best_name} ({best_score:.2f})"
                            color = (255, 200, 0)
                    else:
                        label = f"Unknown ({best_score:.2f})"
                        color = (0, 0, 255)

                    new_tracked.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                        "label": label, "color": color, "hold": TRACK_HOLD_FRAMES})

                with self.track_lock:
                    self.tracked_faces = new_tracked

            with self.track_lock:
                self.tracked_faces = [f for f in self.tracked_faces if f["hold"] > 0]
                for f in self.tracked_faces:
                    f["hold"] -= 1

            elapsed = time.time() - loop_start
            if frame_time - elapsed > 0:
                time.sleep(frame_time - elapsed)


# =============================================================================
#  MAIN
# =============================================================================
# Load face model globally
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# Load embeddings
print("Loading embeddings …")
_known_all = _load_embeddings_from_file(EMBEDDINGS_FILE)
for name, mat in _known_all.items():
    print(f"  {name}: {len(mat)} embeddings")
print(f"Total employees loaded: {len(_known_all)}\n")


def main():
    db.init_db()
    db.sync_roster(list(_known_all.keys()))
    n = db.sync_employee_photos_from_dir()
    if n:
        print(f"[photos] {n} profile photo(s) linked")
    _rebuild_cache_from_db()

    threading.Thread(target=_watch_embeddings, daemon=True, name="embeddings-watcher").start()
    print(f"[reload] Embeddings watcher started (check every {RELOAD_CHECK_INTERVAL}s)")

    threading.Thread(target=_watch_pending_photos, daemon=True, name="pending-photos-watcher").start()
    print(f"[pending-photo] Watcher started (check every {PENDING_PHOTOS_POLL_INTERVAL}s) — "
          f"picks up photos uploaded from machines without a face model")

    running = {"value": True}
    workers = [CameraWorker(cfg, running) for cfg in RTSP_CAMERAS]

    for w in workers:
        w.open()
    for w in workers:
        w.start()

    print("[main] Headless mode — cameras running in background")

    while running["value"]:
        time.sleep(0.5)

    print("All cameras stopped.")
    time.sleep(1)
    os._exit(0)


if __name__ == "__main__":
    main()