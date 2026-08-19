"""
=============================================================================
  db.py  —  Postgres (Supabase) access + offline-safe local outbox.

  entry_cameras.py / exit_cameras.py / server.py import this module exactly
  as before — NO CHANGES NEEDED in those files. Function names and return
  values are identical to the old version.

  WHAT CHANGED AND WHY
  ---------------------------------------------------------------------------
  Old bug: mark_attendance()/mark_exit() called the network FIRST and only
  computed `now = datetime.now()` AFTER those calls returned. When Supabase
  was unreachable, psycopg2 would hang (no connect_timeout / statement_timeout
  set), so the function only resumed once internet came back — and by then
  `now` was captured at THAT moment, not at the moment the face was actually
  detected. That's why entry/exit times were wrong after an outage.

  Fix:
    1. Timestamp is captured as the very first line of mark_attendance/mark_exit,
       before any DB call.
    2. Every DB connection now has connect_timeout + statement_timeout +
       TCP keepalives, so a dead network fails in ~5-10s instead of hanging.
    3. If the Postgres write fails for ANY reason, the row (with the correct
       timestamp) is written to a local SQLite outbox instead, and the
       function still returns "marked" — attendance keeps working offline.
    4. A background thread (started automatically on import) drains the
       outbox into Supabase every SYNC_INTERVAL_SECONDS once connectivity
       returns, oldest rows first, so timestamps stay correct.
    5. Status reads (get_last_status_today / read_day_rows / count_today_rows)
       also fall back to the local outbox when Postgres is unreachable, so
       duplicate-entry checks keep working during an outage.

  ONE-TIME SUPABASE MIGRATION (recommended, not required):
      ALTER TABLE attendance ADD COLUMN IF NOT EXISTS client_uuid TEXT UNIQUE;
  This lets the sync worker safely retry without ever creating a duplicate
  row if it crashes mid-sync. The code works without it too (falls back to
  a name+date+time+camera duplicate check), just slightly less bullet-proof.
=============================================================================
"""

# import os
# import sqlite3
# import threading
# import time
# import uuid
# from datetime import datetime, date

# import psycopg2
# from psycopg2 import pool as pg_pool
# from psycopg2.extras import RealDictCursor

# import config

# # =============================================================================
# #  TUNABLES
# # =============================================================================
# CONNECT_TIMEOUT_SECONDS   = 5      # fail fast instead of hanging on a dead network
# STATEMENT_TIMEOUT_MS      = 5000   # kill a query server-side if it takes >5s
# SYNC_INTERVAL_SECONDS     = 15     # how often the background worker retries
# LOCAL_DB_PATH             = getattr(config, "LOCAL_QUEUE_DB", "offline_queue.db")

# # =============================================================================
# #  CONNECTION POOL  (now with real timeouts)
# # =============================================================================
# _pool = None
# _pool_lock = threading.Lock()


# def get_pool():
#     global _pool
#     if _pool is None:
#         with _pool_lock:
#             if _pool is None:
#                 _pool = pg_pool.ThreadedConnectionPool(
#                     minconn=1,
#                     maxconn=8,
#                     dsn=config.DATABASE_URL,
#                     connect_timeout=CONNECT_TIMEOUT_SECONDS,
#                     keepalives=1,
#                     keepalives_idle=5,
#                     keepalives_interval=3,
#                     keepalives_count=2,
#                     options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
#                 )
#     return _pool


# class _ConnCtx:
#     """Small wrapper so existing code (`conn = get_conn() ... conn.close()`)
#        keeps working unchanged, while actually returning the connection to
#        the pool instead of really closing the socket."""

#     def __init__(self, raw_conn):
#         self._raw = raw_conn

#     def cursor(self, *args, **kwargs):
#         kwargs.setdefault("cursor_factory", RealDictCursor)
#         return self._raw.cursor(*args, **kwargs)

#     def commit(self):
#         self._raw.commit()

#     def rollback(self):
#         self._raw.rollback()

#     def close(self):
#         get_pool().putconn(self._raw)


# def get_conn():
#     raw = get_pool().getconn()
#     return _ConnCtx(raw)


# def init_db():
#     """Tables banti hain schema.sql se. Yahan sirf connectivity check karte
#        hain taaki startup pe galat DATABASE_URL ka pata turant chal jaaye.
#        Ab yeh startup pe hang nahi karega — 5s me fail ho jayega agar DGX
#        offline hai, aur local outbox already ready hoga."""
#     _local_init()
#     _local_log_init()
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute("SELECT 1")
#             cur.fetchone()
#             cur.close()
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] WARNING: could not reach Supabase at startup ({e}). "
#               f"Running in offline mode — events will queue locally and "
#               f"sync automatically once internet is back.")
#     _start_sync_worker()


# # =============================================================================
# #  LOCAL OUTBOX  (SQLite — always available, never blocks on network)
# # =============================================================================
# _local_lock = threading.Lock()


# def _local_conn():
#     conn = sqlite3.connect(LOCAL_DB_PATH, check_same_thread=False, timeout=10)
#     conn.row_factory = sqlite3.Row
#     return conn


# def _local_init():
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             conn.execute("""
#                 CREATE TABLE IF NOT EXISTS pending_attendance (
#                     id          INTEGER PRIMARY KEY AUTOINCREMENT,
#                     client_uuid TEXT UNIQUE,
#                     name        TEXT NOT NULL,
#                     att_date    TEXT NOT NULL,
#                     att_time    TEXT NOT NULL,
#                     status      TEXT NOT NULL,
#                     camera_id   TEXT,
#                     confidence  REAL,
#                     photo_path  TEXT,
#                     created_at  TEXT NOT NULL,
#                     retry_count INTEGER NOT NULL DEFAULT 0
#                 )
#             """)
#             conn.commit()
#         finally:
#             conn.close()


# def _local_insert(name, att_date, att_time, status, camera_id, confidence, photo_path):
#     row_uuid = str(uuid.uuid4())
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             conn.execute(
#                 """INSERT INTO pending_attendance
#                    (client_uuid, name, att_date, att_time, status, camera_id,
#                     confidence, photo_path, created_at)
#                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
#                 (row_uuid, name, att_date, att_time, status, camera_id,
#                  confidence, photo_path, datetime.now().isoformat()),
#             )
#             conn.commit()
#         finally:
#             conn.close()
#     print(f"[db] OFFLINE — queued locally: {name} {status} at {att_time} "
#           f"(will sync when internet returns)")
#     return row_uuid


# def _local_last_status_today(name):
#     """Most recent status for `name` today, considering ONLY unsynced local
#        rows (already-synced rows are covered by the normal Postgres read
#        when it's reachable; this is purely the offline fallback)."""
#     today = date.today().isoformat()
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             row = conn.execute(
#                 """SELECT status FROM pending_attendance
#                    WHERE name = ? AND att_date = ?
#                    ORDER BY id DESC LIMIT 1""",
#                 (name, today),
#             ).fetchone()
#             return row["status"] if row else None
#         finally:
#             conn.close()


# def _local_count_today(name):
#     today = date.today().isoformat()
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             row = conn.execute(
#                 "SELECT COUNT(*) AS c FROM pending_attendance WHERE name = ? AND att_date = ?",
#                 (name, today),
#             ).fetchone()
#             return row["c"]
#         finally:
#             conn.close()


# def _local_day_rows(target_date):
#     """target_date: date object or 'YYYY-MM-DD' string."""
#     td = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             rows = conn.execute(
#                 """SELECT name, att_time AS time, status, camera_id AS camera,
#                           photo_path AS photo
#                    FROM pending_attendance WHERE att_date = ?
#                    ORDER BY att_time DESC""",
#                 (td,),
#             ).fetchall()
#             return [dict(r) for r in rows]
#         finally:
#             conn.close()


# def _local_pending_rows():
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             rows = conn.execute(
#                 "SELECT * FROM pending_attendance ORDER BY id ASC"
#             ).fetchall()
#             return [dict(r) for r in rows]
#         finally:
#             conn.close()


# def _local_delete(row_id):
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             conn.execute("DELETE FROM pending_attendance WHERE id = ?", (row_id,))
#             conn.commit()
#         finally:
#             conn.close()


# def _local_bump_retry(row_id):
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             conn.execute(
#                 "UPDATE pending_attendance SET retry_count = retry_count + 1 WHERE id = ?",
#                 (row_id,),
#             )
#             conn.commit()
#         finally:
#             conn.close()


# def pending_sync_count():
#     """Handy to show on the dashboard: how many events are waiting to sync."""
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             return conn.execute("SELECT COUNT(*) AS c FROM pending_attendance").fetchone()["c"]
#         finally:
#             conn.close()


# # =============================================================================
# #  DETECTION AUDIT LOG  —  every recognition event, regardless of outcome.
# #  This is a PURE LOCAL record (never synced to Supabase) whose only job is
# #  to give you an exact, independently-verifiable timestamp for every face
# #  the camera recognized — including "already_present" hits that never touch
# #  mark_attendance()/mark_exit() at all. Use this to audit real-world timing
# #  accuracy against what actually shows up in Supabase.
# # =============================================================================
# def _local_log_init():
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             conn.execute("""
#                 CREATE TABLE IF NOT EXISTS detection_log (
#                     id          INTEGER PRIMARY KEY AUTOINCREMENT,
#                     name        TEXT NOT NULL,
#                     camera_id   TEXT,
#                     confidence  REAL,
#                     result      TEXT NOT NULL,
#                     event_time  TEXT NOT NULL
#                 )
#             """)
#             conn.commit()
#         finally:
#             conn.close()


# def log_detection(name, camera_id, confidence, result):
#     """Call this on EVERY recognized face above threshold, no matter what
#        happens next. Captures the exact moment of detection, independent of
#        any network/DB call, so you always have ground truth to check against."""
#     event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#     with _local_lock:
#         conn = _local_conn()
#         try:
#             conn.execute(
#                 """INSERT INTO detection_log (name, camera_id, confidence, result, event_time)
#                    VALUES (?, ?, ?, ?, ?)""",
#                 (name, camera_id, confidence, result, event_time),
#             )
#             conn.commit()
#         finally:
#             conn.close()


# # =============================================================================
# #  EMPLOYEE ROSTER
# # =============================================================================
# def get_or_create_employee(name, department=None, designation=None):
#     """Roster me employee dhoondo, na ho to bana do. Returns employee_id.
#        Offline hone par None return karta hai — sync worker isse baad me
#        resolve kar lega, isliye entry/exit marking abhi bhi block nahi hoti."""
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute("SELECT id FROM employees WHERE name = %s", (name,))
#             row = cur.fetchone()
#             if row:
#                 return row["id"]
#             cur.execute(
#                 """INSERT INTO employees (name, department, designation)
#                    VALUES (%s, %s, %s) RETURNING id""",
#                 (name, department, designation),
#             )
#             emp_id = cur.fetchone()["id"]
#             conn.commit()
#             cur.close()
#             return emp_id
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] get_or_create_employee offline fallback for {name}: {e}")
#         return None


# def sync_roster(names):
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             for name in names:
#                 cur.execute(
#                     "INSERT INTO employees (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
#                     (name,),
#                 )
#             conn.commit()
#             cur.close()
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] sync_roster skipped (offline): {e}")


# # =============================================================================
# #  STATUS READ  (Postgres, falls back to local outbox)
# # =============================================================================
# def get_last_status_today(name):
#     """Aaj is naam ka sabse recent status ('Present'/'Exit') ya None.
#        Postgres unreachable ho to local outbox se best-effort answer deta hai."""
#     today = date.today()
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute(
#                 """SELECT status FROM attendance
#                    WHERE name = %s AND att_date = %s
#                    ORDER BY att_time DESC, id DESC LIMIT 1""",
#                 (name, today),
#             )
#             row = cur.fetchone()
#             cur.close()
#             pg_status = row["status"] if row else None
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] get_last_status_today offline fallback for {name}: {e}")
#         return _local_last_status_today(name)

#     # Even when Postgres IS reachable, there may be local rows written during
#     # a recent outage that haven't synced yet — those are more recent truth.
#     local_status = _local_last_status_today(name)
#     return local_status if local_status is not None else pg_status


# # =============================================================================
# #  ATTENDANCE MARKING  — timestamp captured FIRST, network is best-effort
# # =============================================================================
# def mark_attendance(name, camera_id, confidence=None, photo_path=None,
#                      department=None):
#     """
#     Duplicate-prevention wali attendance mark:
#       - Agar aaj already "Present" hai -> "already_present", kuch nahi likhta.
#       - Warna naya row insert (Present ya, agar pehle Exit tha, to Re-Entry
#         bhi "Present" hi likha jaata hai).
#     Returns "marked" | "already_present".

#     CRITICAL: `now` is captured before any network call, so the recorded
#     time is always the moment the face was actually detected — even if
#     Supabase is unreachable and the row has to queue locally.
#     """
#     now = datetime.now()

#     last_status = get_last_status_today(name)
#     if last_status == "Present":
#         return "already_present"

#     try:
#         emp_id = get_or_create_employee(name, department=department)
#         if emp_id is None:
#             raise RuntimeError("no employee_id (offline)")

#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute(
#                 """INSERT INTO attendance
#                    (employee_id, name, att_date, att_time, status, camera_id,
#                     confidence, photo_path)
#                    VALUES (%s, %s, %s, %s, 'Present', %s, %s, %s)""",
#                 (emp_id, name, now.date(), now.time().strftime("%H:%M:%S"),
#                  camera_id, confidence, photo_path),
#             )
#             conn.commit()
#             cur.close()
#         finally:
#             conn.close()
#         return "marked"

#     except Exception as e:
#         print(f"[db] mark_attendance falling back to local outbox for {name}: {e}")
#         _local_insert(
#             name, now.date().isoformat(), now.time().strftime("%H:%M:%S"),
#             "Present", camera_id, confidence, photo_path,
#         )
#         return "marked"


# def mark_exit(name, camera_id, confidence=None, photo_path=None):
#     """Mirrors mark_attendance() — timestamp first, network best-effort,
#        falls back to local outbox on any failure."""
#     now = datetime.now()

#     last_status = get_last_status_today(name)
#     if last_status != "Present":
#         return "not_inside"

#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute("SELECT id FROM employees WHERE name = %s", (name,))
#             row = cur.fetchone()
#             emp_id = row["id"] if row else get_or_create_employee(name)
#             if emp_id is None:
#                 raise RuntimeError("no employee_id (offline)")

#             cur.execute(
#                 """INSERT INTO attendance
#                    (employee_id, name, att_date, att_time, status, camera_id,
#                     confidence, photo_path)
#                    VALUES (%s, %s, %s, %s, 'Exit', %s, %s, %s)""",
#                 (emp_id, name, now.date(), now.time().strftime("%H:%M:%S"),
#                  camera_id, confidence, photo_path),
#             )
#             conn.commit()
#             cur.close()
#         finally:
#             conn.close()
#         return "marked"

#     except Exception as e:
#         print(f"[db] mark_exit falling back to local outbox for {name}: {e}")
#         _local_insert(
#             name, now.date().isoformat(), now.time().strftime("%H:%M:%S"),
#             "Exit", camera_id, confidence, photo_path,
#         )
#         return "marked"


# def count_today_rows(name):
#     today = date.today()
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute(
#                 "SELECT COUNT(*) AS c FROM attendance WHERE name = %s AND att_date = %s",
#                 (name, today),
#             )
#             n = cur.fetchone()["c"]
#             cur.close()
#             return n + _local_count_today(name)
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] count_today_rows offline fallback for {name}: {e}")
#         return _local_count_today(name)


# # =============================================================================
# #  READ HELPERS  —  used by the Flask API / status cache refresh
# # =============================================================================
# def _dictify(cur):
#     return [dict(row) for row in cur.fetchall()]


# def read_day_rows(target_date):
#     """target_date: date object or 'YYYY-MM-DD' string. Falls back to the
#        local outbox (merged in even when online, since those rows haven't
#        synced yet) so the in-process status cache never crashes or goes
#        stale during an outage."""
#     local_rows = _local_day_rows(target_date)
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             cur.execute(
#                 """SELECT name, att_time AS time, status, camera_id AS camera,
#                           photo_path AS photo
#                    FROM attendance WHERE att_date = %s
#                    ORDER BY att_time DESC""",
#                 (target_date,),
#             )
#             rows = _dictify(cur)
#             cur.close()
#             for r in rows:
#                 r["time"] = str(r["time"])
#             return local_rows + rows
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] read_day_rows offline fallback: {e}")
#         return local_rows


# def read_month_rows(month):
#     """month = 'YYYY-MM'."""
#     conn = get_conn()
#     try:
#         cur = conn.cursor()
#         cur.execute(
#             """SELECT name, att_date AS date, att_time AS time, status,
#                       camera_id AS camera, photo_path AS photo
#                FROM attendance WHERE TO_CHAR(att_date, 'YYYY-MM') = %s
#                ORDER BY att_time ASC""",
#             (month,),
#         )
#         rows = _dictify(cur)
#         cur.close()
#         for r in rows:
#             r["date"] = str(r["date"])
#             r["time"] = str(r["time"])
#         return rows
#     finally:
#         conn.close()


# def all_dates():
#     conn = get_conn()
#     try:
#         cur = conn.cursor()
#         cur.execute("SELECT DISTINCT att_date FROM attendance ORDER BY att_date DESC")
#         dates = [str(r["att_date"]) for r in cur.fetchall()]
#         cur.close()
#         return dates
#     finally:
#         conn.close()


# def available_months():
#     conn = get_conn()
#     try:
#         cur = conn.cursor()
#         cur.execute(
#             "SELECT DISTINCT TO_CHAR(att_date, 'YYYY-MM') AS m FROM attendance "
#             "ORDER BY 1 DESC"
#         )
#         months = [r["m"] for r in cur.fetchall()]
#         cur.close()
#         return months
#     finally:
#         conn.close()


# def get_roster():
#     """Returns (names_list, source='employees')."""
#     conn = get_conn()
#     try:
#         cur = conn.cursor()
#         cur.execute("SELECT name FROM employees WHERE active = TRUE ORDER BY name")
#         names = [r["name"] for r in cur.fetchall()]
#         cur.close()
#         return names, "employees"
#     finally:
#         conn.close()


# def get_employee_photo(name):
#     conn = get_conn()
#     try:
#         cur = conn.cursor()
#         cur.execute("SELECT photo_path FROM employees WHERE name = %s", (name,))
#         row = cur.fetchone()
#         cur.close()
#         return row["photo_path"] if row and row["photo_path"] else None
#     finally:
#         conn.close()


# def sync_employee_photos_from_dir():
#     """Offline-safe: if Supabase is unreachable at startup, this just skips
#        and returns 0 instead of crashing main() (which used to trigger a
#        systemd crash-restart loop when the DGX had no internet at boot)."""
#     import os as _os

#     IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
#     if not _os.path.isdir(config.PHOTOS_DIR):
#         return 0

#     updated = 0
#     try:
#         conn = get_conn()
#         try:
#             cur = conn.cursor()
#             for entry in sorted(_os.listdir(config.PHOTOS_DIR)):
#                 emp_dir = _os.path.join(config.PHOTOS_DIR, entry)
#                 if not _os.path.isdir(emp_dir):
#                     continue

#                 photo_file = None
#                 for fname in sorted(_os.listdir(emp_dir)):
#                     fpath = _os.path.join(emp_dir, fname)
#                     if _os.path.isfile(fpath) and fname.lower().endswith(IMG_EXT):
#                         photo_file = fname
#                         break
#                 if not photo_file:
#                     continue

#                 rel_path = f"{entry}/{photo_file}"

#                 cur.execute("SELECT id, photo_path FROM employees WHERE name = %s", (entry,))
#                 row = cur.fetchone()
#                 if row is None:
#                     cur.execute(
#                         "INSERT INTO employees (name, photo_path) VALUES (%s, %s)",
#                         (entry, rel_path),
#                     )
#                     updated += 1
#                 elif not row["photo_path"]:
#                     cur.execute(
#                         "UPDATE employees SET photo_path = %s WHERE id = %s",
#                         (rel_path, row["id"]),
#                     )
#                     updated += 1
#             conn.commit()
#             cur.close()
#         finally:
#             conn.close()
#     except Exception as e:
#         print(f"[db] sync_employee_photos_from_dir skipped (offline): {e}")
#         return 0
#     return updated


# def list_employees():
#     conn = get_conn()
#     try:
#         cur = conn.cursor()
#         cur.execute(
#             """SELECT id, name, department, designation, photo_path, active
#                FROM employees ORDER BY name"""
#         )
#         rows = _dictify(cur)
#         cur.close()
#         return rows
#     finally:
#         conn.close()


# # =============================================================================
# #  BACKGROUND SYNC WORKER  —  drains the local outbox into Supabase
# # =============================================================================
# _sync_thread_started = False
# _sync_thread_lock = threading.Lock()


# def _has_client_uuid_column(cur):
#     cur.execute(
#         """SELECT 1 FROM information_schema.columns
#            WHERE table_name='attendance' AND column_name='client_uuid'"""
#     )
#     return cur.fetchone() is not None


# def _sync_one_row(cur, row, has_uuid_col):
#     emp_id = get_or_create_employee(row["name"])
#     if emp_id is None:
#         raise RuntimeError("still offline")

#     if has_uuid_col:
#         cur.execute(
#             """INSERT INTO attendance
#                (employee_id, name, att_date, att_time, status, camera_id,
#                 confidence, photo_path, client_uuid)
#                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#                ON CONFLICT (client_uuid) DO NOTHING""",
#             (emp_id, row["name"], row["att_date"], row["att_time"], row["status"],
#              row["camera_id"], row["confidence"], row["photo_path"], row["client_uuid"]),
#         )
#     else:
#         # best-effort de-dup without the migration: skip if an identical
#         # row already exists (same name/date/time/camera).
#         cur.execute(
#             """SELECT 1 FROM attendance
#                WHERE name=%s AND att_date=%s AND att_time=%s AND camera_id=%s""",
#             (row["name"], row["att_date"], row["att_time"], row["camera_id"]),
#         )
#         if cur.fetchone() is None:
#             cur.execute(
#                 """INSERT INTO attendance
#                    (employee_id, name, att_date, att_time, status, camera_id,
#                     confidence, photo_path)
#                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
#                 (emp_id, row["name"], row["att_date"], row["att_time"], row["status"],
#                  row["camera_id"], row["confidence"], row["photo_path"]),
#             )


# def _sync_loop():
#     while True:
#         time.sleep(SYNC_INTERVAL_SECONDS)
#         pending = _local_pending_rows()
#         if not pending:
#             continue

#         try:
#             conn = get_conn()
#         except Exception:
#             continue  # still offline, try again next cycle

#         try:
#             cur = conn.cursor()
#             has_uuid_col = _has_client_uuid_column(cur)

#             synced = 0
#             for row in pending:
#                 try:
#                     _sync_one_row(cur, row, has_uuid_col)
#                     conn.commit()
#                     _local_delete(row["id"])
#                     synced += 1
#                 except Exception as e:
#                     conn.rollback()
#                     _local_bump_retry(row["id"])
#                     print(f"[db] sync retry failed for {row['name']} "
#                           f"({row['att_date']} {row['att_time']}): {e}")

#             cur.close()
#             if synced:
#                 print(f"[db] SYNC OK — {synced} offline event(s) pushed to Supabase.")
#         finally:
#             conn.close()


# def _start_sync_worker():
#     global _sync_thread_started
#     with _sync_thread_lock:
#         if _sync_thread_started:
#             return
#         _sync_thread_started = True
#         t = threading.Thread(target=_sync_loop, daemon=True, name="supabase-sync-worker")
#         t.start()
#         print(f"[db] background sync worker started "
#               f"(retries every {SYNC_INTERVAL_SECONDS}s, local queue: {LOCAL_DB_PATH})")




#-----------------------------------NEW CODE-------------------------------------------------------------------------------
"""
=============================================================================
  db.py  —  SQLite-FIRST architecture + smart Supabase sync

  FLOW:
    1. Har face detection → HAMESHA SQLite mein save (synced=0)
    2. In-memory cache turant update (camera display real-time rehta hai)
    3. Background worker har 5 min → SQLite pending rows → Supabase smart push
    4. Smart push logic:
         - NULL      → Present/Exit insert
         - Present   → same status aaya? SKIP | Exit aaya? insert Exit
         - Exit      → Present aaya? insert (re-entry) | same Exit? SKIP
         - Duplicate consecutive same-status within 5 min → DEDUP SKIP
    5. synced=1 mark hoti hai successfully pushed rows
    6. 7 din baad synced=1 rows purge hoti hain SQLite se

  entry_cameras.py / exit_cameras.py / server.py mein koi changes nahi —
  function signatures same hain.
=============================================================================
"""
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, date, timedelta

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

import config

# =============================================================================
#  TUNABLES
# =============================================================================
CONNECT_TIMEOUT_SECONDS = 5
STATEMENT_TIMEOUT_MS    = 5000
SYNC_INTERVAL_SECONDS   = 300     # 5 min
PURGE_AFTER_DAYS        = 7       # 7 din baad synced rows delete
PURGE_CHECK_INTERVAL    = 12      # har 12 cycles (~1 ghanta)
DEDUP_WINDOW_SECONDS    = 300     # 5 min mein same person same status → skip

_DEFAULT_LOCAL_DB = "/home/cyamsys/Desktop/test_Attendance/HR_attendance/offline_queue.db"
LOCAL_DB_PATH     = getattr(config, "LOCAL_QUEUE_DB", _DEFAULT_LOCAL_DB)

# =============================================================================
#  IN-MEMORY STATUS CACHE
# =============================================================================
_mem_cache      = {}
_mem_cache_lock = threading.Lock()
_cache_date     = None

# Guards the check-then-write in mark_attendance()/mark_exit() — without
# this, two cameras (or two duplicate camera processes) recognizing the same
# person within the same instant could both pass the "not already marked"
# check before either writes, producing duplicate Present/Exit rows.
_mark_lock = threading.Lock()


def _mem_get(name: str):
    with _mem_cache_lock:
        return _mem_cache.get(name, None)


def _mem_set(name: str, status: str):
    with _mem_cache_lock:
        _mem_cache[name] = status


def _mem_reset_if_new_day():
    global _cache_date
    today = date.today().isoformat()
    with _mem_cache_lock:
        if _cache_date != today:
            _cache_date = today
            _mem_cache.clear()
            print(f"[cache] Naya din {today} — in-memory cache reset.")


def _mem_seed_from_local():
    """
    Startup pe SQLite ke aaj ke UNSYNCED rows se cache seed karo.

    BUG FIX: pehle _local_day_rows() use karta tha jo synced=1 rows bhi
    return karta tha. Agar koi employee aaj Supabase mein synced ho chuka
    tha (synced=1) aur SQLite mein Entry tha, toh cache mein galat status
    aa jaata tha. Ab sirf synced=0 rows se seed karo — jo pending hain.
    Supabase se baki status _get_effective_status() mein fallback se milega.
    """
    global _cache_date
    today = date.today().isoformat()
    # sirf synced=0 rows — pending wali
    rows  = _local_day_rows(today)   # ab yeh sirf synced=0 return karta hai
    seen  = set()
    with _mem_cache_lock:
        for r in rows:
            if r["name"] not in seen:
                _mem_cache[r["name"]] = r["status"]
                seen.add(r["name"])
        _cache_date = today
    print(f"[cache] Startup: {len(seen)} employee(s) ka status SQLite (pending) se load kiya.")


# =============================================================================
#  PRESENT CACHE  —  entry_cameras.py ka "Inside Now" overlay
# =============================================================================
present_cache      = set()
present_cache_lock = threading.Lock()


def _sync_present_cache_from_mem():
    with _mem_cache_lock:
        snapshot = dict(_mem_cache)
    with present_cache_lock:
        present_cache.clear()
        for name, status in snapshot.items():
            if status == "Present":
                present_cache.add(name)


# =============================================================================
#  CONNECTION POOL
# =============================================================================
_pool      = None
_pool_lock = threading.Lock()


def get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=8,
                    dsn=config.DATABASE_URL,
                    connect_timeout=CONNECT_TIMEOUT_SECONDS,
                    keepalives=1,
                    keepalives_idle=5,
                    keepalives_interval=3,
                    keepalives_count=2,
                    options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
                )
    return _pool


class _ConnCtx:
    def __init__(self, raw_conn):
        self._raw = raw_conn

    def cursor(self, *args, **kwargs):
        kwargs.setdefault("cursor_factory", RealDictCursor)
        return self._raw.cursor(*args, **kwargs)

    def commit(self):   self._raw.commit()
    def rollback(self): self._raw.rollback()

    def close(self):
        get_pool().putconn(self._raw)


def get_conn():
    raw = get_pool().getconn()
    return _ConnCtx(raw)


def init_db():
    _local_init()
    _local_log_init()
    _mem_seed_from_local()
    _sync_present_cache_from_mem()

    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            print("[db] Supabase connection OK.")
        finally:
            conn.close()
    except Exception as e:
        print(
            f"[db] WARNING: Supabase unreachable at startup ({e}).\n"
            f"     Offline mode — events SQLite mein queue honge:\n"
            f"     {LOCAL_DB_PATH}"
        )
    _start_sync_worker()


# =============================================================================
#  LOCAL SQLite OUTBOX
# =============================================================================
_local_lock = threading.Lock()


def _local_conn():
    os.makedirs(os.path.dirname(LOCAL_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(LOCAL_DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _local_init():
    with _local_lock:
        conn = _local_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_attendance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_uuid TEXT UNIQUE,
                    name        TEXT NOT NULL,
                    att_date    TEXT NOT NULL,
                    att_time    TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    camera_id   TEXT,
                    confidence  REAL,
                    photo_path  TEXT,
                    created_at  TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    synced      INTEGER NOT NULL DEFAULT 0
                )
            """)
            try:
                conn.execute(
                    "ALTER TABLE pending_attendance "
                    "ADD COLUMN synced INTEGER NOT NULL DEFAULT 0"
                )
            except Exception:
                pass
            conn.commit()
        finally:
            conn.close()


def _local_insert(name, att_date, att_time, status, camera_id, confidence, photo_path):
    row_uuid = str(uuid.uuid4())
    with _local_lock:
        conn = _local_conn()
        try:
            conn.execute(
                """INSERT INTO pending_attendance
                   (client_uuid, name, att_date, att_time, status, camera_id,
                    confidence, photo_path, created_at, synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (row_uuid, name, att_date, att_time, status,
                 camera_id, confidence, photo_path,
                 datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
    print(f"[SQLite] Saved → {name} | {status} | {att_time} | {camera_id}")
    return row_uuid


def _local_mark_synced(row_id):
    with _local_lock:
        conn = _local_conn()
        try:
            conn.execute(
                "UPDATE pending_attendance SET synced = 1 WHERE id = ?", (row_id,)
            )
            conn.commit()
        finally:
            conn.close()


def _local_day_rows(target_date):
    """
    BUG FIX: Sirf synced=0 rows return karo.

    Pehle yahan synced filter nahi tha — matlab synced=1 rows bhi return
    hoti thi jo Supabase mein already hain. read_day_rows() mein jab
    SQLite + Supabase merge hota tha toh DOUBLE ENTRY ho jaati thi.

    Ab sirf pending (synced=0) rows return hoti hain — jo abhi Supabase
    nahi gayi. Supabase wali rows pg_rows se aayengi aur dedup se handle
    ho jaayengi.
    """
    td = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
    with _local_lock:
        conn = _local_conn()
        try:
            rows = conn.execute(
                """SELECT name, att_time AS time, status,
                          camera_id AS camera, photo_path AS photo
                   FROM pending_attendance
                   WHERE att_date = ? AND synced = 0
                   ORDER BY att_time DESC""",
                (td,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def _local_pending_rows():
    """Sirf synced=0 rows — jo abhi Supabase nahi gayi."""
    with _local_lock:
        conn = _local_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM pending_attendance WHERE synced = 0 ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def _local_last_status_today(name):
    """
    SQLite mein is naam ka aaj ka sabse recent status.
    Dono synced=0 aur synced=1 rows check karo — kyunki startup ke baad
    kuch rows synced=1 ho sakti hain aur in-memory cache empty hota hai.
    """
    today = date.today().isoformat()
    with _local_lock:
        conn = _local_conn()
        try:
            row = conn.execute(
                """SELECT status FROM pending_attendance
                   WHERE name = ? AND att_date = ?
                   ORDER BY id DESC LIMIT 1""",
                (name, today),
            ).fetchone()
            return row["status"] if row else None
        finally:
            conn.close()


def _local_count_today(name):
    today = date.today().isoformat()
    with _local_lock:
        conn = _local_conn()
        try:
            row = conn.execute(
                """SELECT COUNT(*) AS c FROM pending_attendance
                   WHERE name = ? AND att_date = ?""",
                (name, today),
            ).fetchone()
            return row["c"]
        finally:
            conn.close()


def _local_bump_retry(row_id):
    with _local_lock:
        conn = _local_conn()
        try:
            conn.execute(
                "UPDATE pending_attendance SET retry_count = retry_count + 1 WHERE id = ?",
                (row_id,),
            )
            conn.commit()
        finally:
            conn.close()


def pending_sync_count():
    with _local_lock:
        conn = _local_conn()
        try:
            return conn.execute(
                "SELECT COUNT(*) AS c FROM pending_attendance WHERE synced = 0"
            ).fetchone()["c"]
        finally:
            conn.close()


# =============================================================================
#  7-DAY PURGE
# =============================================================================
def _purge_old_local_rows():
    cutoff = (date.today() - timedelta(days=PURGE_AFTER_DAYS)).isoformat()
    with _local_lock:
        conn = _local_conn()
        try:
            skipped = conn.execute(
                """SELECT COUNT(*) AS c FROM pending_attendance
                   WHERE att_date < ? AND synced = 0""",
                (cutoff,),
            ).fetchone()["c"]

            if skipped > 0:
                print(
                    f"[purge] WARNING: {skipped} unsynced row(s) 7+ din purani — "
                    f"DELETE nahi kar rahe. Internet aane pe sync hogi."
                )

            cur = conn.execute(
                "DELETE FROM pending_attendance WHERE att_date < ? AND synced = 1",
                (cutoff,),
            )
            deleted = cur.rowcount
            conn.commit()
            if deleted:
                print(f"[purge] {deleted} synced row(s) deleted (< {cutoff}). Supabase pe safe hai.")
            return deleted, skipped
        finally:
            conn.close()


# =============================================================================
#  DETECTION AUDIT LOG
# =============================================================================
def _local_log_init():
    with _local_lock:
        conn = _local_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    camera_id   TEXT,
                    confidence  REAL,
                    result      TEXT NOT NULL,
                    event_time  TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()


def log_detection(name, camera_id, confidence, result):
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with _local_lock:
        conn = _local_conn()
        try:
            conn.execute(
                """INSERT INTO detection_log
                   (name, camera_id, confidence, result, event_time)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, camera_id, confidence, result, event_time),
            )
            conn.commit()
        finally:
            conn.close()


# =============================================================================
#  EMPLOYEE ROSTER
# =============================================================================
def get_or_create_employee(name, department=None, designation=None):
    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM employees WHERE name = %s", (name,))
            row = cur.fetchone()
            if row:
                return row["id"]
            cur.execute(
                """INSERT INTO employees (name, department, designation)
                   VALUES (%s, %s, %s) RETURNING id""",
                (name, department, designation),
            )
            emp_id = cur.fetchone()["id"]
            conn.commit()
            cur.close()
            return emp_id
        finally:
            conn.close()
    except Exception as e:
        print(f"[db] get_or_create_employee offline ({name}): {e}")
        return None


def sync_roster(names):
    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            for name in names:
                cur.execute(
                    "INSERT INTO employees (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                    (name,),
                )
            conn.commit()
            cur.close()
        finally:
            conn.close()
    except Exception as e:
        print(f"[db] sync_roster skipped (offline): {e}")


# =============================================================================
#  CORE STATUS HELPER
# =============================================================================
def _get_effective_status(name: str):
    """
    In-memory cache → SQLite fallback (dono synced=0 aur synced=1).
    Network call nahi hoti — zero latency.
    """
    status = _mem_get(name)
    if status is not None:
        return status
    # SQLite fallback — dono synced rows check karo (latest status chahiye)
    return _local_last_status_today(name)


# =============================================================================
#  ATTENDANCE MARKING  —  SQLite FIRST
# =============================================================================
def mark_attendance(name, camera_id, confidence=None, photo_path=None,
                    department=None):
    """Returns: 'marked' | 'already_present' """
    now = datetime.now()   # ← TIMESTAMP FIRST

    _mem_reset_if_new_day()

    with _mark_lock:
        if _get_effective_status(name) == "Present":
            return "already_present"

        _local_insert(
            name,
            now.date().isoformat(),
            now.strftime("%H:%M:%S"),
            "Present",
            camera_id,
            confidence,
            photo_path,
        )

        _mem_set(name, "Present")

    with present_cache_lock:
        present_cache.add(name)

    return "marked"


def mark_exit(name, camera_id, confidence=None, photo_path=None):
    """Returns: 'marked' | 'not_inside' """
    now = datetime.now()   # ← TIMESTAMP FIRST

    _mem_reset_if_new_day()

    with _mark_lock:
        if _get_effective_status(name) != "Present":
            return "not_inside"

        _local_insert(
            name,
            now.date().isoformat(),
            now.strftime("%H:%M:%S"),
            "Exit",
            camera_id,
            confidence,
            photo_path,
        )

        _mem_set(name, "Exit")

    with present_cache_lock:
        present_cache.discard(name)

    return "marked"


def get_last_status_today(name: str):
    _mem_reset_if_new_day()
    return _get_effective_status(name)


def count_today_rows(name):
    today = date.today()
    local_n = _local_count_today(name)
    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) AS c FROM attendance WHERE name = %s AND att_date = %s",
                (name, today),
            )
            pg_n = cur.fetchone()["c"]
            cur.close()
            return local_n + pg_n
        finally:
            conn.close()
    except Exception:
        return local_n


# =============================================================================
#  READ HELPERS  —  Flask dashboard ke liye
# =============================================================================
def _dictify(cur):
    return [dict(row) for row in cur.fetchall()]


def read_day_rows(target_date):
    """
    SQLite (synced=0 only) + Supabase (synced rows) merge karo.
    DEDUP: name+time key se ensure karo duplicate na aaye.

    Order:
      1. Supabase rows pehle (authoritative source)
      2. SQLite pending rows baad mein (jo abhi sync nahi hue)
      3. Final result newest-first sort
    """
    # sirf pending rows (synced=0) — synced=1 Supabase mein already hain
    local_rows = _local_day_rows(target_date)

    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT name, att_time AS time, status,
                          camera_id AS camera, photo_path AS photo
                   FROM attendance WHERE att_date = %s
                   ORDER BY att_time DESC""",
                (target_date,),
            )
            pg_rows = _dictify(cur)
            cur.close()
            for r in pg_rows:
                r["time"] = str(r["time"])
        finally:
            conn.close()
    except Exception as e:
        print(f"[db] read_day_rows offline fallback: {e}")
        return local_rows

    # DEDUP merge — name+time combination ek baar hi aaye
    seen   = set()
    merged = []

    # Supabase rows pehle (authoritative)
    for r in pg_rows:
        key = (r["name"], str(r["time"]))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    # SQLite pending rows — sirf jo Supabase mein nahi hain
    for r in local_rows:
        key = (r["name"], str(r["time"]))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    # Newest first
    merged.sort(key=lambda x: x.get("time", ""), reverse=True)
    return merged


def read_month_rows(month):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT name, att_date AS date, att_time AS time, status,
                      camera_id AS camera, photo_path AS photo
               FROM attendance WHERE TO_CHAR(att_date, 'YYYY-MM') = %s
               ORDER BY att_time ASC""",
            (month,),
        )
        rows = _dictify(cur)
        cur.close()
        for r in rows:
            r["date"] = str(r["date"])
            r["time"] = str(r["time"])
        return rows
    finally:
        conn.close()


def all_dates():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT att_date FROM attendance ORDER BY att_date DESC"
        )
        dates = [str(r["att_date"]) for r in cur.fetchall()]
        cur.close()
        return dates
    finally:
        conn.close()


def available_months():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT TO_CHAR(att_date, 'YYYY-MM') AS m "
            "FROM attendance ORDER BY 1 DESC"
        )
        months = [r["m"] for r in cur.fetchall()]
        cur.close()
        return months
    finally:
        conn.close()


def get_roster():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM employees WHERE active = TRUE ORDER BY name"
        )
        names = [r["name"] for r in cur.fetchall()]
        cur.close()
        return names, "employees"
    finally:
        conn.close()


def get_all_face_embeddings():
    """
    Sabhi ACTIVE employees ke face_embeddings Supabase se ek saath fetch
    karo — { name: [raw_bytes, raw_bytes, ...] }.

    Yeh entry_cameras.py ko allow karta hai ki koi bhi machine se (admin
    portal Jetson pe ho ya Render pe deployed ho) add/update kiya gaya
    employee turant camera pe recognizable ho jaaye — kyunki admin panel
    har photo ke embedding ko is table mein daalta hai (auth_db.py), chahe
    request kahin se bhi aayi ho. Local embeddings.pkl sirf offline cache
    hai; Supabase yahan source of truth hai.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT e.name AS name, fe.embedding AS embedding
               FROM face_embeddings fe
               JOIN employees e ON e.id = fe.employee_id
               WHERE e.active = TRUE"""
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    out = {}
    for r in rows:
        out.setdefault(r["name"], []).append(bytes(r["embedding"]))
    return out


def get_employee_photo(name):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT photo_path FROM employees WHERE name = %s", (name,))
        row = cur.fetchone()
        cur.close()
        return row["photo_path"] if row and row["photo_path"] else None
    finally:
        conn.close()


def _safe_folder_name(name):
    """Same sanitize rule as portal.py's _safe_name() / entry_cameras.py's
       _safe() — spaces/special chars -> '_'. Must stay identical to those
       so folder <-> employee-name matching is consistent everywhere."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _find_profile_photo(emp_dir):
    """
    PHOTOS_DIR/<folder>/ ke andar ek usable photo dhoondo.
    Priority: top-level "profile_*" file > koi bhi top-level image >
    (fallback) sabse recent date-subfolder ka camera capture.
    Returns (filename, subdir_or_None) ya (None, None).
    """
    import os as _os
    IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

    top_files = sorted(
        f for f in _os.listdir(emp_dir)
        if _os.path.isfile(_os.path.join(emp_dir, f)) and f.lower().endswith(IMG_EXT)
    )
    for f in top_files:
        if f.startswith("profile_"):
            return f, None
    if top_files:
        return top_files[0], None

    # Fallback — date subfolders (entry/exit camera captures). Most
    # employees ONLY have this — no top-level profile photo — jab tak
    # unhe kabhi portal se add/update na kiya gaya ho.
    for sub in sorted(_os.listdir(emp_dir), reverse=True):   # newest date pehle
        sub_path = _os.path.join(emp_dir, sub)
        if _os.path.isdir(sub_path):
            sub_files = sorted(
                f for f in _os.listdir(sub_path)
                if _os.path.isfile(_os.path.join(sub_path, f)) and f.lower().endswith(IMG_EXT)
            )
            if sub_files:
                return sub_files[0], sub
    return None, None


def sync_employee_photos_from_dir():
    """
    Employees ke photo_path ko local PHOTOS_DIR se backfill karo.

    IMPORTANT:
      - Employee ka real DB name kabhi bhi photo-folder name se create nahi hoga.
      - Folder names sanitized hote hain, e.g. "Amitabh_prajapati".
      - DB employee name "Amitabh prajapati" hi rahega.
      - Existing employees ko sanitized folder-name ke through match karke
        sirf photo_path update kiya jayega.
      - Agar folder ka matching employee nahi milta, folder ko skip kiya jayega;
        koi naya employee INSERT nahi hoga.
    """
    import os as _os

    if not _os.path.isdir(config.PHOTOS_DIR):
        return 0

    updated = 0

    try:
        conn = get_conn()
        try:
            cur = conn.cursor()

            # Existing employees ka safe-folder-name -> employee row map.
            # Example:
            #   DB name    = "Amitabh prajapati"
            #   folder     = "Amitabh_prajapati"
            #   map key    = "Amitabh_prajapati"
            cur.execute("SELECT id, name, photo_path FROM employees")
            employees = cur.fetchall()
            emp_by_safe = {
                _safe_folder_name(r["name"]): r
                for r in employees
            }

            for entry in sorted(_os.listdir(config.PHOTOS_DIR)):
                emp_dir = _os.path.join(config.PHOTOS_DIR, entry)

                if not _os.path.isdir(emp_dir):
                    continue

                fname, subdir = _find_profile_photo(emp_dir)
                if not fname:
                    continue

                rel_path = (
                    f"{entry}/{subdir}/{fname}"
                    if subdir
                    else f"{entry}/{fname}"
                )

                # IMPORTANT: entry is a sanitized folder name, NOT the
                # employee's database name.
                row = emp_by_safe.get(entry)

                if row is None:
                    # NEVER INSERT `entry` into employees.name.
                    # Doing so creates duplicates such as:
                    #   "Amitabh prajapati"
                    #   "Amitabh_prajapati"
                    print(
                        f"[db] Skipping unmatched photo folder: {entry} "
                        f"(no matching employee found)"
                    )
                    continue

                # Employee already exists. Only backfill photo_path.
                if not row["photo_path"]:
                    cur.execute(
                        "UPDATE employees SET photo_path = %s WHERE id = %s",
                        (rel_path, row["id"]),
                    )
                    updated += 1
                    print(
                        f"[db] Photo path updated → {row['name']} | {rel_path}"
                    )

            conn.commit()
            cur.close()

        finally:
            conn.close()

    except Exception as e:
        print(f"[db] sync_employee_photos_from_dir skipped (offline): {e}")
        return 0

    return updated


def list_employees():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, name, department, designation, photo_path, active
               FROM employees ORDER BY name"""
        )
        rows = _dictify(cur)
        cur.close()
        return rows
    finally:
        conn.close()


# =============================================================================
#  BACKGROUND SYNC + PURGE WORKER
# =============================================================================
_sync_thread_started = False
_sync_thread_lock    = threading.Lock()


def _supabase_last_status_today(cur, name: str):
    today = date.today().isoformat()
    cur.execute(
        """SELECT status FROM attendance
           WHERE name = %s AND att_date = %s
           ORDER BY att_time DESC, id DESC LIMIT 1""",
        (name, today),
    )
    row = cur.fetchone()
    return row["status"] if row else None


def _should_push(supabase_status, incoming_status: str) -> bool:
    if supabase_status is None:
        return True
    if supabase_status == "Present":
        return incoming_status == "Exit"
    if supabase_status == "Exit":
        return incoming_status == "Present"
    return True


def _has_client_uuid_column(cur) -> bool:
    cur.execute(
        """SELECT 1 FROM information_schema.columns
           WHERE table_name='attendance' AND column_name='client_uuid'"""
    )
    return cur.fetchone() is not None


def _push_row_to_supabase(cur, row: dict, has_uuid_col: bool, emp_id: int):
    if has_uuid_col:
        cur.execute(
            """INSERT INTO attendance
               (employee_id, name, att_date, att_time, status,
                camera_id, confidence, photo_path, client_uuid)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (client_uuid) DO NOTHING""",
            (emp_id, row["name"], row["att_date"], row["att_time"],
             row["status"], row["camera_id"], row["confidence"],
             row["photo_path"], row["client_uuid"]),
        )
    else:
        cur.execute(
            """SELECT 1 FROM attendance
               WHERE name=%s AND att_date=%s AND att_time=%s AND camera_id=%s""",
            (row["name"], row["att_date"], row["att_time"], row["camera_id"]),
        )
        if cur.fetchone() is None:
            cur.execute(
                """INSERT INTO attendance
                   (employee_id, name, att_date, att_time, status,
                    camera_id, confidence, photo_path)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (emp_id, row["name"], row["att_date"], row["att_time"],
                 row["status"], row["camera_id"], row["confidence"],
                 row["photo_path"]),
            )


def _sync_loop():
    purge_counter = 0

    while True:
        time.sleep(SYNC_INTERVAL_SECONDS)

        # Purge check
        purge_counter += 1
        if purge_counter >= PURGE_CHECK_INTERVAL:
            purge_counter = 0
            try:
                _purge_old_local_rows()
            except Exception as e:
                print(f"[purge] Error (non-fatal): {e}")

        pending = _local_pending_rows()
        if not pending:
            continue

        print(f"[sync] {len(pending)} pending row(s) — Supabase push shuru ...")

        try:
            conn = get_conn()
        except Exception:
            print("[sync] Supabase unreachable — 5 min mein retry hogi.")
            continue

        try:
            cur = conn.cursor()
            has_uuid_col = _has_client_uuid_column(cur)

            synced  = 0
            skipped = 0
            failed  = 0
            dedup_tracker: dict = {}

            for row in pending:
                name   = row["name"]
                status = row["status"]
                key    = (name, status, row["att_date"])

                try:
                    # Dedup window check
                    last_time = dedup_tracker.get(key)
                    if last_time is not None:
                        try:
                            fmt = "%Y-%m-%d %H:%M:%S"
                            t1  = datetime.strptime(f"{row['att_date']} {last_time}", fmt)
                            t2  = datetime.strptime(f"{row['att_date']} {row['att_time']}", fmt)
                            if abs((t2 - t1).total_seconds()) < DEDUP_WINDOW_SECONDS:
                                print(f"[sync] DEDUP SKIP → {name} | {status} | {row['att_time']}")
                                _local_mark_synced(row["id"])
                                skipped += 1
                                continue
                        except Exception:
                            pass

                    # Supabase smart logic
                    sb_status = _supabase_last_status_today(cur, name)

                    # GUARD: Exit kabhi bhi push mat karo jab tak Supabase mein
                    # Present already na dikhe — warna agar us employee ka
                    # Present row abhi tak sync nahi hua (pichli cycle fail/
                    # pending), toh Exit akela chala jaata: "entry missing,
                    # exit marked" bug. Row ko PENDING hi chhodo (mark_synced
                    # mat karo) — Present sync hone ke baad agli cycle mein
                    # yeh khud-ba-khud push ho jaayega.
                    if status == "Exit" and sb_status != "Present":
                        print(f"[sync] DEFER (entry not synced yet) → {name} | Exit | {row['att_time']}")
                        continue

                    if not _should_push(sb_status, status):
                        print(f"[sync] SMART SKIP → {name} | Supabase={sb_status} | incoming={status}")
                        _local_mark_synced(row["id"])
                        skipped += 1
                        continue

                    # Employee ID resolve
                    emp_id = get_or_create_employee(name)
                    if emp_id is None:
                        raise RuntimeError("emp_id resolve nahi hua (offline?)")

                    # Supabase insert
                    _push_row_to_supabase(cur, row, has_uuid_col, emp_id)
                    conn.commit()
                    _local_mark_synced(row["id"])
                    dedup_tracker[key] = row["att_time"]
                    print(f"[sync] PUSHED → {name} | {status} | {row['att_date']} {row['att_time']}")
                    synced += 1

                except Exception as e:
                    conn.rollback()
                    _local_bump_retry(row["id"])
                    failed += 1
                    print(f"[sync] FAILED → {name} | retry#{row['retry_count']+1} | {e}")

            cur.close()
            print(
                f"[sync] COMPLETE — pushed:{synced} | skipped:{skipped} | "
                f"failed:{failed} | still pending:{pending_sync_count()}"
            )

        finally:
            conn.close()


def _start_sync_worker():
    global _sync_thread_started
    with _sync_thread_lock:
        if _sync_thread_started:
            return
        _sync_thread_started = True
        t = threading.Thread(
            target=_sync_loop,
            daemon=True,
            name="supabase-sync-worker",
        )
        t.start()
        print(
            f"[db] Sync worker started:\n"
            f"     Interval : {SYNC_INTERVAL_SECONDS}s (5 min)\n"
            f"     Purge    : {PURGE_AFTER_DAYS} din baad (synced rows only)\n"
            f"     Queue    : {LOCAL_DB_PATH}"
        )