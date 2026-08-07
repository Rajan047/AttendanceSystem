"""
auth_db.py — login accounts + admin writes, layered on top of your existing
Postgres/Supabase schema WITHOUT touching db.py.

Design rules kept identical to your db.py:
  - borrow connections via db.get_conn() (same pool, same timeouts)
  - additive schema only: new `users` table + IF NOT EXISTS columns/enum values
  - link a login to an employee by `name` (your natural key everywhere)
  - all admin attendance writes land in your existing `attendance` table so the
    dashboard you already have reflects them automatically.

Run init_auth_schema() once at server startup (server.py calls it).
"""
import os
from datetime import datetime, date

from werkzeug.security import generate_password_hash, check_password_hash

import db  # your existing module — we reuse its pool via db.get_conn()


# =============================================================================
#  SCHEMA  — additive migrations, safe to run on every startup
# =============================================================================
def init_auth_schema(seed_admin=True):
    conn = db.get_conn()
    try:
        cur = conn.cursor()

        # 1) login accounts
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                username      VARCHAR(150) NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          VARCHAR(20)  NOT NULL DEFAULT 'employee',
                emp_name      VARCHAR(150),   -- links to employees.name; NULL for pure admins
                created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2) HR/leave request queue
        cur.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id          SERIAL PRIMARY KEY,
                emp_name    VARCHAR(150) NOT NULL,
                req_date    DATE NOT NULL,
                type        VARCHAR(20) NOT NULL,       -- wfh / leave
                reason      TEXT,
                status      VARCHAR(20) NOT NULL DEFAULT 'pending',
                reviewed_by VARCHAR(150),
                created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3) extra employee columns your admin panel collects (schema.sql
        #    only had name/department/designation/photo_path/active).
        for col, ddl in [
            ("email",     "ALTER TABLE employees ADD COLUMN IF NOT EXISTS email VARCHAR(150)"),
            ("phone",     "ALTER TABLE employees ADD COLUMN IF NOT EXISTS phone VARCHAR(40)"),
            ("join_date", "ALTER TABLE employees ADD COLUMN IF NOT EXISTS join_date DATE"),
        ]:
            cur.execute(ddl)

        conn.commit()
        cur.close()

        # 4) extend the attendance_status enum so manual HR marks can live in
        #    the same attendance table the dashboard already reads.
        #    ALTER TYPE ... ADD VALUE must run in its OWN autocommit session
        #    (it can't be rolled back and can't share a txn that then uses the
        #    value). We reach into the raw psycopg2 connection for this.
        _extend_status_enum(conn)

        if seed_admin:
            _seed_default_admin(conn)

    finally:
        conn.close()


def _extend_status_enum(conn):
    """Add WFH/Leave/Absent/HalfDay to the attendance_status enum, each in its
       own autocommit transaction. Safe to run repeatedly (IF NOT EXISTS)."""
    raw = getattr(conn, "_raw", conn)  # unwrap db.py's _ConnCtx to the real conn
    prev_autocommit = raw.autocommit
    try:
        raw.autocommit = True
        for val in ("WFH", "Leave", "Absent", "HalfDay"):
            try:
                c = raw.cursor()
                c.execute(f"ALTER TYPE attendance_status ADD VALUE IF NOT EXISTS '{val}'")
                c.close()
            except Exception as e:
                print(f"[auth_db] enum extend note ({val}): {e}")
    finally:
        raw.autocommit = prev_autocommit


def _seed_default_admin(conn):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, 'admin')",
            ("admin", generate_password_hash("admin123")),
        )
        conn.commit()
        print("[auth_db] Seeded default admin -> admin / admin123 "
              "(CHANGE THIS after first login)")
    cur.close()


# =============================================================================
#  AUTH
# =============================================================================
def verify_login(username, password):
    """Returns a dict {id, username, role, emp_name} on success, else None."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return None
    return {"id": row["id"], "username": row["username"],
            "role": row["role"], "emp_name": row["emp_name"]}


def create_login(username, password, role="employee", emp_name=None):
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, role, emp_name) "
            "VALUES (%s, %s, %s, %s)",
            (username, generate_password_hash(password), role, emp_name),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def change_password(user_id, old_pw, new_pw):
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        if not row or not check_password_hash(row["password_hash"], old_pw):
            cur.close()
            return False
        cur.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                    (generate_password_hash(new_pw), user_id))
        conn.commit()
        cur.close()
        return True
    finally:
        conn.close()


# =============================================================================
#  ADMIN — EMPLOYEE MANAGEMENT
# =============================================================================
def list_employees_full():
    """Everything the admin table needs, including whether a face embedding
       exists (checked against face_embeddings table if present)."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT e.id, e.name, e.department, e.designation, e.email, e.phone,
                   e.join_date, e.photo_path, e.active,
                   EXISTS(SELECT 1 FROM face_embeddings f WHERE f.employee_id = e.id) AS has_embedding,
                   EXISTS(SELECT 1 FROM users u WHERE u.emp_name = e.name) AS has_login
            FROM employees e
            ORDER BY e.name
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        for r in rows:
            if r.get("join_date"):
                r["join_date"] = str(r["join_date"])
        return rows
    finally:
        conn.close()


def add_employee(name, department=None, designation=None, email=None,
                 phone=None, join_date=None, photo_path=None):
    """Insert or update-on-conflict (name is unique). Returns employee id."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO employees (name, department, designation, email, phone,
                                   join_date, photo_path, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (name) DO UPDATE SET
                department  = COALESCE(EXCLUDED.department, employees.department),
                designation = COALESCE(EXCLUDED.designation, employees.designation),
                email       = COALESCE(EXCLUDED.email, employees.email),
                phone       = COALESCE(EXCLUDED.phone, employees.phone),
                join_date   = COALESCE(EXCLUDED.join_date, employees.join_date),
                photo_path  = COALESCE(EXCLUDED.photo_path, employees.photo_path)
            RETURNING id
        """, (name, department, designation, email, phone, join_date, photo_path))
        emp_id = cur.fetchone()["id"]
        conn.commit()
        cur.close()
        return emp_id
    finally:
        conn.close()


def update_employee(emp_id, **fields):
    # Date fields mein empty string = NULL
    for date_field in ('join_date', 'resignation'):
        if date_field in fields and fields[date_field] == '':
            fields[date_field] = None

    allowed = {"name", "department", "designation", "email", "phone",
               "join_date", "photo_path", "active"}
    sets, params = [], []
    for k, v in fields.items():
        if k in allowed and v is not None:
            sets.append(f"{k} = %s")
            params.append(v)
    if not sets:
        return False
    params.append(emp_id)
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"UPDATE employees SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()
        cur.close()
        return True
    finally:
        conn.close()


def set_employee_embedding(emp_id, embedding_bytes):
    """Replace ALL of this employee's face_embeddings rows with a single new
       one. Used by the "regenerate from scratch" flow, which already
       recomputes every embedding from the photos on disk — a full replace
       is correct there. For adding photos incrementally, use
       add_employee_embedding() instead (it doesn't delete existing rows)."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM face_embeddings WHERE employee_id = %s", (emp_id,))
        cur.execute(
            "INSERT INTO face_embeddings (employee_id, embedding) VALUES (%s, %s)",
            (emp_id, psycopg2_binary(embedding_bytes)),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def add_employee_embedding(emp_id, embedding_bytes):
    """Append one more face_embeddings row for this employee — does NOT
       touch existing rows. Used when uploading additional photos (add/edit
       employee), so multi-photo uploads accumulate instead of each new
       photo silently replacing the previous one. entry_cameras.py averages
       all rows for a name together, same as it does with embeddings.pkl."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO face_embeddings (employee_id, embedding) VALUES (%s, %s)",
            (emp_id, psycopg2_binary(embedding_bytes)),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def psycopg2_binary(b):
    import psycopg2
    return psycopg2.Binary(b)


def deactivate_employee(emp_id):
    """Soft delete — sets active=FALSE so live cameras stop counting them
       (get_roster() filters on active=TRUE) but history is preserved."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE employees SET active = FALSE WHERE id = %s", (emp_id,))
        conn.commit()
        cur.close()
    finally:
        conn.close()


def delete_employee(emp_id):
    """Hard delete — cascades to attendance + face_embeddings (FK ON DELETE
       CASCADE in your schema). Also removes any linked login."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM employees WHERE id = %s", (emp_id,))
        row = cur.fetchone()
        if row:
            cur.execute("DELETE FROM users WHERE emp_name = %s", (row["name"],))
        cur.execute("DELETE FROM employees WHERE id = %s", (emp_id,))
        conn.commit()
        cur.close()
    finally:
        conn.close()


def get_employee(emp_id):
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
        row = cur.fetchone()
        cur.close()
        d = dict(row) if row else None
        if d and d.get("join_date"):
            d["join_date"] = str(d["join_date"])
        return d
    finally:
        conn.close()


# =============================================================================
#  ADMIN — MANUAL ATTENDANCE  (writes into your existing attendance table)
# =============================================================================
_VALID_MANUAL = {"Present", "Exit", "WFH", "Leave", "Absent", "HalfDay"}


def manual_mark(name, status, on_date=None, in_time=None, marked_by="admin"):
    """
    HR/admin manual mark. Upserts one row per (name, date) for manual statuses
    so re-marking corrects rather than duplicates. Present/Exit still append
    (they're event-based like the cameras produce).
    """
    if status not in _VALID_MANUAL:
        raise ValueError(f"status must be one of {sorted(_VALID_MANUAL)}")

    on_date = on_date or date.today().isoformat()
    now_time = in_time or datetime.now().strftime("%H:%M:%S")

    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM employees WHERE name = %s", (name,))
        row = cur.fetchone()
        if not row:
            cur.close()
            raise ValueError(f"No employee named {name}")
        emp_id = row["id"]

        # For manual day-statuses, replace any prior manual mark for that day
        # (but never clobber camera Present/Exit event rows).
        if status in ("WFH", "Leave", "Absent", "HalfDay"):
            cur.execute(
                "DELETE FROM attendance WHERE name = %s AND att_date = %s "
                "AND status IN ('WFH','Leave','Absent','HalfDay')",
                (name, on_date),
            )

        cur.execute("""
            INSERT INTO attendance
                (employee_id, name, att_date, att_time, status, camera_id, confidence, photo_path)
            VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL)
        """, (emp_id, name, on_date, now_time, status, f"manual:{marked_by}"))
        conn.commit()
        cur.close()
        return "marked"
    finally:
        conn.close()


# =============================================================================
#  REQUESTS  (WFH / leave)
# =============================================================================
def submit_request(emp_name, req_date, req_type, reason=""):
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO requests (emp_name, req_date, type, reason) "
            "VALUES (%s, %s, %s, %s)",
            (emp_name, req_date, req_type, reason),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def list_requests(status="pending"):
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM requests WHERE status = %s ORDER BY created_at DESC",
            (status,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        for r in rows:
            r["req_date"] = str(r["req_date"])
            r["created_at"] = str(r["created_at"])
        return rows
    finally:
        conn.close()


def list_requests_for(emp_name):
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM requests WHERE emp_name = %s ORDER BY created_at DESC",
            (emp_name,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        for r in rows:
            r["req_date"] = str(r["req_date"])
            r["created_at"] = str(r["created_at"])
        return rows
    finally:
        conn.close()


def review_request(req_id, decision, reviewer):
    """decision: 'approved' | 'rejected'. Approving auto-marks attendance."""
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM requests WHERE id = %s", (req_id,))
        req = cur.fetchone()
        if not req:
            cur.close()
            return False
        cur.execute(
            "UPDATE requests SET status = %s, reviewed_by = %s WHERE id = %s",
            (decision, reviewer, req_id),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()

    if decision == "approved":
        status = "WFH" if req["type"] == "wfh" else "Leave"
        manual_mark(req["emp_name"], status,
                    on_date=str(req["req_date"]), marked_by=f"req:{reviewer}")
    return True
