"""
portal.py — login + admin panel + employee panel, Flask Blueprint.
UPDATED: Supabase Storage mein photos upload karo.
"""
from dataclasses import fields
import os
import uuid
import calendar
from functools import wraps
from datetime import datetime, date
from collections import defaultdict

from flask import (
    Blueprint, request, session, redirect, url_for,
    render_template, jsonify
)
from werkzeug.utils import secure_filename

import config
import db
import auth_db

# Supabase Storage helper
try:
    import supabase_storage as _storage
    STORAGE_AVAILABLE = True
except ImportError:
    _storage = None
    STORAGE_AVAILABLE = False

# face_embedding — Render pe available nahi hogi, Jetson pe hogi
try:
    import face_embedding as _fe
    FACE_EMBEDDING_AVAILABLE = True
except ImportError:
    _fe = None
    FACE_EMBEDDING_AVAILABLE = False

portal = Blueprint("portal", __name__, template_folder="templates")

ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp"}
LOW_THRESHOLD = 75
MAX_PHOTOS    = 10


# =============================================================================
#  DECORATORS
# =============================================================================
def login_required(fn):
    @wraps(fn)
    def w(*a, **k):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "login required"}), 401
            return redirect(url_for("portal.login"))
        return fn(*a, **k)
    return w


def admin_required(fn):
    @wraps(fn)
    def w(*a, **k):
        if session.get("role") != "admin":
            if request.path.startswith("/api/"):
                return jsonify({"error": "admin only"}), 403
            return redirect(url_for("portal.login"))
        return fn(*a, **k)
    return w


def employee_required(fn):
    @wraps(fn)
    def w(*a, **k):
        if session.get("role") != "employee":
            if request.path.startswith("/api/"):
                return jsonify({"error": "employee only"}), 403
            return redirect(url_for("portal.login"))
        return fn(*a, **k)
    return w


def _safe_name(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# =============================================================================
#  AUTH
# =============================================================================
@portal.route("/portal/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "user_id" in session:
            return redirect(url_for("portal.admin_home") if session["role"] == "admin"
                            else url_for("portal.employee_home"))
        return render_template("login.html")

    data = request.get_json(silent=True) or request.form
    user = auth_db.verify_login((data.get("username") or "").strip(),
                                data.get("password") or "")
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    session.clear()
    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    session["role"]     = user["role"]
    session["emp_name"] = user["emp_name"]
    session.permanent   = True

    dest = (url_for("portal.admin_home") if user["role"] == "admin"
            else url_for("portal.employee_home"))
    return jsonify({"ok": True, "role": user["role"], "redirect": dest})


@portal.route("/portal/logout")
def logout():
    session.clear()
    return redirect(url_for("portal.login"))


@portal.route("/api/portal/change-password", methods=["POST"])
@login_required
def change_password():
    d = request.get_json(force=True)
    if len(d.get("new_password", "")) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400
    ok = auth_db.change_password(session["user_id"], d.get("old_password", ""),
                                 d.get("new_password", ""))
    return (jsonify({"ok": True}) if ok
            else (jsonify({"error": "Current password is incorrect"}), 400))


@portal.route("/portal/signup", methods=["POST"])
def signup():
    data     = request.get_json(silent=True) or request.form
    name     = (data.get("name") or "").strip()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not name or not username or len(password) < 6:
        return jsonify({"error": "Name, username and password (>=6) required"}), 400
    try:
        auth_db.create_login(username, password, role="employee", emp_name=name)
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "Signup failed (username may be taken)"}), 400


# =============================================================================
#  ADMIN PAGES
# =============================================================================
@portal.route("/portal/admin")
@admin_required
def admin_home():
    return render_template("admin_dashboard.html", username=session.get("username"))


@portal.route("/portal/employee")
@employee_required
def employee_home():
    return render_template("employee_dashboard.html", username=session.get("username"))


# =============================================================================
#  PHOTO HELPERS
# =============================================================================
def _save_photo(photo, emp_name, prefix="profile"):
    """
    Photo save karo — Supabase Storage mein (preferred) ya local fallback.
    Returns (storage_path, public_url_or_abs_path) or (None, error_msg)
    """
    ext = photo.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return None, "Photo must be jpg/jpeg/png/webp"

    safe         = _safe_name(emp_name)
    fname        = f"{prefix}_{uuid.uuid4().hex}.{ext}"
    storage_path = f"{safe}/{fname}"

    # ── Supabase Storage (Render + Jetson dono ke liye persistent) ──────────
    if STORAGE_AVAILABLE:
        file_bytes   = photo.read()
        content_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
        public_url   = _storage.upload_photo(file_bytes, storage_path, content_type)
        if public_url:
            print(f"[storage] ✅ Uploaded to Supabase: {storage_path}")
            return storage_path, public_url

    # ── Local fallback (Jetson pe direct run karte waqt) ────────────────────
    abs_dir  = os.path.join(config.PHOTOS_DIR, safe)
    os.makedirs(abs_dir, exist_ok=True)
    abs_path = os.path.join(abs_dir, fname)
    photo.seek(0)
    photo.save(abs_path)
    print(f"[storage] local save (Supabase unavailable): {abs_path}")
    return storage_path, abs_path


def _process_photos_and_embed(photos, emp_name, emp_id, clear_existing=False):
    """
    Multiple photos process karo — save + embedding generate.
    Returns result dict.
    """
    saved, embedded, failed = 0, 0, 0
    warnings    = []
    first_photo = None
    new_embeddings = []

    if not FACE_EMBEDDING_AVAILABLE:
        warnings.append(
            "Face recognition is not available on this server — photo(s) "
            "saved to Supabase Storage. Jetson ka photo_sync.py automatically "
            "download karke embedding generate kar dega (30s mein)."
        )

    if clear_existing and FACE_EMBEDDING_AVAILABLE:
        _fe.remove_from_pickle(emp_name)

    for i, photo in enumerate(photos):
        if not photo or not photo.filename:
            continue
        if saved >= MAX_PHOTOS:
            warnings.append(f"Max {MAX_PHOTOS} photos — baaki skip ho gayi.")
            break

        prefix = "profile" if i == 0 else f"face_{i}"
        storage_path, result = _save_photo(photo, emp_name, prefix=prefix)

        if storage_path is None:
            warnings.append(f"Photo {i+1}: {result}")
            failed += 1
            continue

        saved += 1
        if first_photo is None:
            first_photo = storage_path

        if not FACE_EMBEDDING_AVAILABLE:
            # Photo Supabase mein save ho gayi — Jetson sync karega
            continue

        # Embedding generate karo (sirf Jetson pe)
        try:
            # result = abs_path ya public_url
            abs_path = result if result.startswith("/") else None
            if abs_path:
                emb = _fe.generate_embedding(abs_path)
                if emb is None:
                    warnings.append(f"Photo {i+1}: face detect nahi hua.")
                    failed += 1
                else:
                    new_embeddings.append(emb)
                    try:
                        auth_db.add_employee_embedding(
                            emp_id, _fe.embedding_to_bytes(emb)
                        )
                    except Exception:
                        pass
                    embedded += 1
        except Exception as e:
            warnings.append(f"Photo {i+1}: embedding error — {e}")
            failed += 1

    if new_embeddings:
        _fe.add_many_to_pickle(emp_name, new_embeddings)
        _touch_embeddings_file()

    return {
        "saved":       saved,
        "embedded":    embedded,
        "failed":      failed,
        "first_photo": first_photo,
        "warnings":    warnings,
    }


def _touch_embeddings_file():
    """embeddings.pkl ka mtime update karo — file-watcher detect karega."""
    try:
        path = config.EMBEDDINGS_FILE
        if os.path.exists(path):
            os.utime(path, None)
    except Exception as e:
        print(f"[portal] embeddings touch failed: {e}")


# =============================================================================
#  ADMIN API — EMPLOYEES
# =============================================================================
@portal.route("/api/portal/employees", methods=["GET"])
@admin_required
def api_list_employees():
    return jsonify(auth_db.list_employees_full())


@portal.route("/api/portal/employees/<int:emp_id>", methods=["GET"])
@admin_required
def api_get_employee(emp_id):
    e = auth_db.get_employee(emp_id)
    return (jsonify(e) if e else (jsonify({"error": "not found"}), 404))


@portal.route("/api/portal/employees", methods=["POST"])
@admin_required
def api_add_employee():
    f    = request.form
    name = (f.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    emp_id = auth_db.add_employee(
        name=name,
        department=f.get("department") or None,
        designation=f.get("designation") or None,
        email=f.get("email") or None,
        phone=f.get("phone") or None,
        join_date=f.get("join_date") or None,
        photo_path=None,
    )

    db.sync_roster([name])

    photos = request.files.getlist("photos[]")
    if not photos or not any(p.filename for p in photos):
        single = request.files.get("photo")
        if single and single.filename:
            photos = [single]

    result   = _process_photos_and_embed(photos, name, emp_id, clear_existing=False)
    warnings = result["warnings"]

    if result["first_photo"]:
        auth_db.update_employee(emp_id, photo_path=result["first_photo"])

    username = (f.get("username") or "").strip()
    password = f.get("password") or ""
    if username:
        if len(password) < 6:
            warnings.append("Login NOT created: password must be at least 6 characters.")
        else:
            try:
                auth_db.create_login(username, password, role="employee", emp_name=name)
            except Exception:
                warnings.append("Login NOT created: username already taken.")

    resp = {
        "ok":       True,
        "id":       emp_id,
        "photos":   result["saved"],
        "embedded": result["embedded"],
        "failed":   result["failed"],
    }
    if warnings:
        resp["warnings"] = warnings
    return jsonify(resp)


@portal.route("/api/portal/employees/<int:emp_id>", methods=["PUT"])
@admin_required
def api_update_employee(emp_id):

    f = request.form if request.form else (request.get_json(silent=True) or {})

    fields = {k: f.get(k) for k in
          ("name", "department", "designation", "email", "phone",
           "join_date", "active", "username", "password")
          if f.get(k) is not None}
    # Password ko hash karke database mein save karo
    if "password" in fields:
        password = str(fields["password"]).strip()

    if password:
        fields["password"] = generate_password_hash(password)
    else:
        fields.pop("password")
    # Date fields mein empty string → None
    for date_field in ("join_date", "resignation"):
        if date_field in fields and fields[date_field] == "":
            fields[date_field] = None

    # Active conversion
    if "active" in fields:
        fields["active"] = str(fields["active"]).lower() in (
            "1", "true", "yes"
        )

    # Password ko hash karke save karo
    if "password" in fields:
        password = str(fields["password"]).strip()

        if password:
            from werkzeug.security import generate_password_hash
            fields["password"] = generate_password_hash(password)
        else:
            # Empty password aaye to existing password ko change mat karo
            fields.pop("password")

    photos = request.files.getlist("photos[]") if request.files else []

    if not photos or not any(p.filename for p in photos):
        single = request.files.get("photo") if request.files else None

        if single and single.filename:
            photos = [single]

    resp = {"ok": True}

    if photos and any(p.filename for p in photos):

        emp = auth_db.get_employee(emp_id)

        if emp:
            result = _process_photos_and_embed(
                photos,
                emp["name"],
                emp_id,
                clear_existing=False
            )

            if result["first_photo"] and not emp.get("photo_path"):
                fields["photo_path"] = result["first_photo"]

            resp["photos"] = result["saved"]
            resp["embedded"] = result["embedded"]
            resp["failed"] = result["failed"]

            if result["warnings"]:
                resp["warnings"] = result["warnings"]

    auth_db.update_employee(emp_id, **fields)

    return jsonify(resp)

@portal.route("/api/portal/employees/<int:emp_id>/regenerate-embedding", methods=["POST"])
@admin_required
def api_regen_embedding(emp_id):
    if not FACE_EMBEDDING_AVAILABLE:
        return jsonify({
            "error": "Face recognition is not available on this server. "
                     "Jetson ka photo_sync.py automatically embeddings generate karta hai."
        }), 503

    emp = auth_db.get_employee(emp_id)
    if not emp:
        return jsonify({"error": "Employee not found"}), 404

    safe    = _safe_name(emp["name"])
    emp_dir = os.path.join(config.PHOTOS_DIR, safe)
    IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

    all_photos = []
    if os.path.isdir(emp_dir):
        for fname in sorted(os.listdir(emp_dir)):
            fpath = os.path.join(emp_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(IMG_EXT):
                all_photos.append(fpath)
        for sub in sorted(os.listdir(emp_dir)):
            sub_path = os.path.join(emp_dir, sub)
            if os.path.isdir(sub_path):
                for fname in sorted(os.listdir(sub_path)):
                    fpath = os.path.join(sub_path, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(IMG_EXT):
                        all_photos.append(fpath)

    if not all_photos:
        return jsonify({"error": "Koi photo nahi mili"}), 400

    _fe.remove_from_pickle(emp["name"])
    embedded, failed, warnings = 0, 0, []

    for fpath in all_photos[:MAX_PHOTOS]:
        try:
            emb = _fe.generate_embedding(fpath)
            if emb is None:
                failed += 1
                warnings.append(f"{os.path.basename(fpath)}: face detect nahi hua")
            else:
                _fe.add_to_pickle(emp["name"], emb)
                embedded += 1
        except Exception as e:
            failed += 1
            warnings.append(f"{os.path.basename(fpath)}: {e}")

    if embedded == 0:
        return jsonify({"error": "Kisi photo mein face nahi mila.", "warnings": warnings}), 422

    _touch_embeddings_file()
    return jsonify({
        "ok":       True,
        "embedded": embedded,
        "failed":   failed,
        "warnings": warnings if warnings else None,
        "message":  f"{embedded} embedding(s) generate ho gayi. Camera reload ho jayega.",
    })


@portal.route("/api/portal/employees/<int:emp_id>", methods=["DELETE"])
@admin_required
def api_delete_employee(emp_id):
    hard = request.args.get("hard") == "1"
    emp  = auth_db.get_employee(emp_id)
    if hard:
        auth_db.delete_employee(emp_id)
        if emp and FACE_EMBEDDING_AVAILABLE:
            try:
                _fe.remove_from_pickle(emp["name"])
                _touch_embeddings_file()
            except Exception:
                pass
    else:
        auth_db.deactivate_employee(emp_id)
    return jsonify({"ok": True, "hard_deleted": hard})


# =============================================================================
#  ADMIN API — ATTENDANCE
# =============================================================================
@portal.route("/api/portal/attendance", methods=["GET"])
@admin_required
def api_admin_attendance():
    day  = request.args.get("date", date.today().isoformat())
    rows = db.read_day_rows(day)

    latest = {}
    for r in sorted(rows, key=lambda x: x.get("time", "")):
        latest[r["name"]] = r
    roster, _ = db.get_roster()

    marked_names = set(latest.keys())
    unmarked     = [n for n in roster if n not in marked_names]

    counts = defaultdict(int)
    for r in latest.values():
        counts[r.get("status", "")] += 1

    return jsonify({
        "date": day,
        "marked": [
            {"name": r["name"], "status": r.get("status", ""),
             "time": r.get("time", ""), "camera": r.get("camera", ""),
             "photo": r.get("photo")}
            for r in latest.values()
        ],
        "unmarked": unmarked,
        "counts": {
            "present":  counts.get("Present", 0),
            "wfh":      counts.get("WFH", 0),
            "leave":    counts.get("Leave", 0),
            "absent":   counts.get("Absent", 0),
            "halfday":  counts.get("HalfDay", 0),
            "exited":   counts.get("Exit", 0),
            "unmarked": len(unmarked),
            "roster":   len(roster),
        },
    })


@portal.route("/api/portal/attendance/mark", methods=["POST"])
@admin_required
def api_admin_mark():
    d = request.get_json(force=True)
    try:
        auth_db.manual_mark(
            d["name"], d["status"],
            on_date=d.get("date"), in_time=d.get("in_time"),
            marked_by=session.get("username", "admin"),
        )
        return jsonify({"ok": True})
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


# =============================================================================
#  MONTHLY REPORT
# =============================================================================
def _calendar_working_dates(month, today):
    y, m = int(month[:4]), int(month[5:7])
    days_in = calendar.monthrange(y, m)[1]
    out = []
    for d in range(1, days_in + 1):
        dt = date(y, m, d)
        ds = dt.strftime("%Y-%m-%d")
        if ds > today:
            break
        if dt.weekday() in config.WEEKEND_DAYS:
            continue
        out.append(ds)
    return out


def _build_monthly(month, only_name=None):
    if not month:
        month = datetime.now().strftime("%Y-%m")
    month_rows = db.read_month_rows(month)
    today      = datetime.now().strftime("%Y-%m-%d")

    if config.WORKING_DAYS_MODE == "calendar":
        working_dates = _calendar_working_dates(month, today)
    else:
        working_dates = sorted({r["date"] for r in month_rows})
    working_set = set(working_dates)

    present_by = defaultdict(set)
    io_map     = defaultdict(lambda: defaultdict(
        lambda: {"ins": [], "outs": [], "status": "Present"}
    ))
    for r in month_rows:
        nm, dt = r["name"], r["date"]
        st     = (r.get("status") or "")
        present_by[nm].add(dt)
        cell = io_map[nm][dt]
        if st == "Exit":
            cell["outs"].append(r.get("time", ""))
        else:
            cell["ins"].append(r.get("time", ""))
            if st in ("WFH", "Leave", "Absent", "HalfDay"):
                cell["status"] = st

    days_by = defaultdict(dict)
    for nm, days in io_map.items():
        for dt, io in days.items():
            first_in = min(io["ins"]) if io["ins"] else (min(io["outs"]) if io["outs"] else "")
            last_out = max(io["outs"]) if io["outs"] else ""
            days_by[nm][dt] = {
                "in":     first_in or "—",
                "out":    last_out or "—",
                "status": io["status"],
            }

    roster, source = db.get_roster()
    everyone       = sorted(set(roster) | set(present_by.keys()))
    if only_name:
        everyone = [n for n in everyone if n == only_name]

    wd = len(working_dates)
    employees, sum_pct, low = [], 0.0, 0
    for name in everyone:
        pres = sorted(d for d in present_by.get(name, set()) if d in working_set)
        pd   = len(pres)
        ad   = max(wd - pd, 0)
        pct  = round(pd / wd * 100, 1) if wd else 0.0
        sum_pct += pct
        if pct < LOW_THRESHOLD and wd > 0:
            low += 1
        employees.append({
            "name":           name,
            "photo":          db.get_employee_photo(name),
            "present_days":   pd,
            "absent_days":    ad,
            "attendance_pct": pct,
            "present_dates":  pres,
            "days":           days_by.get(name, {}),
        })

    return {
        "month":            month,
        "today":            today,
        "mode":             config.WORKING_DAYS_MODE,
        "source":           source,
        "working_days":     wd,
        "working_dates":    working_dates,
        "employees":        employees,
        "available_months": db.available_months(),
        "summary": {
            "total_employees": len(everyone),
            "avg_attendance":  round(sum_pct / len(everyone), 1) if everyone else 0.0,
            "low_attendance":  low,
        },
    }


@portal.route("/api/portal/monthly", methods=["GET"])
@admin_required
def api_admin_monthly():
    return jsonify(_build_monthly(request.args.get("month", "").strip()))


@portal.route("/api/portal/employees/<int:emp_id>/monthly", methods=["GET"])
@admin_required
def api_employee_monthly(emp_id):
    emp = auth_db.get_employee(emp_id)
    if not emp:
        return jsonify({"error": "not found"}), 404
    data  = _build_monthly(request.args.get("month", "").strip(), only_name=emp["name"])
    stats = data["employees"][0] if data["employees"] else {
        "name": emp["name"], "photo": db.get_employee_photo(emp["name"]),
        "present_days": 0, "absent_days": 0, "attendance_pct": 0.0,
        "present_dates": [], "days": {},
    }
    return jsonify({
        "month":         data["month"],
        "working_days":  data["working_days"],
        "working_dates": data["working_dates"],
        "stats":         stats,
    })


# =============================================================================
#  ADMIN API — REQUESTS
# =============================================================================
@portal.route("/api/portal/requests", methods=["GET"])
@admin_required
def api_admin_requests():
    return jsonify(auth_db.list_requests(request.args.get("status", "pending")))


@portal.route("/api/portal/requests/<int:req_id>/review", methods=["POST"])
@admin_required
def api_admin_review(req_id):
    d = request.get_json(force=True)
    if d.get("decision") not in ("approved", "rejected"):
        return jsonify({"error": "decision must be approved/rejected"}), 400
    ok = auth_db.review_request(req_id, d["decision"], session.get("username"))
    return (jsonify({"ok": True}) if ok else (jsonify({"error": "not found"}), 404))


# =============================================================================
#  EMPLOYEE API
# =============================================================================
@portal.route("/api/portal/me", methods=["GET"])
@employee_required
def api_me():
    name = session.get("emp_name")
    emps = auth_db.list_employees_full()
    me   = next((e for e in emps if e["name"] == name), None)
    return (jsonify(me) if me else (jsonify({"error": "no employee record linked"}), 404))


@portal.route("/api/portal/me/monthly", methods=["GET"])
@employee_required
def api_me_monthly():
    return jsonify(_build_monthly(request.args.get("month", "").strip(),
                                  only_name=session.get("emp_name")))


@portal.route("/api/portal/me/requests", methods=["GET"])
@employee_required
def api_me_requests():
    return jsonify(auth_db.list_requests_for(session.get("emp_name")))


@portal.route("/api/portal/me/requests", methods=["POST"])
@employee_required
def api_me_submit_request():
    d = request.get_json(force=True)
    if d.get("type") not in ("wfh", "leave") or not d.get("date"):
        return jsonify({"error": "type must be wfh/leave and date required"}), 400
    auth_db.submit_request(session.get("emp_name"), d["date"], d["type"],
                           d.get("reason", ""))
    return jsonify({"ok": True})