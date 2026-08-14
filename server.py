"""
=============================================================================
  ATTENDANCE DASHBOARD  —  Postgres(Supabase)-backed REST API + dashboard
=============================================================================
"""

import os
import io
import csv
import calendar
from datetime import datetime, date
from collections import defaultdict

from flask import Flask, jsonify, send_file, send_from_directory, request, Response, redirect
from flask_cors import CORS

import config
import db
import auth_db
from portal import portal as portal_bp

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-please-in-production")
CORS(app)

_HERE = os.path.dirname(os.path.abspath(__file__))

db.init_db()

try:
    auth_db.init_auth_schema()
except Exception as e:
    print(f"[auth_db] WARNING: schema init skipped ({e}).")

app.register_blueprint(portal_bp)

_n = db.sync_employee_photos_from_dir()
if _n:
    print(f"[photos] {_n} employee profile photo(s) linked from {config.PHOTOS_DIR}")


# =============================================================================
#  PAYLOAD BUILDERS
# =============================================================================
def _build_payload(target_date):
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    all_dates = db.all_dates()
    day_rows  = db.read_day_rows(target_date)
    profile_photo_by_name = {e["name"]: e["photo_path"] for e in db.list_employees()}

    last_status = {}
    first_seen  = {}
    last_photo  = {}
    for r in sorted(day_rows, key=lambda x: x.get("time", "")):
        last_status[r["name"]] = r.get("status", "")
        first_seen.setdefault(r["name"], r.get("time", ""))
        if r.get("photo"):
            last_photo[r["name"]] = r["photo"]

    inside = sorted([n for n, s in last_status.items() if s == "Present"])

    per_camera = defaultdict(int)
    for r in day_rows:
        per_camera[r.get("camera") or "—"] += 1

    last_entry = None
    if day_rows:
        top = day_rows[0]
        last_entry = {"name": top["name"], "time": top.get("time", ""),
                      "camera": top.get("camera") or "—",
                      "photo": top.get("photo")}

    return {
        "date": target_date,
        "today": datetime.now().strftime("%Y-%m-%d"),
        "server_time": datetime.now().strftime("%H:%M:%S"),
        "available_dates": all_dates,
        "records": [
            {
                "name":          r["name"],
                "time":          r.get("time", ""),
                "status":        r.get("status", ""),
                "camera":        r.get("camera") or "—",
                "first":         first_seen.get(r["name"], ""),
                "photo":         r.get("photo"),
                "profile_photo": profile_photo_by_name.get(r["name"]),
            }
            for r in day_rows
        ],
        "inside": inside,
        "inside_photos": {n: last_photo.get(n) for n in inside},
        "profile_photos": profile_photo_by_name,
        "stats": {
            "present_today":    len(last_status),
            "currently_inside": len(inside),
            "total_entries":    len(day_rows),
            "unique_people":    len(last_status),
        },
        "per_camera": dict(sorted(per_camera.items())),
        "last_entry": last_entry,
        "csv_exists": True,
    }


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


def _build_monthly(month):
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
    io_map = defaultdict(lambda: defaultdict(lambda: {"ins": [], "outs": []}))
    for r in month_rows:
        nm, dt = r["name"], r["date"]
        tm = r.get("time", "")
        st = (r.get("status", "") or "").lower()
        if st in ("present", "exit"):
            present_by[nm].add(dt)
        if st == "exit":
            io_map[nm][dt]["outs"].append(tm)
        else:
            io_map[nm][dt]["ins"].append(tm)

    times_by = defaultdict(dict)
    for nm, days in io_map.items():
        for dt, io in days.items():
            first_in = min(io["ins"]) if io["ins"] else (min(io["outs"]) if io["outs"] else "")
            last_out = max(io["outs"]) if io["outs"] else ""
            times_by[nm][dt] = {"in": first_in, "out": last_out}

    roster, source = db.get_roster()
    everyone = sorted(set(roster) | set(present_by.keys()))

    wd = len(working_dates)
    employees = []
    sum_pct = 0.0
    full = low = 0

    for name in everyone:
        pres_dates = sorted(d for d in present_by.get(name, set()) if d in working_set)
        pd = len(pres_dates)
        ad = max(wd - pd, 0)
        pct = round(pd / wd * 100, 1) if wd else 0.0
        sum_pct += pct
        if ad == 0 and wd > 0:
            full += 1
        if pct < 75 and wd > 0:
            low += 1
        employees.append({
            "name": name,
            "photo": db.get_employee_photo(name),
            "present_days": pd,
            "absent_days": ad,
            "attendance_pct": pct,
            "present_dates": pres_dates,
            "days": times_by.get(name, {}),
        })

    months = db.available_months()

    return {
        "month": month,
        "today": today,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": config.WORKING_DAYS_MODE,
        "source": source,
        "working_days": wd,
        "working_dates": working_dates,
        "roster_size": len(roster),
        "employees": employees,
        "available_months": months,
        "summary": {
            "total_employees": len(everyone),
            "avg_attendance": round(sum_pct / len(everyone), 1) if everyone else 0.0,
            "full_attendance": full,
            "low_attendance": low,
        },
    }


# =============================================================================
#  ROUTES
# =============================================================================
@app.route("/")
def index():
    return send_file(os.path.join(_HERE, "dashboard.html"))


@app.route("/photos/<path:filename>")
def serve_photo(filename):
    """
    Photos Supabase Storage se serve karo.
    Render pe local filesystem mein photos nahi hoti — Supabase public URL pe redirect karo.
    """
    SUPABASE_URL = os.environ.get(
        "SUPABASE_URL", "https://bjgiirtpqfjaaxnftptf.supabase.co"
    )
    BUCKET = "employee-photos"
    return redirect(
        f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{filename}",
        code=302
    )


@app.route("/api/data")
def api_data():
    target_date = request.args.get("date", "").strip()
    return jsonify(_build_payload(target_date))


@app.route("/api/export")
def api_export():
    target_date = request.args.get("date", "").strip() or datetime.now().strftime("%Y-%m-%d")
    rows = db.read_day_rows(target_date)
    rows.sort(key=lambda r: r.get("time", ""))

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Name", "Date", "Time", "Status", "Camera"])
    for r in rows:
        w.writerow([r["name"], target_date, r.get("time", ""),
                    r.get("status", ""), r.get("camera", "")])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{target_date}.csv"},
    )


@app.route("/api/monthly")
def api_monthly():
    month = request.args.get("month", "").strip()
    return jsonify(_build_monthly(month))


@app.route("/api/monthly/export")
def api_monthly_export():
    month = request.args.get("month", "").strip() or datetime.now().strftime("%Y-%m")
    rep = _build_monthly(month)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"Monthly Attendance Report — {month}"])
    w.writerow([f"Working days: {rep['working_days']}", f"Mode: {rep['mode']}",
                f"Generated: {rep['generated']}"])
    w.writerow([])
    w.writerow(["Name", "Present Days", "Absent Days", "Working Days", "Attendance %"])
    for e in rep["employees"]:
        w.writerow([e["name"], e["present_days"], e["absent_days"],
                    rep["working_days"], e["attendance_pct"]])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=monthly_report_{month}.csv"},
    )


@app.route("/api/monthly/grid-export")
def api_monthly_grid_export():
    month = request.args.get("month", "").strip() or datetime.now().strftime("%Y-%m")
    rep = _build_monthly(month)
    dates = rep["working_dates"]

    buf = io.StringIO()
    w = csv.writer(buf)

    def _hdr(d):
        try:
            return d[-2:] + " " + datetime.strptime(d, "%Y-%m-%d").strftime("%a")
        except Exception:
            return d[-2:]

    w.writerow(["Name"] + [_hdr(d) for d in dates] + ["Present", "Absent", "%"])
    for e in rep["employees"]:
        pres = set(e["present_dates"])
        cells = ["P" if d in pres else "A" for d in dates]
        w.writerow([e["name"]] + cells +
                   [e["present_days"], e["absent_days"], e["attendance_pct"]])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=monthly_grid_{month}.csv"},
    )


@app.route("/api/employees", methods=["GET"])
def api_employees():
    return jsonify(db.list_employees())


@app.route("/api/attendance/today", methods=["GET"])
def api_attendance_today():
    return jsonify(db.read_day_rows(datetime.now().strftime("%Y-%m-%d")))


@app.route("/api/attendance", methods=["GET"])
def api_attendance_by_date():
    target_date = request.args.get("date", "").strip() or datetime.now().strftime("%Y-%m-%d")
    return jsonify(db.read_day_rows(target_date))


@app.route("/api/attendance/mark", methods=["POST"])
def api_attendance_mark():
    payload = request.get_json(force=True, silent=True) or {}
    name      = payload.get("name")
    camera_id = payload.get("camera_id", "Manual")
    if not name:
        return jsonify({"error": "name is required"}), 400
    result = db.mark_attendance(name, camera_id)
    return jsonify({"result": result})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


if __name__ == "__main__":
    print("=" * 60)
    print("  ATTENDANCE DASHBOARD  (Postgres / Supabase)")
    print(f"  Photos dir : {config.PHOTOS_DIR}")
    print(f"  Local      : http://localhost:{config.PORT}")
    print(f"  Portal     : http://localhost:{config.PORT}/portal/login")
    print("=" * 60)
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)