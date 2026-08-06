# HR Attendance System — full project

Face-recognition attendance (entry + exit cameras) → Supabase Postgres → live
dashboard + **login-gated admin & employee portal**.

This is the complete fresh project. Nothing needs hand-merging — the login/
admin portal is already wired into `server.py`.

---

## What's in here

```
HR_attendance/
├── server.py                 # Flask app: live dashboard + REST API + PORTAL (login/admin/employee)
├── db.py                     # Postgres/Supabase access + offline outbox  (your file, unchanged)
├── config.py                 # shared settings (DATABASE_URL, paths, etc.) (your file, unchanged)
├── schema.sql                # Supabase schema + the login/admin migrations
├── dashboard.html            # your existing public live dashboard
├── entry_cameras.py          # 4 entry cameras → mark Present   (your file, unchanged)
├── exit_camera.py            # exit camera → mark Exit           (your file, unchanged)
│
├── auth_db.py                # NEW: users/requests tables + admin writes (uses db.get_conn())
├── portal.py                 # NEW: login + admin panel + employee panel (Flask blueprint)
├── face_embedding.py         # NEW: photo → buffalo_l embedding → embeddings.pkl (+ Postgres)
├── templates/
│   ├── login.html            # NEW
│   ├── admin_dashboard.html  # NEW
│   └── employee_dashboard.html # NEW
│
├── requirements.txt
├── Procfile                  # for Render
└── .gitignore
```

---

## STEP 1 — Supabase database (one time)

1. Open your Supabase project → SQL Editor.
2. Paste the whole of `schema.sql` and run it. This creates `employees`,
   `attendance`, `cameras`, `face_embeddings`, **`users`**, **`requests`**,
   adds `email/phone/join_date` to employees, and extends the status enum with
   `WFH/Leave/Absent/HalfDay`.
   *(If you already ran the old schema before, running this again is safe — it's
   all `IF NOT EXISTS` / `ADD VALUE IF NOT EXISTS`.)*

---

## STEP 2 — environment variables

```bash
# your Supabase Transaction Pooler string (port 6543)
export DATABASE_URL='DATABASE_URL="postgresql://postgres.irssbmmqytirucedrknn:Cyamsys%40123@aws-1-ap-south-1.pooler.supabase.com:6543/postgres?pgbouncer=true"'

# a long random string for login sessions (REQUIRED for the portal)
export SECRET_KEY='paste-any-long-random-string-here'

# only on the camera machine (DGX) — where embeddings.pkl lives:
export EMBEDDINGS_FILE='/home/cyamsys/Downloads/HR_face/embeddings/embeddings.pkl'
export PHOTOS_DIR='/home/cyamsys/Downloads/HR_face/photos'
```

Tip: put these in a file called `.env` and `source .env` before running, or set
them in the systemd unit / Render dashboard. (`.env` is gitignored.)

---

## STEP 3 — install dependencies

```bash
cd HR_attendance
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

On a **dashboard-only** box (e.g. Render) you can comment out the face-recognition
lines in `requirements.txt` (numpy/opencv/insightface/onnxruntime) to keep the
build small — the portal + dashboard don't need them. You only need those on the
DGX/camera machine (and for the admin "add employee → generate embedding" step).

---

## STEP 4 — run the server

```bash
python server.py
```

You'll see:
```
  Local   : http://localhost:5400
  Portal  : http://localhost:5400/portal/login
```

Open:
- `http://localhost:5400/` — public live dashboard (as before)
- `http://localhost:5400/portal/login` — login screen

**First-time admin login:** `admin` / `admin123`
→ change it immediately (Admin panel → or POST `/api/portal/change-password`).

---

## STEP 5 — run the cameras (on the DGX)

Separate processes, same folder, same `DATABASE_URL`:

```bash
python entry_cameras.py     # entry gates → marks Present
python exit_camera.py       # exit gate   → marks Exit
```

These are unchanged from your setup — they write into the same `attendance`
table the dashboard and portal read.

---

## How the portal fits together

- **Public dashboard** (`/`) — untouched, live face-recognition view.
- **Admin panel** (`/portal/admin`):
  - *Today* — present/WFH/leave/absent/exited/unmarked counts + quick-mark
    anyone for any date (writes to `attendance`, `camera_id='manual:<admin>'`).
  - *Employees* — add (name, dept, designation, email, phone, join date, photo,
    optional login), edit, deactivate, delete, regenerate face embedding.
  - *Monthly* — Summary + Day-by-Day, same working-days logic as your dashboard.
  - *Requests* — approve/reject WFH/leave; approving auto-marks that day.
- **Employee panel** (`/portal/employee`) — own profile, own monthly attendance,
  own WFH/leave requests. Server-side scoped: an employee can't see anyone else.

### Add-employee → live cameras

When an admin adds an employee **with a photo**, `face_embedding.py` runs
**buffalo_l** (same model your cameras use) and appends the vector to
`embeddings.pkl` in the `{name: [emb,...]}` shape `entry_cameras.py` loads, plus
a backup in the `face_embeddings` Postgres table, and ensures the roster row is
active. To make the new face recognizable on the cameras, reload `embeddings.pkl`:
`sudo systemctl restart entry-cameras.service` (and exit), or add a mtime-based
hot-reload in `entry_cameras.py` (see the note at the bottom of `face_embedding.py`).

> Run the admin panel on the **DGX** for enrollment — that's where `embeddings.pkl`
> and `PHOTOS_DIR` physically live (and where the GPU is). Render has neither.

---

## Deploy to Render (dashboard + portal)

1. Push this folder to a Git repo.
2. Render → New Web Service → point at the repo.
3. Environment tab: set `DATABASE_URL` and `SECRET_KEY`.
4. Render auto-detects the `Procfile` (`gunicorn server:app ...`).

Note: employee-photo enrollment should still be done on the DGX (see above);
Render can serve the dashboard, viewing, and manual marking fine.

---

## Security checklist

- Set a real `SECRET_KEY` (done in Step 2). Never commit it.
- Change the default `admin` / `admin123` on first login.
- You're behind Tailscale Funnel already — keep the portal on that private URL.
- Photo upload checks extension; if ever exposed publicly, also verify the file
  is a real image before saving.
