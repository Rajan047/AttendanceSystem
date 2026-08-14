-- =============================================================================
--  ATTENDANCE SYSTEM — POSTGRES SCHEMA (Supabase)
--  Run once in the Supabase SQL Editor (or via psql).
-- =============================================================================

-- ---------------------------------------------------------------------------
--  STATUS ENUM  (Postgres has real enum types, unlike MySQL's inline ENUM)
--  NOTE: the login/admin add-on (auth_db.py) automatically extends this enum
--  with 'WFH', 'Leave', 'Absent', 'HalfDay' on first run. You can also add
--  them manually here (see the ALTER TYPE lines at the bottom of this file).
-- ---------------------------------------------------------------------------
DO $$ BEGIN
    CREATE TYPE attendance_status AS ENUM ('Present', 'Exit');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ---------------------------------------------------------------------------
--  EMPLOYEES  — roster (source of truth for "who is registered")
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS employees (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(150) NOT NULL UNIQUE,
    department  VARCHAR(100) DEFAULT NULL,
    designation VARCHAR(100) DEFAULT NULL,
    photo_path  VARCHAR(255) DEFAULT NULL,   -- profile photo (relative to /photos)
    active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
--  CAMERAS  — optional registry of RTSP entry/exit points
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    id          VARCHAR(50) PRIMARY KEY,      -- e.g. "Entry-1"
    location    VARCHAR(150) DEFAULT NULL,
    rtsp_url    VARCHAR(255) DEFAULT NULL,
    active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
--  ATTENDANCE  — every recognized-and-marked event
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance (
    id           BIGSERIAL PRIMARY KEY,
    employee_id  INT          NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    name         VARCHAR(150) NOT NULL,        -- denormalised copy, fast reads
    att_date     DATE         NOT NULL,
    att_time     TIME         NOT NULL,
    status       attendance_status NOT NULL DEFAULT 'Present',
    camera_id    VARCHAR(50)  DEFAULT NULL,
    confidence   REAL         DEFAULT NULL,
    photo_path   VARCHAR(255) DEFAULT NULL,    -- captured snapshot for this event
    created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_att_date          ON attendance (att_date);
CREATE INDEX IF NOT EXISTS idx_att_name_date      ON attendance (name, att_date);
CREATE INDEX IF NOT EXISTS idx_att_employee_date  ON attendance (employee_id, att_date);

-- ---------------------------------------------------------------------------
--  FACE EMBEDDINGS  — optional: move embeddings.pkl into Postgres too.
--  The admin panel stores a backup copy of each added employee's embedding
--  here (as float32[512] bytes), in addition to embeddings.pkl.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_embeddings (
    id          SERIAL PRIMARY KEY,
    employee_id INT NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    embedding   BYTEA NOT NULL,     -- serialized float32[512] vector
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
--  RECOMMENDED: client_uuid for safe offline-sync de-duplication
--  (db.py's offline outbox uses this to avoid duplicate rows on retry)
-- ---------------------------------------------------------------------------
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS client_uuid TEXT UNIQUE;

-- =============================================================================
--  LOGIN + ADMIN ADD-ON MIGRATIONS
--  These are also applied automatically by auth_db.init_auth_schema() on
--  first server startup, so running them here is OPTIONAL — but doing it once
--  up front is cleaner (no DDL privileges needed at boot).
-- =============================================================================

-- extra employee fields the admin panel collects
ALTER TABLE employees ADD COLUMN IF NOT EXISTS email     VARCHAR(150);
ALTER TABLE employees ADD COLUMN IF NOT EXISTS phone     VARCHAR(40);
ALTER TABLE employees ADD COLUMN IF NOT EXISTS join_date DATE;

-- allow manual HR statuses in the same attendance table the dashboard reads
ALTER TYPE attendance_status ADD VALUE IF NOT EXISTS 'WFH';
ALTER TYPE attendance_status ADD VALUE IF NOT EXISTS 'Leave';
ALTER TYPE attendance_status ADD VALUE IF NOT EXISTS 'Absent';
ALTER TYPE attendance_status ADD VALUE IF NOT EXISTS 'HalfDay';

-- login accounts (username/password/role, linked to employees by id + name)
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(150) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          VARCHAR(20)  NOT NULL DEFAULT 'employee',
    emp_id        INT,                     -- links to employees.id
    emp_name      VARCHAR(150),            -- denormalised copy for quick lookups
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- WFH / leave request queue
CREATE TABLE IF NOT EXISTS requests (
    id          SERIAL PRIMARY KEY,
    emp_name    VARCHAR(150) NOT NULL,
    req_date    DATE NOT NULL,
    type        VARCHAR(20) NOT NULL,
    reason      TEXT,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    reviewed_by VARCHAR(150),
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
