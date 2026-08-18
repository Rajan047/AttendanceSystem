"""
=============================================================================
  SHARED CONFIG  —  entry_cameras.py aur server.py dono isi file ko use
  karte hain, taaki DB settings ek hi jagah edit karni padein.
=============================================================================
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
#  POSTGRES DATABASE (Supabase)
# =============================================================================
_THIS_PROJECT_REF    = "bjgiirtpqfjaaxnftptf"
_CORRECT_DATABASE_URL = (
    "postgresql://postgres.bjgiirtpqfjaaxnftptf:Cyamsys%40123"
    "@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
)

_env_url = os.environ.get("DATABASE_URL", "").strip()

# Stale DATABASE_URL from another project → ignore karo
if _env_url and _THIS_PROJECT_REF not in _env_url:
    print(
        f"[config] WARNING: DATABASE_URL env var points to a different "
        f"Supabase project (expected '{_THIS_PROJECT_REF}'). Ignoring it and "
        f"using this project's own database instead. "
        f"If this is unexpected, run: unset DATABASE_URL"
    )
    _env_url = ""

DATABASE_URL = _env_url or _CORRECT_DATABASE_URL

# psycopg2 doesn't understand ?pgbouncer=true (Prisma ke liye hota hai)
if "?" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?", 1)[0]

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL set nahi hai. Supabase Transaction Pooler string set karo:\n"
        "  export DATABASE_URL='postgresql://postgres.xxxx:password@host:6543/postgres'"
    )

# =============================================================================
#  FACE EMBEDDINGS
# =============================================================================
EMBEDDINGS_FILE = os.environ.get(
    "EMBEDDINGS_FILE",
    "/home/cyamsys/Documents/HR_attendance/embeddings/embeddings.pkl",
)

# =============================================================================
#  PHOTO STORAGE
# =============================================================================
PHOTOS_DIR = os.environ.get("PHOTOS_DIR", os.path.join(BASE_DIR, "photos"))
os.makedirs(PHOTOS_DIR, exist_ok=True)

# =============================================================================
#  ATTENDANCE BEHAVIOUR
# =============================================================================
MARK_COOLDOWN_SECONDS = int(os.environ.get("MARK_COOLDOWN_SECONDS", 10))
STATUS_CACHE_TTL       = int(os.environ.get("STATUS_CACHE_TTL", 5))

WORKING_DAYS_MODE = os.environ.get("WORKING_DAYS_MODE", "office_open")
WEEKEND_DAYS      = [5, 6]   # 0=Mon ... 6=Sun

# =============================================================================
#  FLASK SERVER
# =============================================================================
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5500)))