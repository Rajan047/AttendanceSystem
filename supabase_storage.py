"""
supabase_storage.py — Supabase Storage upload/delete helpers.
portal.py import karta hai isse.
"""
import os
import requests

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://bjgiirtpqfjaaxnftptf.supabase.co"
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJqZ2lpcnRwcWZqYWF4bmZ0cHRmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODU5OTM5NTMsImV4cCI6MjEwMTU2OTk1M30.-HBEfhHPJ6KhFUwhGad90mWLiJzQe7Ylw2pMRhxVFaU"
)
BUCKET = "employee-photos"

_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}


def upload_photo(file_bytes, storage_path, content_type="image/jpeg"):
    """
    Supabase Storage mein photo upload karo.
    storage_path example: "Rajan/profile_abc123.jpg"
    Returns public URL ya None on failure.
    """
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{storage_path}"
    headers = {**_HEADERS, "Content-Type": content_type, "x-upsert": "true"}
    resp = requests.put(url, headers=headers, data=file_bytes, timeout=30)
    if resp.status_code in (200, 201):
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
        return public_url
    print(f"[storage] upload failed: {resp.status_code} {resp.text}")
    return None


def delete_photo(storage_path):
    """Supabase Storage se photo delete karo."""
    url  = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{storage_path}"
    resp = requests.delete(url, headers=_HEADERS, timeout=10)
    return resp.status_code in (200, 204)


def public_url(storage_path):
    """Storage path se public URL banao."""
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"