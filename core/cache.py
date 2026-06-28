"""
Cache dokumen terproses berdasarkan hash file.

Menyimpan tuple (chunks, metadatas, ids, embeddings) agar dokumen yang tidak berubah
tidak perlu diekstrak dan di-embedding ulang pada proses ingest berikutnya.
"""

import hashlib
import logging
import pickle
from pathlib import Path


CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_cache_path(file_hash):
    ensure_cache_dir()
    return CACHE_DIR / f"{file_hash}.pkl"


def load_document_cache(file_path):
    file_hash = compute_file_hash(file_path)
    cache_path = get_cache_path(file_hash)
    if not cache_path.exists():
        return None, file_hash

    try:
        with open(cache_path, "rb") as file_handle:
            return pickle.load(file_handle), file_hash
    except Exception as exc:
        logging.warning(f"Gagal memuat cache dokumen {file_path}: {exc}")
        return None, file_hash


def save_document_cache(file_hash, payload):
    cache_path = get_cache_path(file_hash)
    try:
        with open(cache_path, "wb") as file_handle:
            pickle.dump(payload, file_handle)
    except Exception as exc:
        logging.warning(f"Gagal menyimpan cache dokumen {file_hash}: {exc}")