"""Shared helpers for the experiment scripts (path setup, JSON I/O, resumable per-unit caching)."""
import json
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

from main import to_jsonable  # noqa: E402,F401


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    os.replace(tmp, path)          # atomic: a crash never leaves a half-written cache file


def cached(path, fn, resume=True):
    """Return the cached result at ``path`` if present (resume), else compute ``fn()`` and cache it."""
    if resume and os.path.exists(path):
        return load_json(path)
    r = fn()
    save_json(r, path)
    return r
