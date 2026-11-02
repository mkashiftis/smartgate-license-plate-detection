"""
SQLite database for the society: flats, the vehicles registered to them,
and a log of every vehicle seen at the gate.

    flats     (flat_no)
    vehicles  (plate PRIMARY KEY, owner, flat_no -> flats)
    entries   (time, plate, owner, flat_no, allowed, plate_type, source)

A plate is the primary key, so one vehicle can belong to only ONE flat,
while one flat can have several vehicles.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime

from .ocr import clean, correct_indian_plate

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "society.db")
_lock = threading.Lock()


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS flats (
            flat_no TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS vehicles (
            plate    TEXT PRIMARY KEY,
            owner    TEXT NOT NULL,
            flat_no  TEXT NOT NULL REFERENCES flats(flat_no),
            added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            time       TEXT NOT NULL,
            plate      TEXT NOT NULL,
            owner      TEXT,
            flat_no    TEXT,
            allowed    INTEGER NOT NULL,
            plate_type TEXT,
            source     TEXT
        );
        """)


def normalise_plate(text: str) -> tuple[str, bool]:
    """Store plates without spaces, e.g. 'mh 02 ab 1234' -> 'MH02AB1234'."""
    r = correct_indian_plate(text)
    return (r["plate"], True) if r["valid"] else (clean(text), False)


def pretty(plate: str) -> str:
    r = correct_indian_plate(plate)
    return r["formatted"] if r["valid"] else plate


def add_vehicle(plate: str, owner: str, flat_no: str) -> dict:
    plate_n, valid = normalise_plate(plate)
    owner = (owner or "").strip()
    flat_no = (flat_no or "").strip().upper()
    if not plate_n or not owner or not flat_no:
        return {"ok": False, "error": "Plate, owner and flat are all required."}
    with _lock, _conn() as c:
        row = c.execute("SELECT flat_no, owner FROM vehicles WHERE plate=?", (plate_n,)).fetchone()
        if row:
            return {"ok": False,
                    "error": f"{pretty(plate_n)} is already registered to {row['owner']} (flat {row['flat_no']}). "
                             "A vehicle can belong to only one flat - remove it first to re-assign it."}
        c.execute("INSERT OR IGNORE INTO flats(flat_no) VALUES (?)", (flat_no,))
        c.execute("INSERT INTO vehicles(plate, owner, flat_no, added_at) VALUES (?,?,?,?)",
                  (plate_n, owner, flat_no, datetime.now().isoformat(timespec="seconds")))
    warn = None if valid else "This doesn't look like a standard Indian plate (e.g. MH 02 AB 1234); saved anyway."
    return {"ok": True, "plate": plate_n, "formatted": pretty(plate_n), "warning": warn}


def remove_vehicle(plate: str) -> bool:
    plate_n, _ = normalise_plate(plate)
    with _lock, _conn() as c:
        cur = c.execute("DELETE FROM vehicles WHERE plate=?", (plate_n,))
        c.execute("DELETE FROM flats WHERE flat_no NOT IN (SELECT flat_no FROM vehicles)")
        return cur.rowcount > 0


def find(plate: str):
    with _lock, _conn() as c:
        row = c.execute("SELECT plate, owner, flat_no FROM vehicles WHERE plate=?", (plate,)).fetchone()
        return dict(row) if row else None


def list_vehicles():
    with _lock, _conn() as c:
        rows = c.execute("SELECT plate, owner, flat_no, added_at FROM vehicles ORDER BY flat_no, owner").fetchall()
    return [dict(r, formatted=pretty(r["plate"])) for r in rows]


def log_entry(plate, owner, flat_no, allowed, plate_type, source):
    with _lock, _conn() as c:
        c.execute("INSERT INTO entries(time, plate, owner, flat_no, allowed, plate_type, source) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), plate, owner, flat_no,
                   1 if allowed else 0, plate_type, source))


def list_entries(limit=200):
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM entries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r, formatted=pretty(r["plate"])) for r in rows]


def clear_entries():
    with _lock, _conn() as c:
        c.execute("DELETE FROM entries")
