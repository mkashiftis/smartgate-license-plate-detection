"""
Reading the characters on the plate.

Two independent readers run on every plate:

1. EasyOCR -- a ready-made text reader. Reliable on real plates.
2. Template matching -- pure syllabus. Each segmented character is compared
   with reference images of 0-9 and A-Z, and the one at the smallest
   Euclidean distance wins (distance measures, Unit 1).

Both raw readings then go through `correct_indian_plate`, which uses the
fixed layout of Indian plates (e.g. MH 02 AB 1234) to fix the usual OCR
confusions: O/0, I/1, S/5, B/8, Z/2, G/6.
"""

from __future__ import annotations

import os
import re
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
TPL_W, TPL_H = 24, 40

# ---------------------------------------------------------------------------
# Indian plate format
# ---------------------------------------------------------------------------

STATE_CODES = set("""AN AP AR AS BR CH CG DD DL DN GA GJ HR HP JK JH KA KL LA LD MP MH MN ML MZ
NL OD OR PY PB RJ SK TN TS TR UP UK UA WB""".split())

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "7",
            "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


def clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _fit(chunk: str, kind: str):
    """Force every char of chunk to be a letter ('L') or digit ('D').
    Returns (fixed, n_substitutions) or None if impossible."""
    out, cost = [], 0
    for ch in chunk:
        if kind == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch]); cost += 1
            else:
                return None
        else:
            if ch.isalpha():
                out.append(ch)
            elif ch in TO_LETTER:
                out.append(TO_LETTER[ch]); cost += 1
            else:
                return None
    return "".join(out), cost


def _parse_standard(s: str):
    """SS RR LLL NNNN : state(2 letters) rto(1-2 digits) series(0-3 letters) number(4 digits)."""
    best = None
    n = len(s)
    for rto in (2, 1):
        for series in (2, 1, 3, 0):
            if 2 + rto + series + 4 != n:
                continue
            parts = [(s[0:2], "L"), (s[2:2 + rto], "D"),
                     (s[2 + rto:2 + rto + series], "L"), (s[2 + rto + series:], "D")]
            fixed, cost = [], 0
            for chunk, kind in parts:
                r = _fit(chunk, kind)
                if r is None:
                    break
                fixed.append(r[0]); cost += r[1]
            else:
                if fixed[0] not in STATE_CODES:
                    cost += 3
                if rto == 1:
                    cost += 1
                cand = (cost, " ".join(p for p in fixed if p), "".join(fixed))
                if best is None or cand[0] < best[0]:
                    best = cand
    return best


def _parse_bharat(s: str):
    """Bharat series: YY BH NNNN L(L)  e.g. 22 BH 1234 AA."""
    if len(s) not in (9, 10):
        return None
    parts = [(s[0:2], "D"), (s[2:4], "L"), (s[4:8], "D"), (s[8:], "L")]
    fixed, cost = [], 0
    for chunk, kind in parts:
        r = _fit(chunk, kind)
        if r is None:
            return None
        fixed.append(r[0]); cost += r[1]
    if fixed[1] != "BH":
        return None
    return cost, " ".join(fixed), "".join(fixed)


def correct_indian_plate(raw: str) -> dict:
    """
    Returns {"plate": "MH02AB1234", "formatted": "MH 02 AB 1234",
             "valid": bool, "fixes": int, "raw": cleaned_raw}
    Tries dropping up to 3 stray characters at either end (e.g. 'IND' from
    the blue strip), each costing 1.
    """
    s = clean(raw)
    if s.startswith("IND") and len(s) > 10:
        s = s[3:]
    best = None
    for a in range(0, min(4, len(s))):
        for b in range(0, min(4, len(s) - a)):
            sub = s[a:len(s) - b] if b else s[a:]
            for p in (_parse_standard(sub), _parse_bharat(sub)):
                if p is None:
                    continue
                cost = p[0] + a + b
                if best is None or cost < best[0]:
                    best = (cost, p[1], p[2])
    if best is None or best[0] > 5:
        return {"plate": s, "formatted": s, "valid": False, "fixes": 0, "raw": s}
    return {"plate": best[2], "formatted": best[1], "valid": True, "fixes": best[0], "raw": s}


# ---------------------------------------------------------------------------
# Template matching reader (syllabus-only)
# ---------------------------------------------------------------------------

def _norm_glyph(binary_char: np.ndarray) -> np.ndarray:
    """Tight-crop, pad to the template aspect ratio, resize, scale to 0..1."""
    ys, xs = np.nonzero(binary_char)
    if len(xs) == 0:
        return np.zeros((TPL_H, TPL_W), np.float32)
    g = binary_char[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    target_w = int(round(h * TPL_W / TPL_H))
    if w < target_w:
        pad = target_w - w
        g = cv2.copyMakeBorder(g, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=0)
    g = cv2.resize(g, (TPL_W, TPL_H), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


class TemplateReader:
    def __init__(self):
        self.templates: list[tuple[str, np.ndarray]] = []
        fonts = [f for f in sorted(os.listdir(FONT_DIR)) if f.lower().endswith(".ttf")]
        for fname in fonts:
            font = ImageFont.truetype(os.path.join(FONT_DIR, fname), 120)
            for ch in ALNUM:
                img = Image.new("L", (160, 160), 0)
                ImageDraw.Draw(img).text((20, 10), ch, fill=255, font=font)
                arr = np.array(img)
                _, arr = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
                self.templates.append((ch, _norm_glyph(arr)))

    def read_char(self, binary_char: np.ndarray) -> tuple[str, float]:
        g = _norm_glyph(binary_char)
        best_ch, best_d = "?", 1e9
        for ch, t in self.templates:
            d = float(np.sqrt(((g - t) ** 2).sum()))   # Euclidean distance
            if d < best_d:
                best_ch, best_d = ch, d
        return best_ch, best_d

    def read(self, char_images) -> tuple[str, list]:
        out, detail = [], []
        for c in char_images:
            ch, d = self.read_char(c)
            out.append(ch)
            detail.append({"char": ch, "distance": round(d, 2)})
        return "".join(out), detail


# ---------------------------------------------------------------------------
# EasyOCR reader
# ---------------------------------------------------------------------------

class EasyReader:
    def __init__(self):
        self._reader = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.error = None

    def load(self):
        """Load the model once (slow: a few seconds, and a ~100 MB download the
        very first time). Called from a background thread at start-up."""
        with self._load_lock:
            if self._reader is None and self.error is None:
                try:
                    import easyocr  # imported here so the site starts fast
                    self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                except Exception as e:  # missing package or model download failed
                    self.error = str(e)
        return self._reader

    @property
    def ready(self) -> bool:
        return self._reader is not None

    def read(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        reader = self._reader          # never blocks: until loaded, template matching reads alone
        if reader is None:
            return "", 0.0
        h = plate_bgr.shape[0]
        img = plate_bgr
        if h < 100:
            f = 100.0 / h
            img = cv2.resize(plate_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with self._lock:
            res = reader.readtext(gray, allowlist=ALNUM, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        # order by line (top to bottom) then left to right
        Hh = gray.shape[0]
        def key(r):
            ys = [p[1] for p in r[0]]; xs = [p[0] for p in r[0]]
            return (round((sum(ys) / 4) / (Hh * 0.35)), min(xs))
        res = sorted(res, key=key)
        # ignore tiny text (e.g. 'IND' on the blue strip)
        heights = [max(p[1] for p in r[0]) - min(p[1] for p in r[0]) for r in res]
        hmax = max(heights)
        keep = [r for r, hh in zip(res, heights) if hh >= 0.45 * hmax]
        text = "".join(r[1] for r in keep)
        conf = float(np.mean([r[2] for r in keep])) if keep else 0.0
        return clean(text), conf
