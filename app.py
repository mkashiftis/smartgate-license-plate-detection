"""
SmartGate - Society Vehicle Access
==================================

Run:   python app.py            then open http://localhost:8000
       python app.py --camera 1 (if the wrong camera opens)

The camera is read on a background thread; a second thread runs the
image-processing pipeline (lpr/pipeline.py) a few times a second, reads the
plate (lpr/ocr.py), checks it against the society database (lpr/database.py)
and publishes the result to the web page.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import platform
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from lpr import database, pipeline
from lpr.ocr import EasyReader, TemplateReader, correct_indian_plate

SITE_NAME = "SmartGate"
SITE_TAGLINE = "Society Vehicle Access"

VOTES_NEEDED = 3        # a REGISTERED plate must be read this many times ...
VOTES_TO_DENY = 5       # ... an unknown plate this many (a misread must not lock a resident out)
VOTE_WINDOW = 6         # ... within the last this-many reads
VERDICT_HOLD = 6.0      # seconds a green/red result stays on screen
LOG_DEBOUNCE = 60.0     # don't log the same plate twice within this many seconds
MOTION_LIMIT = 0.25     # fraction of changing pixels above which the frame is "moving"
UPLOAD_HOLD = 10.0      # seconds an uploaded photo's result stays before live resumes

app = Flask(__name__)


def jpg_b64(img: np.ndarray, width: int = 380, quality: int = 80) -> str:
    if img is None or img.size == 0:
        return ""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    elif w < width // 2:
        f = (width // 2) / w
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode() if ok else ""


def template_strip(char_images, detail) -> np.ndarray | None:
    """Each segmented character next to the letter template matching chose."""
    if not char_images:
        return None
    tiles = []
    for img, d in zip(char_images, detail):
        h = 64
        w = max(8, int(img.shape[1] * h / max(img.shape[0], 1)))
        g = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        tile = np.full((h + 46, max(w, 40) + 10, 3), 30, np.uint8)
        tile[4:4 + h, 5:5 + w] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, d["char"], (8, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 255), 2)
        cv2.putText(tile, f"{d['distance']:.1f}", (6, h + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        tiles.append(tile)
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    return cv2.hconcat(tiles)


class Gate:
    def __init__(self, camera_index: int, source: str | None = None):
        self.camera_index = camera_index
        self.source = source         # optional video/image file used instead of a webcam
        self.cap = None
        self.frame = None
        self.frame_time = 0.0
        self.camera_ok = False
        self.camera_msg = "Starting camera..."
        self.lock = threading.Lock()

        self.easy = EasyReader()
        self.tpl = TemplateReader()

        self.votes: deque = deque(maxlen=VOTE_WINDOW)
        self.box = None              # plate corners in frame coords
        self.box_time = 0.0
        self.verdict = None          # dict, see _decide
        self.verdict_time = 0.0
        self.last_logged: dict[str, float] = {}
        self.hold_until = 0.0        # live processing paused (after an upload)
        self.upload_frame = None     # shown in the video area while paused
        self.motion = 0.0
        self.readings = {}
        self.steps = []
        self.steps_version = 0
        self.fps = 0.0
        self.reader_status = "loading"
        self._prev_small = None
        self._stop = False
        self._switch = threading.Event()

    # ---------------------------------------------------------------- camera
    def _open(self):
        if self.cap is not None:
            self.cap.release()
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else (
            cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.camera_index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap = cap
        if cap.isOpened():
            self.camera_ok, self.camera_msg = True, f"Camera {self.camera_index} connected"
        else:
            self.camera_ok = False
            self.camera_msg = (f"Camera {self.camera_index} could not be opened. Try another camera number, "
                               "or check camera permission (Mac: System Settings > Privacy & Security > Camera). "
                               "You can still upload photos.")

    def _file_loop(self):
        """Feed a photo or video file in place of the webcam (for testing)."""
        img = cv2.imread(self.source)
        self.camera_ok, self.camera_msg = True, f"Test source: {os.path.basename(self.source)}"
        cap = None if img is not None else cv2.VideoCapture(self.source)
        while not self._stop:
            if img is not None:
                frame = img
            else:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            with self.lock:
                self.frame, self.frame_time = frame.copy(), time.time()
            time.sleep(1 / 25)

    def camera_loop(self):
        if self.source:
            return self._file_loop()
        self._open()
        fails = 0
        while not self._stop:
            if self._switch.is_set():
                self._switch.clear()
                self._open()
            if not self.camera_ok:
                time.sleep(0.5)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    self.camera_ok = False
                    self.camera_msg = "The camera stopped sending frames. Select it again or upload a photo."
                time.sleep(0.03)
                continue
            fails = 0
            with self.lock:
                self.frame = frame
                self.frame_time = time.time()

    def switch_camera(self, index: int):
        self.camera_index = index
        self.camera_ok = False
        self.camera_msg = f"Opening camera {index}..."
        self._switch.set()

    # ------------------------------------------------------------ processing
    def _motion(self, frame) -> np.ndarray:
        """Frame differencing (Unit 8): how much of the picture changed since
        the previous processed frame?"""
        small = cv2.cvtColor(cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1]))),
                             cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if self._prev_small is None or self._prev_small.shape != small.shape:
            self._prev_small = small
            self.motion = 0.0
            return np.zeros_like(small)
        diff = cv2.absdiff(small, self._prev_small)
        self._prev_small = small
        moving = diff > 25
        self.motion = float(moving.mean())
        return cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=4), cv2.COLORMAP_INFERNO)

    def analyse(self, frame: np.ndarray, live: bool, motion_img=None) -> dict:
        res = pipeline.run(frame, with_steps=True)
        readings = {"easy": "", "easy_conf": 0.0, "template": "", "corrected": "", "formatted": "",
                    "valid": False, "fixes": 0, "chars": 0, "reader": self.reader_status,
                    "skipped": ""}
        tpl_strip = None
        if res.best is not None:
            readings["chars"] = res.best.n_chars
            if live and self.easy.error is None and not self.easy.ready:
                readings["skipped"] = "Text reader (EasyOCR) is still loading - one moment"
            elif live and self.motion > MOTION_LIMIT:
                readings["skipped"] = "Plate is moving - hold it still"
            else:
                tpl_text, detail = self.tpl.read(res.char_images)
                tpl_strip = template_strip(res.char_images, detail)
                easy_text, conf = self.easy.read(res.best.crop)
                readings.update(easy=easy_text, easy_conf=round(conf, 2), template=tpl_text,
                                reader=self.reader_status)
                # prefer EasyOCR; fall back to template matching if it gave no valid plate
                ce, ct = correct_indian_plate(easy_text), correct_indian_plate(tpl_text)
                pick = ce if ce["valid"] else (ct if ct["valid"] else ce)
                readings.update(corrected=pick["plate"], formatted=pick["formatted"],
                                valid=pick["valid"], fixes=pick["fixes"],
                                used="EasyOCR" if pick is ce else "Template matching")

        steps = [{"key": s.key, "title": s.title, "unit": s.unit, "text": s.explanation, "img": jpg_b64(s.image)}
                 for s in res.steps]
        if tpl_strip is not None:
            steps.append({"key": "template", "title": "14. Template matching (Euclidean distance)",
                          "unit": "Unit 1 + Unit 6",
                          "text": "Each character is resized to 24x40 and compared with reference images of 0-9 "
                                  "and A-Z. The template at the smallest Euclidean distance wins (the distance is "
                                  "shown under each letter). This is the reader built only from syllabus methods.",
                          "img": jpg_b64(tpl_strip, width=520)})
        if motion_img is not None:
            steps.append({"key": "motion", "title": "15. Frame differencing (motion)", "unit": "Unit 8",
                          "text": f"|frame(t) - frame(t-1)|: temporal correlation between consecutive video frames. "
                                  f"{self.motion * 100:.0f}% of pixels changed. Reading waits until the plate is held "
                                  f"still (below {MOTION_LIMIT * 100:.0f}%).",
                          "img": jpg_b64(motion_img)})
        box = pipeline.box_in_frame(res)
        return {"readings": readings, "steps": steps, "box": box, "plate_type": res.plate_type,
                "plate_type_label": res.plate_type_label}

    def _decide(self, plate: str, plate_type_label: str, source: str):
        rec = database.find(plate)
        now = time.time()
        v = {"plate": plate, "formatted": database.pretty(plate), "plate_type": plate_type_label,
             "time": datetime.now().strftime("%I:%M:%S %p"), "source": source}
        if rec:
            v.update(status="granted", owner=rec["owner"], flat=rec["flat_no"])
        else:
            v.update(status="denied", owner=None, flat=None)
        self.verdict, self.verdict_time = v, now
        if now - self.last_logged.get(plate + v["status"], 0) > LOG_DEBOUNCE or source == "upload":
            database.log_entry(plate, v["owner"], v["flat"], v["status"] == "granted", plate_type_label, source)
            self.last_logged[plate + v["status"]] = now
        return v

    def process_loop(self):
        # load EasyOCR in the background so the page appears immediately
        threading.Thread(target=self._load_reader, daemon=True).start()
        last = 0.0
        while not self._stop:
            if time.time() < self.hold_until:
                time.sleep(0.1)
                continue
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is None:
                time.sleep(0.1)
                continue
            t0 = time.time()
            motion_img = self._motion(frame)
            try:
                out = self.analyse(frame, live=True, motion_img=motion_img)
            except Exception as e:   # never let one bad frame stop the gate
                print("frame skipped:", repr(e))
                time.sleep(0.2)
                continue
            r = out["readings"]

            if out["box"] is not None:
                self.box, self.box_time = out["box"], time.time()
            self.votes.append(r["corrected"] if r["valid"] else None)
            counts = Counter(v for v in self.votes if v)
            top = counts.most_common(1)
            r["votes"] = top[0][1] if top else 0
            needed = VOTES_NEEDED
            if top and database.find(top[0][0]) is None:
                needed = VOTES_TO_DENY
            r["votes_needed"] = needed
            if top and top[0][1] >= needed:
                plate = top[0][0]
                registered = database.find(plate) is not None
                v = self.verdict
                fresh = (v is None or v["plate"] != plate or
                         time.time() - self.verdict_time > VERDICT_HOLD or
                         # registered/removed while the plate was in view
                         (v["status"] != "visitor" and (v["status"] == "granted") != registered))
                if fresh:
                    self._decide(plate, out["plate_type_label"], "camera")
                else:
                    self.verdict_time = time.time()  # still in view: keep it on screen

            self.readings = r
            self.steps = out["steps"]
            self.steps_version += 1
            dt = time.time() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(dt, 1e-3)) if last else 1.0 / max(dt, 1e-3)
            last = t0
            time.sleep(max(0.0, 0.25 - dt))

    def _load_reader(self):
        self.easy.load()
        self.reader_status = "ready" if self.easy.error is None else "unavailable"

    def analyse_upload(self, frame: np.ndarray) -> dict:
        self.hold_until = time.time() + UPLOAD_HOLD
        out = self.analyse(frame, live=False)
        r = out["readings"]
        self.readings, self.steps = r, out["steps"]
        self.steps_version += 1
        self.votes.clear()
        self.box, self.box_time = None, 0   # the live-feed box doesn't apply to an upload
        shown = frame.copy()
        if out["box"] is not None:
            colour = (60, 200, 60) if r["valid"] and database.find(r["corrected"]) else (60, 60, 230)
            cv2.polylines(shown, [out["box"]], True, colour, max(3, frame.shape[1] // 300))
        self.upload_frame = shown
        if r["valid"]:
            self._decide(r["corrected"], out["plate_type_label"], "upload")
        else:
            self.verdict = None
        return out

    def allow_visitor(self):
        if not self.verdict or self.verdict["status"] != "denied":
            return False
        v = dict(self.verdict)
        v.update(status="visitor", time=datetime.now().strftime("%I:%M:%S %p"))
        self.verdict, self.verdict_time = v, time.time()
        database.log_entry(v["plate"], None, None, True, v["plate_type"], "visitor (guard override)")
        return True

    # ------------------------------------------------------------- outputs
    def current_verdict(self):
        if self.verdict is None:
            return None
        hold = UPLOAD_HOLD if self.verdict.get("source") == "upload" else VERDICT_HOLD
        if self.verdict["status"] == "visitor":
            hold = VERDICT_HOLD
        if time.time() - self.verdict_time > hold:
            return None
        return self.verdict

    def mjpeg(self):
        while not self._stop:
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            if time.time() < self.hold_until and self.upload_frame is not None:
                frame = self.upload_frame.copy()
                h, w = frame.shape[:2]
                if w > 1280:
                    frame = cv2.resize(frame, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                time.sleep(0.1)
                continue
            if frame is None:
                frame = np.full((480, 854, 3), 35, np.uint8)
                cv2.putText(frame, "No camera feed", (280, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                cv2.putText(frame, "Upload a photo instead", (275, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (150, 150, 150), 1)
            else:
                v = self.current_verdict()
                if self.box is not None and time.time() - self.box_time < 1.0:
                    colour = (0, 190, 255)
                    if v and v["status"] == "granted":
                        colour = (60, 200, 60)
                    elif v and v["status"] == "denied":
                        colour = (60, 60, 230)
                    elif v and v["status"] == "visitor":
                        colour = (230, 160, 40)
                    cv2.polylines(frame, [self.box], True, colour, 4)
                    if v:
                        x, y = self.box[:, 0].min(), self.box[:, 1].min()
                        label = v["formatted"]
                        cv2.rectangle(frame, (x, max(0, y - 40)), (x + 22 * len(label) + 20, y), colour, -1)
                        cv2.putText(frame, label, (x + 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.05)


gate: Gate | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", site_name=SITE_NAME, tagline=SITE_TAGLINE)


@app.route("/video_feed")
def video_feed():
    return Response(gate.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def state():
    since = request.args.get("since", type=int, default=-1)
    data = {
        "camera": {"ok": gate.camera_ok, "message": gate.camera_msg, "index": gate.camera_index},
        "reader": gate.reader_status,
        "verdict": gate.current_verdict(),
        "readings": gate.readings,
        "motion": round(gate.motion, 3),
        "fps": round(gate.fps, 1),
        "paused": time.time() < gate.hold_until,
        "steps_version": gate.steps_version,
    }
    if since != gate.steps_version:
        data["steps"] = gate.steps
    return jsonify(data)


@app.route("/api/camera", methods=["POST"])
def camera():
    idx = int((request.get_json(silent=True) or {}).get("index", 0))
    gate.switch_camera(idx)
    return jsonify(ok=True)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify(ok=False, error="No image uploaded."), 400
    data = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, error="That file is not an image OpenCV can read (use JPG or PNG)."), 400
    out = gate.analyse_upload(frame)
    return jsonify(ok=True, found=out["box"] is not None, readings=out["readings"])


@app.route("/api/resume", methods=["POST"])
def resume():
    gate.hold_until = 0
    gate.verdict = None
    return jsonify(ok=True)


@app.route("/api/visitor", methods=["POST"])
def visitor():
    return jsonify(ok=gate.allow_visitor())


@app.route("/api/vehicles", methods=["GET", "POST"])
def vehicles():
    if request.method == "POST":
        d = request.get_json(silent=True) or {}
        return jsonify(database.add_vehicle(d.get("plate", ""), d.get("owner", ""), d.get("flat", "")))
    return jsonify(database.list_vehicles())


@app.route("/api/vehicles/<plate>", methods=["DELETE"])
def delete_vehicle(plate):
    return jsonify(ok=database.remove_vehicle(plate))


@app.route("/api/log")
def log():
    return jsonify(database.list_entries())


@app.route("/api/log", methods=["DELETE"])
def clear_log():
    database.clear_entries()
    return jsonify(ok=True)


@app.route("/api/log.csv")
def log_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "plate", "owner", "flat", "result", "plate type", "source"])
    for e in database.list_entries(100000):
        w.writerow([e["time"], e["formatted"], e["owner"] or "", e["flat_no"] or "",
                    "allowed" if e["allowed"] else "denied", e["plate_type"] or "", e["source"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=entry_log.csv"})


def main():
    global gate
    ap = argparse.ArgumentParser(description="SmartGate - society vehicle access")
    ap.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", 0)))
    ap.add_argument("--port", type=int, default=8000)  # 5000 is taken by AirPlay on Macs
    ap.add_argument("--source", help="use a photo or video file instead of the webcam (for testing)")
    args = ap.parse_args()

    database.init()
    gate = Gate(args.camera, args.source)
    threading.Thread(target=gate.camera_loop, daemon=True).start()
    threading.Thread(target=gate.process_loop, daemon=True).start()
    print(f"\n  {SITE_NAME} is running.  Open  http://localhost:{args.port}  in your browser.\n"
          "  (Press Ctrl+C here to stop.)\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
