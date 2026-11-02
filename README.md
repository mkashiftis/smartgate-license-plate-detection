# SmartGate — License Plate Detection for Society Gate Entry

An IVP (Image & Video Processing) course project. A webcam watches the society
gate. When a vehicle's number plate is shown to it, the system **finds the
plate, reads the number, and checks it against the society's resident
database**:

- 🟢 **Registered vehicle** → green screen: *ACCESS GRANTED — Rahul Sharma · Flat B-402*
- 🔴 **Outsider** → red screen: *ACCESS DENIED*, with an **Allow as visitor** button for the guard
- every vehicle seen is written to an **entry log**

The plate is found with **classical image processing from the syllabus**
(units 1–8), and a **Behind the Scenes** panel on the website shows every step
live, labelled with its syllabus unit.

---

## 1. Running it

### Mac
1. Install Python 3.10–3.12 from https://www.python.org/downloads/ (if not already installed).
2. Double-click **`run_mac.command`**.
   If macOS blocks it: right-click → Open, or in Terminal: `bash run_mac.command`
3. The first run installs everything (5–10 min, needs internet). After that it starts in seconds and works offline.
4. The website opens at **http://localhost:8000**
5. **Camera permission:** the first time, macOS asks whether Terminal may use the camera — click *Allow*.
   If you missed it: System Settings → Privacy & Security → Camera → turn on Terminal.

### Windows
1. Install Python 3.10–3.12 from https://www.python.org/downloads/ — tick **"Add python.exe to PATH"**.
2. Double-click **`run_windows.bat`**. First run installs everything (5–10 min).
3. The website opens at **http://localhost:8000**

### Manual (any OS)
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip ...
.venv/bin/python app.py                          # then open http://localhost:8000
```
Options: `--camera 1` (use a different camera), `--port 8080`,
`--source sample_images/scene_MH02AB1234.jpg` (use a photo/video file instead of a webcam).

To stop: press **Ctrl + C** in the terminal window.

---

## 2. The website

| Page | What it does |
|---|---|
| **Gate** | Live camera with the plate boxed. Big result card (green / red / visitor), what each reader read, and the **Behind the Scenes** panel. Also: camera selector, **Upload a photo** (fallback if the camera misbehaves), sound on/off. |
| **Residents** | Register a vehicle: plate + owner + flat. Search and remove. A vehicle belongs to exactly one flat; a flat can have several vehicles. |
| **Entry Log** | Every vehicle seen: time, plate, granted / denied / visitor, owner, flat, plate type. Download as CSV. |

The database starts **empty**. Add your residents on the Residents page.

---

## 3. Demo plan (about 5 minutes)

1. Print the plates: **`plates/printable_plates.pdf`** (or make your own, see §6). Use matte paper.
2. Open the site. Show the **Residents** page is empty.
3. Hold up `MH 02 AB 1234` → **RED, Access denied** (not registered). Click **Allow as visitor**.
4. Go to **Residents**, add `MH 02 AB 1234 · Rahul Sharma · B-402`.
5. Hold it up again → **GREEN, Access granted — Rahul Sharma · Flat B-402**.
6. Hold up the **yellow** plate → the card says *commercial vehicle* (HSV colour classification, Unit 7).
7. Hold up the **two-line bike** plate → it still reads correctly.
8. Scroll to **Behind the Scenes** and walk through steps 1–15, pointing out each unit
   (tick **Freeze** to stop the images changing while you explain).
9. Open **Entry Log** — every attempt is there.

**Tips for a reliable live demo**
- Hold the plate **still**, roughly 30–60 cm from the camera, filling about half the width.
- Avoid glare: tilt the paper slightly away from ceiling lights. Don't show plates on a phone screen.
- Test in the actual room beforehand. If the camera fails on the day, use **Upload a photo**
  with the pictures in `sample_images/`.

---

## 4. How it works — and where the syllabus is used

```
camera frame ─► grayscale ─► equalise + smooth ─► black-hat ─► Sobel-x ─► closing + Otsu ─┐
                         └─► Canny edges ─► 4-corner contours ─────────────────────────────┤
                                                                                          ▼
        plate candidates  ─►  perspective-correct  ─►  Otsu binarise  ─►  connected components
                                                                                 │
                    HSV colour ─► plate type                     characters ─────┤
                                                                                 ▼
                                    EasyOCR  +  template matching  ─►  Indian-plate format correction
                                                                                 ▼
                         same reading in 3 frames (5 for unknown plates) ─►  database lookup ─► 🟢 / 🔴
```

| # | Step (as shown in Behind the Scenes) | Syllabus |
|---|---|---|
| 1 | Frame is a H×W×3 matrix | Unit 1 — image as a matrix |
| 2 | Grayscale: Y = 0.299R + 0.587G + 0.114B | Unit 1 — colour & gray levels |
| 3 | Histogram equalisation (tile-wise) + 5×5 Gaussian filter | Unit 2 — histogram processing, smoothing filters |
| 4 | Black-hat = closing − image: dark text on light plate | Unit 4 — morphology |
| 5 | Sobel x-gradient: characters = dense vertical edges | Unit 5 — first-order derivative |
| 6 | Wide closing merges characters; Otsu threshold; opening removes specks | Units 4 & 6 |
| 7 | Canny edge detector finds the plate outline | Unit 5 — Canny |
| 8 | Candidate regions scored by aspect ratio and character count | Unit 6 — region analysis |
| 9 | Perspective correction; Euclidean corner distances | Unit 1 — distance measures |
| 10 | HSV hue slicing → white / yellow / green / black plate | Unit 7 — colour models, colour slicing |
| 11 | Otsu binarisation of the plate | Unit 6 — Otsu |
| 12 | Connected-component labelling, 8-connectivity | Unit 6 — component labelling |
| 13 | 2-D DFT log-magnitude spectrum of the plate | Unit 3 — Fourier transform |
| 14 | Template matching by Euclidean distance (A–Z, 0–9) | Unit 1 — distance measures |
| 15 | Frame differencing \|f(t) − f(t−1)\| — waits until the plate is still | Unit 8 — temporal correlation |

**Why two readers?** *Template matching* uses only syllabus methods, so it's shown
to demonstrate the classical approach. *EasyOCR* (a ready-made text reader) is
more reliable on real plates, so the gate decision prefers it. Both are shown
side by side on the Gate page.

**Plate-format correction.** Indian plates follow `SS 00 XX 0000` (state,
RTO number, series, number) or the Bharat series `00 BH 0000 XX`. Knowing
which positions must be letters and which digits fixes the classic OCR
confusions: O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6.

**Why wait for several reads?** One frame can be misread. A resident is let in
after the same plate is read in 3 frames; an unknown plate needs 5 before it's
refused, so a single misread never locks a resident out.

---

## 5. Project structure

```
app.py                 web server, camera thread, decision logic
lpr/pipeline.py        all the image processing (steps 1–13)
lpr/ocr.py             EasyOCR, template matching, Indian plate-format correction
lpr/database.py        SQLite: flats, vehicles, entry log
templates/index.html   the website (one page, three tabs)
static/style.css       styling
static/app.js          live updates, residents form, entry log
tools/make_plates.py   make printable demo plates
tools/make_test_scenes.py  make synthetic test photos
plates/                printable plates (PNG + PDF)
sample_images/         test photos for "Upload a photo"
data/society.db        the database (created on first run)
assets/fonts/          DejaVu fonts (free licence) for templates and plates
```

---

## 6. Making more demo plates

```bash
.venv/bin/python tools/make_plates.py "MH 02 AB 1234" "MH 47 K 5521:yellow" "MH 01 EV 2024:green" "MH 02 DZ 3110:white:bike"
```
Format: `PLATE[:colour[:bike]]`, colour = white | yellow | green | black.
Output: `plates/*.png` and `plates/printable_plates.pdf`.

Photos of real plates work too: print them reasonably large (at least the plate
filling half an A4 page).

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| "Camera 0 could not be opened" | Pick another number in the **Camera** dropdown (Macs sometimes list an iPhone as camera 0). On Mac, check camera permission for Terminal. Close other apps using the camera (Zoom, Teams, Photo Booth). |
| "OCR loading…" for a long time | First start loads the model (5–20 s). If it says *OCR unavailable*, re-run the installer with internet on. Template matching still works meanwhile. |
| Plate found but never confirmed | Hold it still, closer, and avoid glare. Watch **Behind the Scenes** step 8 to see whether the plate is being picked. |
| Wrong reading | Check the Gate page: if EasyOCR is right but the result isn't, the plate may not follow the Indian format; register it exactly as read. |
| Port 8000 busy | `python app.py --port 8080` |

---

## 8. Limitations (good to mention in the viva)

- Works best with the plate facing the camera; strong tilt, motion blur and glare reduce accuracy.
- Template matching is font-sensitive; real plates use several fonts, which is why EasyOCR makes the final call.
- Non-standard "fancy" plates (stylised fonts, extra text) may be misread — and are illegal anyway.
- A printed photo of a resident's plate would be accepted: real systems pair ANPR with RFID tags (FASTag) or a guard.
- Night-time / infrared operation is not covered.
