"""
The image-processing pipeline: from a camera frame to a cropped, segmented
number plate.

Every stage is a classical technique from the IVP syllabus, and every stage
records a picture of its output (a "step") so the website can show the work
being done, live, in the Behind-the-Scenes panel.

    Unit 1  image as a matrix, colour -> grayscale
    Unit 2  histogram equalisation, Gaussian smoothing
    Unit 3  Fourier spectrum of the plate
    Unit 4  black-hat, closing, opening (morphology)
    Unit 5  Sobel gradient, Canny edge detector
    Unit 6  Otsu thresholding, connected-component labelling (8-connectivity)
    Unit 7  HSV colour model, colour slicing to classify the plate type
    Unit 8  frame differencing (temporal correlation) -- done in camera.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_WIDTH = 800  # every frame is scaled to this width so kernel sizes mean the same thing


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Step:
    key: str
    title: str
    unit: str
    explanation: str
    image: np.ndarray  # BGR or gray


@dataclass
class Candidate:
    box: np.ndarray            # 4 corner points in WORK-frame coordinates
    crop: np.ndarray           # perspective-corrected plate, BGR
    source: str                # "morphology" or "canny-contour"
    n_chars: int = 0
    char_boxes: list = field(default_factory=list)
    binary: np.ndarray | None = None
    two_line: bool = False
    score: float = 0.0


@dataclass
class PipelineResult:
    steps: list[Step]
    best: Candidate | None
    candidates: list[Candidate]
    scale: float               # original_width / WORK_WIDTH
    plate_type: str = "unknown"
    plate_type_label: str = ""
    char_images: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _warp(img: np.ndarray, corners: np.ndarray, out_h: int = 140) -> np.ndarray:
    """Perspective-correct the quadrilateral to a flat, front-on rectangle."""
    tl, tr, br, bl = corners
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))   # Euclidean distance (Unit 1)
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if h < 1 or w < 1:
        return np.zeros((out_h, out_h * 4, 3), np.uint8)
    out_w = int(round(out_h * w / h))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def _norm8(a: np.ndarray) -> np.ndarray:
    return cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plate-level analysis: binarise, label components, find characters
# ---------------------------------------------------------------------------

def segment_characters(plate_bgr: np.ndarray):
    """
    Otsu-binarise the plate and keep the connected components that are
    shaped like characters.  Returns (binary, char_boxes, two_line).
    Characters are WHITE (255) in the returned binary image.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    H, W = gray.shape

    # Otsu picks the threshold that best separates two intensity classes (Unit 6)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Characters are the minority class.  On a white/yellow/green plate they
    # are dark; on a black plate they are light.  Decide from the border,
    # which is always plate background.
    border = np.concatenate([bw[:4].ravel(), bw[-4:].ravel(), bw[:, :4].ravel(), bw[:, -4:].ravel()])
    if border.mean() > 127:          # background is white in bw -> chars are black -> invert
        bw = cv2.bitwise_not(bw)

    # A character can touch the plate's border line and merge with it.  A
    # small opening (erosion then dilation, Unit 4) snaps thin lines off;
    # we label both versions and keep whichever finds more characters.
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    a = _components(bw, H, W)
    b = _components(opened, H, W)
    if len(b[0]) > len(a[0]):
        return opened, b[0], b[1]
    return bw, a[0], a[1]


def _components(bw: np.ndarray, H: int, W: int):
    # Connected-component labelling, 8-connectivity (Unit 6)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 0.22 * H or h > 0.95 * H:
            continue
        if w > 1.3 * h or w < 2:
            continue
        if area < 0.12 * w * h:      # too sparse to be a glyph
            continue
        if x <= 1 or x + w >= W - 1:  # touching the side: plate border or strip
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return [], False

    # Keep components of roughly the most common height (rejects bolts, logos)
    hs = np.array([b[3] for b in boxes], float)
    med = np.median(hs)
    boxes = [b for b in boxes if 0.6 * med <= b[3] <= 1.45 * med]

    # One line or two?  Two-line (bike) plates have characters about half the
    # plate height, with centres falling in two separate bands.
    two_line = False
    if len(boxes) >= 4:
        # split at the largest vertical gap between character centres; it is
        # two lines only if the gap is bigger than half a character height
        order = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
        cys = [b[1] + b[3] / 2 for b in order]
        gaps = np.diff(cys)
        k = int(np.argmax(gaps))
        if gaps[k] > 0.6 * med and k + 1 >= 2 and len(order) - k - 1 >= 2:
            two_line = True
            top, bot = order[:k + 1], order[k + 1:]
            boxes = sorted(top, key=lambda b: b[0]) + sorted(bot, key=lambda b: b[0])
    if not two_line:
        # single line: keep the dominant row (rejects small text above/below)
        cys = np.array([b[1] + b[3] / 2 for b in boxes])
        mc = np.median(cys)
        boxes = [b for b, c in zip(boxes, cys) if abs(c - mc) < 0.35 * med]
        boxes = sorted(boxes, key=lambda b: b[0])

    # The blue "IND" strip on HSRP plates sits at the far left with small
    # letters; the height filter above removes it.
    return boxes, two_line


def classify_plate_colour(plate_bgr: np.ndarray, binary: np.ndarray | None, char_boxes=None):
    """
    HSV colour model + colour slicing (Unit 7).  Indian plates are coloured by
    use: white = private, yellow = commercial, green = electric, black =
    self-drive rental.  We look only at background pixels (not characters).
    Returns (type_key, label, slice_visualisation).
    """
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    if binary is not None and binary.shape[:2] == plate_bgr.shape[:2]:
        bg = cv2.erode(cv2.bitwise_not(binary), np.ones((3, 3), np.uint8))
    else:
        bg = np.full(plate_bgr.shape[:2], 255, np.uint8)
    if char_boxes:
        # only look right around the characters, so paper or car body
        # around the plate is not counted as plate colour
        x0 = min(b[0] for b in char_boxes); y0 = min(b[1] for b in char_boxes)
        x1 = max(b[0] + b[2] for b in char_boxes); y1 = max(b[1] + b[3] for b in char_boxes)
        pad = int(0.15 * (y1 - y0))
        region = np.zeros_like(bg)
        region[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 255
        bg = cv2.bitwise_and(bg, region)
    pix = hsv[bg > 0]
    if len(pix) < 50:
        pix = hsv.reshape(-1, 3)
    h, s, v = np.median(pix[:, 0]), np.median(pix[:, 1]), np.median(pix[:, 2])

    # Colour slicing: mask of "yellow" and "green" hue ranges, for the panel
    yellow = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
    green = cv2.inRange(hsv, (40, 60, 50), (90, 255, 255))
    vis = np.zeros_like(plate_bgr)
    vis[yellow > 0] = (0, 215, 255)
    vis[green > 0] = (60, 180, 60)
    vis[(yellow == 0) & (green == 0)] = (90, 90, 90)

    if v < 80:
        return "black", "Black plate - self-drive rental", vis
    if 15 <= h <= 40 and s > 70:
        return "yellow", "Yellow plate - commercial vehicle", vis
    if 40 < h <= 90 and s > 60:
        return "green", "Green plate - electric vehicle", vis
    if s < 60:
        return "white", "White plate - private vehicle", vis
    return "unknown", "Plate colour not recognised", vis


def fourier_spectrum(plate_bgr: np.ndarray) -> np.ndarray:
    """Log-magnitude Fourier spectrum, centred (Unit 3)."""
    g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(F))
    return cv2.applyColorMap(_norm8(mag), cv2.COLORMAP_INFERNO)


def label_visual(binary: np.ndarray, boxes) -> np.ndarray:
    """Colour each accepted character component differently."""
    vis = cv2.cvtColor(binary // 4, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(7)
    for (x, y, w, h) in boxes:
        col = tuple(int(c) for c in rng.integers(80, 255, 3))
        sub = binary[y:y + h, x:x + w] > 0
        vis[y:y + h, x:x + w][sub] = col
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), col, 1)
    return vis


# ---------------------------------------------------------------------------
# Frame-level detection
# ---------------------------------------------------------------------------

def _score(c: Candidate) -> float:
    """Plates have roughly 6-11 characters.  Prefer that, then bigger plates."""
    n = c.n_chars
    if n < 4:
        return 0.0
    closeness = 1.0 - min(abs(n - 9.5), 6) / 6.0
    area = cv2.contourArea(c.box.astype(np.float32))
    return 1.0 + closeness * 2.0 + min(area / 60000.0, 1.0)


def _aspect_ok(w: float, h: float) -> bool:
    if h <= 0:
        return False
    a = w / h
    return 1.4 <= a <= 7.5


def run(frame_bgr: np.ndarray, with_steps: bool = True) -> PipelineResult:
    steps: list[Step] = []
    scale = frame_bgr.shape[1] / WORK_WIDTH
    work = cv2.resize(frame_bgr, (WORK_WIDTH, int(round(frame_bgr.shape[0] / scale))),
                      interpolation=cv2.INTER_AREA)
    H, W = work.shape[:2]
    frame_area = H * W

    def add(key, title, unit, text, img):
        if with_steps:
            steps.append(Step(key, title, unit, text, img))

    add("input", "1. Input frame", "Unit 1",
        f"The camera frame is a {H}x{W}x3 matrix: every pixel holds three numbers (B, G, R) from 0 to 255.",
        work)

    # --- Unit 1: grayscale -------------------------------------------------
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    add("gray", "2. Grayscale", "Unit 1",
        "Colour is not needed to find the plate, so each pixel becomes one intensity: Y = 0.299R + 0.587G + 0.114B.",
        gray)

    # --- Unit 2: enhancement -----------------------------------------------
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(eq, (5, 5), 0)
    add("enhance", "3. Histogram equalisation + Gaussian smoothing", "Unit 2",
        "Histogram equalisation (applied tile by tile) spreads out the intensities so dim plates gain contrast; "
        "a 5x5 Gaussian filter then removes sensor noise.",
        smooth)

    # --- Unit 4: black-hat -------------------------------------------------
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, rect)
    add("blackhat", "4. Black-hat morphology", "Unit 4",
        "Black-hat = closing(image) - image. It keeps only small DARK details on a LIGHT background "
        "- exactly what plate characters are.",
        _norm8(blackhat))

    # --- Unit 5: Sobel -----------------------------------------------------
    gx = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    gx = _norm8(np.abs(gx))
    add("sobel", "5. Sobel gradient (x-direction)", "Unit 5",
        "First-order derivative in x. A row of characters produces many strong VERTICAL edges close together, "
        "which few other objects do.",
        gx)

    # --- Unit 4 + 6: close, Otsu, open ------------------------------------
    gxb = cv2.GaussianBlur(gx, (5, 5), 0)
    closed = cv2.morphologyEx(gxb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
    _, th = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)
    add("blob", "6. Closing + Otsu threshold", "Unit 4 + Unit 6",
        "A wide closing merges neighbouring character edges into one solid blob; Otsu's method picks the "
        "threshold automatically; an opening removes small specks.",
        th)

    # --- Unit 5: Canny (second route to the plate: its border) -----------
    canny = cv2.Canny(smooth, 60, 180)
    add("canny", "7. Canny edge detector", "Unit 5",
        "Canny = Gaussian smoothing + gradient + non-maximum suppression + hysteresis thresholding. "
        "Used to find the plate's rectangular outline.",
        canny)

    # --- candidates --------------------------------------------------------
    cands: list[Candidate] = []

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.004 * frame_area or w * h > 0.6 * frame_area:
            continue
        if not _aspect_ok(w, h):
            continue
        # the blob covers the characters; pad out to the plate edge
        px, py = int(0.06 * w), int(0.25 * h)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(W, x + w + px), min(H, y + h + py)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        cands.append(Candidate(corners, _warp(work, corners), "morphology"))

    edge = cv2.dilate(canny, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.004 * frame_area or area > 0.7 * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        corners = _order_corners(approx)
        tl, tr, br, bl = corners
        w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        if not _aspect_ok(w, h):
            continue
        # shrink slightly so the plate's own border is not inside the crop
        ctr = corners.mean(axis=0)
        inner = ctr + (corners - ctr) * 0.97
        cands.append(Candidate(inner.astype(np.float32), _warp(work, inner), "canny-contour"))

    for c in cands:
        c.binary, c.char_boxes, c.two_line = segment_characters(c.crop)
        c.n_chars = len(c.char_boxes)
        c.score = _score(c)

    cands.sort(key=lambda c: c.score, reverse=True)
    best = cands[0] if cands and cands[0].score > 0 else None

    vis = work.copy()
    for c in cands[:8]:
        cv2.polylines(vis, [c.box.astype(np.int32)], True, (0, 200, 255), 1)
    if best is not None:
        cv2.polylines(vis, [best.box.astype(np.int32)], True, (0, 255, 0), 3)
    add("candidates", "8. Plate candidates", "Unit 6",
        f"{len(cands)} rectangular regions with a plate-like aspect ratio were found (thin orange). "
        "Each is scored by how many character-shaped components it contains; the winner is in green.",
        vis)

    result = PipelineResult(steps, best, cands, scale)
    if best is None:
        return result

    # --- plate-level steps --------------------------------------------------
    ptype, plabel, slice_vis = classify_plate_colour(best.crop, best.binary, best.char_boxes)
    result.plate_type, result.plate_type_label = ptype, plabel
    add("crop", "9. Plate, perspective-corrected", "Unit 1",
        f"The plate's four corners are mapped to a flat rectangle (found by: {best.source}).",
        best.crop)
    add("colour", "10. HSV colour slicing", "Unit 7",
        f"In HSV, hue separates colour from brightness. Slicing the yellow and green hue ranges classifies the plate: "
        f"{plabel}.",
        slice_vis)
    add("otsu", "11. Otsu binarisation of the plate", "Unit 6",
        "A second Otsu threshold, on the plate alone, separates characters (white) from background (black).",
        best.binary)
    add("components", "12. Connected-component labelling", "Unit 6",
        f"Pixels are grouped with 8-connectivity; components with character-like height and width are kept "
        f"({best.n_chars} characters{', two lines' if best.two_line else ''}).",
        label_visual(best.binary, best.char_boxes))
    add("fourier", "13. Fourier spectrum of the plate", "Unit 3",
        "The 2-D DFT, log magnitude, centred. Sharp character strokes put energy far from the centre "
        "(high frequencies); a blurry plate would only light up the middle.",
        fourier_spectrum(best.crop))

    for (x, y, w, h) in best.char_boxes:
        result.char_images.append(best.binary[y:y + h, x:x + w])
    return result


def box_in_frame(result: PipelineResult) -> np.ndarray | None:
    """Best plate's corners in ORIGINAL frame coordinates."""
    if result.best is None:
        return None
    return (result.best.box * result.scale).astype(np.int32)
