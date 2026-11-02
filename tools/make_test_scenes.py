"""
Make synthetic 'camera' photos of the demo plates for testing without a webcam:
each plate is printed on a sheet, tilted in perspective and placed in a
cluttered scene with blur and noise.  Output: sample_images/
"""
import glob, os, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

def scene(W=1280, H=720):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = rng.integers(60, 200, 3)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = rng.integers(0, W), rng.integers(0, H)
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + rng.integers(20, 300), y + rng.integers(20, 300)), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(10, 120)), c, -1)
    return cv2.GaussianBlur(img, (7, 7), 0)

os.makedirs(os.path.join(ROOT, "sample_images"), exist_ok=True)
for path in sorted(glob.glob(os.path.join(ROOT, "plates", "*.png"))):
    plate = cv2.imread(path)
    ph, pw = plate.shape[:2]
    sheet = np.full((ph + 2 * ph // 3, pw + pw // 6, 3), 245, np.uint8)
    sheet[ph // 3: ph // 3 + ph, pw // 12: pw // 12 + pw] = plate
    img = scene()
    H, W = img.shape[:2]
    sw = rng.uniform(0.45, 0.7) * W
    sh = sw * sheet.shape[0] / sheet.shape[1]
    cx, cy = W / 2 + rng.uniform(-100, 100), H / 2 + rng.uniform(-60, 60)
    j = lambda: rng.uniform(-0.06, 0.06) * sw
    dst = np.float32([[cx - sw/2 + j(), cy - sh/2 + j()], [cx + sw/2 + j(), cy - sh/2 + j()],
                      [cx + sw/2 + j(), cy + sh/2 + j()], [cx - sw/2 + j(), cy + sh/2 + j()]])
    src = np.float32([[0, 0], [sheet.shape[1], 0], [sheet.shape[1], sheet.shape[0]], [0, sheet.shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, M, (W, H))
    mask = cv2.warpPerspective(np.full(sheet.shape[:2], 255, np.uint8), M, (W, H))
    img[mask > 0] = warped[mask > 0]
    shade = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.8, 1.1), W)[None, :, None]
    img = np.clip(img * shade, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    name = "scene_" + os.path.basename(path).replace(".png", ".jpg")
    cv2.imwrite(os.path.join(ROOT, "sample_images", name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote sample_images/" + name)
