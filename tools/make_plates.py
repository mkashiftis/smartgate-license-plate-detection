"""
Make printable demo number plates (Indian HSRP style) for holding up to the camera.

    python tools/make_plates.py                         # the default demo set
    python tools/make_plates.py "MH 02 AB 1234" "MH 47 K 5521:yellow" "MH 01 EV 2024:green"
    python tools/make_plates.py "MH 02 CD 7788:white:bike"

Each argument is  PLATE[:colour[:bike]]   colour = white | yellow | green | black

Writes PNGs to plates/ and one printable PDF: plates/printable_plates.pdf
(A4 landscape, two plates per page). Print on MATTE paper, not glossy.
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "assets", "fonts", "DejaVuSansCondensed-Bold.ttf")
OUT = os.path.join(ROOT, "plates")

COLOURS = {   # background, text
    "white": ((255, 255, 255), (15, 15, 15)),
    "yellow": ((250, 204, 20), (15, 15, 15)),
    "green": ((20, 140, 70), (255, 255, 255)),
    "black": ((20, 20, 20), (250, 204, 20)),
}

DEFAULT = [
    "MH 02 AB 1234",
    "MH 01 CX 4821",
    "MH 04 FT 0786",
    "MH 43 BK 9090",
    "MH 47 K 5521:yellow",
    "MH 01 EV 2024:green",
    "MH 02 DZ 3110:white:bike",
    "22 BH 4567 AA",
]


def _fit_font(draw, text, max_w, max_h):
    size = max_h
    while size > 10:
        f = ImageFont.truetype(FONT, size)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if r - l <= max_w and b - t <= max_h:
            return f
        size -= 2
    return ImageFont.truetype(FONT, 10)


def make_plate(text: str, colour: str = "white", bike: bool = False) -> Image.Image:
    bg, fg = COLOURS[colour]
    if bike:
        W, H = 800, 400            # 200 x 100 mm at 4 px/mm
    else:
        W, H = 2000, 480           # 500 x 120 mm at 4 px/mm
    img = Image.new("RGB", (W, H), (230, 230, 230))
    d = ImageDraw.Draw(img)
    r = int(H * 0.08)
    d.rounded_rectangle([4, 4, W - 5, H - 5], radius=r, fill=bg, outline=fg, width=max(6, H // 40))

    strip_w = int(W * (0.075 if not bike else 0.12))
    m = H // 20
    d.rectangle([m + 6, m + 6, m + 6 + strip_w, H - m - 6], fill=(20, 60, 160))
    sf = ImageFont.truetype(FONT, max(12, strip_w // 3))
    l, t, rr, b = d.textbbox((0, 0), "IND", font=sf)
    d.text((m + 6 + (strip_w - (rr - l)) // 2 - l, H - m - 12 - (b - t) - t - H // 20), "IND",
           font=sf, fill=(255, 255, 255))

    x0 = m + strip_w + 24
    area_w = W - x0 - 70
    if bike:
        parts = text.split()
        line1 = " ".join(parts[:2])
        line2 = " ".join(parts[2:])
        for i, line in enumerate((line1, line2)):
            f = _fit_font(d, line, area_w, int(H * 0.36))
            l, t, rr, b = d.textbbox((0, 0), line, font=f)
            cy = H * (0.29 if i == 0 else 0.71)
            d.text((x0 + (area_w - (rr - l)) / 2 - l, cy - (b - t) / 2 - t), line, font=f, fill=fg)
    else:
        f = _fit_font(d, text, area_w, int(H * 0.56))
        l, t, rr, b = d.textbbox((0, 0), text, font=f)
        d.text((x0 + (area_w - (rr - l)) / 2 - l, H / 2 - (b - t) / 2 - t), text, font=f, fill=fg)
    return img


def parse(arg: str):
    bits = arg.split(":")
    text = bits[0].strip().upper()
    colour = bits[1].strip().lower() if len(bits) > 1 and bits[1].strip() else "white"
    bike = len(bits) > 2 and bits[2].strip().lower() == "bike"
    return text, colour, bike


def main(args):
    os.makedirs(OUT, exist_ok=True)
    specs = [parse(a) for a in (args or DEFAULT)]
    images = []
    for text, colour, bike in specs:
        im = make_plate(text, colour, bike)
        name = text.replace(" ", "") + ("_bike" if bike else "") + ".png"
        im.save(os.path.join(OUT, name))
        images.append(im)
        print("wrote", os.path.join("plates", name))

    # A4 landscape at 150 dpi: 1754 x 1240 px, two plates per page
    PW, PH = 1754, 1240
    pages = []
    for i in range(0, len(images), 2):
        page = Image.new("RGB", (PW, PH), "white")
        for j, im in enumerate(images[i:i + 2]):
            maxw, maxh = PW - 120, PH // 2 - 120
            s = min(maxw / im.width, maxh / im.height)
            r = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
            page.paste(r, ((PW - r.width) // 2, j * PH // 2 + (PH // 2 - r.height) // 2))
        pages.append(page)
    pdf = os.path.join(OUT, "printable_plates.pdf")
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
    print("wrote", os.path.join("plates", "printable_plates.pdf"), f"({len(pages)} pages)")


if __name__ == "__main__":
    main(sys.argv[1:])
