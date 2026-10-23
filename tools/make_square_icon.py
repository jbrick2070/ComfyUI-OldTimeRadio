"""Draw the square registry icon in the card art's own style.

The Comfy Registry specification asks for a square icon of at most 400x400;
the animated card (assets/otr_icon.gif) is 800x400, and a centre crop of it
cuts the title in half. This draws a 400x400 mark from the same parts -- the
title in the card's typeface (assets/fonts/InterDisplay-Black.otf), the
magenta rule, the signal wave and the grid floor, on the card's background --
so the icon and the card read as one thing.

    python tools/make_square_icon.py            # writes assets/otr_icon_400.png
    python tools/make_square_icon.py --preview  # also writes a 96/400 px sheet
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[1]
FONT = REPO / "assets" / "fonts" / "InterDisplay-Black.otf"
OUT = REPO / "assets" / "otr_icon_400.png"

BG_TOP, BG_BOTTOM = (18, 10, 46), (38, 25, 78)
MAGENTA, CYAN, GREEN, GRID, INK = (240, 43, 143), (62, 233, 255), (78, 240, 154), (90, 63, 160), (244, 240, 255)
SIZE = 400
SCALE = 4        # drawn at 1600 and reduced, so edges and glows are smooth


def _background(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), BG_TOP)
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        px_row = tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM))
        for x in range(size):
            px[x, y] = px_row
    return img


def _grid(draw: ImageDraw.ImageDraw, size: int, horizon: int) -> None:
    """The perspective floor from the card: lines fanning from a vanishing
    point above the horizon, and rows that bunch toward it."""
    vanish = (size // 2, horizon - size // 3)
    width = max(1, size // 400)
    for i in range(-10, 11):
        x_bottom = size // 2 + i * size // 8
        draw.line([vanish, (x_bottom, size)], fill=GRID, width=width)
    for k in range(1, 9):
        y = horizon + int((size - horizon) * (k / 8) ** 1.8)
        draw.line([(0, y), (size, y)], fill=GRID, width=width)
    draw.line([(0, horizon), (size, horizon)], fill=MAGENTA, width=width * 2)


def _wave(draw: ImageDraw.ImageDraw, size: int, y_mid: int) -> None:
    """The signal: one bright pulse on a quiet carrier."""
    width = max(2, size // 160)
    points = []
    for x in range(size // 8, size - size // 8):
        t = (x - size // 2) / (size / 10)
        y = y_mid - int(size * 0.075 * math.exp(-t * t) * math.cos(3.5 * t)) - int(size * 0.006 * math.sin(x / 9))
        points.append((x, y))
    draw.line(points, fill=GREEN, width=width, joint="curve")


def _title(img: Image.Image, size: int) -> None:
    font = ImageFont.truetype(str(FONT), int(size * 0.21))
    draw = ImageDraw.Draw(img)
    for line, y in (("OLD TIME", int(size * 0.095)), ("RADIO", int(size * 0.315))):
        tracked = " ".join(line) if line == "RADIO" else line
        w = draw.textlength(tracked, font=font)
        x = (size - w) / 2
        # The card's pink shadow, then the ink.
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((x + size * 0.012, y + size * 0.012), tracked, font=font, fill=MAGENTA + (255,))
        img.paste(shadow, (0, 0), shadow)
        draw.text((x, y), tracked, font=font, fill=INK)


def render(size: int = SIZE) -> Image.Image:
    big = size * SCALE
    img = _background(big)
    draw = ImageDraw.Draw(img)
    horizon = int(big * 0.72)
    _grid(draw, big, horizon)
    # A glow behind the wave, so it reads at icon size.
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _wave(ImageDraw.Draw(glow), big, int(big * 0.60))
    glow = glow.filter(ImageFilter.GaussianBlur(big // 60))
    img.paste(glow, (0, 0), glow)
    _wave(draw, big, int(big * 0.60))
    _title(img, big)
    # The rule under the title, as on the card.
    draw.rectangle([big * 0.18, big * 0.535, big * 0.82, big * 0.545], fill=MAGENTA)
    return img.resize((size, size), Image.LANCZOS)


def preview(icon: Image.Image, art: Image.Image, path: Path) -> None:
    """Both candidates at the registry's row size and at full size."""
    sheet = Image.new("RGB", (1000, 560), (30, 32, 40))
    for i, im in enumerate((icon, art)):
        sheet.paste(im.resize((400, 400), Image.LANCZOS), (40 + i * 480, 40))
        sheet.paste(im.resize((96, 96), Image.LANCZOS), (40 + i * 480, 456))
        sheet.paste(im.resize((48, 48), Image.LANCZOS), (160 + i * 480, 480))
    sheet.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", help="write a comparison sheet to this path")
    args = ap.parse_args(argv)
    icon = render()
    icon.save(OUT, optimize=True)
    print("wrote", OUT, icon.size)
    if args.preview:
        art = Image.open(REPO / "assets" / "otr_gallery_thumb_master.webp").convert("RGB")
        preview(icon, art, Path(args.preview))
        print("wrote", args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
