"""Make the square registry icon: the banner's cathedral tube radio.

The Comfy Registry specification asks for a square icon of at most 400x400.
The icon is the glowing cathedral radio from the banner art
(assets/otr_banner_master.webp), cropped square around its cabinet, so the icon
shows an old-time radio rather than only the words, and the icon and the banner
read as one thing (operator, 2026-11-20: "make this more old time radio").

The crop box is measured, not guessed: in the 1672x941 master the bronze
cabinet spans x 993..1457 and y 127..530, the title's pink "o" ends at x 992,
and the step cards start at y 540. A 480 px square from (985, 58) holds the
whole cabinet and its glow and stops above the cards; the sliver of the "o" that
falls inside the left edge is painted over with the background beside it.

    python tools/make_square_icon.py            # writes assets/otr_icon_400.png
    python tools/make_square_icon.py --preview sheet.png  # also a 400/96/48 px sheet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "assets" / "otr_banner_master.webp"
OUT = REPO / "assets" / "otr_icon_400.png"

SIZE = 400
#: (left, top, right, bottom) in the master: the cabinet, its glow, no cards.
CROP_BOX = (985, 58, 1465, 538)
#: Rows of the 400 px icon where the title's "o" pokes in at the left edge,
#: and the column whose background replaces it.
LETTER_ROWS = range(66, 146)
LETTER_COLUMNS = 13


def render(size: int = SIZE) -> Image.Image:
    art = Image.open(SOURCE).convert("RGB")
    icon = art.crop(CROP_BOX).resize((size, size), Image.LANCZOS)
    px = icon.load()
    for y in LETTER_ROWS:
        background = px[LETTER_COLUMNS, y]
        for x in range(LETTER_COLUMNS):
            px[x, y] = background
    return icon


def preview(icon: Image.Image, path: Path) -> None:
    """The icon at full size and at the sizes ComfyUI-Manager shows it."""
    sheet = Image.new("RGB", (620, 440), (30, 32, 40))
    sheet.paste(icon, (20, 20))
    sheet.paste(icon.resize((96, 96), Image.LANCZOS), (450, 40))
    sheet.paste(icon.resize((48, 48), Image.LANCZOS), (474, 170))
    sheet.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", help="write a size-comparison sheet to this path")
    args = ap.parse_args(argv)
    icon = render()
    icon.save(OUT, optimize=True)
    print("wrote", OUT, icon.size)
    if args.preview:
        preview(icon, Path(args.preview))
        print("wrote", args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
