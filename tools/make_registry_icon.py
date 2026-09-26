"""Animated registry icon for ComfyUI-OldTimeRadio.

A 1980s broadcast-schematic take on the episode pipeline: a simplified
process slide with beveled cells, scientific tick marks, and a carrier
pulse that runs the chain while the master cell scopes a live trace.
800x400 (the registry ceiling), looping GIF.

DO NOT REGENERATE OVER assets/otr_icon.gif EXPECTING THE SAME PICTURE.
The shipped card was hand-finished after this script was last committed and
carries furniture this file does not draw: the LOCAL / OFFLINE flag, the
REQUEST-HASH DETERMINISTIC footer, and T0-T3 tick labels rather than bare
digits. Running this writes a plainer slide and silently downgrades the
registry listing art, which is the one image a stranger sees first.

The card is served live from the `main` branch -- pyproject's `Icon` pins
`raw.githubusercontent.com/.../main/assets/otr_icon.gif` -- so editing that
file updates the public listing with NO version bump and no publish.

NO COUNTS ON THE CARD (operator, 2026-09-21): "better to leave things that
can change like node count out of the image, not just a new number, make it
future proof." The footer used to read "34 NODES / 5 STORY BANKS"; the pack
declares 25 nodes, so it was wrong as well as rot-prone. It was removed
rather than renumbered. Do not add a count back, here or to the art.
"""
import math
from PIL import Image, ImageDraw, ImageFont

W, H = 800, 400
FRAMES = 36
DUR_MS = 55

BOLD = r"C:\Windows\Fonts\arialbd.ttf"
REG = r"C:\Windows\Fonts\arial.ttf"

INK_BG_TOP = (18, 10, 46)
INK_BG_BOT = (38, 18, 74)
CYAN = (64, 224, 232)
MAGENTA = (255, 45, 149)
AMBER = (255, 178, 44)
MINT = (86, 255, 176)
PAPER = (232, 240, 255)
GRID = (98, 84, 168)

f_title = ImageFont.truetype(BOLD, 27)
f_fig = ImageFont.truetype(REG, 13)
f_cell = ImageFont.truetype(BOLD, 15)
f_sub = ImageFont.truetype(REG, 12)
f_tick = ImageFont.truetype(REG, 10)

CELLS = [
    ("SCRIPT", "story bank", AMBER),
    ("VOICES", "cast + host", CYAN),
    ("MASTER", "48 kHz frozen", MINT),
    ("EPISODE", "final mp4", MAGENTA),
]
BOX_W, BOX_H, BOX_Y = 148, 78, 196
XS = [38, 232, 426, 620]


def spaced(d, xy, text, font, fill, track=3):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track
    return x - track


def spaced_w(d, text, font, track=3):
    return sum(d.textlength(c, font=font) for c in text) + track * (len(text) - 1)


def backdrop():
    img = Image.new("RGB", (W, H), INK_BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(H):  # vertical gradient
        t = y / H
        d.line([(0, y), (W, y)], fill=(
            int(INK_BG_TOP[0] + (INK_BG_BOT[0] - INK_BG_TOP[0]) * t),
            int(INK_BG_TOP[1] + (INK_BG_BOT[1] - INK_BG_TOP[1]) * t),
            int(INK_BG_TOP[2] + (INK_BG_BOT[2] - INK_BG_TOP[2]) * t),
        ))
    # perspective floor grid -- the obligatory 80s horizon
    hz = 322
    d.line([(0, hz), (W, hz)], fill=GRID, width=1)
    for i in range(-9, 20):
        x_top = W / 2 + i * 34
        d.line([(x_top, hz), (W / 2 + i * 150, H)], fill=GRID, width=1)
    step = 6
    y = hz + 4
    while y < H:
        d.line([(0, y), (W, y)], fill=GRID, width=1)
        step = int(step * 1.42) + 1
        y += step
    return img, d


def bevel(d, x, y, w, h, face, lit, shade, width=3):
    d.rectangle([x, y, x + w, y + h], fill=face)
    d.line([(x, y), (x + w, y)], fill=lit, width=width)
    d.line([(x, y), (x, y + h)], fill=lit, width=width)
    d.line([(x, y + h), (x + w, y + h)], fill=shade, width=width)
    d.line([(x + w, y), (x + w, y + h)], fill=shade, width=width)


def mix(c, other, t):
    return tuple(int(c[i] + (other[i] - c[i]) * t) for i in range(3))


def draw_frame(k):
    phase = k / FRAMES
    img, d = backdrop()

    # ---- slide furniture -------------------------------------------------
    title = "OLD TIME RADIO"
    tw = spaced_w(d, title, f_title, 6)
    tx = (W - tw) / 2
    spaced(d, (tx + 2, 34), title, f_title, (86, 40, 120), 6)   # hard offset shadow
    spaced(d, (tx, 32), title, f_title, PAPER, 6)
    d.rectangle([tx - 6, 68, tx + tw + 6, 73], fill=MAGENTA)
    d.text((tx - 6, 80), "FIG. 1   EPISODE SIGNAL PATH", font=f_fig, fill=CYAN)

    # scientific tick rule under the chain
    base = 300
    d.line([(38, base), (768, base)], fill=GRID, width=2)
    for i in range(0, 74):
        x = 38 + i * 10
        tall = (i % 5 == 0)
        d.line([(x, base), (x, base - (9 if tall else 4))], fill=GRID, width=1)
    for i, lbl in enumerate(["0", "1", "2", "3"]):
        d.text((XS[i] + BOX_W / 2 - 3, base + 6), lbl, font=f_tick, fill=(150, 138, 200))

    # ---- connectors ------------------------------------------------------
    for i in range(3):
        x0 = XS[i] + BOX_W
        x1 = XS[i + 1]
        ym = BOX_Y + BOX_H / 2
        d.line([(x0 + 6, ym), (x1 - 16, ym)], fill=(126, 108, 196), width=5)
        d.polygon([(x1 - 18, ym - 11), (x1 - 2, ym), (x1 - 18, ym + 11)], fill=(150, 130, 220))

    # ---- cells -----------------------------------------------------------
    lead = phase * 4.0  # which cell the carrier is energising
    for i, (name, sub, hue) in enumerate(CELLS):
        x = XS[i]
        heat = max(0.0, 1.0 - abs(lead - (i + 0.35)) * 1.7)
        face = mix((36, 24, 74), (58, 40, 108), heat)
        bevel(d, x, BOX_Y, BOX_W, BOX_H, face, mix((120, 96, 190), hue, heat),
              (22, 12, 48))
        d.rectangle([x + 3, BOX_Y + 3, x + BOX_W - 3, BOX_Y + 21],
                    fill=mix(hue, PAPER, heat * 0.45))
        d.text((x + 10, BOX_Y + 4), name, font=f_cell, fill=(20, 12, 40))
        d.text((x + 10, BOX_Y + 54), sub, font=f_sub, fill=mix((168, 156, 214), PAPER, heat))

        if i == 2:  # oscilloscope trace inside MASTER
            cx0, cx1 = x + 10, x + BOX_W - 10
            mid = BOX_Y + 40
            pts = []
            for px in range(int(cx0), int(cx1)):
                u = (px - cx0) / (cx1 - cx0)
                amp = 13 * math.sin(u * 9.4 + phase * 6.28) * math.sin(u * math.pi)
                amp += 5 * math.sin(u * 23 - phase * 9.1)
                pts.append((px, mid + amp))
            d.line(pts, fill=MINT, width=2)
        else:
            for r in range(3):  # stacked "data" bars
                yb = BOX_Y + 30 + r * 8
                wfrac = [0.72, 0.5, 0.62][r]
                d.line([(x + 10, yb), (x + 10 + (BOX_W - 20) * wfrac, yb)],
                       fill=mix((96, 82, 158), hue, heat * 0.8), width=3)

    # ---- carrier pulse ---------------------------------------------------
    span_x0, span_x1 = XS[0] + 10, XS[3] + BOX_W - 10
    px = span_x0 + (span_x1 - span_x0) * phase
    py = BOX_Y + BOX_H / 2
    for rad, alpha in ((13, 60), (9, 120), (5, 255)):
        col = mix((58, 40, 108), PAPER, alpha / 255)
        d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col)
    d.line([(px - 26, py), (px - 12, py)], fill=MAGENTA, width=3)

    # corner registration marks -- the technical-diagram tell
    for cx, cy in ((16, 16), (W - 16, 16), (16, H - 16), (W - 16, H - 16)):
        d.line([(cx - 9, cy), (cx + 9, cy)], fill=GRID, width=1)
        d.line([(cx, cy - 9), (cx, cy + 9)], fill=GRID, width=1)

    return img


# --------------------------------------------------------------------------
# SQUARE GALLERY THUMBNAIL (2026-09-25). ComfyUI's template gallery shows a
# square card per workflow and, for a custom pack, requests
# /api/workflow_templates/<pack>/<stem>.jpg -- a static JPEG. Same schematic
# language as the registry card (operator: "functional, not literal"), laid
# out as a 2x2 grid of the four stages so it reads at gallery size. NO COUNTS
# on it, for the same reason as the card above.
# --------------------------------------------------------------------------
SQ = 400
SQ_CELL_W, SQ_CELL_H = 164, 104
SQ_XS = [26, 210]
SQ_YS = [112, 232]


def square_thumbnail():
    img = Image.new("RGB", (SQ, SQ), INK_BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(SQ):
        d.line([(0, y), (SQ, y)], fill=mix(INK_BG_TOP, INK_BG_BOT, y / SQ))
    hz = 354
    d.line([(0, hz), (SQ, hz)], fill=GRID, width=1)
    for i in range(-7, 16):
        d.line([(SQ / 2 + i * 26, hz), (SQ / 2 + i * 110, SQ)], fill=GRID, width=1)
    step, y = 4, hz + 3
    while y < SQ:
        d.line([(0, y), (SQ, y)], fill=GRID, width=1)
        step = int(step * 1.5) + 1
        y += step

    title = "OLD TIME RADIO"
    f_sq_title = ImageFont.truetype(BOLD, 30)
    tw = spaced_w(d, title, f_sq_title, 4)
    tx = (SQ - tw) / 2
    spaced(d, (tx + 2, 26), title, f_sq_title, (86, 40, 120), 4)
    spaced(d, (tx, 24), title, f_sq_title, PAPER, 4)
    d.rectangle([tx - 4, 64, tx + tw + 4, 69], fill=MAGENTA)
    d.text((tx - 4, 78), "EPISODE SIGNAL PATH", font=f_fig, fill=CYAN)

    f_sq_cell = ImageFont.truetype(BOLD, 18)
    f_sq_sub = ImageFont.truetype(REG, 14)
    order = [(0, 0), (1, 0), (0, 1), (1, 1)]   # reading order: Z path
    for i, ((name, sub, hue), (cx, cy)) in enumerate(zip(CELLS, order)):
        x, y = SQ_XS[cx], SQ_YS[cy]
        lit = 1.0 if i == 2 else 0.25   # MASTER glows, as on the card
        bevel(d, x, y, SQ_CELL_W, SQ_CELL_H, mix((36, 24, 74), (58, 40, 108), lit),
              mix((120, 96, 190), hue, lit), (22, 12, 48))
        d.rectangle([x + 3, y + 3, x + SQ_CELL_W - 3, y + 26], fill=hue)
        d.text((x + 10, y + 5), name, font=f_sq_cell, fill=(20, 12, 40))
        d.text((x + 10, y + SQ_CELL_H - 24), sub, font=f_sq_sub,
               fill=mix((168, 156, 214), PAPER, lit))
        if i == 2:
            x0, x1, mid = x + 10, x + SQ_CELL_W - 10, y + 52
            pts = [(px, mid + 14 * math.sin((px - x0) / (x1 - x0) * 9.4)
                    * math.sin((px - x0) / (x1 - x0) * math.pi))
                   for px in range(int(x0), int(x1))]
            d.line(pts, fill=MINT, width=2)
        else:
            for r, wfrac in enumerate((0.74, 0.52, 0.64)):
                yb = y + 38 + r * 10
                d.line([(x + 10, yb), (x + 10 + (SQ_CELL_W - 20) * wfrac, yb)],
                       fill=mix((96, 82, 158), hue, 0.6), width=3)

    # connectors: SCRIPT -> VOICES, down and back to MASTER, across to EPISODE
    arrow = (150, 130, 220)
    ym = SQ_YS[0] + SQ_CELL_H / 2
    d.line([(SQ_XS[0] + SQ_CELL_W + 4, ym), (SQ_XS[1] - 10, ym)], fill=arrow, width=4)
    d.polygon([(SQ_XS[1] - 12, ym - 8), (SQ_XS[1] - 1, ym), (SQ_XS[1] - 12, ym + 8)], fill=arrow)
    xr = SQ_XS[1] + SQ_CELL_W / 2
    xl = SQ_XS[0] + SQ_CELL_W / 2
    yv0, yv1 = SQ_YS[0] + SQ_CELL_H + 3, SQ_YS[1] - 3
    ymid = (yv0 + yv1) / 2
    d.line([(xr, yv0), (xr, ymid)], fill=arrow, width=4)
    d.line([(xr, ymid), (xl, ymid)], fill=arrow, width=4)
    d.line([(xl, ymid), (xl, yv1 - 8)], fill=arrow, width=4)
    d.polygon([(xl - 8, yv1 - 10), (xl, yv1), (xl + 8, yv1 - 10)], fill=arrow)
    ym2 = SQ_YS[1] + SQ_CELL_H / 2
    d.line([(SQ_XS[0] + SQ_CELL_W + 4, ym2), (SQ_XS[1] - 10, ym2)], fill=arrow, width=4)
    d.polygon([(SQ_XS[1] - 12, ym2 - 8), (SQ_XS[1] - 1, ym2), (SQ_XS[1] - 12, ym2 + 8)], fill=arrow)

    for cx, cy in ((12, 12), (SQ - 12, 12), (12, SQ - 12), (SQ - 12, SQ - 12)):
        d.line([(cx - 7, cy), (cx + 7, cy)], fill=GRID, width=1)
        d.line([(cx, cy - 7), (cx, cy + 7)], fill=GRID, width=1)
    return img


if __name__ == "__main__":
    import sys

    ASSETS = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\assets"
    thumb = ASSETS + r"\otr_gallery_thumb.jpg"
    square_thumbnail().save(thumb, quality=90)
    print("wrote", thumb)
    # The registry card is hand-finished beyond this script (see the module
    # docstring), so it is only rewritten on an explicit request. On
    # 2026-09-25 an unguarded run overwrote it; it was restored from git.
    if "--registry-gif" in sys.argv:
        frames = [draw_frame(k) for k in range(FRAMES)]
        out = ASSETS + r"\otr_icon.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=DUR_MS,
                       loop=0, optimize=True, disposal=2)
        frames[8].save(out.replace(".gif", ".png"))
        print("wrote", out)
