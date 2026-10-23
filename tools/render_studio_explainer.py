"""Render the Radio Play Studio explainer rough cut from its script.

The script is planning/studio-explainer/STUDIO_EXPLAINER.md, in the house
motion-graphics format (one block per scene: VISUAL, NARRATION, ON-SCREEN
TEXT). This draws one wireframe frame per scene in the Studio's own palette
(contest/radio-play-studio/radio_play_studio/static/index.html: ink, panel,
paper, muted, line, clay), narrates each scene with the local Kokoro voice,
and pipes the frames to ffmpeg. It shares its engine -- fonts, dashes, glows,
narration, the timeline, the mix, the ffmpeg pipe -- with
tools/render_explainer.py, the App Mode cut.

    python tools/render_studio_explainer.py --out cut.mp4 [--voice bm_george] [--cache DIR]
    python tools/render_studio_explainer.py --stills DIR      # one frame per scene, no audio
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_explainer import (FPS, H, REPO, SR, W, dashed, ease, font, mix_audio,  # noqa: E402
                              narrate, otr_proc, timeline)

SCRIPT = REPO / "planning" / "studio-explainer" / "STUDIO_EXPLAINER.md"

# The Studio's palette, as its page declares it for the dark scheme.
INK, PANEL = (23, 19, 15), (33, 27, 21)
PAPER, MUTED, LINE = (240, 230, 214), (168, 154, 134), (58, 48, 38)
CLAY, CLAY_INK = (224, 112, 63), (26, 16, 7)
OK, BAD = (126, 194, 138), (240, 138, 126)
RAISED = (44, 36, 28)

# The page's own words (static/i18n/en.json), never retyped.
STAGES = [("Stills", "22 painted, 1 to paint"), ("Video", "21 of 22 filmed"),
          ("Publish", "after every shot is filmed")]
BADGES = ["Changed, not painted yet", "Action changed, not filmed yet", "Painted, not filmed",
          "To film again", "Filmed", "Kept", "Failed"]
CAST = ["Announcer", "Jeffrey", "MagnoliaRex"]
CARD_LINE = "Goodnight, old friend. You kept us honest for twenty years."
CARD_PICTURE = "a man in his late 40s with heavy eyelids, a grease-stained work shirt, alone at a dimmed broadcast console"
CARD_MOTION = "He rests his forehead against the console and slowly rotates a large brass dial clockwise."
LANE_FIXED = "This shot's motion is set by its lane."
HINT = "Only the changed shots are painted; everything else is kept. Paint them before filming."
LANGUAGES = ["English", "Español", "Français", "हिन्दी", "Italiano", "日本語", "Português", "简体中文", "繁體中文"]


# --------------------------------------------------------------------------- #
# the script
# --------------------------------------------------------------------------- #

def read_script() -> tuple:
    text = SCRIPT.read_text(encoding="utf-8")
    scenes = []
    for m in re.finditer(r"^## SCENE (\d+) — (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        body = m.group(3)
        narr = re.search(r"\*\*NARRATION:\*\*\s*(.+?)(?:\n\n|\Z)", body, re.S).group(1)
        narr = " ".join(narr.split())
        ost = re.search(r"\*\*ON-SCREEN TEXT:\*\* (.+)", body).group(1).strip()
        scenes.append({"n": int(m.group(1)), "title": m.group(2).strip(), "narration": narr,
                       "ost": ost})
    sources = re.findall(r"^• (.+)$", text, re.M)
    return scenes, sources


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def background(t: float) -> Image.Image:
    px = np.linspace(0, 1, H)[:, None, None]
    arr = np.array(INK)[None, None, :] * (1 - px) + np.array(PANEL)[None, None, :] * px
    img = Image.fromarray(np.repeat(arr.astype(np.uint8), W, axis=1))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    horizon = int(H * 0.68)
    vanish = (W // 2, horizon - H // 3)
    col = MUTED + (40,)
    for i in range(-14, 15):
        d.line([vanish, (W // 2 + i * W // 10, H)], fill=col, width=1)
    for k in range(1, 12):
        phase = ((k + t * 0.35) % 11) / 11
        y = horizon + int((H - horizon) * phase ** 1.9)
        d.line([(0, y), (W, y)], fill=col, width=1)
    d.line([(0, horizon), (W, horizon)], fill=CLAY + (90,), width=2)
    img.paste(layer, (0, 0), layer)
    return img


def glow_rect(frame: Image.Image, box, color, strength=10):
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rectangle(box, outline=color + (200,), width=6)
    layer = layer.filter(ImageFilter.GaussianBlur(strength))
    frame.paste(layer, (0, 0), layer)


def caption(d: ImageDraw.ImageDraw, frame: Image.Image, text: str):
    band = Image.new("RGBA", (W, 74), (0, 0, 0, 160))
    frame.paste(band, (0, H - 74), band)
    f = font("mono", 28)
    while d.textlength(text, font=f) > W - 80 and len(text) > 10:
        text = text[:-4] + "…"
    d.text((40, H - 54), text, font=f, fill=CLAY)


def title(d: ImageDraw.ImageDraw, n: int, text: str):
    d.text((60, 44), text.upper(), font=font("title", 40), fill=PAPER)
    d.text((60, 96), "SCENE %02d" % n, font=font("mono", 20), fill=MUTED)


#: Fonts for the words Segoe UI has no glyphs for: the page's own language names.
_SCRIPT_FONTS = {"devanagari": "C:/Windows/Fonts/Nirmala.ttc", "cjk": "C:/Windows/Fonts/msyh.ttc"}
_script_font_cache: dict = {}


def font_for(text: str, size: int):
    """The body font, or one that can draw ``text`` when it is Devanagari or CJK."""
    from PIL import ImageFont
    kind = None
    for ch in text:
        if 0x0900 <= ord(ch) <= 0x097F:
            kind = "devanagari"
        elif ord(ch) >= 0x3000:
            kind = "cjk"
    if kind is None:
        return font("body", size)
    key = (kind, size)
    if key not in _script_font_cache:
        _script_font_cache[key] = ImageFont.truetype(_SCRIPT_FONTS[kind], size)
    return _script_font_cache[key]


def box(d, xy, lit=False, fill=None, width=2):
    d.rectangle(xy, fill=fill, outline=CLAY if lit else LINE, width=3 if lit else width)


def button(d, frame, xy, label, kind="quiet", size=22, lit=False):
    """kind: 'primary' (solid clay), 'quiet' (outlined), 'dim' (outlined, muted)."""
    x0, y0, x1, y1 = xy
    if kind == "primary":
        d.rounded_rectangle(xy, radius=8, fill=CLAY)
        ink = CLAY_INK
    else:
        d.rounded_rectangle(xy, radius=8, outline=CLAY if lit else LINE, width=2)
        ink = PAPER if kind == "quiet" else MUTED
    f = font("bold", size)
    tw = d.textlength(label, font=f)
    d.text(((x0 + x1 - tw) / 2, y0 + (y1 - y0 - size) / 2 - 3), label, font=f, fill=ink)
    if lit:
        glow_rect(frame, xy, CLAY, 8)


def badge(d, x, y, text, tone="clay", size=16):
    f = font("bold", size)
    tw = int(d.textlength(text, font=f))
    bg, ink = {"clay": ((59, 42, 26), (240, 160, 122)), "ok": ((31, 51, 36), OK),
               "bad": ((58, 34, 32), BAD), "kept": ((44, 41, 34), (200, 185, 163))}[tone]
    d.rounded_rectangle((x, y, x + tw + 20, y + size + 12), radius=999, fill=bg)
    d.text((x + 10, y + 4), text, font=f, fill=ink)
    return tw + 20


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def text_box(d, xy, label, text, lit=False, rows=2):
    x0, y0, x1, y1 = xy
    d.text((x0, y0 - 26), label, font=font("body", 18), fill=MUTED)
    box(d, xy, lit=lit, fill=INK)
    f = font("body", 18)
    for i, line in enumerate(wrap(d, text, f, x1 - x0 - 24)[:rows]):
        d.text((x0 + 12, y0 + 8 + i * 24), line, font=f, fill=PAPER)


def still(d, xy, label="", lit=False, redraw=0.0):
    """A wireframe still: a radio console under a lamp. redraw 0..1 flashes it."""
    x0, y0, x1, y1 = xy
    box(d, xy, lit=lit, fill=RAISED)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    w, h = x1 - x0, y1 - y0
    col = CLAY if lit else MUTED
    # console
    d.rectangle((cx - w * 0.3, cy, cx + w * 0.3, cy + h * 0.3), outline=col, width=2)
    for i in range(5):
        d.ellipse((cx - w * 0.25 + i * w * 0.12, cy + h * 0.08, cx - w * 0.25 + i * w * 0.12 + 14,
                   cy + h * 0.08 + 14), outline=col, width=2)
    # the dial and the lamp
    d.ellipse((cx + w * 0.16, cy + h * 0.06, cx + w * 0.26, cy + h * 0.24), outline=col, width=2)
    d.line([(cx, y0 + 8), (cx, cy - h * 0.18)], fill=col, width=2)
    d.polygon([(cx - w * 0.12, cy - h * 0.18), (cx + w * 0.12, cy - h * 0.18), (cx, cy - h * 0.05)],
              outline=col)
    if redraw > 0 and math_sin_window(redraw) > 0.2:
        # The redraw: the frame flashes clay for the beat the still is repainted.
        d.rectangle(xy, outline=CLAY, width=4)
    if label:
        d.text((x0 + 10, y1 - 30), label, font=font("mono", 15), fill=MUTED)


def math_sin_window(x: float) -> float:
    import math
    return max(0.0, math.sin(min(1.0, max(0.0, x)) * math.pi))


def card(frame, d, xy, who, line, picture, motion, lit_box=None, lane_fixed=False,
         badge_text="Changed, not painted yet", badge_tone="clay", redraw=0.0, lit=False):
    x0, y0, x1, y1 = xy
    box(d, xy, lit=lit, fill=PANEL)
    if lit:
        glow_rect(frame, xy, CLAY, 8)
    sw = int((x1 - x0) * 0.26)
    still(d, (x0 + 18, y0 + 18, x0 + 18 + sw, y0 + 18 + int(sw * 9 / 16)), lit=lit_box == "still",
          redraw=redraw)
    badge(d, x0 + 18, y0 + 30 + int(sw * 9 / 16), badge_text, badge_tone)
    tx = x0 + 36 + sw
    bw = x1 - 190 - tx
    f = font("body", 20)
    d.text((tx, y0 + 18), who, font=font("bold", 20), fill=CLAY)
    yy = y0 + 46
    for ln in wrap(d, line, f, bw)[:2]:
        d.text((tx, yy), ln, font=f, fill=PAPER)
        yy += 26
    text_box(d, (tx, yy + 32, tx + bw, yy + 32 + 62), "Picture", picture, lit=lit_box == "picture")
    if lane_fixed:
        d.text((tx, yy + 112), LANE_FIXED, font=font("body", 18), fill=MUTED)
    else:
        text_box(d, (tx, yy + 128, tx + bw, yy + 128 + 62), "Motion", motion, lit=lit_box == "motion")
    bx = x1 - 170
    button(d, frame, (bx, y0 + 18, x1 - 18, y0 + 60), "Paint again", "quiet", 18, lit=lit_box == "paint")
    button(d, frame, (bx, y0 + 70, x1 - 18, y0 + 112), "Film again", "quiet", 18, lit=lit_box == "film")
    button(d, frame, (bx, y0 + 122, x1 - 18, y0 + 164), "Keep", "quiet", 18, lit=lit_box == "keep")


def stage_markers(frame, d, x, y, w, scale=1.0, lit_until=3, bar=0.62):
    """The three stages with their state lines; returns the height used."""
    r = int(18 * scale)
    gap = w // 3
    for i, (name, state) in enumerate(STAGES):
        cx = x + i * gap + r + 10
        state_of = "done" if i == 0 else ("on" if i == 1 else "off")
        if i < lit_until:
            if state_of == "done":
                d.ellipse((cx - r, y, cx + r, y + 2 * r), fill=CLAY)
                d.text((cx - 6 * scale, y + 2 * scale), str(i + 1), font=font("bold", int(20 * scale)), fill=CLAY_INK)
            elif state_of == "on":
                d.ellipse((cx - r, y, cx + r, y + 2 * r), outline=CLAY, width=3)
                d.text((cx - 6 * scale, y + 2 * scale), str(i + 1), font=font("bold", int(20 * scale)), fill=CLAY)
            else:
                d.ellipse((cx - r, y, cx + r, y + 2 * r), outline=LINE, width=3)
                d.text((cx - 6 * scale, y + 2 * scale), str(i + 1), font=font("bold", int(20 * scale)), fill=MUTED)
        d.text((cx + r + 12, y - 2), name, font=font("bold", int(24 * scale)),
               fill=PAPER if state_of != "off" else MUTED)
        d.text((cx + r + 12, y + int(28 * scale)), state, font=font("mono", int(17 * scale)), fill=MUTED)
    by = y + int(66 * scale)
    d.rectangle((x, by, x + w, by + 8), fill=LINE)
    d.rectangle((x, by, x + int(w * bar), by + 8), fill=CLAY)
    return int(80 * scale)


def portrait(d, xy, name, lit=False):
    box(d, xy, lit=lit, fill=RAISED)
    x0, y0, x1, y1 = xy
    cx = (x0 + x1) // 2
    col = CLAY if lit else MUTED
    r = (x1 - x0) // 5
    d.ellipse((cx - r, y0 + (y1 - y0) * 0.18, cx + r, y0 + (y1 - y0) * 0.18 + 2 * r), outline=col, width=2)
    d.arc((cx - 2 * r, y0 + (y1 - y0) * 0.52, cx + 2 * r, y1 + r), 180, 360, fill=col, width=2)
    d.text((x0 + 8, y1 - 28), name, font=font("bold", 16), fill=PAPER)


def action_bar(frame, d, y, pressed=0.0, lit=True, counts=True):
    d.rectangle((0, y, W, y + 96), fill=PANEL)
    d.line([(0, y), (W, y)], fill=LINE, width=2)
    label = "Paint the stills (1 changed)" if counts else "Paint the stills"
    bx = 40
    bw = int(d.textlength(label, font=font("bold", 22))) + 44
    yy = y + 26 + int(4 * pressed)
    button(d, frame, (bx, yy, bx + bw, yy + 44), label, "primary", 22, lit=lit)
    bx += bw + 16
    label = "Film the shots (0 to film)" if counts else "Film the shots"
    bw = int(d.textlength(label, font=font("bold", 22))) + 44
    button(d, frame, (bx, y + 26, bx + bw, y + 70), label, "dim", 22)
    bx += bw + 16
    bw = int(d.textlength("Publish the episode", font=font("bold", 22))) + 44
    button(d, frame, (bx, y + 26, bx + bw, y + 70), "Publish the episode", "dim", 22)
    bx += bw + 24
    f = font("body", 18)
    for i, ln in enumerate(wrap(d, HINT, f, W - bx - 40)[:2]):
        d.text((bx, y + 28 + i * 22), ln, font=f, fill=MUTED)


def window(d, xy, name, dark=False):
    x0, y0, x1, y1 = xy
    d.rectangle(xy, fill=INK if dark else PANEL, outline=LINE if dark else PAPER, width=2)
    d.rectangle((x0, y0, x1, y0 + 40), fill=RAISED, outline=LINE if dark else PAPER, width=2)
    d.text((x0 + 14, y0 + 8), name, font=font("bold", 20), fill=MUTED if dark else PAPER)


# --------------------------------------------------------------------------- #
# the scenes
# --------------------------------------------------------------------------- #

def scene_frame(scene: dict, t: float, sources: list) -> Image.Image:
    n = scene["n"]
    frame = background(t * 6)
    d = ImageDraw.Draw(frame)
    e = ease(t)

    if n == 1:
        # One card, floating; the Picture box lights, the still redraws once.
        lit = "picture" if 0.3 < t < 0.75 else None
        redraw = (t - 0.5) / 0.2 if 0.5 < t < 0.7 else 0.0
        card(frame, d, (980, 280, 1860, 640), "Jeffrey", CARD_LINE, CARD_PICTURE, CARD_MOTION,
             lit_box=lit, redraw=redraw, lit=t > 0.3)
        d.text((60, 320), "ONE PICTURE.", font=font("title", 110), fill=PAPER)
        d.text((60, 450), "ONE STILL. ONE CLIP.", font=font("title", 72), fill=CLAY)
        d.text((60, 560), "The other twenty-one are reused.", font=font("body", 36), fill=MUTED)

    elif n == 2:
        # The Studio page: title, language picker, two mode cards, the two buttons.
        x, y = 260, 170
        d.rectangle((x, y, x + 1400, y + 720), fill=PANEL, outline=LINE, width=2)
        d.text((x + 40, y + 36), "Radio Play Studio", font=font("title", 44), fill=PAPER)
        d.text((x + 40, y + 100), "Press the button. Your own ComfyUI writes, casts, performs and films a radio play.",
               font=font("body", 22), fill=MUTED)
        lx = x + 1000
        box(d, (lx, y + 36, x + 1360, y + 80), fill=INK)
        d.text((lx + 14, y + 44), "English", font=font("body", 22), fill=PAPER)
        d.polygon([(x + 1330, y + 52), (x + 1346, y + 52), (x + 1338, y + 64)], fill=PAPER)
        for i, lang in enumerate(LANGUAGES):
            d.text((lx + 14, y + 96 + i * 26), lang, font=font_for(lang, 18), fill=MUTED)
        cards = [("A classic, or surprise me", "Adapt Shakespeare or a public-domain story, or roll the dice.", True),
                 ("Tell my own story", "Write who is in it, what happens and where. The rest is filled in for you.", t > 0.35)]
        for i, (head, sub, on) in enumerate(cards):
            cy = y + 170 + i * 150
            box(d, (x + 40, cy, x + 940, cy + 120), lit=on and (i == 1 or t < 0.35), fill=INK)
            d.ellipse((x + 62, cy + 46, x + 86, cy + 70), outline=CLAY if on else LINE, width=3,
                      fill=CLAY if (on and i == 1 and t > 0.35) else None)
            d.text((x + 104, cy + 24), head, font=font("bold", 28), fill=PAPER)
            d.text((x + 104, cy + 66), sub, font=font("body", 20), fill=MUTED)
        button(d, frame, (x + 40, y + 500, x + 360, y + 560), "Make the episode", "primary", 24)
        button(d, frame, (x + 384, y + 500, x + 700, y + 560), "Direct the shots first", "quiet", 22,
               lit=t > 0.6)
        if t > 0.6:
            dashed(d, (x + 980, y + 470), (x + 712, y + 530), PAPER, width=3)
            d.text((x + 990, y + 446), "the quiet button", font=font("mono", 20), fill=PAPER)
        d.text((x + 40, y + 600), "Direct the shots first: the play is written, cast and voiced as usual, then it stops after the stills.",
               font=font("body", 20), fill=MUTED)

    elif n == 3:
        # The board in three bands.
        bands = [("Stages", 0.12), ("Cast", 0.38), ("Scenes", 0.5), ("One bar", 0.75)]
        lit = {name: t >= at for name, at in bands}
        x0 = 220
        d.rectangle((x0 - 20, 150, x0 + 1500, 940), fill=PANEL, outline=LINE, width=2)
        stage_markers(frame, d, x0 + 20, 190, 1400, scale=1.0)
        if lit["Stages"]:
            glow_rect(frame, (x0, 170, x0 + 1460, 290), CLAY, 8)
        for i, name in enumerate(CAST):
            portrait(d, (x0 + 20 + i * 150, 330, x0 + 150 + i * 150, 500), name, lit=lit["Cast"])
            button(d, frame, (x0 + 20 + i * 150, 508, x0 + 150 + i * 150, 540), "Paint again", "quiet", 14)
        if lit["Cast"]:
            glow_rect(frame, (x0, 320, x0 + 480, 550), CLAY, 8)
        sx = x0 + 540
        for i, (name, extra) in enumerate([("Scene 1", "9 shots · the announcer"), ("Scene 2", "13 shots · 1 changed")]):
            yy = 330 + i * 110
            box(d, (sx, yy, x0 + 1460, yy + 90), lit=lit["Scenes"], fill=INK)
            d.text((sx + 20, yy + 18), name, font=font("bold", 26), fill=PAPER)
            d.text((sx + 20, yy + 54), extra, font=font("body", 18), fill=MUTED)
            for k in range(6):
                box(d, (sx + 360 + k * 90, yy + 16, sx + 436 + k * 90, yy + 74), fill=RAISED, width=1)
        if lit["Scenes"]:
            glow_rect(frame, (sx - 10, 320, x0 + 1470, 550), CLAY, 8)
        action_bar(frame, d, 830, lit=lit["One bar"])
        if lit["One bar"]:
            glow_rect(frame, (0, 830, W, 926), CLAY, 8)

    elif n == 4:
        # The three markers, large, with their exact state lines.
        lit_until = 1 + int(min(2, t * 3.3))
        stage_markers(frame, d, 240, 420, 1440, scale=2.0, lit_until=lit_until)
        for i, (name, state) in enumerate(STAGES):
            cx = 240 + i * 480 + 60
            d.text((cx, 300), '"%s"' % state, font=font("mono", 22),
                   fill=CLAY if i < lit_until else LINE)

    elif n == 5:
        # The cast strip; a dashed line from Jeffrey's portrait to his cards.
        for i, name in enumerate(CAST):
            lit = name == "Jeffrey" and t > 0.35
            portrait(d, (140 + i * 260, 300, 360 + i * 260, 600), name, lit=lit)
            button(d, frame, (140 + i * 260, 616, 360 + i * 260, 660), "Paint again", "quiet", 20, lit=lit)
        d.text((140, 700), "A new portrait repaints every shot that character is in, except the kept ones,",
               font=font("body", 22), fill=MUTED)
        d.text((140, 732), "so try the shots first.", font=font("body", 22), fill=MUTED)
        cards = [("Jeffrey", "Filmed", "ok"), ("MagnoliaRex", "Filmed", "ok"), ("Jeffrey", "Kept", "kept"),
                 ("Jeffrey", "Filmed", "ok"), ("MagnoliaRex", "Filmed", "ok"), ("Jeffrey", "Kept", "kept")]
        for k, (who, state, tone) in enumerate(cards):
            cy = 220 + k * 120
            dim = tone == "kept"
            box(d, (1180, cy, 1800, cy + 100), fill=INK if not dim else PANEL,
                lit=(who == "Jeffrey" and not dim and t > 0.5))
            d.text((1200, cy + 14), who, font=font("bold", 20), fill=CLAY if not dim else MUTED)
            badge(d, 1200, cy + 50, state, tone)
            if who == "Jeffrey" and t > 0.5:
                if not dim:
                    dashed(d, (360 + 260, 450), (1180, cy + 50), PAPER, width=2)
        d.text((1180, 950), "dashed: where a new portrait would reach", font=font("mono", 18), fill=PAPER)

    elif n == 6:
        # One card, full frame, each element lit as it is named; a lane-fixed twin.
        order = [None, "still", None, "picture", "motion", "paint", "film", "keep"]
        lit = order[min(len(order) - 1, int(t * len(order)))]
        card(frame, d, (120, 200, 1260, 560), "Jeffrey", CARD_LINE, CARD_PICTURE, CARD_MOTION,
             lit_box=lit, badge_text="Painted, not filmed", badge_tone="clay")
        card(frame, d, (1300, 560, 1880, 900), "ANNOUNCER", "Good evening.",
             "a vintage tabletop tube radio receiver in a broadcast booth", "", lane_fixed=True,
             badge_text="Filmed", badge_tone="ok")
        d.text((1300, 520), "where the lane fixes the motion:", font=font("mono", 18), fill=PAPER)

    elif n == 7:
        # Seven badges, lighting in turn.
        lit_n = 1 + int(t * 8)
        for i, text in enumerate(BADGES):
            y = 180 + i * 96
            tone = "bad" if text == "Failed" else ("ok" if text == "Filmed" else ("kept" if text == "Kept" else "clay"))
            if i < lit_n:
                w = badge(d, 160, y, text, tone, size=28)
                d.text((160 + w + 30, y + 10), ["waiting for a Stills run", "the motion moved; waiting for a Video run",
                                                "waiting for a Video run", "a new take is asked for",
                                                "done", "frozen as it is", "blocks Publish until painted, filmed or kept"][i],
                       font=font("body", 22), fill=MUTED)
            else:
                d.rounded_rectangle((160, y, 160 + 360, y + 40), radius=999, outline=LINE, width=2)
        d.text((1200, 300), "7 STATES", font=font("title", 72), fill=PAPER)
        d.text((1200, 400), "FAILED BLOCKS", font=font("title", 72), fill=BAD)
        d.text((1200, 490), "PUBLISH", font=font("title", 72), fill=BAD)

    elif n == 8:
        # The sticky bar, large; the clay button depresses at the end.
        action_bar(frame, d, 440, pressed=1.0 if t > 0.88 else 0.0)
        glow_rect(frame, (0, 440, W, 536), CLAY, 6)
        lines = ["Only what changed is made again.",
                 "A Stills run paints the changed stills and keeps the rest.",
                 "A Video run films the shots that are not filmed yet, or that you changed.",
                 "Publish mixes the voices and music under the filmed clips and loads no generation weights."]
        for i, ln in enumerate(lines):
            if t > 0.15 + i * 0.17:
                d.text((120, 600 + i * 48), ln, font=font("body", 28), fill=PAPER if i == 0 else MUTED)

    elif n == 9:
        # The receipts ledger: four solid rows, one dashed.
        rows = [("one picture edit", "1 still repainted"), ("then Video", "1 clip filmed, 21 reused"),
                ("one motion edit", "1 clip filmed, 21 reused"), ("no edit", "22 reused")]
        x, y = 200, 220
        d.text((x, y - 60), "2026-10-23 · 16 GB laptop · 22 shots", font=font("mono", 26), fill=MUTED)
        for i, (a, b) in enumerate(rows):
            yy = y + i * 120
            if t > 0.12 + i * 0.16:
                d.rectangle((x, yy, x + 1500, yy + 90), outline=CLAY, width=3, fill=PANEL)
                d.text((x + 30, yy + 24), a, font=font("bold", 32), fill=PAPER)
                d.text((x + 700, yy + 24), "→", font=font("bold", 32), fill=CLAY)
                d.text((x + 780, yy + 24), b, font=font("mono", 32), fill=CLAY)
        yy = y + 4 * 120
        if t > 0.8:
            for (a, b) in [((x, yy), (x + 1500, yy)), ((x, yy + 90), (x + 1500, yy + 90)),
                           ((x, yy), (x, yy + 90)), ((x + 1500, yy), (x + 1500, yy + 90))]:
                dashed(d, a, b, PAPER, width=2)
            d.text((x + 30, yy + 24), "your edit", font=font("bold", 32), fill=PAPER)
            d.text((x + 700, yy + 24), "→", font=font("bold", 32), fill=PAPER)
            d.text((x + 780, yy + 24), "?", font=font("mono", 32), fill=PAPER)

    elif n == 10:
        # Two windows; ComfyUI goes dark and comes back; one card re-films.
        dark = 0.3 < t < 0.55
        window(d, (120, 200, 920, 900), "Radio Play Studio")
        window(d, (1000, 200, 1800, 900), "ComfyUI", dark=dark)
        stage_markers(frame, d, 160, 270, 720, scale=0.9, bar=0.62 if t < 0.3 else (0.62 if t < 0.6 else 0.62 + 0.3 * ease((t - 0.6) / 0.4)))
        if 0.3 < t:
            d.text((160, 380), "ComfyUI no longer has this run.", font=font("mono", 22), fill=BAD if t < 0.6 else MUTED)
        if t > 0.6:
            d.text((160, 420), "Video, again:", font=font("bold", 22), fill=PAPER)
        for k in range(6):
            yy = 470 + k * 64
            refilm = k == 1 and t > 0.6
            box(d, (160, yy, 880, yy + 52), lit=refilm, fill=INK)
            d.text((180, yy + 14), ["shot_shot_001_b1", "shot_shot_001_b2", "shot_shot_001_b3",
                                    "shot_shot_001_b4", "shot_shot_001_b5", "shot_shot_001_b6"][k],
                   font=font("mono", 18), fill=PAPER if refilm else MUTED)
            if t > 0.6:
                badge(d, 640, yy + 10, "filming" if refilm else "REUSED", "clay" if refilm else "kept", 14)
        if not dark:
            for k in range(8):
                d.line([(1040, 300 + k * 60), (1760, 300 + k * 60)], fill=LINE, width=1)
            d.text((1040, 820), "the render" if t < 0.3 else "back; the lost run is gone", font=font("mono", 20), fill=MUTED)
        else:
            d.text((1300, 540), "— off —", font=font("mono", 28), fill=LINE)

    elif n == 11:
        # Three branches of a folder tree; the obs file solid clay.
        def tree(x, y, items, lit_last=False):
            for i, (depth, name) in enumerate(items):
                xx, yy = x + depth * 44, y + i * 44
                last = lit_last and i == len(items) - 1
                if depth:
                    d.line([(xx - 30, yy - 20), (xx - 30, yy + 14), (xx - 8, yy + 14)], fill=LINE, width=2)
                d.text((xx, yy), name, font=font("mono", 24), fill=CLAY if last else PAPER)
                if last:
                    d.polygon([(xx - 60, yy + 4), (xx - 60, yy + 28), (xx - 40, yy + 16)], fill=CLAY)
        if t > 0.05:
            tree(140, 200, [(0, "output/"), (1, "otr/"), (2, "episodes/"), (3, "_replay/"), (4, "p/"), (5, "‹project›/"),
                            (6, "01-stills/"), (6, "02-stills/"), (6, "03-video/"), (6, "…"), (6, "clips/")])
        if t > 0.4:
            tree(940, 200, [(0, "radio-play-studio/"), (1, "episodes/"), (2, "projects/"), (3, "‹project›/"),
                            (4, "project.json"), (4, "media/"), (4, "attempts/")])
        if t > 0.7:
            tree(940, 580, [(0, "output/"), (1, "otr/"), (2, "obs/"), (3, "last_frequency_…_final.mp4")], lit_last=True)

    title(d, n, scene["title"])
    caption(d, frame, scene["ost"])
    return frame


def end_card_frame(t: float, sources: list) -> Image.Image:
    frame = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(frame)
    if t > 0.25:
        d.text((120, 160), "SOURCES", font=font("mono", 34), fill=CLAY)
        scroll = int(40 * ease((t - 0.25) / 0.75))
        for i, line in enumerate(sources):
            d.text((120, 240 + i * 52 - scroll), "• " + line, font=font("mono", 26), fill=CLAY)
    return frame


# --------------------------------------------------------------------------- #
# the cut
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="the mp4 to write")
    ap.add_argument("--stills", help="write one frame per scene into this folder and stop")
    ap.add_argument("--voice", default="bm_george", help="a Kokoro voice id from the local voices folder")
    ap.add_argument("--cache", default=str(REPO / "tmp" / "studio_explainer_cache"))
    args = ap.parse_args(argv)
    scenes, sources = read_script()
    if args.stills:
        out = Path(args.stills)
        out.mkdir(parents=True, exist_ok=True)
        for s in scenes:
            for t in (0.35, 0.9):
                scene_frame(s, t, sources).save(out / f"scene_{s['n']:02d}_{int(t * 100):02d}.png")
        end_card_frame(0.9, sources).save(out / "end_card.png")
        print("wrote stills to", out)
        return 0
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    narrations = narrate(scenes, cache, args.voice)
    plan = timeline(scenes, narrations)
    audio = mix_audio(plan)
    import soundfile as sf
    wav = cache / "mix.wav"
    sf.write(str(wav), audio, SR)
    total_s = plan[-1]["start"] + plan[-1]["dur"]
    frames = int(total_s * FPS)
    print("cut: %d scenes, %.1f s, %d frames" % (len(scenes), total_s, frames))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)]
    proc = otr_proc.popen(cmd, stdin=subprocess.PIPE)
    i = 0
    for p in plan:
        n = int(p["dur"] * FPS)
        for k in range(n):
            t = k / max(1, n - 1)
            frame = scene_frame(p["scene"], t, sources) if p["scene"] else end_card_frame(t, sources)
            proc.stdin.write(frame.tobytes())
            i += 1
            if i % 240 == 0:
                print("  %d / %d frames" % (i, frames), flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", out, "rc", proc.returncode)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
