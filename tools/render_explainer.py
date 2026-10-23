"""Render the app-view explainer as a rough cut, in the house style.

Reads planning/app-view-explainer/APP_VIEW_EXPLAINER.md (the scenes, their
narration and on-screen text) and config/app_mode.json (the form's own labels
and help lines), draws each scene as an isometric wireframe world in the
script's locked palette, narrates it dry with the pack's own Kokoro voice from
the local model, and muxes the cut with ffmpeg. Developer tooling: tools/ does
not ship, and the cut is a rough one for the operator's pipeline to replace.

    python tools/render_explainer.py --stills OUT_DIR   # one frame per scene, to look at
    python tools/render_explainer.py --out cut.mp4      # the whole cut
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "planning" / "app-view-explainer" / "APP_VIEW_EXPLAINER.md"
CONFIG = REPO / "config" / "app_mode.json"
VOICES = Path(r"C:\ComfyUI-Models\TTS\KokoroTTS\voices")
W, H, FPS, SR = 1920, 1080, 24, 24000

BASE, BASE2 = (18, 10, 46), (38, 25, 78)
PAPER, MAGENTA, CYAN = (242, 239, 232), (240, 43, 143), (62, 233, 255)
DIM = (150, 146, 170)

FONTS = {
    "title": str(REPO / "assets" / "fonts" / "InterDisplay-Black.otf"),
    "body": r"C:\Windows\Fonts\segoeui.ttf",
    "bold": r"C:\Windows\Fonts\segoeuib.ttf",
    "mono": r"C:\Windows\Fonts\consola.ttf",
}
_font_cache: dict = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    key = (kind, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONTS[kind], size)
    return _font_cache[key]


# --------------------------------------------------------------------------- #
# the script and the form
# --------------------------------------------------------------------------- #

def read_script() -> tuple:
    text = SCRIPT.read_text(encoding="utf-8")
    scenes = []
    for m in re.finditer(r"^## SCENE (\d+) — (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        body = m.group(3)
        narr = re.search(r"\*\*NARRATION:\*\* (.+)", body).group(1).strip()
        ost = re.search(r"\*\*ON-SCREEN TEXT:\*\* (.+)", body).group(1).strip()
        scenes.append({"n": int(m.group(1)), "title": m.group(2).strip(), "narration": narr,
                       "ost": ost})
    sources = re.findall(r"^• (.+)$", text, re.M)
    return scenes, sources


def read_form() -> tuple:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for node, widget, note_key in cfg["form"]:
        rows.append({"key": widget, "label": cfg["labels"][f"{node}.{widget}"],
                     "note": cfg[note_key]})
    return rows, {r["key"]: r for r in rows}


#: The value each row shows on the shipped app workflow, as the form draws it.
VALUES = {
    "episode_language": "English", "num_characters": "2", "act_count": "1",
    "source_bank": "roll (any eligible bank)", "visual_style": "roll (any style)",
    "announcer_video_model": "viz_mxc_cpu (16:9) (audio-reactive, no scene image)",
    "character_video_model": "viz_camera (16:9) (audio-reactive, no scene image)",
    "music_video_model": "viz_green (16:9) (audio-reactive, no scene image)",
    "announcer_image_model": "z_image_turbo", "character_image_model": "z_image_turbo",
    "music_image_model": "z_image_turbo", "announcer_voice_engine": "kokoro",
    "char_voice_engine": "kokoro", "engine": "stable_audio_3",
    "creative_writing_model": "comfy_native:gemma4-e2b-it-int8-convrot",
    "technical_model": "comfy_native:gemma4-e2b-it-int8-convrot",
    "openrouter_slot_a_model": "(enable OpenRouter)", "openrouter_slot_b_model": "(enable OpenRouter)",
    "comfy_slot_a_model": "(enable Comfy Credits)", "comfy_slot_b_model": "(enable Comfy Credits)",
    "google_api_slot_a_model": "(select Google API model)",
    "google_api_slot_b_model": "(select Google API model)",
    "upscale_engine": "off", "lemmy_cameo": "roll (~11% chance)",
    "asset_cleanup": "full (keep only the published video)",
}
SWITCHES = {"roll_video_lanes", "roll_still_models", "roll_audio_engines"}
TEXTS = {"story_characters", "character_photos", "story_plot", "story_setting", "story_author",
         "music_style", "source_ref"}
POOLS = {"language_roll_pool", "bank_roll_pool", "style_roll_pool", "video_roll_pool",
         "still_roll_pool"}

#: Which rows each scene shows and lights, in order; "special" scenes draw their own world.
SCENE_ROWS = {
    1: "all",
    2: "install",
    3: "gallery",
    4: ["episode_language", "num_characters", "act_count"],
    5: ["source_bank", "source_ref", "visual_style"],
    6: ["roll_video_lanes", "roll_still_models", "roll_audio_engines"],
    7: ["announcer_video_model", "announcer_image_model", "character_video_model",
        "character_image_model", "music_video_model", "music_image_model"],
    8: ["announcer_voice_engine", "char_voice_engine", "engine", "creative_writing_model",
        "technical_model"],
    9: ["openrouter_slot_a_model", "openrouter_slot_b_model", "comfy_slot_a_model",
        "comfy_slot_b_model", "google_api_slot_a_model", "google_api_slot_b_model"],
    10: ["upscale_engine", "lemmy_cameo"],
    11: ["story_characters", "character_photos", "story_plot", "story_setting", "story_author",
         "music_style", "asset_cleanup", "language_roll_pool", "bank_roll_pool",
         "style_roll_pool", "video_roll_pool", "still_roll_pool"],
    12: "run",
}


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def background(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BASE)
    px = np.linspace(0, 1, H)[:, None]
    arr = (np.array(BASE)[None, None, :] * (1 - px[:, :, None]) + np.array(BASE2)[None, None, :] * px[:, :, None])
    img = Image.fromarray(np.repeat(arr.astype(np.uint8), W, axis=1))
    # The grid floor: lines fanning from a vanishing point, rows that drift toward the camera.
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    horizon = int(H * 0.68)
    vanish = (W // 2, horizon - H // 3)
    col = CYAN + (46,)
    for i in range(-14, 15):
        d.line([vanish, (W // 2 + i * W // 10, H)], fill=col, width=1)
    for k in range(1, 12):
        phase = ((k + t * 0.35) % 11) / 11
        y = horizon + int((H - horizon) * phase ** 1.9)
        d.line([(0, y), (W, y)], fill=col, width=1)
    d.line([(0, horizon), (W, horizon)], fill=MAGENTA + (110,), width=2)
    img.paste(layer, (0, 0), layer)
    return img


def dashed(d: ImageDraw.ImageDraw, a, b, fill, width=2, dash=12, gap=8):
    (x0, y0), (x1, y1) = a, b
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(length, pos + dash)
        d.line([(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)], fill=fill, width=width)
        pos += dash + gap


def glow_rect(frame: Image.Image, box, color, strength=10):
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rectangle(box, outline=color + (200,), width=6)
    layer = layer.filter(ImageFilter.GaussianBlur(strength))
    frame.paste(layer, (0, 0), layer)


def caption(d: ImageDraw.ImageDraw, frame: Image.Image, text: str):
    band = Image.new("RGBA", (W, 74), (0, 0, 0, 150))
    frame.paste(band, (0, H - 74), band)
    f = font("mono", 28)
    while d.textlength(text, font=f) > W - 80 and len(text) > 10:
        text = text[:-4] + "…"
    d.text((40, H - 54), text, font=f, fill=CYAN)


def title(d: ImageDraw.ImageDraw, n: int, text: str):
    d.text((60, 44), text.upper(), font=font("title", 40), fill=PAPER)
    d.text((60, 96), "SCENE %02d" % n, font=font("mono", 20), fill=DIM)


def row_height(row: dict, compact: bool) -> int:
    if compact:
        return 126 if row["key"] in TEXTS else 112
    return 150 if row["key"] in TEXTS else 136


def draw_row(frame: Image.Image, d: ImageDraw.ImageDraw, row: dict, x: int, y: int, w: int,
             lit: float, dim: bool = False, compact: bool = False) -> int:
    """One form row as the form draws it: bold label, one help line, the control.
    ``lit`` 0..1 is how far its highlight has come in. Returns the row's height."""
    h = row_height(row, compact)
    outline = PAPER if not dim else DIM
    if lit > 0:
        glow_rect(frame, (x, y, x + w, y + h), MAGENTA, strength=8 + int(6 * lit))
        d.rectangle((x, y, x + w, y + h), outline=MAGENTA, width=3)
    else:
        d.rectangle((x, y, x + w, y + h), outline=outline, width=2)
    d.text((x + 18, y + 10 if compact else y + 12), row["label"], font=font("bold", 26), fill=PAPER if not dim else DIM)
    d.text((x + 18, y + 42 if compact else y + 46), row["note"], font=font("body", 20), fill=(205, 200, 220) if not dim else DIM)
    cy = y + h - (40 if compact else 44)
    if row["key"] in SWITCHES:
        d.rounded_rectangle((x + w - 92, cy - 2, x + w - 30, cy + 26), radius=14, outline=outline, width=2)
        d.ellipse((x + w - 88, cy + 2, x + w - 68, cy + 22), fill=outline)
    elif row["key"] in POOLS:
        d.text((x + 18, cy), "▸  All eligible (none ticked)", font=font("mono", 20), fill=CYAN)
    elif row["key"] in TEXTS:
        d.rectangle((x + 18, cy - 10, x + w - 18, cy + 28), outline=outline, width=1)
    else:
        value = VALUES.get(row["key"], "")
        d.rectangle((x + 18, cy - 8, x + w - 18, cy + 28), outline=outline, width=1)
        f = font("mono", 20)
        while d.textlength(value, font=f) > w - 60 and len(value) > 8:
            value = value[:-4] + "…"
        d.text((x + 30, cy - 2), value, font=f, fill=CYAN)
    return h


def panel(frame: Image.Image, d: ImageDraw.ImageDraw, rows: list, x: int, y: int, w: int, t: float,
          dim_all: bool = False, lit_from: float = 0.1, lit_to: float = 0.75) -> list:
    """The form panel with these rows, lighting them one by one between
    ``lit_from`` and ``lit_to`` of the scene. Returns each row's box."""
    pad = 16
    budget = H - 100 - y
    compact = sum(row_height(r, False) + 12 for r in rows) > budget
    gap = 8 if compact else 12
    total = sum(row_height(r, compact) + gap for r in rows) + pad * 2
    d.rectangle((x - pad, y - pad, x + w + pad, y - pad + total), outline=(90, 80, 130), width=2)
    boxes = []
    yy = y
    for i, row in enumerate(rows):
        start = lit_from + (lit_to - lit_from) * i / max(1, len(rows))
        lit = ease((t - start) / 0.08) if t >= start else 0.0
        h = draw_row(frame, d, row, x, yy, w, lit, dim=dim_all, compact=compact)
        boxes.append((x, yy, x + w, yy + h))
        yy += h + gap
    return boxes


def run_button(d: ImageDraw.ImageDraw, frame: Image.Image, cx: int, cy: int, pressed: float, scale=1.0):
    w, h = int(300 * scale), int(74 * scale)
    box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    glow_rect(frame, box, MAGENTA, strength=14)
    fill = tuple(int(c * (1 - 0.25 * pressed)) for c in MAGENTA)
    d.rounded_rectangle(box, radius=int(12 * scale), fill=fill, outline=PAPER, width=2)
    f = font("title", int(34 * scale))
    tw = d.textlength("RUN", font=f)
    d.text((cx - tw / 2, cy - int(22 * scale)), "RUN", font=f, fill=PAPER)


# --------------------------------------------------------------------------- #
# the scenes
# --------------------------------------------------------------------------- #

def scene_frame(scene: dict, t: float, rows: list, by_key: dict, sources: list) -> Image.Image:
    """One frame of a scene at ``t`` in 0..1."""
    n = scene["n"]
    frame = background(t * 6)
    d = ImageDraw.Draw(frame)
    spec = SCENE_ROWS.get(n)
    if spec == "all":
        # The whole form, forty thin rows, and the button the camera pushes toward.
        x, y, w = 1180, 150, 560
        rh = 17
        d.rectangle((x - 14, y - 14, x + w + 14, y + len(rows) * rh + 14), outline=(90, 80, 130), width=2)
        for i, row in enumerate(rows):
            d.rectangle((x, y + i * rh, x + w, y + i * rh + 12), outline=PAPER, width=1)
            d.text((x + 8, y + i * rh - 2), row["label"], font=font("body", 11), fill=DIM)
        run_button(d, frame, x + w // 2, y + len(rows) * rh + 70, pressed=0.0, scale=0.9)
        d.text((60, 300), "ONE FORM.", font=font("title", 120), fill=PAPER)
        d.text((60, 440), "ONE RADIO PLAY.", font=font("title", 120), fill=MAGENTA)
        d.text((60, 600), "Forty rows and one button.", font=font("body", 36), fill=(205, 200, 220))
    elif spec == "install":
        # A ComfyUI window: the menu path lights one item at a time, the search types itself.
        x, y, w, h = 160, 170, 1100, 620
        d.rectangle((x, y, x + w, y + h), outline=PAPER, width=2)
        d.rectangle((x, y, x + w, y + 54), fill=(28, 18, 66), outline=PAPER, width=2)
        items = [("Workflow", 0.0), ("Extensions", 0.12), ("Node Manager", 0.26)]
        xx = x + 24
        for label, at in items:
            lit = t >= at
            d.text((xx, y + 12), label, font=font("bold", 26), fill=MAGENTA if lit else DIM)
            if lit:
                d.line([(xx, y + 48), (xx + d.textlength(label, font=font("bold", 26)), y + 48)], fill=MAGENTA, width=3)
            xx += int(d.textlength(label, font=font("bold", 26))) + 48
        # the search field
        d.rectangle((x + 40, y + 110, x + 760, y + 170), outline=PAPER, width=2)
        typed = "old time radio"[: int(max(0, (t - 0.32)) / 0.22 * 14)]
        d.text((x + 60, y + 124), typed + ("|" if (t * 4) % 1 < 0.5 and t < 0.6 else ""), font=font("mono", 30), fill=CYAN)
        # the card
        if t > 0.5:
            cx, cy = x + 40, y + 220
            d.rectangle((cx, cy, cx + 720, cy + 180), outline=PAPER, width=2)
            d.text((cx + 24, cy + 20), "Old-Time Radio", font=font("bold", 34), fill=PAPER)
            d.text((cx + 24, cy + 70), "fluxus  ·  comfyui-old-time-radio", font=font("mono", 22), fill=DIM)
            press = ease((t - 0.62) / 0.1)
            box = (cx + 520, cy + 110, cx + 696, cy + 160)
            if press > 0:
                glow_rect(frame, box, MAGENTA, strength=12)
            d.rounded_rectangle(box, radius=8, fill=MAGENTA if press > 0 else None, outline=PAPER, width=2)
            d.text((cx + 556, cy + 118), "Install", font=font("bold", 26), fill=PAPER)
        # the terminal
        tx, ty = 1340, 420
        d.rectangle((tx, ty, tx + 500, ty + 150), outline=CYAN, width=2)
        if t > 0.72:
            d.text((tx + 18, ty + 18), "> winget install Gyan.FFmpeg", font=font("mono", 22), fill=CYAN)
            d.text((tx + 18, ty + 54), "> brew install ffmpeg", font=font("mono", 22), fill=CYAN)
        if t > 0.88:
            d.text((tx + 18, ty + 104), "restart ComfyUI", font=font("mono", 22), fill=MAGENTA)
    elif spec == "gallery":
        # A shelf of workflow cards; one lifts and unfolds into the form.
        crumbs = ["WORKFLOW", "BROWSE TEMPLATES", "EXTENSIONS", "OLD-TIME RADIO"]
        xx = 60
        for i, c in enumerate(crumbs):
            lit = t >= 0.05 + i * 0.09
            d.text((xx, 150), c, font=font("mono", 24), fill=MAGENTA if lit else DIM)
            xx += int(d.textlength(c, font=font("mono", 24))) + 20
            if i < len(crumbs) - 1:
                d.text((xx, 150), "›", font=font("mono", 24), fill=DIM)
                xx += 30
        names = ["otr_16gb_foley", "otr_8gb_video", "otr_app", "otr_mac16_still", "otr_canonical", "otr_cloud_low"]
        for i, name in enumerate(names):
            col, rowi = i % 3, i // 3
            cx, cy = 100 + col * 330, 230 + rowi * 230
            lift = ease((t - 0.42) / 0.15) if name == "otr_app" else 0.0
            cy -= int(40 * lift)
            box = (cx, cy, cx + 290, cy + 180)
            if lift > 0:
                glow_rect(frame, box, MAGENTA, strength=10)
            d.rectangle(box, outline=MAGENTA if lift > 0 else PAPER, width=3 if lift > 0 else 2)
            d.text((cx + 16, cy + 140), name, font=font("mono", 20), fill=PAPER if lift > 0 else DIM)
        # the form appears on the right
        appear = ease((t - 0.6) / 0.2)
        if appear > 0:
            shown = rows[:5]
            x = 1180 + int((1 - appear) * 200)
            panel(frame, d, shown, x, 200, 600, t, lit_from=2, lit_to=3)
            d.text((x, 150), "APP MODE", font=font("title", 30), fill=CYAN)
    elif spec == "run":
        # RUN depresses, the outputs tray slides out, the folder tree draws itself.
        pressed = ease((t - 0.05) / 0.1)
        run_button(d, frame, 1480, 300, pressed=pressed, scale=1.4)
        tray = ease((t - 0.25) / 0.2)
        if tray > 0:
            th = int(200 * tray)
            d.rectangle((0, H - 74 - th, W, H - 74), fill=(26, 16, 60), outline=CYAN, width=2)
            d.text((60, H - 74 - th + 24), "Outputs", font=font("bold", 28), fill=PAPER)
            if tray >= 1:
                d.rectangle((60, H - 74 - th + 70, 60 + 220, H - 74 - th + 70 + 110), outline=MAGENTA, width=3)
                d.polygon([(150, H - 74 - th + 100), (150, H - 74 - th + 150), (195, H - 74 - th + 125)], fill=MAGENTA)
        # the folder tree
        tx, ty = 120, 180
        lines = [("output/", 0.4), ("  otr/", 0.5), ("    obs/", 0.6), ("      last_frequency_… .mp4", 0.7)]
        for i, (label, at) in enumerate(lines):
            if t >= at:
                d.text((tx, ty + i * 54), label, font=font("mono", 34), fill=MAGENTA if i == 3 else CYAN)
        if t > 0.45:
            dashed(d, (900, 180), (900, 420), CYAN, width=2)
            d.text((920, 280), "first run: the models download", font=font("body", 24), fill=(205, 200, 220))
    else:
        shown = [by_key[k] for k in spec]
        dim_all = n == 9
        x, y, w = 1100, 150, 700
        if len(shown) > 6:
            x, w = 1000, 820
            y = 130
        if n == 11:
            # Twelve rows: two columns.
            left, right = shown[:6], shown[6:]
            boxes = panel(frame, d, left, 980, 130, 440, t)
            panel(frame, d, right, 1450, 130, 420, t, lit_from=0.55, lit_to=0.9)
        else:
            boxes = panel(frame, d, shown, x, y, w, t, dim_all=dim_all)
        # The dashed and solid marks the script calls for.
        if n == 6 and boxes:
            for b in boxes:
                ax = b[0] + 40
                d.line([(ax, b[3]), (ax, b[3] + 10)], fill=MAGENTA, width=3)
                d.polygon([(ax - 8, b[3] + 8), (ax + 8, b[3] + 8), (ax, b[3] + 18)], fill=MAGENTA)
        if n == 7 and boxes:
            for k in (1, 3, 5):
                a, b = boxes[k - 1], boxes[k]
                dashed(d, (a[0] - 30, a[1] + 40), (b[0] - 30, b[1] + 40), CYAN, width=2)
                d.text((a[0] - 330, b[1] + 30), "no scene image →", font=font("mono", 24), fill=CYAN)
        if n == 9 and boxes:
            layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            top = boxes[0][1] - 20
            for yy in range(top, boxes[-1][3] + 20, 14):
                dashed(ld, (boxes[0][0] - 20, yy), (boxes[0][2] + 20, yy), CYAN + (60,), width=1)
            frame.paste(layer, (0, 0), layer)
            dashed(d, (boxes[0][0] - 40, boxes[0][1]), (boxes[0][0] - 40, 100), CYAN, width=2)
            d.text((boxes[0][0] - 420, 150), "← also set a Writer row", font=font("mono", 22), fill=CYAN)
        if n == 11 and boxes:
            d.line([(boxes[0][0] - 24, boxes[0][1] + 50), (boxes[0][0] - 60, boxes[0][1] + 50), (boxes[0][0] - 60, 60)], fill=MAGENTA, width=3)
            d.text((boxes[0][0] - 520, 70), "STORY BANK → my_story", font=font("mono", 24), fill=MAGENTA)
        if n == 4 and boxes:
            dashed(d, (boxes[2][0] - 24, boxes[2][1]), (boxes[2][0] - 24, boxes[2][3]), CYAN, width=2)
            d.text((boxes[2][0] - 330, boxes[2][1] + 40), "1 act: a good first try", font=font("mono", 22), fill=CYAN)
        if n == 5 and boxes:
            fan = ["roll (any eligible bank)", "original", "scifi_news_pro", "media_archive", "shakespeare", "public_domain", "my_story"]
            fx, fy = boxes[0][0] - 420, boxes[0][1]
            for i, item in enumerate(fan):
                show = t > 0.15 + i * 0.03
                if show:
                    d.text((fx, fy + i * 34), item, font=font("mono", 22), fill=MAGENTA if i in (0, 6) else PAPER)
        if n == 8 and boxes:
            d.line([(boxes[3][0] - 24, boxes[3][1]), (boxes[3][0] - 24, boxes[4][3])], fill=MAGENTA, width=4)
            d.text((boxes[3][0] - 330, boxes[3][1] + 70), "two writers, local", font=font("mono", 22), fill=MAGENTA)
        if n == 10 and boxes:
            # Lemmy: a peaked cap, tipped.
            cx, cy = 600, 520
            d.ellipse((cx - 50, cy - 30, cx + 50, cy + 70), outline=PAPER, width=3)
            tilt = 8 * math.sin(min(1, t * 2) * math.pi)
            d.arc((cx - 60, cy - 90 + tilt, cx + 60, cy - 10 + tilt), 180, 360, fill=MAGENTA, width=4)
            d.line([(cx - 64, cy - 50 + tilt), (cx + 80, cy - 50 + tilt)], fill=MAGENTA, width=4)
            d.text((cx - 60, cy + 70), "LEMMY", font=font("title", 28), fill=PAPER)
    # The slow push-in the script's camera makes: crop a little and resize;
    # the title and the caption then sit on the frame itself.
    z = 1.0 + 0.045 * ease(t)
    cw, ch = int(W / z), int(H / z)
    left = (W - cw) // 2 + int((t - 0.5) * 20)
    top = (H - ch) // 2
    frame = frame.crop((left, top, left + cw, top + ch)).resize((W, H), Image.BILINEAR)
    d = ImageDraw.Draw(frame)
    title(d, n, scene["title"])
    caption(d, frame, scene["ost"])
    return frame


def end_card_frame(t: float, sources: list) -> Image.Image:
    frame = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(frame)
    if t > 0.25:
        d.text((120, 160), "SOURCES", font=font("mono", 34), fill=CYAN)
        scroll = int(40 * ease((t - 0.25) / 0.75))
        for i, line in enumerate(sources):
            d.text((120, 240 + i * 52 - scroll), "• " + line, font=font("mono", 26), fill=CYAN)
    return frame


# --------------------------------------------------------------------------- #
# narration
# --------------------------------------------------------------------------- #

def narrate(scenes: list, cache: Path, voice_id: str) -> list:
    """One wav per scene, from the local Kokoro model; cached by text."""
    import hashlib
    import soundfile as sf
    import torch
    os.environ.setdefault("HF_HOME", r"C:\ComfyUI-Models\huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from kokoro import KPipeline
    pipe = None
    voice = torch.load(str(VOICES / f"{voice_id}.pt"), weights_only=True)
    out = []
    for s in scenes:
        digest = hashlib.sha256((voice_id + "|" + s["narration"]).encode("utf-8")).hexdigest()[:16]
        path = cache / f"narr_{s['n']:02d}_{digest}.wav"
        if not path.exists():
            if pipe is None:
                pipe = KPipeline(lang_code=voice_id[0] if voice_id[0] in "ab" else "b", device="cpu",
                                 repo_id="hexgrad/Kokoro-82M")
            chunks = [a for _, _, a in pipe(s["narration"], voice=voice, speed=1.0)]
            audio = np.concatenate([np.asarray(c.numpy() if hasattr(c, "numpy") else c) for c in chunks])
            sf.write(str(path), audio, SR)
        data, rate = sf.read(str(path), dtype="float32")
        assert rate == SR
        out.append(data)
        print("narration %02d: %.1f s" % (s["n"], len(data) / SR))
    return out


def tick(seconds: float = 0.05) -> np.ndarray:
    n = int(SR * seconds)
    noise = np.random.default_rng(7).standard_normal(n).astype(np.float32)
    env = np.exp(-np.linspace(0, 9, n)).astype(np.float32)
    return noise * env * 0.08


# --------------------------------------------------------------------------- #
# the cut
# --------------------------------------------------------------------------- #

def timeline(scenes: list, narrations: list, lead=0.5, tail=0.9, end_card=6.0) -> list:
    """Each scene's start and length in seconds."""
    plan, at = [], 0.0
    for s, a in zip(scenes, narrations):
        dur = lead + len(a) / SR + tail
        plan.append({"scene": s, "start": at, "dur": dur, "narration": a, "lead": lead})
        at += dur
    plan.append({"scene": None, "start": at, "dur": end_card, "narration": None, "lead": 0})
    return plan


def mix_audio(plan: list) -> np.ndarray:
    total = int(SR * (plan[-1]["start"] + plan[-1]["dur"])) + SR
    mix = np.zeros(total, dtype=np.float32)
    click = tick()
    for p in plan:
        start = int(p["start"] * SR)
        if p["narration"] is not None:
            mix[start:start + len(click)] += click
            a = p["narration"]
            at = start + int(p["lead"] * SR)
            mix[at:at + len(a)] += a
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.9).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="the mp4 to write")
    ap.add_argument("--stills", help="write one frame per scene into this folder and stop")
    ap.add_argument("--voice", default="bm_george", help="a Kokoro voice id from the local voices folder")
    ap.add_argument("--cache", default=str(REPO / "tmp" / "explainer_cache"))
    args = ap.parse_args(argv)
    scenes, sources = read_script()
    rows, by_key = read_form()
    if args.stills:
        out = Path(args.stills)
        out.mkdir(parents=True, exist_ok=True)
        for s in scenes:
            for t in (0.35, 0.85):
                scene_frame(s, t, rows, by_key, sources).save(out / f"scene_{s['n']:02d}_{int(t * 100):02d}.png")
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    i = 0
    for p in plan:
        n = int(p["dur"] * FPS)
        for k in range(n):
            t = k / max(1, n - 1)
            frame = scene_frame(p["scene"], t, rows, by_key, sources) if p["scene"] else end_card_frame(t, sources)
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
