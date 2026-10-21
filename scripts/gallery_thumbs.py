"""Per-graph gallery cards: the operator's art, lettered with what the graph is.

ComfyUI's template gallery shows one square JPEG per shipped graph
(``workflows/<stem>.jpg``). Until 2026-10-14 every graph carried the same
picture, so 27 cards looked alike and a Mac or AMD owner had to read file
names to find theirs. Each card now keeps the master art
(``assets/otr_gallery_thumb_master.webp``) and letters its middle in big type:
the HARDWARE the graph is built for, then the video LANE it renders with --
"NVIDIA 8 GB" over "LTX VIDEO", "MAC 16 GB" over "ANIMATEDIFF".

THE WORDS COME FROM THE MATRIX. ``build_variants`` hands each card the graph's
profile (``config/workflow_matrix.json`` through ``load_workflow_row``): the
hardware from the row's ``gpu_vendor`` and the size in its id, the lane from
its ``character_visual`` engine. There is no second list of graph names. A
shipping row whose vendor or engine has no words here raises, and ``--check``
reports it, so a new lane cannot ship a card that says nothing.

THE CHECK READS THE LABEL, NOT THE PIXELS. Every card carries a JPEG comment
naming its words, a hash of the master art and a hash of the look (layout
version, palettes, card size, font). ``--check`` compares that comment with the
one derived now, and requires the file to be a whole JPEG, but never redraws: two
machines with different Pillow builds encode the same picture to different
bytes, so a byte check would fail on whichever box did not draw it. ``--all``
redraws a card only when its comment is stale.

Dev-only: scripts/ and assets/ (the art and the bundled Inter Display font,
SIL OFL 1.1, assets/fonts/OFL.txt) do not ship; the finished cards do.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MASTER_ART = REPO / "assets" / "otr_gallery_thumb_master.webp"
FONT = REPO / "assets" / "fonts" / "InterDisplay-Black.otf"

#: Bump when the drawing in render() changes (positions, sizes), so --all
#: redraws every card and --check flags any it did not. The palettes, the card
#: size and the font are hashed into the comment on their own (_look_hash).
LAYOUT = "v2"
CARD_SIZE = 400

#: The video lane each character_visual engine renders, in the gallery's words.
LANES = {
    "viz_camera": "VISUALIZER",
    "still_motion": "STILL + MOTION",
    "still_flat": "STILLS",
    "ltx_8gb": "LTX VIDEO",
    "ltx098_low_video": "LTX VIDEO",
    "ltx25_high_video": "LTX VIDEO",
    "ltx25_foley_16gb": "LTX FOLEY",
    "ltx25_foley_24gb": "LTX FOLEY",
    "cloud_ltx25_foley_plus": "LTX FOLEY",
    "ltx25_mime_16gb": "LTX MIME",
    "ltx25_audio_in_16gb": "LTX AUDIO-IN",
    "ltx25_low_video": "LTX VIDEO",
    "ltx25_foley_8gb": "LTX FOLEY",
    "ltx25_mime_8gb": "LTX MIME",
    "ltx25_audio_in_8gb": "LTX AUDIO-IN",
    "cloud_ltx25_audio_in": "LTX AUDIO-IN",
    "animatediff15_v3_haunted_video": "ANIMATEDIFF",
    "animatediff15_lightning_video": "ANIMATEDIFF",
    "cloud_vidu_q2_pro_fast_720p": "VIDU VIDEO",
}

#: Each hardware class gets its own look, so a Mac or AMD owner finds their
#: card at a glance: (field top, field bottom, hardware text, lane text, rule).
PALETTES = {
    "nvidia": ((118, 185, 0), (24, 64, 0), (255, 255, 255), (234, 255, 196), (196, 255, 92)),
    "apple": ((246, 246, 250), (176, 178, 190), (22, 22, 26), (58, 58, 66), (110, 110, 122)),
    "amd": ((237, 28, 36), (92, 0, 12), (255, 255, 255), (255, 214, 214), (255, 120, 120)),
    "cloud": ((56, 152, 255), (10, 32, 96), (255, 255, 255), (214, 234, 255), (140, 200, 255)),
    "google": ((255, 214, 64), (234, 134, 0), (32, 33, 36), (60, 48, 20), (120, 80, 0)),
    "any": ((150, 64, 255), (64, 0, 104), (255, 255, 255), (244, 218, 255), (255, 120, 220)),
}


@dataclass(frozen=True)
class Card:
    brand: str      # the biggest word: NVIDIA, MAC, AMD, CLOUD, GOOGLE, ANY
    tier: str       # under it: 8 GB, 16 GB, RADEON, COMFY CREDITS ...
    lane: str       # the video lane: LTX VIDEO, ANIMATEDIFF ...
    extra: str      # only where two cards would otherwise match: acts, deluxe
    palette: str    # a key of PALETTES


#: The two graphs that are not matrix rows.
CANONICAL_CARD = Card("ANY", "MACHINE", "AUTO-DETECT", "", "any")
APP_CARD = Card("ANY", "MACHINE", "APP VIEW", "", "any")


def card_for_profile(profile: dict) -> Card:
    """The card for one shipping matrix row. Raises ValueError when the row's
    vendor or video engine has no words here."""
    pid = str(profile["id"])
    vendor = str(profile.get("gpu_vendor") or "none")
    engine = str((profile.get("role_overrides") or {}).get("character_visual"))
    lane = LANES.get(engine)
    if lane is None:
        raise ValueError("%s: video engine %r has no gallery words in "
                         "scripts/gallery_thumbs.LANES" % (pid, engine))
    size = re.search(r"(\d+)gb", pid)
    extra = ""
    if vendor == "nvidia" and size:
        gb = int(size.group(1))
        brand, tier = "NVIDIA", "%d GB%s" % (gb, "+" if gb >= 24 else "")
    elif vendor == "apple":
        mac = re.search(r"mac(\d+)", pid)
        brand, tier = "MAC", ("%s GB" % mac.group(1)) if mac else "APPLE SILICON"
    elif vendor == "amd":
        brand, tier = "AMD", "RADEON"
    elif vendor == "none" and "cloud" in pid:
        brand, tier = "CLOUD", "COMFY CREDITS"
        acts = str((profile.get("features") or {}).get("act_count") or "")
        if acts:
            extra = "%s ACT%s" % (acts, "" if acts == "1" else "S")
        if "deluxe" in pid:
            extra = ("DELUXE, " + extra) if extra else "DELUXE"
        vendor = "cloud"
    elif vendor == "none" and "google" in pid:
        brand, tier, vendor = "GOOGLE", "GEMINI API", "google"
    elif vendor == "nvidia":
        raise ValueError("%s: an NVIDIA row needs its VRAM size in the id "
                         "(otr_<N>gb_...) for its gallery card" % pid)
    else:
        raise ValueError("%s: vendor %r has no gallery words in "
                         "scripts/gallery_thumbs.card_for_profile" % (pid, vendor))
    if (profile.get("role_overrides") or {}).get("character_image") == "flux3_klein_ref":
        # The faces come from reference pictures: the one thing that tells this
        # card apart from the same lane's plain graph.
        extra = ("FLUX 3 REF, " + extra) if extra else "FLUX 3 REF"
    return Card(brand, tier, lane, extra, vendor)


def _look_hash() -> str:
    """Everything that changes the picture without changing the words."""
    h = hashlib.sha256()
    h.update(("%s|%d|%r" % (LAYOUT, CARD_SIZE, sorted(PALETTES.items()))).encode())
    h.update(FONT.read_bytes())
    return h.hexdigest()[:12]


def comment_for(card: Card, master: bytes) -> str:
    """What a card's JPEG comment must say: its words, the art and the look."""
    art = hashlib.sha256(master).hexdigest()[:12]
    return "OTR gallery card %s: %s | %s | %s | %s | art %s | look %s" % (
        LAYOUT, card.brand, card.tier, card.lane, card.extra, art, _look_hash())


def is_whole_jpeg(data: bytes) -> bool:
    """Starts with SOI and ends with EOI: a truncated card is a blank card in
    the gallery even when its header still carries the right comment."""
    return len(data) > 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def jpeg_comment(data: bytes):
    """The first COM segment of a JPEG, or None (also for a non-JPEG)."""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:          # start of scan: no headers after this
            return None
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if marker == 0xFE:
            return data[i + 4:i + 2 + length].decode("utf-8", "replace")
        i += 2 + length
    return None


def _fit(draw, text, font_path, max_w, max_size):
    from PIL import ImageFont
    size = max_size
    while size > 8:
        font = ImageFont.truetype(str(font_path), size)
        left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= max_w:
            return font
        size -= 2
    return ImageFont.truetype(str(font_path), size)


def _logo(art):
    """The OLD TIME RADIO lettering cut out of the master art: its dark
    ground made transparent, its two side captions left behind."""
    from PIL import Image
    w, h = art.size
    logo = art.crop((round(w * 0.129), round(h * 0.040),
                     round(w * 0.950), round(h * 0.236)))
    alpha = logo.convert("L").point(lambda v: 0 if v < 85 else min(255, (v - 85) * 5))
    # The right-hand caption column (WORDS, VOICES ...) sits below the lettering.
    lw, lh = logo.size
    cut = Image.new("L", (lw - round(w * 0.760), lh - round(h * 0.145)), 0)
    alpha.paste(cut, (round(w * 0.760), round(h * 0.145)))
    logo.putalpha(alpha)
    return logo


def _gradient(size, top, bottom):
    from PIL import Image
    mask = Image.linear_gradient("L").resize(size)
    return Image.composite(Image.new("RGB", size, bottom),
                           Image.new("RGB", size, top), mask)


def render(card: Card, master: bytes, comment: str) -> bytes:
    """Draw one card: a small OLD TIME RADIO banner over a field in the
    hardware class's colours, lettered big with what the graph is."""
    from PIL import Image, ImageDraw

    art = Image.open(io.BytesIO(master)).convert("RGB")
    w, h = art.size
    top, bottom, brand_c, lane_c, rule_c = PALETTES[card.palette]
    band = round(h * 0.25)
    out = Image.new("RGB", (w, h), (7, 7, 13))
    out.paste(_gradient((w, h - band), top, bottom), (0, band))
    logo = _logo(art)
    scale = min(w * 0.86 / logo.width, band * 0.80 / logo.height)
    logo = logo.resize((round(logo.width * scale), round(logo.height * scale)),
                       Image.LANCZOS)
    out.paste(logo, ((w - logo.width) // 2, (band - logo.height) // 2), logo)
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, band - round(h * 0.006), w, band + round(h * 0.006)), fill=rule_c)

    max_w = w * 0.86
    lines = [(card.brand, brand_c, round(h * 0.21)),
             (card.tier, brand_c, round(h * 0.12)),
             (card.lane, lane_c, round(h * 0.10))]
    if card.extra:
        lines.append((card.extra, lane_c, round(h * 0.07)))
    fitted = [(text, colour, _fit(draw, text, FONT, max_w, size))
              for text, colour, size in lines]
    gap = round(h * 0.03)
    boxes = [draw.textbbox((0, 0), text, font=font) for text, _c, font in fitted]
    heights = [b[3] - b[1] for b in boxes]
    y = (band + h) / 2 - (sum(heights) + gap * (len(fitted) - 1)) / 2
    for i, ((text, colour, font), b, height) in enumerate(zip(fitted, boxes, heights)):
        x = (w - (b[2] - b[0])) / 2 - b[0]
        draw.text((x, y - b[1]), text, font=font, fill=colour)
        y += height + gap
        if i == 1:      # a rule centred in the gap between hardware and lane
            mid, half = y - gap / 2, h * 0.002
            draw.rectangle((w * 0.30, mid - half, w * 0.70, mid + half), fill=rule_c)

    small = out.resize((CARD_SIZE, CARD_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=90, comment=comment.encode("utf-8"))
    return buf.getvalue()
