"""OTR_CreditsRoll -- the ONE late viewer-credits surface, rendered as the
operator's 3-column SIGNAL LOST console (credits enrichment 2026-07-03, redesign
per docs/2026-07-03-credits-enrichment/CREDITS_OVERLAY_BUILD_PLAN.md).

WIRED 93 -> OTR_CreditsRoll (node 95) -> 85. Renders the COMPLETE viewer credits
LATE, from post-render truth: the DURABLE production-ledger singleton (S2 stamps)
+ the clip manifest (node 92 slot 1). Appended as a SILENT tail to node 93's
output; the video ENDS at credit-end.

Presentation model (operator lock 2026-07-03):
- COLUMNS 1 + 2 are a STATIC dashboard (title / MODELS / [PRODUCTION LEDGER] /
  [SYSTEM]  |  CAST & VOICES / [WRITER-LLM-CONFIG]) -- held the whole time.
- COLUMN 3 SCROLLS the full narrative (STORY SPINE -> full CLASSIFIED TRANSCRIPT
  -> SOURCE INTERCEPT -> DIAGNOSTIC). NOTHING is dropped; the whole script rolls.
  The credits-tail DURATION follows the col-3 scroll length and is DECLARED to the
  credits-aware mux (node 85) so its guard permits it.

Hard contracts:
- NO FALLBACKS. A missing RECEIPT (title/style/cast/engines/seed/commit) RAISES
  CreditsDataError before mux. Only probe fields ([SYSTEM]/VRAM) and pure in-world
  flavor text may be soft; frozen story facts (spine/news) are omitted-if-absent,
  never a quiet "(not recorded)".
- LOOK CONTRACT (operator 2026-06-17): the backdrop under the console is the LAST
  drama clip LOOPED (plan_backdrop), darkened; the radial console panel is
  composited at partial alpha so the clip ghosts through (never credits-over-black).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

log = logging.getLogger("OldTimeRadio")

# --------------------------------------------------------------------------- #
# Palette (from the operator's Credits Overlay design). RGB.
# --------------------------------------------------------------------------- #
_TEAL = (94, 234, 212)          # #5eead4 -- hero / headers
_GREEN = (74, 222, 128)         # #4ade80 -- base / labels
_BRIGHT = (187, 247, 208)       # #bbf7d0 -- bright values / body
_SUB = (167, 243, 208)          # #a7f3d0 -- sub-headers
_BG_A = (36, 16, 23)            # #241017 radial center (warm)
_BG_B = (10, 13, 10)            # #0a0d0a
_BG_C = (5, 7, 5)               # #050705 outer
_NEON = [(255, 45, 107), (192, 38, 211), (37, 99, 235), (14, 165, 233)]

# alpha tiers
_A_LABEL = 140
_A_TAG = 140
_A_MICRO = 110
_A_FOOTER = 120
_A_RULE = 51

# --------------------------------------------------------------------------- #
# Geometry (1920x1080 reference; scaled by h/1080 for other sizes).
# --------------------------------------------------------------------------- #
_REF_H = 1080
_MARGIN_L = 56
_MARGIN_TOP = 48
_COL1_X, _COL1_W = 56, 598
_COL2_X, _COL2_W = 742, 424
_COL3_X, _COL3_W = 1256, 608
_RULE1_X, _RULE2_X = 698, 1210
_COL3_VIEW_Y = 128
_COL3_VIEW_H = 908

# scroll / hold (seconds)
_SCROLL_PPS = 60.0
_LEAD_HOLD_S = 3.0
_TAIL_HOLD_S = 4.0
_MAX_HOLD_S = 120.0             # ceiling -> speed up, never truncate
_FADE_IN_S = 0.6
_FADE_OUT_S = 0.8

# font point sizes @1080 (tiers). Auto-scaled by h.
_PT_HERO_MAX = 76
_PT_HERO_MIN = 48
_PT_H1 = 28
_PT_H2 = 22
_PT_SUBHEAD = 18
_PT_NAME = 22
_PT_SPEAKER = 19
_PT_BODY = 20
_PT_GRID = 16
_PT_GRIDL = 18
_PT_TAG = 15
_PT_MICRO = 14
_PT_FOOTER = 15


class CreditsDataError(RuntimeError):
    """A required credits receipt is missing / unrenderable. LOUD by design --
    the no-fallback contract forbids a quiet placeholder."""


_STORY_STYLE_STATUS_SCAFFOLD_OFF = "story_scaffold_off"


def _require(container, key, source):
    v = (container or {}).get(key)
    if v is None or v == "" or v == {} or v == []:
        raise CreditsDataError(
            f"required credits receipt missing: {source}.{key} -- the durable "
            f"stamp for it never landed (no-fallback: refusing a placeholder)")
    return v


def _story_style_receipt(meta: dict) -> str:
    """Return the durable story-style receipt for credits.

    Normal runs require ``meta.style``. A story-scaffold-off run has no story
    contract by design, so the writer stamps a separate explicit status. That
    status is accepted only when paired with ``story_scaffold_enabled is False``;
    a plain missing ``meta.style`` remains a hard data error.
    """
    style = (meta or {}).get("style")
    if style is not None and style != "":
        return str(style).strip()
    if (
        (meta or {}).get("story_scaffold_enabled") is False
        and (meta or {}).get("story_style_status")
        == _STORY_STYLE_STATUS_SCAFFOLD_OFF
    ):
        return _STORY_STYLE_STATUS_SCAFFOLD_OFF
    return str(_require(meta, "style", "meta")).strip()


# --------------------------------------------------------------------------- #
# Declared sources (date / GPU / git -- each has a declared source, no ad-hoc)
# --------------------------------------------------------------------------- #
def _git_short_sha() -> str:
    """Repo-file read (.git/HEAD -> ref). Raises when unresolvable (the
    COMMIT line is a receipt)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head_path = os.path.join(repo, ".git", "HEAD")
    try:
        with open(head_path, "r", encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1].strip()
            ref_path = os.path.join(repo, ".git", *ref.split("/"))
            if os.path.exists(ref_path):
                with open(ref_path, "r", encoding="utf-8") as f:
                    return f.read().strip()[:8]
            packed = os.path.join(repo, ".git", "packed-refs")
            with open(packed, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(ref) and not line.startswith("#"):
                        return line.split()[0][:8]
            raise FileNotFoundError(ref)
        return head[:8]
    except Exception as exc:  # noqa: BLE001
        raise CreditsDataError(
            f"git short SHA unresolvable from {head_path!r}: {exc}") from exc


def _sys_specs() -> dict:
    try:
        from ._otr_sys_specs import collect_system_specs
        return collect_system_specs() or {}
    except Exception:  # noqa: BLE001 -- probe block, degrades to "(unknown)"
        return {}


# =========================================================================== #
# DATA -> LAYOUT (the durable-receipt reader). Every block is (label, value,
# tier) tuples the renderers two-tone. No-fallback on receipts.
# =========================================================================== #
_FAMILY_PRETTY = {
    "audio_driven_face": "audio-driven face",
    "text_to_video": "text-to-video",
    "image_to_video": "image-to-video",
    "audio_conditioned_video": "audio-in video",
    "static_image_gen": "still",
}


def _fam(family) -> str:
    f = str(family or "").strip()
    return _FAMILY_PRETTY.get(f, f.replace("_", " ")) if f else ""


def _hist_rows(by_role: dict) -> list:
    """(role, 'eng1, eng2 x2') rows in role order."""
    rows = []
    for role in sorted(by_role or {}):
        engs = by_role[role] or {}
        val = ", ".join("%s%s" % (e, "" if n == 1 else " x%d" % n)
                        for e, n in sorted(engs.items()))
        rows.append((role, val))
    return rows


def _video_role_rows(render_engines: dict) -> list:
    """(role, engine, family-suffix) for the MODELS.VIDEO block. The family
    suffix comes from by_engine[eng]['family'] (S-B stamp).

    When the roll-up lists ``family`` in its ``varied`` the engine has NO
    single family across its clips (2026-07-28), and the row says "mixed". A
    blank suffix there would read as "this engine has no family", which is a
    different statement and an untrue one."""
    by_role = render_engines.get("by_role") or {}
    by_engine = render_engines.get("by_engine") or {}
    rows = []
    for role in sorted(by_role):
        engs = by_role[role] or {}
        eng = sorted(engs)[0] if engs else "?"
        rollup = by_engine.get(eng) or {}
        fam = ("mixed" if "family" in (rollup.get("varied") or [])
               else _fam(rollup.get("family")))
        rows.append((role, eng, fam))
    return rows


def _recipe_suffix(render_engines: dict, eng: str) -> str:
    """The recipe line a MODELS.VIDEO row carries for one engine.

    Reads the PER-FIELD by_engine roll-up: a field the engine did NOT hold to
    one value across its clips is absent from the row and named in ``varied``,
    so the card reports ``mixed <fields>`` rather than dropping the fact
    silently or printing one clip's value as though it were the engine's.
    Pre-2026-07-28 ledgers carry no ``varied`` key and read exactly as before."""
    r = (render_engines.get("by_engine") or {}).get(eng) or {}
    bits = []
    if r.get("recipe"):
        bits.append(str(r["recipe"]))
    if r.get("quant"):
        bits.append(str(r["quant"]))
    if r.get("use_lora"):
        bits.append("lora")
    if r.get("render_canvas"):
        bits.append(str(r["render_canvas"]))
    # family has its OWN slot on the row (_video_role_rows), so naming it here
    # too would print the same fact twice; this line reports only its own.
    mixed = [f for f in (r.get("varied") or []) if f != "family"]
    if mixed:
        bits.append("mixed " + ", ".join(mixed))
    return " · ".join(bits)


def _fmt_gib(mb) -> str:
    try:
        return "%.1f GiB" % (float(mb) / 1024.0)
    except (TypeError, ValueError):
        return "(unknown)"


_DIAGNOSTICS = [
    ">> DIAGNOSTIC: CARRIER RE-ACQUIRED. {vram} PEAK CONTAINED. RESIDUAL HUM WITHIN TOLERANCE.",
    ">> DIAGNOSTIC: TRANSCRIPT SEALED. {ncast} VOICES ACCOUNTED FOR. NO UNLISTED SPEAKERS DETECTED.",
    ">> DIAGNOSTIC: SIGNAL DEGRADED GRACEFULLY. NOTHING FOLLOWED US OUT.",
    ">> DIAGNOSTIC: PLAYBACK COMPLETE. THE FREQUENCY REMEMBERS.",
    ">> DIAGNOSTIC: TAPE ENDS. THE HISS KEEPS TALKING.",
]


def build_credits_layout(led: dict, *, w: int, h: int, manifest: dict) -> dict:
    """The console layout from the DURABLE ledger (S2 stamps). Returns a dict
    with hero/subtitle + col1_blocks + col2_blocks + col3_flow. RAISES on any
    missing receipt (no-fallback)."""
    if not isinstance(led, dict) or not led:
        raise CreditsDataError("durable ledger missing/empty at credits time")
    meta = led.get("meta")
    if not isinstance(meta, dict) or not meta:
        raise CreditsDataError("durable ledger has no meta at credits time")
    manifest = manifest if isinstance(manifest, dict) else {}

    # --- hero / subtitle (title tweak: episode title is the HERO) -----------
    title = str(_require(meta, "episode_title", "meta")).strip()
    style = _story_style_receipt(meta)

    # --- COL 1 -------------------------------------------------------------
    gp = meta.get("gen_params_initial") or {}
    ren = _require(meta, "render_engines", "meta")
    img = _require(meta, "image_engines", "meta")
    music = _require(meta, "music_engine", "meta")

    col1 = []
    # MODELS
    models = {"header": "MODELS", "tag": "GENERATIVE STACK · THIS EPISODE",
              "img_rev": (img.get("image_revision")),
              "vid_rev": (ren.get("video_revision")),
              "image_rows": _hist_rows(img.get("by_role") or {}),
              "video_rows": _video_role_rows(ren),
              "video_suffix": {r[1]: _recipe_suffix(ren, r[1])
                               for r in _video_role_rows(ren)},
              "music": str(music)}
    col1.append(("models", models))

    # [ PRODUCTION LEDGER ]
    sysd = _sys_specs()
    frames = manifest.get("total_target_frames")
    fps = manifest.get("fps")
    clip_count = manifest.get("clip_count") or len(manifest.get("clips") or [])
    seed = (meta.get("cast_contract") or {}).get("cast_seed",
                                                 meta.get("episode_seed"))
    if seed is None:
        raise CreditsDataError(
            "required credits receipt missing: cast seed "
            "(meta.cast_contract.cast_seed / meta.episode_seed)")
    ledger_grid = []
    ledger_grid.append(("VRAM:", "%s of %s" % (
        _fmt_gib(ren.get("vram_peak_mb")), (sysd.get("vram") or "GPU"))))
    if frames and fps:
        ledger_grid.append(("FRAMES:", "%s @ %s fps · %s clips"
                            % (frames, fps, clip_count)))
    ledger_grid.append(("SEED:", "cast %s%s" % (
        seed, " (%s)" % gp.get("seed_source") if gp.get("seed_source") else "")))
    ledger_grid.append(("COMMIT:", _git_short_sha()))
    ledger_grid.append(("REV:", "img %s · vid %s" % (
        img.get("image_revision"), ren.get("video_revision"))))
    col1.append(("grid", {"header": "[ PRODUCTION LEDGER ]", "rows": ledger_grid}))

    # [ SYSTEM ] -- soft probe block. Operator 2026-07-03 / roundtable R1: SYSTEM
    # moves OUT of static col 1 INTO the TOP of the col-3 scroll (below), so the
    # scroll carries "all the same details as the old right panel". Built here
    # (sysd is in scope), prepended to col3_flow. Col 1 keeps title / MODELS /
    # [PRODUCTION LEDGER] / footer -- it gains room, no duplication.
    _cpu = sysd.get("cpu") or "(unknown)"
    if sysd.get("cpu_cores"):
        _cpu += " (%s)" % sysd["cpu_cores"]
    _ram = sysd.get("ram") or "(unknown)"
    if sysd.get("ram_peak"):
        _ram += " (peak %s)" % sysd["ram_peak"]
    _gpu = sysd.get("gpu") or "(unknown)"
    if sysd.get("vram"):
        _gpu += " (%s)" % sysd["vram"]
    sys_grid = [
        ("Host:", "%s · %s" % (sysd.get("hostname") or "?",
                               sysd.get("os") or "(unknown)")),
        ("CPU:", _cpu),
        ("RAM:", _ram),
        ("GPU:", _gpu),
        ("CUDA:", "%s · torch %s · Python %s" % (
            sysd.get("cuda") or "?", sysd.get("torch") or "?",
            sysd.get("python") or "?")),
    ]

    # --- COL 2 -------------------------------------------------------------
    cast = _require(led, "cast", "ledger")
    voice_slots = (meta.get("cast_voice_slots")
                   if isinstance(meta.get("cast_voice_slots"), dict) else {})
    cast_rows = []
    for entry in cast:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("char_id") or "").strip()
        ref = entry.get("voice_ref_id") or entry.get("voice_preset")
        eng = entry.get("voice_engine") or (
            "bark" if entry.get("voice_preset") else "")
        if not ref or not eng:
            raise CreditsDataError(
                f"cast entry {name or '?'} has no delivered voice stamp "
                f"(voice_ref_id/voice_preset + voice_engine)")
        sig = str(((voice_slots.get(str(entry.get("char_id") or "")) or {})
                   ).get("speech_signature") or "").strip()
        cast_rows.append({"name": name, "line": "%s · %s" % (eng, ref),
                          "sig": sig})
    if not cast_rows:
        raise CreditsDataError("cast is empty at credits time")

    creative = gp.get("creative_writing_model") or meta.get("creative_writing_model")
    technical = gp.get("technical_model") or meta.get("technical_model")
    if not creative or not technical:
        raise CreditsDataError(
            "required credits receipt missing: writer models "
            "(meta.gen_params_initial.creative_writing_model / technical_model)")
    writer_grid = [("Creative (A):", str(creative)),
                   ("Technical (B):", str(technical))]
    if meta.get("slot_transitions") is not None:
        writer_grid.append(("Slot routing:", "%s A<->B transition(s)"
                            % meta.get("slot_transitions")))
    if gp.get("creativity"):
        writer_grid.append(("Creativity:", str(gp.get("creativity"))))
    if gp.get("temperature") is not None or gp.get("top_p") is not None:
        writer_grid.append(("Temp / top_p:", "%s / %s" % (
            gp.get("temperature"), gp.get("top_p"))))
    if gp.get("target_words") is not None:
        writer_grid.append(("Words:", "target %s / actual %s (char %s / ann %s)" % (
            gp.get("target_words"), meta.get("total_word_count"),
            meta.get("character_word_count"), meta.get("announcer_word_count"))))
    col2 = {"cast_rows": cast_rows, "writer_grid": writer_grid}

    # --- COL 3 (scrolls) ---------------------------------------------------
    # Order: [ SYSTEM ] -> [ STORY SPINE ] -> full [ CLASSIFIED TRANSCRIPT ] ->
    # >> SOURCE INTERCEPT -> >> DIAGNOSTIC. SYSTEM leads (operator/R1: the scroll
    # carries all the same details as the old right panel).
    flow = []
    flow.append(("system", {"header": "[ SYSTEM ]", "rows": sys_grid}))
    # STORY SPINE
    spine = []
    news = meta.get("news") if isinstance(meta.get("news"), dict) else {}
    ds = (meta.get("dramatic_state")
          if isinstance(meta.get("dramatic_state"), dict) else {})
    # Meta split (2026-07-09): the premise line is the PRODUCED story's
    # logline (K.5.6 post-composition summary) -- the actual episode, not
    # the pre-generation source digest. Old ledgers without the field keep
    # the legacy news read. The SOURCE INTERCEPT block below stays on
    # meta["news"] deliberately: that label IS the source.
    produced = (meta.get("produced_story")
                if isinstance(meta.get("produced_story"), dict) else {})
    if produced.get("logline"):
        spine.append(("Premise:", str(produced["logline"])))
    elif news.get("script_brief"):
        spine.append(("Premise:", str(news["script_brief"])))
    if ds.get("dramatic_question"):
        spine.append(("Question:", str(ds["dramatic_question"])))
    if ds.get("character_a_wants"):
        spine.append(("A wants:", str(ds["character_a_wants"])))
    if ds.get("character_b_wants"):
        spine.append(("B wants:", str(ds["character_b_wants"])))
    if ds.get("ending_change"):
        spine.append(("Ending:", str(ds["ending_change"])))
    if spine:
        flow.append(("spine", {"header": "[ STORY SPINE ]", "rows": spine}))

    # CLASSIFIED TRANSCRIPT (full)
    dialog = []
    try:
        from . import _otr_ledger_consumers as _OTRLC
        for line in _OTRLC.iter_lines(led):
            role = line.get("speaker_role") or ""
            text = (line.get("text") or "").strip()
            if not text or role not in ("character", "announcer"):
                continue
            spk = _OTRLC.speaker_name(led, line)
            # Use the same alias-aware cast join as the display speaker. A
            # content-owned announcer line may carry the role tag
            # char_id="announcer" while the delivered cast row is keyed c01.
            cast_row = _OTRLC.cast_lookup(
                led, str(line.get("char_id") or "")
            )
            voice = (
                cast_row.get("voice_ref_id")
                or cast_row.get("voice_preset")
                or ""
            )
            dialog.append({"speaker": spk, "voice": voice, "text": text})
    except Exception as exc:  # noqa: BLE001 -- transcript is a durable-ledger read
        raise CreditsDataError(
            "transcript unreadable from durable ledger lines: %s" % exc) from exc
    flow.append(("transcript",
                 {"header": "[ CLASSIFIED TRANSCRIPT ]",
                  "tag": "EPISODE // %s · SCENE 1" % title.upper(),
                  "lines": dialog}))

    # SOURCE INTERCEPT (news, optional)
    if news.get("script_brief"):
        flow.append(("intercept",
                     {"text": ">> SOURCE INTERCEPT: %s" % news["script_brief"]}))

    # SOURCE line (kibitz r2-r4, printed-layer provenance): rendered when
    # the writer stamped meta["credits_source_line"] from the bank row's
    # defaults. Data-driven -- no bank branch here: original_radio's
    # machine-generation disclosure is unconditional for that lane because
    # its bank row always defines the default; banks without the default
    # (science et al.) are byte-identical.
    _src_line = str(meta.get("credits_source_line") or "")
    if _src_line:
        flow.append(("intercept", {"text": ">> SOURCE: %s" % _src_line}))

    # DIAGNOSTIC (seeded, no fabricated numbers)
    try:
        idx = int(seed) % len(_DIAGNOSTICS)
    except (TypeError, ValueError):
        idx = 0
    diag = _DIAGNOSTICS[idx].format(
        vram=_fmt_gib(ren.get("vram_peak_mb")), ncast=len(cast_rows))
    flow.append(("diagnostic", {"text": diag}))

    return {
        "hero": title.upper(),
        "subtitle": "SIGNAL LOST",
        "subtitle_tag": "EPISODE TREATMENT",
        "meta_strip": "%s · %dx%d · %s" % (
            style, w, h, _today()),
        "col1": col1,
        "col2": col2,
        "col3_flow": flow,
        "footer_l": "Made with OTR v2.0-alpha — 100%% generated",
        "footer_c": ">> voices from the CastLock final stamp — "
                    "delivered, not planned.",
    }


def _today() -> str:
    import time
    return time.strftime("%Y-%m-%d")


# =========================================================================== #
# PIL rendering
# =========================================================================== #
_FONT_CACHE: dict = {}


def _load_font(pt: int):
    """Resolve a monospace truetype at ``pt`` via absolute font paths (bare
    names do not resolve reliably). Cached. RAISES if nothing resolves --
    no-fallback: a point-size-less bitmap hero is unacceptable (Fable risk #3)."""
    from PIL import ImageFont
    key = int(pt)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    fd = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for path in (
        os.path.join(fd, "JetBrainsMono-Bold.ttf"),
        os.path.join(fd, "consola.ttf"),
        os.path.join(fd, "cour.ttf"),
        os.path.join(fd, "lucon.ttf"),
        "JetBrainsMono-Bold.ttf", "consola.ttf", "cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "DejaVuSansMono.ttf",
    ):
        try:
            f = ImageFont.truetype(path, key)
            _FONT_CACHE[key] = f
            return f
        except Exception:  # noqa: BLE001
            continue
    raise CreditsDataError(
        "no monospace truetype font resolved -- refusing to render a "
        "point-size-less bitmap hero (no-fallback)")


def _fw(draw, s, font) -> int:
    try:
        return int(draw.textlength(s, font=font))
    except Exception:  # noqa: BLE001
        return len(s) * font.size


def _fh(font) -> int:
    return int(font.size * 1.28)


def _sc(pt: int, h: int) -> int:
    return max(8, int(round(pt * h / _REF_H)))


def _rgba(rgb, a=255):
    return (rgb[0], rgb[1], rgb[2], a)


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if _fw(draw, t, font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines or [""]


def _autoshrink_pt(draw, text, max_w, hi, lo, mkfont):
    pt = hi
    while pt > lo and _fw(draw, text, mkfont(pt)) > max_w:
        pt -= 2
    return max(lo, pt)


def _draw_grid(draw, x, y, header, rows, h, *, label_w=None):
    """Bracket header + dim-label/bright-value rows. Returns new y."""
    from PIL import ImageFont  # noqa: F401
    fh = _load_font(_sc(_PT_H2, h))
    draw.text((x, y), header, fill=_rgba(_TEAL), font=fh)
    y += _fh(fh) + _sc(6, h)
    fg = _load_font(_sc(_PT_GRID, h))
    lw = label_w if label_w is not None else _sc(96, h)
    for label, value in rows:
        draw.text((x, y), label, fill=_rgba(_GREEN, _A_LABEL), font=fg)
        for i, ln in enumerate(_wrap(draw, str(value), fg, _COL_W_FOR(x, h) - lw)):
            draw.text((x + lw, y + i * _fh(fg)), ln, fill=_rgba(_BRIGHT), font=fg)
        y += _fh(fg) * max(1, len(_wrap(draw, str(value), fg,
                                        _COL_W_FOR(x, h) - lw))) + _sc(4, h)
    return y + _sc(14, h)


def _COL_W_FOR(x, h):
    sx = h / _REF_H
    if x < _RULE1_X * sx:
        return int(_COL1_W * sx)
    if x < _RULE2_X * sx:
        return int(_COL2_W * sx)
    return int(_COL3_W * sx)


def render_static_base(layout: dict, w: int, h: int):
    """The full static console frame (bg + neon + cols 1-2 + footers +
    scanlines). Column 3 is left as background; the scroll canvas overlays it.
    Returns a PIL RGBA image (w x h)."""
    from PIL import Image, ImageDraw, ImageFilter
    sx = h / _REF_H

    # --- radial background ---
    base = _radial_bg(w, h)

    # --- neon accent bar ---
    bar = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bx, bw = int(1832 * sx), int(30 * sx)
    by, bh = int(54 * sx), int(520 * sx)
    for i in range(bh):
        t = i / max(1, bh - 1)
        col = _neon_lerp(t)
        bd.line([(bx, by + i), (bx + bw, by + i)], fill=col + (200,))
    bar = bar.filter(ImageFilter.GaussianBlur(int(26 * sx)))
    base = Image.alpha_composite(base, Image.blend(
        Image.new("RGBA", (w, h), (0, 0, 0, 0)), bar, 0.5))

    # --- text layer (cols 1-2 + footers) ---
    txt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)

    x1, top = int(_COL1_X * sx), int(_MARGIN_TOP * sx)
    y = _draw_col1(d, x1, top, layout, w, h)
    # col1 footer (bottom)
    ff = _load_font(_sc(_PT_FOOTER, h))
    d.text((x1, h - int(56 * sx)), layout["footer_l"] % (), fill=_rgba(_TEAL, 200),
           font=ff)

    x2 = int(_COL2_X * sx)
    _draw_col2(d, x2, top, layout, w, h)
    d.text((x2, h - int(56 * sx)), layout["footer_c"], fill=_rgba(_GREEN, _A_FOOTER),
           font=ff)

    # column rules
    for rx in (_RULE1_X, _RULE2_X):
        d.line([(int(rx * sx), top), (int(rx * sx), h - int(40 * sx))],
               fill=_rgba(_GREEN, _A_RULE), width=1)

    base = _compose_glow(base, txt)
    base = _scanlines(base)
    return base


def _draw_col1(d, x, top, layout, w, h):
    """Column 1, with the recipe notes sized to what the column can AFFORD.

    Col1 flows its blocks downward and has no backstop -- whatever it draws
    past ``h - 56*sx`` lands in the footer band or off the canvas entirely,
    and PIL clips it silently. The per-engine recipe note (2026-07-28) is the
    only OPTIONAL content here, so it is measured first: draw the whole column
    onto a scratch canvas at the largest note allowance, and step down until
    the column ends above the footer. At 1280x720 -- the size this repo's own
    render tests use -- the full-length allowance overruns by 27px, which is
    exactly the regression this pass exists to not ship.

    It does NOT rescue a column that overflows on its REQUIRED content alone
    (h=480 already did before any of this); at that point the allowance is
    zero and the pre-existing overflow is unchanged, not hidden."""
    floor_y = h - int(56 * (h / _REF_H))
    allowance = _NOTE_LINES_MAX
    while allowance > 0:
        if _flow_col1(_scratch_draw(w, h), x, top, layout, w, h,
                      allowance) <= floor_y:
            break
        allowance -= 1
    return _flow_col1(d, x, top, layout, w, h, allowance)


def _scratch_draw(w, h):
    """A throwaway ImageDraw for MEASUREMENT passes. Real fonts and real
    textlength, nothing kept."""
    from PIL import Image, ImageDraw
    return ImageDraw.Draw(Image.new("RGBA", (max(1, w), max(1, h)),
                                    (0, 0, 0, 0)))


def _flow_col1(d, x, top, layout, w, h, note_lines):
    sx = h / _REF_H
    y = top
    # hero (auto-shrink to col1 width)
    hero = layout["hero"]
    hero_pt = _autoshrink_pt(d, hero, int(_COL1_W * sx),
                             _sc(_PT_HERO_MAX, h), _sc(_PT_HERO_MIN, h),
                             lambda p: _load_font(p))
    fh = _load_font(hero_pt)
    d.text((x, y), hero, fill=_rgba(_TEAL), font=fh)
    y += _fh(fh)
    # subtitle "SIGNAL LOST" at 50%
    fs = _load_font(max(8, hero_pt // 2))
    d.text((x, y), layout["subtitle"], fill=_rgba(_TEAL), font=fs)
    ft = _load_font(_sc(_PT_TAG, h))
    d.text((x + _fw(d, layout["subtitle"], fs) + _sc(16, h),
            y + (_fh(fs) - _fh(ft))), layout["subtitle_tag"],
           fill=_rgba(_GREEN, _A_TAG), font=ft)
    y += _fh(fs) + _sc(6, h)
    ffine = _load_font(_sc(_PT_FOOTER, h))
    d.text((x, y), layout["meta_strip"], fill=_rgba(_BRIGHT, 150), font=ffine)
    y += _fh(ffine) + _sc(24, h)

    for kind, block in layout["col1"]:
        if kind == "models":
            y = _draw_models(d, x, y, block, w, h, note_lines=note_lines)
        elif kind == "grid":
            y = _draw_grid(d, x, y, block["header"], block["rows"], h)
    return y


# How many wrapped lines a per-engine recipe note may occupy at most. The
# column's own measurement pass (_draw_col1) spends DOWN from here; two lines
# holds a full LANE 2 receipt with its departure list at 1080.
_NOTE_LINES_MAX = 2


def _draw_models(d, x, y, m, w, h, note_lines=_NOTE_LINES_MAX):
    sx = h / _REF_H
    fh1 = _load_font(_sc(_PT_H1, h))
    d.text((x, y), m["header"], fill=_rgba(_TEAL), font=fh1)
    y += _fh(fh1)
    ftag = _load_font(_sc(_PT_TAG, h))
    d.text((x, y), m["tag"], fill=_rgba(_GREEN, _A_TAG), font=ftag)
    y += _fh(ftag) + _sc(14, h)
    fsub = _load_font(_sc(_PT_SUBHEAD, h))
    fbody = _load_font(_sc(_PT_BODY, h))
    fmicro = _load_font(_sc(_PT_MICRO, h))
    colw = int(_COL1_W * sx)

    gap = _sc(12, h)

    def _fit(s, font, max_w):
        """Trim ``s`` to ``max_w`` pixels, ending in a visible "...".

        Binary search rather than a per-character walk: the credits card is
        drawn per frame, and a ~90-character LANE 2 receipt would otherwise
        cost ~90 textlength calls a row. Returns "" only when even the ellipsis
        does not fit -- above that the cut is always MARKED, because a value
        silently shortened to fit reads as the whole value."""
        if max_w <= 0:
            return ""
        if _fw(d, s, font) <= max_w:
            return s
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _fw(d, s[:mid] + "...", font) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return (s[:lo] + "...") if _fw(d, "...", font) <= max_w else ""

    def _row(label, value, suffix=""):
        d.text((x, cur_y[0]), label, fill=_rgba(_GREEN, _A_LABEL), font=fbody)
        # CLAMP (2026-07-28, the LANE 2 fan-out rider). This right-aligned
        # block used to start at ``x + colw - width`` with no bound, so a value
        # wider than the column began LEFT of the label and drew straight over
        # it -- and a LANE 2 recipe receipt is ~90 characters against a layout
        # sized for short engine ids. The annotation gives way first: an engine
        # id that cannot be read is worse than a missing family suffix.
        # The whole block is measured in fbody even though the suffix DRAWS in
        # fmicro, which is the convention this row already used. It
        # over-estimates the suffix, so the clamp errs toward more space, never
        # less; correcting the measurement would move every existing row.
        avail = colw - _fw(d, label, fbody) - gap
        if _fw(d, value + (("  " + suffix) if suffix else ""), fbody) > avail:
            if suffix:
                suffix = _fit(suffix, fbody,
                              avail - _fw(d, value + "  ", fbody))
            if not suffix:
                value = _fit(value, fbody, avail)
        vx = x + colw - _fw(d, value + (("  " + suffix) if suffix else ""), fbody)
        d.text((vx, cur_y[0]), value, fill=_rgba(_BRIGHT), font=fbody)
        if suffix:
            d.text((vx + _fw(d, value + "  ", fbody), cur_y[0] + _sc(3, h)),
                   suffix, fill=_rgba(_GREEN, _A_MICRO), font=fmicro)
        cur_y[0] += _fh(fbody) + _sc(3, h)

    def _micro_note(text, max_lines=note_lines):
        """A wrapped continuation note under a row, in micro type.

        The recipe receipt does NOT go in ``_row``'s suffix slot: it is ~90
        characters since LANE 2, so the clamp above would cut it to nothing
        and the row would say less than saying nothing. It gets its own
        indented lines, wrapped to the column.

        ``max_lines`` comes from the COLUMN, which measured what it can afford
        (see _draw_col1) -- col1 flows downward with no backstop, and a fixed
        two-line allowance overruns the footer at 720p. Overflow is folded into
        the last line and ellipsized rather than dropped, because a silently
        truncated recipe reads as the whole recipe; at an allowance of zero the
        note is absent entirely, which is honest -- there is no room for it."""
        if not text or max_lines <= 0:
            return
        ix = x + _sc(10, h)
        room = colw - (ix - x)
        lines = _wrap(d, text, fmicro, room)
        if not lines:
            return
        shown = lines[:max_lines]
        if len(lines) > max_lines:
            shown[-1] = " ".join(lines[max_lines - 1:])
        # EVERY line goes through the clamp, not just the folded one: _wrap
        # splits on whitespace and cannot break a single long token, and a
        # departure list like
        # RECIPE_LTX8_I2V_v2+prequalification[tiled_vae=off,t5_device=cpu]
        # is exactly that -- one token wider than the column, which _wrap
        # returns intact and which would otherwise draw straight off the edge.
        shown = [_fit(ln, fmicro, room) for ln in shown]
        for ln in shown:
            if not ln:
                continue
            d.text((ix, cur_y[0]), ln, fill=_rgba(_GREEN, _A_MICRO),
                   font=fmicro)
            cur_y[0] += _fh(fmicro)
        # No trailing pad. _sc() floors at 8 because it scales FONT sizes, so
        # a nominal "2px" gap would silently be 8px at EVERY canvas height --
        # 24px of invisible drift across three video rows, spent out of the
        # same budget the note itself is competing for. The micro line advance
        # already separates the note from the row below it.

    cur_y = [y]
    # IMAGE
    d.text((x, cur_y[0]), "IMAGE", fill=_rgba(_SUB), font=fsub)
    if m.get("img_rev") is not None:
        d.text((x + _fw(d, "IMAGE  ", fsub), cur_y[0] + _sc(3, h)),
               "REV %s" % m["img_rev"], fill=_rgba(_GREEN, _A_MICRO), font=fmicro)
    cur_y[0] += _fh(fsub) + _sc(4, h)
    for role, val in (m["image_rows"] or [("stills", "(none)")]):
        _row(role, val)
    cur_y[0] += _sc(8, h)
    # VIDEO
    d.text((x, cur_y[0]), "VIDEO", fill=_rgba(_SUB), font=fsub)
    d.text((x + _fw(d, "VIDEO  ", fsub), cur_y[0] + _sc(3, h)),
           "%d RENDER ROLES" % len(m["video_rows"]),
           fill=_rgba(_GREEN, _A_MICRO), font=fmicro)
    cur_y[0] += _fh(fsub) + _sc(4, h)
    # The recipe receipt finally reaches the CARD (2026-07-28). It has been
    # written into ``video_suffix`` since the S-E5 stamp and read by nobody --
    # one write, zero readers -- so the ledger knew what rendered each beat and
    # the credits sheet did not. It draws UNDER its engine row rather than
    # beside it; see _micro_note for why the suffix slot cannot hold it.
    suffixes = m.get("video_suffix") or {}
    for role, eng, fam in m["video_rows"]:
        _row(role, eng, ("· " + fam) if fam else "")
        _micro_note(suffixes.get(eng) or "")
    cur_y[0] += _sc(8, h)
    # MUSIC
    d.text((x, cur_y[0]), "MUSIC", fill=_rgba(_SUB), font=fsub)
    cur_y[0] += _fh(fsub) + _sc(4, h)
    _row("theme", m["music"], "· closing cue looped")
    return cur_y[0] + _sc(16, h)


def _draw_col2(d, x, top, layout, w, h):
    sx = h / _REF_H
    col2 = layout["col2"]
    y = top
    fh1 = _load_font(_sc(_PT_H1, h))
    d.text((x, y), "CAST & VOICES", fill=_rgba(_TEAL), font=fh1)
    y += _fh(fh1)
    ftag = _load_font(_sc(_PT_TAG, h))
    d.text((x, y), "DELIVERED VOICE · PERSISTENT", fill=_rgba(_GREEN, _A_TAG),
           font=ftag)
    y += _fh(ftag) + _sc(18, h)
    fname = _load_font(_sc(_PT_NAME, h))
    fline = _load_font(_sc(_PT_SPEAKER, h))
    for c in col2["cast_rows"]:
        d.text((x, y), c["name"], fill=_rgba(_BRIGHT), font=fname)
        y += _fh(fname)
        line = c["line"] + (('  "%s"' % c["sig"]) if c["sig"] else "")
        d.text((x, y), line, fill=_rgba(_GREEN, 150), font=fline)
        y += _fh(fline) + _sc(14, h)
    y += _sc(12, h)
    d.line([(x, y), (x + int(_COL2_W * sx), y)], fill=_rgba(_GREEN, 40), width=1)
    y += _sc(18, h)
    y = _draw_grid(d, x, y, "[ WRITER / LLM CONFIG ]", col2["writer_grid"], h,
                   label_w=_sc(150, h))
    return y


def render_scroll_canvas(flow: list, col_w: int, h: int):
    """The tall col-3 content strip (STORY SPINE -> full transcript -> intercept
    -> diagnostic). Returns a PIL RGBA image (col_w x content_h)."""
    from PIL import Image, ImageDraw
    # measure by drawing to a throwaway, then render for real
    tmp = Image.new("RGBA", (col_w, 8), (0, 0, 0, 0))
    md = ImageDraw.Draw(tmp)
    total = _scroll_height(md, flow, col_w, h)
    img = Image.new("RGBA", (col_w, max(total, 8)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _scroll_draw(d, flow, col_w, h)
    return _compose_glow(Image.new("RGBA", img.size, (0, 0, 0, 0)), img)


def _scroll_render_ops(d, flow, col_w, h, *, measure):
    """Shared walk; when measure=True only advances y (draw is a no-op)."""
    sx = h / _REF_H
    y = 0
    fh1 = _load_font(_sc(_PT_H1, h))
    ftag = _load_font(_sc(_PT_TAG, h))
    fspk = _load_font(_sc(_PT_SPEAKER, h))
    fbody = _load_font(_sc(_PT_BODY, h))
    fgrid = _load_font(_sc(_PT_GRID, h))
    ffoot = _load_font(_sc(_PT_FOOTER, h))

    def T(x, yy, s, font, fill):
        if not measure:
            d.text((x, yy), s, fill=fill, font=font)

    for kind, block in flow:
        if kind in ("system", "spine"):
            # bracket header + dim-label / bright-value grid (SYSTEM leads the
            # scroll; STORY SPINE follows -- same shape).
            _lw = _sc(110, h) if kind == "spine" else _sc(72, h)
            T(0, y, block["header"], fh1, _rgba(_TEAL))
            y += _fh(fh1) + _sc(8, h)
            for label, val in block["rows"]:
                T(0, y, label, fgrid, _rgba(_GREEN, _A_LABEL))
                lines = _wrap(d, val, fgrid, col_w - _lw)
                for i, ln in enumerate(lines):
                    T(_lw, y + i * _fh(fgrid), ln, fgrid, _rgba(_BRIGHT))
                y += _fh(fgrid) * len(lines) + _sc(6, h)
            y += _sc(24, h)
        elif kind == "transcript":
            T(0, y, block["header"], fh1, _rgba(_TEAL))
            y += _fh(fh1)
            T(0, y, block["tag"], ftag, _rgba(_GREEN, _A_TAG))
            y += _fh(ftag) + _sc(18, h)
            for ln in block["lines"]:
                head = ln["speaker"] + (
                    "  [%s]" % ln["voice"] if ln["voice"] else "")
                T(0, y, head, fspk, _rgba(_TEAL))
                y += _fh(fspk) + _sc(2, h)
                for wl in _wrap(d, ln["text"], fbody, col_w):
                    T(0, y, wl, fbody, _rgba(_BRIGHT))
                    y += _fh(fbody)
                y += _sc(14, h)
            y += _sc(18, h)
        elif kind in ("intercept", "diagnostic"):
            for wl in _wrap(d, block["text"], ffoot, col_w):
                T(0, y, wl, ffoot, _rgba(_GREEN, _A_FOOTER))
                y += _fh(ffoot)
            y += _sc(16, h)
    return y


def _scroll_height(d, flow, col_w, h):
    return _scroll_render_ops(d, flow, col_w, h, measure=True)


def _scroll_draw(d, flow, col_w, h):
    _scroll_render_ops(d, flow, col_w, h, measure=False)


# --------------------------------------------------------------------------- #
# compositing primitives
# --------------------------------------------------------------------------- #
def _radial_bg(w, h):
    from PIL import Image
    import numpy as np
    sw, sh = 480, 270
    yy, xx = np.mgrid[0:sh, 0:sw].astype("float32")
    cx, cy = sw * 0.82, sh * 0.42
    r = np.sqrt(((xx - cx) / (sw * 1.2)) ** 2 + ((yy - cy) / (sh * 0.9)) ** 2)
    r = np.clip(r / 0.62, 0.0, 1.0)
    a, b, c = (np.array(_BG_A, "float32"), np.array(_BG_B, "float32"),
               np.array(_BG_C, "float32"))
    t1 = np.clip(r / 0.44, 0, 1)[..., None]
    mid = a * (1 - t1) + b * t1
    t2 = np.clip((r - 0.44) / 0.56, 0, 1)[..., None]
    out = mid * (1 - t2) + c * t2
    arr = np.dstack([out.astype("uint8"),
                     np.full((sh, sw, 1), 255, "uint8")])
    small = Image.fromarray(arr, "RGBA")
    return small.resize((w, h), Image.BILINEAR)


def _neon_lerp(t):
    stops = [(0.0, _NEON[0]), (0.38, _NEON[1]), (0.72, _NEON[2]), (1.0, _NEON[3])]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / max(1e-6, t1 - t0)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
    return _NEON[-1]


def _compose_glow(base, text_layer):
    from PIL import Image, ImageFilter
    glow = text_layer.filter(ImageFilter.GaussianBlur(6))
    out = Image.alpha_composite(base, Image.blend(
        Image.new("RGBA", base.size, (0, 0, 0, 0)), glow, 0.45))
    return Image.alpha_composite(out, text_layer)


def _scanlines(base):
    from PIL import Image, ImageDraw
    sl = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sl)
    for y in range(0, base.size[1], 4):
        sd.line([(0, y), (base.size[0], y)], fill=(0, 0, 0, 66))
    return Image.alpha_composite(base, sl)


# =========================================================================== #
# Duration + backdrop
# =========================================================================== #
def compute_credits_duration_s(roll_px: int, view_h: int,
                               pps: float = _SCROLL_PPS) -> tuple:
    """Returns (dur_s, effective_pps) for the CLASSIC credits roll.

    ``roll_px`` = the full travel distance of the col-3 canvas (content_h +
    view_h): the content enters from BELOW the viewport and fully exits above
    it, so EVERY line passes through the viewport -- nothing is ever clipped,
    and it ALWAYS rolls regardless of transcript length (roundtable R1: the old
    overflow-only ``content_h - view_h`` model left short episodes static +
    clipped). Over the ceiling -> SPEED UP (never truncate)."""
    if pps <= 0:
        raise CreditsDataError(f"pps must be > 0 (got {pps})")
    roll_px = max(1, int(roll_px))
    dur = _LEAD_HOLD_S + roll_px / pps + _TAIL_HOLD_S
    eff = pps
    if dur > _MAX_HOLD_S:
        eff = roll_px / max(1.0, _MAX_HOLD_S - _LEAD_HOLD_S - _TAIL_HOLD_S)
        dur = _MAX_HOLD_S
    return (round(dur, 3), eff)


def plan_backdrop(clip_manifest: dict) -> dict:
    """LOOK CONTRACT (operator 2026-06-17 / BUG-410): the console rides over the
    LAST existing drama clip, LOOPED. RAISES when the manifest has no usable clip
    (credits-over-black is a bounce)."""
    # Exclude DIRECTORY clips (3D mesh_stage writes a frame directory, not an
    # mp4): ffmpeg `-stream_loop -1 -i <dir>` would crash. kibitz R3. (not-isdir
    # keeps non-existent fixture paths usable while rejecting real directories.)
    rows = [r for r in ((clip_manifest or {}).get("clips") or [])
            if isinstance(r, dict) and r.get("exists") and r.get("path")
            and not os.path.isdir(str(r.get("path")))]
    if not rows:
        raise CreditsDataError(
            "clip manifest has no loopable (file) clip to ride under the credits "
            "console (look contract: credits-over-black / directory clip is a bounce)")
    if any(r.get("start_s") is not None for r in rows):
        return max(rows, key=lambda r: float(r.get("start_s") or 0.0))
    return rows[-1]


# =========================================================================== #
# ffmpeg render + append (every failure RAISES; no source-copy)
# =========================================================================== #
def _ffmpeg_bin() -> str:
    # Honor OTR_FFMPEG (the sibling video nodes' config path) before PATH, so
    # credits never fail on a box where ffmpeg is configured but not on PATH.
    cand = (os.environ.get("OTR_FFMPEG") or "").strip()
    p = (cand if (cand and (os.path.isfile(cand) or shutil.which(cand)))
         else shutil.which("ffmpeg"))
    if not p:
        raise CreditsDataError(
            "ffmpeg not found (OTR_FFMPEG / PATH) -- cannot render credits")
    return p


def _ffprobe_bin() -> str:
    cand = (os.environ.get("OTR_FFPROBE") or "").strip()
    p = (cand if (cand and (os.path.isfile(cand) or shutil.which(cand)))
         else shutil.which("ffprobe"))
    if not p:
        raise CreditsDataError(
            "ffprobe not found (OTR_FFPROBE / PATH) -- cannot render credits")
    return p


def _probe_video(path: str) -> dict:
    out = subprocess.run(
        [_ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise CreditsDataError(f"ffprobe failed on {path!r}: {out.stderr}")
    data = json.loads(out.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate") or "25/1").partition("/")
    fps = float(num) / float(den or 1)
    try:
        dur = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    return {"w": int(st.get("width") or 0), "h": int(st.get("height") or 0),
            "fps": fps, "duration": dur}


def render_credits_clip(layout: dict, backdrop_path: str, out_path: str,
                        *, w: int, h: int, fps: float) -> float:
    """Render the SILENT credits console clip: static base (cols 1-2) + the
    scrolling col-3 canvas, over the LOOPED darkened backdrop, with a fade in/out.
    Returns the clip duration (== declared credits tail). RAISES on any failure."""
    base_img = render_static_base(layout, w, h)
    scroll_img = render_scroll_canvas(layout["col3_flow"],
                                      _sc(_COL3_W, h), h)
    col3_x = _sc(_COL3_X, h)
    col3_y = _sc(_COL3_VIEW_Y, h)
    view_h = min(_sc(_COL3_VIEW_H, h), h - col3_y - _sc(40, h))
    # CLASSIC ROLL (roundtable R1 fix): pad the scroll canvas view_h transparent
    # at TOP and BOTTOM so the content ENTERS from below the viewport and EXITS
    # above the top -- EVERY line passes through the viewport, nothing is ever
    # clipped, and it ALWAYS rolls regardless of transcript length. The crop
    # window travels y = 0 .. roll_px where roll_px = content_h + view_h.
    from PIL import Image as _Image
    _padded = _Image.new(
        "RGBA", (scroll_img.width, scroll_img.height + 2 * view_h), (0, 0, 0, 0))
    _padded.paste(scroll_img, (0, view_h))
    scroll_img = _padded
    roll_px = scroll_img.height - view_h            # = content_h + view_h
    dur, eff_pps = compute_credits_duration_s(roll_px, view_h)

    base_png = out_path + ".base.png"
    scroll_png = out_path + ".scroll.png"
    base_img.convert("RGBA").save(base_png)
    scroll_img.convert("RGBA").save(scroll_png)
    try:
        yexpr = "clip((t-%.3f)*%.4f\\,0\\,%d)" % (_LEAD_HOLD_S, eff_pps, roll_px)
        fade_out_st = max(0.0, dur - _FADE_OUT_S)
        cmd = [
            _ffmpeg_bin(), "-y",
            "-stream_loop", "-1", "-i", backdrop_path,
            # base + scroll are STILLS: they MUST be looped image inputs at the clip
            # fps so each carries an advancing per-frame timestamp. Fed as a plain
            # `-i <png>` they are a SINGLE frame at t=0, so the col-3 crop y-expr
            # (which scrolls on `t`) is frozen at y=0 -> the blank top pad, and the
            # whole scroll column renders empty for the entire roll (the shipped bug).
            "-loop", "1", "-framerate", f"{fps}", "-i", base_png,
            "-loop", "1", "-framerate", f"{fps}", "-i", scroll_png,
            "-filter_complex",
            (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
             f"crop={w}:{h},eq=brightness=-0.32,fps={fps}[bg];"
             f"[bg][1:v]overlay=0:0[b1];"
             # The crop y-expr scrolls per-frame (this ffmpeg evaluates crop x/y
             # per frame; proven by the text-scroll test). The REAL fix is the
             # classic-roll model above (roll_px = content_h + view_h): a large,
             # length-independent travel so col 3 ALWAYS visibly rolls and nothing
             # is clipped (the old overflow-only model left short episodes static).
             f"[2:v]crop=w={_sc(_COL3_W, h)}:h={view_h}:x=0:y='{yexpr}'[sc];"
             f"[b1][sc]overlay={col3_x}:{col3_y}[cv];"
             f"[cv]fade=t=in:st=0:d={_FADE_IN_S:.2f},"
             f"fade=t=out:st={fade_out_st:.2f}:d={_FADE_OUT_S:.2f}[v]"),
            "-map", "[v]", "-an",
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", "18", out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_path) \
                or os.path.getsize(out_path) == 0:
            raise CreditsDataError(
                f"credits console render failed (rc={r.returncode}): "
                f"{(r.stderr or '')[-1000:]}")
    finally:
        for p in (base_png, scroll_png):
            try:
                os.remove(p)
            except OSError:
                pass
    return dur


def append_credits(body_path: str, credits_path: str, out_path: str) -> str:
    """Concat body + credits (both silent, matching params) via the concat
    demuxer, stream-copy. NO source-copy fallback: any failure RAISES."""
    if not os.path.exists(body_path):
        raise CreditsDataError(f"body video missing: {body_path!r}")
    if not os.path.exists(credits_path):
        raise CreditsDataError(f"credits clip missing: {credits_path!r}")
    lst = out_path + ".concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in (body_path, credits_path):
            f.write("file '%s'\n" % p.replace("\\", "/").replace("'", r"'\''"))
    try:
        r = subprocess.run(
            [_ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", lst,
             "-c", "copy", out_path],
            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_path) \
                or os.path.getsize(out_path) == 0:
            raise CreditsDataError(
                f"credits append (concat) failed (rc={r.returncode}): "
                f"{(r.stderr or '')[-800:]}")
    finally:
        try:
            os.remove(lst)
        except OSError:
            pass
    return out_path


# =========================================================================== #
# The terminal node (wired 93 -> OTR_CreditsRoll -> 85)
# =========================================================================== #
class OTRCreditsRoll:
    """Registered as ``OTR_CreditsRoll``. Appends the late SIGNAL LOST console
    to node 93's output and hands the mux (85) the video path + the DECLARED
    credits-tail duration (credits-aware guard)."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "roll"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("video_with_credits_path", "declared_credits_tail_s",
                    "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {
                "forceInput": True,
                "tooltip": "Node 93 output (captions/bars already burned)."}),
            "clip_manifest_json": ("STRING", {
                "forceInput": True,
                "tooltip": "Node 92 slot 1 (third fan-out consumer)."}),
        }}

    def roll(self, video_path, clip_manifest_json):
        from .production_ledger import get_ledger

        if not video_path or not os.path.exists(video_path):
            raise CreditsDataError(
                f"credits input video missing: {video_path!r}")
        try:
            manifest = json.loads(clip_manifest_json or "")
        except (TypeError, ValueError) as exc:
            raise CreditsDataError(
                f"clip_manifest_json unparseable: {exc}") from exc

        led = get_ledger()
        vp = _probe_video(video_path)
        w = vp["w"] or 1920
        h = vp["h"] or 1080
        layout = build_credits_layout(led.data, w=w, h=h, manifest=manifest)
        backdrop = plan_backdrop(manifest)

        base, ext = os.path.splitext(video_path)
        credits_clip = base + "_credits" + (ext or ".mp4")
        out = base + "_with_credits" + (ext or ".mp4")

        tail_s = render_credits_clip(
            layout, str(backdrop["path"]), credits_clip,
            w=w, h=h, fps=vp["fps"])
        append_credits(video_path, credits_clip, out)

        report = json.dumps({
            "ok": True, "declared_credits_tail_s": round(tail_s, 3),
            "hero": layout["hero"], "cols": [len(layout["col1"]),
                                             len(layout["col3_flow"])],
            "backdrop_shot": backdrop.get("shot_id"), "output": out},
            ensure_ascii=True)
        log.info("[OTR_CreditsRoll] appended %.1fs console (hero=%r backdrop=%s) "
                 "-> %s", tail_s, layout["hero"], backdrop.get("shot_id"), out)
        return (out, float(tail_s), report)


__all__ = [
    "OTRCreditsRoll", "CreditsDataError", "build_credits_layout",
    "compute_credits_duration_s", "plan_backdrop", "render_static_base",
    "render_scroll_canvas", "render_credits_clip", "append_credits",
]
