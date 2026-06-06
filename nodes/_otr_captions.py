"""otr_captions.py -- build burned-in SDH open captions (.ass) from an OTR ledger.

Generates an ASS (libass) subtitle file from the per-episode ``*_ledger.json``
``lines[]`` timing (``start_s`` / ``dur_s`` / ``text`` / ``speaker_role`` /
``char_id``), cross-referencing ``cast[]`` for speaker display names.

Design (per Jeffrey's go-forward feedback 2026-05-30):
  * Default style ``sdh_standard``: Arial 52 px, WHITE dialogue, ~65%-opaque
    black box, bottom-center, max 2 lines. Accessibility master.
  * Optional style ``otr_crt``: green-CRT themed, for A/B QA only -- NOT default.
  * Speaker label coloring ONLY: the ``NAME:`` prefix is colored per speaker;
    the dialogue text stays white. No rainbow captions.
  * Sound/music cues are sparse and bracketed (``[STATIC HISS]`` / music note).
  * SDH line rules: <=2 lines, <=37 chars/line, target <=17 CPS (hard cap 20),
    min 1.0 s on screen, no overlap (later cue start clamps earlier cue end).

This module is pure stdlib and import-safe (no side effects at import). The
node burn step (P1+) calls :func:`build_ass_from_ledger`; the CLI builds an
``.ass`` next to a given ledger and prints the file plus a lint report for QA.

Audio is never touched -- the burn happens on the video stream only, downstream.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

# -- SDH line rules ---------------------------------------------------------
MAX_CHARS_PER_LINE = 37
MAX_LINES_PER_CUE = 2
TARGET_CPS = 17.0
HARD_CPS_CAP = 20.0
MIN_CUE_DUR_S = 1.0

# -- Speaker label color (ASS \3c outline-override format: &Hbbggrr&) -------
# ACCESSIBILITY (Jeffrey 2026-05-30): color is NEVER the speaker cue --
# color-sensitive / color-blind viewers must not depend on it. The speaker
# is identified by the BOLD WHITE "NAME:" label (text + weight). Per-speaker
# color is applied ONLY as a subtle OUTLINE on the name; the fill stays white
# and the dialogue stays white on the opaque box, so every caption is
# constantly, equally legible regardless of color perception.
_WHITE = "&HFFFFFF&"
# Desaturated PASTEL hues so the name outline is a subtle tint, never a neon
# block. Speaker ID never depends on these -- they are a secondary cue only.
_NAME_COLORS_BBGGRR = [
    "&HE0E0A0&",  # soft teal  (#A0E0E0)
    "&HA0E0A0&",  # soft green (#A0E0A0)
    "&HA0E0E0&",  # soft straw (#E0E0A0)
    "&HE0B0E0&",  # soft mauve (#E0B0E0)
]
_ANNOUNCER_COLOR_BBGGRR = "&H80C8FF&"  # soft amber (#FFC880) -- announcer distinct

# -- Style presets ----------------------------------------------------------
# ASS V4+ Style fields: Name, Fontname, Fontsize, PrimaryColour,
# SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline,
# StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,
# Alignment, MarginL, MarginR, MarginV, Encoding.
# Colors here are &HAABBGGRR (alpha 00 = opaque, FF = transparent).
STYLES = {
    # Accessibility master. White text, ~55%-opaque black box (BorderStyle=3).
    "sdh_standard": {
        "font": "Arial",
        "size": 26,
        "primary": "&H00FFFFFF",   # opaque white
        "outline_col": "&H00000000",
        "back": "&H70000000",      # alpha 0x70 -> ~56% opaque black box (lighter)
        "bold": 0,
        "border_style": 3,         # opaque box (uses BackColour)
        "outline": 5,              # box padding -- MUST be >0 or libass draws no box
        "shadow": 0,
        "margin_v": 90,            # P2 QA: clears bottom overscan + procgen waveform HUD
        "announcer_color": _ANNOUNCER_COLOR_BBGGRR,
    },
    # Themed variant for QA comparison only. Green CRT, thin outline (no box).
    "otr_crt": {
        "font": "Consolas",
        "size": 50,
        "primary": "&H0066FF33",   # opaque OTR green (#33FF66 -> bbggrr 66FF33)
        "outline_col": "&H00000000",
        "back": "&H7F000000",      # ~50% box (mostly unused at border_style 1)
        "bold": 0,
        "border_style": 1,         # outline + shadow
        "outline": 2,
        "shadow": 0,
        "margin_v": 70,
        "announcer_color": "&H00BFFF&",
    },
}

# Roles that produce spoken dialogue captions.
_SPEECH_ROLES = {"announcer", "character"}


def ass_timecode(seconds: float) -> str:
    """Format seconds as ASS H:MM:SS.cs (centisecond precision)."""
    if seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Escape characters special to ASS dialogue text."""
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def wrap_words(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """Greedy word-wrap into physical lines of at most ``max_chars``."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
        # hard-break a single word longer than the line
        while len(cur) > max_chars:
            lines.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        lines.append(cur)
    return lines or [""]


def chunk_into_cues(text: str, max_chars: int = MAX_CHARS_PER_LINE,
                    max_lines: int = MAX_LINES_PER_CUE) -> list[str]:
    """Wrap text and group physical lines into cues of <= ``max_lines`` lines.

    Returns a list of cue strings; multi-line cues join physical lines with the
    literal ASS line break ``\\N``.
    """
    phys = wrap_words(text, max_chars)
    cues: list[str] = []
    for i in range(0, len(phys), max_lines):
        cues.append("\\N".join(phys[i:i + max_lines]))
    return cues or [""]


def distribute_time(n: int, start: float, end: float,
                    min_dur: float = MIN_CUE_DUR_S) -> list[tuple[float, float]]:
    """Split [start, end] into ``n`` consecutive (s, e) spans.

    Even split; if the window is too short to give each cue ``min_dur`` the
    spans simply share the available time (a CPS lint warning is raised by the
    caller). Spans never overlap and never exceed ``end``.
    """
    if n <= 0:
        return []
    total = max(0.0, end - start)
    span = total / n
    out: list[tuple[float, float]] = []
    t = start
    for i in range(n):
        s = t
        e = end if i == n - 1 else min(end, t + span)
        out.append((s, e))
        t = e
    return out


def color_for(char_id: str, role: str, style: dict, order_map: dict) -> str:
    """Return the \\c bbggrr color for a speaker's NAME label."""
    if role == "announcer":
        return style.get("announcer_color", _ANNOUNCER_COLOR_BBGGRR)
    idx = order_map.setdefault(char_id, len(order_map))
    return _NAME_COLORS_BBGGRR[idx % len(_NAME_COLORS_BBGGRR)]


def _cast_names(ledger: dict) -> dict:
    names = {}
    for c in (ledger.get("cast") or []):
        cid = str(c.get("char_id") or "")
        nm = c.get("name") or c.get("display_name") or ""
        if cid:
            names[cid] = str(nm).strip()
    return names


def _ass_header(style: dict, margin_v: int) -> str:
    s = style
    style_line = (
        "Style: SDH,{font},{size},{primary},&H000000FF,{outline_col},{back},"
        "{bold},0,0,0,100,100,0,0,{bs},{outline},{shadow},2,60,60,{mv},1"
    ).format(
        font=s["font"], size=s["size"], primary=s["primary"],
        outline_col=s["outline_col"], back=s["back"], bold=s["bold"],
        bs=s["border_style"], outline=s["outline"], shadow=s["shadow"],
        mv=margin_v,
    )
    return (
        "[Script Info]\n"
        "; OTR SDH open captions -- generated by scripts/otr_captions.py\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style_line}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )


def build_ass_from_ledger(ledger_path, style: str = "sdh_standard",
                          margin_v: Optional[int] = None,
                          out_path=None) -> tuple[Optional[str], str]:
    """Build an .ass caption file from a ledger. Returns (out_path, report).

    On any failure returns (None, reason). Best-effort: never raises.
    """
    try:
        lp = Path(ledger_path)
        ledger = json.loads(lp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return (None, f"could not load ledger {ledger_path}: {type(exc).__name__}: {exc}")

    st = STYLES.get(style)
    if st is None:
        return (None, f"unknown style {style!r}; choices: {sorted(STYLES)}")
    mv = int(margin_v) if margin_v is not None else int(st["margin_v"])

    names = _cast_names(ledger)
    lines = ledger.get("lines") or []
    events: list[str] = []
    lint: list[str] = []
    prev_char = None

    # Sort by start time so clamping logic is correct even if unordered.
    dlines = [ln for ln in lines if str(ln.get("speaker_role") or "") in _SPEECH_ROLES]
    dlines.sort(key=lambda ln: float(ln.get("start_s") or 0.0))

    for li, ln in enumerate(dlines):
        role = str(ln.get("speaker_role") or "")
        cid = str(ln.get("char_id") or "")
        start = float(ln.get("start_s") or 0.0)
        dur = float(ln.get("dur_s") or 0.0)
        end = start + max(0.0, dur)
        # Clamp end to the next speech line's start (no overlap).
        if li + 1 < len(dlines):
            nxt = float(dlines[li + 1].get("start_s") or end)
            if nxt < end:
                end = nxt
        text = _ass_escape(str(ln.get("text") or ""))
        if not text:
            continue

        is_turn_start = cid != prev_char
        prev_char = cid
        nm = names.get(cid, "").strip()

        # Prepend speaker label to the text of the FIRST cue of a turn so
        # wrapping accounts for its width; color is injected after wrapping.
        body = text
        label = f"{nm}: " if (is_turn_start and nm) else ""
        cues = chunk_into_cues(label + body)
        spans = distribute_time(len(cues), start, end)

        # CPS lint on the whole line.
        visible = len(text)
        line_dur = max(0.001, end - start)
        cps = visible / line_dur
        if cps > HARD_CPS_CAP:
            lint.append(f"line[{li}] {nm or role}: {cps:.1f} CPS > hard cap {HARD_CPS_CAP} "
                        f"({visible} chars in {line_dur:.2f}s)")
        elif cps > TARGET_CPS:
            lint.append(f"line[{li}] {nm or role}: {cps:.1f} CPS > target {TARGET_CPS}")

        for ci, (cue, (s, e)) in enumerate(zip(cues, spans)):
            if (e - s) < MIN_CUE_DUR_S:
                lint.append(f"line[{li}] cue[{ci}] {nm or role}: on-screen {e - s:.2f}s < {MIN_CUE_DUR_S}s")
            disp = cue
            # Speaker label = BOLD WHITE (color-blind-safe primary cue). The
            # name shares the SINGLE caption box -- no separate box/border or
            # color around it (those drew the ugly highlighted block behind the
            # name). Distinguished by weight only; \r resets to the box style.
            if ci == 0 and label:
                lbl = label.strip()  # "NAME:"
                disp = disp.replace(
                    lbl, f"{{\\b1}}{lbl}{{\\r}}", 1)
            events.append(
                f"Dialogue: 0,{ass_timecode(s)},{ass_timecode(e)},SDH,,0,0,0,,{disp}"
            )

    if not events:
        return (None, "no speech lines found in ledger (nothing to caption)")

    ass_text = _ass_header(st, mv) + "\n".join(events) + "\n"

    if out_path is None:
        eid = str(ledger.get("episode_id") or lp.stem.replace("_ledger", ""))
        out_path = lp.with_name(f"{eid}_captions.ass")
    out_path = Path(out_path)
    try:
        out_path.write_text(ass_text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return (None, f"could not write {out_path}: {type(exc).__name__}: {exc}")

    report = [
        f"captions: {len(events)} events from {len(dlines)} speech lines "
        f"-> {out_path.name} (style={style}, margin_v={mv})",
    ]
    if lint:
        report.append(f"  LINT ({len(lint)}):")
        report.extend("    - " + w for w in lint)
    else:
        report.append("  LINT: clean (all cues within CPS + duration rules)")
    return (str(out_path), "\n".join(report))


def _cli(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not argv:
        print("usage: otr_captions.py <ledger.json> [style] [margin_v]")
        return 2
    ledger = argv[0]
    style = argv[1] if len(argv) > 1 else "sdh_standard"
    mv = int(argv[2]) if len(argv) > 2 else None
    out, report = build_ass_from_ledger(ledger, style=style, margin_v=mv)
    print(report)
    if out:
        print("\n=== ASS FILE ===")
        print(Path(out).read_text(encoding="utf-8"))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
