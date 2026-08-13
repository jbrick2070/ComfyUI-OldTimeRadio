"""otr_captions.py -- build burned-in SDH open captions (.ass) from an OTR ledger.

Generates an ASS (libass) subtitle file from the per-episode ``*_ledger.json``
``lines[]`` timing (``start_s`` / ``dur_s`` / ``text`` / ``speaker_role`` /
``char_id``), cross-referencing ``cast[]`` for speaker display names.

Design (per Jeffrey's go-forward feedback 2026-05-30):
  * Default style ``sdh_standard``: Arial 36 ASS units, WHITE dialogue, ~65%-opaque
    black box, bottom-center, max 2 lines. Accessibility master.
  * Optional style ``otr_crt``: green-CRT themed, for A/B QA only -- NOT default.
  * Speaker label coloring ONLY: the ``NAME:`` prefix is colored per speaker;
    the dialogue text stays white. No rainbow captions.
  * There are NO sound-effect captions. The ``sfx`` speaker_role was removed
    2026-07-01 (rip-sfx-broll) and a ledger still carrying one is an invariant
    error (``_otr_ledger_freeze.py:97``); a line is music, announcer, or
    character. ``_SPEECH_ROLES`` captions the latter two, so the older
    "sparse bracketed ``[STATIC HISS]`` cue" note described a lane that no
    longer exists.
  * PERFORMANCE DIRECTION IS SHOWN ON PURPOSE (operator ruling 2026-08-05).
    A caption burns the RAW ledger line, so a parenthetical like
    ``(forcefully winding the clock)`` appears on screen even though the voice
    never speaks it -- TTS independently strips it via ``clean_spoken_text``
    (``_otr_script_prep.py:21``). Caption and audio therefore diverge BY DESIGN;
    measured at 255 cues across 95 of 915 shipped episodes. The operator's call:
    "it's a nice easter egg as long as it's built and we know and it's
    documented." This is that documentation, and
    ``tests/test_otr_captions.py`` pins it so the divergence cannot be
    "corrected" by accident later.
    The same raw text is LOAD-BEARING on the visual side, which is the real
    reason the ledger keeps the line as written rather than stripping it
    upstream. It feeds (a) the still-image prompt
    (``otr_meta_brief_image_prompt.py:1313``), (b) the MOTION CLAUSE that
    directs i2v video -- ``_otr_motion_clause._line_text_index`` reads raw
    ``lines[].text`` and hands it to ``build_clause_messages``, under the
    standing operator directive recorded at ``_otr_motion_clause.py:47``
    ("the line drives the motion") -- and (c) the HUD / full-script print
    (``video_engine.py:1311`` and ``:1962``). Stripping direction out of
    ``lines[].text`` would quietly degrade stills AND motion, so it stays.
    NOT covered by this ruling: a SOURCE CITATION or URL leaking into line text
    is a writer defect, not an easter egg, and is tracked separately.
  * SDH line rules: <=2 lines, <=44 chars/line, target <=17 CPS (hard cap 20),
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

# Shared hero-title-card arithmetic. RELATIVE with a bare-name fallback, never
# `from nodes...` -- an absolute package import resolves under pytest and RAISES
# on a live server (lesson L23).
try:
    from . import _otr_title_card as _OTRTC  # type: ignore
except ImportError:  # loaded with nodes/ on sys.path
    import _otr_title_card as _OTRTC  # type: ignore

# -- ASS coordinate space ---------------------------------------------------
# The header pins these, so every position and size in the file is expressed in
# them and libass scales the whole lot to whatever the video actually is. The
# title planner works in the PROCGEN frame's pixels, which need not match, so
# the emitter scales into this space rather than moving PlayRes -- moving it
# would silently rescale every SDH caption's font and margins too.
PLAY_RES_X = 1920
PLAY_RES_Y = 1080

# -- SDH line rules ---------------------------------------------------------
MAX_CHARS_PER_LINE = 44
MAX_LINES_PER_CUE = 2
TARGET_CPS = 17.0
HARD_CPS_CAP = 20.0
MIN_CUE_DUR_S = 1.0
CAPTION_MARGIN_X = 40

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
        "size": 36,
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


# -- Hero title style -------------------------------------------------------
# A SECOND style line, deliberately additive: the SDH bytes above are unchanged.
#
# This is the whole point of moving the title out of the procgen layer. There
# the composite is `screen` + `green_only`, a LIGHTEN-ONLY blend where
# screen(A, 0) = A, so a black outline is the blend's identity and draws
# nothing -- measured 1.13:1 over a lit monitor (Bible 07.32). Here, downstream
# of that blend, BorderStyle=1 with a real Outline width is an actual dark edge
# against actual pixels.
#
# Alignment 7 (top-left) because the planner hands over PIL top-left origins,
# and every event carries an explicit \pos, so libass collision handling can
# never shift the hero off its planned mark.
TITLE_STYLE_NAME = "TITLE"
# 3 -> 6, and a real shadow, after the FIRST live leg (2026-08-13). Operator, on
# the published artifact: "a bit hard to see from a distance but much improved".
# The measurement said the same thing and says why. On that episode the glyph
# EDGE cleared 13.66:1 against its outline -- the gate passed -- while the glyph
# FILL was only 1.21:1 against the lit gold radio panel behind it, barely up
# from the 1.01:1 the pre-fix control measured. So the outline was carrying the
# whole result: legible close up, and at a distance the 3px border falls below
# what the eye resolves and the phosphor fill sinks back into a bright scene.
#
# Widening the border and adding a dropped shadow buys weight that survives
# downscaling, WITHOUT going to BorderStyle=3 -- an opaque box would be the
# strongest answer and is what the SDH captions use, but it would replace the
# CRT phosphor look with a caption slab, and the Matrix decode is explicitly
# kept. Re-measure `core vs scene` on the next leg: that is the number that has
# to move, not `core vs outline`, which was already passing.
TITLE_OUTLINE_W = 6
TITLE_SHADOW = 2
_TITLE_STYLE_LINE = (
    f"Style: {TITLE_STYLE_NAME},Consolas,48,&H0041FF00,&H000000FF,&H00000000,"
    f"&H00000000,1,0,0,0,100,100,0,0,1,{TITLE_OUTLINE_W},{TITLE_SHADOW},7,0,0,0,1"
)


def _ass_header(style: dict, margin_v: int) -> str:
    s = style
    style_line = (
        "Style: SDH,{font},{size},{primary},&H000000FF,{outline_col},{back},"
        "{bold},0,0,0,100,100,0,0,{bs},{outline},{shadow},2,{mx},{mx},{mv},1"
    ).format(
        font=s["font"], size=s["size"], primary=s["primary"],
        outline_col=s["outline_col"], back=s["back"], bold=s["bold"],
        bs=s["border_style"], outline=s["outline"], shadow=s["shadow"],
        mx=CAPTION_MARGIN_X, mv=margin_v,
    )
    style_line = style_line + "\n" + _TITLE_STYLE_LINE
    return (
        "[Script Info]\n"
        "; OTR SDH open captions -- generated by scripts/otr_captions.py\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {PLAY_RES_X}\n"
        f"PlayResY: {PLAY_RES_Y}\n"
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


class TitlePlanError(ValueError):
    """A supplied hero-title plan could not be turned into ASS events.

    Its own exception type because the CALLER must be able to tell it apart from
    every other caption failure. Captions are best-effort and may pass through;
    a title that was asked for and cannot be drawn is a REFUSAL, because the
    alternative is publishing an episode with no title and a clean log.
    """


def title_events_from_plan(plan) -> list[str]:
    """Per-frame ASS Dialogue events for the hero title card.

    ONE event per visible line per frame, each bounded to ``[fi/fps,
    (fi+1)/fps)`` with fixed position, size, colour and text. That is not a
    workaround for missing tag interpolation -- it is the only shape that
    reproduces the frame-SEEDED scramble, the 1-2 frame POP and the framewise
    dock resize deterministically. Consecutive events share a boundary exactly
    (this frame's end is next frame's start, same rounding), so there is no
    1-centisecond gap to flicker through.

    Raises ``TitlePlanError`` on anything it cannot render faithfully.
    """
    if not isinstance(plan, dict):
        raise TitlePlanError(f"title plan is {type(plan).__name__}, expected a mapping")
    version = plan.get("version")
    if version != _OTRTC.PLAN_VERSION:
        raise TitlePlanError(
            f"title plan version {version!r} != supported {_OTRTC.PLAN_VERSION}; "
            f"refusing to guess at its shape"
        )
    fps = float(plan.get("fps") or 0)
    if fps <= 0:
        raise TitlePlanError(f"title plan fps is {plan.get('fps')!r}")
    res = plan.get("play_res") or []
    if len(res) != 2 or not all(float(v) > 0 for v in res):
        raise TitlePlanError(f"title plan play_res is {res!r}")

    # The plan's coordinates are in the PROCGEN frame's space; the ASS header
    # pins PlayResX/Y to 1920x1080 for the captions. Scale the title into that
    # space rather than moving PlayRes, which would silently rescale every SDH
    # caption's font size and margins along with it.
    sx = PLAY_RES_X / float(res[0])
    sy = PLAY_RES_Y / float(res[1])
    # Font size has ONE number, so it scales vertically; positions scale per
    # axis. When the procgen frame's aspect differs from the ASS coordinate
    # space those disagree, and the glyph ends up a different width than the
    # centring math assumed -- 832x480 into 1920x1080 is a real 2.5% skew, and
    # 832x480 is a shipped choice on this node. \fscx restores the horizontal
    # extent exactly; at matching aspects the ratio is 1.0 and this is a no-op.
    fscx = 100.0 * sx / sy

    events: list[str] = []
    for frame in plan.get("frames") or []:
        fi = int(frame.get("f"))
        start = ass_timecode(fi / fps)
        end = ass_timecode((fi + 1) / fps)
        for item in frame.get("items") or []:
            kind = item.get("kind")
            colour = str(item.get("colour") or "")
            x = int(round(float(item["x"]) * sx))
            y = int(round(float(item["y"]) * sy))
            if kind == "text":
                size = max(1, int(round(float(item["size"]) * sy)))
                # Escape BEFORE the override tags go on, never after: an
                # unescaped brace in a title would otherwise open an override
                # block and swallow the rest of the line.
                text = _ass_escape(str(item.get("text") or ""))
                if not text:
                    continue
                scale = "" if abs(fscx - 100.0) < 0.05 else f"\\fscx{fscx:.3f}"
                tags = f"{{\\pos({x},{y})\\fs{size}{scale}\\c{colour}}}"
                events.append(
                    f"Dialogue: 1,{start},{end},{TITLE_STYLE_NAME},,0,0,0,,{tags}{text}"
                )
            elif kind == "rect":
                w = max(1, int(round(float(item["w"]) * sx)))
                h = max(1, int(round(float(item["h"]) * sy)))
                # A filled block via an ASS drawing command. \bord0 so the
                # cursor does not inherit the TITLE style's outline.
                tags = (f"{{\\pos({x},{y})\\c{colour}\\bord0\\shad0\\p1}}")
                shape = f"m 0 0 l {w} 0 l {w} {h} l 0 {h}"
                events.append(
                    f"Dialogue: 1,{start},{end},{TITLE_STYLE_NAME},,0,0,0,,"
                    f"{tags}{shape}{{\\p0}}"
                )
            else:
                raise TitlePlanError(f"unknown title plan item kind {kind!r}")
    if not events:
        raise TitlePlanError(
            "title plan produced no events -- a plan was supplied, so an empty "
            "result is a defect, not an episode without a title card"
        )
    return events


def build_ass_from_ledger(ledger_path, style: str = "sdh_standard",
                          margin_v: Optional[int] = None,
                          out_path=None,
                          title_plan=None) -> tuple[Optional[str], str]:
    """Build an .ass caption file from a ledger. Returns (out_path, report).

    On any failure returns (None, reason). Best-effort: never raises -- EXCEPT
    for ``TitlePlanError``, which propagates on purpose. Captions may be absent
    from a deliverable; a hero title that was planned and then silently dropped
    may not, and the caller cannot fail closed on a failure it never sees.

    With ``title_plan`` supplied this may produce a TITLE-ONLY .ass: an episode
    whose ledger will not load, or that has no speech lines at all, still gets
    its title card.
    """
    title_events = title_events_from_plan(title_plan) if title_plan else []

    ledger = None
    ledger_err = ""
    try:
        lp = Path(ledger_path)
        ledger = json.loads(lp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        ledger_err = (f"could not load ledger {ledger_path}: "
                      f"{type(exc).__name__}: {exc}")
        if not title_events:
            return (None, ledger_err)
        lp = Path(str(ledger_path) or "episode")

    st = STYLES.get(style)
    if st is None:
        # Reachable, and it used to be silent: this (None, reason) becomes a
        # ValueError in burn_captions_on_video, which the node caught and turned
        # into a clean-master passthrough -- dropping the title with no error.
        return (None, f"unknown style {style!r}; choices: {sorted(STYLES)}")
    mv = int(margin_v) if margin_v is not None else int(st["margin_v"])

    try:
        from . import _otr_ledger_consumers as _OTRLC  # type: ignore
    except ImportError:  # pragma: no cover - direct CLI execution
        import _otr_ledger_consumers as _OTRLC  # type: ignore
    events: list[str] = []
    lint: list[str] = []
    prev_char = None

    # The shared iterator owns the canonical mute contract. Filtering skipped
    # rows BEFORE sorting/clamping prevents a muted row from both leaking into
    # the ASS and shortening the preceding spoken cue.
    dlines = (list(_OTRLC.iter_lines(ledger, roles=_SPEECH_ROLES))
              if ledger is not None else [])
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
        # RAW line text, deliberately -- see the performance-direction note in
        # the module docstring. The caption is NOT the spoken surface here, and
        # that divergence is intended, reviewed and operator-approved.
        text = _ass_escape(str(ln.get("text") or ""))
        if not text:
            continue

        is_turn_start = cid != prev_char
        prev_char = cid
        # Resolve role-tag aliases (notably line char_id="announcer" against a
        # canonical ANNOUNCER cast row whose id is c01) through the shared cast
        # authority. Preserve the legacy display_name fallback on the resolved
        # row rather than rebuilding another raw-id map here.
        cast_row = _OTRLC.cast_lookup(ledger, cid)
        nm = str(
            cast_row.get("name") or cast_row.get("display_name") or ""
        ).strip()

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

    # Reordered 2026-08-12: this used to bail before the title events were even
    # considered, so an episode with no speech lines produced no .ass at all --
    # which, once the title lives ONLY in ASS, is an episode with no title.
    # Title-only output is valid ASS and is the correct answer here.
    if not events and not title_events:
        return (None, "no speech lines found in ledger (nothing to caption)")

    # Title events LAST so the hero draws over the captions on the rare frame
    # where a card and a caption overlap; layer 1 vs layer 0 already says so,
    # and the ordering makes it true independently of libass layer handling.
    ass_text = _ass_header(st, mv) + "\n".join(events + title_events) + "\n"

    if out_path is None:
        eid = str((ledger or {}).get("episode_id")
                  or lp.stem.replace("_ledger", ""))
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
