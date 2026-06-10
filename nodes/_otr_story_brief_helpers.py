"""nodes/_otr_story_brief_helpers.py -- central consumer helpers (C5b).

Visual + audio consumers do NOT parse the brief prose themselves. They
call one of these five helpers, each shaped to the consumer's prompt
budget and narrative concern:

  get_story_brief_full     -> full brief prose, "" if absent or failed
  get_story_brief_ltx      -> sentence-boundary-trimmed fragment for LTX
  get_story_brief_lighting -> lighting + atmosphere terms, comma joined
  get_story_brief_music_mood -> list[str] of in-vocab mood terms
  get_story_brief_status   -> 'ok' | 'failed' | 'absent'

One helper per consumer shape. The alternative was N slightly-different
bad implementations across N consumer files (refinement section 5).

Module is PURE: no I/O, no GPU, no ComfyUI imports, no MusicGen import.
The dependency direction is consumer -> helper, NEVER the reverse.
(2026-06-10 gap audit: the historical music consumer
`nodes/musicgen_theme.py` no longer exists -- the live music lane reads
the brief via `nodes/_otr_music_prompt.py`'s own protocol. The live
VISUAL consumers are the prompt finisher's callers: ShotLock M4, the
image-prompt deriver, and the render driver's scene composer.) The
`test_get_music_mood_no_musicgen_import` test in
`tests/test_story_brief_helpers_c5b.py` locks the no-reverse-import
property via AST inspection.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

from typing import Any


# MusicGen mood vocabulary -- 16 terms that the audio model handles
# well as a prepended mood prefix. Atmosphere terms outside this set
# are filtered out so the MusicGen prompt stays in known-vocabulary
# space (refinement section 6.3).
_MUSIC_MOOD_VOCAB: frozenset[str] = frozenset({
    "tense", "ominous", "melancholic", "hopeful", "urgent", "calm",
    "eerie", "sombre", "playful", "menacing", "wistful", "frantic",
    "reverent", "uneasy", "stoic", "yearning",
})


def _meta(obj: Any) -> dict:
    """Accept either a meta dict OR a parent dict carrying a meta key."""
    if isinstance(obj, dict):
        if "story_brief_status" in obj or "story_brief" in obj:
            return obj
        sub = obj.get("meta")
        if isinstance(sub, dict):
            return sub
    return {}


def get_story_brief_status(meta: Any) -> str:
    """Return 'ok' / 'failed' / 'absent'.

    'absent' is the legacy-ledger value (no story_brief_status key);
    consumers fall through to legacy prompt construction on this path
    per refinement section 8.2. 'failed' surfaces the L-6 sentinel so
    the consumer can log story_brief_status in its render output
    (E-07 pattern).
    """
    m = _meta(meta)
    status = m.get("story_brief_status")
    if status in ("ok", "failed"):
        return status
    return "absent"


def get_story_brief_full(meta: Any) -> str:
    """Full brief prose, empty string if absent or failed.

    Returning "" on non-ok status lets consumers fall through to
    legacy prompt construction without branching on status -- a
    one-liner truthiness check (`if brief: ...`) does the right
    thing in both ok and non-ok cases.
    """
    m = _meta(meta)
    if get_story_brief_status(m) != "ok":
        return ""
    return (m.get("story_brief") or "").strip()


def get_story_brief_ltx(meta: Any, max_chars: int = 90) -> str:
    """Brief fragment safe for LTX motion prompts.

    Per refinement section 6.1: LTX motion budget is 220-240 chars
    total with 80-100 chars for the brief fragment. Default max_chars
    is 90 (centerpoint of the 80-100 window). Trimmed at the nearest
    sentence-end or clause-boundary before max_chars; NEVER mid-word.
    """
    full = get_story_brief_full(meta)
    if not full or len(full) <= max_chars:
        return full

    candidate = full[:max_chars]
    # Prefer sentence-end (. ! ?) then clause boundary (, ; :) then
    # word boundary. All inside the [0:max_chars] window.
    for sep in (". ", "! ", "? ", "; ", ", ", ": ", " "):
        idx = candidate.rfind(sep)
        if idx >= 20:  # avoid trimming so aggressively the brief is empty
            return candidate[: idx + (0 if sep == " " else 1)].rstrip()
    # Fall back: hard-trim and append ellipsis. Never returns mid-word
    # because the " " sep above catches that case.
    return candidate.rstrip()


def get_story_brief_lighting(meta: Any) -> str:
    """lighting + atmosphere terms, comma-joined.

    Per refinement section 6.2: portrait builders want lighting and
    atmosphere terms WITHOUT setting-terms noise (which would push
    the portrait prompt toward env composition). Returns empty
    string when the brief is absent or failed.
    """
    m = _meta(meta)
    if get_story_brief_status(m) != "ok":
        return ""
    terms = m.get("story_brief_terms") or {}
    if not isinstance(terms, dict):
        return ""
    lighting = [str(t).strip() for t in (terms.get("lighting") or []) if str(t).strip()]
    atmosphere = [str(t).strip() for t in (terms.get("atmosphere") or []) if str(t).strip()]
    return ", ".join(lighting + atmosphere)


def log_story_brief_disposition(meta: Any, consumer_id: str, log: Any) -> str:
    """Sprint E E3 / H4: uniform one-line disposition log for every
    visual + audio consumer.

    Returns the resolved status string so the caller can branch on it
    if needed. Each consumer calls this exactly ONCE per run with its
    own `log` (logging.Logger) and a string consumer_id from the
    canonical set (refreshed 2026-06-10 gap audit -- the legacy batch
    renderers are gone; the live consumers are):

        ltx_scene_open  render_driver.run_real_episode (scene composer)
        shotlock_m4     OTR_ShotLock (per-beat creative derivation)
        flux_portrait   OTR_MetaBriefImagePromptGen (portrait prompts)

    The log line format is uniform across consumers so soak diagnostics
    can grep one canonical pattern instead of N consumer-specific log
    formats. Per refinement E-07 / Sprint E E3 plan H4 fix:

        [story_brief:<consumer_id>] status=<status> brief_chars=<N> terms=<counts>

    Where <counts> is a compact `setting=N lighting=N atmosphere=N`
    summary. status="absent" or "failed" yields brief_chars=0 terms=0/0/0
    and the consumer's subsequent helper calls return safe empty values.
    """
    status = get_story_brief_status(meta)
    m = _meta(meta)
    brief = (m.get("story_brief") or "") if status == "ok" else ""
    terms = m.get("story_brief_terms") or {} if status == "ok" else {}
    if not isinstance(terms, dict):
        terms = {}
    def _n(key):
        v = terms.get(key)
        return len(v) if isinstance(v, list) else (1 if v else 0)

    n_setting = _n("setting")
    n_lighting = _n("lighting")
    n_atmosphere = _n("atmosphere")
    log.info(
        "[story_brief:%s] status=%s brief_chars=%d "
        "terms=setting=%d/lighting=%d/atmosphere=%d",
        consumer_id, status, len(brief),
        n_setting, n_lighting, n_atmosphere,
    )
    return status


def get_story_brief_music_mood(meta: Any) -> list[str]:
    """Mood keywords from atmosphere_terms, intersected with the
    MusicGen mood vocabulary.

    Per refinement section 6.3: MusicGen never sees prose. The helper
    extracts in-vocab terms from atmosphere_terms so the MusicGen
    prompt stays in known-vocabulary space. Returns an empty list
    when the brief is absent, failed, or carries no in-vocab atmosphere
    terms -- the caller (wired at C5g per E-12) treats an empty list
    as "fall through to legacy prompt construction" per refinement
    section 8.2.

    Dependency direction: consumer -> helper. This module does NOT
    import musicgen_theme; the consumer imports THIS helper.

    DEPRECATED-IN-PLACE (2026-06-10 gap audit): the consumer named here
    historically, ``nodes/musicgen_theme.py``, no longer exists (audio
    cleanbreak); the LIVE music lane reads the brief through its own
    protocol in ``nodes/_otr_music_prompt.py`` (v2 music_mood_terms -> v1
    fallback) and does NOT call this helper. Kept for compatibility; do
    not wire new consumers to it without checking _otr_music_prompt first.
    """
    m = _meta(meta)
    if get_story_brief_status(m) != "ok":
        return []
    terms = m.get("story_brief_terms") or {}
    if not isinstance(terms, dict):
        return []
    atmosphere = terms.get("atmosphere") or []
    if not isinstance(atmosphere, list):
        return []
    # Preserve order, intersect against vocab.
    out: list[str] = []
    for raw in atmosphere:
        t = str(raw).strip().lower()
        if t in _MUSIC_MOOD_VOCAB and t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# The prompt FINISHER (2026-06-10 brief-downstream gap audit, F1).
#
# The CW-1 teardown deleted otr_video_plan.py, the only consumer that appended
# the brief's era prose + the film style tail to visual prompts -- every
# post-refactor prompt rendered without them (gap G2/G3, roundtable-hardened
# fix docs/2026-06-10-brief-downstream-gaps/). These helpers restore that
# finishing as ONE shared seam. Pure functions: no logging here (the
# disposition log keeps its once-per-run contract at the NODE level), no
# dedupe, no style presets (3-model panel consensus cuts).
# ---------------------------------------------------------------------------

#: Era-tail fallback when the brief is absent/failed/empty (legacy
#: _DEFAULT_ERA_TAIL, otr_video_plan.py).
ERA_TAIL_DEFAULT = "timeless cinematic aesthetic"

#: The film aesthetic tail (legacy _DEFAULT_STYLE_TAIL, otr_video_plan.py).
STYLE_TAIL_DEFAULT = ("cinematic, 35mm film look, subtle film grain, "
                      "volumetric lighting")

#: The render-constraint clause the LTX scene prompts carry; preserved
#: verbatim through max_chars trimming.
NO_TEXT_CLAUSE = "no on-screen text"


def get_era_tail(meta: Any) -> str:
    """The brief-derived era/aesthetic tail; NEVER empty, never raises.

    Ports the legacy ``_resolve_era_tail`` precedence (Sprint 8.7):
    ``atmosphere_line`` -> ``visual_palette`` (top 3) -> v1
    lighting+atmosphere (:func:`get_story_brief_lighting`) -> the
    :data:`ERA_TAIL_DEFAULT` constant. v2 fields come through the canonical
    brief reader; every failure path degrades, fail-soft.
    """
    atmosphere_line = ""
    palette: list[str] = []
    try:
        try:
            from ._otr_brief_reader import _read_brief_field  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_brief_reader import _read_brief_field  # type: ignore
        raw_line = _read_brief_field(meta, "atmosphere_line", default="")
        if isinstance(raw_line, str):
            atmosphere_line = raw_line.strip()
        raw_palette = _read_brief_field(meta, "visual_palette", default=[])
        if isinstance(raw_palette, list):
            palette = [str(t).strip() for t in raw_palette
                       if str(t).strip()][:3]
    except Exception:  # noqa: BLE001 -- reader unavailable -> v1-only
        pass
    v1_tail = (get_story_brief_lighting(meta) or "").strip()
    parts: list[str] = []
    if atmosphere_line:
        parts.append(atmosphere_line)
    if palette:
        parts.extend(palette)
    if v1_tail:
        parts.append(v1_tail)
    return ", ".join(parts) or ERA_TAIL_DEFAULT


def finish_visual_prompt(meta: Any, prompt: str, *, max_chars: int = 0,
                         style_tail: bool = True) -> str:
    """``prompt + ", " + era_tail [+ ", " + STYLE_TAIL_DEFAULT]`` -- the one
    shared finishing seam every visual prompt site calls.

    ``max_chars`` (0 = uncapped): word-boundary trim of the FINISHED string
    for budgeted consumers (LTX motion budget is 220-240 chars); a trailing
    :data:`NO_TEXT_CLAUSE` present in ``prompt`` survives the trim (it is a
    render constraint, not flavor). Callers run their guards BEFORE this and
    compute prompt hashes AFTER it. Pure; never raises; empty ``prompt``
    returns '' (finishing never invents a subject).
    """
    base = (prompt or "").strip().rstrip(",")
    if not base:
        return ""
    # Preserve the clause only when TRAILING (pass-02 panel: an occurrence
    # mid-prompt is content, not a render constraint to relocate).
    keep_no_text = base.endswith(NO_TEXT_CLAUSE)
    if keep_no_text:
        base = base[: -len(NO_TEXT_CLAUSE)].strip().rstrip(",").strip()
    pieces = [base, get_era_tail(meta)]
    if style_tail:
        pieces.append(STYLE_TAIL_DEFAULT)
    out = ", ".join(p for p in pieces if p)
    if max_chars and len(out) > max_chars:
        budget = max_chars - (len(NO_TEXT_CLAUSE) + 2 if keep_no_text else 0)
        cut = out[:max(budget, 20)]
        idx = cut.rfind(" ")
        if idx >= 20:
            cut = cut[:idx]
        out = cut.rstrip(" ,")
    if keep_no_text:
        out = f"{out}, {NO_TEXT_CLAUSE}"
    if max_chars and len(out) > max_chars:
        # Hard guarantee for pathological small caps (pass-02 panel): the
        # cap wins even over the preserved clause + the 20-char floor.
        out = out[:max_chars].rstrip(" ,")
    return out
