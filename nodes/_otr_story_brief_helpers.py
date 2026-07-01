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

#: BUG-411 restore (2026-06-14): the 6/5 FLUX image pipeline
#: (visual/batch_flux_render.py) appended a RICHER cinematic grade than the
#: shared STYLE_TAIL_DEFAULT. The image-pipeline rewrite into
#: _otr_image_engines dropped the "anamorphic lens, heavy vignette, muted color
#: grade, sharp focus" grade descriptors (legacy _DEFAULT_STYLE_SUFFIX), which
#: flattened the look. Re-added on the IMAGE STILL path ONLY
#: (compose_still_prompt, after STYLE_TAIL_DEFAULT); the shared tail that LTX
#: scene clips + character video (style_tail=True) use stays untouched.
IMAGE_GRADE_TAIL = ("anamorphic lens, heavy vignette, muted color grade, "
                    "sharp focus")

#: BUG-411: the 6/5 radio bookend / radio stills carried a broadcast-distress
#: identity suffix appended to every radio prompt (legacy _RADIO_PROMPT_SUFFIX);
#: the "35mm film grain, broadcast-distressed" grade IS the lush distressed
#: tint the operator wants back. Re-added to the radio scene stills
#: (open/announcer/music) so they read as a distressed period broadcast still.
RADIO_BROADCAST_TAIL = ("35mm film grain, broadcast-distressed cinematic "
                        "aesthetic, centered composition")

#: The render-constraint clause the LTX scene prompts carry; preserved
#: verbatim through max_chars trimming.
NO_TEXT_CLAUSE = "no on-screen text"


def get_era_tail(meta: Any, profile: str = "full") -> str:
    """The brief-derived era/aesthetic tail; NEVER empty, never raises.

    ``profile="full"`` (default; every pre-still-spine call site, behavior
    unchanged) ports the legacy ``_resolve_era_tail`` precedence (Sprint
    8.7): ``atmosphere_line`` -> ``visual_palette`` (top 3) -> v1
    lighting+atmosphere (:func:`get_story_brief_lighting`) -> the
    :data:`ERA_TAIL_DEFAULT` constant. v2 fields come through the canonical
    brief reader; every failure path degrades, fail-soft.

    ``profile="still"`` (still-spine ST-1): the TRIMMED tail for still-image
    prompts -- atmosphere line + palette top-2 + lighting top-2, capped at
    ~120 chars on a word boundary. Stills carry their subject up front; the
    full tail diet would push FLUX toward atmosphere soup. The per-episode
    palette color (e.g. Mars = red) comes through HERE by design -- the
    still profile trims the tail, never deletes it.
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
    if profile == "portrait":
        # Portrait era tail: NEVER includes the episode's ambient colour
        # palette (which skews blue on sci-fi, red on period drama and bleeds
        # directly into every character face). Lighting terms (quality/mood,
        # not palette colours) are safe and preserved (BUG-LOCAL-113).
        parts = []
        if atmosphere_line:
            parts.append(atmosphere_line)
        m2 = _meta(meta)
        terms2 = m2.get("story_brief_terms") or {}
        if isinstance(terms2, dict):
            lighting = [str(t).strip() for t in (terms2.get("lighting") or [])
                        if str(t).strip()]
            parts.extend(lighting)
        return ", ".join(parts) or ERA_TAIL_DEFAULT
    if profile == "still":
        m = _meta(meta)
        terms = m.get("story_brief_terms") or {}
        if not isinstance(terms, dict):
            terms = {}
        lighting = [str(t).strip() for t in (terms.get("lighting") or [])
                    if str(t).strip()][:2]
        parts = []
        if atmosphere_line:
            parts.append(atmosphere_line)
        parts.extend(palette[:2])
        parts.extend(lighting)
        out = ", ".join(parts) or ERA_TAIL_DEFAULT
        if len(out) > 120:                 # word-boundary trim (~120 chars)
            cut = out[:120]
            idx = cut.rfind(" ")
            out = (cut[:idx] if idx >= 20 else cut).rstrip(" ,")
        return out
    v1_tail = (get_story_brief_lighting(meta) or "").strip()
    parts = []
    if atmosphere_line:
        parts.append(atmosphere_line)
    if palette:
        parts.extend(palette)
    if v1_tail:
        parts.append(v1_tail)
    return ", ".join(parts) or ERA_TAIL_DEFAULT


# ---------------------------------------------------------------------------
# Still-spine ST-1 (2026-06-10): the shared OPEN SUBJECT + the 5-layer still
# prompt composer. The render driver's concrete radio-set subject wording
# MOVES here (one source of truth -- the driver's LTX text prompt and the
# scene STILL prompt for the same beat lead with the SAME subject string;
# the parity test in tests/test_still_spine_helpers.py locks it). Layer
# order restores the proven 6/5 composer (legacy otr_video_plan.py PASS-3:
# subject first, scene context, framing hint, era tail, style tail -- FLUX
# weights earlier tokens more heavily).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BRIEF-DRIVEN RADIO FORM (2026-07-01) -- kills the hardcoded-1940s anchors.
#
# A DETERMINISTIC brief -> radio-form-noun map (NO LLM: nondeterminism would
# break the reproducibility invariant, and a ~7-entry explicit map is the
# deterministic source of truth, not "drift"). It decides ONLY the physical
# RADIO OBJECT the imagery is built around; the era TEXTURE still rides
# :func:`get_era_tail`. Lives HERE (the pure brief-helper) so both the image
# node (build_radio_host_prompt) and get_open_subject can share it without a
# circular import. Ordered most-specific-first; first keyword hit wins; the
# default is a neutral tube radio (a radio FORM, never a 1940s studio anchor).
# ---------------------------------------------------------------------------
_RADIO_FORM_MAP = (
    (("space", "orbital", "docking", "spacecraft", "starship", "space station",
      "sci-fi", "science fiction", "futuristic", "galactic", "interstellar"),
     "a sleek space-station communications console"),
    (("undersea", "submarine", "deep sea", "abyssal", "ocean floor", "bathyscaphe"),
     "a sealed submarine radio station"),
    (("war", "military", "battlefield", "trench", "frontline", "front line"),
     "a rugged portable field radio transceiver"),
    (("western", "frontier", "old west", "dustbowl", "prairie"),
     "a weathered cathedral-style wooden radio set"),
    (("victorian", "steampunk", "gaslight", "brass", "clockwork"),
     "an ornate brass-and-mahogany valve wireless set"),
    (("mid-century", "atomic age", "1950s", "1960s", "retro-futuristic"),
     "a mid-century tabletop transistor radio"),
    (("noir", "detective", "deco", "art deco", "1930s", "1940s", "prohibition"),
     "an art-deco bakelite tube radio"),
)
_RADIO_FORM_DEFAULT = "a vintage tabletop tube radio receiver"


def _radio_form_haystack(meta: Any) -> str:
    """Lowercased brief signal text for the form resolver: ``meta.style`` plus
    ``story_brief_terms`` setting/atmosphere. Pure; tolerant of a bare meta."""
    m = meta if isinstance(meta, dict) else {}
    parts = [str(m.get("style") or "")]
    terms = m.get("story_brief_terms")
    if isinstance(terms, dict):
        for key in ("setting", "atmosphere"):
            v = terms.get(key)
            if isinstance(v, list):
                parts.extend(str(t) for t in v)
            elif v:
                parts.append(str(v))
    return " ".join(parts).lower()


def radio_form_from_meta(meta: Any) -> str:
    """DETERMINISTIC brief -> radio-form noun phrase (NO LLM). The first keyword
    match in :data:`_RADIO_FORM_MAP` wins; :data:`_RADIO_FORM_DEFAULT` (a neutral
    tube radio, NOT the retired 1940s studio anchor) when nothing matches. Pure;
    never empty."""
    hay = _radio_form_haystack(meta)
    for keys, form in _RADIO_FORM_MAP:
        if any(k in hay for k in keys):
            return form
    return _RADIO_FORM_DEFAULT


def get_open_subject(role: str, synthetic: bool, meta: Any = None) -> str:
    """The CONCRETE, FACELESS radio subject for an OPEN / bookend beat (r5b
    operator catch: image models render narrative loglines as murk -- opens lead
    with a picture, never a sentence).

    BRIEF-DRIVEN (2026-07-01): the physical radio FORM comes from
    :func:`radio_form_from_meta` (deterministic, no LLM), so a non-1940s brief no
    longer opens on the hardcoded 1940s set. FACELESS by contract -- ONLY HuMo
    gets a face; this still is what ltx_audio_in / still_pan / still_flat show for
    the bookends. Pure; never empty. ``meta`` optional (bare -> neutral tube
    radio form)."""
    form = radio_form_from_meta(meta or {})
    if synthetic:
        return ("%s warming up on a table, glowing dials and tubes, "
                "warm filament glow" % form)
    if str(role or "") == "announcer_visual":
        return "%s in a broadcast booth, glowing warmly, lit dials and tubes" % form
    return "%s glowing warmly, vacuum tubes and dials" % form


#: Framing hints (layer 3 of the 5-layer still composer). The macro framing
#: is the 6/5 look the operator wants back; the portrait framing matches the
#: round-5 three-quarter STYLE_ANCHOR decision ("more body -- better").
STILL_FRAMING_OPEN = "full-frame macro, centered subject"
STILL_FRAMING_PORTRAIT = ("three-quarter framing, full head and shoulders with "
                          "clear headroom above, face unobstructed")
#: Scene-BEAT framing (person beats, 2026-06-16 framing fix): the old scene path
#: reused STILL_FRAMING_OPEN ("full-frame macro") which has NO headroom directive
#: -> LTX i2v inherits the tight still and crops heads. Positive-only (FLUX plants
#: negated tokens -- the c01 lesson); keeps the operator's wider three-quarter
#: look, just guarantees the whole head + headroom. scene_open (the radio-set
#: object beat) KEEPS the macro -- no person, no head to frame.
STILL_FRAMING_SCENE_BEAT = ("cinematic three-quarter framing, people shown with "
                            "full heads and clear headroom inside frame, faces "
                            "unobstructed, balanced composition")
#: Scene-CHARACTER framing (BUG 1, 2026-06-20 operator directive): the LANDSCAPE
#: still-only character beat (still_flat / still_pan / ltx_video on a character
#: line). Leads with the CHARACTER (compose_still_prompt subject = appearance) in
#: a WIDE 16:9 environment shot -- a medium shot framing the person inside the
#: scene, NEVER the vertical portrait (which pillarboxes -> the radio-booth floor
#: fills the sides) and NEVER the generic radio-set scene subject. Positive-only
#: (FLUX plants negated tokens). The radio broadcast tail is dropped for this kind
#: so the frame reads as the character in their world, not an on-air booth.
STILL_FRAMING_SCENE_CHARACTER = (
    "cinematic medium shot, the character framed within a wide 16:9 environment, "
    "full head and shoulders with clear headroom inside frame, face unobstructed, "
    "balanced landscape composition")


def compose_still_prompt(meta: Any, *, kind: str, role: str = "",
                         beat_id: str = "", char_entry: Any = None) -> str:
    """ONE still-image prompt in the legacy 5-layer order: subject /
    setting top-2 / framing hint / TRIMMED era tail (still profile) /
    style tail. Scene kinds append the :data:`NO_TEXT_CLAUSE` (a render
    constraint stills must carry; portraits never need it).

    ``kind``: ``"portrait"`` leads with the character's own description
    (``portrait_prompt`` -> ``appearance`` -> ``character_description`` on
    ``char_entry``); ``"scene_character"`` (BUG 1, 2026-06-20) ALSO leads with
    the character description but composes a WIDE 16:9 medium shot (the landscape
    still-only character beat -- still_flat/still_pan/ltx_video) and DROPS the
    radio-booth tail; every other scene kind (``scene_open`` / ``scene_beat``)
    leads with :func:`get_open_subject` (``synthetic`` = kind=="scene_open"
    -- the b000 opening-music beat is the synthetic open). Pure; never
    raises; never returns empty (the subject layer always exists).
    """
    is_portrait = (kind == "portrait")
    is_char_scene = (kind == "scene_character")
    if is_portrait or is_char_scene:
        # Both lead with the CHARACTER's own description; scene_character just
        # frames them in a wide environment instead of a tight portrait.
        ce = char_entry if isinstance(char_entry, dict) else {}
        subject = str(ce.get("portrait_prompt") or ce.get("appearance")
                      or ce.get("character_description") or "").strip()
        if not subject:
            subject = "a period-dressed character, face clearly visible"
    else:
        subject = get_open_subject(role, synthetic=(kind == "scene_open"), meta=meta)
    m = _meta(meta)
    terms = m.get("story_brief_terms") or {}
    if not isinstance(terms, dict):
        terms = {}
    setting = ", ".join([str(t).strip() for t in (terms.get("setting") or [])
                         if str(t).strip()][:2])
    if is_portrait:
        framing = STILL_FRAMING_PORTRAIT
    elif is_char_scene:
        framing = STILL_FRAMING_SCENE_CHARACTER
    elif kind == "scene_beat":
        framing = STILL_FRAMING_SCENE_BEAT
    else:
        framing = STILL_FRAMING_OPEN
    pieces = [subject]
    if setting:
        pieces.append(setting)
    pieces.append(framing)
    pieces.append(get_era_tail(meta, profile="still"))
    pieces.append(STYLE_TAIL_DEFAULT)
    if not is_portrait:
        # BUG-411: restore the 6/5 cinematic grade + the radio broadcast-distress
        # identity the image-pipeline rewrite dropped. Scene stills are all radio
        # context (open/announcer/music), so both tails apply; appended AFTER the
        # shared STYLE_TAIL_DEFAULT and BEFORE the NO_TEXT_CLAUSE so the 5-layer
        # order and the no-text contract are preserved. Portraits are unchanged
        # (they keep their three-quarter framing + the shared style tail).
        pieces.append(IMAGE_GRADE_TAIL)
        # BUG 1 (2026-06-20): the character SCENE still is NOT a radio booth --
        # it shows the character in their world, so the broadcast-booth tail
        # ("centered composition", broadcast-distress) is dropped for this kind.
        if not is_char_scene:
            pieces.append(RADIO_BROADCAST_TAIL)
    out = ", ".join(p.strip().rstrip(",") for p in pieces if p and p.strip())
    if not is_portrait:
        out = f"{out}, {NO_TEXT_CLAUSE}"
    return out


def finish_visual_prompt(meta: Any, prompt: str, *, max_chars: int = 0,
                         style_tail: bool = True,
                         era_profile: str = "full") -> str:
    """``prompt + ", " + era_tail [+ ", " + STYLE_TAIL_DEFAULT]`` -- the one
    shared finishing seam every visual prompt site calls.

    ``max_chars`` (0 = uncapped): word-boundary trim of the FINISHED string
    for budgeted consumers (LTX motion budget is 220-240 chars); a trailing
    :data:`NO_TEXT_CLAUSE` present in ``prompt`` survives the trim (it is a
    render constraint, not flavor). Callers run their guards BEFORE this and
    compute prompt hashes AFTER it. Pure; never raises; empty ``prompt``
    returns '' (finishing never invents a subject).

    ``era_profile`` (LTX-I2V ticket Part A, 2026-06-11): passed through to
    :func:`get_era_tail`. Default ``"full"`` = every pre-existing call site,
    byte-identical. The LTX scene composer passes ``"still"`` so VIDEO
    prompts share the stills' trimmed palette diet (the full top-3 palette
    tail is what dragged LTX scene clips reddish while the FLUX stills,
    already on the still profile, stayed balanced).
    """
    base = (prompt or "").strip().rstrip(",")
    if not base:
        return ""
    # Preserve the clause only when TRAILING (pass-02 panel: an occurrence
    # mid-prompt is content, not a render constraint to relocate).
    keep_no_text = base.endswith(NO_TEXT_CLAUSE)
    if keep_no_text:
        base = base[: -len(NO_TEXT_CLAUSE)].strip().rstrip(",").strip()
    pieces = [base, get_era_tail(meta, profile=era_profile)]
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
