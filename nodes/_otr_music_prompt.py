"""nodes/_otr_music_prompt.py -- the single source of truth for theme-music cue
prompts, routed through the Meta brief protocol.

Every music cue prompt (opening / closing / interstitial) is composed HERE from
the story/Meta brief on the ledger meta, read via the brief-reader protocol
(`_otr_brief_reader._read_brief_field`) -- never by poking meta directly with a
local template. This is the same downstream-consumer contract the visual slots
(FLUX / LTX / HuMo / portraits) follow, so music generation pulls period,
setting, and mood from the same propagating creative brief as every other
creative call. Cue-specific shaping (the per-cue character template) is layered
on top of the brief-derived context.

Mood resolution cascade (preserved from the audited musicgen path so a given
brief yields the same music): v2 `music_mood_terms` (top 3) -> v1
`story_brief_terms.atmosphere` (top 3) -> keyword-mined `news.script_brief` ->
neutral "atmospheric". Plus a setting "evokes ..." clause, an optional period
descriptor, the cue character, and the instrumental-only tail.

PURE: no I/O, no GPU, no engine imports. Consumers (the theme node + the legacy
musicgen node until it is retired) import from here; this module imports only the
brief reader. UTF-8 no BOM, ASCII-only, no em-dashes.
"""
from __future__ import annotations

from ._otr_brief_reader import _read_brief_field, spoken_term

# Three fixed cues + durations (seconds). Durations are part of cue identity;
# keep stable (mirrors the legacy MusicGen cue durations).
CUE_DURATIONS: dict[str, int] = {"opening": 12, "closing": 8, "interstitial": 4}

# Universal instrumental-only tail, applied last so it always lands at the end.
_PROMPT_TAIL = ", instrumental only, no dialogue, no vocals"

# Per-cue musical character templates (the cue-specific shaping layered on top of
# the brief-derived mood/setting/period context).
_CUE_CHARACTER: dict[str, str] = {
    "opening":      "slow atmospheric build, introduces the scene, "
                    "ends on a sustained chord, instrumental intro",
    "closing":      "resolving cadence, gentle decay into silence, "
                    "instrumental outro",
    "interstitial": "brief textural bridge, unresolved, short "
                    "instrumental transition",
}

# Keyword-mined mood tags from news.script_brief (last-ditch fallback when the
# brief carries no music_mood_terms and no atmosphere). Case-insensitive.
_MOOD_TAGS: dict[str, str] = {
    "betrayal":   "minor mode, unresolved tension",
    "discovery":  "rising figure, slight upward motion",
    "loss":       "subdued, slow decay",
    "urgent":     "tighter rhythm, percussive accents",
    "isolation":  "sparse texture, wide stereo field",
    "danger":     "building tension, dissonant cluster",
    "mystery":    "harmonic ambiguity, slow modulation",
    "triumph":    "resolving cadence, brighter register",
    "conflict":   "rhythmic accents, opposing voices",
    "silence":    "minimal density, long pauses",
}


def _mood_suffix(script_brief: str) -> str:
    """Mine mood tags from the news script_brief. Returns a comma-prefixed
    suffix (e.g. ', minor mode, unresolved tension') or '' if nothing matches."""
    if not script_brief:
        return ""
    low = script_brief.lower()
    tags: list[str] = []
    seen: set[str] = set()
    for keyword, tag in _MOOD_TAGS.items():
        if keyword in low and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return (", " + ", ".join(tags)) if tags else ""


def compose_music_prompt(meta: dict, cue_id: str) -> tuple[str, int]:
    """Compose a music cue prompt from the Meta brief, returning (prompt,
    duration_sec). Reads every brief field through the brief-reader protocol;
    never crashes on an absent / malformed brief (falls through to a neutral
    atmospheric default + the cue character template).
    """
    terms = (meta.get("story_brief_terms") or {}) if isinstance(meta, dict) else {}
    if not isinstance(terms, dict):
        terms = {}

    setting_raw = terms.get("setting") or []
    if not isinstance(setting_raw, list):
        setting_raw = []
    # Normalised: this is composed into the MusicGen TEXT prompt below, and the
    # brief emits identifier case (PBUG-20260903-04).
    setting_terms = [spoken_term(t) for t in setting_raw if spoken_term(t)]

    # Mood: v2 music_mood_terms (via the protocol reader) -> v1 atmosphere ->
    # keyword-mined news.script_brief.
    mood_terms: list[str] = []
    music_mood_raw = _read_brief_field(meta, "music_mood_terms", default=[])
    if isinstance(music_mood_raw, list):
        mood_terms = [str(t).strip() for t in music_mood_raw if str(t).strip()]
    if mood_terms:
        mood_terms = mood_terms[:3]
    else:
        atmosphere_raw = terms.get("atmosphere") or []
        if not isinstance(atmosphere_raw, list):
            atmosphere_raw = []
        atmosphere = [str(t).strip() for t in atmosphere_raw if str(t).strip()]
        if atmosphere:
            mood_terms = atmosphere[:3]
        else:
            # Meta split (2026-07-09): the last-ditch mood mining prefers
            # the PRODUCED story's logline (a summary of the actual
            # episode) over the pre-generation source digest; the digest
            # survives only as the final floor for old ledgers.
            produced = (meta.get("produced_story") or {}) if isinstance(meta, dict) else {}
            if not isinstance(produced, dict):
                produced = {}
            news_meta = (meta.get("news") or {}) if isinstance(meta, dict) else {}
            if not isinstance(news_meta, dict):
                news_meta = {}
            seed_text = (
                produced.get("logline") or news_meta.get("script_brief") or ""
            )
            kw = _mood_suffix(seed_text).lstrip(", ").strip()
            mood_terms = [t.strip() for t in kw.split(",") if t.strip()] if kw else []

    setting_str = ", ".join(setting_terms[:2]) if setting_terms else ""

    # Period overlay: gen_params_initial.period_voice descriptor, read through
    # the protocol so a future nested move needs no consumer change.
    period_descriptor = ""
    gen_params = meta.get("gen_params_initial") if isinstance(meta, dict) else None
    if isinstance(gen_params, dict):
        pv = gen_params.get("period_voice")
        if isinstance(pv, dict):
            descriptor = pv.get("descriptor") or pv.get("music_descriptor")
            if isinstance(descriptor, str) and descriptor.strip():
                period_descriptor = descriptor.strip()

    parts: list[str] = []
    parts.append(", ".join(mood_terms) if mood_terms else "atmospheric")
    if setting_str:
        parts.append(f"evokes {setting_str}")
    if period_descriptor:
        parts.append(period_descriptor)
    parts.append(_CUE_CHARACTER[cue_id])
    prompt = ", ".join(parts) + _PROMPT_TAIL
    return prompt, CUE_DURATIONS[cue_id]
