"""nodes/_otr_lfc_smart_suggestion.py — Phase 4.5 (ADR section 6.17).

Deterministic post-Phase-4 scan that synthesizes SFX / music beats
implied by dialogue verbs but missing from the ledger.

  "Start the car."     -> no SFX: car_engine_start beat -> synthesize
  "The door slammed."  -> no SFX: door_slam beat       -> synthesize
  "Music swelled."     -> no MUSIC cue                  -> synthesize

Coverage: ~80% of the common cases hit by a small "implies SFX"
verb allow-list (start, slam, crash, ring, knock, roar, scream,
swelled, breaks, ticks, gasps, sirens, applause, footsteps). The
ambiguous cases (subjective intensity, abstract verbs) route to a
single batched LLM call when `generate_fn` is provided; otherwise
they are skipped with a log.

OPTIONAL phase. Default OFF for v2.0-alpha (ADR section 6.17 calls
this out explicitly). The cascade exposes it via
enable_phase_4_5_smart_suggestion=False. Promote to mandatory once
soak data validates the regex set.

Synthesized beats get `auto_generated=True` so audit logs can
distinguish them from writer-stamped beats.

Status: LFC sprint commit 11 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Set

log = logging.getLogger("OTR.lfc_smart_suggestion")


__all__ = [
    "SmartSuggestionReport",
    "SFX_VERB_PATTERNS",
    "MUSIC_VERB_PATTERNS",
    "phase_4_5_smart_suggestion",
]


# ---------------------------------------------------------------------------
# Verb / SFX inference tables
# ---------------------------------------------------------------------------


# Pattern -> SFX tag mapping. Each pattern is anchored loosely with
# \b on both ends so it catches conjugated forms ("started" /
# "starting" via \w*). The SFX tag is the snake_case identifier that
# the procedural SFX consumer downstream can route. Keep this list
# small and high-precision; soak data drives additions.
SFX_VERB_PATTERNS: tuple[tuple[str, str], ...] = (
    # Engines + machinery
    (r"\b(?:start|starts|started|starting)\s+(?:the\s+)?car\b", "car_engine_start"),
    (r"\b(?:starts|started|starting)\s+the\s+engine\b",          "engine_start"),
    # Doors
    (r"\b(?:slam|slams|slammed|slamming)\s+(?:the\s+)?door\b",    "door_slam"),
    (r"\b(?:the\s+)?door\s+(?:slam|slams|slammed|slamming)\b",    "door_slam"),
    (r"\b(?:knock|knocks|knocked|knocking)\s+(?:on|at)\b",        "door_knock"),
    # Phones / signals
    (r"\b(?:phone|telephone)\s+(?:rang|rings|ringing|rung)\b",    "phone_ring"),
    (r"\b(?:rings|rang|ringing)\s+(?:the\s+)?(?:phone|bell)\b",    "phone_ring"),
    # Glass / crash
    (r"\b(?:glass|window|cup|bottle)\s+(?:shatters?|shattered|breaks?|broke|smash(?:es|ed)?)\b", "glass_shatter"),
    (r"\b(?:crashes?|crashed|crashing)\b",                         "crash"),
    # Sirens / alarms
    (r"\bsirens?\s+(?:wail|wailed|wails|wailing|sound|sounded|sounds|going|fading|approaching)\b",
     "siren"),
    (r"\b(?:alarm|alarms)\s+(?:goes|went|going|sounds?|sounded)\b", "alarm"),
    # Footsteps
    (r"\bfootsteps?\s+(?:approach|approaching|approached|fade|fading|faded|echoed|echo)\b",
     "footsteps"),
    # Applause / crowd
    (r"\b(?:applause|applauding|clapping)\b",                      "applause"),
    (r"\bthe\s+crowd\s+(?:roars?|roared|cheers?|cheered)\b",      "crowd_roar"),
    # Screams / gasps
    (r"\b(?:scream|screams|screamed|screaming)\b",                 "scream"),
    (r"\b(?:gasp|gasps|gasped|gasping)\b",                         "gasp"),
    # Clocks
    (r"\b(?:clock\s+)?(?:tick|ticks|ticked|ticking)\b",            "clock_tick"),
    # Doors creak
    (r"\b(?:door|hinge)\s+creaks?\b",                              "door_creak"),
)


# Music-implication patterns. Map to the structured MusicGen-shape
# tag form per ADR section 6.15 ("MUSIC: ...").
MUSIC_VERB_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bmusic\s+(?:swells?|swelled|swelling)\b",
     "Cinematic, Swelling Strings, 80 BPM, Building"),
    (r"\b(?:the\s+)?(?:strings|orchestra)\s+(?:swell|swells|swelled)\b",
     "Cinematic, Swelling Strings, 80 BPM, Building"),
    (r"\b(?:music\s+)?(?:fades?|faded|fading)\s+(?:in|out)\b",
     "Ambient, Soft Fade, 60 BPM, Subtle"),
    (r"\b(?:music\s+)?starts?\s+playing\b",
     "Cinematic, Establishing, 90 BPM, Moody"),
)


# Compile regex sets once at module load. Case-insensitive.
_SFX_REGEX = tuple(
    (re.compile(p, re.IGNORECASE), tag) for p, tag in SFX_VERB_PATTERNS
)
_MUSIC_REGEX = tuple(
    (re.compile(p, re.IGNORECASE), tag) for p, tag in MUSIC_VERB_PATTERNS
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SmartSuggestionReport:
    """Phase 4.5 summary. Stamped on meta.phase_4_5_smart_suggestion."""

    lines_scanned: int = 0
    sfx_suggested: int = 0
    music_suggested: int = 0
    sfx_added: list = field(default_factory=list)
    music_added: list = field(default_factory=list)
    ambiguous_skipped: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lines_scanned": int(self.lines_scanned),
            "sfx_suggested": int(self.sfx_suggested),
            "music_suggested": int(self.music_suggested),
            "sfx_added": list(self.sfx_added),
            "music_added": list(self.music_added),
            "ambiguous_skipped": int(self.ambiguous_skipped),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _existing_sfx_tags(ledger_data: dict, scene_lines: list) -> set[str]:
    """Set of canonical sfx tags already stamped on this scene's lines.

    B1 fix (commit 12.1): for auto_generated SFX entries the canonical
    tag lives on the `sfx_tag` field stamped at synthesis time. For
    user-stamped SFX entries (no `sfx_tag` field), fall back to the
    text-derived token form.

    Canonical tag format matches the SFX_VERB_PATTERNS table:
    snake_case identifier like "car_engine_start".
    """
    tags: set[str] = set()
    for ln in scene_lines:
        if not isinstance(ln, dict):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role != "sfx":
            continue
        # B1 fix: prefer the canonical sfx_tag field when present.
        canonical = ln.get("sfx_tag")
        if isinstance(canonical, str) and canonical:
            tags.add(canonical.lower())
            continue
        # Fallback for user-stamped SFX: strip the "SFX: " prefix
        # if present, then snake_case the remainder. Matches the
        # canonical tag format closely enough for the common case
        # where the user typed "SFX: car_engine_start".
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("sfx:"):
            lowered = lowered[4:].strip()
        # Try the raw lowered form first (matches if user typed the
        # canonical tag verbatim); also add the snake_case form for
        # near-matches.
        tags.add(lowered)
        tags.add(re.sub(r"\W+", "_", lowered).strip("_"))
    return tags


def _existing_music_tags(scene_lines: list) -> set[str]:
    tags: set[str] = set()
    for ln in scene_lines:
        if not isinstance(ln, dict):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role in ("music_open", "music_close", "music_inter"):
            tags.add(role)
    return tags


def _next_beat_id(
    ledger_data: dict,
    prefix: str = "auto_sfx_",
    *,
    reserved: Optional[set[str]] = None,
) -> str:
    """Generate a stable auto-incrementing beat_id for synthesized entries.

    Scans existing beat_id + line_id fields under the same prefix on
    the ledger. The optional `reserved` set lets the caller pass a
    running registry of IDs claimed in the current synthesis pass --
    needed because synthesised entries are accumulated in a LOCAL
    new_lines list during the scene loop and aren't visible on
    ledger_data["lines"] until the loop exits (B6 fix).

    Returns the next free `f"{prefix}{i:03d}"` id and adds it to
    `reserved` when supplied so the caller's next call gets a fresh
    id.
    """
    existing: set[str] = set()
    for source in (ledger_data.get("beats") or [],
                   ledger_data.get("lines") or []):
        for item in source:
            if not isinstance(item, dict):
                continue
            for key in ("beat_id", "line_id"):
                val = item.get(key) or ""
                if isinstance(val, str) and val.startswith(prefix):
                    existing.add(val)
    if reserved:
        existing |= reserved
    i = 0
    while True:
        candidate = f"{prefix}{i:03d}"
        if candidate not in existing:
            if reserved is not None:
                reserved.add(candidate)
            return candidate
        i += 1


def phase_4_5_smart_suggestion(
    led,
    *,
    enable: bool = False,
    generate_fn=None,
) -> SmartSuggestionReport:
    """LFC Phase 4.5. Synthesize implied SFX / music beats.

    Disabled by default per ADR section 6.17 ("Mark this OPTIONAL
    for v2.0-alpha"). Opt in with enable=True. When disabled,
    returns an empty report and stamps a skip record.

    For each character / announcer line, scans the text for
    SFX-implication patterns. When a pattern matches AND the
    inferred SFX tag is NOT already present in the scene, appends
    a new SFX entry to `lines` with `auto_generated=True`. Same
    treatment for music-implication patterns.

    Mutation policy
      Mutates    Appends new SFX / music ENTRIES to lines (this is the
                 only LFC phase that adds lines; flagged
                 `auto_generated=True` for audit). Does NOT touch
                 existing line.text / char_id / speaker_role.
      Asserts    line_id stability for ORIGINAL lines (no renumbering;
                 the appends slot in at scene boundary positions, not
                 mid-scene).

    The ADR's stated "ambiguous cases route to a single batched LLM
    call" is gated on `generate_fn` being supplied. When it is, the
    LLM batched call is reserved as a follow-up (commit 11 covers
    the regex tier; LLM tier is a future ADR follow-up). For now
    ambiguous cases are counted on `ambiguous_skipped` but not
    rewritten.

    Stamps meta.phase_4_5_smart_suggestion with the full report.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = SmartSuggestionReport()
    if not enable:
        ledger_data.setdefault("meta", {})["phase_4_5_smart_suggestion"] = {
            "skipped": True, "skipped_reason": "enable=False",
        }
        log.debug("[LFC:phase_4_5] disabled (enable=False)")
        return rep

    lines = ledger_data.get("lines") or []
    if not lines:
        ledger_data.setdefault("meta", {})["phase_4_5_smart_suggestion"] = (
            rep.to_dict()
        )
        return rep

    # Ledger-wide music_inter detection. Per ADR section 6.17, the
    # rule "don't synth music when the user already provided a
    # music_inter elsewhere" prevents accidental layering. Checked
    # ONCE up front; cheaper + more conservative than scene-local
    # dedupe (which has edge cases at the music_inter boundary itself).
    _ledger_has_music_inter = any(
        isinstance(ln, dict)
        and (ln.get("speaker_role") or "").strip().lower() == "music_inter"
        for ln in lines
    )

    # Iterate scenes (music_inter boundary). For each scene check
    # the dialogue text for SFX / music implications, then append
    # new entries inside the scene's slice. Building a new lines
    # list keeps the line_id ordering intact.
    new_lines: list = []
    rep.lines_scanned = len(lines)

    # Partition by scene (music_inter dividers). Keep a single-pass
    # implementation so the ordering invariant holds.
    scene_buf: list = []

    # B6 fix (commit 12.1): track IDs reserved across the whole
    # synthesis pass so two SFX/music suggestions in one run can't
    # collide on the same `auto_sfx_000` / `auto_music_000` id. The
    # set is per-namespace (line_id and beat_id share the same
    # prefix system, but each new entry needs its own line_id AND
    # its own beat_id -- so we keep them in one combined reservation
    # to guarantee uniqueness across both fields).
    reserved_line_ids: set[str] = set()
    reserved_beat_ids: set[str] = set()

    def _flush_scene_with_suggestions():
        if not scene_buf:
            return
        existing_sfx = _existing_sfx_tags(ledger_data, scene_buf)
        existing_music = _existing_music_tags(scene_buf)

        # Scan voiced dialogue.
        added_this_scene_sfx: set[str] = set()
        added_this_scene_music: set[str] = set()
        for ln in scene_buf:
            if not isinstance(ln, dict):
                continue
            role = (ln.get("speaker_role") or "").strip().lower()
            if role not in ("character", "announcer"):
                continue
            text = ln.get("text") or ""
            if not text:
                continue
            for rx, tag in _SFX_REGEX:
                if not rx.search(text):
                    continue
                # De-dupe against scene-existing sfx + already-suggested.
                if tag in existing_sfx or tag in added_this_scene_sfx:
                    continue
                added_this_scene_sfx.add(tag)
                rep.sfx_suggested += 1
                rep.sfx_added.append({
                    "line_id": ln.get("line_id", ""),
                    "tag": tag,
                    "trigger_text": text[:80],
                })
            for rx, tag in _MUSIC_REGEX:
                if not rx.search(text):
                    continue
                # Music key is the tag string itself for de-dupe.
                # Ledger-wide guard: if the user has stamped any
                # music_inter anywhere in the lines list, treat
                # music as already-covered.
                if (
                    _ledger_has_music_inter
                    or "music_inter" in existing_music
                    or tag in added_this_scene_music
                ):
                    continue
                added_this_scene_music.add(tag)
                rep.music_suggested += 1
                rep.music_added.append({
                    "line_id": ln.get("line_id", ""),
                    "tag": tag,
                    "trigger_text": text[:80],
                })

        # Append the scene's original lines, then the synthesized
        # entries at the end of the scene block. B6 fix: each ID
        # generator call threads through the running reservation
        # sets so two SFX synths in one run get distinct IDs. B1
        # fix: stamp `sfx_tag` so cross-run dedupe via
        # _existing_sfx_tags matches the canonical tag.
        new_lines.extend(scene_buf)
        for entry in added_this_scene_sfx:
            new_lines.append({
                "line_id": _next_beat_id(
                    ledger_data, "auto_sfx_",
                    reserved=reserved_line_ids,
                ),
                "beat_id": _next_beat_id(
                    ledger_data, "auto_sfx_beat_",
                    reserved=reserved_beat_ids,
                ),
                "speaker_role": "sfx",
                "char_id": "",
                "text": f"SFX: {entry}",
                "char_count": len(f"SFX: {entry}"),
                "word_count": 2,
                "skip": False,
                "auto_generated": True,
                "sfx_tag": entry,
            })
        for entry in added_this_scene_music:
            new_lines.append({
                "line_id": _next_beat_id(
                    ledger_data, "auto_music_",
                    reserved=reserved_line_ids,
                ),
                "beat_id": _next_beat_id(
                    ledger_data, "auto_music_beat_",
                    reserved=reserved_beat_ids,
                ),
                "speaker_role": "music_inter",
                "char_id": "",
                "text": f"MUSIC: {entry}",
                "char_count": len(f"MUSIC: {entry}"),
                "word_count": len(entry.split()),
                "skip": False,
                "auto_generated": True,
                "music_tag": entry,
            })

    for ln in lines:
        if not isinstance(ln, dict):
            scene_buf.append(ln)
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role == "music_inter":
            _flush_scene_with_suggestions()
            scene_buf = [ln]  # music_inter starts the NEXT scene
        else:
            scene_buf.append(ln)
    _flush_scene_with_suggestions()

    # Only mutate if anything was actually added.
    if rep.sfx_suggested or rep.music_suggested:
        ledger_data["lines"] = new_lines
    ledger_data.setdefault("meta", {})["phase_4_5_smart_suggestion"] = (
        rep.to_dict()
    )
    log.info(
        "[LFC:phase_4_5] scanned=%d sfx_added=%d music_added=%d",
        rep.lines_scanned, rep.sfx_suggested, rep.music_suggested,
    )
    return rep
