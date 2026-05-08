"""nodes/_otr_voice_resolver.py

Parse 'engine:preset' voice specs from cast contract entries.
Phase 0+ Cast Contract Extensions §3 dependency, AND Voice Backend
Abstraction §3 (voice spec format). See ROADMAP.md.

Status: parser only. Backend dispatch lives in Voice Backend Abstraction
work (separate Phase 0+ candidate, not in scope for this session).

Public surface:
    VoiceSpec — frozen dataclass with engine + preset
    parse_voice_spec(s: str) -> VoiceSpec
    KNOWN_ENGINES — set of engine names this resolver currently recognizes

Forward-compat: parse_voice_spec accepts unknown engine names (returns
them in VoiceSpec.engine unchanged) so future Voice Backend drivers can
register without needing to update this module first.
"""

from __future__ import annotations

from dataclasses import dataclass


KNOWN_ENGINES: set[str] = {"bark", "kokoro", "cosyvoice", "xtts", "piper"}


@dataclass(frozen=True)
class VoiceSpec:
    engine: str
    preset: str

    def to_str(self) -> str:
        return f"{self.engine}:{self.preset}"


def parse_voice_spec(s: str) -> VoiceSpec:
    """Parse 'engine:preset' string into VoiceSpec.

    Examples:
        'bark:v2/en_speaker_5' -> VoiceSpec(engine='bark', preset='v2/en_speaker_5')
        'kokoro:bm_fable'      -> VoiceSpec(engine='kokoro', preset='bm_fable')

    The preset may itself contain colons (e.g. 'speaker_3:variant_a'); only
    the FIRST colon delimits engine from preset.

    Raises ValueError if:
      - The string does not contain a colon
      - The engine portion is empty
      - The preset portion is empty
    """
    if not s or ":" not in s:
        raise ValueError(
            f"voice spec {s!r} must be 'engine:preset' form (contain ':')"
        )
    engine, _, preset = s.partition(":")
    engine = engine.strip().lower()
    preset = preset.strip()
    if not engine:
        raise ValueError(f"voice spec {s!r} has empty engine portion")
    if not preset:
        raise ValueError(f"voice spec {s!r} has empty preset portion")
    # Forward-compat: unknown engines pass through. Voice Backend Abstraction
    # is where engine validation will live (each backend driver registers).
    return VoiceSpec(engine=engine, preset=preset)
