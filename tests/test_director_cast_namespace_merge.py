"""Regression test for BUG-LOCAL-068: Director cast namespace duplicates.

Bug: ScriptWriter cast rows use script-side spaced names ("KAEL VAUGHN") while
the Director's voice_assignments dict keys carry the underscored variant
("KAEL_VAUGHN"). The Director-stage cast-merge loop did a naive equality
lookup that failed across the underscore/space gap, so it APPENDED a
duplicate row for every Director-known character. Voice presets attached
to the new (underscored) rows; dialogue stayed on the original (spaced)
rows; Bark fell back to default voices.

Live evidence (2026-04-26 PM, "Signal Abyss" Captain-Eris-Violet trial):

    cast (5 entries instead of 3):
      c01  ANNOUNCER       lines=3  words=70  voice=v2/en_speaker_0  ✓
      c02  KAEL VAUGHN     lines=1  words=18  voice=None             ✗
      c03  REI HOWARD      lines=1  words=15  voice=None             ✗
      c04  REI_HOWARD      lines=0  words=0   voice=v2/en_speaker_4  ✗
      c05  KAEL_VAUGHN     lines=0  words=0   voice=v2/en_speaker_3  ✗

Fix: normalize both sides via `_norm_name_key()` (uppercase, strip,
underscore -> space) BEFORE the dict lookup. New rows get the spaced
form so the ledger doesn't carry mixed entries that would re-trigger
the same bug for downstream consumers.

This test exercises the cast-merge logic without requiring transformers /
torch / comfy by re-implementing the merge as a standalone helper that
mirrors the production code's algorithm.
"""

from __future__ import annotations


def _norm_name_key(s: str) -> str:
    """Mirror of the namespace-normaliser added in BUG-066 fix."""
    return (s or "").strip().upper().replace("_", " ")


def _merge_voice_assignments(existing_cast, voice_assignments):
    """Standalone re-implementation of the cast-merge algorithm.

    Mirrors the post-BUG-068 production code in
    nodes/story_orchestrator.py around line 7846.
    """
    existing_cast = list(existing_cast or [])
    name_to_row = {_norm_name_key(r.get("name")): r for r in existing_cast}
    for char_name, entry in (voice_assignments or {}).items():
        if not isinstance(entry, dict):
            continue
        key = _norm_name_key(char_name)
        row = name_to_row.get(key)
        if row is None:
            row = {"char_id": f"c{len(existing_cast)+1:02d}", "name": key}
            existing_cast.append(row)
            name_to_row[key] = row
        row["voice_preset"] = entry.get("voice_preset") or row.get("voice_preset")
        if entry.get("gender"):
            row["gender"] = entry["gender"]
        if entry.get("notes"):
            row["description"] = entry["notes"]
    return existing_cast


# ---------------------------------------------------------------------------
# The exact failure shape from the live 2026-04-26 run
# ---------------------------------------------------------------------------


class TestSpacedAndUnderscoredMerge:

    def _signal_abyss_existing_cast(self):
        return [
            {"char_id": "c01", "name": "ANNOUNCER",
             "voice_preset": "v2/en_speaker_0"},
            {"char_id": "c02", "name": "KAEL VAUGHN", "voice_preset": None},
            {"char_id": "c03", "name": "REI HOWARD", "voice_preset": None},
        ]

    def _signal_abyss_voice_assignments(self):
        return {
            "ANNOUNCER":     {"voice_preset": "v2/en_speaker_0",
                              "notes": "Female, mature, authoritative"},
            "KAEL_VAUGHN":   {"voice_preset": "v2/en_speaker_3",
                              "notes": "male, 30s, intense"},
            "REI_HOWARD":    {"voice_preset": "v2/en_speaker_4",
                              "notes": "female, 30s, urgent"},
        }

    def test_no_duplicate_rows_after_merge(self):
        cast = self._signal_abyss_existing_cast()
        merged = _merge_voice_assignments(
            cast, self._signal_abyss_voice_assignments()
        )
        # 3 in -> 3 out (no underscored duplicates appended)
        assert len(merged) == 3, (
            f"expected 3 cast rows after merge, got {len(merged)}: "
            f"{[r['name'] for r in merged]}"
        )

    def test_voice_preset_lands_on_speaking_row(self):
        cast = self._signal_abyss_existing_cast()
        merged = _merge_voice_assignments(
            cast, self._signal_abyss_voice_assignments()
        )
        by_name = {r["name"]: r for r in merged}
        assert "KAEL VAUGHN" in by_name, "spaced KAEL VAUGHN must survive"
        assert by_name["KAEL VAUGHN"]["voice_preset"] == "v2/en_speaker_3"
        assert "REI HOWARD" in by_name, "spaced REI HOWARD must survive"
        assert by_name["REI HOWARD"]["voice_preset"] == "v2/en_speaker_4"

    def test_no_underscored_names_survive(self):
        cast = self._signal_abyss_existing_cast()
        merged = _merge_voice_assignments(
            cast, self._signal_abyss_voice_assignments()
        )
        names = [r["name"] for r in merged]
        for n in names:
            assert "_" not in n, (
                f"normalised cast must not contain underscored names; "
                f"found {n!r} in {names}"
            )


class TestNewCharactersFromDirectorOnly:
    """When the Director adds a brand-new character (e.g. atmospheric
    voice that the script never named), it should land as a fresh row
    using the spaced canonical form, not underscored."""

    def test_new_underscored_character_becomes_spaced_row(self):
        existing = [
            {"char_id": "c01", "name": "ANNOUNCER",
             "voice_preset": "v2/en_speaker_0"},
        ]
        voice_assignments = {
            "AUTHORITY_VOICE": {"voice_preset": "v2/en_speaker_5",
                                "notes": "male, deep, official"},
        }
        merged = _merge_voice_assignments(existing, voice_assignments)
        names = [r["name"] for r in merged]
        # New character merged in
        assert "AUTHORITY VOICE" in names
        # Spaced form, not underscored
        assert "AUTHORITY_VOICE" not in names


class TestSymmetricNormalization:
    """Whichever side has underscores must merge with whichever side has spaces."""

    def test_existing_underscored_voice_assignment_spaced(self):
        existing = [
            {"char_id": "c01", "name": "DR_KAREN_VANCE", "voice_preset": None},
        ]
        voice_assignments = {
            "DR KAREN VANCE": {"voice_preset": "v2/en_speaker_2",
                               "notes": "female, 50s, clinical"},
        }
        merged = _merge_voice_assignments(existing, voice_assignments)
        assert len(merged) == 1
        assert merged[0]["voice_preset"] == "v2/en_speaker_2"

    def test_existing_spaced_voice_assignment_underscored(self):
        existing = [
            {"char_id": "c01", "name": "MEREDITH SAGE", "voice_preset": None},
        ]
        voice_assignments = {
            "MEREDITH_SAGE": {"voice_preset": "v2/en_speaker_9",
                              "notes": "female, 40s"},
        }
        merged = _merge_voice_assignments(existing, voice_assignments)
        assert len(merged) == 1
        assert merged[0]["voice_preset"] == "v2/en_speaker_9"
        assert merged[0]["name"] == "MEREDITH SAGE"  # spaced form preserved


class TestNormaliserBoundaryCases:

    def test_empty_input_is_safe(self):
        assert _norm_name_key("") == ""
        assert _norm_name_key(None) == ""
        assert _norm_name_key("   ") == ""

    def test_already_normalised_is_idempotent(self):
        assert _norm_name_key("KAEL VAUGHN") == "KAEL VAUGHN"

    def test_lowercase_and_mixed_case_uppercased(self):
        assert _norm_name_key("kael vaughn") == "KAEL VAUGHN"
        assert _norm_name_key("Kael_Vaughn") == "KAEL VAUGHN"

    def test_multiple_underscores_collapse_to_spaces(self):
        assert _norm_name_key("DR_KAREN_VANCE") == "DR KAREN VANCE"

    def test_leading_trailing_whitespace_stripped(self):
        assert _norm_name_key("  KAEL VAUGHN  ") == "KAEL VAUGHN"
