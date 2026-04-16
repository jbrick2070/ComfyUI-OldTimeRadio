"""
test_p0_features.py
====================
Offline tests for the two P0 features:
  P0 #1: min_line_count_per_character (self-critique guard)
  P0 #2: Director JSON schema validator

These tests extract the logic directly from story_orchestrator.py so they
can run without the full ComfyUI import chain.

Run:
    python -m pytest tests/test_p0_features.py -v
"""

import re
import logging
import pytest

# ---------------------------------------------------------------------------
# Extracted logic: _count_character_lines (from LLMScriptWriter)
# ---------------------------------------------------------------------------
def _count_character_lines(text):
    """Count dialogue lines per character in script text.

    Exact copy of LLMScriptWriter._count_character_lines logic from
    nodes/story_orchestrator.py so we can test without ComfyUI imports.
    """
    if not text:
        return {}

    _struct_exclude = frozenset([
        "TITLE", "SCENE", "ACT", "SFX", "ENV", "MUSIC", "BEAT",
        "PAUSE", "NARRATOR", "SYSTEM_SENTINEL"
    ])

    pattern = r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'

    character_counts = {}
    for line in text.split('\n'):
        match = re.match(pattern, line)
        if match:
            char_name = match.group(1).strip()
            if char_name not in _struct_exclude:
                character_counts[char_name] = character_counts.get(char_name, 0) + 1

    return character_counts


# ---------------------------------------------------------------------------
# Extracted logic: _validate_director_plan (from LLMDirector)
# ---------------------------------------------------------------------------
_DIRECTOR_SCHEMA = {
    "required_keys": {
        "voice_assignments": dict,
        "sfx_plan": list,
        "music_plan": list,
    },
    "optional_keys": {
        "episode_title": str,
        "pacing": dict,
        "visual_plan": dict,
    },
    "voice_assignment_required": {
        "voice_preset": str,
    },
    "sfx_entry_required": {
        "cue_id": str,
        "generation_prompt": str,
    },
    "music_entry_required": {
        "cue_id": str,
        "duration_sec": (int, float),
        "generation_prompt": str,
    },
    "music_cue_ids": {"opening", "closing", "interstitial"},
}

log = logging.getLogger("otr.test_p0")


def _validate_director_plan(plan):
    """Validate and repair the Director's JSON production plan.

    Exact copy of LLMDirector._validate_director_plan logic from
    nodes/story_orchestrator.py so we can test without ComfyUI imports.
    """
    if not isinstance(plan, dict):
        plan = {}

    # --- Part 1: Check and add required keys with defaults ---
    for key, expected_type in _DIRECTOR_SCHEMA["required_keys"].items():
        if key not in plan:
            if expected_type == dict:
                plan[key] = {}
            elif expected_type == list:
                plan[key] = []
        elif not isinstance(plan[key], expected_type):
            if expected_type == dict:
                plan[key] = {}
            elif expected_type == list:
                plan[key] = []

    # --- Part 2: Validate voice_assignments ---
    voice_assignments = plan.get("voice_assignments", {})
    if isinstance(voice_assignments, dict):
        for char_name, char_data in list(voice_assignments.items()):
            if not isinstance(char_data, dict):
                voice_assignments[char_name] = {}
                char_data = voice_assignments[char_name]

            if "voice_preset" not in char_data:
                fallback = "v2/en_speaker_0"
                char_data["voice_preset"] = fallback
            elif not isinstance(char_data["voice_preset"], str):
                fallback = "v2/en_speaker_0"
                char_data["voice_preset"] = fallback

    # --- Part 3: Validate SFX plan ---
    sfx_plan = plan.get("sfx_plan", [])
    if not isinstance(sfx_plan, list):
        plan["sfx_plan"] = []
        sfx_plan = []
    else:
        valid_sfx = []
        for i, sfx_entry in enumerate(sfx_plan):
            if not isinstance(sfx_entry, dict):
                continue
            if "cue_id" not in sfx_entry or "generation_prompt" not in sfx_entry:
                continue
            valid_sfx.append(sfx_entry)
        plan["sfx_plan"] = valid_sfx

    # --- Part 4: Validate and synthesize music_plan ---
    music_plan = plan.get("music_plan", [])
    if not isinstance(music_plan, list):
        plan["music_plan"] = []
        music_plan = []

    music_dict = {entry.get("cue_id"): entry for entry in music_plan
                  if isinstance(entry, dict) and "cue_id" in entry}

    music_defaults = {
        "opening": {
            "cue_id": "opening",
            "duration_sec": 12,
            "generation_prompt": "1940s old time radio opening theme, warm brass fanfare, upright bass, snare brushes, mono AM radio character, tube saturation, confident and mysterious, ends on a held chord"
        },
        "closing": {
            "cue_id": "closing",
            "duration_sec": 8,
            "generation_prompt": "1940s old time radio closing sting, brass and strings, resolving cadence, warm tube saturation, fades to silence"
        },
        "interstitial": {
            "cue_id": "interstitial",
            "duration_sec": 4,
            "generation_prompt": "short old time radio act-break stinger, single brass hit with cymbal swell, mono, tube warmth"
        }
    }

    for cue_id in _DIRECTOR_SCHEMA["music_cue_ids"]:
        if cue_id not in music_dict:
            plan["music_plan"].append(music_defaults[cue_id].copy())
        else:
            cue_entry = music_dict[cue_id]
            duration = cue_entry.get("duration_sec")
            if not isinstance(duration, (int, float)) or duration <= 0:
                default_duration = music_defaults[cue_id]["duration_sec"]
                cue_entry["duration_sec"] = default_duration

            if "generation_prompt" not in cue_entry or not isinstance(cue_entry.get("generation_prompt"), str):
                default_prompt = music_defaults[cue_id]["generation_prompt"]
                cue_entry["generation_prompt"] = default_prompt

    return plan


# ===========================================================================
# P0 #1: _count_character_lines
# ===========================================================================
class TestCountCharacterLines:
    """Tests for the character line counting helper."""

    def test_basic_dialogue(self):
        text = (
            "ALICE: Hello there.\n"
            "BOB: Hi Alice.\n"
            "ALICE: How are you?\n"
        )
        counts = _count_character_lines(text)
        assert counts["ALICE"] == 2
        assert counts["BOB"] == 1

    def test_excludes_structural_tokens(self):
        text = (
            "TITLE: The Last Frequency\n"
            "SCENE: A dark room\n"
            "SFX: Thunder\n"
            "ALICE: I hear thunder.\n"
            "ENV: Rainy night\n"
            "MUSIC: Opening theme\n"
        )
        counts = _count_character_lines(text)
        assert "TITLE" not in counts
        assert "SCENE" not in counts
        assert "SFX" not in counts
        assert "ENV" not in counts
        assert "MUSIC" not in counts
        assert counts.get("ALICE", 0) == 1

    def test_announcer_counted(self):
        text = (
            "ANNOUNCER: Welcome to Signal Lost.\n"
            "ANNOUNCER: And now, the story.\n"
        )
        counts = _count_character_lines(text)
        assert counts["ANNOUNCER"] == 2

    def test_multiword_name(self):
        text = (
            "DOCTOR CHEN: The readings are off.\n"
            "DOCTOR CHEN: Run it again.\n"
            "OFFICER HALL: Copy that.\n"
        )
        counts = _count_character_lines(text)
        assert counts["DOCTOR CHEN"] == 2
        assert counts["OFFICER HALL"] == 1

    def test_empty_text(self):
        counts = _count_character_lines("")
        assert counts == {}

    def test_no_dialogue(self):
        text = "Just some narration with no character lines."
        counts = _count_character_lines(text)
        assert counts == {}

    def test_asterisk_wrapped_name(self):
        text = (
            "**ALICE**: Something dramatic.\n"
            "*BOB*: A reply.\n"
        )
        counts = _count_character_lines(text)
        assert counts.get("ALICE", 0) == 1
        assert counts.get("BOB", 0) == 1

    def test_emotion_parenthetical(self):
        text = (
            "ALICE (whispering): Be quiet.\n"
            "BOB (angry): No!\n"
            "ALICE: Fine.\n"
        )
        counts = _count_character_lines(text)
        assert counts["ALICE"] == 2
        assert counts["BOB"] == 1


# ===========================================================================
# P0 #2: _validate_director_plan
# ===========================================================================
class TestValidateDirectorPlan:
    """Tests for the Director JSON schema validator."""

    def _good_plan(self):
        return {
            "episode_title": "Test Episode",
            "voice_assignments": {
                "ALICE": {"voice_preset": "v2/en_speaker_2", "notes": "Female"},
                "BOB": {"voice_preset": "v2/en_speaker_1", "notes": "Male"},
            },
            "sfx_plan": [
                {"cue_id": "sfx_001", "type": "sfx",
                 "description": "Thunder", "generation_prompt": "Distant thunder rolling"},
            ],
            "music_plan": [
                {"cue_id": "opening", "duration_sec": 12,
                 "generation_prompt": "1940s brass opening fanfare"},
                {"cue_id": "closing", "duration_sec": 8,
                 "generation_prompt": "1940s closing sting"},
                {"cue_id": "interstitial", "duration_sec": 4,
                 "generation_prompt": "Short brass hit"},
            ],
        }

    def test_valid_plan_passes_through(self):
        plan = self._good_plan()
        result = _validate_director_plan(plan)
        assert result["voice_assignments"]["ALICE"]["voice_preset"] == "v2/en_speaker_2"
        assert len(result["music_plan"]) == 3
        assert len(result["sfx_plan"]) == 1

    def test_missing_voice_assignments_gets_default(self):
        plan = {"sfx_plan": [], "music_plan": []}
        result = _validate_director_plan(plan)
        assert "voice_assignments" in result
        assert isinstance(result["voice_assignments"], dict)

    def test_missing_sfx_plan_gets_default(self):
        plan = {"voice_assignments": {}, "music_plan": []}
        result = _validate_director_plan(plan)
        assert "sfx_plan" in result
        assert isinstance(result["sfx_plan"], list)

    def test_missing_music_plan_gets_synthesized(self):
        plan = {"voice_assignments": {}, "sfx_plan": []}
        result = _validate_director_plan(plan)
        assert "music_plan" in result
        cue_ids = {c["cue_id"] for c in result["music_plan"]}
        assert "opening" in cue_ids
        assert "closing" in cue_ids
        assert "interstitial" in cue_ids

    def test_missing_music_cue_gets_synthesized(self):
        plan = self._good_plan()
        plan["music_plan"] = [c for c in plan["music_plan"] if c["cue_id"] != "closing"]
        result = _validate_director_plan(plan)
        cue_ids = {c["cue_id"] for c in result["music_plan"]}
        assert "closing" in cue_ids

    def test_broken_sfx_entries_filtered(self):
        plan = self._good_plan()
        plan["sfx_plan"].append({"cue_id": "bad_sfx"})  # missing generation_prompt
        result = _validate_director_plan(plan)
        assert len(result["sfx_plan"]) == 1

    def test_missing_voice_preset_gets_fallback(self):
        plan = self._good_plan()
        plan["voice_assignments"]["ALICE"] = {"notes": "Female, no preset"}
        result = _validate_director_plan(plan)
        assert "voice_preset" in result["voice_assignments"]["ALICE"]
        assert isinstance(result["voice_assignments"]["ALICE"]["voice_preset"], str)

    def test_none_input_returns_dict(self):
        result = _validate_director_plan(None)
        assert isinstance(result, dict)
        assert "voice_assignments" in result
        assert "sfx_plan" in result
        assert "music_plan" in result

    def test_empty_dict_input(self):
        result = _validate_director_plan({})
        assert "voice_assignments" in result
        assert "sfx_plan" in result
        assert "music_plan" in result

    def test_invalid_duration_sec_clamped(self):
        plan = self._good_plan()
        for cue in plan["music_plan"]:
            cue["duration_sec"] = -5
        result = _validate_director_plan(plan)
        for cue in result["music_plan"]:
            assert cue["duration_sec"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
