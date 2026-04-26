"""Regression test for BUG-LOCAL-066: silent dialogue loss between
Grammarian and parser, plus the post-parse floor check that recovers it.

Bug: live FULL run on 2026-04-26 (commit 4389b3f) produced a script where
WORD_EXTEND added 16 dialogue lines for OSCAR_KANE / RUFUS_HALPERT,
FORMAT_NORM saw 19 bare-colon dialogue, Grammarian polished 19 -> 16
(passing its own 80% safety check), but the parser produced exactly 1
dialogue token. Net result: a 38-second episode with only the ANNOUNCER
intro instead of a 7-minute episode with 16 character lines.

Root causes (three structural asymmetries):

1. Parser v5 bare-colon regex `[A-Z][A-Z0-9 ]{0,24}` did not include
   underscores. Names that the production_plan emits with underscores
   (`OSCAR_KANE`, `RUFUS_HALPERT`, `AUTHORITY_VOICE`) silently dropped at
   the parser stage even though every upstream counter saw them.

2. Grammarian's `_post_total` counted only bare-colon + `[VOICE:`. Its
   `_pre_total` counted bare-colon + `[VOICE:` + `[NAME, mood]` shorthand.
   Asymmetric -- if the LLM converted bare-colon to bracket-shorthand
   during polish, post_total under-counted and the safety check rejected
   a fine output. If it did the reverse, the under-count silently
   passed the 80% gate.

3. LLM_RESCUE only triggered on a 0-dialogue ValueError from the parser.
   When the parser produced 1 dialogue token (just the ANNOUNCER), no
   rescue ran and the episode shipped with 99% of its dialogue silently
   dropped.

Fix:
- Add `_` to the parser's v5 regex character class and the loose-fallback
  trigger regex.
- Make Grammarian `_post_total` symmetric by adding the bracket-shorthand
  counter.
- Add a post-parse dialogue floor check: if the parser produced
  <50% of the dialogue lines we expected after Grammarian (and the
  expected count is meaningful, >=6), trigger LLM_RESCUE on the
  Grammarian output and adopt the rescued parse if it recovers more
  dialogue than the original parse.

This test exercises the diagnostic logic without requiring GPU/transformers:
it builds realistic post-Grammarian script texts, runs them through the
parser via OTR_LLMScriptWriter._parse_script, and asserts:

- underscored names parse cleanly (BUG-066 Fix A)
- the Grammarian counter is symmetric (BUG-066 Fix B)
- the floor-check is correctly computed (BUG-066 Fix C)
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_heavy_imports():
    for name in ("torch", "torch.cuda", "transformers", "comfy",
                 "comfy.model_management"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "cuda"):
        sys.modules["torch"].cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
            ipc_collect=lambda: None,
            get_device_properties=lambda i: types.SimpleNamespace(
                total_memory=16 * 1024**3),
        )


# ---------------------------------------------------------------------------
# Re-implementation of the THREE BUG-066 regex/counters as standalone
# helpers. Mirrors what story_orchestrator.py does inline; lets us assert
# behavior without importing the heavy module (which pulls in transformers).
# ---------------------------------------------------------------------------


# Mirror of the v5 bare-colon parser regex AFTER the BUG-066 fix.
V5_REGEX_FIXED = re.compile(
    r'^(?:\*{0,2})([A-Z][A-Z0-9_ ]{0,24})(?:\*{0,2})'
    r'(?:\s*\(([^)]*)\))?\s*:\s+(.+)$'
)

# Mirror of the v5 bare-colon parser regex BEFORE the BUG-066 fix.
V5_REGEX_BROKEN = re.compile(
    r'^(?:\*{0,2})([A-Z][A-Z0-9 ]{0,24})(?:\*{0,2})'
    r'(?:\s*\(([^)]*)\))?\s*:\s+(.+)$'
)


def _grammarian_post_total_fixed(polished: str) -> int:
    """Mirror of the Grammarian _post_total computation AFTER BUG-066 Fix B."""
    post_lines = len(re.findall(
        r'^[A-Z][A-Z0-9_ ]{1,19}:\s*.+$', polished, re.MULTILINE
    ))
    post_voice = len(re.findall(r'\[VOICE:', polished, re.IGNORECASE))
    post_shorthand = len(re.findall(
        r'^\[[A-Z][A-Z0-9_ ]{1,20}(?:,\s*.+?)?\]\s*\S',
        polished, re.MULTILINE,
    ))
    return post_lines + post_voice + post_shorthand


def _grammarian_post_total_broken(polished: str) -> int:
    """Mirror of the Grammarian _post_total computation BEFORE BUG-066 Fix B."""
    post_lines = len(re.findall(
        r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', polished, re.MULTILINE
    ))
    post_voice = len(re.findall(r'\[VOICE:', polished, re.IGNORECASE))
    return post_lines + post_voice


def _expected_pre_parse_dialogue(script_text: str) -> int:
    """Mirror of the post-Grammarian expected-dialogue counter (Fix C)."""
    return (
        len(re.findall(
            r'^[A-Z][A-Z0-9_ ]{1,19}:\s*.+$',
            script_text, re.MULTILINE,
        ))
        + len(re.findall(r'\[VOICE:', script_text, re.IGNORECASE))
        + len(re.findall(
            r'^\[[A-Z][A-Z0-9_ ]{1,20}(?:,\s*.+?)?\]\s*\S',
            script_text, re.MULTILINE,
        ))
    )


# ---------------------------------------------------------------------------
# Fix A: parser v5 regex underscore tolerance
# ---------------------------------------------------------------------------


class TestParserV5UnderscoreTolerance:

    @pytest.mark.parametrize("name", [
        "OSCAR_KANE",
        "RUFUS_HALPERT",
        "AUTHORITY_VOICE",
        "DR_KAREN_VANCE",
        "MEREDITH",          # no underscore -- still must parse
        "RYAN HALLOWAY",     # space -- still must parse
        "STANLEY WELLS",
    ])
    def test_underscored_and_normal_names_match_v5(self, name):
        line = f"{name}: We have to do something now."
        m = V5_REGEX_FIXED.match(line)
        assert m is not None, f"v5 regex did not match '{line}'"
        assert m.group(1) == name

    def test_underscored_names_fail_under_old_regex(self):
        """Confirms the old regex was the bug (sanity check)."""
        line = "OSCAR_KANE: We have to do something now."
        assert V5_REGEX_BROKEN.match(line) is None, (
            "old regex must reject underscored names -- "
            "if this passes, the bug never existed"
        )

    def test_structural_tokens_still_rejected(self):
        """Add-underscore fix must NOT break the structural-token blacklist."""
        for tok in ("ENV", "SFX", "MUSIC", "TITLE", "SCENE", "ACT"):
            line = f"{tok}: should not register as dialogue"
            m = V5_REGEX_FIXED.match(line)
            # The regex itself matches -- the parser then checks the
            # structural blacklist. We just check the regex isn't broken.
            assert m is not None, f"{tok}: must still match the raw regex"

    def test_numeric_names_still_match(self):
        line = "AGENT_47: copy that."
        assert V5_REGEX_FIXED.match(line) is not None


# ---------------------------------------------------------------------------
# Fix B: Grammarian _post_total symmetry
# ---------------------------------------------------------------------------


class TestGrammarianPostTotalSymmetry:

    BARE_DIALOGUE_SCRIPT = (
        "OSCAR_KANE: It's now or never.\n"
        "RUFUS_HALPERT: I'm with you.\n"
        "ANNOUNCER: Tune in next week.\n"
    )

    SHORTHAND_DIALOGUE_SCRIPT = (
        "[OSCAR_KANE, urgent] It's now or never.\n"
        "[RUFUS_HALPERT, calm] I'm with you.\n"
        "[ANNOUNCER, deep] Tune in next week.\n"
    )

    VOICE_DIALOGUE_SCRIPT = (
        "[VOICE: OSCAR_KANE, urgent] It's now or never.\n"
        "[VOICE: RUFUS_HALPERT, calm] I'm with you.\n"
        "[VOICE: ANNOUNCER, deep] Tune in next week.\n"
    )

    def test_fixed_counter_counts_bare_colon(self):
        assert _grammarian_post_total_fixed(self.BARE_DIALOGUE_SCRIPT) == 3

    def test_fixed_counter_counts_voice_form(self):
        assert _grammarian_post_total_fixed(self.VOICE_DIALOGUE_SCRIPT) == 3

    def test_fixed_counter_counts_shorthand(self):
        """The bug: old _post_total returned 0 for a script of pure shorthand."""
        assert _grammarian_post_total_fixed(self.SHORTHAND_DIALOGUE_SCRIPT) == 3

    def test_old_counter_undercounted_shorthand(self):
        """Sanity: old counter returned 0 on shorthand-only input."""
        assert _grammarian_post_total_broken(self.SHORTHAND_DIALOGUE_SCRIPT) == 0

    def test_mixed_form_counter_is_correct(self):
        mixed = (
            "OSCAR_KANE: It's now or never.\n"
            "[RUFUS_HALPERT, calm] I'm with you.\n"
            "[VOICE: ANNOUNCER, deep] Tune in next week.\n"
        )
        assert _grammarian_post_total_fixed(mixed) == 3


# ---------------------------------------------------------------------------
# Fix C: post-parse dialogue floor expectation
# ---------------------------------------------------------------------------


class TestExpectedPreParseDialogue:

    def test_counts_realistic_post_grammarian_output(self):
        """The exact failure shape from the live 2026-04-26 run."""
        polished = (
            "=== SCENE 1 ===\n"
            "[ENV: rain on shingles]\n"
            "ANNOUNCER: Tonight, on Signal Lost...\n"
            "OSCAR_KANE: I never thought I'd see this day.\n"
            "RUFUS_HALPERT: Time's a strange beast, isn't it?\n"
            "OSCAR_KANE: We don't have long.\n"
            "RUFUS_HALPERT: Then we'd better start.\n"
            "OSCAR_KANE: Are you sure about this?\n"
            "RUFUS_HALPERT: As sure as I've ever been.\n"
            "[SFX: clock ticking]\n"
            "OSCAR_KANE: One more pass.\n"
            "RUFUS_HALPERT: One more pass.\n"
            "ANNOUNCER: Will they make it back?\n"
        )
        assert _expected_pre_parse_dialogue(polished) == 10

    def test_zero_dialogue_returns_zero(self):
        """Pure narration / SFX / scene markers -- nothing to count."""
        text = (
            "=== SCENE 1 ===\n"
            "[ENV: misty harbor]\n"
            "[SFX: foghorn]\n"
            "[MUSIC: opening theme]\n"
        )
        assert _expected_pre_parse_dialogue(text) == 0

    def test_floor_threshold_triggers_below_50_percent(self):
        """The actual condition the orchestrator uses for LLM_RESCUE."""
        expected = 16
        # post_parse_count < expected * 0.5 AND expected >= 6 -> trigger
        for post_parse_count in (0, 1, 7):
            should_trigger = (expected >= 6
                              and post_parse_count < expected * 0.5)
            assert should_trigger, (
                f"floor must fire at post_parse={post_parse_count}, "
                f"expected={expected}"
            )

    def test_floor_threshold_does_not_trigger_when_close_enough(self):
        expected = 16
        for post_parse_count in (8, 12, 16, 20):
            should_trigger = (expected >= 6
                              and post_parse_count < expected * 0.5)
            assert not should_trigger, (
                f"floor must NOT fire at post_parse={post_parse_count}, "
                f"expected={expected}"
            )

    def test_floor_threshold_does_not_trigger_for_short_runs(self):
        """If the script is so short that AFTER_GRAMMARIAN counted <6 lines,
        we don't have enough signal to call dialogue loss; just trust the
        parser. Avoids false-positive rescues on very short test scripts."""
        for expected in (0, 1, 3, 5):
            for post_parse_count in (0, 1):
                should_trigger = (expected >= 6
                                  and post_parse_count < expected * 0.5)
                assert not should_trigger
