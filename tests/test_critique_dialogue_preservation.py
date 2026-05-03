"""BUG-LOCAL-027 regression: critique/revision must not strip dialogue.

Three failure modes pinned by these tests:

1. The character-line counter (``_count_character_lines``) must parse the
   writer's actual ``[N] CHARNAME: text`` numbered-bracket format. Prior
   to the 2026-05-03 fix the regex required the line to START with the
   uppercase name, so ``[12] FLETCHER WELLS: ...`` returned ``{}`` and
   the per-character preservation gate became a no-op.

2. The total-collapse hard gate (added 2026-05-03) must reject a
   revision that wipes character dialogue across the board, even when
   no single character drops below the per-character floor (because
   they were ALL wiped at once -- the per-character loop iterates the
   draft dict and silently exits if the parser already returned ``{}``).

3. The revision prompt (passed verbatim to ``str.format``) must be
   safe -- no unescaped ``{`` braces in the prose body. BUG-LOCAL-026
   lesson: prose containing literal ``{}`` crashes ``.format()`` with
   ``IndexError``. The new ABSOLUTE REQUIREMENT clause must pass a
   format-safety smoke.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Match other tests' import pattern -- repo root on path so
# ``import nodes.story_orchestrator`` works.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# CUDA mask to mirror conftest behavior in case this file is run alone.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


# ---------------------------------------------------------------------------
# (1) Parser tests -- _count_character_lines must handle [N] CHARNAME: format
# ---------------------------------------------------------------------------

def _load_count_fn():
    """Load the staticmethod under test. Heavy module import is fine here
    because conftest masks CUDA so torch import won't try to bind GPU."""
    from nodes import story_orchestrator as so  # noqa: WPS433
    return so.LLMScriptWriter._count_character_lines


def test_parser_handles_bare_charname_format():
    """Plain ``CHARNAME: text`` -- the original supported format."""
    count = _load_count_fn()
    text = """=== SCENE 1 ===
ENV: A dim lab.
ALICE: Hello there.
BOB: Hello back.
ALICE: How are you?
"""
    counts = count(text)
    assert counts.get("ALICE") == 2, f"expected ALICE=2, got {counts}"
    assert counts.get("BOB") == 1, f"expected BOB=1, got {counts}"
    # Structural tokens must NEVER be counted as characters.
    assert "SCENE" not in counts
    assert "ENV" not in counts


def test_parser_handles_bracketed_number_prefix():
    """``[N] CHARNAME: text`` -- the actual writer output format.

    BUG-LOCAL-027 root cause: this format silently parsed to {} because
    the original regex required ``\\s*`` (only whitespace) before the
    uppercase name. The fix added an optional ``(?:\\[\\d+\\]\\s+)?``
    non-capturing group so the bracketed prefix is tolerated.
    """
    count = _load_count_fn()
    text = """=== SCENE 1 ===
ENV: A dim lab.
[1] ALICE: Hello there.
[2] BOB: Hello back.
[3] ALICE: How are you?
[15] FLETCHER WELLS: Risk is mounting fast.
"""
    counts = count(text)
    assert counts.get("ALICE") == 2, f"expected ALICE=2, got {counts}"
    assert counts.get("BOB") == 1, f"expected BOB=1, got {counts}"
    assert counts.get("FLETCHER WELLS") == 1, (
        f"expected FLETCHER WELLS=1 (multi-word name with bracket prefix), "
        f"got {counts}"
    )


def test_parser_mixed_format_in_same_text():
    """Real revision passes can mix both formats. Both must count."""
    count = _load_count_fn()
    text = """=== SCENE 1 ===
ALICE: Bare format here.
[2] BOB: Bracketed format here.
ALICE: Bare again.
[5] BOB: Bracketed again.
"""
    counts = count(text)
    assert counts.get("ALICE") == 2
    assert counts.get("BOB") == 2


def test_parser_excludes_structural_tokens():
    """SFX/ENV/SCENE/ACT/MUSIC/etc must not be counted as characters."""
    count = _load_count_fn()
    text = """TITLE: My Episode
=== SCENE 1 ===
ENV: dark room
SFX: thunder
[1] ALICE: real dialogue
MUSIC: theme rises
[2] BOB: more dialogue
ACT 2:
PAUSE: brief silence
NARRATOR: omniscient prose
"""
    counts = count(text)
    assert counts == {"ALICE": 1, "BOB": 1}, (
        f"only ALICE+BOB should count, got {counts}"
    )


def test_parser_empty_text_returns_empty_dict():
    count = _load_count_fn()
    assert count("") == {}
    assert count(None) == {}  # type: ignore[arg-type]


def test_parser_announcer_counted_as_character():
    """ANNOUNCER is a real voice (Kokoro) and must count as a character."""
    count = _load_count_fn()
    text = """[1] ANNOUNCER: Tonight's broadcast comes from deep space.
[2] ALICE: Receiving signal.
[3] ANNOUNCER: That concludes our broadcast.
"""
    counts = count(text)
    assert counts.get("ANNOUNCER") == 2
    assert counts.get("ALICE") == 1


# ---------------------------------------------------------------------------
# (2) Total-collapse hard gate logic test
#
# We can't easily invoke the full _critique_and_revise (it calls the LLM),
# but we CAN exercise the core decision logic by directly verifying the
# math the gate uses. The gate computes
#   draft_total = sum(draft_char_counts.values())
#   revised_total = sum(revised_char_counts.values())
#   if draft_total >= 3 and revised_total < ceil(draft_total * 0.5): reject
# ---------------------------------------------------------------------------

import math


def _gate_fires(draft_counts, revised_counts):
    """Pure-python mirror of the BUG-LOCAL-027 hard gate.

    Returns True if the gate would reject the revision (i.e. fall back
    to draft), False if it would accept.
    """
    draft_total = sum(draft_counts.values())
    revised_total = sum(revised_counts.values())
    if draft_total < 3:
        return False  # below threshold to apply ratio
    min_revised = max(1, math.ceil(draft_total * 0.5))
    return revised_total < min_revised


def test_gate_rejects_total_dialogue_wipe():
    """The exact failure mode from BUG-LOCAL-027 soak runs."""
    draft = {"ANNOUNCER": 2, "FLETCHER WELLS": 8, "KENJI BERNARD": 8}  # 18 lines
    revised = {}  # zero -- the bug
    assert _gate_fires(draft, revised), (
        "gate must REJECT revision that drops 18 lines to 0"
    )


def test_gate_rejects_announcer_only_revision():
    """Revision keeps narrator but wipes character dialogue."""
    draft = {"ANNOUNCER": 2, "ALICE": 5, "BOB": 5}  # 12 lines
    revised = {"ANNOUNCER": 2}  # 2 lines (90.4% sim, 94% len -- looks fine on
                                # surface metrics, the actual L46713 case)
    assert _gate_fires(draft, revised), (
        "gate must REJECT revision that keeps only ANNOUNCER"
    )


def test_gate_accepts_minor_dialogue_trim():
    """Trimming a few lines for pacing is fine; gate must NOT over-fire."""
    draft = {"ALICE": 6, "BOB": 6}  # 12 lines
    revised = {"ALICE": 5, "BOB": 5}  # 10 lines (83% retained -- valid revision)
    assert not _gate_fires(draft, revised), (
        "gate must ACCEPT revision that retains 83%% of dialogue"
    )


def test_gate_accepts_at_threshold():
    """Revision exactly at 50% should pass (>= min_revised, not strict <)."""
    draft = {"ALICE": 4, "BOB": 4}  # 8 lines
    revised = {"ALICE": 2, "BOB": 2}  # 4 lines == ceil(8 * 0.5) = 4
    assert not _gate_fires(draft, revised), (
        "revision exactly at 50%% must pass (>= min_revised = 4)"
    )


def test_gate_skips_short_drafts():
    """Drafts with < 3 character lines are too short to apply ratio."""
    draft = {"ALICE": 2}
    revised = {}  # full wipe but draft below threshold
    assert not _gate_fires(draft, revised), (
        "draft_total < 3 -- gate must defer to per-character floor instead"
    )


def test_gate_handles_empty_dicts_safely():
    """Both empty -- gate must not raise."""
    assert not _gate_fires({}, {})


# ---------------------------------------------------------------------------
# (3) Format-safety smoke for the revision prompt (BUG-026 lesson)
# ---------------------------------------------------------------------------

def test_revision_prompt_no_unescaped_braces_in_added_clause():
    """The new ABSOLUTE REQUIREMENT clause must not introduce ``{}`` literals.

    BUG-LOCAL-026 was a 10-minute pipeline crash caused by literal ``{}``
    in DIRECTOR_PROMPT prose -- ``str.format()`` interpreted them as
    positional arg slots. Any new clause added to a ``.format()`` template
    must pass this gate.

    We extract the relevant block from the source file and assert no
    bare ``{`` outside the legitimate f-string placeholders. The check
    is intentionally lenient: it allows ``{style.replace(...)}`` and
    ``{target_words}`` and ``{min_line_count_per_character}`` (the f-string
    interpolations in the surrounding prose).
    """
    src_path = _REPO_ROOT / "nodes" / "story_orchestrator.py"
    src = src_path.read_text(encoding="utf-8")

    # Locate the revision_prompt assignment.
    m = re.search(
        r'revision_prompt = f"""(.*?)"""',
        src,
        flags=re.DOTALL,
    )
    assert m is not None, "revision_prompt f-string not found in source"
    body = m.group(1)

    # The legitimate placeholders we expect in this f-string:
    legit_placeholders = {
        "{style.replace(\"_\", \" \")}",
        "{target_words}",
        "{min_line_count_per_character}",
        "{critique_text}",
        "{draft_text}",
    }
    body_stripped = body
    for p in legit_placeholders:
        body_stripped = body_stripped.replace(p, "")

    # After stripping legit placeholders, no ``{`` should remain.
    assert "{" not in body_stripped and "}" not in body_stripped, (
        "BUG-026-class footgun: revision_prompt prose contains unescaped "
        "``{}`` (would crash str.format with IndexError). Escape with "
        "``{{ ... }}`` or remove the braces from prose."
    )


def test_revision_prompt_preserves_dialogue_clause_present():
    """Sanity: the new dialogue-preservation clause is actually in the prompt.

    Guards against a future refactor accidentally deleting the BUG-027
    fix from the prompt template.
    """
    src_path = _REPO_ROOT / "nodes" / "story_orchestrator.py"
    src = src_path.read_text(encoding="utf-8")
    assert "ABSOLUTE REQUIREMENT" in src, (
        "ABSOLUTE REQUIREMENT clause missing from story_orchestrator.py "
        "(BUG-027 prompt-side fix regressed)"
    )
    assert "DIALOGUE MUST SURVIVE THE REVISION" in src, (
        "DIALOGUE MUST SURVIVE THE REVISION header missing from "
        "story_orchestrator.py (BUG-027 prompt-side fix regressed)"
    )


# ---------------------------------------------------------------------------
# (4) ULTRA_SMOKE format normalization (BUG-027 extension, 2026-05-03 EVENING)
# ---------------------------------------------------------------------------

def _load_normalize_fn():
    from nodes import story_orchestrator as so  # noqa: WPS433
    return so.LLMScriptWriter._normalize_voice_format_to_standard


def test_normalize_standalone_voice_prefix_to_charname():
    """`[VOICE: NAME, attr, ...]: text` -> `NAME: text`."""
    norm = _load_normalize_fn()
    src = "[VOICE: KENJI BERNARD, male, 50s, weathered, low]: Code input confirmed."
    out = norm(src)
    assert out.startswith("KENJI BERNARD: Code input confirmed."), (
        f"expected normalized to start with 'KENJI BERNARD: Code input confirmed.', "
        f"got {out!r}"
    )


def test_normalize_voice_with_no_attrs():
    """`[VOICE: NAME]: text` (no comma-attrs) -> `NAME: text`."""
    norm = _load_normalize_fn()
    src = "[VOICE: ANNOUNCER]: Welcome to Signal Lost."
    assert norm(src) == "ANNOUNCER: Welcome to Signal Lost."


def test_normalize_strips_inline_voice_block_from_dialogue():
    """`[N] CHARNAME: [VOICE: ...]: text` keeps `[N] CHARNAME:` and removes inline VOICE."""
    norm = _load_normalize_fn()
    src = "[18] KENJI BERNARD: [VOICE: KENJI BERNARD, male, 50s, weathered, low]: continue"
    out = norm(src)
    # The inline [VOICE: ...] block should be stripped; the [N] CHARNAME: prefix preserved.
    assert "[VOICE:" not in out, f"inline VOICE block not stripped: {out!r}"
    assert "[18]" in out and "KENJI BERNARD" in out, f"prefix lost: {out!r}"
    assert "continue" in out, f"dialogue lost: {out!r}"


def test_normalize_idempotent_on_standard_format():
    """Already-standard text passes through unchanged."""
    norm = _load_normalize_fn()
    src = """=== SCENE 1 ===
ENV: dim lighting
[1] ALICE: Hello there.
[2] BOB: Hello back.
"""
    assert norm(src) == src


def test_normalize_handles_empty_and_none():
    norm = _load_normalize_fn()
    assert norm("") == ""
    assert norm(None) is None


def test_normalize_then_count_recovers_dialogue_for_ultra_smoke():
    """End-to-end: normalize + count yields correct character counts on a
    realistic ULTRA_SMOKE-style draft. This is the actual failure mode from
    the 2026-05-03 EVENING soak (otr_runtime.log L47868)."""
    norm = _load_normalize_fn()
    count = _load_count_fn()
    # Draft mixing standalone VOICE prefixes + standard format
    draft = """=== SCENE 1 ===
ENV: tunnel ambience
[VOICE: ANNOUNCER, female, 40s, warm broadcast]: Tonight's broadcast comes from deep beneath.
[VOICE: REN KANE, male, 30s, urgent]: Did ya check the relay logs yet?
[VOICE: PETER ECKELS, male, 40s, scientist, calm]: Logs useless now.
[VOICE: ANNOUNCER, female, 40s, warm broadcast]: Listen for truth in the noise tonight.
"""
    normalized = norm(draft)
    counts = count(normalized)
    assert counts.get("ANNOUNCER") == 2, f"ANNOUNCER expected 2 got {counts}"
    assert counts.get("REN KANE") == 1, f"REN KANE expected 1 got {counts}"
    assert counts.get("PETER ECKELS") == 1, f"PETER ECKELS expected 1 got {counts}"


def test_critique_calls_normalize_at_function_entry():
    """Static check: _critique_and_revise body calls _normalize_voice_format_to_standard
    before the critique pass runs, on BOTH draft_text AND revised_text."""
    src_path = _REPO_ROOT / "nodes" / "story_orchestrator.py"
    src = src_path.read_text(encoding="utf-8")
    # Locate the _critique_and_revise function body.
    fn_match = re.search(
        r"def _critique_and_revise\(self.*?\n(.+?)\n    def ",
        src,
        flags=re.DOTALL,
    )
    assert fn_match is not None, "could not find _critique_and_revise body in source"
    body = fn_match.group(1)
    assert "_normalize_voice_format_to_standard(draft_text)" in body, (
        "_critique_and_revise must call _normalize_voice_format_to_standard "
        "on draft_text at function entry (BUG-027 extension)"
    )
    assert "_normalize_voice_format_to_standard(revised_text)" in body, (
        "_critique_and_revise must call _normalize_voice_format_to_standard "
        "on revised_text after the revision pass (BUG-027 extension)"
    )
