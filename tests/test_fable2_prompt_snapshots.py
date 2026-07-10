"""scifi_fable2 seam-prompt snapshots -- shape assertions (S0, doc s10).

Not byte-golden files: SHAPE pins on the load-bearing contract lines of each
of the 9 seams, robust to whitespace rewrap (assertions run on
whitespace-collapsed text). Both pitch modes (1 and 3) are pinned, the
critic problem classes are pinned as a SUBSET of the audit classes (r4
anchor), and the FORMAT block is pinned byte-identical between the script
and revision seams (grammar single source of truth, doc s4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as ROUTING  # noqa: E402

_CRITIC_CLASSES = (
    "register_bleed", "on_the_nose", "stakes_sag", "ending_unearned",
    "continuity_break", "cast_unused", "announcer_contract", "word_budget",
    "subtext_flat",
)
_AUDIT_ONLY_CLASSES = (
    "speaker_not_in_cast", "verbatim_break", "skeleton_break",
    "news_source_framing", "machine_attribution", "weapons_smoking",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def stages():
    ROUTING._REGISTRY = None
    try:
        return dict(ROUTING.resolve_story_pack("scifi_fable2").prompt_stages)
    finally:
        ROUTING._REGISTRY = None


class TestSeamShapes:
    def test_json_seams_declare_json_only_output(self, stages):
        for seam in ("fable2_dossier_system", "fable2_pitch_system",
                     "fable2_select_system", "fable2_treatment_system",
                     "fable2_critic_system", "fable2_casting_system",
                     "fable2_audit_system"):
            assert "Return one JSON object only" in _flat(stages[seam]), seam

    def test_dossier_contract(self, stages):
        flat = _flat(stages["fable2_dossier_system"])
        for needle in ("facts_to_keep", "allowed_numbers", "named_entities",
                       "dramatizable_vectors", "Copy, never invent"):
            assert needle in flat

    def test_pitch_seam_carries_both_modes(self, stages):
        # r3/M5: the pitch prompt is parameterized for 1-or-3 mode; the
        # low-budget single pitch carries no divergence duty.
        flat = _flat(stages["fable2_pitch_system"])
        assert "(1 or 3 -- the count and the cards are in the user message)" \
            in flat
        assert "array of EXACTLY the requested count" in flat
        assert "When THREE pitches are requested" in flat
        assert ("A single requested pitch carries no divergence duty."
                in flat)
        assert "pitch i uses card i" in flat

    def test_select_seam(self, stages):
        flat = _flat(stages["fable2_select_system"])
        assert "chosen_pitch_id" in flat
        assert "It is never spoken on air." in flat
        assert "Judge; do not rewrite any pitch." in flat

    def test_treatment_contract(self, stages):
        flat = _flat(stages["fable2_treatment_system"])
        for needle in (
                "dramatic_question", "cast_shapes", "priced_ending",
                "news_thread", "news_close_read",
                'never "ANNOUNCER"',
                "these exact names are the ONLY legal speakers in the",
                "priced_ending is required for ALL FOUR shapes",
                "SFW. No weapons, no smoking."):
            assert needle in flat
        # S1b read-split: the treatment no longer writes the factual
        # close -- the seam says so explicitly.
        assert "do NOT write it here" in flat

    def test_news_read_seam_contract(self, stages):
        # P2c (S1b read-split, kibitz r2 Q1): single-purpose factual
        # close under the anti-invention subset laws.
        flat = _flat(stages["fable2_news_read_system"])
        for needle in (
                "news_close_read", "Copy, never invent",
                "allowed_numbers",
                "NEVER name the drama's fictional characters",
                "FORBIDDEN NAMES",
                "no computed totals or averages"):
            assert needle in flat

    def test_script_seam_format_block(self, stages):
        flat = _flat(stages["fable2_script_system"])
        for needle in (
                "FORMAT -- every non-blank line of your output must match "
                "EXACTLY ONE shape:",
                "TITLE: <episode title>",
                "SCENE <n>: <one-line concrete setting>",
                "CODA: <one pivot clause ending with a colon>",
                "REQUIRED SKELETON, in order:",
                "HARD BANS: no parentheses ( ) or brackets [ ] anywhere",
                "nothing after END."):
            assert needle in flat

    def test_script_seam_writing_rules(self, stages):
        flat = _flat(stages["fable2_script_system"])
        for needle in (
                "plus or minus 30% per scene",
                "CODA: at most 16 words, ends with a colon",
                "never mention machines writing stories",
                "Start with TITLE: and stop after END."):
            assert needle in flat

    def test_script_example_is_format_only(self, stages):
        # r1 anchor: the format example must carry its imitation guard.
        flat = _flat(stages["fable2_script_system"])
        assert "TITLE: The Long Count" in flat
        assert ("the example shows FORMAT ONLY -- never imitate its plot, "
                "names, or imagery" in flat)

    def test_format_block_identical_in_script_and_revision(self, stages):
        # Grammar single source of truth (doc s4): one FORMAT block, twice.
        def block(seam):
            text = stages[seam]
            start = text.index("FORMAT -- every non-blank line")
            end = text.index("nothing after END.") + len("nothing after END.")
            return text[start:end]
        assert block("fable2_script_system") == \
            block("fable2_revision_system")

    def test_revision_law(self, stages):
        flat = _flat(stages["fable2_revision_system"])
        for needle in (
                "Output the ENTIRE episode again, top to bottom -- not a "
                "diff.",
                "any scene with no note against it is copied VERBATIM",
                "Keep the same TITLE, CAST, scene count and order",
                "Apply EVERY note."):
            assert needle in flat

    def test_critic_is_judge_not_writer(self, stages):
        flat = _flat(stages["fable2_critic_system"])
        assert "NEVER write replacement dialogue" in flat
        assert '"ship" with an empty notes array is a legitimate verdict' \
            in flat
        assert "word_budget is for SCENE-LEVEL pacing only" in flat
        for cls in _CRITIC_CLASSES:
            assert cls in flat, f"critic class {cls} missing"

    def test_critic_classes_are_a_subset_of_audit_classes(self, stages):
        # r4 anchor snapshot: every critic problem class must exist in the
        # audit checklist (deferred P8 repair maps AuditFindings ->
        # CriticNotes and reuses the revision seam).
        audit_flat = _flat(stages["fable2_audit_system"])
        for cls in _CRITIC_CLASSES:
            assert cls in audit_flat, f"audit missing critic class {cls}"
        for cls in _AUDIT_ONLY_CLASSES:
            assert cls in audit_flat, f"audit missing class {cls}"

    def test_casting_orders_by_menu_id(self, stages):
        # r2/S4: menu entries carry STABLE IDS; the LLM copies ids, never
        # descriptions.
        flat = _flat(stages["fable2_casting_system"])
        assert 'copy ONE menu id (like "m03")' in flat
        assert "ids only, never invent one" in flat
        assert "never the ANNOUNCER" in flat
        assert "SFW. No weapons, no smoking." in flat

    def test_audit_reports_never_rewrites(self, stages):
        flat = _flat(stages["fable2_audit_system"])
        assert "You report; you never rewrite." in flat
        assert "array of 0-12 objects" in flat
        assert ("An empty findings array is a legitimate, common result."
                in flat)

    def test_sfw_line_on_the_creative_seams(self, stages):
        for seam in ("fable2_pitch_system", "fable2_treatment_system",
                     "fable2_casting_system"):
            assert "SFW. No weapons, no smoking." in _flat(stages[seam]), seam
