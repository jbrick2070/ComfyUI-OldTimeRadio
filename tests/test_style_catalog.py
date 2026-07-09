"""Story-grammar: 100-style catalog + climax-class ending taxonomy + the
deterministic selector (2026-06-24). Pure / CPU. UTF-8 no BOM, SFW.

Chunk 1 (l12): climax_role param + climax-class validator (default-preserving).
Chunk 2 (catalog): ending_tag per style + ENDING_TEMPLATES + validate_catalog.
Chunk 3 (catalog): deterministic select_style.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_style_catalog as CAT  # noqa: E402
from nodes import _otr_story_quality_l12 as L12  # noqa: E402


# ---------------------------------------------------------------------------
# Chunk 1 -- climax-class roles + validator (default-preserving)
# ---------------------------------------------------------------------------
class TestClimaxClassRoles:
    def test_irreversible_choice_is_one_of_nine_climax_classes(self):
        assert "irreversible_choice" in L12.CLIMAX_CLASS_ROLES
        assert len(L12.CLIMAX_CLASS_ROLES) == 9

    def test_default_assign_is_byte_identical(self):
        # No climax_role => irreversible_choice everywhere (pre-2026-06-24 behavior).
        ids = [f"b{i:03d}" for i in range(1, 6)]
        roles = L12.assign_beat_roles(ids)
        assert roles[ids[-1]] == "irreversible_choice"
        L12.validate_beat_roles(roles, ids)

    def test_climax_role_drives_last_beat(self):
        ids = ["b001", "b002", "b003"]
        roles = L12.assign_beat_roles(ids, climax_role="confession")
        assert roles["b003"] == "confession"
        assert roles["b001"] == "setup"
        assert roles["b002"] == "personal_stake"
        L12.validate_beat_roles(roles, ids)   # a non-irreversible climax validates

    def test_invalid_climax_role_falls_back(self):
        roles = L12.assign_beat_roles(["b001"], climax_role="not_a_role")
        assert roles["b001"] == "irreversible_choice"

    def test_validator_accepts_any_single_climax_class_last(self):
        for tag in L12.CLIMAX_CLASS_ROLES:
            roles = L12.assign_beat_roles(["b001", "b002", "b003"], climax_role=tag)
            L12.validate_beat_roles(roles, ["b001", "b002", "b003"])

    def test_validator_rejects_two_climax_class(self):
        bad = {"b001": "revelation", "b002": "confession"}
        with pytest.raises(L12.BeatRoleViolation):
            L12.validate_beat_roles(bad, ["b001", "b002"])

    def test_validator_rejects_climax_not_last(self):
        bad = {"b001": "reversal", "b002": "pressure", "b003": "setup"}
        with pytest.raises(L12.BeatRoleViolation):
            L12.validate_beat_roles(bad, ["b001", "b002", "b003"])

    def test_validator_rejects_zero_climax(self):
        bad = {"b001": "setup", "b002": "pressure", "b003": "consequence"}
        with pytest.raises(L12.BeatRoleViolation):
            L12.validate_beat_roles(bad, ["b001", "b002", "b003"])


# ---------------------------------------------------------------------------
# Chunk 2 -- catalog ending taxonomy
# ---------------------------------------------------------------------------
class TestCatalogEndings:
    def test_validate_catalog_passes(self):
        CAT.validate_catalog()

    def test_all_100_have_climax_class_tag(self):
        assert len(CAT.STYLE_CATALOG) == 100
        for s in CAT.STYLE_CATALOG:
            assert s["ending_tag"] in L12.CLIMAX_CLASS_ROLES

    def test_ending_tags_match_climax_classes(self):
        assert set(CAT.ENDING_TAGS) == set(L12.CLIMAX_CLASS_ROLES)

    def test_variety_and_irreversible_is_rare(self):
        used = {s["ending_tag"] for s in CAT.STYLE_CATALOG}
        assert len(used) >= 8                     # not collapsed to one ending
        n_irrev = sum(1 for s in CAT.STYLE_CATALOG
                      if s["ending_tag"] == "irreversible_choice")
        assert n_irrev <= 12                       # demoted from 100% to a sliver

    def test_every_tag_has_a_template(self):
        for tag in CAT.ENDING_TAGS:
            assert CAT.ENDING_TEMPLATES[tag].strip()

    def test_templates_steer_toward_human_on_mic_endings(self):
        # The whole point: every template pulls toward a human, on-mic ending and
        # away from machinery (the irreversible_choice one names the banned
        # devices explicitly; the rest say "not on a device/machine/action").
        _steer = ("not on", "no machinery", "on-mic", "no console", "no device",
                  "do not resolve")
        for tag, tmpl in CAT.ENDING_TEMPLATES.items():
            low = tmpl.lower()
            assert any(p in low for p in _steer), f"{tag} template lacks steering"

    def test_ending_template_for_unknown_slug_defaults(self):
        assert CAT.ending_template_for("no_such_style").strip()

    def test_ending_mode_prose_preserved(self):
        # The rename trap: render_style_grammar still reads ending_mode.
        assert "Ending mode:" in CAT.render_style_grammar("locked_room_suspense")


# ---------------------------------------------------------------------------
# Chunk 3 -- deterministic selector
# ---------------------------------------------------------------------------
class TestSelectStyle:
    def test_deterministic(self):
        a = CAT.select_style("a quiet town keeps a secret", {}, 42)
        b = CAT.select_style("a quiet town keeps a secret", {}, 42)
        assert a == b
        assert a in CAT.all_slugs()

    def test_seed_varies_selection(self):
        picks = {CAT.select_style("a town", {}, s) for s in range(20)}
        assert len(picks) > 1                      # different seeds -> variety

    def test_non_disaster_premise_picks_non_emergency(self):
        slug = CAT.select_style("two old friends share a long overdue dinner", {}, 7)
        assert slug in CAT.non_emergency_slugs()

    def test_disaster_premise_allows_emergency(self):
        assert CAT.premise_wants_emergency("a reactor meltdown forces an evacuation", {})
        assert not CAT.premise_wants_emergency("a museum acquires a quiet painting", {})

    def test_emergency_only_on_disaster_keywords(self):
        # Across many seeds, a calm premise never reaches an emergency-tagged style.
        emergency = set(CAT.all_slugs()) - set(CAT.non_emergency_slugs())
        for s in range(40):
            assert CAT.select_style("a slow afternoon at the library", {}, s) not in emergency


# ---------------------------------------------------------------------------
# Chunk 4 -- media archive style pool (QA must-fix test gap)
# ---------------------------------------------------------------------------
class TestMediaArchiveStylePool:
    """Proves the curated media_archive style pool is correct and integrated."""

    # --- all curated slugs exist in the catalog ---
    def test_all_media_archive_slugs_exist_in_catalog(self):
        all_known = CAT.all_slugs()
        for slug in CAT.media_archive_slugs():
            assert slug in all_known, f"media_archive slug {slug!r} not in STYLE_CATALOG"

    # --- select_style with source_bank="media_archive" returns from curated pool ---
    def test_select_style_uses_media_archive_pool(self):
        pool = set(CAT.media_archive_slugs())
        for seed in range(50):
            slug = CAT.select_style(
                "an archivist finds a lost recording", {"source_bank": "media_archive"}, seed
            )
            assert slug in pool, f"seed {seed} -> {slug!r} not in media_archive pool"

    # --- curated pool excludes the 6 pruned drift-prone slugs ---
    _PRUNED = (
        "newsroom_emergency_bulletin",
        "pulp_serial_cliffhanger",
        "missing_person_audio_diary",
        "lost_expedition_recordings",
        "old_theater_phantom_rehearsal",
        "live_variety_show_disaster",
    )

    def test_pruned_slugs_absent(self):
        pool = set(CAT.media_archive_slugs())
        for slug in self._PRUNED:
            assert slug not in pool, f"drift-prone slug {slug!r} still in media_archive pool"

    # --- pool size is exactly 18 after pruning ---
    def test_pool_size(self):
        assert len(CAT.media_archive_slugs()) == 18

    # --- curated pool excludes obvious sci-fi, emergency, crime, horror,
    #     disaster, ghost/phantom, and survival styles ---
    _EXCLUDED_CATEGORIES = {
        "emergency", "ghost", "phantom", "killer", "murder",
        "expedition_recording", "disaster",
    }

    def test_no_obvious_drift_category_in_pool(self):
        for slug in CAT.media_archive_slugs():
            for cat in self._EXCLUDED_CATEGORIES:
                assert cat not in slug, (
                    f"media_archive slug {slug!r} contains excluded category {cat!r}"
                )

    # --- science/default style selection is byte-identical for pinned seeds ---
    def test_science_selection_unchanged(self):
        """The media_archive pool must NOT affect the default (science) path."""
        a = CAT.select_style("a quiet town keeps a secret", {}, 42)
        b = CAT.select_style("a quiet town keeps a secret", {}, 42)
        assert a == b
        # Must be from the full catalog, not the media_archive pool
        assert a in CAT.all_slugs()

    def test_media_archive_deterministic(self):
        a = CAT.select_style("archive recording", {"source_bank": "media_archive"}, 99)
        b = CAT.select_style("archive recording", {"source_bank": "media_archive"}, 99)
        assert a == b

    def test_visual_style_metadata_does_not_affect_story_style(self):
        """Visual packs are downstream render metadata, not story grammar."""
        premise = "archive recording"
        base_meta = {"source_bank": "media_archive"}
        base = CAT.select_style(premise, base_meta, 99)
        for visual_style in ("sci_fi_radio", "recur_frac", "video_art"):
            styled = CAT.select_style(
                premise,
                {**base_meta, "visual_style": visual_style},
                99,
            )
            assert styled == base

    def test_media_archive_seed_varies(self):
        picks = {
            CAT.select_style("archive", {"source_bank": "media_archive"}, s)
            for s in range(30)
        }
        assert len(picks) > 1  # different seeds -> variety
