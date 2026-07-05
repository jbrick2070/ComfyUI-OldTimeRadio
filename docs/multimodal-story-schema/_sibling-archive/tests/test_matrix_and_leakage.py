"""Declared-matrix invariants + story/visual leakage (kibitz r1/r2 scope:
valid pack pairs x styles + one typed negative per axis; no full Cartesian)."""

from __future__ import annotations

import time

import pytest

from upstream_story_lab.bridge import build_spec
from upstream_story_lab.compat import PRODUCTION_VISUAL_TAILS
from upstream_story_lab.preview import (
    render_prompt_preview,
    render_visual_preview,
    scan_story_leakage,
    scan_visual_leakage,
)
from upstream_story_lab.profiles import resolve_profile
from upstream_story_lab.registry import UnknownIdError

RUNNABLE_BANKS = ("science_news", "media_archive", "public_domain_story")

#: Terms no NON-SCIENCE story preview may contain (the science lane is the
#: only lane where science/news words are allowed).
NON_SCIENCE_STORY_LEAKAGE = (
    "science-fiction audio drama",
    "sci-fi radio drama",
    "real science",
    "news facts",
    "spaceship",
    "mission control",
    "laboratory containment",
)

#: Visual terms the non-cinematic styles must never emit.
NON_CINEMA_VISUAL_LEAKAGE = ("35mm", "film grain", "radio studio")


def test_declared_matrix_resolves_and_previews(registry) -> None:
    """Every runnable (bank, model, pipeline) pack x every style: resolution
    succeeds, profile builds, spec assembles, previews render. Budget <10s."""

    start = time.monotonic()
    combos = 0
    for (bank_id, model_id, pipeline_id), (pack, _path) in registry.packs.items():
        if bank_id not in RUNNABLE_BANKS:
            continue
        if not registry.pipeline(pipeline_id).executable_in_lab:
            # legacy_many_pass: still resolvable + previewable.
            pass
        for style_id in registry.styles:
            spec = build_spec(
                registry,
                source_bank_id=bank_id,
                story_model_id=model_id,
                story_pipeline_id=pipeline_id,
                visual_style_id=style_id,
            )
            assert spec.story_model_id == model_id
            assert spec.visual_style_id == style_id
            assert render_prompt_preview(spec)
            assert render_visual_preview(spec)
            combos += 1
    assert combos >= 55  # 11 runnable-bank packs x 5 styles
    assert time.monotonic() - start < 10.0


def test_auto_defaults_are_recorded_not_invisible(registry) -> None:
    spec = build_spec(registry, source_bank_id="media_archive")
    decisions = {d.axis: d for d in spec.resolution.decisions}
    assert decisions["story_model"].default_applied is True
    assert decisions["story_model"].decided_by == "banks.json:default_story_model"
    assert decisions["visual_style"].resolved == "archival_documentary"
    assert spec.provenance.pack_sha256 and spec.provenance.banks_sha256


def test_one_negative_per_axis(registry) -> None:
    with pytest.raises(UnknownIdError):
        registry.resolve(source_bank_id="unknown_bank")
    with pytest.raises(UnknownIdError):
        registry.resolve(source_bank_id="media_archive", story_model_id="space_opera")
    with pytest.raises(UnknownIdError):
        registry.resolve(source_bank_id="media_archive", story_pipeline_id="mystery_pipe")
    with pytest.raises(UnknownIdError):
        registry.resolve(source_bank_id="media_archive", visual_style_id="vaporwave")


def test_non_science_story_previews_are_clean(registry) -> None:
    for (bank_id, model_id, pipeline_id), _ in registry.packs.items():
        if bank_id not in ("media_archive", "public_domain_story"):
            continue
        spec = build_spec(
            registry, source_bank_id=bank_id, story_model_id=model_id,
            story_pipeline_id=pipeline_id,
        )
        preview = render_prompt_preview(spec).lower()
        hits = [t for t in NON_SCIENCE_STORY_LEAKAGE if t in preview]
        assert not hits, f"{bank_id}/{model_id} leaked {hits}"
        assert not scan_story_leakage(registry, spec)


def test_science_lane_still_speaks_science(registry) -> None:
    profile = resolve_profile(registry, "science_news", "science_news_default",
                              "legacy_many_pass")
    assert "science" in profile.story_form_label
    assert profile.coda_mode == "real_news_report"


def test_non_cinematic_styles_emit_no_cinema_tails(registry) -> None:
    for style_id in ("anime", "cartoon", "paper_origami"):
        spec = build_spec(
            registry, source_bank_id="media_archive", visual_style_id=style_id,
        )
        preview = render_visual_preview(spec).lower()
        hits = [t for t in NON_CINEMA_VISUAL_LEAKAGE if t in preview]
        assert not hits, f"{style_id} leaked {hits}"
        assert not scan_visual_leakage(spec)


def test_sci_fi_radio_reproduces_production_tails_byte_identically(registry) -> None:
    policy = registry.style("sci_fi_radio")
    assert policy.era_tail == PRODUCTION_VISUAL_TAILS["ERA_TAIL_DEFAULT"]
    assert policy.positive_tail == PRODUCTION_VISUAL_TAILS["STYLE_TAIL_DEFAULT"]
    assert policy.image_grade_tail == PRODUCTION_VISUAL_TAILS["IMAGE_GRADE_TAIL"]
    assert policy.broadcast_tail == PRODUCTION_VISUAL_TAILS["RADIO_BROADCAST_TAIL"]
    assert policy.allow_radio_tails is True


def test_archival_documentary_motion_keys_match_production_vocabulary(registry) -> None:
    policy = registry.style("archival_documentary")
    assert set(policy.motion_prompts) == {
        "announcer", "music_open", "music_close", "music_inter",
    }
