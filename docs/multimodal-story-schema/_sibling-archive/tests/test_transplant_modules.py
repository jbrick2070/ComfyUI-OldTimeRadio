"""Pure tests for the staged production modules (transplant_work/). These are
standalone dict-in/dict-out modules: no lab imports, no production imports
(kibitz r2: lab-side testing of production-shaped code is pure-module only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "transplant_work" / "production_new_modules"
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))

import _otr_ledger_input_adapter as adapter  # noqa: E402
import _otr_source_interpreter as facade  # noqa: E402
import _otr_story_prompt_profile as spp  # noqa: E402
import _otr_visual_style_policy as vsp  # noqa: E402

from upstream_story_lab.bridge import build_bridge_artifact, build_spec  # noqa: E402


def _artifact_dict(registry, bank="media_archive"):
    spec = build_spec(registry, source_bank_id=bank)
    return build_bridge_artifact(spec).model_dump(mode="json")


def test_adapter_accepts_real_bridge_artifacts(registry) -> None:
    for bank in ("science_news", "media_archive", "public_domain_story"):
        adapter.validate_bridge_artifact(_artifact_dict(registry, bank))


def test_adapter_rejects_missing_and_inconsistent(registry) -> None:
    data = _artifact_dict(registry)
    broken = {k: v for k, v in data.items() if k != "meta_mirrors"}
    with pytest.raises(adapter.BridgeInputError, match="meta_mirrors"):
        adapter.validate_bridge_artifact(broken)

    twisted = _artifact_dict(registry)
    twisted["ledger_writing_spec"]["prompt_profile"]["story_model_id"] = "other"
    with pytest.raises(adapter.BridgeInputError, match="prompt_profile"):
        adapter.validate_bridge_artifact(twisted)

    drifted = _artifact_dict(registry)
    drifted["meta_mirrors"]["news"].pop("decoder_profile")
    with pytest.raises(adapter.BridgeInputError, match="NewsBriefs"):
        adapter.validate_bridge_artifact(drifted)


def test_profile_helpers_produce_locked_kwargs(registry) -> None:
    data = _artifact_dict(registry)
    profile = data["ledger_writing_spec"]["prompt_profile"]
    outline = spp.outline_request_fields(profile)
    assert outline["source_material_label"] == "Media archive item"
    overrides = spp.style_picker_overrides(profile)
    assert set(overrides) == {
        "inventor_system_prompt", "chooser_system_prompt", "chooser_user_template",
    }
    # Non-science lanes must OVERRIDE the production sci-fi inventor persona
    # (the pack prompt may still mention sci-fi as a negation; the leakage
    # scan governs term policy, not this test).
    assert overrides["inventor_system_prompt"].strip()
    assert overrides["inventor_system_prompt"] != "You are a sci-fi radio drama showrunner."
    assert spp.coda_mode(profile) == "archive_source_note"


def test_science_profile_leaves_style_picker_constants(registry) -> None:
    data = _artifact_dict(registry, "science_news")
    profile = data["ledger_writing_spec"]["prompt_profile"]
    overrides = spp.style_picker_overrides(profile)
    assert overrides == {
        "inventor_system_prompt": "",
        "chooser_system_prompt": "",
        "chooser_user_template": "",
    }  # empty = production module constants stay byte-identical


def test_visual_policy_stamp_and_tails(registry) -> None:
    data = _artifact_dict(registry)
    policy = data["ledger_writing_spec"]["visual_policy"]
    meta: dict = {}
    vsp.stamp_meta(meta, policy)
    assert vsp.read_style_id(meta) == "archival_documentary"
    with pytest.raises(vsp.VisualPolicyError, match="refusing to restamp"):
        vsp.stamp_meta({"visual_style": "sci_fi_radio"}, policy)
    tails = vsp.tail_overrides(policy)
    assert tails["style_tail"]
    with pytest.raises(vsp.VisualPolicyError, match="unknown motion role"):
        vsp.motion_prompt_for_role(policy, "scene_broll")


def test_facade_routes_and_refuses(registry) -> None:
    data = _artifact_dict(registry)
    story_input = data["ledger_writing_spec"]["story_input"]
    out = facade.interpret_source("media_archive", bridge_story_input=story_input)
    assert set(out) == set(facade.LOGICAL_FIELDS)

    with pytest.raises(facade.SourceInterpreterError, match="packet-driven"):
        facade.interpret_source("media_archive")
    with pytest.raises(facade.SourceInterpreterError, match="news_briefs_builder"):
        facade.interpret_source("science_news")
    with pytest.raises(facade.SourceInterpreterError, match="not runnable"):
        facade.interpret_source("custom_source_bank")
    with pytest.raises(facade.SourceInterpreterError, match="cross-lane"):
        facade.interpret_source("public_domain_story", bridge_story_input=story_input)


def test_facade_science_wraps_builder_verbatim() -> None:
    class FakeBriefs:
        def model_dump(self):
            return {
                "casting_brief": "c", "script_brief": "s",
                "news_close_brief": "n", "key_terms": ["k"],
            }

    calls = {}

    def builder(**kwargs):
        calls.update(kwargs)
        return FakeBriefs()

    out = facade.interpret_source(
        "science_news",
        article={"headline": "h", "summary": "sum", "full_text": "ft",
                 "source": "Outlet", "date": "2026-07-02", "link": "",
                 "seed_text": "h sum"},
        news_briefs_builder=builder,
        technical_fn="tf-sentinel",
    )
    assert out["close_brief"] == "n"
    # kibitz r4 (Codex M1): the facade maps the article dict onto the REAL
    # build_news_briefs keyword set - no `article` kwarg exists in production.
    assert "article" not in calls
    assert calls["headline"] == "h"
    assert calls["outlet"] == "Outlet"
    assert calls["pub_date"] == "2026-07-02"
    assert calls["technical_fn"] == "tf-sentinel"
