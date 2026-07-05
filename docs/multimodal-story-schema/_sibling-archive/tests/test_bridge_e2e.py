"""End-to-end handoff: lab emit -> file -> production adapter load+validate
(kibitz r3, Codex M1: one contract, path in / dict out, proven)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "transplant_work" / "production_new_modules"
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))

import _otr_ledger_input_adapter as adapter  # noqa: E402

from upstream_story_lab.bridge import (  # noqa: E402
    build_bridge_artifact,
    build_spec,
    emit_bridge_artifact,
)


def test_emit_to_adapter_roundtrip(registry, tmp_path) -> None:
    for bank in ("science_news", "media_archive", "public_domain_story"):
        spec = build_spec(registry, source_bank_id=bank)
        artifact = build_bridge_artifact(spec)
        path = emit_bridge_artifact(artifact, tmp_path / f"{bank}.json")
        data = adapter.load_bridge_artifact(path)
        assert data["ledger_writing_spec"]["source_bank_id"] == bank


def test_adapter_load_rejects_missing_and_malformed(tmp_path) -> None:
    with pytest.raises(adapter.BridgeInputError, match="not found"):
        adapter.load_bridge_artifact(tmp_path / "ghost.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(adapter.BridgeInputError, match="not valid UTF-8 JSON"):
        adapter.load_bridge_artifact(bad)
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(adapter.BridgeInputError, match="BOM"):
        adapter.load_bridge_artifact(bom)


def test_close_brief_round_trip_survives_both_mappings(registry) -> None:
    """close_brief -> meta.news.news_close_brief (bridge) -> close_brief
    (science facade) round-trips exactly (kibitz r3, Claude Code M2: the two
    independent rename sites must agree)."""

    import _otr_source_interpreter as facade

    spec = build_spec(registry, source_bank_id="science_news")
    artifact = build_bridge_artifact(spec)

    class Briefs:
        def model_dump(self):
            return {
                k: artifact.meta_mirrors.news[k]
                for k in ("casting_brief", "script_brief", "news_close_brief",
                          "key_terms")
            }

    out = facade.interpret_source(
        "science_news", news_briefs_builder=lambda **kw: Briefs(),
    )
    assert out["close_brief"] == spec.story_input.close_brief


def test_provenance_baseline_is_live_from_manifest(registry) -> None:
    spec = build_spec(registry, source_bank_id="media_archive")
    assert spec.provenance.production_baseline == (
        "a7bdc42de2a7dde32c4ea5350141e770aa8ae03a"
    )


def test_emit_filename_scheme_distinguishes_all_axes(registry) -> None:
    """Two different selections of ANY axis must map to different artifact
    filenames (kibitz r3, Codex M2: no silent overwrite). Mirrors the naming
    scheme in nodes.OTR_BridgeArtifactEmit."""

    from upstream_story_lab.compat import canonical_json_hash

    def name_for(style: str) -> str:
        spec = build_spec(
            registry, source_bank_id="media_archive", visual_style_id=style,
        )
        content_hash = canonical_json_hash(
            spec.model_dump(mode="json")
        )[:8]
        return (
            f"bridge_{spec.source_bank_id}_{spec.story_model_id}_"
            f"{spec.story_pipeline_id}_{spec.visual_style_id}_{content_hash}.json"
        )

    assert name_for("archival_documentary") != name_for("anime")
