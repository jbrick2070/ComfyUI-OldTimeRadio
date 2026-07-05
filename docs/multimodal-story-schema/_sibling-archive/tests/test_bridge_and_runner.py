"""Bridge artifact emission + simple_4 runner loud-failure contract."""

from __future__ import annotations

import json

import pytest

from upstream_story_lab.bridge import (
    BridgeError,
    build_bridge_artifact,
    build_spec,
    emit_bridge_artifact,
)
from upstream_story_lab.compat import (
    NEWS_ARTICLE_KEYS,
    NEWS_BRIEFS_FIELDS,
    NEWS_SEED_KEYS,
)
from upstream_story_lab.contracts import BridgeArtifact
from upstream_story_lab.runner import PipelineRunError, run_pipeline


def test_bridge_mirrors_are_complete_and_mapped(registry) -> None:
    for bank in ("science_news", "media_archive", "public_domain_story"):
        spec = build_spec(registry, source_bank_id=bank)
        artifact = build_bridge_artifact(spec)
        news = artifact.meta_mirrors.news
        assert set(news) == set(NEWS_BRIEFS_FIELDS)
        assert isinstance(news["key_terms"], list)  # freeze invariant
        # close_brief -> news_close_brief mapping is explicit
        assert news["news_close_brief"] == spec.story_input.close_brief
        seed = artifact.meta_mirrors.news_seed
        assert set(seed) == set(NEWS_SEED_KEYS)
        article = artifact.adapter_news_article
        assert set(article) == set(NEWS_ARTICLE_KEYS)
        assert article["seed_text"].strip()


def test_bridge_artifact_round_trips_to_disk(registry, tmp_path) -> None:
    spec = build_spec(registry, source_bank_id="media_archive")
    artifact = build_bridge_artifact(spec)
    path = emit_bridge_artifact(artifact, tmp_path / "bridge" / "artifact.json")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")  # no BOM
    reparsed = BridgeArtifact(**json.loads(raw.decode("utf-8")))
    assert reparsed.ledger_writing_spec.story_model_id == spec.story_model_id
    assert reparsed.ledger_writing_spec.provenance.pack_sha256


def test_bridge_refuses_empty_seed(registry) -> None:
    spec = build_spec(registry, source_bank_id="media_archive")
    stripped = spec.model_copy(deep=True)
    stripped.source_material.source_title = ""
    stripped.source_material.source_summary = ""
    stripped.source_material.raw_text = ""
    with pytest.raises(BridgeError, match="no title/summary/raw_text"):
        build_bridge_artifact(stripped)


# ---------------- simple_4 runner ----------------


def _experimental(registry):
    pack = registry.pack(
        "custom_source_bank", "simple_4_prompt_experimental",
        "simple_4_prompt_experimental",
    )
    pipeline = registry.pipeline("simple_4_prompt_experimental")
    return pack, pipeline


def test_runner_happy_path_runs_all_four_passes(registry) -> None:
    pack, pipeline = _experimental(registry)
    calls = []

    def fake_llm(slot, pass_id, prompt):
        calls.append((slot, pass_id))
        return f"output for {pass_id}"

    outputs = run_pipeline(pack, pipeline, fake_llm)
    assert list(outputs) == [p.pass_id for p in pipeline.passes]
    assert [s for s, _ in calls] == ["creative", "creative", "technical", "technical"]


@pytest.mark.parametrize("fail_at", [0, 1, 2, 3])
def test_runner_names_the_exact_failing_pass(registry, fail_at) -> None:
    pack, pipeline = _experimental(registry)
    target = pipeline.passes[fail_at].pass_id

    def fake_llm(slot, pass_id, prompt):
        if pass_id == target:
            raise RuntimeError("model exploded")
        return "ok"

    with pytest.raises(PipelineRunError, match=target):
        run_pipeline(pack, pipeline, fake_llm)


def test_runner_refuses_descriptive_pipeline(registry) -> None:
    pack = registry.pack("media_archive", "gentle_thriller", "legacy_many_pass")
    pipeline = registry.pipeline("legacy_many_pass")
    with pytest.raises(PipelineRunError, match="descriptive"):
        run_pipeline(pack, pipeline, lambda *a: "ok")


def test_runner_empty_output_fails_loud_no_fallback(registry) -> None:
    pack, pipeline = _experimental(registry)
    with pytest.raises(PipelineRunError, match="no fallback"):
        run_pipeline(pack, pipeline, lambda *a: "")
