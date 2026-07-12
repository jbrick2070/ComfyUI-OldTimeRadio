import json
from types import SimpleNamespace

import pytest

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_source_payload as source
from nodes import _otr_story_routing as routing
from nodes import _otr_story_rules as rules


def test_atomic_bank_pipeline_pack_rules_fetcher_and_runner_resolve():
    routing._REGISTRY = None
    rules._clear_caches()
    registry = routing._ensure_loaded()
    bank = registry.banks["original_codex56sol"]
    assert bank.runnable is True
    assert bank.default_story_model == "original_codex56sol_v1"
    assert bank.default_story_pipeline == "acoustic_puzzle_v1"
    assert registry.pipelines["acoustic_puzzle_v1"].executable is True
    pack = routing.resolve_story_pack("original_codex56sol")
    assert pack.story_pipeline_id == "acoustic_puzzle_v1"
    assert rules.resolve_story_rules("original_codex56sol").rules_id == bank.source_bank_id
    assert source.resolve_fetcher(bank).fetch is not None
    assert "acoustic_puzzle_v1" in writer._RUNNER_BY_PIPELINE
    assert list(registry.banks)[-2:] == ["original_codex56sol", "custom_source_bank"]


def test_local_seed_fetch_is_exact_deterministic_and_rejects_source_ref(monkeypatch):
    routing._REGISTRY = None
    bank = routing._ensure_loaded().banks["original_codex56sol"]
    entry = source.resolve_fetcher(bank)
    monkeypatch.setenv("OTR_ORIGINAL_SEED", "repeatable")
    a = entry.fetch(bank=bank, technical_model="unused", source_ref="")
    b = entry.fetch(bank=bank, technical_model="unused", source_ref="")
    assert a == b
    assert set(a.payload) == source.SOURCE_PAYLOAD_KEYS
    assert all(isinstance(v, str) for v in a.payload.values())
    assert a.payload["seed_text"]
    draw = json.loads(a.payload["seed_text"])
    assert draw == a.source_meta["constraint_draw"]
    assert len(draw["deck_sha256"]) == 64
    assert draw["device_spoken_anchor"]
    assert draw["resolution_spoken_anchor"]
    assert a.source_rights["rights_label"] == "synthetic-original"
    assert not a.source_rights["license_url"]
    with pytest.raises(source.SourcePayloadContractError, match="rejects source_ref"):
        entry.fetch(bank=bank, technical_model="unused", source_ref="https://example.test")


def test_canonical_default_and_bytes_remain_pinned():
    raw = (routing._STORY_PACKS_ROOT.parent.parent / "workflows" /
           "otr_canonical.json").read_bytes()
    workflow = json.loads(raw)
    writer_nodes = [n for n in workflow["nodes"] if n.get("type") == "OTR_LedgerScriptWriter"]
    assert len(writer_nodes) == 1
    assert writer_nodes[0]["widgets_values"][23] == "science_news"


def test_custom_title_override_receipts_name_the_real_bank():
    assert writer._title_source_for_custom_override(SimpleNamespace(
        source_bank_id="scifi_fable2",
    )) == "fable2_script_title"
    assert writer._title_source_for_custom_override(SimpleNamespace(
        source_bank_id="original_codex56sol",
    )) == "original_codex56sol_script_title"


def test_broadcast_score_prompt_closes_music_bookend_key_sets():
    routing._REGISTRY = None
    seam = routing.resolve_story_pack(
        "original_codex56sol").prompt_stages["codex56_broadcast_score"]
    assert "exactly the keys description and generation_prompt" in seam
    assert "never add music_file" in seam
    assert "shot_01, shot_01, shot_02 is valid" in seam
    assert "shot_01, shot_02, shot_01 is forbidden" in seam
    assert "beats array is chronological and MUST never be reordered" in seam
    assert "requires a new shot row with a new unique shot_id" in seam


@pytest.mark.parametrize("hint", ["", "please emphasize patient teamwork"])
def test_custom_premise_is_only_an_operator_hint_and_fetcher_still_runs(
        monkeypatch, hint):
    monkeypatch.setenv("OTR_ORIGINAL_SEED", "ingress-precedence")
    resolved = writer._resolve_inputs(
        source_bank="original_codex56sol", custom_premise=hint,
    )
    assert resolved["seed_source"] == "original_llm"
    assert json.loads(resolved["news_article"]["seed_text"])["constraint_id"]
    assert resolved["source_meta"]["operator_hint"] == hint
