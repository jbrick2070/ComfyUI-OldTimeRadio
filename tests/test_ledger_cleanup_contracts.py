"""Focused regression coverage for the pre-video ledger contracts."""

from pathlib import Path

import pytest

from nodes import _otr_content_safety as safety
from nodes import _otr_openrouter_backend as openrouter
from nodes import _otr_scifi_fable2 as fable2
from nodes._otr_video_engines import render_driver as rd


def _still_ledger(tmp_path, *, pool_path="", receipt=True):
    row = {
        "object_id": "still_b001",
        "kind": "scene_beat",
        "beat_id": "b001",
        "path": "",
        "pool_path": pool_path,
        "content_hash": "abc123456789",
    }
    images = {"images": [row]}
    if receipt:
        images["required_scene_targets"] = [{
            "object_id": "still_b001",
            "kind": "scene_beat",
            "beat_id": "b001",
            "path": "",
        }]
    return {
        "episode_id": "ep_contract",
        "lines": [{"line_id": "b001", "char_id": "",
                   "speaker_role": "background"}],
        "images": images,
        "video": {"shots": [{
            "shot_id": "shot_b001",
            "source_line_ids": ["b001"],
            "engine_id": "still_pan",
            "family": "static_image_gen",
        }]},
    }


def test_still_spine_repairs_pool_path_into_active_episode(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    pool = tmp_path / "pool.png"
    pool.write_bytes(b"still")
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(output_root))
    ledger = _still_ledger(tmp_path, pool_path=str(pool))

    receipt = rd.validate_and_repair_still_spine(ledger)

    repaired = Path(ledger["images"]["images"][0]["path"])
    assert repaired.is_file()
    assert repaired.parent == output_root / "otr" / "episodes" / "ep_contract" / "stills"
    assert receipt["validated"][0]["path"] == str(repaired)
    assert ledger["images"]["required_scene_targets"][0]["materialized_path"] == str(repaired)


def test_still_spine_fails_closed_without_authoritative_target_receipt(
        tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _still_ledger(tmp_path, receipt=False)
    with pytest.raises(rd.RenderError, match="required_scene_targets"):
        rd.validate_and_repair_still_spine(ledger)


def test_safety_cleanup_rejects_replacement_with_empty_spoken_surface(
        monkeypatch):
    ledger = {"lines": [{
        "line_id": "l1", "speaker_role": "character", "text": "damn it"
    }]}
    monkeypatch.setattr(
        safety, "propose_safety_patches",
        lambda _ledger, _slot: ({"l1": "(sound effect)"}, {"status": "patched"}),
    )

    receipt = safety.apply_safety_cleanup(ledger, lambda _prompt: "unused")

    assert receipt["status"] == "apply_failed"
    assert "spoken surface" in receipt["error"]
    assert ledger["lines"][0]["text"] == "damn it"


def test_bounded_provider_capacity_fails_before_network(monkeypatch):
    class BoundedProviderMessages(list):
        _otr_reserve_remaining_output_capacity = True
        _otr_fail_on_output_limit = True

    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        openrouter, "_post_chat_completion",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.raises(openrouter.OpenRouterConfigError, match="complete requested output"):
        openrouter.OpenRouterBackend().generate(
            {
                "slug": "test/model",
                "context_cap": 8192,
                "max_tokens_cap": 8192,
                "base_url": "https://example.invalid",
            },
            BoundedProviderMessages([{"role": "user", "content": "x" * 28000}]),
            temperature=0.2,
            max_new_tokens=2000,
        )
    assert calls == []


def test_scifi_markup_repair_names_the_exact_standalone_row():
    note = fable2._standalone_stage_direction_repair_note(
        ["BAD_LINE_SHAPE: (A sharp beep sounds.) (line 6)"]
    )
    assert "(A sharp beep sounds.)" in note
    assert "must not appear as a standalone output row" in note
    assert "Return no explanation" in note
