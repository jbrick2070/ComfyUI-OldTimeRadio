"""Focused regression coverage for the pre-video ledger contracts."""

from pathlib import Path

import pytest

from nodes import _otr_scifi_news_pro_markup as markup
from nodes import _otr_openrouter_backend as openrouter
from nodes import _otr_scifi_news_pro as scifi_news_pro
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
    # A POOL STILL IS A REAL PNG IN THE REAL POOL (2026-09-05). This used to
    # plant b"still" at tmp_path/pool.png -- outside the pool and not an image
    # -- which is the exact shape `_trusted_still_source` now refuses, because a
    # ledger-carried `pool_path` naming any readable file was copied into a
    # `/view`-served directory. Production's pool is
    # `<output>/otr/episodes/_shared/cache`.
    pool_dir = output_root / "otr" / "episodes" / "_shared" / "cache"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool = pool_dir / "pool.png"
    pool.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(120))
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


def test_still_spine_allows_visualizer_without_scene_manifest(
        tmp_path, monkeypatch):
    """No-still visualizers do not need a scene-target receipt."""
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = {
        "episode_id": "ep_visualizer",
        "lines": [{"line_id": "b001", "char_id": "",
                   "speaker_role": "background"}],
        "images": {"images": []},
        "video": {"shots": [{
            "shot_id": "shot_b001",
            "source_line_ids": ["b001"],
            "engine_id": "viz_mxc_cpu",
            "family": "abstract",
        }]},
    }

    receipt = rd.validate_and_repair_still_spine(ledger)

    assert receipt["validated"] == []
    assert ledger["images"]["still_spine_receipt"]["validated"] == []


# `test_safety_cleanup_rejects_replacement_with_empty_spoken_surface` was here
# and is DELETED (2026-08-23). It exercised `apply_safety_cleanup`'s internal
# invariant -- an LLM-proposed replacement that emptied a spoken row -- and that
# whole rewrite pass is gone from `_otr_content_safety`, unwired at its caller
# since 2026-08-05. A test for machinery that cannot run is not coverage. What
# replaces it is stronger and lives in `test_ledger_cleanup_pass.py`: an
# assertion that the rewrite entry points DO NOT EXIST.


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


def test_scifi_markup_repair_names_the_offending_standalone_row():
    """The repair rung must quote the parser's own evidence back to the model.

    Takes TYPED defects since PBUG-20260812-03: `str(defect)` appends
    ` (line N)`, and re-parsing that string is what corrupted the token the note
    looks up. The note now reads `code`, `detail` and `line_no` directly, so the
    line number no longer has to be smuggled through the detail.
    """
    note = scifi_news_pro._standalone_stage_direction_repair_note(
        (markup.ParseDefect(markup.NewsProParseDefect.BAD_LINE_SHAPE,
                            "(A sharp beep sounds.)", 6),),
        cast_names=("Ada", "Bo"),
    )
    assert "(A sharp beep sounds.)" in note
    assert "line 6" in note
    assert "must not appear as a standalone output row" in note
    assert "Return no explanation" in note
