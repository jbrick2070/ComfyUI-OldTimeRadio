"""The ACTUAL render receipt (campaign item 0, 2026-09-02).

One receipt per rendered segment, built in ``render_driver.render_shot`` from the request
the engine consumed and the clip it returned; ``actual_request_sha`` hashes the CAUSAL fields
only, so two A/A nulls agree and a changed prompt, seed, still, adapter strength or graph
setting disagrees by construction; wall seconds, the peak and the run id never enter the
hash. Node 92 stamps the trace ONCE under ``meta.render_trace``; the planned ``video``
section survives every later save (``TOP_PRESERVE``); the haunted engine pins its sampler
inputs from the same constants its graph builder uses.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_video_engines import render_driver as rd  # noqa: E402


class _Engine:
    name = "stub_engine"
    recipe_receipt_id = "stub_recipe_v1"
    family = "text_to_video"
    implementation_version = "stub-1"

    def __init__(self, artifacts=()):
        self._artifacts = list(artifacts)

    def sampler_inputs_for(self, request):
        return {"steps": 20, "cfg": 8.0, "sampler": "euler", "denoise": 1.0}

    def model_artifacts(self):
        return list(self._artifacts)


def _request(prompt="a lean figure, one clear action", seed=1234, still=""):
    return {
        "shot_id": "shot_b001", "role": "character_video",
        "text_prompt": prompt, "negative_prompt": "text, watermark",
        "seed_bundle": {"episode_seed": 1, "request_seed": seed, "variation_seed": 0},
        "timing": {"target_frame_count": 95, "target_duration_s": 7.6, "start_s": 1.0},
        "canvas": {"w": 512, "h": 288, "fps": 25},
        "asset_refs": {"init_image": still},
        "observability": {"prompt_sha8": hashlib.sha256(prompt.encode()).hexdigest()[:8]},
    }


SHOT = {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "stub_engine",
        "render_request_hash": "abc123def4567890", "role": "character_video"}
CLIP = {"path": "C:/x/shot_b001.mp4", "frame_count": 95, "recipe": "stub_recipe_v1",
        "vram_peak_mb": 13000}


def test_receipt_is_deterministic_over_the_causal_fields():
    r1 = rd.build_actual_receipt(_Engine(), SHOT, _request(), CLIP, wall_s=1.0)
    r2 = rd.build_actual_receipt(_Engine(), SHOT, _request(), dict(CLIP, vram_peak_mb=9000),
                                 wall_s=9.0)
    assert r1["actual_request_sha"] == r2["actual_request_sha"], "wall/peak are not causal"
    assert r1["receipt_version"] == rd.RENDER_RECEIPT_VERSION
    assert r1["seed"] == 1234 and r1["comparison_seed_hash"] == "abc123def4567890"
    assert r1["sampler_inputs"]["steps"] == 20 and r1["engine_id"] == "stub_engine"
    assert r1["wall_s"] == 1.0 and r2["wall_s"] == 9.0
    assert r1["prompt_sha8"] == _request()["observability"]["prompt_sha8"]


@pytest.mark.parametrize("change", ["prompt", "seed", "still", "sampler"])
def test_a_changed_causal_input_changes_the_sha(tmp_path, change):
    base = rd.build_actual_receipt(_Engine(), SHOT, _request(), CLIP)
    if change == "prompt":
        other = rd.build_actual_receipt(_Engine(), SHOT, _request(prompt="a tall figure"), CLIP)
    elif change == "seed":
        other = rd.build_actual_receipt(_Engine(), SHOT, _request(seed=99), CLIP)
    elif change == "still":
        still = tmp_path / "still.png"; still.write_bytes(b"\x89PNG stub")
        other = rd.build_actual_receipt(_Engine(), SHOT, _request(still=str(still)), CLIP)
        assert other["still_sha256"] == hashlib.sha256(b"\x89PNG stub").hexdigest()
    else:
        eng = _Engine(); eng.sampler_inputs_for = lambda req: {"steps": 30}
        other = rd.build_actual_receipt(eng, SHOT, _request(), CLIP)
    assert other["actual_request_sha"] != base["actual_request_sha"]


def test_model_artifacts_are_hashed_once_per_process(tmp_path, monkeypatch):
    weights = tmp_path / "mm.ckpt"; weights.write_bytes(b"motion module bytes" * 100)
    calls = {"n": 0}
    real_open = open

    def counting_open(path, *a, **k):
        if str(path) == str(weights) and "rb" in a:
            calls["n"] += 1
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", counting_open)
    rd._ARTIFACT_DIGESTS.clear()
    eng = _Engine(artifacts=[("motion_module", str(weights))])
    r1 = rd.build_actual_receipt(eng, SHOT, _request(), CLIP)
    r2 = rd.build_actual_receipt(eng, SHOT, _request(seed=5), CLIP)
    assert r1["model_artifacts"][0]["sha256"] == hashlib.sha256(b"motion module bytes" * 100).hexdigest()
    assert r1["model_artifacts"][0]["path"] == "mm.ckpt"
    assert r2["model_artifacts"] == r1["model_artifacts"]
    assert calls["n"] == 1, "the weight file is hashed once, then looked up"


def test_a_missing_engine_hook_yields_null_sampler_inputs_not_a_crash():
    class Bare:
        name = "cheap"
    r = rd.build_actual_receipt(Bare(), SHOT, _request(), {"path": "x.mp4"})
    assert r["sampler_inputs"] is None and r["model_artifacts"] == []
    assert r["actual_request_sha"]


def test_first_receipt_field_reads_single_and_multi_segment_clips():
    single = {"receipt": {"prompt_sha8": "aa", "seed": 7}}
    multi = {"receipts": [{"prompt_sha8": "bb", "seed": 8}, {"prompt_sha8": "cc", "seed": 9}]}
    assert rd._first_receipt_field(single, "seed") == 7
    assert rd._first_receipt_field(multi, "prompt_sha8") == "bb"
    assert rd._first_receipt_field({}, "seed") is None
    assert rd._first_receipt_field("a/path.mp4", "seed") is None


def test_the_haunted_engine_pins_its_sampler_inputs_from_its_own_constants():
    from nodes._otr_video_engines import eng_ghost_signal as g
    from nodes._otr_video_engines import eng_ghost_signal_official as go
    eng = go.GhostSignalV3HauntedEngine()
    si = eng.sampler_inputs_for(_request())
    assert si["checkpoint"] == g.GHOST_CHECKPOINT_NAME
    assert si["adapter"] == go.ADAPTER_V3_NAME and si["adapter_strength"] == float(eng.lora_strength)
    assert (si["steps"], si["cfg"], si["sampler"], si["scheduler"], si["denoise"]) == (
        g.GHOST_STEPS, g.GHOST_CFG, g.GHOST_SAMPLER_NAME, g.GHOST_SCHEDULER, g.GHOST_DENOISE)
    assert (si["canvas_w"], si["canvas_h"]) == (g.GHOST_CANVAS_W, g.GHOST_CANVAS_H)
    assert (si["context_length"], si["context_overlap"], si["context_fuse_method"]) == (
        g.GHOST_CONTEXT_LENGTH, g.GHOST_CONTEXT_OVERLAP, g.GHOST_CONTEXT_FUSE_METHOD)
    assert si["latent"] == "EmptyLatentImage" and si["init_image"] is None
    names = [n for n, _p in eng.model_artifacts()]
    assert names == ["checkpoint", "motion_module", "adapter"]


def test_the_adapter_strength_env_override_reaches_the_receipt(monkeypatch):
    from nodes._otr_video_engines import eng_ghost_signal_official as go
    monkeypatch.setenv(go.ADAPTER_V3_STRENGTH_ENV, "0.25")
    eng = go.GhostSignalV3HauntedEngine()
    a = rd.build_actual_receipt(eng, SHOT, _request(), CLIP)
    monkeypatch.setenv(go.ADAPTER_V3_STRENGTH_ENV, "1.0")
    b = rd.build_actual_receipt(eng, SHOT, _request(), CLIP)
    assert a["sampler_inputs"]["adapter_strength"] == 0.25
    assert a["actual_request_sha"] != b["actual_request_sha"]


def test_the_planned_video_section_survives_a_disk_merge(tmp_path):
    from nodes import production_ledger as PL
    on_disk = {"schema_version": "x", "episode_id": "ep", "video": {"shots": [{"shot_id": "s1"}]},
               "meta": {"freeze_timestamp": "t1"}, "audio": {"master_audio_sha256": "aa"}}
    path = tmp_path / "ep_ledger.json"; path.write_text(json.dumps(on_disk), encoding="utf-8")
    in_mem = {"schema_version": "x", "episode_id": "ep", "meta": {"freeze_timestamp": "t1"}}
    merged = PL.Ledger._merge_with_disk(copy.deepcopy(in_mem), str(path))
    assert merged.get("video") == on_disk["video"], "TOP_PRESERVE must keep the planned section"


def test_node_92_stamps_the_trace_once_with_a_run_id(monkeypatch):
    from nodes import otr_video_render_batch as vb
    from nodes import production_ledger as PL
    PL.new_ledger(episode_id="ep_trace_test")
    receipts = [{"shot_id": "s1", "seed": 1, "actual_request_sha": "aa"},
                {"shot_id": "s2", "seed": 2, "actual_request_sha": "bb"}, "not a dict"]
    n = vb._stamp_render_trace(receipts, render_run_id="run123")
    meta = PL.get_ledger().data["meta"]
    assert n == 2 and len(meta["render_trace"]) == 2
    assert [r["order"] for r in meta["render_trace"]] == [0, 1]
    assert all(r["render_run_id"] == "run123" for r in meta["render_trace"])
    assert meta["render_trace_version"] == "render_trace_v1"
    vb._stamp_render_trace([{"shot_id": "s9", "seed": 9}], render_run_id="run456")
    assert [r["shot_id"] for r in PL.get_ledger().data["meta"]["render_trace"]] == ["s9"], \
        "a second stamp REPLACES the trace, it never appends"
