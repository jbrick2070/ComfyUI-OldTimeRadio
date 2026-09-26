"""The Comfy-native Gemma 4 writer (plan row 0n): identity, pinned weight, planning.

Slice A2 of kibitz-runs/2026-09-26-comfy-gemma-writer. CPU only, no network, no
ComfyUI: every assertion calls the real function it is about.
"""
from __future__ import annotations

import pytest

from nodes import _otr_comfy_textgen_backend as native
from nodes import _otr_visual_assets as va


def _prompt(creative="Qwen/Qwen3.5-4B", technical="Qwen/Qwen3.5-4B", replay=""):
    return {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "1": {"class_type": "OTR_LedgerScriptWriter",
              "inputs": {"gate_in": ["63", 0], "replay_from": replay,
                         "creative_writing_model": creative,
                         "technical_model": technical}},
    }


def _plan(prompt):
    return va.plan_prompt(prompt, "63", resolve_video=lambda v: v,
                          freeze_video=lambda v: v, role_video_slots={})


def test_the_manifest_row_is_the_backend_pin():
    """One pin, two readers: the fetch table and the backend must name the same
    file at the same revision, size and hash, or the download verifies a file the
    loader then refuses."""
    spec = va.MANIFEST[(native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN)]
    assert spec == {"repo_id": native.WEIGHT_REPO,
                    "filename": native.WEIGHT_FILENAME,
                    "revision": native.WEIGHT_REVISION,
                    "size": native.WEIGHT_SIZE,
                    "sha256": native.WEIGHT_SHA256}
    assert native.WEIGHT_FILENAME.rsplit("/", 1)[-1] == native.WEIGHT_TOKEN


def test_only_the_exact_native_id_needs_a_weight():
    assert native.native_writer_weights(native.MODEL_ID) == (
        (native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN),)
    for other in ("google/gemma-4-E2B-it", "Qwen/Qwen3.5-4B", "comfy:slot-a",
                  "", None, native.MODEL_ID + " (5.2 GB download)"):
        assert native.native_writer_weights(other) == (), other
    assert native.is_native_writer(native.MODEL_ID)
    assert not native.is_native_writer("google/gemma-4-E2B-it")


@pytest.mark.parametrize("creative, technical", [
    (native.MODEL_ID, "Qwen/Qwen3.5-4B"),
    ("Qwen/Qwen3.5-4B", native.MODEL_ID + " (5.2 GB download)"),
    (native.MODEL_ID + " (5.2 GB download)", native.MODEL_ID),
])
def test_either_slot_plans_the_native_writer_once(creative, technical):
    plan = _plan(_prompt(creative, technical))
    assert plan["writer_models"] == {native.MODEL_ID}


def test_hf_and_cloud_writer_picks_plan_no_writer_weight():
    for creative, technical in (("Qwen/Qwen3.5-4B", "google/gemma-4-12b-it"),
                                ("comfy:slot-a", "comfy:slot-b"),
                                ("google_api:slot-a", "google_api:slot-b")):
        assert _plan(_prompt(creative, technical))["writer_models"] == set()


def test_a_replay_plans_no_writer_weight():
    """A replay's writer passes through frozen; nothing it names is fetched."""
    plan = _plan(_prompt(native.MODEL_ID, native.MODEL_ID, replay="C:/bundle"))
    assert plan["replay"] is True
    assert plan["writer_models"] == set()


def test_both_slots_request_the_pinned_weight_once():
    requests = va.native_requests(set(), folder_paths=va._NothingInstalled,
                                  writer_models={native.MODEL_ID})
    assert [(r["category"], r["token"]) for r in requests] == [
        (native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN)]
    assert requests[0]["path"] is None
    assert requests[0]["spec"]["revision"] == native.WEIGHT_REVISION
    assert requests[0]["spec"]["sha256"] == native.WEIGHT_SHA256


def test_no_writer_models_requests_nothing_extra():
    assert va.native_requests(set(), folder_paths=va._NothingInstalled) == []


def test_a_linked_writer_widget_never_refuses_the_readiness_pass():
    """The video weights must still plan when the writer's model is wired from
    another node; the linked slot is noted, never a refusal."""
    prompt = _prompt()
    prompt["1"]["inputs"]["creative_writing_model"] = ["99", 0]
    del prompt["1"]["inputs"]["technical_model"]
    plan = _plan(prompt)
    assert plan["writer_models"] == set()
    assert any("creative_writing_model is linked" in n for n in plan["skipped"])
