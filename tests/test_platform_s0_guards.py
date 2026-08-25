"""S0 platform-portability guards (docs/2026-07-09-platform-portability-final.md).

Committed-state defects that broke every non-CUDA tier outright:

  - ``_otr_model_loader``: unguarded ``torch.cuda.get_device_capability()``
    (the loader :257 crash) + a CUDA-device-0-keyed ``max_memory`` dict
    built from model-id string tags alone, handed to transformers on hosts
    with no CUDA device 0.
  - ``vram_context_test``: every ``torch.cuda.*`` memory counter unguarded,
    so the diagnostics node crashed on exactly the hosts it should measure.
  - ``_otr_workflow_validator._detect_host``: MPS-blind and vendor-blind.

The suite env hides CUDA (conftest sets ``CUDA_VISIBLE_DEVICES=''``), so
every direct call below IS the non-CUDA host case the defects broke.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_loader as ml
from nodes import vram_context_test as vct
from nodes._otr_workflow_validator import WorkflowValidator


# --------------------------------------------------------------------------
# _plan_max_memory: backend-keyed VRAM budgeting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", [
    "google/gemma-2-2b-it",                 # 2b tag branch
    "mistralai/Mistral-Nemo-Instruct-2407", # 12b tag branch
    "google/gemma-2-9b-it",                 # 9b tag branch
    "some/tiny-model",                      # no branch
])
def test_plan_max_memory_is_none_off_cuda(model_id):
    """No CUDA -> no CUDA-device-0 plan, regardless of model-id tags.

    The pre-S0 code keyed ``{0: ...}`` off string tags alone; the 2b/9b/12b
    branches fired on CUDA-less hosts and handed transformers a device map
    for hardware that does not exist."""
    assert ml._plan_max_memory(model_id, 0.0, cuda_available=False) is None
    # Paranoia: even a (mis)reported total_vram must not resurrect the plan.
    assert ml._plan_max_memory(model_id, 16.0, cuda_available=False) is None


def test_plan_max_memory_cuda_branches_unchanged():
    """On CUDA the plan is byte-identical to the pre-S0 behaviour."""
    assert ml._plan_max_memory("anything", 16.0, cuda_available=True) == {
        0: "13.5GiB", "cpu": "32GiB"}
    assert ml._plan_max_memory("google/gemma-2-2b-it", 8.0,
                               cuda_available=True) == {
        0: "3.2GiB", "cpu": "32GiB"}
    # NB: ids ENDING in "2b" (e.g. "foo-12b") hit the 2b branch -- that is
    # the pre-S0 tag semantics, preserved verbatim. A 9b id exercises the
    # mid branch cleanly.
    assert ml._plan_max_memory("google/gemma-2-9b-it", 8.0,
                               cuda_available=True) == {
        0: "6.8GiB", "cpu": "32GiB"}
    assert ml._plan_max_memory("some/tiny-model", 8.0,
                               cuda_available=True) is None


# --------------------------------------------------------------------------
# _apply_matmul_precision_policy: the loader :257 crash
# --------------------------------------------------------------------------

def test_matmul_precision_policy_survives_cuda_less_host(monkeypatch):
    """get_device_capability() must never be reached without CUDA."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _boom(*a, **k):  # the exact pre-S0 crash, now unreachable
        raise RuntimeError("No CUDA GPUs are available")

    monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
    ml._apply_matmul_precision_policy()  # must not raise


# --------------------------------------------------------------------------
# vram_context_test: accelerator-keyed probe helpers
# --------------------------------------------------------------------------

def test_vram_node_accel_helpers_survive_no_accelerator(monkeypatch):
    """accel='none' path: zeros, no torch.cuda.* reached, no raise."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        monkeypatch.setattr(mps, "is_available", lambda: False)

    accel = vct._accel_state()
    assert accel == "none"

    def _boom(*a, **k):
        raise RuntimeError("No CUDA GPUs are available")

    for fn_name in ("empty_cache", "reset_peak_memory_stats",
                    "memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, fn_name, _boom)

    _ = vct._accel_empty_cache(accel)
    _ = vct._accel_reset_peak(accel)
    assert vct._accel_allocated_gb(accel) == 0.0
    assert vct._accel_peak_gb(accel) == 0.0


def test_vram_node_accel_state_is_valid_on_this_host():
    """Whatever the live host is, the state must be a known enum value."""
    assert vct._accel_state() in ("cuda", "mps", "none")


# --------------------------------------------------------------------------
# _detect_host: MPS + vendor awareness (additive keys)
# --------------------------------------------------------------------------

def test_detect_host_shape_and_vendor_consistency():
    host = WorkflowValidator._detect_host()
    assert set(host) == {"platform", "has_cuda", "has_mps", "vendor",
                         "vram_mb"}
    assert host["platform"] in ("win", "mac", "any")
    assert isinstance(host["has_cuda"], bool)
    assert isinstance(host["has_mps"], bool)
    assert isinstance(host["vram_mb"], int)
    assert host["vendor"] in ("none", "nvidia", "amd", "apple")
    if not host["has_cuda"] and not host["has_mps"]:
        assert host["vendor"] == "none"
    if host["has_cuda"]:
        assert host["vendor"] in ("nvidia", "amd")
        # The lie detector: claiming CUDA with 0 MB VRAM was the pre-S0
        # failure shape (is_available() True, device 0 unopenable).
        assert host["vram_mb"] > 0


# --------------------------------------------------------------------------
# bark registry ruling + cpu_floor runnability
# --------------------------------------------------------------------------

def test_bark_registry_row_admits_cpu_but_stays_impractical_there():
    """2026-08-25: _generate_single_line no longer hardcodes CUDA (it asks
    the loaded model for its real device), so a CPU-loaded bark generates
    correctly instead of crashing on the first line -- "cpu" belongs in
    device_backends now. It still is not a PRACTICAL cpu_floor choice: bark
    is a ~1B-parameter three-stage autoregressive stack, so
    practical_without_gpu stays False and the profile-fit reason below must
    become REASON_IMPRACTICAL_ON_CPU rather than REASON_REQUIRES_CUDA."""
    from nodes._otr_audio_engines import registry as areg

    assert areg.CAPABILITIES["bark"]["device_backends"] == ["cuda", "cpu"]
    assert areg.CAPABILITIES["bark"]["practical_without_gpu"] is False


def test_cpu_floor_profile_is_runnable_on_cpu():
    """The committed cpu_floor profile must select only engines whose
    declarations admit the cpu backend, and the bark_legacy voice bank goes
    with bark. bark itself stays excluded from cpu_floor -- CUDA is no
    longer REQUIRED (device_backends includes "cpu"), but running a
    three-stage autoregressive TTS stack on CPU is not PRACTICAL, so the
    exclusion reason changes from REASON_REQUIRES_CUDA to
    REASON_IMPRACTICAL_ON_CPU rather than disappearing."""
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_shared import capability_profiles as cp

    prof = cp.load_profile("cpu_floor")
    avail = cp.availability(prof, areg.CAPABILITIES)
    assert avail["bark"] == cp.REASON_IMPRACTICAL_ON_CPU
    for slot in ("char_voice_engine", "announcer_voice_engine",
                 "music_engine"):
        eng = prof["slot_overrides"][slot]
        assert avail.get(eng) == cp.REASON_OK, (slot, eng)
    assert prof["slot_overrides"]["char_voice_engine"] == "kokoro"
    assert prof["slot_overrides"]["voice_bank"] == "default"
    # Post-ship audit (2026-07-10): the IMAGE roles must be overridden too
    # -- inheriting canonical's cuda-only z_image_turbo shipped a GPU
    # engine on the no-GPU tier. Every image override must fit cpu.
    from nodes._otr_image_engines import registry as ireg_caps

    img_avail = cp.availability(prof, ireg_caps.CAPABILITIES)
    for slot in ("announcer_image", "music_image", "character_image"):
        eng = prof["role_overrides"].get(slot)
        assert eng, f"cpu_floor must override {slot} (canonical is cuda-only)"
        assert img_avail.get(eng) == cp.REASON_OK, (slot, eng)


# --------------------------------------------------------------------------
# indextts2 installer gap: the scripts its errors name must exist
# --------------------------------------------------------------------------

def test_indextts2_named_install_scripts_exist():
    """eng_indextts2's fail-closed errors point operators at two scripts;
    pre-S0 NEITHER existed anywhere in the repo (only the worker did)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    eng = (repo / "nodes" / "_otr_audio_engines"
           / "eng_indextts2.py").read_text(encoding="utf-8")
    for name in ("_otr_indextts2_install.ps1", "_otr_idx_download_weights.py",
                 "_otr_indextts2_worker.py"):
        assert name in eng or name == "_otr_indextts2_worker.py", name
        assert (repo / "scripts" / name).exists(), (
            f"scripts/{name} is named by eng_indextts2 error text (or is the "
            "worker) but does not exist")


# --------------------------------------------------------------------------
# S4: host_caps shape + the CastLock voice_device ledger stamp
# --------------------------------------------------------------------------

def test_build_host_caps_minimum_shape():
    """Verify-at-build register: the minimum host_caps shape every adapter
    receives (S4 killed the empty-dict adapter boundary)."""
    from nodes._otr_shared.host_caps import build_host_caps

    caps = build_host_caps()
    assert set(caps) == {"cuda_available", "mps_available", "vendor",
                         "total_vram_gb"}
    assert isinstance(caps["cuda_available"], bool)
    assert isinstance(caps["mps_available"], bool)
    assert caps["vendor"] in ("none", "nvidia", "amd", "apple")
    assert isinstance(caps["total_vram_gb"], float)
    if caps["cuda_available"]:
        assert caps["total_vram_gb"] > 0


def test_cast_lock_stamps_voice_device():
    """S4: meta.voice_device rides the ledger like the engine stamps --
    every voice adapter + theme music reads ONE explicit device."""
    import json

    from nodes.cast_lock import CastLock

    led = {"episode_id": "ep_vd", "cast": [], "meta": {}}
    out = CastLock().lock(script_json=json.dumps(led),
                          cast_voice_policy="preserve_ledger",
                          voice_device="cpu")
    patched = json.loads(out[0])
    assert patched["meta"]["voice_device"] == "cpu"
    # Default = the nv50 baseline.
    out2 = CastLock().lock(script_json=json.dumps(led),
                           cast_voice_policy="preserve_ledger")
    assert json.loads(out2[0])["meta"]["voice_device"] == "cuda"
    # Invalid device fails loud.
    import pytest as _pytest
    with _pytest.raises(ValueError, match="voice_device"):
        CastLock().lock(script_json=json.dumps(led),
                        cast_voice_policy="preserve_ledger",
                        voice_device="tpu")


# --------------------------------------------------------------------------
# HuMo artifact alignment: script vs engine default vs registry label
# --------------------------------------------------------------------------

def test_humo_download_script_fetches_engine_default_unet():
    """Fresh-install alignment (S0): the download script fetched Comfy-Org's
    humo_17B file while the engine default resolves Kijai's 14B_KJ file --
    ~16 GB downloaded, engine still reported not-installed."""
    import re
    from pathlib import Path

    from nodes._otr_video_engines import eng_humo

    script = (Path(__file__).resolve().parents[1] / "scripts"
              / "download_humo_models.ps1")
    text = script.read_text(encoding="utf-8")
    assert eng_humo._HUMO_DEFAULT_UNET in text, (
        "download_humo_models.ps1 no longer fetches the engine default UNET "
        f"({eng_humo._HUMO_DEFAULT_UNET}) -- fresh installs will break")
    assert "humo_17B_fp8_e4m3fn.safetensors" not in text, (
        "the misaligned Comfy-Org 17B entry is back")
    m = re.search(r"url = '([^']*HuMo[^']*)'", text)
    assert m is not None and eng_humo._HUMO_DEFAULT_UNET in m.group(1), (
        "HuMo download URL does not point at the engine default artifact")


def test_humo_registry_labels_name_the_kijai_artifact_family():
    from nodes._otr_video_engines import registry as vreg

    caps = vreg.CAPABILITIES
    assert caps["humo"]["model_requirements"] == ["HuMo-14B-KJ"]
    assert caps["humo_14B_169"]["model_requirements"] == ["HuMo-14B-KJ"]


def test_detect_host_reports_mps(monkeypatch):
    """A (simulated) Apple-Silicon host is no longer invisible."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        pytest.skip("torch build has no mps backend module")
    monkeypatch.setattr(mps, "is_available", lambda: True)

    host = WorkflowValidator._detect_host()
    assert host["has_mps"] is True
    assert host["has_cuda"] is False
    assert host["vendor"] == "apple"
