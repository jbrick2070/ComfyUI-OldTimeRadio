"""C3 -- third image engine (Qwen-Image GGUF), generic peer, default-OFF. CPU tests.

Carries the model-agnostic image registry from >=2 engines (C2) to an OPEN set of
>=3, and proves a DIFFERENT integration mode behind the SAME protocol: Z-Image
(C2) runs in a cu128 sidecar; Qwen-Image GGUF rides the in-stack ComfyUI-GGUF
loader, so its fail-closed gate is the GGUF WEIGHTS file (OTR_QWEN_IMAGE_GGUF),
not a sidecar venv. Flux stays gen 1; Qwen-Image is an opt-in peer, greyed until
OTR_ENABLE_QWEN_IMAGE=1 AND its checkpoint exists. The live render is the operator
GPU smoke -- NOT covered here.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib

import pytest

from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines import qwen_image as qwi
from nodes._otr_shared import role_compat as rc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_registry_holds_three_engines():
    names = set(ireg.all_engine_names())
    assert {"flux_gen1", "z_image_turbo", "qwen_image"} <= names
    assert len(names) >= 3  # model-agnostic: an OPEN, growable set of image engines


def test_qwen_image_registry_round_trip():
    # registry round-trip: registered by import, fetchable, name matches.
    assert ireg.is_registered("qwen_image")
    assert "qwen_image" in ireg.all_engine_names()
    assert ireg.get_engine("qwen_image").name == "qwen_image"


def test_qwen_image_default_off_gated_by_flag(monkeypatch):
    monkeypatch.delenv(qwi.ENABLE_FLAG, raising=False)
    # default-OFF: the registry greys it (GATED_BY_FLAG) for a role it serves
    with pytest.raises(ireg.EngineUnusable) as ei:
        ireg.assert_usable("qwen_image", "announcer_visual")
    assert ei.value.reason is ireg.EngineUsabilityReason.GATED_BY_FLAG
    # flag ON -> the registry admits it (the deeper disk check is the adapter's)
    monkeypatch.setenv(qwi.ENABLE_FLAG, "1")
    assert ireg.assert_usable("qwen_image", "announcer_visual") == "qwen_image"


def test_flux_remains_gen1_default():
    # Flux is still the in-stack default everywhere; qwen_image is an OPT-IN peer.
    assert ireg.get_engine("qwen_image").default_roles == ()
    assert ireg.get_engine("flux_gen1").default_roles  # non-empty (gen-1 default)


def test_qwen_image_protocol_parity():
    eng = ireg.get_engine("qwen_image")
    for attr in ("name", "roles", "default_roles", "commercial_clean",
                 "requires_flag", "required_inputs"):
        assert hasattr(eng, attr)
    for meth in ("load", "unload", "assert_usable", "prepare", "render_image", "teardown"):
        assert callable(getattr(eng, meth))
    assert not hasattr(eng, "canonicalize")     # reduced prompt->image set (AS-4)
    assert eng.required_inputs == ("text_prompt",)
    assert eng.commercial_clean is True         # Apache-2.0 (per the C2 matrix)
    assert eng.requires_flag == "OTR_ENABLE_QWEN_IMAGE"


def test_qwen_image_role_filter_shared():
    descs = [
        {"engine_id": n, "roles": tuple(ireg.get_engine(n).roles),
         "required_inputs": tuple(getattr(ireg.get_engine(n), "required_inputs", ()))}
        for n in ireg.all_engine_names()
    ]
    # needs only text_prompt -> fits every role (incl. text-only background_abstract)
    assert "qwen_image" in rc.filter_engines_for_role("background_abstract", descs)
    assert "qwen_image" in rc.filter_engines_for_role("character_video", descs)


def test_qwen_image_adapter_assert_usable_fail_closed(monkeypatch):
    """The adapter's OWN assert_usable fails closed (MISSING_MODEL) until the GGUF
    checkpoint exists -- ABSENT/greyed, never a stub (BUG-046). The gate is the
    weights file (in-stack GGUF), not a sidecar venv."""
    monkeypatch.delenv(qwi.MODEL_ENV, raising=False)
    eng = ireg.get_engine("qwen_image")
    with pytest.raises(ireg.EngineUnusable) as ei:
        eng.assert_usable({}, {"role": "character_video"})
    assert ei.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL


def test_qwen_image_assert_usable_passes_when_ckpt_present(monkeypatch, tmp_path):
    """With a real checkpoint file on disk the adapter's disk check passes -- the
    fail-closed gate is genuinely the weights file, not a hardcoded raise."""
    ckpt = tmp_path / "qwen-image-Q4_K_M.gguf"
    ckpt.write_bytes(b"GGUF\x00\x00\x00\x00")  # presence only; the real load is the GPU smoke
    monkeypatch.setenv(qwi.MODEL_ENV, str(ckpt))
    eng = ireg.get_engine("qwen_image")
    assert eng.assert_usable({}, {"role": "character_video"}) == "qwen_image"


def test_qwen_image_render_is_operator_gpu_smoke():
    eng = ireg.get_engine("qwen_image")
    with pytest.raises(NotImplementedError):
        eng.render_image({"prompt": "x"}, {})


def test_qwen_image_cold_import_clean():
    code = (
        "import sys; import nodes._otr_image_engines.qwen_image;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
