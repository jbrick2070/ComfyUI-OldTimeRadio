"""C2 -- second image engine (Z-Image-Turbo), build-or-NO-GO, default-OFF. CPU tests.

Proves the image registry holds >=2 engines + the model-agnostic layer (Flux stays
gen 1; Z-Image is an opt-in peer, greyed until OTR_ENABLE_ZIMAGE=1 AND its cu128
sidecar exists). The live render is the operator cu128-sidecar GPU smoke -- NOT
covered here.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib

import pytest

from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines import z_image_turbo as zit
from nodes._otr_shared import role_compat as rc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_registry_holds_two_engines():
    names = set(ireg.all_engine_names())
    assert "flux_gen1" in names and "z_image_turbo" in names
    assert len(names) >= 2  # model-agnostic: the registry holds >=2 image engines


def test_z_image_selectable_no_flag(monkeypatch):
    # Registry IS the menu: z_image_turbo is selectable for a role it serves with
    # NO flag gate (the deeper disk/dep check is the adapter's).
    monkeypatch.delenv(zit.ENABLE_FLAG, raising=False)
    assert ireg.assert_usable("z_image_turbo", "announcer_visual") == "z_image_turbo"


def test_flux_remains_gen1_default():
    # Flux is still the in-stack default everywhere; z_image is an OPT-IN peer.
    assert ireg.get_engine("z_image_turbo").default_roles == ()
    assert ireg.get_engine("flux_gen1").default_roles  # non-empty (gen-1 default)


def test_z_image_protocol_parity():
    eng = ireg.get_engine("z_image_turbo")
    for attr in ("name", "roles", "default_roles", "commercial_clean",
                 "requires_flag", "required_inputs"):
        assert hasattr(eng, attr)
    for meth in ("load", "unload", "assert_usable", "prepare", "render_image", "teardown"):
        assert callable(getattr(eng, meth))
    assert not hasattr(eng, "canonicalize")     # reduced prompt->image set (AS-4)
    assert eng.required_inputs == ("text_prompt",)
    assert eng.commercial_clean is True         # Apache-2.0 (per the C2 matrix)
    assert eng.requires_flag is None            # registry IS the menu (no flag gate)


def test_z_image_role_filter_shared():
    descs = [
        {"engine_id": n, "roles": tuple(ireg.get_engine(n).roles),
         "required_inputs": tuple(getattr(ireg.get_engine(n), "required_inputs", ()))}
        for n in ireg.all_engine_names()
    ]
    # needs only text_prompt -> fits every role (incl. text-only background_abstract)
    assert "z_image_turbo" in rc.filter_engines_for_role("background_abstract", descs)
    assert "z_image_turbo" in rc.filter_engines_for_role("character_video", descs)


def test_z_image_adapter_assert_usable_fail_closed(monkeypatch):
    """The adapter's OWN assert_usable fails closed (MISSING_MODEL) until the
    diffusion-model file is configured -- ABSENT/greyed, never a stub (BUG-046).
    2026-06-18: in-process now (the stale cu128 sidecar was dropped), so the gate
    is the WEIGHTS env (MODEL_ENV), mirroring lumina_image."""
    monkeypatch.delenv(zit.MODEL_ENV, raising=False)
    eng = ireg.get_engine("z_image_turbo")
    with pytest.raises(ireg.EngineUnusable) as ei:
        eng.assert_usable({}, {"role": "character_video"})
    assert ei.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL


def test_z_image_graph_is_well_formed():
    """In-process build (2026-06-18): _zimage_params resolves the low-VRAM
    converged defaults and _build_zimage_graph emits the lumina-style split-file
    AuraFlow graph. CPU-pure (no torch/comfy); the live render is the operator
    GPU smoke."""
    eng = ireg.get_engine("z_image_turbo")
    params = eng._zimage_params({"prompt": "p", "seed": 7, "width": 832, "height": 1216})
    assert params["steps"] == 8 and params["cfg"] == 2.0 and params["shift"] == 3.0
    assert params["scheduler"] == "normal" and params["sampler_name"] == "euler"
    assert params["negative"]                       # live negative default
    assert params["width"] == 832 and params["height"] == 1216   # request dims honored

    class _W:
        def __init__(self, n, s): self.n, self.s = n, s

    graph = eng._build_zimage_graph(params, _W)
    assert set(graph) == {"unet", "clip", "vae", "sampling", "pos", "neg",
                          "latent", "ksampler", "decode"}
    assert graph["clip"]["inputs"]["type"] == params["clip_type"]
    assert graph["ksampler"]["inputs"]["steps"] == 8


def test_z_image_cold_import_clean():
    code = (
        "import sys; import nodes._otr_image_engines.z_image_turbo;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
