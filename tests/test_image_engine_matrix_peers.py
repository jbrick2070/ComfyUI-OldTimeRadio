"""Matrix image-engine peers (C4+) -- parametrized CPU contract suite.

Beyond Flux gen-1 (default), Z-Image (C2) and Qwen-Image (C3, each with their own
file), the C2 dep/license matrix lists further candidates. Each is scaffolded as
an OPT-IN peer (default-OFF, fail-closed, no model "primary") that registers
identically -- proving the model-agnostic image layer holds an OPEN, growable set
and honours each engine's real license. This data-driven suite asserts the SHARED
contract for every such peer so adding one is a single table row, not a new file.

The live render of every peer is the operator GPU smoke (NotImplementedError) --
NOT covered here. ``commercial_clean`` is asserted against the matrix's license
column (MIT / Apache -> True; conditional / verify -> False), kept HONEST.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import pathlib

import pytest

from nodes._otr_image_engines import registry as ireg
from nodes._otr_shared import role_compat as rc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# (adapter_module, engine_name, enable_flag, model_env, expected_commercial_clean)
# One row per matrix peer beyond C2/C3. Extend with new candidates here only.
# Clean-license (MIT/Apache) -> True; conditional/unconfirmed license -> False.
PEERS = [
    ("hidream_i1", "hidream_i1", "OTR_ENABLE_HIDREAM", "OTR_HIDREAM_GGUF", True),
    # chroma_hd DROPPED 2026-06-18 (operator: it is a de-restricted/uncensored
    # FLUX finetune; OTR will not ship a path to that content).
    # lumina_image GRADUATED to a BUILT engine (real render_image) 2026-06-14 --
    # it now has its own suite (test_lumina_image_engine.py), like flux_gen1, and
    # is no longer a NotImplementedError stub peer (so it leaves this matrix).
    ("sd35_large", "sd35_large", "OTR_ENABLE_SD35", "OTR_SD35_CKPT", False),
    # flux2_klein GRADUATED to a BUILT engine (real render_image) 2026-06-18 -- it
    # now has its own suite (test_flux2_klein_engine.py), like lumina_image, and is
    # no longer a NotImplementedError stub peer (so it leaves this matrix).
]

PEER_IDS = [p[1] for p in PEERS]


def test_registry_holds_full_peer_set():
    """The registry holds Flux gen-1 + C2/C3 + every matrix peer -- an open set."""
    names = set(ireg.all_engine_names())
    base = {"flux_gen1", "z_image_turbo", "qwen_image"}
    assert base <= names
    assert {p[1] for p in PEERS} <= names
    assert len(names) >= len(base) + len(PEERS)


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_registry_round_trip(mod, name, flag, model_env, clean):
    assert ireg.is_registered(name)
    assert name in ireg.all_engine_names()
    assert ireg.get_engine(name).name == name


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_selectable_no_flag(monkeypatch, mod, name, flag, model_env, clean):
    # Registry IS the menu: a registered peer is selectable for a role it serves
    # with NO flag gate (the deeper disk check is the adapter's).
    monkeypatch.delenv(flag, raising=False)
    assert ireg.assert_usable(name, "announcer_visual") == name


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_is_opt_in_not_default(mod, name, flag, model_env, clean):
    # no matrix peer is "primary" -- Flux stays the in-stack default everywhere.
    assert ireg.get_engine(name).default_roles == ()
    assert ireg.get_engine("flux_gen1").default_roles  # non-empty


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_protocol_parity(mod, name, flag, model_env, clean):
    eng = ireg.get_engine(name)
    for attr in ("name", "roles", "default_roles", "commercial_clean",
                 "requires_flag", "required_inputs"):
        assert hasattr(eng, attr)
    for meth in ("load", "unload", "assert_usable", "prepare", "render_image", "teardown"):
        assert callable(getattr(eng, meth))
    assert not hasattr(eng, "canonicalize")     # reduced prompt->image set (AS-4)
    assert eng.required_inputs == ("text_prompt",)
    assert eng.requires_flag == flag


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_commercial_clean_honest(mod, name, flag, model_env, clean):
    """commercial_clean matches the matrix license column -- never optimistic."""
    eng = ireg.get_engine(name)
    assert isinstance(eng.commercial_clean, bool)
    assert eng.commercial_clean is clean


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_role_filter_shared(mod, name, flag, model_env, clean):
    descs = [
        {"engine_id": n, "roles": tuple(ireg.get_engine(n).roles),
         "required_inputs": tuple(getattr(ireg.get_engine(n), "required_inputs", ()))}
        for n in ireg.all_engine_names()
    ]
    assert name in rc.filter_engines_for_role("background_abstract", descs)
    assert name in rc.filter_engines_for_role("character_video", descs)


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_assert_usable_fail_closed(monkeypatch, mod, name, flag, model_env, clean):
    """The adapter's OWN assert_usable fails closed (MISSING_MODEL) until its
    weights exist -- ABSENT/greyed, never a stub (BUG-046)."""
    monkeypatch.delenv(model_env, raising=False)
    eng = ireg.get_engine(name)
    with pytest.raises(ireg.EngineUnusable) as ei:
        eng.assert_usable({}, {"role": "character_video"})
    assert ei.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_assert_usable_passes_with_weights(monkeypatch, tmp_path, mod, name, flag, model_env, clean):
    """With the weights file present the disk check passes -- the gate is genuinely
    the file, not a hardcoded raise."""
    ckpt = tmp_path / f"{name}.weights"
    ckpt.write_bytes(b"\x00\x01\x02\x03")
    monkeypatch.setenv(model_env, str(ckpt))
    eng = ireg.get_engine(name)
    assert eng.assert_usable({}, {"role": "character_video"}) == name


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_render_is_operator_gpu_smoke(mod, name, flag, model_env, clean):
    eng = ireg.get_engine(name)
    with pytest.raises(NotImplementedError):
        eng.render_image({"prompt": "x"}, {})


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_env_constants_exported(mod, name, flag, model_env, clean):
    """The adapter module exports its flag + model-env constants (the operator's
    contract for enabling it)."""
    m = importlib.import_module(f"nodes._otr_image_engines.{mod}")
    assert m.ENABLE_FLAG == flag
    assert m.MODEL_ENV == model_env


@pytest.mark.parametrize("mod,name,flag,model_env,clean", PEERS, ids=PEER_IDS)
def test_peer_cold_import_clean(mod, name, flag, model_env, clean):
    code = (
        f"import sys; import nodes._otr_image_engines.{mod};"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
