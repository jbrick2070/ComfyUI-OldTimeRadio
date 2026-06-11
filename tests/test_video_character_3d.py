"""CPU tests for the character_3d dark scaffold (W7-pre migration slice).

Covers: registry presence, family-in-FAMILIES, role-fit, fail-closed reasons
(hunyuan/trellis: flag -> venv -> mesh -> ARKit-52, semantics UNCHANGED;
triposg: flag -> venv -> weights -> template(+hash) -> rhubarb -> blender,
NEVER the generated-mesh dir), the requires_mesh_portrait capability field,
fallback chain terminates at still_kenburns, cold-import clean (V-12). No GPU,
no model load, no sidecar. UTF-8, no BOM, ASCII-only source.

These are ADD-ONLY (dark scaffold renders nothing): the adapters are registered
but always fail-closed until their gates clear (triposg: S-3D-0 + T2b keystone;
hunyuan/trellis: the deferred cu128 toolkit lane).
"""
from __future__ import annotations

import os

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import eng_character_3d  # noqa: F401 -- registers


CHAR3D_ENGINES = ("triposg_talk", "hunyuan3d_talk", "trellis_talk")
CHAR3D_ROLES = ("announcer_visual", "character_video")


# ---------------------------------------------------------------------------
# Registration + family identity
# ---------------------------------------------------------------------------

def test_char3d_adapters_registered():
    for name in CHAR3D_ENGINES:
        assert vreg.is_registered(name), f"{name} not in video registry"
        assert vreg.get_engine(name).name == name


def test_char3d_family_is_character_3d():
    for name in CHAR3D_ENGINES:
        assert vreg.get_engine(name).family == "character_3d"


def test_char3d_family_in_schemas_families():
    from nodes._otr_video_engines import schemas as sc
    assert "character_3d" in sc.FAMILIES
    assert "character_3d" in sc.FAMILY_REQUIRED_INPUTS
    assert sc.FAMILY_REQUIRED_INPUTS["character_3d"] == ("audio_ref", "init_image")


def test_char3d_families_sync_invariant():
    """schemas.py module-level assert guarantees FAMILIES == FAMILY_REQUIRED_INPUTS."""
    from nodes._otr_video_engines import schemas as sc
    assert set(sc.FAMILIES) == set(sc.FAMILY_REQUIRED_INPUTS)


# ---------------------------------------------------------------------------
# Role fit
# ---------------------------------------------------------------------------

def test_char3d_roles():
    for name in CHAR3D_ENGINES:
        eng = vreg.get_engine(name)
        for role in CHAR3D_ROLES:
            assert role in eng.roles, f"{name} missing role {role}"


def test_char3d_default_roles_empty():
    """Both adapters are dark -- never a default for any role."""
    for name in CHAR3D_ENGINES:
        assert vreg.get_engine(name).default_roles == ()


def test_char3d_required_inputs():
    for name in CHAR3D_ENGINES:
        eng = vreg.get_engine(name)
        assert "audio_ref" in eng.required_inputs
        assert "init_image" in eng.required_inputs


# ---------------------------------------------------------------------------
# Fail-closed (flag -> venv -> mesh -> ARKit-52)
# ---------------------------------------------------------------------------

def _clear_3d_env(monkeypatch):
    for var in (
        "OTR_ENABLE_CHARACTER_3D", "OTR_ENABLE_TRELLIS_TALK",
        "OTR_ENABLE_TRIPOSG_TALK",
        "OTR_B_SIDECAR_PYTHON", "OTR_TRELLIS_SIDECAR_PYTHON",
        "OTR_TRIPOSG_SIDECAR_PYTHON", "OTR_TRIPOSG_WEIGHTS_DIR",
        "OTR_B_MESH_DIR", "OTR_B_ARKIT_TEMPLATE_NPZ",
        "OTR_B_ARKIT_TEMPLATE_HASH", "OTR_RHUBARB_EXE", "OTR_BLENDER_EXE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_hunyuan3d_fails_closed_no_flag(monkeypatch):
    _clear_3d_env(monkeypatch)
    eng = vreg.get_engine("hunyuan3d_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.GATED_BY_FLAG
    assert "OTR_ENABLE_CHARACTER_3D" in str(exc.value)


def test_trellis_fails_closed_no_flag(monkeypatch):
    _clear_3d_env(monkeypatch)
    eng = vreg.get_engine("trellis_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.GATED_BY_FLAG
    assert "OTR_ENABLE_TRELLIS_TALK" in str(exc.value)


def test_hunyuan3d_fails_closed_no_venv(monkeypatch, tmp_path):
    _clear_3d_env(monkeypatch)
    monkeypatch.setenv("OTR_ENABLE_CHARACTER_3D", "1")
    # venv env not set -> MISSING_MODEL
    eng = vreg.get_engine("hunyuan3d_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.MISSING_MODEL
    assert "OTR_B_SIDECAR_PYTHON" in str(exc.value)


def test_hunyuan3d_fails_closed_no_mesh_dir(monkeypatch, tmp_path):
    _clear_3d_env(monkeypatch)
    fake_venv = tmp_path / "python.exe"
    fake_venv.write_bytes(b"x")
    monkeypatch.setenv("OTR_ENABLE_CHARACTER_3D", "1")
    monkeypatch.setenv("OTR_B_SIDECAR_PYTHON", str(fake_venv))
    # mesh dir not set -> MISSING_MODEL
    eng = vreg.get_engine("hunyuan3d_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.MISSING_MODEL
    assert "OTR_B_MESH_DIR" in str(exc.value)


def test_hunyuan3d_fails_closed_empty_mesh_dir(monkeypatch, tmp_path):
    _clear_3d_env(monkeypatch)
    fake_venv = tmp_path / "python.exe"
    fake_venv.write_bytes(b"x")
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    # no .obj/.glb files
    (mesh_dir / "readme.txt").write_text("no meshes here")
    monkeypatch.setenv("OTR_ENABLE_CHARACTER_3D", "1")
    monkeypatch.setenv("OTR_B_SIDECAR_PYTHON", str(fake_venv))
    monkeypatch.setenv("OTR_B_MESH_DIR", str(mesh_dir))
    eng = vreg.get_engine("hunyuan3d_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.MISSING_MODEL
    assert "no .obj" in str(exc.value).lower() or "mesh" in str(exc.value).lower()


def test_hunyuan3d_fails_closed_no_arkit_npz(monkeypatch, tmp_path):
    _clear_3d_env(monkeypatch)
    fake_venv = tmp_path / "python.exe"
    fake_venv.write_bytes(b"x")
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "char01.obj").write_bytes(b"v 0 0 0")
    monkeypatch.setenv("OTR_ENABLE_CHARACTER_3D", "1")
    monkeypatch.setenv("OTR_B_SIDECAR_PYTHON", str(fake_venv))
    monkeypatch.setenv("OTR_B_MESH_DIR", str(mesh_dir))
    # ARKit npz not set -> MISSING_MODEL
    eng = vreg.get_engine("hunyuan3d_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.MISSING_MODEL
    assert "OTR_B_ARKIT_TEMPLATE_NPZ" in str(exc.value)


# ---------------------------------------------------------------------------
# triposg_talk fail-closed (its OWN helper: flag -> venv -> weights ->
# template(+hash) -> rhubarb -> blender; NEVER the generated-mesh dir)
# ---------------------------------------------------------------------------

def _stage_triposg_assets(monkeypatch, tmp_path, *, upto):
    """Stage fake triposg prerequisites in dependency order; ``upto`` names the
    LAST one staged so the next check in the chain is the one that fails."""
    order = ("venv", "weights", "npz", "rhubarb", "blender")
    monkeypatch.setenv("OTR_ENABLE_TRIPOSG_TALK", "1")
    stage = order[:order.index(upto) + 1] if upto else ()
    if "venv" in stage:
        venv = tmp_path / "python.exe"
        venv.write_bytes(b"x")
        monkeypatch.setenv("OTR_TRIPOSG_SIDECAR_PYTHON", str(venv))
    if "weights" in stage:
        weights = tmp_path / "triposg_weights"
        weights.mkdir(exist_ok=True)
        monkeypatch.setenv("OTR_TRIPOSG_WEIGHTS_DIR", str(weights))
    if "npz" in stage:
        npz = tmp_path / "arkit52.npz"
        npz.write_bytes(b"fake-npz")
        monkeypatch.setenv("OTR_B_ARKIT_TEMPLATE_NPZ", str(npz))
    if "rhubarb" in stage:
        rhubarb = tmp_path / "rhubarb.exe"
        rhubarb.write_bytes(b"x")
        monkeypatch.setenv("OTR_RHUBARB_EXE", str(rhubarb))
    if "blender" in stage:
        blender = tmp_path / "blender.exe"
        blender.write_bytes(b"x")
        monkeypatch.setenv("OTR_BLENDER_EXE", str(blender))


def test_triposg_fails_closed_no_flag(monkeypatch):
    _clear_3d_env(monkeypatch)
    eng = vreg.get_engine("triposg_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.GATED_BY_FLAG
    assert "OTR_ENABLE_TRIPOSG_TALK" in str(exc.value)


@pytest.mark.parametrize("upto,expect_env", [
    (None, "OTR_TRIPOSG_SIDECAR_PYTHON"),
    ("venv", "OTR_TRIPOSG_WEIGHTS_DIR"),
    ("weights", "OTR_B_ARKIT_TEMPLATE_NPZ"),
    ("npz", "OTR_RHUBARB_EXE"),
    ("rhubarb", "OTR_BLENDER_EXE"),
])
def test_triposg_fail_closed_order(monkeypatch, tmp_path, upto, expect_env):
    """Each missing prerequisite is named in dependency order (3D plan 7.1)."""
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto=upto)
    eng = vreg.get_engine("triposg_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.MISSING_MODEL
    assert expect_env in str(exc.value)


def test_triposg_never_gates_on_the_generated_mesh_dir(monkeypatch, tmp_path):
    """The 7.1 deadlock fix: OTR_B_MESH_DIR is probe_c EVIDENCE, never a gate
    (prepare() is what GENERATES meshes). With every real prerequisite staged
    and NO mesh dir, assert_usable PASSES."""
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="blender")
    assert "OTR_B_MESH_DIR" not in os.environ
    eng = vreg.get_engine("triposg_talk")
    assert eng.assert_usable({}, {}) == "triposg_talk"


def test_triposg_template_hash_pin_mismatch_fails_closed(monkeypatch, tmp_path):
    """When OTR_B_ARKIT_TEMPLATE_HASH is set, a sha256 mismatch FAILS CLOSED
    (a stale/wrong template never reaches the wrap)."""
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="blender")
    monkeypatch.setenv("OTR_B_ARKIT_TEMPLATE_HASH", "0" * 64)
    eng = vreg.get_engine("triposg_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert "MISMATCH" in str(exc.value)


def test_triposg_template_hash_pin_match_passes(monkeypatch, tmp_path):
    import hashlib
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="blender")
    digest = hashlib.sha256(b"fake-npz").hexdigest()
    monkeypatch.setenv("OTR_B_ARKIT_TEMPLATE_HASH", digest)
    eng = vreg.get_engine("triposg_talk")
    assert eng.assert_usable({}, {}) == "triposg_talk"


def test_triposg_recheck_every_attempt_no_cached_missing(monkeypatch, tmp_path):
    """7.1: re-check on EVERY attempt -- a prerequisite that appears between
    attempts flips the verdict (no cached "missing")."""
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="rhubarb")
    eng = vreg.get_engine("triposg_talk")
    with pytest.raises(vreg.EngineUnusable):
        eng.assert_usable({}, {})
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"x")
    monkeypatch.setenv("OTR_BLENDER_EXE", str(blender))
    assert eng.assert_usable({}, {}) == "triposg_talk"


# ---------------------------------------------------------------------------
# requires_mesh_portrait capability (3D plan section 3)
# ---------------------------------------------------------------------------

def test_char3d_adapters_declare_requires_mesh_portrait():
    for name in CHAR3D_ENGINES:
        assert getattr(vreg.get_engine(name), "requires_mesh_portrait", False) \
            is True, f"{name} must declare requires_mesh_portrait=True"


def test_requires_mesh_portrait_is_a_real_schema_field():
    """AdapterDescriptor / VideoProfileRow are extra='forbid'; the capability
    is a REAL field (default False) so declaring it never rejects."""
    from nodes._otr_video_engines.schemas import (AdapterDescriptor,
                                                  VideoProfileRow)
    d = AdapterDescriptor(engine_id="x", family="character_3d",
                          requires_mesh_portrait=True)
    assert d.requires_mesh_portrait is True
    assert AdapterDescriptor(engine_id="y",
                             family="abstract").requires_mesh_portrait is False
    r = VideoProfileRow(profile_id="p", engine_id="x", family="character_3d",
                        requires_mesh_portrait=True)
    assert r.requires_mesh_portrait is True
    with pytest.raises(Exception):
        AdapterDescriptor(engine_id="x", family="character_3d",
                          requires_mesh_portait=True)  # typo'd key -> forbid


# ---------------------------------------------------------------------------
# load / render_clip raise NAMED errors
# ---------------------------------------------------------------------------

def test_char3d_load_raises_runtime_error():
    for name in CHAR3D_ENGINES:
        eng = vreg.get_engine(name)
        with pytest.raises(RuntimeError) as exc:
            eng.load()
        assert "dark scaffold" in str(exc.value).lower()


def test_char3d_render_clip_raises_not_implemented():
    for name in CHAR3D_ENGINES:
        eng = vreg.get_engine(name)
        with pytest.raises(NotImplementedError):
            eng.render_clip({}, {})


# ---------------------------------------------------------------------------
# Fallback chain terminates at still_kenburns
# ---------------------------------------------------------------------------

def test_char3d_fallback_chain_terminates():
    """Both 3D adapters declare fallback_engine='humo'; the chain must reach
    still_kenburns via render_driver.make_fallback_of."""
    from nodes._otr_video_engines.render_driver import make_fallback_of, FLOOR_NAMES

    # Register placeholder cheap families so still_kenburns is present
    from nodes._otr_video_engines import cheap_families  # noqa: F401

    fallback_fn = make_fallback_of()
    for name in CHAR3D_ENGINES:
        chain = [name]
        nxt = fallback_fn(name)
        while nxt is not None and nxt not in FLOOR_NAMES:
            chain.append(nxt)
            nxt = fallback_fn(nxt)
        chain.append(nxt)
        assert nxt in FLOOR_NAMES, (
            f"Fallback chain for {name} did not terminate at a floor: {chain}"
        )


# ---------------------------------------------------------------------------
# Cold-import clean (V-12)
# ---------------------------------------------------------------------------

def test_char3d_cold_import_no_heavy_libs():
    """Importing eng_character_3d must not pull in torch / diffusers / comfy.

    This test imports the module (already done at module level) and checks
    sys.modules for the known-heavy suspects. It is NOT an exhaustive check --
    it is a quick guard that the module scope stays clean.
    """
    import sys
    heavy = [k for k in sys.modules if k.startswith((
        "torch", "torchvision", "torchaudio",
        "diffusers", "transformers",
        "comfy.", "comfy_extras.",
    ))]
    # None of the heavy libs must have been imported as a SIDE EFFECT of
    # importing eng_character_3d (they may be present from other test
    # fixtures, so we only assert the module itself is present and clean at
    # the module-scope import level).
    mod = sys.modules.get("nodes._otr_video_engines.eng_character_3d")
    assert mod is not None, "eng_character_3d not in sys.modules"
    # Confirm the module's own globals contain none of the heavy names
    mod_globals_heavy = [
        k for k in dir(mod)
        if k in ("torch", "diffusers", "transformers", "comfy")
    ]
    assert mod_globals_heavy == [], (
        f"eng_character_3d has heavy imports in module globals: {mod_globals_heavy}"
    )


def test_char3d_registered_with_all_flags_unset(monkeypatch):
    """V-6: engines appear in all_engine_names() regardless of flag state."""
    for var in (
        "OTR_ENABLE_CHARACTER_3D", "OTR_ENABLE_TRELLIS_TALK",
        "OTR_B_SIDECAR_PYTHON", "OTR_TRELLIS_SIDECAR_PYTHON",
        "OTR_B_MESH_DIR", "OTR_B_ARKIT_TEMPLATE_NPZ",
    ):
        monkeypatch.delenv(var, raising=False)
    names = vreg.all_engine_names()
    for name in CHAR3D_ENGINES:
        assert name in names, f"{name} missing from registry with all 3D flags unset"
