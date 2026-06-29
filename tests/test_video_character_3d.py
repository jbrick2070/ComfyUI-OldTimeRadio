"""CPU tests for the character_3d dark scaffolds (UNREGISTERED, C3).

UNREGISTERED 2026-06-29 (C3 -- "registry IS the menu"): the three 3D talkers
(triposg_talk / hunyuan3d_talk / trellis_talk) render NotImplementedError, so
they are no longer registered/selectable and carry no CAPABILITIES row. The
SOURCE classes stay on disk (they return when a real forward ships): these tests
instantiate them DIRECTLY and prove the identity + the fail-closed ladders
(triposg: flag -> venv -> weights -> template(+hash) -> rhubarb -> blender, NEVER
the generated-mesh dir; hunyuan/trellis: flag -> venv -> mesh -> ARKit-52) +
NotImplementedError render are all intact, while the registry no longer lists
them. No GPU, no model load, no sidecar. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines.eng_character_3d import (
    Hunyuan3DTalkEngine,
    TripoSGTalkEngine,
    TrellisTalkEngine,
)


#: name -> SOURCE class (the scaffolds are unregistered, so the tests build them
#: directly instead of through the registry).
ENGINE_CLASSES = {
    "triposg_talk": TripoSGTalkEngine,
    "hunyuan3d_talk": Hunyuan3DTalkEngine,
    "trellis_talk": TrellisTalkEngine,
}
CHAR3D_ENGINES = tuple(ENGINE_CLASSES)
CHAR3D_ROLES = ("announcer_visual", "character_video")


def _eng(name):
    """Instantiate the scaffold's SOURCE class directly (it is unregistered)."""
    return ENGINE_CLASSES[name]()


# ---------------------------------------------------------------------------
# Unregistered contract (C3) + family identity (the family still exists)
# ---------------------------------------------------------------------------

def test_char3d_adapters_unregistered():
    # C3: dropped from the registry + the full dropdown; no CAPABILITIES row.
    for name in CHAR3D_ENGINES:
        assert not vreg.is_registered(name), f"{name} must be unregistered (C3)"
        assert name not in vreg.all_engine_names()
        assert name not in vreg.CAPABILITIES


def test_char3d_source_identity_intact():
    # The source classes keep their name + the character_3d family.
    for name in CHAR3D_ENGINES:
        eng = _eng(name)
        assert eng.name == name
        assert eng.family == "character_3d"


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
        eng = _eng(name)
        for role in CHAR3D_ROLES:
            assert role in eng.roles, f"{name} missing role {role}"


def test_char3d_default_roles_empty():
    """Both adapters are dark -- never a default for any role."""
    for name in CHAR3D_ENGINES:
        assert _eng(name).default_roles == ()


def test_char3d_required_inputs():
    for name in CHAR3D_ENGINES:
        eng = _eng(name)
        assert "audio_ref" in eng.required_inputs
        assert "init_image" in eng.required_inputs


# ---------------------------------------------------------------------------
# Fail-closed (flag -> venv -> mesh -> ARKit-52) -- source behavior preserved
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
    eng = _eng("hunyuan3d_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.GATED_BY_FLAG
    assert "OTR_ENABLE_CHARACTER_3D" in str(exc.value)


def test_trellis_fails_closed_no_flag(monkeypatch):
    _clear_3d_env(monkeypatch)
    eng = _eng("trellis_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert exc.value.reason is vreg.EngineUsabilityReason.GATED_BY_FLAG
    assert "OTR_ENABLE_TRELLIS_TALK" in str(exc.value)


def test_hunyuan3d_fails_closed_no_venv(monkeypatch, tmp_path):
    _clear_3d_env(monkeypatch)
    monkeypatch.setenv("OTR_ENABLE_CHARACTER_3D", "1")
    # venv env not set -> MISSING_MODEL
    eng = _eng("hunyuan3d_talk")
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
    eng = _eng("hunyuan3d_talk")
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
    eng = _eng("hunyuan3d_talk")
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
    eng = _eng("hunyuan3d_talk")
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
    eng = _eng("triposg_talk")
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
    eng = _eng("triposg_talk")
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
    eng = _eng("triposg_talk")
    assert eng.assert_usable({}, {}) == "triposg_talk"


def test_triposg_template_hash_pin_mismatch_fails_closed(monkeypatch, tmp_path):
    """When OTR_B_ARKIT_TEMPLATE_HASH is set, a sha256 mismatch FAILS CLOSED
    (a stale/wrong template never reaches the wrap)."""
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="blender")
    monkeypatch.setenv("OTR_B_ARKIT_TEMPLATE_HASH", "0" * 64)
    eng = _eng("triposg_talk")
    with pytest.raises(vreg.EngineUnusable) as exc:
        eng.assert_usable({}, {})
    assert "MISMATCH" in str(exc.value)


def test_triposg_template_hash_pin_match_passes(monkeypatch, tmp_path):
    import hashlib
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="blender")
    digest = hashlib.sha256(b"fake-npz").hexdigest()
    monkeypatch.setenv("OTR_B_ARKIT_TEMPLATE_HASH", digest)
    eng = _eng("triposg_talk")
    assert eng.assert_usable({}, {}) == "triposg_talk"


def test_triposg_recheck_every_attempt_no_cached_missing(monkeypatch, tmp_path):
    """7.1: re-check on EVERY attempt -- a prerequisite that appears between
    attempts flips the verdict (no cached "missing")."""
    _clear_3d_env(monkeypatch)
    _stage_triposg_assets(monkeypatch, tmp_path, upto="rhubarb")
    eng = _eng("triposg_talk")
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
        assert getattr(_eng(name), "requires_mesh_portrait", False) \
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
# load / render_clip raise NAMED errors (the "returns when built" guard)
# ---------------------------------------------------------------------------

def test_char3d_load_raises_runtime_error():
    for name in CHAR3D_ENGINES:
        eng = _eng(name)
        with pytest.raises(RuntimeError) as exc:
            eng.load()
        assert "dark scaffold" in str(exc.value).lower()


def test_char3d_render_clip_raises_not_implemented():
    for name in CHAR3D_ENGINES:
        eng = _eng(name)
        with pytest.raises(NotImplementedError):
            eng.render_clip({}, {})


# ---------------------------------------------------------------------------
# Fallback: an unregistered 3D name still terminates at the radio floor
# ---------------------------------------------------------------------------

def test_char3d_source_declares_humo_fallback():
    """The source classes keep fallback_engine='humo' (returns when built)."""
    for name in CHAR3D_ENGINES:
        assert _eng(name).fallback_engine == "humo"


def test_char3d_unregistered_name_floors_directly():
    """make_fallback_of resolves an UNREGISTERED 3D name straight to the
    universal radio floor (it is not in the registry and not in the soak
    overlay), so the chain still terminates LOUD rather than dangling."""
    from nodes._otr_video_engines.render_driver import make_fallback_of, FLOOR_NAMES
    from nodes._otr_video_engines import cheap_families  # noqa: F401 - registers floor

    fallback_fn = make_fallback_of()
    for name in CHAR3D_ENGINES:
        nxt = fallback_fn(name)
        assert nxt in FLOOR_NAMES, f"{name} did not floor: {nxt}"


# ---------------------------------------------------------------------------
# Cold-import clean (V-12)
# ---------------------------------------------------------------------------

def test_char3d_cold_import_no_heavy_libs():
    """Importing eng_character_3d must not pull in torch / diffusers / comfy."""
    import sys
    mod = sys.modules.get("nodes._otr_video_engines.eng_character_3d")
    assert mod is not None, "eng_character_3d not in sys.modules"
    mod_globals_heavy = [
        k for k in dir(mod)
        if k in ("torch", "diffusers", "transformers", "comfy")
    ]
    assert mod_globals_heavy == [], (
        f"eng_character_3d has heavy imports in module globals: {mod_globals_heavy}"
    )
