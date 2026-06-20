"""0-E tickets E-1..E-6 CPU tests -- mesh_stage (the traditional 3D chain).

Blender + the hy3d checkpoint are absent in the pytest sandbox, so every
test exercises the COLD path: registration (selectable-not-default, the
E-7 license posture), the fail-closed usability ladder (flag -> Blender
exe -> hy3d ckpt; NEVER the cache dir -- the 7.1 deadlock lesson), the E-2
cache key + manifest, the E-3 command/env builders, the E-4 validate +
ATOMIC publish + stale-tmp sweep, and the Track-3 directory-clip
canonicalize enforcement. The live mesher + Blender render + VRAM probe is
the GPU/operator step (E-1/E-6), NOT covered here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import schemas as sc
from nodes._otr_video_engines import eng_mesh_stage as ms
from nodes._otr_video_engines.eng_mesh_stage import (
    MeshStageEngine,
    build_blender_cmd,
    build_blender_env,
    build_mesh_manifest,
    cleanup_stale_tmp,
    mesh_cache_key,
    portrait_content_hash,
    publish_frame_dir,
    validate_frame_dir,
    write_manifest,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STAGE_SRC = REPO_ROOT / "scripts" / "otr_mesh_stage_blender.py"


def _write_png(path, w=16, h=16, mode="RGBA"):
    from PIL import Image
    Image.new(mode, (w, h), (200, 60, 60, 255) if mode == "RGBA"
              else (200, 60, 60)).save(path)
    return path


# --------------------------------------------------------------------------- #
# Registration: selectable-not-default + the E-5/E-7 posture
# --------------------------------------------------------------------------- #
def test_mesh_stage_registered_selectable_not_default():
    assert vreg.is_registered("mesh_stage")
    eng = vreg.get_engine("mesh_stage")
    assert isinstance(eng, MeshStageEngine)
    # E-5: the family token is a REAL schemas.FAMILIES member and is NOT the
    # talking-3D family (never overload triposg_talk / character_3d).
    assert eng.family == "image_to_video" and eng.family in sc.FAMILIES
    assert eng.family != "character_3d"
    assert eng.default_roles == ()             # NEVER a default until look-QA
    assert eng.requires_flag == "OTR_ENABLE_MESH_STAGE"
    assert eng.required_inputs == ("init_image",)
    assert eng.commercial_clean is False       # Tencent license -- E-7 gate
    assert eng.fallback_engine == "still_parallax"   # E-5 chain head
    assert "mesh_stage" in vreg.all_engine_names()   # full dropdown (V-6)
    assert "no lip-sync" in eng.honest_label   # camera motion only (tier-1)
    for role in rc.ROLES:                      # never the default for ANY role
        assert vreg.default_engine_for_role(role) != "mesh_stage"


def test_mesh_stage_roles_exactly_the_three_e5_slots():
    eng = vreg.get_engine("mesh_stage")
    assert eng.roles == ("music_visual", "announcer_visual", "character_video")
    desc = {"engine_id": "mesh_stage", "roles": eng.roles,
            "required_inputs": eng.required_inputs}
    from nodes.otr_video_director import VIDEO_SLOT_ROLES
    for slot, slot_roles in VIDEO_SLOT_ROLES.items():
        assert any(rc.engine_fits_role(desc, role) for role in slot_roles), slot
    assert rc.engine_fits_role(desc, "background_abstract") is False
    assert rc.engine_fits_role(desc, "scene_broll") is False


def test_registry_gates_mesh_stage_until_flag(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_MESH_STAGE", raising=False)
    eng = vreg.get_engine("mesh_stage")
    for role in eng.roles:
        with pytest.raises(vreg.EngineUnusable):
            vreg.assert_usable("mesh_stage", role)
    monkeypatch.setenv("OTR_ENABLE_MESH_STAGE", "1")
    for role in eng.roles:
        assert vreg.assert_usable("mesh_stage", role) == "mesh_stage"


def test_mesh_stage_assert_usable_flag_blender_then_ckpt(monkeypatch, tmp_path):
    eng = vreg.get_engine("mesh_stage")
    monkeypatch.delenv("OTR_ENABLE_MESH_STAGE", raising=False)
    with pytest.raises(vreg.EngineUnusable) as e1:
        eng.assert_usable(host_caps={}, profile={})
    assert e1.value.reason == vreg.EngineUsabilityReason.GATED_BY_FLAG
    # Flag on, Blender missing -> MISSING_MODEL naming OTR_BLENDER_EXE.
    monkeypatch.setenv("OTR_ENABLE_MESH_STAGE", "1")
    monkeypatch.setenv("OTR_BLENDER_EXE", str(tmp_path / "absent" / "blender.exe"))
    with pytest.raises(vreg.EngineUnusable) as e2:
        eng.assert_usable(host_caps={}, profile={})
    assert e2.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL
    assert "OTR_BLENDER_EXE" in str(e2.value)
    # Blender present, hy3d ckpt missing -> MISSING_MODEL naming the ckpt.
    fake_blender = tmp_path / "blender.exe"
    fake_blender.write_bytes(b"MZ")
    monkeypatch.setenv("OTR_BLENDER_EXE", str(fake_blender))
    monkeypatch.setenv("OTR_HY3D_CKPT", str(tmp_path / "no_ckpt.safetensors"))
    with pytest.raises(vreg.EngineUnusable) as e3:
        eng.assert_usable(host_caps={}, profile={})
    assert e3.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL
    assert "OTR_HY3D_CKPT" in str(e3.value)
    # All three present -> usable. The ladder NEVER gates on the cache dir
    # (meshes are GENERATED at render time -- the 7.1 deadlock lesson).
    ckpt = tmp_path / "hy3d.safetensors"
    ckpt.write_bytes(b"\x00" * 8)
    monkeypatch.setenv("OTR_HY3D_CKPT", str(ckpt))
    monkeypatch.setenv("OTR_MESH_CACHE_DIR", str(tmp_path / "never_created"))
    assert eng.assert_usable(host_caps={}, profile={}) == "mesh_stage"


# --------------------------------------------------------------------------- #
# E-2 -- cache key + manifest (pure)
# --------------------------------------------------------------------------- #
def test_portrait_content_hash_is_content_keyed(tmp_path):
    a = tmp_path / "a.png"
    a.write_bytes(b"pixels-1")
    b = tmp_path / "b.png"                     # same CONTENT, different path
    b.write_bytes(b"pixels-1")
    c = tmp_path / "c.png"
    c.write_bytes(b"pixels-2")
    assert portrait_content_hash(str(a)) == portrait_content_hash(str(b))
    assert portrait_content_hash(str(a)) != portrait_content_hash(str(c))


def test_mesh_cache_key_shape_and_regen_trap():
    sha = "ab" * 32
    key = mesh_cache_key("Announcer Joe", sha)
    assert key == "Announcer_Joe__%s__hy3d2mv_v1" % sha[:16]
    # Same character + same CANONICAL portrait -> the SAME key (episode reuse
    # within episode; stable cast across episodes -- the regen trap stays shut).
    assert mesh_cache_key("Announcer Joe", sha) == key
    # A different mesher id/version NEVER collides (mesher swappable).
    assert mesh_cache_key("Announcer Joe", sha, "triposr", "2") != key
    assert mesh_cache_key("", sha).startswith("uncast__")
    with pytest.raises(ValueError):
        mesh_cache_key("x", "not-a-sha")       # content hash is mandatory


def test_manifest_roundtrip(tmp_path):
    sha = "cd" * 32
    man = build_mesh_manifest("joe", sha, seed=7)
    assert man["schema_version"] == 1
    assert man["mesher_id"] == "hy3d2mv" and man["mesher_version"] == "1"
    assert man["source_portrait_sha256"] == sha and man["seed"] == 7
    assert "Tencent" in man["license"]         # E-7: operator-visible record
    p = tmp_path / "m.manifest.json"
    write_manifest(man, str(p))
    assert json.loads(p.read_text(encoding="utf-8")) == man
    assert not (tmp_path / "m.manifest.json.tmp").exists()   # atomic


# --------------------------------------------------------------------------- #
# E-3 -- command + sanitized env (pure)
# --------------------------------------------------------------------------- #
def test_build_blender_cmd_shape():
    cmd = build_blender_cmd("C:/tools/blender.exe", "C:/m.glb", "C:/out",
                            169, 1472, 832, 42)
    assert cmd[0] == "C:/tools/blender.exe"
    assert "--background" in cmd and "--factory-startup" in cmd
    i = cmd.index("--python-exit-code")        # fail-closed: stage exc -> exit 1
    assert cmd[i + 1] == "1"
    assert "--python" in cmd and "--" in cmd
    tail = cmd[cmd.index("--") + 1:]
    assert tail[tail.index("--frames") + 1] == "169"
    assert tail[tail.index("--width") + 1] == "1472"     # E-6 explicit canvas
    assert tail[tail.index("--height") + 1] == "832"
    assert tail[tail.index("--seed") + 1] == "42"        # E-6 reproducibility
    assert tail[tail.index("--render-engine") + 1] == "WORKBENCH"   # v1 mode
    assert tail[tail.index("--mode") + 1] == "render"


def test_build_blender_env_sanitized():
    base = {"PATH": "C:/x", "PYTHONPATH": "C:/venv/Lib", "PYTHONHOME": "C:/py",
            "VIRTUAL_ENV": "C:/venv", "HF_HOME": "C:/models", "FOO": "1"}
    env = build_blender_env(base)
    for leak in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        assert leak not in env                 # venv never poisons Blender's py
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PATH"] == "C:/x" and env["FOO"] == "1"   # the rest passes thru
    assert base["PYTHONPATH"] == "C:/venv/Lib"           # pure: input untouched


# --------------------------------------------------------------------------- #
# E-4 -- validate + ATOMIC publish + stale-tmp sweep
# --------------------------------------------------------------------------- #
def test_validate_frame_dir_count_dims_rgba(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(3):
        _write_png(str(d / ("frame_%04d.png" % (i + 1))), 32, 24)
    frames = validate_frame_dir(str(d), 3, 32, 24)
    assert len(frames) == 3
    with pytest.raises(ValueError):            # count != ledger target
        validate_frame_dir(str(d), 4, 32, 24)
    with pytest.raises(ValueError):            # dims != explicit canvas
        validate_frame_dir(str(d), 3, 64, 24)
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for i in range(2):
        _write_png(str(rgb / ("frame_%04d.png" % (i + 1))), 32, 24, mode="RGB")
    with pytest.raises(ValueError):            # straight-alpha RGBA required
        validate_frame_dir(str(rgb), 2, 32, 24)


def test_publish_frame_dir_atomic_and_collision(tmp_path):
    tmp_render = tmp_path / "otr_mesh_tmp_abc"
    tmp_render.mkdir()
    _write_png(str(tmp_render / "frame_0001.png"))
    final = tmp_path / "stage_final"
    assert publish_frame_dir(str(tmp_render), str(final)) == str(final)
    assert final.is_dir() and not tmp_render.exists()    # a RENAME, not a copy
    assert (final / "frame_0001.png").exists()
    again = tmp_path / "otr_mesh_tmp_def"
    again.mkdir()
    with pytest.raises(FileExistsError):       # single-writer: never merge
        publish_frame_dir(str(again), str(final))


def test_cleanup_stale_tmp_sweeps_only_old_prefixed(tmp_path):
    old = tmp_path / "otr_mesh_tmp_old"
    old.mkdir()
    fresh = tmp_path / "otr_mesh_tmp_fresh"
    fresh.mkdir()
    other = tmp_path / "keep_me"
    other.mkdir()
    stale_t = time.time() - 48 * 3600
    os.utime(old, (stale_t, stale_t))
    removed = cleanup_stale_tmp(str(tmp_path), max_age_hours=24.0)
    assert [os.path.basename(p) for p in removed] == ["otr_mesh_tmp_old"]
    assert fresh.exists() and other.exists() and not old.exists()
    assert cleanup_stale_tmp(str(tmp_path / "missing")) == []


# --------------------------------------------------------------------------- #
# LOUD missing inputs + the directory-clip canonicalize enforcement
# --------------------------------------------------------------------------- #
def test_render_clip_missing_still_raises():
    eng = vreg.get_engine("mesh_stage")
    req = {"shot_id": "s1", "canvas": {"w": 64, "h": 48, "fps": 25},
           "timing": {"target_frame_count": 3}, "asset_refs": {}}
    with pytest.raises(FileNotFoundError):
        eng.render_clip(req)
    assert rd.classify_failure(FileNotFoundError("x")).name == "DEPENDENCY_MISSING"


def test_canonicalize_enforces_directory_contract(tmp_path):
    eng = vreg.get_engine("mesh_stage")
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(3):
        _write_png(str(d / ("frame_%04d.png" % (i + 1))), 16, 16)
    clip = eng._directory_clip({"shot_id": "s7"}, str(d), 25, 3)
    assert clip["type"] == "directory" and clip["alpha"] == "straight"
    assert clip["pixel_format"] == "rgba" and clip["has_audio"] is False
    assert clip["engine_id"] == "mesh_stage"
    out = eng.canonicalize(clip, {"timing": {"target_frame_count": 3}}, {})
    assert out is clip                         # validated, unchanged
    with pytest.raises(ValueError):            # ledger target mismatch
        eng.canonicalize(clip, {"timing": {"target_frame_count": 4}}, {})
    bad = dict(clip, frame_count=2)            # declared != disk
    with pytest.raises(ValueError):
        eng.canonicalize(bad, {"timing": {"target_frame_count": 2}}, {})


def test_default_canvas_is_explicit_1472x832():
    # E-6: when the request leaves the canvas unset, mesh_stage uses the
    # EXPLICIT landscape contract -- never the silent cheap-family 832x480.
    assert (ms.DEFAULT_W, ms.DEFAULT_H) == (1472, 832)


# --------------------------------------------------------------------------- #
# Driver maps + capability row (BOTH copies; the 0-E wiring contract)
# --------------------------------------------------------------------------- #
def test_engine_family_fallback_chain_both_copies():
    assert rd.ENGINE_FAMILY["mesh_stage"] == "image_to_video"
    assert rd.engine_family("mesh_stage") == "image_to_video"
    fb = rd.make_fallback_of()
    # E-5: the LOUD chain mesh_stage -> still_parallax -> still_kenburns.
    assert fb("mesh_stage") == "still_parallax"
    assert fb("still_parallax") == "still_kenburns"
    assert fb("still_kenburns") is None        # the floor terminates
    soak_src = REPO_ROOT / "scripts" / "otr_video_soak.py"
    spec = importlib.util.spec_from_file_location("otr_video_soak", soak_src)
    soak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soak)
    assert soak.ENGINE_FAMILY["mesh_stage"] == "image_to_video"


def test_capability_row_draft_probe_pending():
    row = vreg.CAPABILITIES["mesh_stage"]
    assert row["cpu_ok"] is False              # the mesher needs the GPU
    assert row["vram_class"] == "medium"       # DRAFT pending the E-1 probe
    assert row["required_toolchain"] is None   # compile-free core nodes
    assert row["requires_sidecar"] is False    # in-process (NOT the cu128 lane)
    assert "hunyuan3d-dit-v2-mv" in row["model_requirements"]
    assert "blender-portable" in row["model_requirements"]


# --------------------------------------------------------------------------- #
# The Blender stage script: pure parser + ASCII (the bpy parts never import)
# --------------------------------------------------------------------------- #
def _load_stage():
    spec = importlib.util.spec_from_file_location("otr_mesh_stage_blender",
                                                  STAGE_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)               # module scope imports NO bpy
    return mod


def test_stage_script_parser_contract():
    stage = _load_stage()
    a = stage.parse_stage_args(
        ["--mode", "render", "--glb", "C:/m.glb", "--out", "C:/o",
         "--frames", "169", "--width", "1472", "--height", "832",
         "--seed", "9"])
    assert (a.frames, a.width, a.height, a.seed) == (169, 1472, 832, 9)
    assert a.render_engine == "WORKBENCH"      # the ONE v1 mode
    assert a.radius == 2.5 and a.threads == 4  # fixed-thread determinism pin
    with pytest.raises(SystemExit):            # render mode REQUIRES the glb
        stage.parse_stage_args(["--mode", "render", "--out", "C:/o",
                                "--frames", "3", "--width", "64",
                                "--height", "64"])
    with pytest.raises(SystemExit):            # EEVEE banned v1 headless
        stage.parse_stage_args(["--mode", "selftest", "--out", "C:/o",
                                "--frames", "3", "--width", "64",
                                "--height", "64", "--render-engine", "EEVEE"])
    with pytest.raises(SystemExit):            # frame floor
        stage.parse_stage_args(["--mode", "selftest", "--out", "C:/o",
                                "--frames", "0", "--width", "64",
                                "--height", "64"])
    sf = stage.parse_stage_args(["--mode", "selftest", "--out", "C:/o",
                                 "--frames", "3", "--width", "256",
                                 "--height", "256"])
    assert sf.mode == "selftest" and sf.glb == ""


def test_stage_script_parser_portrait_and_arc():
    stage = _load_stage()
    a = stage.parse_stage_args(
        ["--mode", "render", "--glb", "C:/m.glb", "--out", "C:/o",
         "--frames", "169", "--width", "1472", "--height", "832",
         "--portrait", "C:/p.png", "--start-angle", "10", "--arc-degrees", "30"])
    assert a.portrait == "C:/p.png"
    assert a.start_angle == 10.0 and a.arc_degrees == 30.0
    # Defaults: no portrait, the bounded-arc defaults.
    b = stage.parse_stage_args(
        ["--mode", "render", "--glb", "C:/m.glb", "--out", "C:/o",
         "--frames", "3", "--width", "64", "--height", "64"])
    assert b.portrait == "" and b.start_angle == 0.0 and b.arc_degrees == 45.0


def test_arc_keyframes_single_and_bounded_sweep():
    stage = _load_stage()
    import math
    # frames==1 -> ONE keyframe at frame 1 (a still hero shot, no fcurve).
    assert stage.arc_keyframes(1, 0.0, 45.0) == [(1, 0.0)]
    keys = stage.arc_keyframes(3, 0.0, 30.0)
    assert [k[0] for k in keys] == [1, 2, 3]                # frames 1..N
    assert keys[0][1] == 0.0                                # start
    assert math.isclose(keys[-1][1], math.radians(30.0))    # start+arc
    assert math.isclose(keys[1][1], math.radians(15.0))     # LINEAR midpoint
    # The arc is CLAMPED to +/-MAX_ARC_DEGREES (the unpainted back never shows).
    over = stage.arc_keyframes(2, 0.0, 90.0)
    assert math.isclose(over[-1][1], math.radians(stage.MAX_ARC_DEGREES))
    assert stage.clamp_arc_degrees(90.0) == stage.MAX_ARC_DEGREES
    assert stage.clamp_arc_degrees(-90.0) == -stage.MAX_ARC_DEGREES
    assert stage.clamp_arc_degrees(20.0) == 20.0


def test_project_uv_front_view_mapping():
    stage = _load_stage()
    # origin -> image center.
    assert stage.project_uv(0.0, 0.0) == (0.5, 0.5)
    # +Y -> right; +Z -> up (t toward 0, the top).
    u_right, _ = stage.project_uv(0.4, 0.0)
    assert u_right > 0.5
    _, t_top = stage.project_uv(0.0, 0.4)
    assert t_top < 0.5
    # Out-of-frame coords clamp into [0, 1].
    assert stage.project_uv(5.0, -5.0) == (1.0, 1.0)
    assert stage.project_uv(-5.0, 5.0) == (0.0, 0.0)


def test_sample_image_nearest_and_flip():
    stage = _load_stage()
    # 2x2 RGBA, bottom-origin: bottom-left red, bottom-right green,
    # top-left blue, top-right white.
    px = [1, 0, 0, 1,   0, 1, 0, 1,    # bottom row (y=0)
          0, 0, 1, 1,   1, 1, 1, 1]    # top row    (y=1)
    # t=1 is the BOTTOM of the image (flip), u=0 -> bottom-left red.
    assert stage.sample_image(px, 2, 2, 0.0, 1.0) == (1.0, 0.0, 0.0)
    # t=0 is the TOP, u=0 -> top-left blue.
    assert stage.sample_image(px, 2, 2, 0.0, 0.0) == (0.0, 0.0, 1.0)
    # t=0 top, u=1 -> top-right white.
    assert stage.sample_image(px, 2, 2, 1.0, 0.0) == (1.0, 1.0, 1.0)


def test_build_blender_cmd_portrait_and_arc_appended():
    # Legacy invocation (no portrait) is byte-identical -- no extra tokens.
    base = build_blender_cmd("C:/b.exe", "C:/m.glb", "C:/o", 3, 64, 64, 0)
    assert "--portrait" not in base
    assert "--start-angle" not in base and "--arc-degrees" not in base
    # Portrait + bounded arc appended at the END (positional widgets untouched).
    cmd = build_blender_cmd("C:/b.exe", "C:/m.glb", "C:/o", 3, 64, 64, 0,
                            portrait="C:/p.png", start_angle=10.0,
                            arc_degrees=30.0)
    assert cmd[cmd.index("--portrait") + 1] == "C:/p.png"
    assert cmd[cmd.index("--start-angle") + 1] == "10.0"
    assert cmd[cmd.index("--arc-degrees") + 1] == "30.0"
    # The frozen positional tail is unchanged.
    tail = cmd[cmd.index("--") + 1:]
    assert tail[tail.index("--frames") + 1] == "3"
    assert tail[tail.index("--render-engine") + 1] == "WORKBENCH"


def test_stage_and_engine_sources_ascii_no_em_dash():
    for src_path in (STAGE_SRC,
                     REPO_ROOT / "nodes" / "_otr_video_engines"
                     / "eng_mesh_stage.py"):
        src = src_path.read_text(encoding="utf-8")
        assert "—" not in src             # em-dash forbidden (CLAUDE.md)
        src.encode("ascii")                    # ASCII-only source


# --------------------------------------------------------------------------- #
# Cold-import (V-12)
# --------------------------------------------------------------------------- #
def test_cold_import_mesh_stage_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_mesh_stage;"
        "heavy=[m for m in ('torch','transformers','diffusers') "
        "if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


# --------------------------------------------------------------------------- #
# A4 schema audit (2026-06-11): explicit widgets vs the INSTALLED core nodes
# --------------------------------------------------------------------------- #
def test_mesh_graph_passes_every_required_widget_explicitly():
    """The installed hy3d nodes are V3 IO.ComfyNode classes whose
    EXECUTE_NORMALIZED backfills NO schema defaults, and core
    CLIPVisionEncode requires its crop combo -- so the declarative graph
    must carry every required widget EXPLICITLY. crop / num_chunks /
    octree_resolution / algorithm were the 2026-06-11 A4 audit gaps; this
    pins them against regression."""
    eng = vreg.get_engine("mesh_stage")
    graph = eng._build_mesh_graph("portrait.png", 7, "prefix")
    assert graph["cv_encode"]["inputs"]["crop"] == "center"
    assert graph["vaedecode"]["inputs"]["num_chunks"] == 8000
    assert graph["vaedecode"]["inputs"]["octree_resolution"] == 256
    assert graph["voxeltomesh"]["inputs"]["algorithm"] == "surface net"
    assert graph["voxeltomesh"]["inputs"]["threshold"] == 0.6
    # KSampler keeps its full required widget set (already complete).
    ks = graph["ksampler"]["inputs"]
    for widget in ("seed", "steps", "cfg", "sampler_name", "scheduler",
                   "denoise"):
        assert widget in ks


def test_mesh_graph_official_blueprint_wiring():
    """A4 audit: the loader is ImageOnlyCheckpointLoader (the all-in-one
    hy3d checkpoint embeds the DINO image encoder + ShapeVAE -- verified
    vae./conditioner.main_image_encoder. keys in the safetensors header),
    the encoder comes from checkpoint slot 1 (NOT a CLIPVisionLoader
    file), and the MODEL is patched by ModelSamplingAuraFlow shift=1
    before the KSampler -- the shipped official blueprint wiring."""
    eng = vreg.get_engine("mesh_stage")
    cands = eng._node_candidates()
    assert cands["checkpoint"] == ("ImageOnlyCheckpointLoader",)
    assert "clipvision" not in cands           # embedded encoder, no loader
    assert cands["modelsampling"] == ("ModelSamplingAuraFlow",)
    graph = eng._build_mesh_graph("portrait.png", 7, "prefix")
    assert "clipvision" not in graph
    cv = graph["cv_encode"]["inputs"]["clip_vision"]
    assert (cv.src, cv.slot) == ("checkpoint", 1)
    vae = graph["vaedecode"]["inputs"]["vae"]
    assert (vae.src, vae.slot) == ("checkpoint", 2)
    msl = graph["modelsampling"]["inputs"]
    assert msl["shift"] == 1.0
    model = graph["ksampler"]["inputs"]["model"]
    assert (model.src, model.slot) == ("modelsampling", 0)
