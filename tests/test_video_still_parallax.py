"""still_parallax (2.5D depth parallax over stills) -- UNREGISTERED dark
scaffold (item 2 rip-out, 2026-06-30, "registry IS the menu").

still_parallax is no longer imported/selectable (the mesh_stage / triposr
fallback chains now degrade directly to still_motion, not via this engine).
The SOURCE class stays on disk (it returns when re-enabled) and keeps its
identity: the fail-closed usability ladder (local model snapshot), the
role_compat fit (all three OTR_VideoDirector slots; background_abstract
excluded -- no still to warp), and the PURE parallax math (deterministic,
frame-0 identity, depth-weighted shift, edge clamp, no-stretch cover box) all
still apply via DIRECT instantiation. The live depth inference + look-QA is
the GPU/operator step, NOT covered here.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import schemas as sc
from nodes._otr_video_engines.eng_still_parallax import (
    StillParallaxEngine,
    fit_cover_box,
    normalize_depth,
    parallax_offsets,
    synth_parallax_frames,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Registration: selectable-not-default until operator look-QA (0-E gate)
# --------------------------------------------------------------------------- #
def test_still_parallax_unregistered_not_selectable():
    # item 2 rip-out (2026-06-30, "registry IS the menu"): dropped from the
    # registry + the full dropdown; no CAPABILITIES row.
    assert not vreg.is_registered("still_parallax")
    assert "still_parallax" not in vreg.all_engine_names()
    assert "still_parallax" not in vreg.CAPABILITIES


def test_still_parallax_source_identity_intact():
    # The source class is preserved (returns when re-enabled): DA-V2-SMALL
    # depth parallax over an existing still, Apache-2.0, degrades to
    # still_motion (matches mesh_stage/triposr's updated chain head).
    eng = StillParallaxEngine()
    assert eng.family == "static_motion" and eng.family in sc.FAMILIES
    assert eng.required_inputs == ("init_image",)
    assert eng.commercial_clean is True        # Apache-2.0 SMALL ckpt pinned
    assert eng.fallback_engine == "still_motion"
    assert "not real 3D" in eng.honest_label   # the honest 0-E label


def test_still_parallax_fits_every_surviving_slot():
    from nodes.otr_video_director import VIDEO_SLOT_ROLES
    eng = StillParallaxEngine()
    desc = {"engine_id": "still_parallax", "roles": eng.roles,
            "required_inputs": eng.required_inputs}
    # rip-sfx-broll (2026-07-01): every surviving slot's role supplies
    # init_image, so the parallax engine fits them all; dead roles raise.
    for slot, slot_roles in VIDEO_SLOT_ROLES.items():
        fits = any(rc.engine_fits_role(desc, role) for role in slot_roles)
        assert fits, slot
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert rc.engine_fits_role(desc, role) is True
    import pytest as _pytest
    for dead in ("scene_broll", "background_abstract"):
        with _pytest.raises(rc.RoleCompatError):
            rc.engine_fits_role(desc, dead)


def test_still_parallax_assert_usable_model(monkeypatch, tmp_path):
    eng = StillParallaxEngine()
    # No flag gate (registry IS the menu). The pinned local snapshot missing ->
    # MISSING_MODEL naming the env knob + the HF repo id (offline-first: never a
    # runtime fetch).
    monkeypatch.setenv("OTR_DEPTH_ANYTHING_V2_DIR", str(tmp_path / "absent"))
    with pytest.raises(vreg.EngineUnusable) as e2:
        eng.assert_usable(host_caps={}, profile={})
    assert e2.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL
    msg = str(e2.value)
    assert "OTR_DEPTH_ANYTHING_V2_DIR" in msg
    assert "Depth-Anything-V2-Small-hf" in msg
    # A bare dir is NOT enough (A-lane fix 2026-06-11): the resolved source
    # must carry config.json or from_pretrained would fail at load time.
    bare = tmp_path / "da2s"
    bare.mkdir()
    monkeypatch.setenv("OTR_DEPTH_ANYTHING_V2_DIR", str(bare))
    with pytest.raises(vreg.EngineUnusable) as e3:
        eng.assert_usable(host_caps={}, profile={})
    assert e3.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL
    # LOADABLE snapshot present -> usable (the cheap CPU gate).
    (bare / "config.json").write_text("{}", encoding="utf-8")
    assert eng.assert_usable(host_caps={}, profile={}) == "still_parallax"


# --------------------------------------------------------------------------- #
# Pure parallax math (deterministic; no RNG, no model, no ffmpeg)
# --------------------------------------------------------------------------- #
def test_normalize_depth_minmax_and_flat():
    import numpy as np
    d = normalize_depth([[0.0, 5.0], [10.0, 5.0]])
    assert d.dtype.name == "float32"
    assert d[0, 0] == 0.0 and d[1, 0] == 1.0 and d[0, 1] == 0.5
    flat = normalize_depth(np.full((4, 4), 7.25))
    assert flat.min() == flat.max() == 0.0     # flat scene -> no parallax


def test_parallax_offsets_loop_and_identity():
    offs = parallax_offsets(5, 3.0)
    assert len(offs) == 5
    assert offs[0] == (0.0, 0.0)               # frame 0 IS the still
    assert abs(offs[-1][0]) < 1e-9 and abs(offs[-1][1]) < 1e-9   # loops
    assert abs(offs[1][0] - 3.0) < 1e-9        # quarter phase = full amp
    assert parallax_offsets(1, 3.0) == [(0.0, 0.0)]   # n=1 degenerate
    assert parallax_offsets(5, 3.0) == offs    # deterministic, no RNG


def test_synth_parallax_frames_depth_weighted_shift_and_clamp():
    import numpy as np
    h, w, amp, n = 8, 8, 2.0, 5
    img = np.zeros((h, w, 3), dtype="uint8")
    img[:, :, 0] = np.arange(w, dtype="uint8")[None, :] * 10   # x-gradient
    depth = np.zeros((h, w), dtype="float32")
    depth[h // 2:, :] = 1.0                    # top = far, bottom = near
    frames = synth_parallax_frames(img, depth, n, amp)
    assert frames.shape == (n, h, w, 3) and frames.dtype.name == "uint8"
    assert np.array_equal(frames[0], img)      # frame-0 identity
    assert np.array_equal(frames[-1], img)     # seamless loop
    # Quarter phase (frame 1): dx=amp, dy=0. Far rows (off=-1) sample x+amp
    # (content slides LEFT); near rows (off=+1) sample x-amp (slides RIGHT).
    assert frames[1][0, 3, 0] == 50            # far: img[., 3+2] = 50
    assert frames[1][7, 3, 0] == 10            # near: img[., 3-2] = 10
    # Edge clamp: the gather never reads outside the still.
    assert frames[1][0, 7, 0] == 70            # far at right edge clamps to 7
    assert np.array_equal(frames, synth_parallax_frames(img, depth, n, amp))
    with pytest.raises(ValueError):
        synth_parallax_frames(img[:, :, 0], depth, n, amp)     # bad img shape
    with pytest.raises(ValueError):
        synth_parallax_frames(img, depth[:4, :], n, amp)       # shape mismatch


def test_fit_cover_box_uniform_no_stretch():
    sw, sh, left, top = fit_cover_box(480, 832, 1280, 720)
    assert sw >= 1280 and sh >= 720            # covers the canvas
    assert abs(sw / 480 - sh / 832) < 0.01     # ONE uniform scale (no stretch)
    assert left == (sw - 1280) // 2 and top == (sh - 720) // 2   # centered
    sw2, sh2, left2, top2 = fit_cover_box(1472, 832, 1472, 832)
    assert (sw2, sh2, left2, top2) == (1472, 832, 0, 0)          # exact fit


# --------------------------------------------------------------------------- #
# LOUD missing-still behavior (honest engine: never a slate under this stamp)
# --------------------------------------------------------------------------- #
def test_render_clip_missing_still_raises_dependency_missing():
    eng = StillParallaxEngine()
    req = {"shot_id": "s1", "canvas": {"w": 64, "h": 48, "fps": 25},
           "timing": {"target_frame_count": 3}, "asset_refs": {}}
    with pytest.raises(FileNotFoundError):
        eng.render_clip(req)
    # The chain classifies it DEPENDENCY_MISSING -> LOUD skip to the floor.
    assert rd.classify_failure(FileNotFoundError("x")).name == "DEPENDENCY_MISSING"


# --------------------------------------------------------------------------- #
# Driver maps + capability row (BOTH copies; the 0-E wiring contract)
# --------------------------------------------------------------------------- #
def test_engine_family_removed_from_driver_kept_for_soak_audit():
    # item 2 rip-out (2026-06-30): dropped from render_driver's live
    # ENGINE_FAMILY (no longer a real dispatch target) + FLOOR_NAMES was never
    # its home (model-gated, not a floor). The historical-audit copy in
    # otr_video_soak.ENGINE_FAMILY keeps its entry so old soak fixtures still
    # classify correctly.
    assert "still_parallax" not in rd.ENGINE_FAMILY
    assert "still_parallax" not in rd.FLOOR_NAMES
    # A dangling fallback_engine self-heals to the radio floor (make_fallback_of
    # contract: an unregistered name that is not a FLOOR_NAMES member falls
    # through to UNIVERSAL_FLOOR).
    assert rd.make_fallback_of()("still_parallax") == "still_motion"
    soak_src = REPO_ROOT / "scripts" / "otr_video_soak.py"
    spec = importlib.util.spec_from_file_location("otr_video_soak", soak_src)
    soak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soak)
    assert soak.ENGINE_FAMILY["still_parallax"] == "static_motion"   # audit-only


def test_canonical_clip_stamps_parallax():
    eng = StillParallaxEngine()
    clip = eng._floor_clip({"shot_id": "s9"}, "C:/p.mp4", 25, 40)
    assert clip["engine_id"] == "still_parallax"
    assert clip["family"] == "static_motion"
    assert clip["has_audio"] is False          # V-1: mux owns audio
    assert clip["pixel_format"] == "yuv420p" and clip["matrix"] == "bt709"
    assert eng.canonicalize(clip, {}, {}) is clip   # identity canonicalize


# --------------------------------------------------------------------------- #
# Cold-import + ASCII source (V-12 / CLAUDE.md)
# --------------------------------------------------------------------------- #
def test_cold_import_still_parallax_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_still_parallax;"
        "heavy=[m for m in ('torch','transformers','diffusers') "
        "if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


def test_still_parallax_source_is_ascii_no_em_dash():
    src_path = REPO_ROOT / "nodes" / "_otr_video_engines" / "eng_still_parallax.py"
    src = src_path.read_text(encoding="utf-8")
    assert "—" not in src                 # em-dash forbidden (CLAUDE.md)
    src.encode("ascii")                        # ASCII-only source


# --------------------------------------------------------------------------- #
# A-lane fix (2026-06-11): HF-cache snapshot resolution for from_pretrained
# --------------------------------------------------------------------------- #
def test_resolve_model_src_hf_cache_layout(tmp_path):
    """from_pretrained CANNOT load an HF cache REPO dir (models--*) -- only
    its snapshots/<rev> (proven live 2026-06-11: repo dir raises OSError,
    snapshot loads). resolve_model_src maps repo-cache dirs to the newest
    config.json-bearing snapshot and passes plain dirs through."""
    from nodes._otr_video_engines.eng_still_parallax import resolve_model_src

    # Plain local model dir: passes through unchanged.
    plain = tmp_path / "plain_model"
    plain.mkdir()
    (plain / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_src(str(plain)) == str(plain)

    # HF cache repo dir: resolves into snapshots/<rev> with config.json.
    repo = tmp_path / "models--depth-anything--Depth-Anything-V2-Small-hf"
    incomplete = repo / "snapshots" / "aaaa"      # no config.json -> skipped
    incomplete.mkdir(parents=True)
    good = repo / "snapshots" / "bbbb"
    good.mkdir()
    (good / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model_src(str(repo)) == str(good)

    # No usable snapshot: falls back to the input (load then fails LOUD).
    empty = tmp_path / "models--empty"
    (empty / "snapshots").mkdir(parents=True)
    assert resolve_model_src(str(empty)) == str(empty)


def test_installed_requires_loadable_snapshot(tmp_path, monkeypatch):
    """_installed() is True only when the RESOLVED source carries a
    config.json -- a bare cache dir without a snapshot is NOT installed
    (fail-closed at assert_usable, not at load time)."""
    eng = StillParallaxEngine()
    repo = tmp_path / "models--depth-anything--Depth-Anything-V2-Small-hf"
    (repo / "snapshots").mkdir(parents=True)
    monkeypatch.setenv("OTR_DEPTH_ANYTHING_V2_DIR", str(repo))
    assert eng._installed() is False
    snap = repo / "snapshots" / "rev0"
    snap.mkdir()
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert eng._installed() is True
