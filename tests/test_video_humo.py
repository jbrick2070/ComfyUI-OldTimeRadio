"""A-S6 / CW-7 CPU tests -- the in-process HuMo audio-driven-face adapter.

HuMo (audio + portrait -> talking-character video) runs IN-PROCESS in the main
cu130 venv (its wrapper is absent in the pytest sandbox), so every test here
exercises the COLD path: the adapter is registered + dark + gated, fails closed
without the flag / install, tolerates SageAttention (unlike ltx_video -- HuMo's
Sage stance is a GPU-smoke verify item, not a hard CPU gate), the portrait
init_image aspect plan never stretches (480x832 native), the AS-3 lease is taken
+ released, the pure request / clip helpers are deterministic, and the declared
fallback chain humo -> humo_1.7B -> still_motion is acyclic and converges on
the radio floor. The live load + the VRAM<=14.5 GB boundary + render-twice pixels
are the A-S6 GPU smoke (operator), NOT covered here.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import types

import pytest

from nodes._otr_shared import fallback as fb
from nodes._otr_shared import gpu_residency as gr
from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as sc
from nodes._otr_video_engines.eng_humo import HuMoEngine

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Registration + default-OFF / dark
# --------------------------------------------------------------------------- #
def test_humo_registered_and_dark():
    assert vreg.is_registered("humo")
    eng = vreg.get_engine("humo")
    assert isinstance(eng, HuMoEngine)
    assert eng.family == "audio_driven_face" and eng.family in sc.FAMILIES
    assert eng.default_roles == ()                 # not a default
    assert eng.requires_flag is None               # registry IS the menu (no flag gate)
    assert eng.required_inputs == ("audio_ref", "init_image")
    # RADIO IS THE HOST (2026-06-30, reversing Route-A 2026-06-28): HuMo serves
    # ONLY character_video now. This tuple is UI-SORT/self-description metadata
    # only -- role_compat.engine_fits_role (below) is capability-only and does
    # not consult it; the actual hard gate against announcer/music is
    # render_driver._enforce_radio_is_host.
    assert eng.roles == ("character_video",)
    assert eng.declared_isolation == mc.ISOLATION_IN_PROCESS
    assert eng.binds_seed is True
    assert eng.fallback_engine == "humo_1.7B"      # 14B degrades to the 1.7B tier first
    assert "humo" in vreg.all_engine_names()       # in the full static dropdown
    # The 1.7B tier degrades straight to the zero-VRAM still floor (latentsync
    # removed 2026-06-17).
    assert vreg.get_engine("humo_1.7B").fallback_engine == "still_motion"


def test_humo_required_inputs_match_family_schema():
    # The adapter's required_inputs are exactly the schema's hard requirements
    # for audio_driven_face (one shared vocabulary -- role_compat + the request
    # validator agree).
    assert set(vreg.get_engine("humo").required_inputs) == set(
        sc.FAMILY_REQUIRED_INPUTS["audio_driven_face"])


# --------------------------------------------------------------------------- #
# Registry gating (dark until the opt-in flag) + role membership
# --------------------------------------------------------------------------- #
def test_registry_humo_selectable_no_flag(monkeypatch):
    # Registry IS the menu: humo is usable for NO flag gate. assert_usable is
    # CAPABILITY-only (role_compat.engine_fits_role) and does NOT consult the
    # `.roles` whitelist, so it still passes announcer_visual/music_visual here
    # -- capability-FIT is necessary but no longer SUFFICIENT for those two
    # roles; render_driver._enforce_radio_is_host is the separate, higher-level
    # policy gate that redirects any such real dispatch to ltx_audio_in
    # (2026-06-30 HuMo-improve plan, "the radio is the host").
    monkeypatch.delenv("OTR_ENABLE_HUMO", raising=False)
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert vreg.assert_usable("humo", role) == "humo"
    for role in ("scene_broll", "background_abstract"):
        with pytest.raises(vreg.EngineUnusable):
            vreg.assert_usable("humo", role)               # role not served


# --------------------------------------------------------------------------- #
# Shared role_compat filter (AS-1): audio_driven_face needs audio_ref+init_image
# --------------------------------------------------------------------------- #
def test_humo_role_fit_audio_driven_face():
    eng = vreg.get_engine("humo")
    desc = {"engine_id": "humo", "roles": eng.roles,
            "required_inputs": eng.required_inputs}
    # announcer_visual, music_visual AND character_video all supply BOTH
    # init_image and audio_ref, so HuMo capability-FITS all three -- this is
    # the low-level input-token check only. Whether HuMo is actually ALLOWED to
    # dispatch for a role is a separate, higher-level policy
    # (render_driver._enforce_radio_is_host, 2026-06-30): announcer_visual /
    # music_visual are policy-forbidden ("radio is the host") even though they
    # fit here; only character_video is both fit AND allowed.
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert rc.engine_fits_role(desc, role) is True
    # rip-sfx-broll (2026-07-01): the dead roles are unknown tokens and raise.
    import pytest as _pytest
    for role in ("scene_broll", "background_abstract"):
        with _pytest.raises(rc.RoleCompatError):
            rc.engine_fits_role(desc, role)
    assert rc.filter_engines_for_role("character_video", [desc]) == ["humo"]
    assert rc.filter_engines_for_role("music_visual", [desc]) == ["humo"]


# --------------------------------------------------------------------------- #
# Fail-closed usability ladder (no heavy import; CPU box)
# --------------------------------------------------------------------------- #
def test_humo_assert_usable_install_tolerates_sage(monkeypatch):
    eng = vreg.get_engine("humo")
    # No flag gate (registry IS the menu). SageAttention resident + ckpt absent
    # -> MISSING_MODEL (NOT INCOMPATIBLE_PROFILE: HuMo loads in-process and does
    # not adopt ltx_video's hard BUG-070 Sage abort; its Sage tolerance is a
    # GPU-smoke verify item).
    monkeypatch.setitem(sys.modules, "sageattention",
                        types.ModuleType("sageattention"))
    monkeypatch.setenv("OTR_HUMO_CKPT", str(REPO_ROOT / "_no_ckpt.safetensors"))
    with pytest.raises(vreg.EngineUnusable) as e2:
        eng.assert_usable(host_caps={}, profile={})
    assert e2.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL


# --------------------------------------------------------------------------- #
# init_image aspect: single uniform scale, never a silent stretch (N9)
# --------------------------------------------------------------------------- #
def test_humo_aspect_plan_portrait_no_stretch():
    eng = vreg.get_engine("humo")
    # HuMo native 480x832 portrait into a 1280x720 landscape canvas.
    req = {"asset_refs": {"init_image": "C:/p.png"}, "init_w": 480, "init_h": 832,
           "canvas": {"w": 1280, "h": 720, "aspect_policy": "pad"}}
    plan = eng._aspect_plan(req)
    assert plan is not None
    sx = plan["scaled_w"] / plan["src_w"]
    sy = plan["scaled_h"] / plan["src_h"]
    assert abs(sx - sy) < 0.01                       # one uniform scale (no stretch)
    assert plan["scaled_w"] % 2 == 0 and plan["scaled_h"] % 2 == 0
    assert eng._aspect_plan({"canvas": {"w": 0, "h": 0}}) is None   # unsized -> None
    with pytest.raises(ValueError):
        eng._aspect_plan({"canvas": {"aspect_policy": "stretch"}})  # bad token


# --------------------------------------------------------------------------- #
# audio_ref + init_image extraction (the audio_driven_face inputs)
# --------------------------------------------------------------------------- #
def test_humo_ref_extraction_handles_str_dict_and_none():
    eng = vreg.get_engine("humo")
    assert eng._ref_path("C:/a.wav") == "C:/a.wav"
    assert eng._ref_path({"path": "C:/b.wav", "content_hash": "x"}) == "C:/b.wav"
    assert eng._ref_path(None) == "" and eng._ref_path("") == ""
    assert eng._init_image_ref({"asset_refs": {"init_image": "C:/p.png"}}) == "C:/p.png"
    assert eng._init_image_ref({"asset_refs": {}}) == ""


# --------------------------------------------------------------------------- #
# AS-3 shared lease: prepare takes it; a load failure must not strand it
# --------------------------------------------------------------------------- #
def test_humo_prepare_releases_lease_on_load_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    monkeypatch.setenv("OTR_HUMO_CKPT", str(tmp_path / "_no_ckpt.safetensors"))
    eng = HuMoEngine()
    assert not gr.is_held()
    with pytest.raises(RuntimeError):
        eng.prepare(host_caps={}, profile={}, session_ctx={})
    assert not gr.is_held()                          # lease released, never stranded


def test_humo_teardown_idempotent_no_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    eng = HuMoEngine()
    eng.unload()                                     # no residency -> no crash
    eng.teardown(None)                               # no lease -> no raise
    eng.teardown({"lease": None})
    eng.teardown({"lease": None, "patchers": []})


# --------------------------------------------------------------------------- #
# Determinism contract (V-7): build the request twice -> identical; seed routes
# --------------------------------------------------------------------------- #
def test_humo_build_render_request_deterministic():
    eng = vreg.get_engine("humo")
    req = {"asset_refs": {"init_image": "C:/p.png"},
           "audio_ref": {"path": "C:/a.wav", "content_hash": "h"},
           "text_prompt": "a calm anchor", "init_w": 480, "init_h": 832,
           "canvas": {"w": 480, "h": 832, "aspect_policy": "pad"},
           "timing": {"target_frame_count": 177}, "seed_bundle": {"request_seed": 5}}
    a, b = eng._build_render_request(req), eng._build_render_request(req)
    assert a == b                                    # render-twice request stability
    assert a["init_image"] == "C:/p.png" and a["audio_path"] == "C:/a.wav"
    assert a["text_prompt"] == "a calm anchor" and a["fps"] == 25
    assert a["target_frame_count"] == 177 and a["seed"] == 5
    assert a["aspect_plan"] is not None
    assert eng._build_render_request(dict(req, seed_bundle={"request_seed": 9})) != a


def test_humo_canonicalize_silent_bt709():
    eng = vreg.get_engine("humo")
    clip = eng.canonicalize({"out_path": "C:/o.mp4", "frame_count": 177},
                            {"shot_id": "s1"}, {})
    assert clip["has_audio"] is False                # V-1: mux owns audio
    assert clip["pixel_format"] == "yuv420p"
    assert clip["matrix"] == clip["transfer"] == clip["color_primaries"] == "bt709"
    assert clip["frame_count"] == 177 and clip["fps"] == 25
    assert clip["engine_id"] == "humo" and clip["family"] == "audio_driven_face"
    assert clip["path"] == "C:/o.mp4" and clip["clip_id"] == "s1"


# --------------------------------------------------------------------------- #
# Fallback chain: humo -> humo_1.7B -> still_motion converges (A-S6 gate)
# --------------------------------------------------------------------------- #
def _registry_fallback_of(name):
    """Single-hop lookup over the live registry (None when terminal/unknown)."""
    if not vreg.is_registered(name):
        return None
    return getattr(vreg.get_engine(name), "fallback_engine", None)


def test_humo_fallback_chain_converges_on_radio_floor():
    chain = fb.resolve_fallback_chain("humo", _registry_fallback_of)
    assert chain == ["humo", "humo_1.7B", "still_motion"]
    # Terminus is the zero-VRAM radio floor: registered, static_motion, no flag.
    floor = vreg.get_engine("still_motion")
    assert floor.family == "static_motion"
    assert getattr(floor, "requires_flag", None) is None
    assert getattr(floor, "fallback_engine", None) is None       # terminal
    assert fb.chain_terminates_at_floor(
        "humo", _registry_fallback_of, {"still_motion"}) is True


def test_fallback_resolver_pure_acyclic_and_bounded():
    table = {"a": "b", "b": "c", "c": None}
    assert fb.resolve_fallback_chain("a", table.get) == ["a", "b", "c"]
    # A cycle is caught fail-closed (never an infinite walk).
    cyc = {"a": "b", "b": "a"}
    with pytest.raises(fb.FallbackCycleError):
        fb.resolve_fallback_chain("a", cyc.get)
    # A runaway (always a fresh name) is bounded by max_hops.
    with pytest.raises(fb.FallbackChainError):
        fb.resolve_fallback_chain("x0", lambda n: "x" + str(int(n[1:]) + 1),
                                  max_hops=8)
    with pytest.raises(ValueError):
        fb.resolve_fallback_chain("", table.get)


# --------------------------------------------------------------------------- #
# Cold-import + ASCII source (V-12 / CLAUDE.md)
# --------------------------------------------------------------------------- #
def test_cold_import_humo_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_humo;"
        "import nodes._otr_shared.fallback;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


def test_humo_source_is_ascii_no_em_dash():
    for rel in ("nodes/_otr_video_engines/eng_humo.py",
                "nodes/_otr_shared/fallback.py"):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "—" not in src, f"em-dash forbidden in {rel} (CLAUDE.md)"
        src.encode("ascii")                          # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
