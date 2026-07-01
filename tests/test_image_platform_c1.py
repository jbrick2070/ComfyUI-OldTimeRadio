"""CW-IMG / C1 -- image-gen adapter platform (CPU tests).

Mirrors tests/test_video_platform_aseam.py for the image namespace: cold-import
(V-12), protocol parity vs the SHIPPED AudioEngine core (AS-4), the SHARED
role_compat filter (AS-1), the per-role director + 3D granularity LOCK, the
Meta-Brief prompt gen (temp=0 / hash-after / reseed / fallback / consistency
gate), and the dispatcher (cache + cregenerate-invalidates + AS-5 content-address
+ image_done). The live Flux render (passthrough-equality + golden-image) is the
operator GPU smoke -- those are marked ``requires_cuda`` (skipped on CPU), never
faked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import pathlib

import pytest

from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines import schemas as isc
from nodes._otr_shared import role_compat as rc
from nodes.otr_image_director import (
    OTRImageDirector, ADD_CUSTOM, enforce_3d_granularity_lock, three_d_locked_slots,
)
from nodes import otr_meta_brief_image_prompt as mbp
from nodes import otr_image_gen_dispatcher as disp

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_image_registry():
    saved = dict(ireg._IMAGE_REGISTRY._registry)
    try:
        yield ireg._IMAGE_REGISTRY
    finally:
        ireg._IMAGE_REGISTRY._registry.clear()
        ireg._IMAGE_REGISTRY._registry.update(saved)


@pytest.fixture
def clean_video_registry():
    from nodes._otr_video_engines import registry as vreg
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    try:
        yield vreg._VIDEO_REGISTRY
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _img_stub(**kw):
    import types
    base = dict(
        name="img_stub", roles=rc.ROLES, default_roles=rc.ROLES,
        commercial_clean=True, requires_flag=None,
        required_inputs=("text_prompt",), engine_version="1",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# Still-spine ST-2: derive_image_prompts emits + dispatch_images consumes the
# ONE versioned {"version": 1, "objects": [...]} payload (pass-02 item 1; no
# dual-schema shims). These builders shape test portrait objects.
def _pobj(cid, prompt, prompt_hash, source="llm", role="character_video"):
    return {"object_id": cid, "kind": "portrait", "role": role,
            "char_id": cid, "w": 832, "h": 1216,
            "prompt": prompt, "prompt_hash": prompt_hash, "source": source}


def _payload(*objs):
    return {"version": 1, "objects": list(objs)}


def _by_id(payload_and_warns_or_payload):
    p = payload_and_warns_or_payload
    if isinstance(p, tuple):
        p = p[0]
    return mbp.objects_by_id(p)


# --------------------------------------------------------------------------- #
def test_image_cold_import_no_heavy_libs():
    """Importing the image registry + nodes pulls NO heavy lib (V-12)."""
    code = (
        "import sys;"
        "import nodes._otr_image_engines;"
        "import nodes._otr_image_engines.registry;"
        "import nodes._otr_image_engines.schemas;"
        "import nodes._otr_image_engines.flux_gen1;"
        "import nodes.otr_image_director;"
        "import nodes.otr_meta_brief_image_prompt;"
        "import nodes.otr_image_gen_dispatcher;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_image_protocol_parity():
    """ImageEngine is a structural superset of the SHIPPED AudioEngine core, and
    shares the usability taxonomy (AS-4); it has the reduced prompt->image
    lifecycle and deliberately NO canonicalize."""
    from nodes._otr_audio_engines.registry import (
        AudioEngine, EngineUsabilityReason as AReason,
    )
    a_ann = AudioEngine.__annotations__
    i_ann = ireg.ImageEngine.__annotations__
    for name, typ in a_ann.items():
        assert name in i_ann, f"ImageEngine missing audio core attr {name!r}"
        assert i_ann[name] == typ, f"annotation mismatch on {name!r}"
    for meth in ("load", "unload"):
        assert hasattr(ireg.ImageEngine, meth)
    for meth in ("assert_usable", "prepare", "render_image", "teardown"):
        assert hasattr(ireg.ImageEngine, meth), f"ImageEngine missing {meth}"
    assert not hasattr(ireg.ImageEngine, "canonicalize")  # reduced set (AS-4)
    assert {r.value for r in ireg.EngineUsabilityReason} == {r.value for r in AReason}


def test_flux_gen1_registered_as_gen1():
    """C-now: the image registry is non-empty and Flux is registered as gen 1
    (just an adapter, not a hardcoded default)."""
    assert "flux_gen1" in ireg.all_engine_names()
    eng = ireg.get_engine("flux_gen1")
    assert eng.required_inputs == ("text_prompt",)
    # default everywhere -> usable for any role without an opt-in flag
    assert ireg.assert_usable("flux_gen1", "character_video") == "flux_gen1"


def test_image_registry_register_and_assert_usable(clean_image_registry):
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="z_image"))
    assert ireg.assert_usable("z_image", "music_visual") == "z_image"
    with pytest.raises(ireg.EngineUnusable):
        ireg.assert_usable("nope", "music_visual")


def test_image_role_filter_shared(clean_image_registry):
    """C's per-role dropdown uses the SAME role_compat filter (AS-1), not a fork."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="txt", required_inputs=("text_prompt",)))
    ireg.register(_img_stub(name="needs_audio", required_inputs=("audio_ref",)))
    descs = [
        {"engine_id": n, "roles": rc.ROLES,
         "required_inputs": tuple(ireg.get_engine(n).required_inputs)}
        for n in ireg.all_engine_names()
    ]
    # background_abstract supplies only text -> the audio-needing engine is out.
    fit = rc.filter_engines_for_role("background_abstract", descs)
    assert "txt" in fit and "needs_audio" not in fit


#: Minimal VALID video policy for director calls with no 3D engine in play
#: (video_policy_json is REQUIRED + fail-closed per the 3D plan section 3).
_EMPTY_VIDEO_POLICY = json.dumps({"video_models": {}})


def test_image_director_policy_json_and_seed(clean_image_registry):
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    out = OTRImageDirector().direct(**_direct_kwargs(
        other_beats_granularity="per_beat", request_seed=7))
    policy = json.loads(out[0])
    assert policy["image_models"]["music_image_model"]["engine_id"] == "flux_gen1"
    assert policy["seed"]["request_seed"] == 7
    assert policy["fresh_cap"] == 15
    # NO widget named 'seed' on the node (V-7)
    assert "seed" not in OTRImageDirector.INPUT_TYPES()["required"]
    assert "request_seed" in OTRImageDirector.INPUT_TYPES()["required"]


def test_image_director_emits_distinct_per_role_picks_bug405(clean_image_registry):
    # BUG-LOCAL-405: per-role image-model selection must reach image_policy
    # VERBATIM per slot. The live bug was the SAVED workflow carrying flux_gen1
    # in all three slots (so every still minted flux_gen1) -- NOT a director or
    # dispatcher defect. Prove direct() emits the operator's DISTINCT picks per
    # slot; the dispatcher then honors a usable engine and fails LOUD (never a
    # silent flux substitution) on an unusable one.
    clean_image_registry._registry.clear()
    for nm in ("eng_ann", "eng_music", "eng_other"):
        ireg.register(_img_stub(name=nm))
    out = OTRImageDirector().direct(**_direct_kwargs(
        announcer_image_model="eng_ann",
        music_image_model="eng_music",
        other_beats_image_model="eng_other",
    ))
    models = json.loads(out[0])["image_models"]
    assert models["announcer_image_model"]["engine_id"] == "eng_ann"
    assert models["music_image_model"]["engine_id"] == "eng_music"
    assert models["other_beats_image_model"]["engine_id"] == "eng_other"


def test_image_director_fail_closed_incompatible_pick(clean_image_registry):
    clean_image_registry._registry.clear()
    # capability-only (2026-06-22): a genuine incompatibility is a required input
    # NO role supplies (an unknown token) -> fits no role -> fail closed, no swap.
    ireg.register(_img_stub(name="needs_unknown", roles=("background_abstract",),
                            required_inputs=("depth_map",)))
    with pytest.raises(ValueError):
        OTRImageDirector().direct(**_direct_kwargs(
            announcer_image_model="needs_unknown", music_image_model="needs_unknown",
            other_beats_image_model="needs_unknown"))


def _direct_kwargs(**over):
    """Baseline director kwargs (per_object everywhere). Image-model picks now
    live in the wired video_policy_json (OTR_VideoDirector is the single home),
    so this helper FOLDS any image-model overrides into that policy's
    ``image_models`` -- call sites read unchanged. A deliberately-malformed
    video_policy_json (the fail-closed tests) passes through untouched."""
    img = {
        "announcer_image_model": over.pop("announcer_image_model", "flux_gen1"),
        "music_image_model": over.pop("music_image_model", "flux_gen1"),
        "other_beats_image_model": over.pop("other_beats_image_model", "flux_gen1"),
    }
    vp_raw = over.pop("video_policy_json", _EMPTY_VIDEO_POLICY)
    try:
        vp = json.loads(vp_raw)
        if isinstance(vp, dict):
            im = vp.get("image_models")
            vp["image_models"] = {**(im if isinstance(im, dict) else {}), **img}
            vp_raw = json.dumps(vp)
    except (ValueError, TypeError):
        pass  # malformed payload passes through for the fail-closed tests
    kw = dict(
        announcer_granularity="per_object",
        music_granularity="per_object", other_beats_granularity="per_object",
        fresh_cap=15, seed_mode="request_hash", request_seed=0,
        video_policy_json=vp_raw,
    )
    kw.update(over)
    return kw


def _mesh3d_stub(**kw):
    import types
    base = dict(
        name="mesh3d", roles=("character_video",),
        default_roles=("character_video",), family="character_3d",
        required_inputs=("text_prompt",), requires_mesh_portrait=True,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_3d_granularity_lock_per_beat_raises(clean_image_registry,
                                             clean_video_registry):
    """3D plan section 3: a role whose paired VIDEO engine declares
    requires_mesh_portrait REJECTS per_beat with a RAISE (no coercion --
    fresh-per-beat would rebuild the mesh per beat)."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    clean_video_registry._registry.clear()
    clean_video_registry._registry["mesh3d"] = _mesh3d_stub()
    video_policy = json.dumps({"video_models": {
        "other_beats_video_model": {"engine_id": "mesh3d", "custom": False},
    }})
    with pytest.raises(ValueError, match="per_object"):
        OTRImageDirector().direct(**_direct_kwargs(
            other_beats_granularity="per_beat",
            video_policy_json=video_policy))


def test_3d_granularity_lock_per_object_passes(clean_image_registry,
                                               clean_video_registry):
    """With per_object picked, the 3D-locked slot is recorded and the policy
    emits (the lock is CHARACTER-level per_object: one portrait per character
    used globally)."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    clean_video_registry._registry.clear()
    clean_video_registry._registry["mesh3d"] = _mesh3d_stub()
    video_policy = json.dumps({"video_models": {
        "other_beats_video_model": {"engine_id": "mesh3d", "custom": False},
    }})
    out = OTRImageDirector().direct(**_direct_kwargs(
        video_policy_json=video_policy))
    policy = json.loads(out[0])
    assert policy["granularity"]["other_beats_image_model"] == "per_object"
    assert "other_beats_image_model" in policy["locked_3d_slots"]


def test_3d_lock_pure_fn():
    """enforce_3d_granularity_lock RAISES on a locked slot that is not
    per_object (fail-closed; the old coercion is gone) and passes through a
    compliant dict unchanged."""
    warns: list = []
    with pytest.raises(ValueError, match="per_object"):
        enforce_3d_granularity_lock(
            {"a": "per_beat", "b": "per_object"}, {"a"}, warns)
    out = enforce_3d_granularity_lock(
        {"a": "per_object", "b": "per_beat"}, {"a"}, warns)
    assert out == {"a": "per_object", "b": "per_beat"}  # unlocked b untouched


def test_video_policy_json_is_required_and_fail_closed():
    """3D plan section 3: video_policy_json is a REQUIRED forceInput link
    (ComfyUI enforces wired connections for required inputs only) and the
    parse fails closed on empty/malformed/non-policy payloads."""
    it = OTRImageDirector.INPUT_TYPES()
    assert "video_policy_json" in it["required"]
    assert "video_policy_json" not in (it.get("optional") or {})
    spec = it["required"]["video_policy_json"][1]
    assert spec.get("forceInput") is True  # never an auto-generated widget
    for bad in ("", "   ", "not json", "[1,2]", "{}",
                '{"video_models": "nope"}'):
        with pytest.raises(ValueError):
            OTRImageDirector().direct(**_direct_kwargs(video_policy_json=bad))


def test_unregistered_video_engine_fails_closed(clean_image_registry,
                                                clean_video_registry):
    """An UNREGISTERED video engine in the policy raises -- its
    requires_mesh_portrait capability cannot be read (covers custom adapters
    that never registered; never a silent not-3D guess)."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    clean_video_registry._registry.clear()
    video_policy = json.dumps({"video_models": {
        "other_beats_video_model": {"engine_id": "ghost_engine", "custom": True},
    }})
    with pytest.raises(ValueError, match="ghost_engine"):
        OTRImageDirector().direct(**_direct_kwargs(
            video_policy_json=video_policy))


def test_char3d_family_without_capability_fails_closed(clean_image_registry,
                                                       clean_video_registry):
    """A registered character_3d-family engine that does NOT declare
    requires_mesh_portrait raises (the family says 3D but the lock cannot
    prove it -- fail closed, never a hard-coded family check)."""
    import types
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    clean_video_registry._registry.clear()
    clean_video_registry._registry["old3d"] = types.SimpleNamespace(
        name="old3d", roles=("character_video",), default_roles=(),
        family="character_3d", required_inputs=("audio_ref", "init_image"),
    )
    video_policy = json.dumps({"video_models": {
        "other_beats_video_model": {"engine_id": "old3d", "custom": False},
    }})
    with pytest.raises(ValueError, match="requires_mesh_portrait"):
        OTRImageDirector().direct(**_direct_kwargs(
            video_policy_json=video_policy))


def test_char3d_engine_locks_via_capability(clean_image_registry,
                                            clean_video_registry):
    """A character_3d-family engine that declares requires_mesh_portrait=True
    drives the lock (capability-based, never a hard-coded name). Uses a
    test-registered stub: the real 3D scaffolds were UNREGISTERED 2026-06-29
    (C3), so the lock is proven against a registered character_3d stub instead."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    clean_video_registry._registry.clear()
    clean_video_registry._registry["mesh3d"] = _mesh3d_stub()
    locked = three_d_locked_slots({"video_models": {
        "other_beats_video_model": {"engine_id": "mesh3d", "custom": False},
    }})
    assert "other_beats_image_model" in locked


def test_non_3d_engine_does_not_lock():
    """A registered engine with requires_mesh_portrait absent and a non-3D
    family locks nothing (capability default False)."""
    from nodes._otr_video_engines import eng_humo  # noqa: F401
    locked = three_d_locked_slots({"video_models": {
        "other_beats_video_model": {"engine_id": "humo", "custom": False},
    }})
    assert locked == set()


def test_dispatcher_halts_on_3d_per_beat_policy():
    """Defense-in-depth (3D plan section 3): a hand-crafted/stale policy with
    a 3D-locked slot at per_beat HALTS the dispatcher before any object is
    dispatched."""
    with pytest.raises(ValueError, match="HALT"):
        disp.dispatch_images(
            {}, {"locked_3d_slots": ["other_beats_image_model"],
                 "granularity": {"other_beats_image_model": "per_beat"}},
            {"version": 1, "objects": []}, gen_fn=lambda req: None)


def test_no_hardcoded_image_engine_name(clean_image_registry):
    """M2 (behavioral): the per-role image COMBO is sourced from the registry, not
    a hardcoded list -- a brand-new engine name appears in the dropdown with NO
    code edit, and with the registry cleared the 'flux_gen1' default is NOT baked
    into the node. Proof that selection is model-agnostic. The image-model
    dropdowns now live in OTR_VideoDirector (the single home, 2026-06-18)."""
    from nodes.otr_video_director import OTRVideoDirector
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="z_image_custom"))
    combo = OTRVideoDirector.INPUT_TYPES()["required"]["music_image_model"][0]
    assert "z_image_custom" in combo
    assert ADD_CUSTOM in combo
    assert "flux_gen1" not in combo  # registry-sourced, not a baked-in default


# --------------------------------------------------------------------------- #
def test_meta_brief_prompt_temp0_hash_reseed_fallback():
    cast = [{"char_id": "c1", "name": "BABA", "portrait_prompt": "a tall weathered spacer"}]
    meta = {"story_brief_terms": {"setting": ["a derelict orbital station"]}}

    # "lined face" keeps the person guard satisfied (look-QA round 4): a
    # portrait prompt with NO person-evidence now falls back to template.
    good = mbp.derive_image_prompts(cast, meta, llm_fn=lambda _p: "a tall weathered spacer, lined face, station, photographic")
    p = _by_id(good)["c1"]
    assert p["source"] == "llm" and p["prompt_hash"]
    assert p["kind"] == "portrait" and p["w"] == 832 and p["h"] == 1216

    # empty LLM -> reseed -> deterministic template (NEVER empty)
    empties = mbp.derive_image_prompts(cast, meta, llm_fn=lambda _p: "")
    p2 = _by_id(empties)["c1"]
    assert p2["source"].startswith("template") and p2["prompt"]
    assert "spacer" in p2["prompt"] and "station" in p2["prompt"]
    # hash is taken AFTER the call (matches the final prompt text)
    import hashlib
    expect = hashlib.sha256(json.dumps(p2["prompt"], ensure_ascii=True,
                            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert p2["prompt_hash"] == expect


def test_meta_brief_appearance_by_char_id():
    assert mbp._appearance_for_char(
        [{"char_id": "c1", "name": "BABA", "portrait_prompt": "unique-token"}], "c1"
    ) == "unique-token"
    assert mbp._appearance_for_char(
        [{"char_id": "c1", "name": "BABA", "portrait_prompt": "unique-token"}], "BABA"
    ) == ""   # never by display name


def test_meta_brief_consistency_gate_fallback():
    cast = [{"char_id": "c1", "name": "X", "portrait_prompt": "scarred veteran"}]
    meta = {"story_brief_terms": {"setting": ["a neon market"]}}
    # LLM returns a prompt missing both appearance + setting -> gate -> template
    out, warns = mbp.derive_image_prompts(cast, meta, llm_fn=lambda _p: "totally unrelated text")
    assert _by_id(out)["c1"]["source"] == "template_consistency"
    assert any("missing appearance/setting" in w for w in warns)


# --------------------------------------------------------------------------- #
def test_dispatcher_prompt_not_path_guard():
    with pytest.raises(ValueError):
        disp._assert_not_path("output/otr/stills/abc.png")
    with pytest.raises(ValueError):
        disp._assert_not_path(r"C:\stills\x.png")
    disp._assert_not_path("a normal portrait prompt, cinematic")  # ok


def test_apply_fresh_cap():
    assert disp.apply_fresh_cap(40, 15, 9) == 9     # beat budget binds
    assert disp.apply_fresh_cap(40, 15, 99) == 15   # fresh_cap binds
    assert disp.apply_fresh_cap(3, 15, 99) == 3     # request binds


def _np_pixels(val):
    import numpy as np
    return np.full((8, 8, 3), int(val), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Skip unused stills for procedural-floor video engines (operator 2026-06-18):
# an all-viz_green episode must invoke NO image model (accessible: works for
# users with no image/video models).
# --------------------------------------------------------------------------- #
def test_still_needed_for_role_gates_on_video_engine_init_image():
    # viz_green ignores init_image -> still NOT needed; wan_ti2v consumes
    # init_image -> still needed.
    pol_vis = {"video_models": {"other_beats_video_model": {"engine_id": "viz_green"}}}
    pol_wan = {"video_models": {"other_beats_video_model": {"engine_id": "wan_ti2v"}}}
    assert disp._still_needed_for_role(pol_vis, "character_video") is False
    assert disp._still_needed_for_role(pol_wan, "character_video") is True


def test_still_needed_for_role_fails_safe():
    # no video_models / unknown role / unknown engine -> keep the still (legacy)
    assert disp._still_needed_for_role({}, "character_video") is True
    assert disp._still_needed_for_role(
        {"video_models": {"other_beats_video_model": {"engine_id": "viz_green"}}},
        "not_a_role") is True
    assert disp._still_needed_for_role(
        {"video_models": {"other_beats_video_model": {"engine_id": "nope_engine"}}},
        "character_video") is True


def test_still_needed_keys_on_accepts_still_capability():
    # Coverage arch (2026-06-18): every real motion lane declares accepts_still=True
    # (MotionEngineBase default), so a SILENT ltx_video clip now consumes the role's
    # selected image -- this is the flux2/flux-on-LTX fix. The viz_green lane opts
    # OUT (accepts_still=False) and mints no still.
    pol_ltx = {"video_models": {"music_video_model": {"engine_id": "ltx_video"}}}
    pol_avm = {"video_models": {"music_video_model": {"engine_id": "viz_green"}}}
    assert disp._still_needed_for_role(pol_ltx, "music_visual") is True
    assert disp._still_needed_for_role(pol_avm, "music_visual") is False


def test_engine_consumes_still_capability_vs_dual_read():
    import types
    # explicit capability wins both ways
    assert disp.engine_consumes_still(
        types.SimpleNamespace(accepts_still=True, required_inputs=())) is True
    assert disp.engine_consumes_still(
        types.SimpleNamespace(accepts_still=False,
                              required_inputs=("init_image",))) is False
    # dual-read fallback: no flag declared -> legacy required_inputs check
    assert disp.engine_consumes_still(
        types.SimpleNamespace(required_inputs=("init_image",))) is True
    assert disp.engine_consumes_still(
        types.SimpleNamespace(required_inputs=("text_prompt",))) is False


def test_c1_still_pan_and_motion_consume_their_scene_still():
    """C1 (D2 BLACK fix, 2026-06-30): still_pan + still_motion declare
    accepts_still=True so the image dispatcher MINTS the role's selected scene
    still for them (instead of the dark floor). still_flat already did; viz_green
    opts OUT (audio-reactive, ignores stills)."""
    from nodes._otr_video_engines import registry as vreg
    assert disp.engine_consumes_still(vreg.get_engine("still_pan")) is True
    assert disp.engine_consumes_still(vreg.get_engine("still_motion")) is True
    assert disp.engine_consumes_still(vreg.get_engine("still_flat")) is True
    assert disp.engine_consumes_still(vreg.get_engine("viz_green")) is False


def test_dispatch_skips_stills_for_all_visualizer_episode(clean_image_registry, tmp_path):
    # All video roles = viz_green (renamed from visualizer 2026-06-30, item 2;
    # no init_image) -> NO still generated, gen_fn never called -> an
    # all-procedural episode needs no image model at all.
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"episode_id": "ep_vis", "cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {
        "image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
        "video_models": {
            "announcer_video_model": {"engine_id": "viz_green"},
            "music_video_model": {"engine_id": "viz_green"},
            "other_beats_video_model": {"engine_id": "viz_green"}},
        "seed": {"request_seed": 0}, "granularity": {}}
    prompts = _payload(
        _pobj("c1", "a spacer, station", "ph1", role="character_video"),
        {"object_id": "b000_music_open", "kind": "scene_open", "role": "music_visual",
         "char_id": "", "w": 1472, "h": 832, "prompt": "open card", "prompt_hash": "ph2"})
    lockdir = tmp_path / "lease.lockdir"

    calls = {"n": 0}
    def gen_fn(_req):
        calls["n"] += 1
        return _np_pixels(50)

    led, _done, _report, _w = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=gen_fn, output_dir=str(tmp_path), lockdir=lockdir)
    assert calls["n"] == 0                          # NO image model invoked
    assert not (led.get("images", {}).get("images") or [])   # no stills produced


def test_dispatch_still_made_when_video_engine_needs_init_image(clean_image_registry, tmp_path):
    # other_beats video = wan_ti2v (consumes init_image) -> the still IS generated.
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"episode_id": "ep_wan", "cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {
        "image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
        "video_models": {"other_beats_video_model": {"engine_id": "wan_ti2v"}},
        "seed": {"request_seed": 0}, "granularity": {}}
    prompts = _payload(_pobj("c1", "a spacer, station", "ph1"))
    lockdir = tmp_path / "lease.lockdir"
    calls = {"n": 0}
    def gen_fn(_req):
        calls["n"] += 1
        return _np_pixels(60)
    disp.dispatch_images(ledger, policy, prompts, gen_fn=gen_fn,
                         output_dir=str(tmp_path), lockdir=lockdir)
    assert calls["n"] == 1                          # wan_ti2v needs the still


def test_dispatcher_cache_and_cregenerate_invalidates(clean_image_registry, tmp_path):
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"episode_id": "ep_test",
              "cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}, "granularity": {}}
    prompts = _payload(_pobj("c1", "a spacer, station", "ph1"))
    lockdir = tmp_path / "lease.lockdir"

    calls = {"n": 0}
    def gen_fn(_req):
        calls["n"] += 1
        return _np_pixels(100 + calls["n"])

    led, done, _report, _w = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=gen_fn, output_dir=str(tmp_path), lockdir=lockdir,
    )
    assert calls["n"] == 1 and done.startswith("image:done:")
    img = led["images"]["images"][0]
    assert img["path"].endswith(".png")
    # ST-3/W3: the row path is the EPISODE-LOCAL copy; the content-addressed
    # cross-episode cache copy (AS-5; OH-1: episodes/_shared/cache) is
    # pool_path and both exist on disk.
    assert "otr/episodes/ep_test/stills/" in img["path"].replace("\\", "/")
    assert img["pool_path"].replace("\\", "/").endswith(
        f"otr/episodes/_shared/cache/{img['portrait_content_hash']}.png"
    )
    assert os.path.exists(img["path"]) and os.path.exists(img["pool_path"])
    # stills_manifest.json written beside the episode stills
    mpath = pathlib.Path(img["path"]).parent / "stills_manifest.json"
    assert mpath.exists()
    man = json.loads(mpath.read_text(encoding="utf-8"))
    assert man["episode_id"] == "ep_test"
    assert man["stills"][0]["object_id"] == "c1"
    # cast was stamped so the AS-5 resolver round-trips to the POOL copy
    from nodes._otr_shared import portrait_ledger as pl
    assert pl.resolve_portrait_path(led, "c1", output_dir=str(tmp_path)) \
        == pathlib.Path(img["pool_path"])

    # re-run, SAME prompt -> cache HIT (no regen) BUT the hit still
    # materializes into the episode + appends a FRESH row (pass-02 Gem-2)
    led2, _d2, _r2, _w2 = disp.dispatch_images(
        led, policy, prompts, gen_fn=gen_fn, output_dir=str(tmp_path), lockdir=lockdir,
    )
    assert calls["n"] == 1               # gen_fn NOT called again
    rows = led2["images"]["images"]
    assert len(rows) == 2                # fresh ledger row per dispatch
    assert rows[1]["provenance"]["source"] == "cache_hit"
    assert os.path.exists(rows[1]["path"])
    assert rows[0]["portrait_content_hash"] == rows[1]["portrait_content_hash"]

    # change the prompt hash -> new cache key -> REGEN -> new content hash/file
    prompts2 = _payload(_pobj("c1", "a spacer, ruined station", "ph2"))
    led3, _d3, _r3, _w3 = disp.dispatch_images(
        led2, policy, prompts2, gen_fn=gen_fn, output_dir=str(tmp_path), lockdir=lockdir,
    )
    assert calls["n"] == 2               # regenerated
    hashes = {i["portrait_content_hash"] for i in led3["images"]["images"]}
    assert len(hashes) == 2              # B's mesh cache would invalidate


def test_dispatcher_hard_fails_on_unusable_engine(clean_image_registry, tmp_path):
    """NO FALLBACKS (operator 2026-06-18): an unusable/absent REQUESTED engine
    HARD-FAILS the episode (ImageRenderError), never skipped, never a silent
    wrong-engine render."""
    clean_image_registry._registry.clear()  # nothing registered -> assert_usable fails
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "ghost"}},
              "seed": {"request_seed": 0}}
    prompts = _payload(_pobj("c1", "x, y", "ph"))
    with pytest.raises(disp.ImageRenderError):
        disp.dispatch_images(
            ledger, policy, prompts, gen_fn=lambda r: _np_pixels(5),
            output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
        )


# --------------------------------------------------------------------------- #
def test_image_schema_extra_forbid():
    from pydantic import ValidationError
    isc.ImageRequest(request_id="r", role="character_video", object_id="c1",
                     engine_id="flux_gen1")
    with pytest.raises(ValidationError):
        isc.ImageRequest(request_id="r", role="character_video", object_id="c1",
                         engine_id="flux_gen1", bogus_key=1)
    with pytest.raises(ValidationError):
        isc.CanonicalImage(image_id="i", role="r", object_id="o", path="p", nope=1)


# --------------------------------------------------------------------------- #
# disk-path handoff guard (PASS-PM C1): sidecar .png is read only once flushed
def test_wait_for_file_ready_ok_and_timeout(tmp_path):
    p = tmp_path / "ok.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 120)
    assert disp.wait_for_file_ready(str(p), attempts=4, sleep_s=0.0) == str(p)
    # missing file -> fail-closed timeout (never a silent bad image)
    with pytest.raises(disp.ImageHandoffTimeout):
        disp.wait_for_file_ready(str(tmp_path / "missing.png"), attempts=2, sleep_s=0.0)
    # 0-byte / truncated handoff -> fail-closed timeout
    z = tmp_path / "zero.png"
    z.write_bytes(b"")
    with pytest.raises(disp.ImageHandoffTimeout):
        disp.wait_for_file_ready(str(z), attempts=2, sleep_s=0.0)


def test_coerce_pixels_reads_png_path(tmp_path):
    """The sidecar disk-path branch: a .png PATH is decoded to pixels (no IMAGE
    tensor crosses the boundary) only after wait_for_file_ready confirms it."""
    import numpy as np
    from PIL import Image
    px = np.full((6, 6, 3), 80, dtype=np.uint8)
    p = tmp_path / "img.png"
    Image.fromarray(px).save(p)
    out = disp._coerce_pixels(str(p), wait_attempts=4, wait_sleep_s=0.0)
    assert out.shape == (6, 6, 3)
    # a never-ready path fails closed, never a wrong image
    with pytest.raises(disp.ImageHandoffTimeout):
        disp._coerce_pixels(str(tmp_path / "ghost.png"), wait_attempts=2, wait_sleep_s=0.0)


@pytest.mark.requires_cuda
def test_flux_gen1_passthrough_equality_gpu():
    """OPERATOR GPU SMOKE (skipped on CPU): the flux_gen1 adapter render must be
    byte-equal to the legacy direct Flux output for the same request. Run on the
    5080; do NOT fake. Placeholder marker so the contract is visible + collected."""
    pytest.skip("operator GPU smoke -- run on the 5080 (passthrough-equality + golden-image)")


# --------------------------------------------------------------------------- #
# CW-IMG hardening -- cross-process disk-path handoff readiness (PASS-PM C1).
# A cu128 image sidecar writes a .png and returns its PATH; the main venv must
# not decode a 0-byte / still-flushing file. wait_for_file_ready guards that
# race and the dispatcher treats a never-ready handoff as a fail-closed miss.
# --------------------------------------------------------------------------- #
def _write_png(path, val=123, size=8):
    from PIL import Image
    import numpy as np
    Image.fromarray(np.full((size, size, 3), int(val), dtype=np.uint8)).save(path, format="PNG")
    return str(path)


def test_wait_for_file_ready_accepts_complete_png(tmp_path):
    p = _write_png(tmp_path / "ready.png")
    assert disp.wait_for_file_ready(p, attempts=5, sleep_s=0.0) == p


def test_wait_for_file_ready_rejects_zero_byte(tmp_path):
    z = tmp_path / "empty.png"
    z.write_bytes(b"")                       # 0-byte handoff (the race)
    with pytest.raises(disp.ImageHandoffTimeout):
        disp.wait_for_file_ready(str(z), attempts=3, sleep_s=0.0)


def test_wait_for_file_ready_rejects_missing(tmp_path):
    with pytest.raises(disp.ImageHandoffTimeout):
        disp.wait_for_file_ready(str(tmp_path / "nope.png"), attempts=3, sleep_s=0.0)


def test_coerce_pixels_reads_sidecar_png_path(tmp_path):
    """The disk-path handoff: a .png PATH decodes to a uint8 pixel array."""
    p = _write_png(tmp_path / "side.png", val=200, size=8)
    px = disp._coerce_pixels(p, wait_attempts=5, wait_sleep_s=0.0)
    assert px.shape == (8, 8, 3) and int(px[0, 0, 0]) == 200


def test_dispatcher_accepts_sidecar_path_handoff(clean_image_registry, tmp_path):
    """End-to-end disk-path handoff: gen_fn returns a .png PATH (sidecar shape),
    the dispatcher waits for it, content-addresses it + stamps the ledger."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"episode_id": "ep_test",
              "cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}}
    prompts = _payload(_pobj("c1", "a spacer, station", "ph1"))
    side = _write_png(tmp_path / "from_sidecar.png", val=77)

    led, done, _r, warns = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=lambda _req: side,
        output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
        handoff_wait_attempts=5, handoff_wait_sleep_s=0.0,
    )
    assert warns == []
    img = led["images"]["images"][0]
    # OH-1: the AS-5 pool moved to the episodes/_shared/cache tier.
    assert img["pool_path"].replace("\\", "/").endswith(
        f"otr/episodes/_shared/cache/{img['portrait_content_hash']}.png"
    )
    assert "otr/episodes/ep_test/stills/" in img["path"].replace("\\", "/")
    assert done.startswith("image:done:")


def test_dispatcher_hard_fails_on_truncated_handoff(clean_image_registry, tmp_path):
    """NO FALLBACKS: a 0-byte sidecar handoff HARD-FAILS (ImageRenderError),
    never a crash and never a silent skip (PASS-PM C1 fail-closed -> now loud)."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}}
    prompts = _payload(_pobj("c1", "a spacer, station", "ph1"))
    empty = tmp_path / "truncated.png"
    empty.write_bytes(b"")

    with pytest.raises(disp.ImageRenderError):
        disp.dispatch_images(
            ledger, policy, prompts, gen_fn=lambda _req: str(empty),
            output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
            handoff_wait_attempts=2, handoff_wait_sleep_s=0.0,
        )


# --------------------------------------------------------------------------- #
# C1 image GATE -- the in-process Flux gen_fn. The graph SPEC + the dispatcher
# fail-closed degrade are CPU-pure; the live Flux forward is the operator GPU
# smoke (test_flux_gen1_passthrough_equality_gpu, requires_cuda).
# --------------------------------------------------------------------------- #
def test_flux_gen1_graph_spec_is_proven_recipe():
    """flux_gen1 builds the proven render_flux_batch recipe as a pure
    wrapper_bridge graph (CheckpointLoaderSimple -> CLIPTextEncode x2 ->
    EmptyLatentImage -> KSampler(euler/simple) -> VAEDecode), seed + prompt from
    the request. Pure data -- no torch / comfy touched."""
    from nodes._otr_image_engines.flux_gen1 import FluxGen1ImageEngine
    from nodes._otr_video_engines.wrapper_bridge import Wire
    eng = FluxGen1ImageEngine()
    params = eng._flux_params({"prompt": "a weathered spacer, station", "seed": 4242})
    assert params["ckpt_name"] == "flux1-dev-fp8.safetensors"   # the box default
    assert params["seed"] == 4242
    assert params["sampler_name"] == "euler" and params["scheduler"] == "simple"
    # BUG-411 restore: FluxGuidance defaults to 3.5 (the 6/5 look lever).
    assert params["guidance"] == 3.5
    graph = eng._build_flux_graph(params, Wire)
    assert set(graph) == {"ckpt", "pos", "neg", "guidance", "latent",
                          "ksampler", "decode"}
    # CheckpointLoaderSimple out slots: 0=MODEL, 1=CLIP, 2=VAE (proven wiring)
    assert graph["pos"]["inputs"]["clip"] == Wire("ckpt", 1)
    # FluxGuidance bakes guidance into the POSITIVE conditioning; the KSampler
    # reads the GUIDED positive, not the raw CLIP encode (BUG-411).
    assert graph["guidance"]["inputs"]["conditioning"] == Wire("pos", 0)
    assert graph["guidance"]["inputs"]["guidance"] == 3.5
    assert graph["ksampler"]["inputs"]["model"] == Wire("ckpt", 0)
    assert graph["ksampler"]["inputs"]["positive"] == Wire("guidance", 0)
    assert graph["ksampler"]["inputs"]["negative"] == Wire("neg", 0)
    assert graph["ksampler"]["inputs"]["latent_image"] == Wire("latent", 0)
    assert graph["decode"]["inputs"]["samples"] == Wire("ksampler", 0)
    assert graph["decode"]["inputs"]["vae"] == Wire("ckpt", 2)
    cands = eng._node_candidates()
    assert cands["ckpt"] == ("CheckpointLoaderSimple",)
    assert cands["guidance"] == ("FluxGuidance",)
    assert cands["decode"] == ("VAEDecode",)


def test_flux_gen1_params_env_overridable(monkeypatch):
    """Checkpoint + dims + sampler params are env-overridable so the operator
    points at the installed model without a code edit; the request seed binds."""
    monkeypatch.setenv("OTR_FLUX_CKPT", "my-flux.safetensors")
    monkeypatch.setenv("OTR_FLUX_STEPS", "8")
    monkeypatch.setenv("OTR_FLUX_WIDTH", "768")
    monkeypatch.setenv("OTR_FLUX_HEIGHT", "1024")
    monkeypatch.setenv("OTR_FLUX_GUIDANCE", "2.0")
    from nodes._otr_image_engines.flux_gen1 import FluxGen1ImageEngine
    params = FluxGen1ImageEngine()._flux_params({"prompt": "p", "seed": 7})
    assert params["ckpt_name"] == "my-flux.safetensors"
    assert params["steps"] == 8
    assert params["width"] == 768 and params["height"] == 1024
    assert params["seed"] == 7
    assert params["guidance"] == 2.0   # BUG-411: env-overridable guidance


def test_dispatcher_hard_fails_on_render_failure(clean_image_registry, tmp_path):
    """NO FALLBACKS (operator 2026-06-18): a gen_fn that RAISES (no CUDA /
    wrapper node missing / OOM) HARD-FAILS the episode (ImageRenderError) -- no
    skip, no radio-floor degrade, no silent flux substitution."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}}
    prompts = _payload(_pobj("c1", "a spacer, station", "ph1"))

    def boom(_req):
        raise RuntimeError("no CUDA / wrapper node missing on this box")

    with pytest.raises(disp.ImageRenderError):
        disp.dispatch_images(
            ledger, policy, prompts, gen_fn=boom,
            output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
        )


def test_inprocess_gen_fn_resolves_engine_from_registry(clean_image_registry):
    """The in-graph gen_fn resolves the request's engine from the registry and
    returns its render_image pixels (model-agnostic; lazy -- no torch import)."""
    import numpy as np
    clean_image_registry._registry.clear()
    captured = {}

    def _render(request, prepared):
        captured["req"] = request
        captured["prepared"] = prepared
        return np.full((4, 4, 3), 9, dtype=np.uint8)

    stub = _img_stub(name="flux_gen1")
    stub.prepare = lambda *a, **k: {"engine_id": "flux_gen1"}
    stub.render_image = _render
    stub.teardown = lambda *a, **k: None
    ireg.register(stub)
    px = disp._inprocess_gen_fn({"engine_id": "flux_gen1", "prompt": "p", "seed": 1})
    assert px.shape == (4, 4, 3) and int(px[0, 0, 0]) == 9
    assert captured["req"]["engine_id"] == "flux_gen1"
    assert captured["prepared"] == {"engine_id": "flux_gen1"}


# --------------------------------------------------------------------------- #
# ANNOUNCER radio-style portrait (the b001/b005 intro/outro starvation fix).
# Announcer beats are talking beats; without an init_image HuMo fails its
# instant guard and the beat starves to the still floor. The announcer is a
# SYNTHETIC non-cast subject (CastLock owns cast) minted from the lines.
# --------------------------------------------------------------------------- #

def test_announcer_line_char_ids_pure():
    lines = [
        {"line_id": "b000", "speaker_role": "music_open", "char_id": "music_open"},
        {"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"},
        {"line_id": "b002", "speaker_role": "character", "char_id": "c01"},
        {"line_id": "b005", "speaker_role": "announcer", "char_id": "announcer"},
        "garbage-row",
        {"line_id": "b006", "speaker_role": "announcer"},   # no char_id -> default
    ]
    assert mbp.announcer_line_char_ids(lines) == ["announcer"]
    assert mbp.announcer_line_char_ids([]) == []
    assert mbp.announcer_line_char_ids(None) == []


def test_meta_brief_announcer_prompt_added_radio_style():
    """Lines with an announcer role mint a radio-style announcer prompt keyed
    by the LINE's char_id; the prompt is never empty and carries the radio
    styling so the portrait reads as period broadcast."""
    cast = [{"char_id": "c1", "name": "EDNA", "portrait_prompt": "an older lighthouse keeper"}]
    meta = {"story_brief_terms": {"setting": ["a fogbound harbor town"]}}
    lines = [
        {"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"},
        {"line_id": "b002", "speaker_role": "character", "char_id": "c1"},
    ]
    out, _warns = mbp.derive_image_prompts(cast, meta, llm_fn=None, lines=lines)
    objs = _by_id(out)
    portraits = {k for k, v in objs.items() if v["kind"] == "portrait"}
    assert portraits == {"c1", "announcer"}
    ann = objs["announcer"]
    assert ann["role"] == "announcer_visual"
    assert ann["prompt"], "announcer prompt must never be empty"
    assert ann["source"].startswith("announcer_")
    low = ann["prompt"].lower()
    assert "radio" in low and "microphone" in low, \
        "announcer portrait must be radio-styled (operator directive)"
    assert ann["prompt_hash"]


def test_meta_brief_announcer_not_added_without_announcer_lines():
    cast = [{"char_id": "c1", "portrait_prompt": "a spacer"}]
    meta = {}
    lines = [{"line_id": "b002", "speaker_role": "character", "char_id": "c1"}]
    out, _w = mbp.derive_image_prompts(cast, meta, llm_fn=None, lines=lines)
    assert "announcer" not in _by_id(out)
    out2, _w2 = mbp.derive_image_prompts(cast, meta, llm_fn=None)  # lines omitted
    assert "announcer" not in _by_id(out2)


def test_meta_brief_announcer_not_duplicated_when_cast_covers_it():
    """If a (weird) episode carries a real cast row with the announcer id, the
    cast entry wins -- no synthetic duplicate."""
    cast = [{"char_id": "announcer", "portrait_prompt": "the station voice in a booth"}]
    lines = [{"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"}]
    out, _w = mbp.derive_image_prompts(cast, {}, llm_fn=None, lines=lines)
    objs = _by_id(out)
    assert [k for k, v in objs.items() if v["kind"] == "portrait"] == ["announcer"]
    assert not objs["announcer"]["source"].startswith("announcer_"), \
        "a real cast row must not be relabeled as synthetic"


def test_meta_brief_announcer_llm_refined_keeps_grounding():
    """The synthetic announcer entry rides the SAME llm + consistency path."""
    lines = [{"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"}]
    out, _w = mbp.derive_image_prompts(
        [], {}, llm_fn=lambda _p: "a velvet-voiced radio announcer, chrome microphone, art deco studio",
        lines=lines)
    ann = _by_id(out)["announcer"]
    assert ann["source"] == "announcer_llm"
    assert "radio" in ann["prompt"].lower()


def test_meta_brief_announcer_gate_requires_radio_grounding():
    """An LLM line that drops the radio styling for pure story-setting flavor
    FAILS the announcer's gate (appearance-only grounding) and falls back to
    the radio template -- even when the line is grounded in the setting.
    (Live finding 2026-06-09: the Ticking Countdown announcer portrait came
    out as a modern control-room figure because the gate accepted
    setting-grounded prompts. Operator directive: radio style.)"""
    meta = {"story_brief_terms": {"setting": ["a mission control countdown"]}}
    lines = [{"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer"}]
    out, warns = mbp.derive_image_prompts(
        [], meta,
        llm_fn=lambda _p: "a tense mission control countdown operator at a console",
        lines=lines)
    ann = _by_id(out)["announcer"]
    assert ann["source"] == "announcer_template_consistency", \
        "setting-only grounding must fail the announcer gate (got %r)" % ann["source"]
    low = ann["prompt"].lower()
    assert "radio" in low and "microphone" in low
    assert any("missing appearance/setting" in w for w in warns)
    # a CHARACTER with the same setting-grounded line still passes (the
    # relaxed gate is announcer-only)
    cast = [{"char_id": "c1", "portrait_prompt": "a flight controller"}]
    out2, _w2 = mbp.derive_image_prompts(
        cast, meta,
        llm_fn=lambda _p: "a tense mission control countdown operator at a console")
    assert _by_id(out2)["c1"]["source"] == "llm"


def test_stamp_portrait_non_cast_strict_vs_relaxed(tmp_path):
    """Default stays BUG-098 fail-closed; require_cast_entry=False writes the
    content-addressed PNG, skips the cast stamp, and NEVER adds a cast row."""
    from nodes._otr_shared import portrait_ledger as pl
    led = {"cast": [{"char_id": "c1", "name": "X"}]}
    px = _np_pixels(42)
    with pytest.raises(pl.PortraitUnresolved):
        pl.stamp_portrait(led, "announcer", px, output_dir=str(tmp_path))
    path = pl.stamp_portrait(led, "announcer", px, output_dir=str(tmp_path),
                             require_cast_entry=False)
    assert path.exists() and path.suffix == ".png"
    assert [c["char_id"] for c in led["cast"]] == ["c1"], \
        "cast is CastLock's frozen authority -- never grown by a portrait stamp"
    assert "portrait_content_hash" not in led["cast"][0]


def test_dispatcher_mints_non_cast_announcer_portrait(clean_image_registry, tmp_path):
    """dispatch_images mints the announcer portrait, records it in
    ledger['images'] keyed object_id='announcer' (the index the render path
    resolves init_image from), and leaves cast untouched."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"episode_id": "ep_test",
              "cast": [{"char_id": "c1", "name": "EDNA"}]}
    policy = {"image_models": {
                  "other_beats_image_model": {"engine_id": "flux_gen1"},
                  "announcer_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}, "granularity": {}}
    prompts = _payload(
        _pobj("c1", "a keeper, harbor", "ph1"),
        _pobj("announcer", mbp.ANNOUNCER_PORTRAIT_ANCHOR, "ph-ann",
              source="announcer_template", role="announcer_visual"),
    )
    led, done, _r, warns = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=lambda _req: _np_pixels(7),
        output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
    )
    assert done.startswith("image:done:")
    by_id = {i["object_id"]: i for i in led["images"]["images"]}
    assert set(by_id) == {"c1", "announcer"}
    assert os.path.exists(by_id["announcer"]["path"])
    assert [c["char_id"] for c in led["cast"]] == ["c1"]
    assert not any("announcer" in w for w in warns), \
        "announcer mint must not warn: %r" % warns
    # the video render path resolves init_image through _portrait_index
    from nodes._otr_video_engines import render_driver as rd
    idx = rd._portrait_index(led)
    assert idx.get("announcer") == by_id["announcer"]["path"]


# --------------------------------------------------------------------------- #
# Operator ticket 2026-06-11: stale pending_* stills dir -- mux-style re-resolve
# --------------------------------------------------------------------------- #
def _episodes_fixture(tmp_path, final_slug="signal_lost_rapid_roots_x"):
    """A renamed-episode disk layout: the final dir holds the newest ledger;
    the pending dir is GONE (the rename already happened)."""
    root = tmp_path / "otr" / "episodes"
    audio = root / final_slug / "audio"
    audio.mkdir(parents=True)
    (audio / f"{final_slug}_ledger.json").write_text("{}", encoding="utf-8")
    return root


def test_reresolve_stale_pending_rekeys_to_renamed_episode(tmp_path, monkeypatch):
    from nodes.otr_image_gen_dispatcher import _reresolve_episode_stills_dir
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = _episodes_fixture(tmp_path)
    warns = []
    stale = str(root / "pending_20260611_010101" / "stills")
    new_dir, new_ep = _reresolve_episode_stills_dir("pending_20260611_010101", stale, warns)
    assert new_ep == "signal_lost_rapid_roots_x"
    assert new_dir == str(root / "signal_lost_rapid_roots_x" / "stills")
    assert warns and "re-resolved" in warns[0]


def test_reresolve_pending_dir_still_live_is_untouched(tmp_path, monkeypatch):
    from nodes.otr_image_gen_dispatcher import _reresolve_episode_stills_dir
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = _episodes_fixture(tmp_path)
    live = root / "pending_20260611_020202"
    (live / "stills").mkdir(parents=True)   # rename has NOT happened yet
    warns = []
    d, e = _reresolve_episode_stills_dir(
        "pending_20260611_020202", str(live / "stills"), warns)
    assert (d, e) == (str(live / "stills"), "pending_20260611_020202")
    assert not warns


def test_reresolve_non_pending_id_untouched(tmp_path, monkeypatch):
    from nodes.otr_image_gen_dispatcher import _reresolve_episode_stills_dir
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = _episodes_fixture(tmp_path)
    target = str(root / "my_final_episode" / "stills")
    warns = []
    assert _reresolve_episode_stills_dir("my_final_episode", target, warns) == \
        (target, "my_final_episode")
    assert not warns


def test_reresolve_skipped_in_test_mode(tmp_path, monkeypatch):
    from nodes.otr_image_gen_dispatcher import _reresolve_episode_stills_dir
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    root = _episodes_fixture(tmp_path)
    stale = str(root / "pending_20260611_030303" / "stills")
    warns = []
    assert _reresolve_episode_stills_dir("pending_20260611_030303", stale, warns) == \
        (stale, "pending_20260611_030303")
    assert not warns
