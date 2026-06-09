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


def test_image_director_policy_json_and_seed(clean_image_registry):
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    out = OTRImageDirector().direct(
        announcer_image_model="flux_gen1", music_image_model="flux_gen1",
        other_beats_image_model="flux_gen1", announcer_granularity="per_object",
        music_granularity="per_object", other_beats_granularity="per_beat",
        fresh_cap=15, seed_mode="request_hash", request_seed=7,
    )
    policy = json.loads(out[0])
    assert policy["image_models"]["music_image_model"]["engine_id"] == "flux_gen1"
    assert policy["seed"]["request_seed"] == 7
    assert policy["fresh_cap"] == 15
    # NO widget named 'seed' on the node (V-7)
    assert "seed" not in OTRImageDirector.INPUT_TYPES()["required"]
    assert "request_seed" in OTRImageDirector.INPUT_TYPES()["required"]


def test_image_director_fail_closed_incompatible_pick(clean_image_registry):
    clean_image_registry._registry.clear()
    # an engine that needs audio_ref cannot serve background (other_beats supplies
    # only text for that role) -> fail closed, no silent swap.
    ireg.register(_img_stub(name="needs_audio", required_inputs=("audio_ref",)))
    with pytest.raises(ValueError):
        OTRImageDirector().direct(
            announcer_image_model="needs_audio", music_image_model="needs_audio",
            other_beats_image_model="needs_audio", announcer_granularity="per_object",
            music_granularity="per_object", other_beats_granularity="per_object",
            fresh_cap=15, seed_mode="request_hash", request_seed=0,
        )


def test_3d_granularity_lock(clean_image_registry, clean_video_registry):
    """A role whose paired VIDEO engine is character_3d is hard-locked to
    per_object; per_beat is rejected with a warning (PASS-IMG MUST-FIX #2)."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    import types
    clean_video_registry._registry.clear()
    clean_video_registry._registry["mesh3d"] = types.SimpleNamespace(
        name="mesh3d", roles=("character_video",), default_roles=("character_video",),
        family="character_3d", required_inputs=("text_prompt",),
    )
    video_policy = json.dumps({"video_models": {
        "other_beats_video_model": {"engine_id": "mesh3d", "custom": False},
    }})
    out = OTRImageDirector().direct(
        announcer_image_model="flux_gen1", music_image_model="flux_gen1",
        other_beats_image_model="flux_gen1", announcer_granularity="per_object",
        music_granularity="per_object", other_beats_granularity="per_beat",
        fresh_cap=15, seed_mode="request_hash", request_seed=0,
        video_policy_json=video_policy,
    )
    policy = json.loads(out[0])
    assert policy["granularity"]["other_beats_image_model"] == "per_object"
    assert "other_beats_image_model" in policy["locked_3d_slots"]
    assert any("locked to per_object" in w for w in policy["warnings"])


def test_3d_lock_pure_fn():
    warns: list = []
    out = enforce_3d_granularity_lock(
        {"a": "per_beat", "b": "per_object"}, {"a"}, warns,
    )
    assert out["a"] == "per_object" and out["b"] == "per_object"
    assert warns


def test_no_hardcoded_image_engine_name(clean_image_registry):
    """M2 (behavioral): the per-role COMBO is sourced from the registry, not a
    hardcoded list -- a brand-new engine name appears in the dropdown with NO
    code edit, and with the registry cleared the 'flux_gen1' default is NOT baked
    into the node. Proof that selection is model-agnostic."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="z_image_custom"))
    combo = OTRImageDirector.INPUT_TYPES()["required"]["music_image_model"][0]
    assert "z_image_custom" in combo
    assert ADD_CUSTOM in combo
    assert "flux_gen1" not in combo  # registry-sourced, not a baked-in default


# --------------------------------------------------------------------------- #
def test_meta_brief_prompt_temp0_hash_reseed_fallback():
    cast = [{"char_id": "c1", "name": "BABA", "portrait_prompt": "a tall weathered spacer"}]
    meta = {"story_brief_terms": {"setting": ["a derelict orbital station"]}}

    good = mbp.derive_image_prompts(cast, meta, llm_fn=lambda _p: "a tall weathered spacer, station, photographic")
    p = good[0]["c1"]
    assert p["source"] == "llm" and p["prompt_hash"]

    # empty LLM -> reseed -> deterministic template (NEVER empty)
    empties = mbp.derive_image_prompts(cast, meta, llm_fn=lambda _p: "")
    p2 = empties[0]["c1"]
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
    assert out["c1"]["source"] == "template_consistency"
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


def test_dispatcher_cache_and_cregenerate_invalidates(clean_image_registry, tmp_path):
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}, "granularity": {}}
    prompts = {"c1": {"prompt": "a spacer, station", "prompt_hash": "ph1", "source": "llm"}}
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
    # content-addressed at output/otr/stills/{portrait_content_hash}.png (AS-5)
    assert img["path"].replace("\\", "/").endswith(
        f"otr/stills/{img['portrait_content_hash']}.png"
    )
    # cast was stamped so the AS-5 resolver round-trips
    from nodes._otr_shared import portrait_ledger as pl
    assert pl.resolve_portrait_path(led, "c1", output_dir=str(tmp_path)) == pathlib.Path(img["path"])

    # re-run, SAME prompt -> cache HIT (no regen)
    led2, _d2, _r2, _w2 = disp.dispatch_images(
        led, policy, prompts, gen_fn=gen_fn, output_dir=str(tmp_path), lockdir=lockdir,
    )
    assert calls["n"] == 1               # gen_fn NOT called again
    assert len(led2["images"]["images"]) == 1

    # change the prompt hash -> new cache key -> REGEN -> new content hash/file
    prompts2 = {"c1": {"prompt": "a spacer, ruined station", "prompt_hash": "ph2", "source": "llm"}}
    led3, _d3, _r3, _w3 = disp.dispatch_images(
        led2, policy, prompts2, gen_fn=gen_fn, output_dir=str(tmp_path), lockdir=lockdir,
    )
    assert calls["n"] == 2               # regenerated
    hashes = {i["portrait_content_hash"] for i in led3["images"]["images"]}
    assert len(hashes) == 2              # B's mesh cache would invalidate


def test_dispatcher_fail_closed_unusable_engine(clean_image_registry, tmp_path):
    """An unusable/absent engine -> that object simply has no image (warned),
    never a silent wrong-engine render."""
    clean_image_registry._registry.clear()  # nothing registered -> assert_usable fails
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "ghost"}},
              "seed": {"request_seed": 0}}
    prompts = {"c1": {"prompt": "x, y", "prompt_hash": "ph", "source": "llm"}}
    led, done, _r, warns = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=lambda r: _np_pixels(5),
        output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
    )
    assert led["images"]["images"] == []
    assert any("not usable" in w for w in warns)


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
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}}
    prompts = {"c1": {"prompt": "a spacer, station", "prompt_hash": "ph1", "source": "llm"}}
    side = _write_png(tmp_path / "from_sidecar.png", val=77)

    led, done, _r, warns = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=lambda _req: side,
        output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
        handoff_wait_attempts=5, handoff_wait_sleep_s=0.0,
    )
    assert warns == []
    img = led["images"]["images"][0]
    assert img["path"].replace("\\", "/").endswith(
        f"otr/stills/{img['portrait_content_hash']}.png"
    )
    assert done.startswith("image:done:")


def test_dispatcher_skips_truncated_handoff_fail_closed(clean_image_registry, tmp_path):
    """A 0-byte sidecar handoff -> that object simply has no image (warned),
    never a crash and never a silent bad render (PASS-PM C1, fail-closed)."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}}
    prompts = {"c1": {"prompt": "a spacer, station", "prompt_hash": "ph1", "source": "llm"}}
    empty = tmp_path / "truncated.png"
    empty.write_bytes(b"")

    led, _done, _r, warns = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=lambda _req: str(empty),
        output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
        handoff_wait_attempts=2, handoff_wait_sleep_s=0.0,
    )
    assert led["images"]["images"] == []
    assert any("handoff not ready" in w for w in warns)


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
    graph = eng._build_flux_graph(params, Wire)
    assert set(graph) == {"ckpt", "pos", "neg", "latent", "ksampler", "decode"}
    # CheckpointLoaderSimple out slots: 0=MODEL, 1=CLIP, 2=VAE (proven wiring)
    assert graph["pos"]["inputs"]["clip"] == Wire("ckpt", 1)
    assert graph["ksampler"]["inputs"]["model"] == Wire("ckpt", 0)
    assert graph["ksampler"]["inputs"]["positive"] == Wire("pos", 0)
    assert graph["ksampler"]["inputs"]["negative"] == Wire("neg", 0)
    assert graph["ksampler"]["inputs"]["latent_image"] == Wire("latent", 0)
    assert graph["decode"]["inputs"]["samples"] == Wire("ksampler", 0)
    assert graph["decode"]["inputs"]["vae"] == Wire("ckpt", 2)
    cands = eng._node_candidates()
    assert cands["ckpt"] == ("CheckpointLoaderSimple",)
    assert cands["decode"] == ("VAEDecode",)


def test_flux_gen1_params_env_overridable(monkeypatch):
    """Checkpoint + dims + sampler params are env-overridable so the operator
    points at the installed model without a code edit; the request seed binds."""
    monkeypatch.setenv("OTR_FLUX_CKPT", "my-flux.safetensors")
    monkeypatch.setenv("OTR_FLUX_STEPS", "8")
    monkeypatch.setenv("OTR_FLUX_WIDTH", "768")
    monkeypatch.setenv("OTR_FLUX_HEIGHT", "1024")
    from nodes._otr_image_engines.flux_gen1 import FluxGen1ImageEngine
    params = FluxGen1ImageEngine()._flux_params({"prompt": "p", "seed": 7})
    assert params["ckpt_name"] == "my-flux.safetensors"
    assert params["steps"] == 8
    assert params["width"] == 768 and params["height"] == 1024
    assert params["seed"] == 7


def test_dispatcher_render_failure_degrades_to_floor(clean_image_registry, tmp_path):
    """The image GATE never crashes the episode: a gen_fn that RAISES (no CUDA /
    wrapper node missing / OOM) is fail-closed -> the object has no portrait
    (warned LOUD) so HuMo degrades to the radio floor downstream."""
    clean_image_registry._registry.clear()
    ireg.register(_img_stub(name="flux_gen1"))
    ledger = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    policy = {"image_models": {"other_beats_image_model": {"engine_id": "flux_gen1"}},
              "seed": {"request_seed": 0}}
    prompts = {"c1": {"prompt": "a spacer, station", "prompt_hash": "ph1", "source": "llm"}}

    def boom(_req):
        raise RuntimeError("no CUDA / wrapper node missing on this box")

    led, done, _r, warns = disp.dispatch_images(
        ledger, policy, prompts, gen_fn=boom,
        output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
    )
    assert led["images"]["images"] == []
    assert any("render failed" in w for w in warns)
    assert done.startswith("image:done:")


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
