"""C2 -- second image engine (Z-Image-Turbo), build-or-NO-GO, default-OFF. CPU tests.

Proves the image registry holds >=2 engines + the model-agnostic layer (Flux stays
gen 1; Z-Image is an opt-in peer selectable with NO flag gate -- the registry is
the menu, and availability is the on-disk model check). The live render is the
operator GPU smoke -- NOT covered here.
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
    # needs only text_prompt -> fits every surviving role
    assert "z_image_turbo" in rc.filter_engines_for_role("announcer_visual", descs)
    assert "z_image_turbo" in rc.filter_engines_for_role("character_video", descs)


def test_z_image_adapter_assert_usable_fail_closed(monkeypatch):
    """The adapter's OWN assert_usable fails closed (MISSING_MODEL) until the
    diffusion-model file is configured -- ABSENT/greyed, never a stub (BUG-046).
    2026-06-18: in-process now (the stale cu128 sidecar was dropped), so the gate
    is the WEIGHTS env (MODEL_ENV), mirroring lumina_image."""
    # env UNSET + nothing installed (conftest folder_paths stub -> []) -> greyed.
    monkeypatch.delenv(zit.MODEL_ENV, raising=False)
    eng = ireg.get_engine("z_image_turbo")
    with pytest.raises(ireg.EngineUnusable) as ei:
        eng.assert_usable({}, {"role": "character_video"})
    assert ei.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL


def _set_installed(monkeypatch, names):
    """Point the conftest folder_paths stub's diffusion_models listing at `names`."""
    import sys as _sys
    fp = _sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_filename_list", lambda *a, **k: list(names))
    return fp


def test_z_image_autodiscovers_installed_unet_when_env_unset(monkeypatch):
    """2026-07-05 root fix: env UNSET but a z_image_turbo*.safetensors IS installed
    -> USABLE (no hand-set env required) + the resolver picks the installed file.
    This is the exact bake-off failure (nvfp4 on disk, bf16 default, env unset)."""
    monkeypatch.delenv(zit.MODEL_ENV, raising=False)
    _set_installed(monkeypatch, ["z_image_turbo_nvfp4.safetensors"])
    eng = ireg.get_engine("z_image_turbo")
    assert eng.assert_usable({}, {"role": "character_video"}) == "z_image_turbo"
    assert zit._resolve_unet_name() == ("z_image_turbo_nvfp4.safetensors", True)


def test_z_image_ranks_nvfp4_over_bf16(monkeypatch):
    """When both quants are installed, nvfp4 (Blackwell) wins the default -- the
    bf16 default must NOT beat an installed nvfp4 (codex should-fix)."""
    monkeypatch.delenv(zit.MODEL_ENV, raising=False)
    _set_installed(monkeypatch, ["z_image_turbo_bf16.safetensors",
                                 "z_image_turbo_nvfp4.safetensors"])
    assert zit._resolve_unet_name()[0] == "z_image_turbo_nvfp4.safetensors"


def test_z_image_env_basename_visible_is_usable(monkeypatch):
    """A basename OTR_ZIMAGE_UNET (as every boot script sets it) that folder_paths
    lists is verified usable -- not rejected by a raw os.path.isfile (codex #1)."""
    monkeypatch.setenv(zit.MODEL_ENV, "z_image_turbo_nvfp4.safetensors")
    _set_installed(monkeypatch, ["z_image_turbo_nvfp4.safetensors"])
    eng = ireg.get_engine("z_image_turbo")
    assert eng.assert_usable({}, {"role": "character_video"}) == "z_image_turbo"


def test_z_image_env_pointing_at_absent_file_greys_out(monkeypatch):
    """env set to a file that is NOT installed -> greyed EARLY (MISSING_MODEL),
    not a deep FileNotFoundError at render."""
    monkeypatch.setenv(zit.MODEL_ENV, "z_image_turbo_bf16.safetensors")
    _set_installed(monkeypatch, [])
    eng = ireg.get_engine("z_image_turbo")
    with pytest.raises(ireg.EngineUnusable) as ei:
        eng.assert_usable({}, {"role": "character_video"})
    assert ei.value.reason is ireg.EngineUsabilityReason.MISSING_MODEL


def test_z_image_params_use_the_resolved_unet(monkeypatch):
    """_zimage_params + assert_usable share ONE resolution -> the render graph's
    unet_name is the auto-discovered installed file (contracts can't diverge)."""
    monkeypatch.delenv(zit.MODEL_ENV, raising=False)
    _set_installed(monkeypatch, ["z_image_turbo_nvfp4.safetensors"])
    eng = ireg.get_engine("z_image_turbo")
    params = eng._zimage_params({"prompt": "p", "seed": 1, "width": 512, "height": 512})
    assert params["unet_name"] == "z_image_turbo_nvfp4.safetensors"


def test_z_image_graph_is_well_formed():
    """In-process build (2026-06-18): _zimage_params resolves the low-VRAM
    converged defaults and _build_zimage_graph emits the lumina-style split-file
    AuraFlow graph. CPU-pure (no torch/comfy); the live render is the operator
    GPU smoke."""
    eng = ireg.get_engine("z_image_turbo")
    params = eng._zimage_params({"prompt": "p", "seed": 7, "width": 832, "height": 1216})
    assert params["steps"] == 8 and params["cfg"] == 1.0 and params["shift"] == 3.0
    assert params["scheduler"] == "normal" and params["sampler_name"] == "euler"
    assert params["negative"]                       # still resolved; inert at cfg 1.0, live if OTR_ZIMAGE_CFG is raised
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


# ---------------------------------------------------------------------------
# Portrait reference conditioning (2026-08-05). No engine consumed a reference
# at all, so a character's face was re-invented from text on every beat.
# ---------------------------------------------------------------------------
def _zimage():
    from nodes._otr_image_engines import registry as ireg
    return ireg.get_engine("z_image_turbo")


def test_no_shipping_image_engine_advertises_unproven_reference_conditioning():
    """A generic installed node is not evidence that a checkpoint was trained
    for that semantic path. Z-Image's matched A/B reproduced square-grid
    corruption, so every shipping image adapter is reference-OFF today.
    """
    from nodes._otr_image_engines import registry as ireg

    for name in ireg.all_engine_names():
        eng = ireg.get_engine(name)
        declared = bool(getattr(eng, "accepts_reference_image", False))
        assert declared is False, name

    zimage = ireg.get_engine("z_image_turbo")
    assert zimage.engine_version == "2", "old gridded cache entries must miss"


def test_google_image_rejects_the_singular_reference_key():
    """The guard checked `reference_images` (plural) only, so the singular key
    the dispatcher actually sends sailed past it into the network call.
    """
    import pytest

    from nodes._otr_image_engines import eng_google_image as g

    with pytest.raises(g.GoogleAPIRequestShapeError):
        g._reject_reference_inputs({"reference_image": "/tmp/portrait.png"})
    # and the pre-existing shapes still raise
    with pytest.raises(g.GoogleAPIRequestShapeError):
        g._reject_reference_inputs({"init_image": "x"})
    g._reject_reference_inputs({"prompt": "a radio"})  # clean request: no raise


def test_unreferenced_zimage_graph_is_byte_identical_to_the_shipped_nine():
    from nodes._otr_video_engines import wrapper_bridge as _wb

    eng = _zimage()
    params = eng._zimage_params({"prompt": "a stern man", "seed": 7})
    assert not params["reference_image"]
    assert set(eng._build_zimage_graph(params, _wb.Wire)) == {
        "unet", "clip", "vae", "sampling", "pos", "neg", "latent",
        "ksampler", "decode",
    }


def test_direct_render_request_cannot_reactivate_reference_graph(monkeypatch):
    """The adapter boundary owns the ban too; bypassing the dispatcher with a
    direct reference_image request must still execute the nine-node base graph.
    """
    from nodes._otr_video_engines import wrapper_bridge as _wb

    captured = {}
    monkeypatch.setattr(
        _wb, "resolve_graph_classes",
        lambda candidates: {key: object() for key in candidates},
    )

    def _run(graph, classes, terminal):
        captured["graph"] = graph
        captured["classes"] = classes
        captured["terminal"] = terminal
        return (object(),)

    monkeypatch.setattr(_wb, "run_graph", _run)
    monkeypatch.setattr(_wb, "images_to_uint8", lambda images: ["frame"])
    monkeypatch.setattr(_wb, "reclaim_idle_models", lambda **kwargs: None)
    monkeypatch.setattr(
        _wb, "stage_into_comfy_input",
        lambda *_args, **_kwargs: pytest.fail("production staged a banned reference"),
    )

    eng = zit.ZImageTurboEngine()
    result = eng.render_image({
        "prompt": "a stern man",
        "seed": 7,
        "reference_image": "portrait.png",
    })
    assert result == "frame"
    assert set(captured["graph"]) == {
        "unet", "clip", "vae", "sampling", "pos", "neg", "latent",
        "ksampler", "decode",
    }
    assert "ReferenceLatent" not in {
        node["class"] for node in captured["graph"].values()
    }
    assert set(captured["classes"]) == {
        "unet", "clip", "vae", "sampling", "pos", "neg", "latent",
        "ksampler", "decode",
    }


def test_referenced_zimage_graph_adds_the_chain_AND_rewires_the_sampler():
    """Diagnostic-only graph retained for the permanent rejection A/B. A graph
    that builds the chain and forgets to consume it passes a node-count check
    and renders nothing different -- so assert the rewire.
    """
    from nodes._otr_video_engines import wrapper_bridge as _wb

    eng = _zimage()
    params = eng._diagnostic_zimage_params(
        {"prompt": "a stern man", "seed": 7, "reference_image": "portrait.png"})
    graph = eng._build_zimage_graph(params, _wb.Wire)
    assert set(graph) == {
        "unet", "clip", "vae", "sampling", "pos", "neg", "latent",
        "ksampler", "decode",
        "load_ref", "scale_ref", "encode_ref", "ref_pos", "ref_neg",
    }
    assert graph["ksampler"]["inputs"]["positive"] == _wb.Wire("ref_pos", 0)
    assert graph["ksampler"]["inputs"]["negative"] == _wb.Wire("ref_neg", 0)
    assert graph["ref_pos"]["inputs"]["conditioning"] == _wb.Wire("pos", 0)
    assert graph["ref_neg"]["inputs"]["conditioning"] == _wb.Wire("neg", 0)


def test_the_scaler_is_spelled_with_every_argument_ImageScale_requires():
    """ImageScale.upscale is (image, upscale_method, width, height, crop), all
    required, and run_graph calls fn(**kwargs) with no fallback -- a short-form
    node is a dead episode, not a warning.
    """
    from nodes._otr_video_engines import wrapper_bridge as _wb

    eng = _zimage()
    params = eng._diagnostic_zimage_params(
        {"prompt": "x", "seed": 1, "reference_image": "p.png"})
    scale = eng._build_zimage_graph(params, _wb.Wire)["scale_ref"]
    assert set(scale["inputs"]) == {
        "image", "upscale_method", "width", "height", "crop"}
    # width=0 lets ImageScale derive the side from the real image, so a
    # head-and-shoulders portrait is not centre-cropped into a 16:9 band.
    assert scale["inputs"]["width"] == 0
    assert scale["inputs"]["crop"] == "disabled"


def test_flux_kontext_scaler_must_not_share_the_candidate_tuple():
    """resolve_graph_classes binds the FIRST installed name, and
    FluxKontextImageScale.execute takes `image` only -- it would receive four
    extra kwargs and TypeError inside a GraphExecutionError.
    """
    eng = _zimage()
    assert eng._REF_CANDIDATES["scale_ref"] == ("ImageScale",)
    assert "FluxKontextImageScale" not in eng._REF_CANDIDATES["scale_ref"]


def test_reference_classes_never_pollute_the_cached_singleton_map():
    """The registry stores ONE instance and render_image caches the resolved
    class map on it. The episode's first mint is an unreferenced portrait, so a
    params-gated main map would be cached without the reference keys and every
    later referenced mint would die with 'class unresolved'.
    """
    eng = _zimage()
    assert "load_ref" not in eng._node_candidates()
    assert "load_ref" not in eng._node_candidates(
        {"reference_image": "p.png", "latent_node": "EmptySD3LatentImage"})
    assert set(eng._REF_CANDIDATES) == {
        "load_ref", "scale_ref", "encode_ref", "ref_pos", "ref_neg"}
