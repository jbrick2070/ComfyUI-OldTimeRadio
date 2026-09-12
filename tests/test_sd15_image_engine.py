"""``sd15`` -- the only image engine proven on Apple Silicon, and it had no tests.

WHY THIS FILE EXISTS. Every other local image engine in the pack carries one
(`test_image_engine_c2.py` for z_image_turbo, `test_flux2_klein_engine.py`,
`test_lumina_image_engine.py`, `test_ideogram4_local_engine.py`). `sd15` had
none -- grepping tests/ for ``_fit_native``, ``_sd15_params`` or ``SD15Engine``
returned nothing -- while being the engine that supplies the still for every
``still_*`` receipt and every `ltx_8gb` receipt on the Mac. The audit that found
that gap named the assertions; this is them.

TWO OF THESE PIN REAL DEFECTS THAT REACHED CODE AND WERE CAUGHT BY REVIEW, not
hypotheticals. Both are the same shape -- confidently wrong output, no error
anywhere, a valid PNG of the wrong thing:

  * the engine read ``text_prompt`` (the ``required_inputs`` name) where the
    dispatcher supplies ``prompt``, so every still would have been minted from
    an EMPTY string;
  * ``_fit_native`` had a 512 default that was dead on every real request,
    because the composer always stamps w/h -- so the canonical's ~832-wide
    canvas went straight through, and SD 1.5 duplicates subjects past 768.

Neither raises. Neither fails a gate. Only a person looking at the picture, or
a test like this one, notices.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._otr_image_engines import registry as ireg  # noqa: E402
from nodes._otr_image_engines import sd15 as S  # noqa: E402


# ---------------------------------------------------------------------------
# The registry row -- the claim that this engine runs on Metal at all
# ---------------------------------------------------------------------------

def test_the_row_claims_mps_and_that_claim_has_a_receipt():
    """`sd15` is the reason the Mac has local stills. If this row loses "mps",
    the profile wizard stops admitting it and `config/profiles/otr_mac_mps.json`
    -- which names sd15 for all three image roles -- becomes incoherent."""
    row = ireg.CAPABILITIES["sd15"]
    backends = list(row["device_backends"])
    assert "mps" in backends, (
        "sd15 published episodes on a Mac mini M4 on 2026-09-08; "
        "config/machine_classes.json carries the receipt")
    assert "cuda" in backends, "removing cuda would strand every NVIDIA profile"
    assert row["model_requirements"], (
        "sd15 needs a checkpoint and the row must say so, or preflight cannot "
        "tell anyone what to fetch")


def test_it_is_a_local_engine_that_needs_a_gpu_to_be_practical():
    row = ireg.CAPABILITIES["sd15"]
    assert row["requires_vendor"] is None, "no vendor lock -- that is the point"
    assert row["requires_sidecar"] is False
    assert row["required_toolchain"] is None
    assert row["needs_fp8_te"] is False and row["needs_fp4_te"] is False, (
        "fp8/fp4 are the two things Metal cannot do; sd15 needs neither, which "
        "is why it works here")


# ---------------------------------------------------------------------------
# _fit_native -- the silent two-headed-still defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w,h,expect", [
    # The canonical's real canvas. 832 > 768, so it MUST come down.
    (832, 480, (768, 440)),
    # The other real one seen in production logs.
    (1472, 832, (768, 432)),
    # Already inside the ceiling: snap to 8, do not upscale.
    (512, 512, (512, 512)),
    (768, 768, (768, 768)),
    # Snap-only, no scaling: 750 -> 744 is the multiple-of-8 floor.
    (750, 750, (744, 744)),
    # Portrait: the LONG side is the one that is clamped.
    (480, 832, (440, 768)),
    # Degenerate input must not produce a zero dimension.
    (1, 1, (8, 8)),
])
def test_fit_native_clamps_the_long_side_and_keeps_the_aspect(w, h, expect):
    assert S._fit_native(w, h, S._DEFAULT_MAX_SIDE) == expect


def test_fit_native_never_returns_a_dimension_that_is_not_a_multiple_of_8():
    """A latent is width/8; a non-multiple silently truncates."""
    for w in range(100, 1600, 37):
        for h in (240, 480, 832):
            fw, fh = S._fit_native(w, h, S._DEFAULT_MAX_SIDE)
            assert fw % 8 == 0 and fh % 8 == 0, (w, h, fw, fh)
            assert fw >= 8 and fh >= 8


def test_the_default_ceiling_is_768_because_sd15_duplicates_past_it():
    """768 is not a preference. SD 1.5 duplicates subjects on a long side past
    it -- confidently wrong output with no error -- and the composer ALWAYS
    stamps width/height, so a lower "default" would simply never bind."""
    assert S._DEFAULT_MAX_SIDE == 768
    assert S.MAX_SIDE_ENV == "OTR_SD15_MAX_SIDE"


def test_raising_the_ceiling_is_possible_and_deliberate(monkeypatch):
    """The clamp is an opinion, not a cage: an operator who wants the larger
    canvas can have it, which is why the log line says so when it clamps."""
    assert S._fit_native(1472, 832, 1536) == (1472, 832)


# ---------------------------------------------------------------------------
# _sd15_params -- the empty-prompt defect
# ---------------------------------------------------------------------------

class _Eng(S.SD15Engine):
    """The adapter with nothing stubbed -- _sd15_params is pure."""


@pytest.mark.parametrize("request_obj", [
    {"prompt": "a brass radio dial", "width": 832, "height": 480},
    pytest.param(
        type("R", (), {"prompt": "a brass radio dial", "width": 832,
                       "height": 480})(),
        id="attribute-style-request"),
])
def test_it_reads_prompt_and_not_text_prompt(request_obj):
    """THE DEFECT, PINNED. `required_inputs` says "text_prompt" because that is
    what the ROLE supplies; the dispatcher hands the engine "prompt". Reading
    the former mints every still from an empty string -- a valid PNG of nothing
    in particular, with no error at any layer."""
    params = _Eng()._sd15_params(request_obj)
    assert params["prompt"] == "a brass radio dial"


def test_a_text_prompt_only_request_does_not_smuggle_a_prompt_in():
    """The mirror of the above: if someone "fixes" this by reading both, this
    test says which one is authoritative."""
    params = _Eng()._sd15_params({"text_prompt": "ignored", "width": 512,
                                  "height": 512})
    assert params["prompt"] == "", (
        "text_prompt is NOT the field the dispatcher supplies; accepting it "
        "hides the defect rather than fixing it")


def test_a_none_prompt_becomes_empty_string_not_the_word_None():
    """``str(None)`` is "None", which is a prompt SD 1.5 will happily render."""
    params = _Eng()._sd15_params({"prompt": None, "width": 512, "height": 512})
    assert params["prompt"] == ""


def test_the_request_dims_are_clamped_on_the_way_through(monkeypatch):
    """_fit_native is not decoration -- the params the graph receives carry the
    clamped values, so a caller cannot route around it."""
    params = _Eng()._sd15_params({"prompt": "x", "width": 832, "height": 480})
    assert (params["width"], params["height"]) == (768, 440)


def test_raising_width_alone_is_still_clamped(monkeypatch):
    """OTR_SD15_WIDTH is a fallback for a request that omits dims, NOT an
    override of the ceiling. Setting it must not reopen the 832 defect."""
    monkeypatch.setenv("OTR_SD15_WIDTH", "1024")
    monkeypatch.setenv("OTR_SD15_HEIGHT", "1024")
    params = _Eng()._sd15_params({"prompt": "x"})
    assert max(params["width"], params["height"]) <= S._DEFAULT_MAX_SIDE


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_it_declares_the_identity_the_registry_bijection_needs():
    eng = _Eng()
    assert eng.name == "sd15"
    assert eng.accepts_reference_image is False, (
        "sd15 is text-to-image here; claiming reference support would let the "
        "director route a reference to a lane that drops it")
    assert getattr(eng, "engine_version", None), (
        "an engine_version is what lets a cached still be invalidated when the "
        "recipe changes")
    assert eng.required_inputs, "the role gate reads this"


def test_the_checkpoint_resolver_is_one_function_and_env_overridable(
        monkeypatch):
    """One resolver shared by assert_usable and the graph builder, so the gate
    cannot check a different file from the one the loader opens."""
    assert S._resolve_ckpt_name() == S._DEFAULT_CKPT
    monkeypatch.setenv(S.CKPT_ENV, "something-else.safetensors")
    assert S._resolve_ckpt_name() == "something-else.safetensors"
    # Read the constant rather than hardcoding the name: a rename should move
    # this test with it, not break it.
    assert S.CKPT_ENV.startswith("OTR_SD15")


def test_a_visual_pack_may_name_its_own_checkpoint_when_that_file_exists(
        monkeypatch):
    """A PACK'S CHECKPOINT IS A PREFERENCE, NOT A GATE (operator, 2026-09-12:
    *"an anime SD1.5 would really pop"*).

    `_installed` asks ComfyUI's `folder_paths`, which does not exist in a test
    process, so the installed check is what is monkeypatched here -- it is also
    the reason a bare probe outside ComfyUI always reports the default.
    """
    monkeypatch.setattr(S, "_installed", lambda name: True)
    assert S._resolve_ckpt_name("anime") == "Counterfeit-V3.0_fp16.safetensors"
    # THE PACK OUTRANKS THE ENV, because it is the more specific statement and
    # the env is a whole-engine escape hatch.
    monkeypatch.setenv(S.CKPT_ENV, "something-else.safetensors")
    assert S._resolve_ckpt_name("anime") == "Counterfeit-V3.0_fp16.safetensors"
    # ...but only for the pack that names one.
    assert S._resolve_ckpt_name("sci_fi_radio") == "something-else.safetensors"


def test_a_pack_checkpoint_that_is_not_installed_costs_nothing(monkeypatch):
    """THE WHOLE SAFETY PROPERTY. A pack naming weights this box never fetched
    must not grey the engine out or fail a render -- it falls through to the
    env override and then the shipped default, exactly as before the field
    existed. `assert_usable` gates on this same resolver, so what it checks is
    always a file that is actually present."""
    monkeypatch.setattr(S, "_installed", lambda name: False)
    assert S._resolve_ckpt_name("anime") == S._DEFAULT_CKPT
    assert S._style_ckpt_name("anime") == ""
    monkeypatch.setenv(S.CKPT_ENV, "operator-choice.safetensors")
    assert S._resolve_ckpt_name("anime") == "operator-choice.safetensors"


def test_an_unknown_or_blank_style_resolves_exactly_as_before():
    """Every pre-existing caller passes nothing. A style lookup must never be
    able to fail a mint, so a bad id is "" rather than an exception."""
    assert S._resolve_ckpt_name() == S._DEFAULT_CKPT
    assert S._resolve_ckpt_name("") == S._DEFAULT_CKPT
    assert S._resolve_ckpt_name("no_such_style") == S._DEFAULT_CKPT
    assert S._style_ckpt_name("no_such_style") == ""
    assert S._style_ckpt_name(None) == ""


def test_the_dispatcher_stamps_the_style_on_the_request():
    """THE WIRING, AT ITS REAL SITE. The per-style checkpoint is unreachable
    unless the request carries the style id, and the request never carried it
    before this change."""
    import inspect as _inspect
    from nodes import otr_image_gen_dispatcher as D
    src = _inspect.getsource(D)
    assert '"visual_style": (str(getattr(_vstyle, "style_id", "") or "")' in src, (
        "the engine request must stamp visual_style")
    # and the engine must read that exact key. `_sd15_params` is a METHOD on
    # the engine class, not a module function.
    engine_cls = next(
        obj for _n, obj in vars(S).items()
        if isinstance(obj, type) and hasattr(obj, "_sd15_params"))
    assert 'get("visual_style")' in _inspect.getsource(
        engine_cls._sd15_params)


def test_the_default_checkpoint_is_the_ungated_archive_copy():
    """`stabilityai/*` historically gated its SD 1.5 repo behind a terms click,
    which breaks the auto-install property. The Comfy-Org archive does not."""
    assert S._DEFAULT_CKPT == "v1-5-pruned-emaonly-fp16.safetensors"
