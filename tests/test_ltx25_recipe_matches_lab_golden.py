"""Our LTX 2.5 constants must agree with the lab's executable golden recipe.

WHY THIS EXISTS. `nodes/_otr_video_engines/ltx25_recipe.py` transcribes numbers
the lab measured and froze. A transcription can drift two ways -- someone edits
our constants, or the lab revises theirs -- and both failures are silent until
a render comes out wrong. This test reads the lab's ACTUAL workflow file and
compares, so the transcription is checked rather than trusted.

IT WAS ALSO WORTH WRITING BECAUSE THE PROSE AND THE JSON DISAGREED. The lab's
QA document describes the first-frame anchor as ``SetLatentNoiseMask`` with
frame 0 at 0.0; the executable recipe uses ``LTXVImgToVideoInplace`` at
strength 1.0. Same intent, different node. And the QA says the scheduler's
latent is connected to ``EmptyLTXVLatentVideo``, while the JSON connects it to
the ImgToVideoInplace OUTPUT -- which on the I2V path is a different tensor.
**The file that ran wins.** These tests encode that rule.

THE LAB FILE LIVES OUTSIDE THIS REPO, so every test here SKIPS cleanly when it
is absent rather than failing. A missing lab checkout must never turn this
suite red -- that would punish the wrong person for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes._otr_video_engines import ltx25_recipe as R
from nodes._otr_video_engines import eng_ltx25
from nodes._otr_video_engines.wrapper_bridge import Wire


GOLDEN = Path(
    r"C:/Users/jeffr/Documents/ComfyUI/vram-recipe-lab/recipes"
    r"/ltx_2_5_golden_i2v_foley.json"
)
TWO_STAGE = Path(
    r"C:/Users/jeffr/Documents/ComfyUI/vram-recipe-lab/recipes"
    r"/ltx_2_5_two_stage.json"
)


def _graph():
    if not GOLDEN.is_file():
        pytest.skip("lab golden recipe not present at %s" % GOLDEN)
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return doc, doc.get("prompt") or {}


def _two_stage_graph():
    if not TWO_STAGE.is_file():
        pytest.skip("lab two-stage recipe not present at %s" % TWO_STAGE)
    doc = json.loads(TWO_STAGE.read_text(encoding="utf-8"))
    return doc, doc.get("prompt") or {}


def _node(graph, class_type):
    """The single node of a class. Asserts uniqueness -- if the lab ever ships
    two of something we assume is unique, that is itself the finding."""
    hits = [(k, v) for k, v in graph.items() if v.get("class_type") == class_type]
    assert len(hits) == 1, "%s appears %d times, expected exactly 1" % (
        class_type, len(hits))
    return hits[0]


# --------------------------------------------------------------------------
# The contract block
# --------------------------------------------------------------------------

def test_canvas_and_length_match_the_lab_contract():
    doc, _ = _graph()
    c = doc["contract"]
    assert (c["width"], c["height"]) == (R.LTX25_CANVAS_W, R.LTX25_CANVAS_H)
    assert c["frames"] == R.LTX25_FRAMES
    assert int(c["fps"]) == R.LTX25_FPS


def test_the_canvas_is_modulo_32_legal_on_both_axes():
    """G2.4. The lab rejected 768x432 because 432/32 = 13.5 corrupts the VAE."""
    assert R.LTX25_CANVAS_W % 32 == 0, R.LTX25_CANVAS_W
    assert R.LTX25_CANVAS_H % 32 == 0, R.LTX25_CANVAS_H


def test_the_frame_count_satisfies_the_temporal_contract():
    """(frames - 1) must be divisible by 8 for temporal downsampling."""
    assert (R.LTX25_FRAMES - 1) % 8 == 0


def test_frames_and_fps_give_the_standard_otr_shot_length():
    doc, _ = _graph()
    assert abs(R.LTX25_FRAMES / R.LTX25_FPS - doc["contract"]["duration_s"]) < 0.01


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

def test_the_dit_is_the_q3_the_operator_locked():
    _doc, g = _graph()
    _k, n = _node(g, "UnetLoaderGGUF")
    assert n["inputs"]["unet_name"] == R.LTX25_DIT_GGUF
    assert "Q3" in R.LTX25_DIT_GGUF, (
        "Q3 is the locked safe quant; Q5 breaches the clamp and is quarantined"
    )


def test_the_text_encoder_matches():
    _doc, g = _graph()
    _k, n = _node(g, "CLIPLoaderGGUF")
    assert n["inputs"]["clip_name"] == R.LTX25_TEXT_ENCODER_GGUF


def test_both_vaes_match_and_the_audio_vae_really_is_loaded():
    """The silent lane still needs the audio VAE -- LTXVEmptyLatentAudio mints
    the audio latent with it, and the concat needs that latent."""
    _doc, g = _graph()
    vaes = {v["inputs"]["vae_name"] for v in g.values()
            if v.get("class_type") == "VAELoader"}
    assert R.LTX25_VIDEO_VAE in vaes
    assert R.LTX25_AUDIO_VAE in vaes
    assert R.LTX25_AUDIO_VAE_REQUIRED_EVEN_WHEN_SILENT is True

    _k, empty_audio = _node(g, "LTXVEmptyLatentAudio")
    assert "audio_vae" in empty_audio["inputs"], (
        "if this input vanishes, a silent lane could skip loading the audio VAE"
    )


# --------------------------------------------------------------------------
# Sampling -- the CFG values are a VRAM contract, not taste
# --------------------------------------------------------------------------

def test_sampler_and_steps_match():
    _doc, g = _graph()
    _k, sel = _node(g, "KSamplerSelect")
    assert sel["inputs"]["sampler_name"] == R.LTX25_SAMPLER
    _k, sched = _node(g, "LTXVScheduler")
    assert sched["inputs"]["steps"] == R.LTX25_STEPS


def test_every_cfg_is_exactly_one():
    """CFG stays at 1.0. This is a memory contract; do not 'tune' it.

    THE REASON GIVEN HERE USED TO BE FALSE. It said 1.0 keeps batch size 1 and
    that anything above "forces batch size 2 (positive + negative)". Under the
    locked sampler ``euler_ancestral_cfg_pp`` that is wrong: CFG++ forces
    ``disable_cfg1_optimization=True``, so the unconditional branch is
    evaluated at 1.0 TOO. The saving from staying at 1.0 is not a halved batch.

    The lock still stands and is still a memory contract -- raising CFG raises
    the peak and this stack has ~150 MiB of headroom on a clean box -- but a
    window that trusts the old wording will mis-predict VRAM and will also
    conclude the negative encode is free to delete. It is not: the negative is
    LIVE and steers every step.
    """
    _doc, g = _graph()
    _k, guider = _node(g, "LTXVDualCFGGuider")
    assert guider["inputs"]["video_cfg"] == R.LTX25_CFG_VIDEO == 1.0
    assert guider["inputs"]["audio_cfg"] == R.LTX25_CFG_AUDIO == 1.0
    _k, modality = _node(g, "LTXVModalityGuidance")
    assert modality["inputs"]["modality_scale"] == R.LTX25_CFG_MODALITY == 1.0


def test_the_negative_prompt_is_empty():
    """The negative TEXT is empty. The negative CONDITIONING is not inert.

    THIS DOCSTRING USED TO SAY *"Inert at CFG 1.0, so it is removed rather than
    carried"* AND THAT WAS FALSE -- the fourth surviving copy of a premise the
    2026-08-19 window believed it had corrected in three places, found by the
    Codex lane on 2026-08-20. The ordinary ComfyUI rule (cfg 1.0 elides the
    uncond) does NOT hold here: the locked sampler ``euler_ancestral_cfg_pp``
    forces ``disable_cfg1_optimization=True`` and consumes ``uncond_denoised``
    in its own step derivative, so the unconditional branch is computed every
    step and steers the output.

    The distinction matters because it is load-bearing twice over: it is why
    the ``neg`` encode may never be wired from ``pos`` as a free optimisation,
    and it is why the empty negative is worth CACHING per episode rather than
    deleting. What this test asserts is unchanged -- the recipe carries an
    empty negative STRING -- only the reason was wrong.
    """
    _doc, g = _graph()
    texts = [v["inputs"].get("text") for v in g.values()
             if v.get("class_type") == "CLIPTextEncode"]
    assert R.LTX25_NEGATIVE_PROMPT in texts, (
        "the golden recipe should carry an empty negative; got %r" % (texts,)
    )


# --------------------------------------------------------------------------
# The two places the prose and the JSON disagreed
# --------------------------------------------------------------------------

def test_the_anchor_node_is_ImgToVideoInplace_not_SetLatentNoiseMask():
    """The QA prose says SetLatentNoiseMask. The file that RAN says otherwise."""
    _doc, g = _graph()
    _k, anchor = _node(g, R.LTX25_I2V_ANCHOR_NODE)
    assert anchor["inputs"]["strength"] == R.LTX25_I2V_ANCHOR_STRENGTH
    assert anchor["inputs"]["bypass"] is False
    assert "SetLatentNoiseMask" not in {
        v.get("class_type") for v in g.values()
    }, "the prose's node appeared after all -- re-read the recipe before coding"


def test_the_scheduler_latent_comes_from_the_anchor_not_the_empty_latent():
    """CORRECTION the JSON forced. Wiring this to EmptyLTXVLatentVideo would
    hand the scheduler a latent with no still baked in, and the failure would
    be wrong-but-running -- the worst kind."""
    _doc, g = _graph()
    sched_key, sched = _node(g, "LTXVScheduler")
    src_key = sched["inputs"]["latent"][0]
    assert g[src_key]["class_type"] == R.LTX25_SCHEDULER_LATENT_SOURCE, (
        "scheduler latent now comes from %s" % g[src_key]["class_type"]
    )
    assert R.LTX25_SCHEDULER_LATENT_MUST_BE_CONNECTED is True


# --------------------------------------------------------------------------
# Closed options stay closed
# --------------------------------------------------------------------------

def test_the_original_golden_remains_one_stage_but_production_selects_two_stage():
    """The old golden stays useful as stage-one evidence; it no longer bans
    the selected production refinement that the operator approved by eye."""
    _doc, g = _graph()
    assert R.LTX25_MULTISHOT_ALLOWED is False
    assert R.LTX25_INGRAPH_UPSCALE_ALLOWED is True
    classes = {v.get("class_type") for v in g.values()}
    upscalers = {c for c in classes if "Upscale" in c or "upscale" in c}
    assert not upscalers, "the golden graph grew an upscaler: %s" % upscalers
    _k, empty = _node(g, "EmptyLTXVLatentVideo")
    assert empty["inputs"]["length"] == R.LTX25_FRAMES == 97, (
        "161-frame multishot is banned; length must stay at the 97 rung"
    )


def test_the_selected_two_stage_lab_recipe_matches_production_constants():
    doc, g = _two_stage_graph()
    assert doc["name"] == R.LTX25_TWO_STAGE_RECIPE_ID
    assert (doc["contract"]["width"], doc["contract"]["height"]) == (832, 480)
    _k, loader = _node(g, "LatentUpscaleModelLoader")
    assert loader["inputs"]["model_name"] == R.LTX25_UPSCALER_MODEL
    _k, manual = _node(g, "ManualSigmas")
    lab_sigmas = manual["inputs"].get("sigmas_string", manual["inputs"].get("sigmas"))
    assert lab_sigmas == R.LTX25_REFINE_SIGMAS
    decodes = [v for v in g.values() if v.get("class_type") == "VAEDecodeTiled"]
    assert len(decodes) == 1
    dec = decodes[0]["inputs"]
    assert dec["tile_size"] == R.LTX25_STAGE2_DECODE_TILE_SIZE
    assert dec["overlap"] == R.LTX25_STAGE2_DECODE_OVERLAP
    assert dec["temporal_size"] == R.LTX25_STAGE2_DECODE_TEMPORAL_SIZE
    assert dec["temporal_overlap"] == R.LTX25_STAGE2_DECODE_TEMPORAL_OVERLAP


def test_the_entire_selected_stage_two_matches_the_lab_byte_for_byte_semantically():
    """Normalize numeric node ids, then compare every class, literal, wire
    endpoint, and output slot in stage two.

    Raw JSON bytes cannot match because the lab uses Comfy API ids while OTR
    uses named direct-call nodes. This is the stricter useful comparison: all
    bytes that can alter execution are equal after that transport translation.
    """
    _doc, lab = _two_stage_graph()
    engine = eng_ltx25.Ltx25VideoEngine()
    prod = engine._build_graph(
        {"text_prompt": "probe", "seed": 42}, "still.png", 97, 832, 480)

    ids = {
        "2": "videovae", "8": "ksel", "9": "noise", "10": "guider",
        "15": "preprocess", "32": "separate", "33": "decode",
        "100": "upscale_loader", "101": "latent_upscale",
        "102": "refine_i2v", "103": "refine_sigmas",
        "104": "refine_sampler", "105": "refine_concat",
        "106": "refine_separate",
    }
    stage = ("upscale_loader", "latent_upscale", "refine_i2v",
             "refine_sigmas", "refine_concat", "refine_sampler",
             "refine_separate", "decode")
    candidates = engine._node_candidates()

    def normalize_value(value, *, production):
        if production and isinstance(value, Wire):
            return [value.src, value.slot]
        if not production and isinstance(value, list) and len(value) == 2 \
                and str(value[0]) in ids and isinstance(value[1], int):
            return [ids[str(value[0])], value[1]]
        if isinstance(value, dict):
            return {k: normalize_value(v, production=production)
                    for k, v in sorted(value.items())}
        return value

    lab_by_name = {ids[node_id]: spec for node_id, spec in lab.items()
                   if node_id in ids and ids[node_id] in stage}
    for name in stage:
        lab_spec = lab_by_name[name]
        prod_spec = prod[name]
        assert candidates[prod_spec["class"]][0] == lab_spec["class_type"], name
        lab_inputs = dict(lab_spec["inputs"])
        prod_inputs = dict(prod_spec["inputs"])
        assert normalize_value(prod_inputs, production=True) == normalize_value(
            lab_inputs, production=False), name


def test_the_tiled_decode_knobs_match_the_lab_and_are_not_the_siblings():
    """The decode knobs are RECIPE values and drift silently if left as
    literals. The sibling ``eng_ltx_av`` decodes whole-clip at 4096/8 by
    default; inheriting that by resemblance would change a measured recipe on
    a lane with 0.02 GiB of headroom, and nothing would say so."""
    _doc, g = _graph()
    _k, dec = _node(g, "VAEDecodeTiled")
    assert dec["inputs"]["tile_size"] == R.LTX25_DECODE_TILE_SIZE
    assert dec["inputs"]["overlap"] == R.LTX25_DECODE_OVERLAP
    assert dec["inputs"]["temporal_size"] == R.LTX25_DECODE_TEMPORAL_SIZE
    assert dec["inputs"]["temporal_overlap"] == R.LTX25_DECODE_TEMPORAL_OVERLAP
    assert R.LTX25_DECODE_TEMPORAL_SIZE != 4096, (
        "4096 is eng_ltx_av's whole-clip default, not this lane's recipe")


def test_the_peak_decomposition_sums_to_the_observed_peak():
    """The lab's correction, pinned: the 14.48 GiB peak is DiT weights plus
    activations plus allocator context, with the encoder and VAEs at ZERO.

    This is what makes 'aggressive staging will bring the peak down' false --
    at the moment of the peak the encoder is not resident to free. Pinned as
    arithmetic so the claim cannot rot into folklore."""
    parts = R.LTX25_PEAK_DECOMPOSITION_GIB
    assert parts["text_encoder"] == 0.0
    assert parts["vaes"] == 0.0
    assert abs(sum(parts.values()) - R.LTX25_LAB_OBSERVED_PEAK_GIB) < 0.005, (
        "decomposition %r does not sum to the observed %r"
        % (parts, R.LTX25_LAB_OBSERVED_PEAK_GIB))
    assert R.LTX25_STAGING_REDUCES_PEAK is False


def test_the_lab_vram_figure_is_recorded_but_not_used_as_qualification():
    """CLAUDE.md 0A: a bench result may never be worded as qualification. The
    number is kept for traceability; OUR envelope comes from our own smoke."""
    doc, _ = _graph()
    assert doc["contract"]["vram_ceiling_gb"] == R.LTX25_LAB_CLAMP_GIB
    assert R.LTX25_LAB_OBSERVED_PEAK_GIB < R.LTX25_LAB_CLAMP_GIB
