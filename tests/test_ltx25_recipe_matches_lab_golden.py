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


GOLDEN = Path(
    r"C:/Users/jeffr/Documents/ComfyUI/vram-recipe-lab/recipes"
    r"/ltx_2_5_golden_i2v_foley.json"
)


def _graph():
    if not GOLDEN.is_file():
        pytest.skip("lab golden recipe not present at %s" % GOLDEN)
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
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
    """Above 1.0 forces batch size 2 (positive + negative) and OOMs past the
    clamp. This is a memory contract; do not 'tune' it."""
    _doc, g = _graph()
    _k, guider = _node(g, "LTXVDualCFGGuider")
    assert guider["inputs"]["video_cfg"] == R.LTX25_CFG_VIDEO == 1.0
    assert guider["inputs"]["audio_cfg"] == R.LTX25_CFG_AUDIO == 1.0
    _k, modality = _node(g, "LTXVModalityGuidance")
    assert modality["inputs"]["modality_scale"] == R.LTX25_CFG_MODALITY == 1.0


def test_the_negative_prompt_is_empty():
    """Inert at CFG 1.0, so it is removed rather than carried."""
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

def test_no_multishot_and_no_in_graph_upscaler():
    """Both were measured to blow the clamp and are banned from this graph.
    The upscaler weight exists on this box and may run as an OFFLINE pass."""
    _doc, g = _graph()
    assert R.LTX25_MULTISHOT_ALLOWED is False
    assert R.LTX25_INGRAPH_UPSCALE_ALLOWED is False
    classes = {v.get("class_type") for v in g.values()}
    upscalers = {c for c in classes if "Upscale" in c or "upscale" in c}
    assert not upscalers, "the golden graph grew an upscaler: %s" % upscalers
    _k, empty = _node(g, "EmptyLTXVLatentVideo")
    assert empty["inputs"]["length"] == R.LTX25_FRAMES == 97, (
        "161-frame multishot is banned; length must stay at the 97 rung"
    )


def test_the_lab_vram_figure_is_recorded_but_not_used_as_qualification():
    """CLAUDE.md 0A: a bench result may never be worded as qualification. The
    number is kept for traceability; OUR envelope comes from our own smoke."""
    doc, _ = _graph()
    assert doc["contract"]["vram_ceiling_gb"] == R.LTX25_LAB_CLAMP_GIB
    assert R.LTX25_LAB_OBSERVED_PEAK_GIB < R.LTX25_LAB_CLAMP_GIB
