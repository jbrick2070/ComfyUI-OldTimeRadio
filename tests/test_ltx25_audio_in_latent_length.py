# -*- coding: utf-8 -*-
"""The audio-in lanes must hand the sampler a picture-length audio latent.

WRITTEN AFTER THE DEFECT SHIPPED AND FAILED FIVE LIVE LEGS (2026-09-23).

The first design swapped the class behind the ``emptyaudio`` graph key so that
``LTXVAudioVAEEncode`` stood where ``LTXVEmptyLatentAudio`` had, leaving
``concat`` to read the encoder directly. It is the obvious move and it is
wrong, for a reason no unit test in this repo could see at the time:

  * ``LTXVEmptyLatentAudio`` is TOLD ``frames_number`` and ``frame_rate`` and
    returns exactly ``round(frames / fps * latents_per_second)`` latents.
  * ``LTXVAudioVAEEncode`` returns whatever its own mel and downsample rounding
    produce for the waveform it is handed.

MEASURED: a 97-frame beat whose reference WAV was ALREADY exactly 3.88 s
(171,108 samples at 44.1 kHz, resampling to exactly the 186,240 the clip needs
at 48 kHz) encoded to ONE latent fewer than the picture, decoded 3,360 samples
short, and was refused by ``_write_foley_stem`` as INVALID_DAG. The input was
never wrong; the encode/decode round trip is lossy and only the empty-latent
node knows the target.

THE FIX IS A SECOND CONCAT, AND ITS VIDEO INPUT IS THE LOAD-BEARING PART.
``LTXVConcatAVLatent`` only runs ``fit_audio`` on the branch where its
``video_latent`` is ALREADY an AV latent (``comfy_extras/nodes_lt.py`` --
``if video_samples.is_nested:``). Wiring ``refconcat`` to ``i2v`` instead of to
``concat`` would look equally sensible, pass every other check in this file,
and silently restore the bug -- so that wire is asserted by name.

A waveform-level reconciliation was tried first and could not fire at all,
because the waveform already had the exact duration asked of it. That is why
this file asserts the GRAPH, not the input audio.
"""
from __future__ import annotations

import inspect

import pytest

from nodes._otr_video_engines import eng_ltx25

AUDIO_IN = (
    eng_ltx25.Ltx25NativeAudioIn16gbEngine,
    eng_ltx25.Ltx25NativeAudioIn24gbEngine,
)

_GRAPH_SRC = inspect.getsource(eng_ltx25.Ltx25NativeAudioInMixin._build_graph)


@pytest.mark.parametrize("cls", AUDIO_IN, ids=lambda c: c.name)
def test_the_empty_latent_node_is_not_hijacked(cls):
    """``emptyaudio`` must stay the node that knows the picture's length."""
    resolved = cls()._node_candidates()["emptyaudio"]
    assert resolved == ("LTXVEmptyLatentAudio",), (
        "%s resolved `emptyaudio` to %r. That key is the ONLY source of the "
        "audio latent length the picture needs; pointing it at the encoder is "
        "the defect that failed five live legs." % (cls.name, resolved))


@pytest.mark.parametrize("cls", AUDIO_IN, ids=lambda c: c.name)
def test_the_reference_encoder_and_its_fitting_concat_are_both_present(cls):
    cand = cls()._node_candidates()
    assert cand.get("refencode") == ("LTXVAudioVAEEncode",)
    assert cand.get("refconcat") == ("LTXVConcatAVLatent",)
    assert cand.get("loadaudio") == ("LoadAudio",)


def test_the_graph_does_not_rebuild_the_empty_audio_node():
    assert 'g["emptyaudio"]' not in _GRAPH_SRC, (
        "the audio-in graph reassigns `emptyaudio`. Leave it alone -- it is "
        "what the encoded stream is fitted TO.")


def test_the_fitting_concat_takes_an_av_latent_so_fit_audio_actually_runs():
    """This is the wire the whole fix rests on.

    ``fit_audio`` is reached only when ``video_latent`` is already nested.
    ``concat`` emits an AV latent; ``i2v`` emits a plain video latent.
    """
    assert '"video_latent": W("concat", 0)' in _GRAPH_SRC, (
        "`refconcat` must take `concat`'s AV latent. Any plain video latent "
        "(`i2v`, `latent_upscale`) skips fit_audio entirely and the encoded "
        "stream reaches the sampler at the encoder's own length.")
    assert '"audio_latent": W("refmask", 0)' in _GRAPH_SRC, (
        "`refconcat` must take the FROZEN latent, not the raw encoder output "
        "-- see test_the_supplied_audio_is_frozen_not_regenerated.")


@pytest.mark.parametrize("cls", AUDIO_IN, ids=lambda c: c.name)
def test_the_freeze_nodes_are_registered(cls):
    cand = cls()._node_candidates()
    assert cand.get("refsolid") == ("SolidMask",)
    assert cand.get("refmask") == ("SetLatentNoiseMask",)


def test_the_supplied_audio_is_frozen_not_regenerated():
    """Without this the lane carries the reference and then discards it.

    ``LTXVAudioVAEEncode`` returns no noise mask.
    ``LTXVConcatAVLatent.execute`` substitutes ``ones_like`` for a missing
    AUDIO mask whenever the VIDEO side has one, and the i2v anchor always
    gives it one. All-ones means "generate this", and stage one starts at
    sigma 1.0 where ``sigma * noise + (1 - sigma) * latent_image`` takes
    exactly nothing from the reference.

    The proven ``ltx_audio_in`` lane rides its audio latent under
    ``SolidMask(0)`` -> ``SetLatentNoiseMask`` for this reason
    (``eng_ltx_av.py``). Raising ``modality_scale`` cannot recover audio that
    was already thrown away. Found by a codex review of `bed7556d`.
    """
    assert '"value": 0.0' in _GRAPH_SRC, (
        "the solid mask must be 0.0 -- a 1.0 mask regenerates the reference, "
        "which is the exact defect this guards.")
    assert '"samples": W("refencode", 0), "mask": W("refsolid", 0)' in _GRAPH_SRC, (
        "the freeze must wrap the ENCODED reference.")


def test_the_sampler_consumes_the_fitted_latent_not_the_unfitted_one():
    assert 'g["sampler"]["inputs"]["latent_image"] = W("refconcat", 0)' in _GRAPH_SRC, (
        "the sampler still reads `concat`, so the fitted audio stream is "
        "built and then thrown away.")


def test_the_reference_is_padded_before_staging_not_after():
    """The pad is useless unless it runs on the file the graph actually loads."""
    assert ('audio_path = self._pad_reference_for_vae_crop(audio_path, length)\n'
            '        audio_name = _wb.stage_into_comfy_input(audio_path)'
            ) in _GRAPH_SRC, (
        "the VAE-crop pad must produce the path that is staged into ComfyUI; "
        "padding a file nobody loads is the shape of the previous dead fix.")


@pytest.mark.parametrize("have,rate", [
    (171108, 44100),   # the real measured slice: 171108 % 4096 == 3172
    (2205, 44100),     # 50 ms -- the old crop reduced this to ZERO samples
    (400000, 44100),   # longer than the clip
    (172032, 44100),   # already a multiple: must be left alone
])
def test_the_padded_reference_is_a_multiple_the_vae_crop_will_not_touch(
        tmp_path, have, rate):
    """Executed, not asserted from source.

    ``vae_encode_crop_pixels`` narrows to ``(n // 4096) * 4096`` taking
    ``(n % 4096) // 2`` off the FRONT, so any non-multiple loses the head of
    the beat's audio. On an exact multiple ``x == dims[d]`` and it never
    narrows.
    """
    import numpy as np

    from nodes._otr_video_engines import foley_stems as fs

    src = tmp_path / "slice_test.wav"
    fs.write_pcm16_wav(str(src), np.zeros((1, have), dtype=np.float32), rate)

    engine = eng_ltx25.Ltx25NativeAudioIn24gbEngine()
    out = engine._pad_reference_for_vae_crop(str(src), 97)
    samples, out_rate = fs.read_pcm16_wav(out)
    n = int(samples.shape[-1])

    assert out_rate == rate
    assert n % 4096 == 0, (
        "%d samples leaves a remainder of %d, so the VAE crop would take "
        "%d samples off the front of the reference"
        % (n, n % 4096, (n % 4096) // 2))
    assert n >= int(round(97 / 25.0 * rate)), "padded below the clip duration"
    assert n >= have, "the reference itself must never be shortened here"
    if have % 4096 == 0 and have >= int(round(97 / 25.0 * rate)):
        assert out == str(src), "an exact multiple must not be rewritten"


def test_stage_two_inherits_the_fixed_length_and_needs_no_second_fix():
    """Documented here so nobody 'fixes' stage two as well and double-pads."""
    base = inspect.getsource(eng_ltx25.Ltx25VideoEngine._build_graph)
    assert '"audio_latent": W("separate", 1)' in base, (
        "stage two no longer takes its audio from stage one's sampled stream; "
        "if that changed, the audio-in length fix may need a stage-two twin.")
