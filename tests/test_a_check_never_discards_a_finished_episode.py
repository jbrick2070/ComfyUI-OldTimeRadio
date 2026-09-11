"""A mixed sample rate is converted, not fatal. Operator: only an OOM should fail.

FOUND BY THE 2026-09-11 FAIL-CLOSED AUDIT -- seven readers over the render path,
every finding put to an adversarial verifier that tried to refute it. Four
survived; this is the one that was fixed.

THE DEFECT. `pack_audio_batch` raised ValueError when a per-line clip's real rate
differed from the rate the caller declared -- AFTER every line in the role had been
generated and paid for. `OTRVoiceNodeBase.generate()` wraps that loop in try/finally
with no except, so the error left the node and ended the whole prompt.

IT HAS ALREADY HAPPENED IN PRODUCTION. `eda8590c` (2026-06-05) records a live cast
crashing on exactly this message with rates [22050, 24000] -- an indextts2 character
(22,050 Hz) beside a bark-fallback one (24,000 Hz). The fix then was a
`resample_audio` helper at the one caller; it was retired along with the fallback
branch that called it, and the hazard came back to a path nothing guarded.

WHY THIS IS NOT THE CARVE-OUT. A raise on the render path is legitimate only against
a silently WRONG render. Resampling is deterministic and content-preserving: the line
says the same words, in the same voice, from the same engine, and only the sample
grid changes. Every downstream consumer demands exactly the one rate this produces.

WHAT THE FINISHED-DIFF REVIEW CHANGED, and it was right. The first cut of the helper
returned the ORIGINAL waveform when conversion failed -- and `pack_audio_batch` then
labelled it with the DESTINATION rate. A 22,050 Hz clip labelled 24,000 Hz plays 8.8%
fast and pitch-shifted, and the slicer, the mux and the duration gates all read the
label rather than the samples. That is precisely the silent-wrong case the change
claimed to avoid. It now CONVERTS OR RAISES, and `MemoryError` propagates untouched
so resource death stays loud.

NOT IN THIS FILE ANY MORE: the caption-burn title-card refusal. It was changed to
degrade, and the change was REVERTED the same hour -- that branch catches every
ValueError, including an unknown caption style, where refusing is correct because a
misconfiguration would otherwise ship untitled episodes forever in silence. Making it
degrade only on a platform-capability gap needs a classification that does not exist
yet, which is a design choice with more than one defensible answer. See the
GO_FORWARD row.
"""
from __future__ import annotations

import logging

import pytest
import torch

from nodes._otr_audio_engines.base import pack_audio_batch
from nodes._otr_audio_utils import ResampleUnavailable, resample_to_rate


def _clip(rate, seconds=0.25, value=0.5):
    n = max(1, int(rate * seconds))
    return {"waveform": torch.full((1, 1, n), float(value)), "sample_rate": int(rate)}


# ---------------------------------------------------------------------------
# 1. The batch survives a mixed rate
# ---------------------------------------------------------------------------
def test_mixed_rates_are_RESAMPLED_rather_than_refused():
    """The defect, inverted."""
    packed = pack_audio_batch(
        [_clip(24000), _clip(22050), _clip(24000)], sample_rate=24000, mono=True)

    assert int(packed["sample_rate"]) == 24000
    assert int(packed["waveform"].shape[0]) == 3, (
        "a clip was dropped; losing a line of recorded dialogue is a WORSE "
        "render than converting it")


def test_the_resampled_clip_KEEPS_ITS_DURATION():
    """Content-preserving is what licenses this. A 0.25s line stays 0.25s."""
    packed = pack_audio_batch([_clip(22050, seconds=0.25)],
                              sample_rate=24000, mono=True)
    got = int(packed["waveform"].shape[-1])
    expected = int(24000 * 0.25)
    assert abs(got - expected) <= 64, (
        "resampled length %d is not ~%d samples at 24 kHz" % (got, expected))


def test_a_DOWNSAMPLE_keeps_its_duration_too():
    """24k -> 22.05k is the real engine pairing from the live incident, and the
    direction where a non-anti-aliased fallback would show."""
    packed = pack_audio_batch([_clip(24000, seconds=0.25)],
                              sample_rate=22050, mono=True)
    got = int(packed["waveform"].shape[-1])
    expected = int(22050 * 0.25)
    assert abs(got - expected) <= 64


def test_a_mixed_rate_is_LOGGED_so_a_broken_engine_stays_visible(caplog):
    """Degrading must not hide an engine that keeps returning the wrong rate."""
    caplog.set_level(logging.WARNING)
    pack_audio_batch([_clip(24000), _clip(22050)], sample_rate=24000, mono=True)

    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "22050" in blob and "24000" in blob, (
        "both rates must appear so the source engine can be identified: %r" % blob)


def test_matching_rates_take_no_resample_path_at_all():
    """The common case must be untouched."""
    a, b = _clip(24000), _clip(24000)
    packed = pack_audio_batch([a, b], sample_rate=24000, mono=True)
    assert int(packed["waveform"].shape[0]) == 2
    assert torch.allclose(packed["waveform"][0], a["waveform"][0])


def test_an_inconsistent_CHANNEL_count_still_raises():
    """Scope check. Channels are not rates, and a degrade here would be guessing
    at what the caller meant."""
    mono = {"waveform": torch.zeros((1, 1, 100)), "sample_rate": 24000}
    stereo = {"waveform": torch.zeros((1, 2, 100)), "sample_rate": 24000}
    with pytest.raises(ValueError, match="channel count"):
        pack_audio_batch([mono, stereo], sample_rate=24000, mono=False)


def test_an_UNCONVERTIBLE_clip_refuses_rather_than_shipping_a_wrong_label(
        monkeypatch):
    """THE REVIEW'S MUST-FIX, pinned.

    Everything after the resample labels the batch with the declared rate, so
    unconverted samples under that label are a silently wrong render. When
    conversion cannot happen the ValueError comes back -- and that is not a
    retreat from "only an OOM should fail", it is the carve-out that outranks
    it."""
    import nodes._otr_audio_engines.base as base

    def cannot(_w, _src, _dst):
        raise ResampleUnavailable("no resampler here")

    monkeypatch.setattr(base, "resample_to_rate", cannot)
    with pytest.raises(ValueError, match="could not be converted"):
        pack_audio_batch([_clip(24000), _clip(22050)],
                         sample_rate=24000, mono=True)


# ---------------------------------------------------------------------------
# 2. The resampler itself
# ---------------------------------------------------------------------------
def test_resample_to_rate_is_a_no_op_at_the_same_rate():
    w = torch.rand((1, 1, 500))
    assert resample_to_rate(w, 24000, 24000) is w


@pytest.mark.parametrize("src,dst", [(0, 24000), (24000, 0), (-1, 24000)])
def test_resample_to_rate_REFUSES_a_nonsense_rate(src, dst):
    """It raises rather than returning the input. Returning it would hand the
    caller unconverted audio to label with the destination rate."""
    with pytest.raises(ResampleUnavailable):
        resample_to_rate(torch.rand((1, 1, 100)), src, dst)


def test_resample_to_rate_REPORTS_failure_instead_of_disguising_it():
    """Never silently returns the input."""
    class Hostile:
        def detach(self):
            raise RuntimeError("no tensor here")

    with pytest.raises(ResampleUnavailable):
        resample_to_rate(Hostile(), 22050, 24000)


def test_a_MEMORY_ERROR_is_never_swallowed():
    """The operator's rule cuts both ways: an OOM must stay loud and fatal.

    `except Exception` catches MemoryError, and the first cut of this helper did
    exactly that -- turning resource death into a quietly degraded render."""
    class Starved:
        def detach(self):
            raise MemoryError("out of memory")

    with pytest.raises(MemoryError):
        resample_to_rate(Starved(), 22050, 24000)


def test_a_KEYBOARD_INTERRUPT_is_never_swallowed():
    """An operator pressing Ctrl-C means stop, not degrade."""
    class Interrupted:
        def detach(self):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        resample_to_rate(Interrupted(), 22050, 24000)
