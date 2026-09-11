"""Shared audio helpers for the v2 audio lane.

ComfyUI AUDIO is ``{"waveform": tensor[B, C, T], "sample_rate": int}``. Never
assume ``waveform.shape[0] == 2`` -- dim 0 is batch, dim 1 is channel. Every
engine output passes through ``canonical_audio`` then ``mono_safe`` before it
reaches SceneSequencer / EpisodeAssembler, so the mono assembly chain stays
untouched and stereo never leaks downstream.
"""
from __future__ import annotations

import logging

import torch

_DEFAULT_SR = 24000


#: This module had no logger before 2026-09-11; `resample_to_rate` needs one to
#: report taking the non-anti-aliased fallback path.
log = logging.getLogger("OTR")


def _split(audio):
    if isinstance(audio, dict):
        return audio.get("waveform"), int(audio.get("sample_rate", _DEFAULT_SR))
    return audio, _DEFAULT_SR


def canonical_audio(audio) -> dict:
    """Return ``{"waveform": tensor[B, C, T], "sample_rate": int}``.

    Promotes ``[T]`` -> ``[1, 1, T]`` and ``[C, T]`` -> ``[1, C, T]``. Accepts
    a raw tensor / array or an AUDIO dict.
    """
    wf, sr = _split(audio)
    if wf is None:
        raise ValueError("canonical_audio: missing waveform")
    if not torch.is_tensor(wf):
        wf = torch.as_tensor(wf, dtype=torch.float32)
    if wf.dim() == 1:
        wf = wf.unsqueeze(0).unsqueeze(0)
    elif wf.dim() == 2:
        wf = wf.unsqueeze(0)
    elif wf.dim() != 3:
        raise ValueError(
            f"canonical_audio: expected 1/2/3-D waveform, got {wf.dim()}-D"
        )
    return {"waveform": wf, "sample_rate": int(sr)}


class ResampleUnavailable(RuntimeError):
    """Audio could not be converted to the requested rate.

    Raised rather than returning the input, because the caller's next act is to
    LABEL the result with the destination rate. Handing back unconverted samples
    under that label is a silently wrong render -- a 22,050 Hz line labelled
    24,000 Hz plays 8.8% fast and pitch-shifted, and every downstream consumer
    (the slicer, the mux, the duration gates) reads the label, not the samples.
    Caught by the 2026-09-11 finished-diff review; the first cut of this helper
    returned the original waveform on failure and did exactly that.
    """


def resample_to_rate(waveform, src_rate: int, dst_rate: int):
    """Resample a ``[1, C, T]`` float tensor to ``dst_rate``.

    CPU ONLY, and that is invariant I-11 rather than a convenience: post-engine
    audio DSP must not touch CUDA or the deterministic audio baseline moves
    under the non-strict determinism default. `scene_sequencer._resample_audio`
    is the same chain for the same reason on the mixing side; this is the
    packing side, and the two are deliberately not shared -- a resampler is
    three lines of policy, and the policy is what differs between lanes.

    Path selection, highest quality first:
      * scipy.signal.resample_poly -- anti-aliased polyphase. scipy is NOT a
        declared dependency of this pack, so it is tried and not required.
      * numpy linear interpolation -- always available, audibly fine for the
        small ratios this sees (a 22.05k engine clip into a 24k batch).

    IT CONVERTS OR IT RAISES `ResampleUnavailable`. It never returns the input
    unchanged for a rate it was asked to change, because the caller then labels
    the result with `dst_rate` and unconverted samples under that label are a
    silently wrong render rather than a rough one.

    A `MemoryError`, `KeyboardInterrupt` or `SystemExit` propagates untouched.
    Resource death stays loud and fatal by operator directive; only an inability
    to CONVERT is reported as this helper's own failure.

    SUPPORTED CONVERSIONS: any ratio, up or down. The polyphase path is
    anti-aliased in both directions. The numpy fallback is LINEAR and therefore
    not anti-aliased, which matters on downsampling (24k -> 22.05k is a real
    engine pairing here, not a hypothetical), so taking it is logged rather than
    left as an unstated assumption.
    """
    import torch

    src = int(src_rate)
    dst = int(dst_rate)
    if src <= 0 or dst <= 0:
        raise ResampleUnavailable(
            "cannot resample %r Hz -> %r Hz: both rates must be positive"
            % (src_rate, dst_rate))
    if src == dst:
        return waveform
    try:
        import math

        import numpy as np

        flat = waveform.detach().to("cpu").float().numpy()
        shape = flat.shape
        chans = flat.reshape(-1, shape[-1])
        g = math.gcd(dst, src)
        up, down = dst // g, src // g
        try:
            from scipy.signal import resample_poly

            out = np.stack([
                resample_poly(row, up, down).astype(np.float32) for row in chans
            ])
        except ImportError:
            log.warning(
                "[resample_to_rate] scipy is unavailable; falling back to LINEAR "
                "interpolation for %d -> %d Hz. It is not anti-aliased, so a "
                "downsample can alias. Install scipy for the polyphase path.",
                src, dst,
            )
            new_len = max(1, int(round(chans.shape[-1] * dst / src)))
            src_x = np.arange(chans.shape[-1])
            dst_x = np.linspace(0, chans.shape[-1] - 1, new_len)
            out = np.stack([
                np.interp(dst_x, src_x, row).astype(np.float32) for row in chans
            ])
        return torch.from_numpy(
            out.reshape(shape[:-1] + (out.shape[-1],))).float()
    except (MemoryError, KeyboardInterrupt, SystemExit):
        # RESOURCE DEATH AND OPERATOR INTERRUPT STAY FATAL. `except Exception`
        # catches MemoryError, and the first cut of this helper swallowed it into
        # a quietly degraded render -- the exact inversion the operator ruled
        # against ("only an out of memory should fail" cuts both ways).
        raise
    except Exception as exc:   # noqa: BLE001 -- reported, never disguised
        raise ResampleUnavailable(
            "could not resample %d Hz -> %d Hz: %s" % (src, dst, exc)) from exc


def mono_safe(audio) -> dict:
    """Downmix to a single channel (mean across channels) when needed.

    Leaves an already-mono buffer untouched, so the byte-path stays identical
    to the mono engines.
    """
    a = canonical_audio(audio)
    wf = a["waveform"]
    if wf.shape[1] > 1:
        wf = wf.mean(dim=1, keepdim=True)
    return {"waveform": wf, "sample_rate": a["sample_rate"]}
