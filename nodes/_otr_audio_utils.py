"""Shared audio helpers for the v2 audio lane.

ComfyUI AUDIO is ``{"waveform": tensor[B, C, T], "sample_rate": int}``. Never
assume ``waveform.shape[0] == 2`` -- dim 0 is batch, dim 1 is channel. Every
engine output passes through ``canonical_audio`` then ``mono_safe`` before it
reaches SceneSequencer / EpisodeAssembler, so the mono assembly chain stays
untouched and stereo never leaks downstream.
"""
from __future__ import annotations

import hashlib

import torch

_DEFAULT_SR = 24000


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


def resample_audio(audio, target_sr) -> dict:
    """Resample an AUDIO buffer to ``target_sr`` Hz. Deterministic, CPU-only.

    Uses ``scipy.signal.resample_poly`` -- the same anti-aliased CPU resampler
    SceneSequencer already uses (``scene_sequencer._resample_audio``), per
    invariant I-11 which removed the GPU torchaudio path so post-engine
    resampling stays deterministic. No-op when the buffer is already at
    ``target_sr``. Returns a canonical AUDIO dict
    ``{"waveform": tensor[B, C, T'], "sample_rate": target_sr}`` in float32.
    scipy is lazy-imported so module import stays side-effect-free (C-5).
    """
    a = canonical_audio(audio)
    src_sr = int(a["sample_rate"])
    target_sr = int(target_sr)
    if src_sr == target_sr:
        return a
    if src_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"resample_audio: rates must be positive "
            f"(src={src_sr}, target={target_sr})"
        )
    wf = a["waveform"].detach().to("cpu").contiguous().to(torch.float32)
    if wf.shape[-1] == 0:
        # Empty buffer: nothing to resample; just restamp the rate.
        return {"waveform": wf, "sample_rate": target_sr}
    from math import gcd
    from scipy.signal import resample_poly  # lazy import keeps C-5 (no import-time deps)

    g = gcd(src_sr, target_sr)
    up, down = target_sr // g, src_sr // g
    res = resample_poly(wf.numpy(), up, down, axis=-1)  # [B, C, T']
    out = torch.from_numpy(res.astype("float32")).contiguous()
    return {"waveform": out, "sample_rate": target_sr}


def audio_sha16(audio) -> str:
    """Deterministic 16-hex digest of an audio buffer (for render logs)."""
    a = canonical_audio(audio)
    arr = a["waveform"].detach().to("cpu").contiguous().to(torch.float32).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
