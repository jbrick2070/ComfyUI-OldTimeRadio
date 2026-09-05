"""Shared audio helpers for the v2 audio lane.

ComfyUI AUDIO is ``{"waveform": tensor[B, C, T], "sample_rate": int}``. Never
assume ``waveform.shape[0] == 2`` -- dim 0 is batch, dim 1 is channel. Every
engine output passes through ``canonical_audio`` then ``mono_safe`` before it
reaches SceneSequencer / EpisodeAssembler, so the mono assembly chain stays
untouched and stereo never leaks downstream.
"""
from __future__ import annotations


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
