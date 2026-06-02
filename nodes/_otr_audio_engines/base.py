"""Audio-engine adapter base + the AUDIO-batch packer (plan piece 4).

Two things live here:

* :class:`AudioEngineAdapter` -- an OPTIONAL common base for engine adapters.
  The registry duck-types against the :class:`~.registry.AudioEngine` Protocol,
  so an adapter does not have to inherit from this; the existing legacy adapters
  do not. New adapters may subclass it to pick up the attribute defaults, the
  cheap ``load`` / ``unload`` no-ops, and the ``supports_external_generator``
  flag. Heavy model residency belongs in ``load`` / ``unload``, never in
  ``__init__`` (C-5: importing the package must never pull weights).

* :func:`pack_audio_batch` -- packs a sequence of per-line engine outputs into
  the EXISTING Bark AUDIO-batch contract ``{"waveform": tensor[B, C, T],
  "sample_rate": int}`` (C-4). The generic voice/music node generates one
  waveform per line, then calls this to emit a single batch the way the legacy
  Bark node does; SceneSequencer unbinds dim 0 (batch) back into per-line clips.
  No socket retype -- same dict shape the mono assembly chain already consumes.

Import-time is side-effect-free (no IO, no network, no CUDA) per C-5.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch

from .._otr_audio_utils import canonical_audio, mono_safe
from .._otr_resolved_request import assert_audio_batch_contract, empty_audio_batch

_DEFAULT_SR = 24000


class AudioEngineAdapter:
    """Optional base for audio-engine adapters (see module docstring).

    ``interface`` is one of ``"batch"`` (delegate the whole ledger to an
    existing legacy node, byte-identical), ``"per_line"`` (call
    ``generate_voice`` per dialogue line) or ``"clip"`` (call ``generate_clip``
    per music cue).
    """

    name: str = ""
    roles: tuple = ()
    default_roles: tuple = ()
    commercial_clean: Optional[bool] = None
    requires_flag: Optional[str] = None
    interface: str = "per_line"
    sample_rate: int = _DEFAULT_SR
    # G1: bit_exact mode requires every forward to bind an external
    # torch.Generator. An adapter flips this True only once the F dependency
    # pilot has verified its forward accepts ``generator=``; until then it is
    # disqualified from bit_exact (never silently downgraded).
    supports_external_generator: bool = False

    def load(self) -> None:  # pragma: no cover - trivial no-op default
        """Bring the model into residency. Cheap when already loaded."""

    def unload(self) -> None:  # pragma: no cover - trivial no-op default
        """Release residency. Cheap when already unloaded."""


def engine_supports_external_generator(engine) -> bool:
    """True iff ``engine`` binds an external ``torch.Generator`` per forward.

    Read via ``getattr`` so duck-typed adapters that never set the flag read as
    ``False`` (correctly disqualified from bit_exact mode until F verifies them).
    """
    return bool(getattr(engine, "supports_external_generator", False))


def pack_audio_batch(
    items: Iterable,
    *,
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> dict:
    """Pack per-line engine outputs into the Bark AUDIO-batch contract.

    ``items`` is a sequence of AUDIO dicts (``{"waveform", "sample_rate"}``) or
    raw tensors -- one per line. Each is canonicalized to ``[1, C, T]`` (and
    downmixed to mono when ``mono``), right-padded with zeros to the longest
    ``T`` in the batch, and stacked along the batch dim -> ``[B, C, max_T]``.
    An empty sequence returns the canonical empty batch ``[1, 1, 0]`` (C-4).

    All items must share one sample rate (raises otherwise); pass
    ``sample_rate`` to assert a specific rate. The result satisfies
    ``assert_audio_batch_contract``.
    """
    items = list(items or [])
    if not items:
        sr = int(sample_rate) if sample_rate else _DEFAULT_SR
        return empty_audio_batch(sr)

    waveforms = []
    rates = []
    for it in items:
        a = mono_safe(it) if mono else canonical_audio(it)
        waveforms.append(a["waveform"])  # [1, C, T]
        rates.append(int(a["sample_rate"]))

    sr = int(sample_rate) if sample_rate else rates[0]
    mismatched = {r for r in rates if r != sr}
    if mismatched:
        raise ValueError(
            f"pack_audio_batch: mixed sample rates {sorted(set(rates))}; "
            f"resample to one rate before packing"
        )

    channels = waveforms[0].shape[1]
    for w in waveforms:
        if w.shape[1] != channels:
            raise ValueError(
                f"pack_audio_batch: inconsistent channel count "
                f"{[int(w.shape[1]) for w in waveforms]}"
            )

    max_t = max(int(w.shape[-1]) for w in waveforms)
    out = torch.zeros(len(waveforms), channels, max_t, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        t = int(w.shape[-1])
        if t:
            out[i, :, :t] = w[0, :, :t].to(torch.float32)
    return assert_audio_batch_contract(
        {"waveform": out, "sample_rate": sr}, where="pack_audio_batch"
    )
