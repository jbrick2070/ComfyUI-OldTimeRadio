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

import os
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
    # Voice-reference policy (model-agnostic dispatch -- replaces the old
    # _OTR_CLONE_ENGINES name tuple). A voice-CLONING engine sets
    # requires_voice_ref=True so the dispatch resolves a per-character reference
    # WAV for it and, when none is available for a char_voice line, FAILS LOUD
    # (no-fallback rip 2026-07-03 -- no bark fallback). voice_ref_kind documents
    # what the reference is ("wav_path" for clip-cloning engines).
    # missing_ref_fallback is retired: it stays None on every engine (a missing
    # ref raises). Non-clone engines keep these defaults and are never sent down
    # the ref-resolution path.
    requires_voice_ref: bool = False
    voice_ref_kind: Optional[str] = None
    missing_ref_fallback: Optional[str] = None
    # G1: bit_exact mode requires every forward to bind an external
    # torch.Generator. An adapter flips this True only once the F dependency
    # pilot has verified its forward accepts ``generator=``; until then it is
    # disqualified from bit_exact (never silently downgraded).
    supports_external_generator: bool = False

    def load(self) -> None:  # pragma: no cover - trivial no-op default
        """Bring the model into residency. Cheap when already loaded."""

    def unload(self) -> None:  # pragma: no cover - trivial no-op default
        """Release residency. Cheap when already unloaded."""

    def identity_params(self, *, resolved_voice_ref=None, resolved_model=None, **_kw) -> dict:
        """Call-time identity fields that must live in the cache key (I-6).

        Default is empty; adapters whose forward has externally-resolved
        identity (e.g. env-selected model) override to return keys folded
        into ResolvedVoiceRequest's provider_model_id / provider_voice_id
        by the per-line wiring in _otr_voice_node_common._render_per_line.
        """
        return {}

    def render_time_params(self) -> dict:
        """Call-time NUMERIC params that must live in the cache key.

        The numeric sibling of :meth:`identity_params`, and it exists because
        that one only had a home for STRING identity. An adapter that resolves a
        knob at GENERATE time -- typically from the environment, so a
        long-running server picks the change up -- has a value that changes the
        audio and is not in ``profile.default_params``, which the request
        captured at BUILD time. The render moves and the cache key does not, so
        the next identical request replays audio made under the old value.

        Returned values are merged into the request's ``params`` and quantized
        into ``quantized_params`` (3 decimal places), so they key like any other
        numeric param. Default EMPTY, which is what preserves every engine that
        has no such knob byte-for-byte -- including its ``IS_CHANGED`` answer.

        The engine must resolve the value through the SAME function the forward
        calls, so the key and the render can never disagree about it.
        """
        return {}


def resolve_voice_ref_path(ref):
    """THE ONE resolver for a voice-bank ``ref_path`` (Lemmy chunk B).

    Bank refs are stored relative to the ComfyUI root
    (``models/TTS/refs/indextts2/x.wav``) and have to become an absolute path an
    ISOLATED WORKER can open regardless of its own cwd.

    WHY THIS IS SHARED NOW. Three cloning adapters (indextts2, chatterbox, dia)
    each carried a private copy that tried exactly ONE candidate --
    ``<comfy_base>/models/<ref>`` -- and fell back to ``os.path.abspath(ref)``,
    which is a cwd-relative path that generally does not exist. The voice node's
    own ``_resolve_ref_to_disk`` meanwhile knew about three MORE places, notably
    the ``C:\\ComfyUI-Models`` root the Comfy Desktop 1.0.4 model-path migration
    introduced. So on a box whose refs live under the migrated root, the NODE's
    existence check found the file and the ADAPTER's resolver did not: preflight
    passed and the worker then failed to open a reference the check had just
    confirmed. Two resolvers over one fact, disagreeing exactly where it hurts.

    ORDER IS THE CONTRACT: the first candidate that EXISTS wins, so a box with
    the file in the historical location resolves exactly as it always did and
    the extra candidates can only turn a miss into a hit. When nothing exists it
    returns the absolute-path fallback -- a path the caller will fail LOUDLY on,
    which is what the adapters already did and is better than a bare ``None``
    the worker would report as an empty filename.

    Passthrough for empty and for an already-absolute path, matching every
    caller's existing expectation.
    """
    if not ref or os.path.isabs(ref):
        return ref
    rp = str(ref).replace("\\", "/")
    stripped = rp[len("models/"):] if rp.startswith("models/") else rp
    candidates = []
    try:
        import folder_paths                     # ComfyUI runtime only

        models_dir = folder_paths.models_dir
        # <base>/models/... -- the historical single candidate, FIRST so no box
        # that resolved before can resolve differently now.
        candidates.append(os.path.join(os.path.dirname(models_dir), ref))
        candidates.append(os.path.join(models_dir, stripped))
    except Exception:  # noqa: BLE001 -- non-Comfy contexts (tests / CLI)
        pass
    # The extra models root after the Comfy Desktop 1.0.4 model-path migration.
    candidates.append(os.path.join("C:\\ComfyUI-Models", stripped))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return os.path.abspath(ref)


def engine_supports_external_generator(engine) -> bool:
    """True iff ``engine`` binds an external ``torch.Generator`` per forward.

    Read via ``getattr`` so duck-typed adapters that never set the flag read as
    ``False`` (correctly disqualified from bit_exact mode until F verifies them).
    """
    return bool(getattr(engine, "supports_external_generator", False))


def supported_kwargs(fn, **kwargs) -> dict:
    """Subset of ``kwargs`` whose names appear in ``fn``'s signature.

    Blind-call guard for engine forwards implemented to the documented
    assumed_call: a kwarg the real library does not accept is dropped instead of
    passed, so a name the F dependency pilot later corrects cannot crash the
    forward. If the signature cannot be introspected (C builtins) or it accepts
    ``**kwargs``, everything passes through unchanged.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


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
