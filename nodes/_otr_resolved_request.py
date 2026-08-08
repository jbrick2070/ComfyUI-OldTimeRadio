"""Frozen ``ResolvedVoiceRequest`` identity contract + seed reduction + the
ComfyUI AUDIO-batch contract asserts (R0a shell; Wave 0 adds the builder and the
integer-tick quantizer on top of this surface).

Why this exists (plan invariants I-6, C-4):
  * **One frozen request == one cache key == one engine input.** The engine never
    sees a raw widget float -- it sees this object, whose ``cache_key`` is a
    sha256 over the IN_KEY fields only. IGNORED fields (debug text, the derived
    engine seed, the release-gate boolean) never perturb identity.
  * **One seed reduction.** ``_seed_to_int64`` is the single sha256->seed helper
    for every deterministic draw in the audio path, so legacy seeding, the
    per-line stable seed, and the per-engine generator all reduce identically.
  * **AUDIO-batch contract.** ComfyUI AUDIO is ``{"waveform": tensor[B, C, T],
    "sample_rate": int}``; the canonical empty batch is ``[1, 1, 0]``. The assert
    guards SceneSequencer + EpisodeAssembler ingest (C-4) with no socket retype.

Import-time is side-effect-free (no IO, no network, no CUDA) per C-5.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields as _dc_fields
from typing import Optional, Tuple

import torch

REQUEST_SCHEMA_VERSION = "2"
_DEFAULT_SR = 24000
_SEP = b"\x1f"  # ASCII unit separator -> unambiguous part join


# ---------------------------------------------------------------------------
# Seed reduction -- the single sha256 -> int helper for the whole audio path
# ---------------------------------------------------------------------------

def _canon_part(part) -> bytes:
    """Canonical bytes for one seed part. Stable across runs and platforms.

    Each part is type-tagged (``i:`` int, ``s:`` str, ``b:`` bool, ``y:`` bytes,
    ``j:`` json) so cross-type values can never collide -- ``1``, ``"1"`` and
    ``True`` reduce to three distinct seeds.
    """
    if isinstance(part, bytes):
        return b"y:" + part
    if isinstance(part, bool):  # before int -- bool is an int subclass
        return b"b:1" if part else b"b:0"
    if isinstance(part, int):
        return b"i:" + str(int(part)).encode("utf-8")
    if isinstance(part, str):
        return b"s:" + part.encode("utf-8")
    return b"j:" + json.dumps(
        part, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _seed_to_int64(*parts) -> int:
    """Reduce any mix of ``str | int | bytes | json-able`` parts to a stable,
    non-negative 63-bit int (safe positive int64).

    The 63-bit value is accepted directly by ``torch.Generator.manual_seed`` and
    ``random.seed``. NumPy callers must mask to 32 bits: ``seed & 0xFFFFFFFF``.
    Order matters -- ``_seed_to_int64(a, b) != _seed_to_int64(b, a)``.
    """
    joined = _SEP.join(_canon_part(p) for p in parts)
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


# ---------------------------------------------------------------------------
# ResolvedVoiceRequest v1 -- frozen identity contract
# ---------------------------------------------------------------------------

# Fields that ARE identity (feed cache_key).
IN_KEY_FIELDS: Tuple[str, ...] = (
    "request_schema_version", "role", "engine_profile_id", "engine_name",
    "engine_impl_version", "provider_model_id", "provider_voice_id",
    "char_id", "line_id", "occurrence",
    "prepared_text_sha256", "voice_ref_id", "voice_preset",
    "delivery_profile_id", "delivery_profile_version", "episode_seed",
    "cast_lock_revision", "stable_line_seed", "sample_rate", "channels",
    "quantized_params", "source_ref_sha256", "commercial_clean",
)
# Fields that are NOT identity: debug text, the derived engine seed, and the
# release gate (a gate is not identity -- I-8).
IGNORED_FIELDS: Tuple[str, ...] = (
    "prepared_text", "engine_seed", "allowed_for_release",
)


@dataclass(frozen=True)
class ResolvedVoiceRequest:
    """Immutable, model-agnostic engine input. ``cache_key`` hashes IN_KEY only.

    ``delivery_profile_id`` + ``delivery_profile_version`` stay in the key as
    identity even while only ``neutral`` ships (dropping them would force a
    future re-baseline).
    """
    # --- IN_KEY (identity) ---
    role: str = ""
    engine_profile_id: str = ""
    engine_name: str = ""
    engine_impl_version: str = ""
    provider_model_id: str = ""
    provider_voice_id: str = ""
    char_id: str = ""
    line_id: str = ""
    occurrence: int = 0
    prepared_text_sha256: str = ""
    voice_ref_id: Optional[str] = None
    voice_preset: Optional[str] = None
    delivery_profile_id: str = "neutral"
    delivery_profile_version: str = "1"
    episode_seed: int = 0
    cast_lock_revision: int = 0
    stable_line_seed: int = 0
    sample_rate: int = _DEFAULT_SR
    channels: int = 1
    quantized_params: Tuple[Tuple[str, int], ...] = ()
    source_ref_sha256: str = ""
    commercial_clean: Optional[bool] = None
    request_schema_version: str = REQUEST_SCHEMA_VERSION
    # --- IGNORED (debug / derived / gate -- never identity) ---
    prepared_text: str = ""
    engine_seed: int = 0
    allowed_for_release: bool = False

    def in_key_dict(self) -> dict:
        """The identity sub-dict (IN_KEY fields only)."""
        return {name: getattr(self, name) for name in IN_KEY_FIELDS}

    def canonical_json(self) -> str:
        """Deterministic JSON over IN_KEY. Sorted keys, tight separators."""
        return json.dumps(
            self.in_key_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def cache_key(self) -> str:
        """sha256 hex of ``canonical_json`` -- the cache + identity key (I-6)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _field_partition_is_complete() -> bool:
    """True iff IN_KEY ∪ IGNORED exactly partitions the dataclass fields.

    Exposed as a function (not an import-time assert) so a box-fresh import can
    never crash the pack (C-5). The R0a contract test calls it.
    """
    declared = {f.name for f in _dc_fields(ResolvedVoiceRequest)}
    tagged = set(IN_KEY_FIELDS) | set(IGNORED_FIELDS)
    overlap = set(IN_KEY_FIELDS) & set(IGNORED_FIELDS)
    return declared == tagged and not overlap


# ---------------------------------------------------------------------------
# AUDIO-batch contract (C-4) -- transcribe the existing Bark output shape
# ---------------------------------------------------------------------------

def empty_audio_batch(sample_rate: int = _DEFAULT_SR) -> dict:
    """The canonical empty AUDIO batch: ``waveform`` ``[1, 1, 0]``."""
    return {
        "waveform": torch.zeros(1, 1, 0, dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }


def assert_audio_batch_contract(audio, *, where: str = "") -> dict:
    """Assert the ComfyUI AUDIO-batch contract and return ``audio`` unchanged.

    Contract: a dict ``{"waveform": torch.Tensor[B, C, T], "sample_rate": int}``;
    the canonical empty batch is ``[1, 1, 0]``. Designed to wrap a consumer
    ingest: ``wf = assert_audio_batch_contract(x, where="SceneSequencer")``.
    """
    ctx = f" [{where}]" if where else ""
    if not isinstance(audio, dict):
        raise TypeError(
            f"AUDIO contract{ctx}: expected dict, got {type(audio).__name__}"
        )
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(
            f"AUDIO contract{ctx}: missing 'waveform'/'sample_rate' keys"
        )
    wf = audio["waveform"]
    if not torch.is_tensor(wf):
        raise TypeError(
            f"AUDIO contract{ctx}: waveform must be a torch.Tensor, "
            f"got {type(wf).__name__}"
        )
    if wf.dim() != 3:
        raise ValueError(
            f"AUDIO contract{ctx}: waveform must be 3-D [B, C, T], "
            f"got {wf.dim()}-D {tuple(wf.shape)}"
        )
    sr = audio["sample_rate"]
    if isinstance(sr, bool) or not isinstance(sr, int) or sr <= 0:
        raise ValueError(
            f"AUDIO contract{ctx}: sample_rate must be a positive int, got {sr!r}"
        )
    return audio


# ---------------------------------------------------------------------------
# Wave 0: integer-tick quantizer + request builder (the engine never keys on a
# raw widget float -- I-6)
# ---------------------------------------------------------------------------

DEFAULT_PARAM_QUANTUM = 1000  # 3 decimal places of tick resolution


def quantize_params(params, *, quantum: int = DEFAULT_PARAM_QUANTUM) -> Tuple[Tuple[str, int], ...]:
    """Map a ``{name: value}`` param dict to a sorted tuple of ``(name, int_tick)``.

    Floats become integer ticks (``round(value * quantum)``) so float-repr noise
    can never perturb the cache key; bools become 0/1; any non-numeric value is
    reduced to a stable 31-bit int tick. Deterministic and order-independent
    (sorted by name).
    """
    out = []
    for name in sorted((params or {}).keys()):
        v = params[name]
        if isinstance(v, bool):
            tick = 1 if v else 0
        elif isinstance(v, (int, float)):
            tick = int(round(float(v) * quantum))
        else:
            tick = _seed_to_int64("param", str(name), str(v)) % (1 << 31)
        out.append((str(name), tick))
    return tuple(out)


def build_resolved_request(
    *,
    role: str,
    engine_name: str,
    engine_profile_id: str,
    engine_impl_version: str,
    char_id: str,
    line_id: str,
    occurrence: int = 0,
    prepared_text: str = "",
    voice_ref_id: Optional[str] = None,
    voice_preset: Optional[str] = None,
    delivery_profile_id: str = "neutral",
    delivery_profile_version: str = "1",
    episode_seed: int = 0,
    cast_lock_revision: int = 0,
    sample_rate: int = _DEFAULT_SR,
    channels: int = 1,
    params=None,
    quantum: int = DEFAULT_PARAM_QUANTUM,
    source_ref_sha256: str = "",
    commercial_clean: Optional[bool] = None,
    provider_model_id: str = "",
    provider_voice_id: str = "",
) -> ResolvedVoiceRequest:
    """Pack already-resolved pieces into the frozen ``ResolvedVoiceRequest``.

    Computes the derived identity fields: ``prepared_text_sha256``, the quantized
    params tuple, and ``stable_line_seed`` (``stable_line_seed_v1`` reduction of
    episode_seed + cast_lock_revision + line_id + char_id + prepared_text_sha256 +
    engine_profile_id + voice_ref_id + delivery_profile_version, per G1). The raw
    ``prepared_text`` is carried only as the IGNORED debug field.
    """
    prepared_text = prepared_text or ""
    prepared_text_sha256 = hashlib.sha256(prepared_text.encode("utf-8")).hexdigest()
    quantized = quantize_params(params, quantum=quantum)
    stable_line_seed = _seed_to_int64(
        "stable_line_seed_v1", int(episode_seed), int(cast_lock_revision),
        line_id, char_id, prepared_text_sha256, engine_profile_id,
        voice_ref_id or "", delivery_profile_version,
    )
    return ResolvedVoiceRequest(
        role=role,
        engine_name=engine_name,
        engine_profile_id=engine_profile_id,
        engine_impl_version=engine_impl_version,
        provider_model_id=str(provider_model_id or ""),
        provider_voice_id=str(provider_voice_id or ""),
        char_id=char_id,
        line_id=line_id,
        occurrence=int(occurrence),
        prepared_text_sha256=prepared_text_sha256,
        voice_ref_id=voice_ref_id,
        voice_preset=voice_preset,
        delivery_profile_id=delivery_profile_id,
        delivery_profile_version=delivery_profile_version,
        episode_seed=int(episode_seed),
        cast_lock_revision=int(cast_lock_revision),
        stable_line_seed=stable_line_seed,
        sample_rate=int(sample_rate),
        channels=int(channels),
        quantized_params=quantized,
        source_ref_sha256=source_ref_sha256,
        commercial_clean=commercial_clean,
        prepared_text=prepared_text,
    )
