"""Device resolver for the upscale namespace.

Turns a profile-level device token (``"cuda"``, ``"cuda:N"``, ``"cpu"``) into a
concrete ``torch.device``, or raises :class:`EngineUnusable` with the classified
reason. Fail-loud on every unavailable / malformed selection per the D-2 codicil
("fail-loud on unsupported hardware") and the portability brief rule 6 ("NO
FALLBACKS ethos holds").

MPS is NOT accepted by the first ship (see registry.CAPABILITIES); it will land
when a Mac user provides an integration receipt.

Torch is imported lazily inside the resolver so this module stays cold-import-clean
(V-12 invariant) and can be exercised from unit tests without pulling torch.
"""
from __future__ import annotations

from .._otr_shared.engine_registry_base import EngineUnusable, EngineUsabilityReason


def resolve_device(profile_value: str):
    """Resolve ``profile_value`` to a ``torch.device``. Raises
    :class:`EngineUnusable` on any invalid selection.

    Accepts: ``"cpu"``, ``"cuda"`` (= ``cuda:0``), ``"cuda:N"``.
    Rejects: ``"mps"`` (until a Mac integration receipt lands), any other token.
    """
    import torch

    v = (profile_value or "cpu").strip().lower()
    if v == "cpu":
        return torch.device("cpu")
    if v == "cuda":
        if not torch.cuda.is_available():
            raise EngineUnusable(
                "<upscale>", "upscale_stage",
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "cuda requested but torch.cuda.is_available() is False",
                kind="upscale")
        return torch.device("cuda:0")
    if v.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise EngineUnusable(
                "<upscale>", "upscale_stage",
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "cuda requested but torch.cuda.is_available() is False",
                kind="upscale")
        try:
            idx = int(v.split(":", 1)[1])
        except ValueError:
            raise EngineUnusable(
                "<upscale>", "upscale_stage",
                EngineUsabilityReason.MALFORMED_CONFIG,
                f"cuda:N token {profile_value!r} has non-integer index",
                kind="upscale")
        if idx < 0 or idx >= torch.cuda.device_count():
            raise EngineUnusable(
                "<upscale>", "upscale_stage",
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                f"cuda:{idx} requested but only {torch.cuda.device_count()} "
                f"device(s) present",
                kind="upscale")
        return torch.device(f"cuda:{idx}")
    raise EngineUnusable(
        "<upscale>", "upscale_stage",
        EngineUsabilityReason.MALFORMED_CONFIG,
        f"unknown device token {profile_value!r}",
        kind="upscale")
