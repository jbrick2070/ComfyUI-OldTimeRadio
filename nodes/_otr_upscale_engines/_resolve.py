"""Device resolver for the upscale namespace.

Turns a profile-level device token (``"cuda"``, ``"cuda:N"``, ``"cpu"``) into a
concrete ``torch.device``, or raises :class:`EngineUnusable` with the classified
reason. Fail-loud on every unavailable / malformed selection per the D-2 codicil
("fail-loud on unsupported hardware") and the portability brief rule 6 ("NO
FALLBACKS ethos holds").

MPS IS ACCEPTED SINCE 2026-09-09, and the condition this docstring set is the
reason it took until then: "it will land when a Mac user provides an integration
receipt." The receipt exists now. On an Apple M4 / 16 GB, RealESRGAN_x2plus
loaded through spandrel 0.4.2 and ran a 128x128 -> 256x256 forward on
``torch.device("mps")`` in 0.51 s, every output finite, and -- the part that
makes it a receipt rather than a hopeful log line -- **bit-exact against the CPU
path on the same input and weights: max |mps - cpu| = 0.00000**. A dead or
silently wrong Metal kernel cannot produce that.

Torch is imported lazily inside the resolver so this module stays cold-import-clean
(V-12 invariant) and can be exercised from unit tests without pulling torch.
"""
from __future__ import annotations

from .._otr_shared.engine_registry_base import EngineUnusable, EngineUsabilityReason


def resolve_device(profile_value: str):
    """Resolve ``profile_value`` to a ``torch.device``. Raises
    :class:`EngineUnusable` on any invalid selection.

    Accepts: ``"cpu"``, ``"mps"``, ``"cuda"`` (= ``cuda:0``), ``"cuda:N"``.
    Rejects: any other token.
    """
    import torch

    v = (profile_value or "cpu").strip().lower()
    if v == "cpu":
        return torch.device("cpu")
    if v == "mps":
        # Same fail-loud shape as the cuda branch below: a profile that ASKS for
        # Metal on a box without it is a misconfiguration to name, never
        # something to quietly satisfy on the CPU. The NO-FALLBACKS ethos in
        # this module's header applies to every backend equally.
        if not (getattr(torch, "backends", None)
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()):
            raise EngineUnusable(
                "<upscale>", "upscale_stage",
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "mps requested but torch.backends.mps.is_available() is False",
                kind="upscale")
        return torch.device("mps")
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
