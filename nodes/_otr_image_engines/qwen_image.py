"""Qwen-Image (GGUF) image adapter -- the THIRD image engine (C3, generic peer).

Carries the model-agnostic image layer from "it holds two engines" (C2) to "it
holds an OPEN, growable set": Qwen-Image registers EXACTLY like ``flux_gen1`` and
``z_image_turbo``, so the registry now holds >=3 peers the user picks per role.
Per ``C2_DEP_LICENSE_MATRIX.md`` Qwen-Image is the runner-up pick: 20B, GGUF Q4,
Apache-2.0 (commercial-clean), an all-rounder (in-image text + an edit variant)
where Z-Image is the portrait specialist.

The C3 demonstration -- a DIFFERENT integration mode behind the SAME protocol.
Z-Image (C2) runs in a cu128 SIDECAR (contaminating FP8 deps). Qwen-Image GGUF
instead rides the ALREADY-INSTALLED in-stack ComfyUI-GGUF loader on the protected
cu130 main venv: a GGUF checkpoint is dtype-isolated and does NOT swap torch or
drag a banned dep onto the stack, so it needs no sidecar. Two engines, two
integration paths, one ``prompt -> image`` contract -- that is the proof the layer
is genuinely model-agnostic, not a two-engine special case.

Because there is no sidecar venv to gate on, the fail-closed check is the GGUF
WEIGHTS FILE: ``assert_usable`` raises MISSING_MODEL until ``OTR_QWEN_IMAGE_GGUF``
points at the downloaded ``.gguf`` checkpoint (ABSENT/greyed, never a silent stub
-- BUG-046). The registry greys the engine entirely until ``OTR_ENABLE_QWEN_IMAGE
=1`` (default-OFF; build-or-NO-GO -- enabled only after the operator's
verify-on-5080 GO, since ~12-15 GB Q4 is TIGHT under the 14.5 GB single-resident
ceiling and the per-quant VRAM must be confirmed on the 5080 before commit).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / the ComfyUI-GGUF loader / the model are NEVER
imported here -- the heavy path is lazy, inside ``render_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.qwen_image")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1"
#: (build-or-NO-GO: enabled only after the operator's verify-on-5080 GO).
ENABLE_FLAG = "OTR_ENABLE_QWEN_IMAGE"

#: Env var pointing at the downloaded Qwen-Image GGUF (Q4) checkpoint file. Absent
#: / not a file -> ``assert_usable`` fails closed (the deeper disk check beyond the
#: flag). Unlike Z-Image this is a WEIGHTS path, not a sidecar python -- Qwen-Image
#: GGUF runs in-stack via the installed ComfyUI-GGUF loader, no cu128 sidecar.
MODEL_ENV = "OTR_QWEN_IMAGE_GGUF"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class QwenImageEngine:
    """The Qwen-Image (GGUF) image adapter (reduced ``prompt -> image`` protocol)."""

    name = "qwen_image"
    #: Can serve any image role; OPT-IN peer (Flux remains the in-stack default
    #: everywhere, so default_roles is empty here -- no model is "primary").
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # Apache-2.0 (C2 matrix; operator confirms weights provenance)
    requires_flag = ENABLE_FLAG      # default-OFF (build-or-NO-GO)
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def load(self) -> None:  # pragma: no cover - residency owned by the GGUF loader
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the Qwen-Image GGUF checkpoint exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag`` (greyed unless ``OTR_ENABLE_QWEN_IMAGE=1``); this is the
        deeper disk check the dispatcher runs before a render. The gate is the
        WEIGHTS file (in-stack GGUF), not a sidecar venv."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"qwen_image GGUF checkpoint not found; set {MODEL_ENV} to the "
                f"downloaded .gguf path after the verify-on-5080 GO (C3 generic "
                f"engine; per-quant VRAM must be confirmed on the 5080 first)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the in-stack ComfyUI-GGUF Qwen-Image loader (disk
        -path .png handoff, never a tensor across the dispatcher boundary).
        GPU/operator smoke; the CPU layer tests only registry / protocol / role /
        fail-closed behaviour. Lazy-imports the heavy path so module import stays
        cold-import clean (V-12)."""
        raise NotImplementedError(
            "qwen_image.render_image is the in-stack GGUF GPU/operator smoke; "
            "download the Qwen-Image GGUF, set OTR_QWEN_IMAGE_GGUF, and run the "
            "verify-on-5080 checklist (per-quant VRAM <= 14.5 GB) first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["QwenImageEngine", "ENABLE_FLAG", "MODEL_ENV"]
