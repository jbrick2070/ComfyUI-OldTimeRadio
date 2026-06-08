"""FLUX.2 Klein image adapter -- a model-agnostic image peer (C8).

A matrix peer from ``C2_DEP_LICENSE_MATRIX.md`` (the GO #2 candidate: 4B, strong
pose + multi-reference character consistency). Two honesty caveats are baked in:

1. **License (verify):** Klein is in the FLUX family, whose dev weights are BFL
   non-commercial; the matrix records the license as "verify (FLUX family)".
   Until the operator confirms a commercial grant, ``commercial_clean = False``
   (conservative -- the license gate treats it as restricted, never optimistic).
2. **Attention landmine (BUG-070):** FLUX-style RoPE int8-PV attention hard-aborts
   on sm_120 if SageAttention is patched onto it. The verify-on-5080 MUST confirm
   SageAttention is NOT patched before the first forward, and ~13 GB native is
   TIGHT on 16 GB -- both are render-time GPU concerns the CPU scaffold records.

Native checkpoint loaded through ComfyUI's own loaders on the protected cu130 main
venv (the shipped Flux gen-1 stack). Flux gen-1 stays the in-stack default; Klein
is an OPT-IN peer (``default_roles=()`` -- no model is "primary"), greyed until
``OTR_ENABLE_FLUX2_KLEIN=1`` AND its checkpoint exists. The fail-closed gate is
the WEIGHTS FILE (``OTR_FLUX2_KLEIN_CKPT``): ``assert_usable`` raises MISSING_MODEL
until it points at the downloaded checkpoint (ABSENT/greyed, never a stub --
BUG-046).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / comfy / the model are NEVER imported here -- the
heavy path is lazy, inside ``render_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.flux2_klein")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_FLUX2_KLEIN"

#: Env var pointing at the downloaded FLUX.2 Klein checkpoint file. Absent / not a
#: file -> ``assert_usable`` fails closed. Native in-stack load (ComfyUI loaders),
#: so this is a WEIGHTS path, not a sidecar python.
MODEL_ENV = "OTR_FLUX2_KLEIN_CKPT"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class Flux2KleinEngine:
    """The FLUX.2 Klein image adapter (reduced ``prompt -> image`` protocol)."""

    name = "flux2_klein"
    roles = ROLES
    default_roles = ()
    #: FLUX-family license is unconfirmed ("verify") -> conservative False until
    #: the operator confirms a commercial grant.
    commercial_clean = False
    requires_flag = ENABLE_FLAG      # default-OFF
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def load(self) -> None:  # pragma: no cover - residency owned by comfy loaders
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the FLUX.2 Klein checkpoint exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the WEIGHTS file). The
        verify-on-5080 must also confirm SageAttention is not patched (BUG-070,
        FLUX-style attn) and the FLUX-family license -- both GPU/operator gates."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"flux2_klein checkpoint not found; set {MODEL_ENV} to the "
                f"downloaded checkpoint path after the verify-on-5080 GO (confirm "
                f"SageAttention not patched -- BUG-070 -- and the FLUX-family license)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the in-stack ComfyUI FLUX.2 Klein loaders
        (disk-path .png handoff, never a tensor across the dispatcher boundary).
        GPU/operator smoke; the CPU layer tests only registry / protocol / role /
        fail-closed behaviour. Lazy-imports the heavy path so module import stays
        cold-import clean (V-12)."""
        raise NotImplementedError(
            "flux2_klein.render_image is the in-stack GPU/operator smoke; download "
            "the FLUX.2 Klein checkpoint, set OTR_FLUX2_KLEIN_CKPT, and run the "
            "verify-on-5080 checklist (BUG-070 SageAttention + license + VRAM) first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["Flux2KleinEngine", "ENABLE_FLAG", "MODEL_ENV"]
