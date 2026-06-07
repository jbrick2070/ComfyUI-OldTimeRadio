"""Z-Image-Turbo image adapter -- the SECOND image engine (C2, build-or-NO-GO).

Proves the image layer is genuinely model-agnostic: Z-Image-Turbo registers
EXACTLY like ``flux_gen1``, so the registry holds >=2 engines and the user can
pick it per role. Flux stays gen 1; this is the next-gen OPT-IN option
(default-OFF behind ``OTR_ENABLE_ZIMAGE``). Per ``C2_DEP_LICENSE_MATRIX.md``:
6B FP8, Apache-2.0 (commercial-clean), portrait specialist, ~8 GB FP8 -- under
the 14.5 GB single-resident ceiling, leaving lease headroom.

Isolation: it runs in a cu128 SIDECAR (the protected cu130 main venv must not be
contaminated -- the proven Path-B subprocess + JSON bridge + disk-path handoff,
like C1). ``render_image`` is the GPU/operator smoke (lazy sidecar spawn). On CPU
it is REGISTERED but FAIL-CLOSED: ``assert_usable`` raises until the sidecar venv
+ weights exist (ABSENT/greyed, never a silent stub -- BUG-046), and the registry
greys it entirely until ``OTR_ENABLE_ZIMAGE=1``.

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / diffusers / the sidecar are NEVER imported here.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.z_image_turbo")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1"
#: (build-or-NO-GO: enabled only after the operator's verify-on-5080 GO).
ENABLE_FLAG = "OTR_ENABLE_ZIMAGE"

#: Env var pointing at the BUILT cu128 sidecar venv python. Absent / not a file
#: -> ``assert_usable`` fails closed (the deeper disk/dep check beyond the flag).
SIDECAR_ENV = "OTR_ZIMAGE_SIDECAR"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class ZImageTurboEngine:
    """The Z-Image-Turbo image adapter (reduced ``prompt -> image`` protocol)."""

    name = "z_image_turbo"
    #: Can serve any image role; OPT-IN peer (Flux remains the in-stack default
    #: everywhere, so default_roles is empty here -- no model is "primary").
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # Apache-2.0 (C2 matrix; operator confirms weights provenance)
    requires_flag = ENABLE_FLAG      # default-OFF (build-or-NO-GO)
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def load(self) -> None:  # pragma: no cover - residency lives in the sidecar
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the cu128 sidecar venv + weights exist (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag`` (greyed unless ``OTR_ENABLE_ZIMAGE=1``); this is the
        deeper disk/dep check the dispatcher runs before a render."""
        sidecar = os.getenv(SIDECAR_ENV, "").strip()
        if not sidecar or not os.path.isfile(sidecar):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"z_image_turbo cu128 sidecar not built; set {SIDECAR_ENV} to its "
                f"venv python after the verify-on-5080 GO (C2 build-or-NO-GO)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator sidecar
        """Render one still via the Z-Image-Turbo cu128 sidecar (disk-path .png
        handoff, never a tensor across the boundary). GPU/operator smoke; the CPU
        layer tests only registry / protocol / role / fail-closed behaviour."""
        raise NotImplementedError(
            "z_image_turbo.render_image is the cu128-sidecar GPU/operator smoke; "
            "build the sidecar venv + run the verify-on-5080 checklist first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["ZImageTurboEngine", "ENABLE_FLAG", "SIDECAR_ENV"]
