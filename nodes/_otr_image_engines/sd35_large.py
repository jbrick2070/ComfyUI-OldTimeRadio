"""Stable Diffusion 3.5 Large image adapter -- a model-agnostic image peer (C7).

A matrix peer from ``C2_DEP_LICENSE_MATRIX.md`` with a CONDITIONAL license, kept
HONEST. SD 3.5 Large ships under the Stability AI Community License: free for
non-commercial use and for commercial use only BELOW a revenue threshold (the
matrix records "Community (<$1M free)"). That is NOT an unconditional commercial
grant, so ``commercial_clean = False`` -- the UI / license gate must treat it as
restricted until the operator confirms OTR's use stays within the free tier (or a
paid Stability license is held). This is the point of the flag: the registry can
hold mixed-license peers and surface each one's real terms instead of pretending
everything is Apache.

Native checkpoint (FP8, ~11 GB per the matrix), loaded through ComfyUI's own
loaders on the protected cu130 main venv (not GGUF, not a sidecar). Flux stays
gen 1; SD 3.5 is an OPT-IN peer (``default_roles=()``), greyed until
``OTR_ENABLE_SD35=1`` AND its checkpoint exists. The fail-closed gate is the
WEIGHTS FILE (``OTR_SD35_CKPT``): ``assert_usable`` raises MISSING_MODEL until it
points at the downloaded checkpoint (ABSENT/greyed, never a stub -- BUG-046).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / comfy / the model are NEVER imported here -- the
heavy path is lazy, inside ``render_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.sd35_large")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_SD35"

#: Env var pointing at the downloaded SD 3.5 Large checkpoint file. Absent / not a
#: file -> ``assert_usable`` fails closed. Native in-stack load (ComfyUI loaders),
#: so this is a WEIGHTS path, not a sidecar python.
MODEL_ENV = "OTR_SD35_CKPT"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


class SD35LargeEngine:
    """The SD 3.5 Large image adapter (reduced ``prompt -> image`` protocol)."""

    name = "sd35_large"
    roles = ROLES
    default_roles = ()
    #: Stability Community License is conditional (free only below a revenue
    #: threshold) -> NOT unconditionally commercial-clean. Honest, conservative.
    commercial_clean = False
    requires_flag = ENABLE_FLAG      # default-OFF
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def load(self) -> None:  # pragma: no cover - residency owned by comfy loaders
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the SD 3.5 Large checkpoint exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the WEIGHTS file). The
        commercial-clean gate is metadata the license gate reads, not a usability
        check -- it is False here (conditional Community license)."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"sd35_large checkpoint not found; set {MODEL_ENV} to the "
                f"downloaded checkpoint path after the verify-on-5080 GO "
                f"(confirm the Community-license revenue tier before commercial use)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the in-stack ComfyUI SD 3.5 Large loaders
        (disk-path .png handoff, never a tensor across the dispatcher boundary).
        GPU/operator smoke; the CPU layer tests only registry / protocol / role /
        fail-closed behaviour. Lazy-imports the heavy path so module import stays
        cold-import clean (V-12)."""
        raise NotImplementedError(
            "sd35_large.render_image is the in-stack GPU/operator smoke; download "
            "the SD 3.5 Large checkpoint, set OTR_SD35_CKPT, and run the "
            "verify-on-5080 checklist (license tier + VRAM <= 14.5 GB) first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["SD35LargeEngine", "ENABLE_FLAG", "MODEL_ENV"]
