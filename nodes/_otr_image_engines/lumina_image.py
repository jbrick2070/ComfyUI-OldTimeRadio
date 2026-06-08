"""Lumina-Image 2.0 image adapter -- a model-agnostic image peer (C6).

A 6th image engine from ``C2_DEP_LICENSE_MATRIX.md`` (the lightweight Apache-2.0
candidate). It registers EXACTLY like the other image peers, growing the open
per-role registry. Lumina-Image 2.0 is a NATIVE checkpoint loaded through
ComfyUI's own loaders on the protected cu130 main venv (not GGUF, not a sidecar);
per the matrix it is lightweight (the smallest-footprint candidate), Apache-2.0
(commercial-clean). Its sm_120 / dep / ComfyUI-support cells are "verify" -- the
operator confirms the loader path + per-quant VRAM on the 5080 before commit.

Flux stays gen 1; Lumina is an OPT-IN peer (``default_roles=()`` -- no model is
"primary"), greyed until ``OTR_ENABLE_LUMINA=1`` AND its checkpoint exists. The
fail-closed gate is the WEIGHTS FILE (``OTR_LUMINA_CKPT``): ``assert_usable``
raises MISSING_MODEL until it points at the downloaded checkpoint (ABSENT/greyed,
never a silent stub -- BUG-046).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / comfy / the model are NEVER imported here -- the
heavy path is lazy, inside ``render_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.lumina_image")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_LUMINA"

#: Env var pointing at the downloaded Lumina-Image 2.0 checkpoint file. Absent /
#: not a file -> ``assert_usable`` fails closed. Native in-stack load (ComfyUI
#: loaders), so this is a WEIGHTS path, not a sidecar python.
MODEL_ENV = "OTR_LUMINA_CKPT"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class LuminaImage2Engine:
    """The Lumina-Image 2.0 image adapter (reduced ``prompt -> image`` protocol)."""

    name = "lumina_image"
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # Apache-2.0 (C2 matrix; operator confirms weights provenance)
    requires_flag = ENABLE_FLAG      # default-OFF
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def load(self) -> None:  # pragma: no cover - residency owned by comfy loaders
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the Lumina-Image 2.0 checkpoint exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check (the WEIGHTS file)."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"lumina_image checkpoint not found; set {MODEL_ENV} to the "
                f"downloaded checkpoint path after the verify-on-5080 GO "
                f"(confirm the ComfyUI loader path + per-quant VRAM on the 5080)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the in-stack ComfyUI Lumina-Image 2.0 loaders
        (disk-path .png handoff, never a tensor across the dispatcher boundary).
        GPU/operator smoke; the CPU layer tests only registry / protocol / role /
        fail-closed behaviour. Lazy-imports the heavy path so module import stays
        cold-import clean (V-12)."""
        raise NotImplementedError(
            "lumina_image.render_image is the in-stack GPU/operator smoke; "
            "download the Lumina-Image 2.0 checkpoint, set OTR_LUMINA_CKPT, and "
            "run the verify-on-5080 checklist (loader path + VRAM <= 14.5 GB) first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["LuminaImage2Engine", "ENABLE_FLAG", "MODEL_ENV"]
