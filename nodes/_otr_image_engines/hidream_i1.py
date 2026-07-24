"""HiDream-I1 (GGUF) image adapter -- a model-agnostic image peer (C4).

A 4th image engine from ``C2_DEP_LICENSE_MATRIX.md`` (the MIT-licensed GGUF
candidate). It registers EXACTLY like ``flux_gen1`` / ``z_image_turbo`` /
``z_image_turbo``, growing the open registry of per-role peers. Like the other
rides the ALREADY-INSTALLED in-stack ComfyUI-GGUF loader -- a GGUF checkpoint is
dtype-isolated and does NOT swap torch or pull a banned dep onto the protected
cu130 main venv, so it needs no cu128 sidecar. Per the matrix: HiDream-I1 Fast is
GGUF, MIT (commercial-clean), ~13-15 GB (TIGHT under the 14.5 GB single-resident
ceiling -- the per-quant VRAM must be confirmed on the 5080 before commit).

Flux stays gen 1; HiDream is an OPT-IN peer (``default_roles=()`` -- no model is
"primary"), greyed until ``OTR_ENABLE_HIDREAM=1`` AND its GGUF weights exist. The
fail-closed gate is the GGUF WEIGHTS FILE (``OTR_HIDREAM_GGUF``): ``assert_usable``
raises MISSING_MODEL until it points at the downloaded ``.gguf`` (ABSENT/greyed,
never a silent stub -- BUG-046).

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary + stdlib. torch / the ComfyUI-GGUF loader / the model are NEVER
imported here -- the heavy path is lazy, inside ``render_image``.
"""
from __future__ import annotations

import logging
import os

from .registry import EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.hidream_i1")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_HIDREAM"

#: Env var pointing at the downloaded HiDream-I1 GGUF checkpoint file. Absent /
#: not a file -> ``assert_usable`` fails closed. In-stack GGUF (the installed
#: ComfyUI-GGUF loader), so this is a WEIGHTS path, not a sidecar python.
MODEL_ENV = "OTR_HIDREAM_GGUF"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


class HiDreamI1Engine:
    """The HiDream-I1 (GGUF) image adapter (reduced ``prompt -> image`` protocol)."""

    name = "hidream_i1"
    roles = ROLES
    default_roles = ()
    commercial_clean = True          # MIT (C2 matrix; operator confirms weights provenance)
    requires_flag = ENABLE_FLAG      # default-OFF
    required_inputs = ("text_prompt",)
    engine_version = "1"

    def load(self) -> None:  # pragma: no cover - residency owned by the GGUF loader
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """FAIL CLOSED until the HiDream-I1 GGUF checkpoint exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check. The gate is the WEIGHTS
        file (in-stack GGUF), not a sidecar venv."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"hidream_i1 GGUF checkpoint not found; set {MODEL_ENV} to the "
                f"downloaded .gguf path after the verify-on-5080 GO (per-quant "
                f"VRAM must be confirmed on the 5080 first -- ~13-15 GB is tight)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the in-stack ComfyUI-GGUF HiDream-I1 loader
        (disk-path .png handoff, never a tensor across the dispatcher boundary).
        GPU/operator smoke; the CPU layer tests only registry / protocol / role /
        fail-closed behaviour. Lazy-imports the heavy path so module import stays
        cold-import clean (V-12)."""
        raise NotImplementedError(
            "hidream_i1.render_image is the in-stack GGUF GPU/operator smoke; "
            "download the HiDream-I1 GGUF, set OTR_HIDREAM_GGUF, and run the "
            "verify-on-5080 checklist (per-quant VRAM <= 14.5 GB) first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["HiDreamI1Engine", "ENABLE_FLAG", "MODEL_ENV"]
