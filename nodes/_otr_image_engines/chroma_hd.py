"""Chroma1-HD image adapter -- a model-agnostic image peer (C5).

A 5th image engine from ``C2_DEP_LICENSE_MATRIX.md`` (the Apache-2.0 open base for
finetunes). It registers EXACTLY like the other image peers, growing the open
per-role registry. Unlike the GGUF peers (Qwen / HiDream, in-stack ComfyUI-GGUF),
Chroma1-HD is a NATIVE checkpoint loaded through ComfyUI's own UNET/CLIP/VAE
loaders on the protected cu130 main venv -- it is FLUX-family, so it shares the
shipped Flux gen-1 stack and the same attention landmine.

Per the matrix: Apache-2.0 (commercial-clean), an open base for finetunes; the
sm_120 / dep risk cell is "verify". Because it is FLUX-style attention, its
verify-on-5080 MUST confirm SageAttention is NOT patched onto it before the first
forward (BUG-070: a FLUX-style RoPE int8-PV path hard-aborts on sm_120 when
SageAttention is patched). That is a render-time GPU concern; the CPU scaffold
keeps the standard fail-closed contract.

Flux stays gen 1; Chroma is an OPT-IN peer (``default_roles=()`` -- no model is
"primary"), greyed until ``OTR_ENABLE_CHROMA=1`` AND its checkpoint exists. The
fail-closed gate is the WEIGHTS FILE (``OTR_CHROMA_CKPT``): ``assert_usable``
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

log = logging.getLogger("OTR.image.chroma_hd")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_CHROMA"

#: Env var pointing at the downloaded Chroma1-HD checkpoint file. Absent / not a
#: file -> ``assert_usable`` fails closed. Native in-stack load (ComfyUI loaders),
#: so this is a WEIGHTS path, not a sidecar python.
MODEL_ENV = "OTR_CHROMA_CKPT"


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class Chroma1HDEngine:
    """The Chroma1-HD image adapter (reduced ``prompt -> image`` protocol)."""

    name = "chroma_hd"
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
        """FAIL CLOSED until the Chroma1-HD checkpoint exists (BUG-046):
        ABSENT/greyed, never a stub. The registry already gates on
        ``requires_flag``; this is the deeper disk check. The verify-on-5080 must
        additionally confirm SageAttention is not patched (BUG-070, FLUX-family)
        -- a GPU concern enforced at the live forward, not on CPU."""
        ckpt = os.getenv(MODEL_ENV, "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"chroma_hd checkpoint not found; set {MODEL_ENV} to the "
                f"downloaded checkpoint path after the verify-on-5080 GO "
                f"(confirm SageAttention not patched -- BUG-070, FLUX-family attn)",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the in-stack ComfyUI Chroma1-HD loaders (disk-path
        .png handoff, never a tensor across the dispatcher boundary). GPU/operator
        smoke; the CPU layer tests only registry / protocol / role / fail-closed
        behaviour. Lazy-imports the heavy path so module import stays cold-import
        clean (V-12)."""
        raise NotImplementedError(
            "chroma_hd.render_image is the in-stack GPU/operator smoke; download "
            "the Chroma1-HD checkpoint, set OTR_CHROMA_CKPT, and run the "
            "verify-on-5080 checklist (BUG-070 SageAttention + VRAM <= 14.5 GB) first"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


__all__ = ["Chroma1HDEngine", "ENABLE_FLAG", "MODEL_ENV"]
