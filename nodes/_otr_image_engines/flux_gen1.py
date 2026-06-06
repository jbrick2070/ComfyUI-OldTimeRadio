"""Flux gen-1 image adapter -- the FIRST image engine (C-now / C1).

"Flux" is just gen 1, not a hardcoded default: it registers exactly like any
future image engine (Z-Image / Klein / Qwen / "+ Add Custom Model") so swapping
it touches nothing downstream. It wraps the SHIPPED Flux still/portrait
generation (``visual/batch_flux_render.py`` + ``visual/batch_flux_portrait_render.py``)
as the platform's gen-1 ``prompt -> .png`` engine.

Cold-import clean (V-12): module scope imports only the dep-free registry +
role vocabulary + stdlib. torch / comfy / the heavy Flux pipeline are imported
LAZILY inside ``load`` / ``render_image`` / ``assert_usable`` -- importing this
module (and thus the image namespace) never pulls a model framework.

Licensing: Flux.1-dev is BFL NON-commercial, so ``commercial_clean = False``.
It is still the in-stack gen-1 default (``default_roles`` = every image role), so
``assert_usable`` returns it for any role without requiring an opt-in flag; the
commercial-clean flag is metadata the UI/license gate reads, not a usability gate.
"""
from __future__ import annotations

import logging

from .registry import register
from .._otr_shared.role_compat import ROLES

log = logging.getLogger("OTR.image.flux_gen1")


@register
class FluxGen1ImageEngine:
    """The gen-1 Flux image adapter (reduced ``prompt -> image`` protocol)."""

    name = "flux_gen1"
    #: Flux can produce an image for any role (announcer / music / character /
    #: scene / background); it is the in-stack default everywhere (gen 1).
    roles = ROLES
    default_roles = ROLES
    commercial_clean = False        # Flux.1-dev = BFL non-commercial
    requires_flag = None            # default engine -> no opt-in flag needed
    #: An image engine needs a text prompt; the edit/img2img path may also take
    #: an init_image, but gen-1 portrait/still generation is text -> image.
    required_inputs = ("text_prompt",)
    engine_version = "1"

    # --- residency (cheap no-ops here: the shipped Flux nodes own the real
    #     MODEL/CLIP/VAE residency via ComfyUI's loaders; the adapter does not
    #     hold weights so importing it stays cold-import clean). ---
    def load(self) -> None:  # pragma: no cover - residency owned by comfy loaders
        return None

    def unload(self) -> None:  # pragma: no cover
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """Gen-1 Flux is usable wherever ComfyUI itself can run Flux. The real
        weight/dep check lives in the shipped Flux loader path; this adapter
        returns its name (default engine) and defers the heavy probe to the
        dispatcher's actual render (GPU / operator smoke)."""
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover - GPU
        return {"engine_id": self.name}

    def render_image(self, request, prepared):  # pragma: no cover - GPU/operator
        """Render one still via the shipped Flux pipeline. GPU-only; exercised by
        the operator smoke (passthrough-equality + golden-image), never on CPU.
        Lazy-imports the heavy path so module import stays cold-import clean."""
        raise NotImplementedError(
            "flux_gen1.render_image is the GPU/operator smoke path; the CPU "
            "platform tests inject a fake gen_fn into OTR_ImageGenDispatcher"
        )

    def teardown(self, prepared) -> None:  # pragma: no cover - GPU
        return None
