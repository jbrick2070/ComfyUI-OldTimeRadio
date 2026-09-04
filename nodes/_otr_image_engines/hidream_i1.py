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

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.image.hidream_i1")

#: Opt-in flag (default-OFF). The registry greys the engine until set to "1".
ENABLE_FLAG = "OTR_ENABLE_HIDREAM"

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "One present-tense sentence, natural prose, under 24 words. Begin with the "
    "subject and action. Preserve every required beat fact. Camera direction and "
    "speed only from Camera; if NONE, no camera wording. No weight syntax, no tags."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
PROVENANCE FIRST: **the operator supplied this directive himself on 2026-08-17,
drafted from PUBLIC DOCS and explicitly labelled "NOT yet validated".** Stored
verbatim, not rewritten. This engine is default-OFF (``requires_flag =
ENABLE_FLAG``), so it is opt-in rather than live in the menu.

REGISTRY NOTES -- v2, VALIDATED by the operator against public docs 2026-08-17.
Recorded as authored; the probe gate still rules and none of this is built.

IDENTITY: 17B sparse-MoE MMDiT, MIT licence, 2025-04-07. **QUAD encoders:** CLIP-L
+ CLIP-G (77-token class, contributing POOLED GLOBAL vectors) + T5-XXL +
Llama-3.1-8B-Instruct intermediate layers, with T5 and Llama concatenating into the
sequence stream.

**THE CORRECTION, AND IT INVALIDATES WHAT THIS FILE SAID AN HOUR EARLIER.** The
first version of this note stated flatly that "HiDream SUPPORTS a negative, unlike
FLUX.2". **That is wrong: negative support is VARIANT-dependent, not model-wide.**
  * **Full** -- 50 steps, cfg 5.0, shift 3.0 -> negative **LIVE**.
  * **Dev** -- 28 steps, cfg 1.0, shift 6.0 -> negative **INERT**.
  * **Fast** -- 16 steps, cfg 1.0, shift 3.0 -> negative **INERT**.
**So the registry MUST record variant + cfg beside the engine, or the negative field
silently lies** -- which is precisely the class of defect item H just closed on the
dispatcher side, arriving here from a different direction. Full at full precision
needs 27GB+ and will NOT fit this box's 16GB; fp8/GGUF quants of any tier will.
**Whatever lands on disk decides the negative reality**, sampler-side.

effective_cap: reference `max_sequence_length` is **128 tokens** (~90 words), with
community reports of degradation past it; SD.Next ships a 256 default via override.
So: 128 default, extendable, QUALITY-GATED -- and a beat budget is far inside it.
(The first draft said "128 reference, 248 tolerated"; the corrected framing is that
extension is possible but costs quality, not that 248 is free.)

style_token_position: **prepend-tolerant**, as drafted, and now with a reason --
ordering pressure lives in the long T5/Llama sequence stream, while the CLIPs
contribute POOLED vectors where position matters least. A/B per gate regardless.

THE ONE-STRING CONTRACT IS CONFIRMED, and it stays the sharp caveat here:
per-encoder prompt fields (clip vs llama splits) DO exist in advanced node packs.
**Do not enable them without the schema change.** A reader who discovers the
capability and wires it against a one-string writer sends one value into a slot
that expects several.

THE GATE, endorsed unchanged and still NOT built: "engine selectable AND directive
present, else hard refuse at selection -- no bare-writer run, no borrowed
directive." Nothing reads these constants yet, so build it in the change that wires
them.

INSTALL-DAY VERIFY, for this engine: **which variant is actually on disk**, because
that single fact decides the negative, the shift and the step count.
"""

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
        ckpt = otr_env.get(MODEL_ENV, "").strip()
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
