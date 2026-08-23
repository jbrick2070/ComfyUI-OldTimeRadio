"""Pydantic schemas for the image-gen platform (C1) -- ``extra="forbid"`` everywhere.

Mirrors the video schemas' discipline (a typo'd key is a hard error, never a
silently-dropped field) but for the reduced ``prompt -> image`` domain. These are
the contracts at the C->A/B seam: what an image request carries, what a rendered
still records, and the per-engine config the director emits.

Dependency note: pydantic is imported at module scope (same as the video
schemas). The cold-import invariant (V-12) bans torch / transformers / diffusers
at import, not pydantic -- the video schemas already establish that.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


#: Image generation granularity. ``per_object`` REUSES one image per
#: character/prop/announcer (maps to mesh-once-per-character for 3D); ``per_beat``
#: generates a FRESH image per beat. Any role on the 3D Model Renderer is
#: hard-locked to ``per_object`` (per_beat -> mesh-rebuild-per-beat; BANNED).
GRANULARITY_MODES: tuple = ("per_object", "per_beat")

# (lean-mean 2026-08-22) ``IMAGE_INPUT_TOKENS`` was here and is DELETED. It
# advertised itself as "shared vocabulary with role_compat" and was neither:
# nothing in the repo read it, and the vocabulary it claimed to share was a
# 2-token subset of the 4-token frozenset that is actually enforced --
# ``nodes/_otr_shared/role_compat.INPUT_TOKENS`` ({text_prompt, init_image,
# audio_ref, base_clip_ref}), checked at role_compat.py's
# ``required_set <= INPUT_TOKENS`` gate.
#
# So it was worse than dead: it was a DECOY. An author adding an image engine
# would have consulted it, seen two legal tokens, and had no way to learn that
# the real gate accepts four and lives in another module. One vocabulary, one
# owner -- import from ``role_compat`` if you need the token set.


class _Forbid(BaseModel):
    """Common base: reject unknown keys (extra='forbid') everywhere."""

    model_config = ConfigDict(extra="forbid")


class ImageEngineConfig(_Forbid):
    """The ``IMAGE_ENGINE_CONFIG`` the director resolves per role (spine §J)."""

    engine_id: str
    role: str
    granularity: str = "per_object"
    resolution: tuple = (1024, 1024)
    enabled: bool = True
    sidecar_venv: Optional[str] = None
    custom: bool = False


class ImageRequest(_Forbid):
    """One image render request for one (role, object) -- the dispatcher unit.

    The cache key is composed from ``(role, object_id, prompt_hash, seed,
    engine_id, engine_version)`` (PASS-IMG MUST-FIX #5); ``prompt`` is the
    LLM-derived text and MUST NOT be a path (the dispatcher asserts this -- a
    distinct prompt-STRING vs path-STRING contract, SHOULD-FIX).
    """

    request_id: str
    role: str
    object_id: str
    engine_id: str
    engine_version: str = "1"
    prompt: str = ""
    prompt_hash: str = ""
    negative_prompt: Optional[str] = None
    init_image: Optional[str] = None     # PATH to a .png, never an IMAGE tensor
    seed: int = 0
    granularity: str = "per_object"
    width: int = 1024
    height: int = 1024


class CanonicalImage(_Forbid):
    """A rendered still recorded in the ledger (content-addressed on disk).

    ``path`` is ``output/otr/stills/{portrait_content_hash}.png`` (AS-5,
    never-overwrite). ``portrait_content_hash`` = sha256 of the DECODED pixels so
    a re-gen yields a new hash and B's mesh cache (keyed on it) invalidates
    correctly. ``has_alpha`` records straight/premultiplied for the 3D overlay.
    """

    image_id: str
    role: str
    object_id: str
    path: str
    width: int = 0
    height: int = 0
    image_format: str = "png"
    engine_id: str = ""
    engine_version: str = "1"
    request_hash: str = ""               # the (role,object,prompt,seed,engine) cache key
    portrait_content_hash: str = ""      # decoded-pixel hash == the file name stem
    prompt_hash: str = ""
    provenance: dict = Field(default_factory=dict)


class ImageLedgerSection(_Forbid):
    """The ``ledger['images']`` section the dispatcher writes back."""

    image_revision: int = 1
    granularity_by_role: dict = Field(default_factory=dict)
    images: list[CanonicalImage] = Field(default_factory=list)
    #: request-cache-key -> image_id (dispatch dedup; a changed key -> regen).
    cache_index: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
