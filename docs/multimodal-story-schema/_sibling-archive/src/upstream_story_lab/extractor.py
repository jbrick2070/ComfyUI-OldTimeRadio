"""Phase A prompt extractor -- returns pack overrides or None (passthrough).

None means "no override -- production caller uses its Python literal".
Reserved solely for intentional empty override. All structural failures
(unknown bank/model/pipeline, unknown seam) raise RegistryError.
"""
from __future__ import annotations

from .contracts import PRODUCTION_TEMPLATE_SEAMS
from .registry import Registry, RegistryError


def get_pack_prompt_or_none(
    registry: Registry,
    source_bank_id: str,
    story_model_id: str,
    story_pipeline_id: str,
    seam_key: str,
) -> str | None:
    """Return pack.prompt_stages[seam_key] if present and non-empty; else None."""
    if seam_key not in PRODUCTION_TEMPLATE_SEAMS:
        raise RegistryError(
            f"unknown Phase A production seam: {seam_key!r}"
        )
    # registry.pack raises UnknownIdError (a RegistryError subclass) on
    # unknown triple -- canonical error text; don't hand-roll .packs.get().
    pack = registry.pack(source_bank_id, story_model_id, story_pipeline_id)
    value = pack.prompt_stages.get(seam_key, "").strip()
    return value or None
