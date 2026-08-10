"""The ONE place that turns a Comfy image SELECTOR into a Google MODEL ID.

WHY THIS EXISTS. The pack offers image models by DISPLAY NAME -- the string an
operator picks in the node -- and resolves that name to a real Google model id at
invoke time. Before this module the resolution lived inline in
``cloud_media_invoke.py``, so the provenance guard, which only ever inspected the
offered display name, could not see the id actually sent to Google. Two real
model ids were shipped with no provenance entry at all, and one of them is a
``preview`` id whose stable twin the pack ALSO ships through a different lane.

The fix is not to teach the guard the mapping -- that would be a second copy of
this table, and two lists that must agree with nothing forcing them to is the
very defect class the guard exists to catch. Instead there is exactly one table,
here, imported by BOTH the invoke path and the inventory collector.

THE FALLTHROUGH IS PART OF THE CONTRACT. Most selectors ARE concrete ids already
(the seedream / krea / photon rows), so an unmapped value resolves to itself.
This function must never raise on an unknown selector: that would break every
one of those rows. It raises only on input that is not a usable selector at all.

Pure data + stdlib. No imports from engine namespaces, no ComfyUI import, so the
provenance test can read it without loading torch.
"""
from __future__ import annotations

from typing import Mapping

#: Display name -> the Google model id actually sent on the wire.
#:
#: Only selectors whose display name DIFFERS from their model id belong here.
#: Everything else resolves by identity, which is why this table is short and
#: must stay short -- an entry that merely restates an id is noise that will
#: drift.
SELECTOR_TO_MODEL_ID: Mapping[str, str] = {
    # NOTE (2026-08-09): this row points at the PREVIEW twin while the stable
    # `gemini-3.1-flash-image` is shipped in parallel by eng_google_image.
    # Measured against Google models.get the same day: both ids resolve, carry
    # IDENTICAL capabilities (65536 in/out, generateContent+countTokens+
    # batchGenerateContent, version 3.0) and the SAME displayName "Nano Banana
    # 2"; neither carries any deprecation, shutdown or sunset field. So this is
    # not a broken route -- it is a preview id used where an equivalent stable
    # id exists. Repointing it changes which model renders stills, which is
    # recipe-adjacent and an operator call, so it is recorded here and NOT
    # changed. See docs/2026-08-09-BUILD-SPEC-slug-provenance-non-video.md.
    "Nano Banana 2 (Gemini 3.1 Flash Image)": "gemini-3.1-flash-image-preview",
    "Nano Banana 2 Lite": "gemini-3.1-flash-lite-image",
}


def resolve_selector_to_model_id(selector: str) -> str:
    """Return the Google model id for ``selector``.

    Mapped selectors resolve through the table; everything else resolves to
    itself, because most offered values are already concrete ids.

    Raises ``ValueError`` only for input that could not be a selector at all --
    a non-string, or an empty/whitespace-only string. An UNKNOWN but well-formed
    selector is NOT an error; it is the identity case.
    """
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError(
            "image model selector must be a non-empty string, got %r" % (selector,)
        )
    return SELECTOR_TO_MODEL_ID.get(selector, selector)


__all__ = ["SELECTOR_TO_MODEL_ID", "resolve_selector_to_model_id"]
