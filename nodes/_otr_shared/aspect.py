r"""Character / HuMo aspect helpers -- one source of truth for the talking-head
render geometry.

The operator wants BOTH looks (2026-06-16), expressed as TWO selectable engines
in the canonical role dropdowns (not a hidden global toggle):

  * humo_1.7B      -- PORTRAIT 480x832, the classic "old-time" talking-head
                      pillarbox (the compositor pillarboxes it into 16:9).
  * humo_1.7B_169  -- WIDE 832x480, the modern 16:9 character beat; the clip
                      fills the 16:9 composite (no pillarbox).

The selecting authority is therefore each ENGINE's ``render_aspect`` attribute,
NOT an environment variable. This module only maps an aspect string to the right
dimensions, so the three call sites stay consistent:

  * eng_humo._native_dims        -> HuMo render size (humo_dims_for_aspect)
  * otr_meta_brief...            -> FLUX character-still size (still_dims_for_aspect)
  * render_driver (canvas)        -> pillarbox vs 16:9 composite (normalize_aspect)

(1.7B framing note: a MEDIUM 832x480 still lip-syncs well; a 832x480 CLOSE-UP
goes mushy on the 1.7B tier -- keep the wide still framing medium.)
"""

PORTRAIT = "portrait"
WIDE = "wide"

#: Spellings that all mean WIDE (16:9), case-insensitive.
_WIDE_ALIASES = frozenset({"wide", "16:9", "169", "landscape"})

#: The 16:9 preset, snapped /32 (the latent-grid contract).
WIDE_DIMS = (832, 480)
#: HuMo's proven portrait render size.
HUMO_PORTRAIT_DIMS = (480, 832)


def normalize_aspect(value):
    """Return ``'wide'`` for any wide spelling, else ``'portrait'`` (the safe
    default for unset / unknown values, keeping the legacy render path)."""
    if str(value or "").strip().lower() in _WIDE_ALIASES:
        return WIDE
    return PORTRAIT


def is_wide(value):
    """True iff ``value`` names the 16:9 aspect."""
    return normalize_aspect(value) == WIDE


def humo_dims_for_aspect(aspect):
    """(w, h) for the HuMo render: wide 832x480 / portrait 480x832."""
    return WIDE_DIMS if is_wide(aspect) else HUMO_PORTRAIT_DIMS


def still_dims_for_aspect(aspect, portrait_w, portrait_h):
    """(w, h) for the FLUX character still.

    Wide -> 832x480 (ONE 16:9 still feeds both LTX scene beats and the
    landscape HuMo, no pillarbox). Portrait -> the caller's proven portrait
    dims ``(portrait_w, portrait_h)`` unchanged."""
    return WIDE_DIMS if is_wide(aspect) else (portrait_w, portrait_h)
