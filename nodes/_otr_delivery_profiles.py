"""Delivery profiles (plan E.3, Wave 1 / 1e).

A delivery profile shapes HOW a line is performed without touching the canonical
spoken words: it may project text and overlay engine params. Only ``neutral``
ships in v2 -- it is a strict IDENTITY (no text projection, no tags, no transform
on top of the engine defaults), so the default audio path is unchanged. The
profile id + version stay in the ``ResolvedVoiceRequest`` cache key as identity
even while only neutral exists (dropping them would force a future re-baseline);
populated profiles arrive in v2.1.

Import-time is side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

DELIVERY_PROFILE_VERSION = "1"
# Versions the cache IS_CHANGED + sidecar key on (E.5). Bumping either
# invalidates cached audio cleanly.
DELIVERY_PROJECTION_VERSION = "1"


class UnknownDeliveryProfile(ValueError):
    """Raised for a delivery profile id that is not registered (fail-closed)."""


@dataclass(frozen=True)
class DeliveryProfile:
    """One delivery profile. ``neutral`` is the identity profile (E.3)."""

    # `version` / `projection_version` fields were removed 2026-08-28: the ONE
    # instance (`_NEUTRAL`) always took the defaults and nothing ever read
    # them, so they advertised per-profile versioning that did not exist. The
    # live consumers read the MODULE constants below directly (cast_lock and
    # the audio cache both do), which is where the real versions live.
    profile_id: str

    def project_text(self, text: str) -> str:
        """Project the spoken text. Neutral = identity (no transform)."""
        return text if text is not None else ""

    def overlay_params(self, params: Optional[dict]) -> dict:
        """Overlay engine params. Neutral = the engine defaults unchanged."""
        return dict(params or {})


_NEUTRAL = DeliveryProfile("neutral")
_PROFILES: Dict[str, DeliveryProfile] = {"neutral": _NEUTRAL}


def available_delivery_profiles() -> List[str]:
    """Stable list of registered delivery profile ids (only ``neutral`` in v2)."""
    return sorted(_PROFILES)


def get_delivery_profile(profile_id: str = "neutral") -> DeliveryProfile:
    """Return the profile for ``profile_id`` or raise :class:`UnknownDeliveryProfile`."""
    profile = _PROFILES.get(profile_id)
    if profile is None:
        raise UnknownDeliveryProfile(
            f"unknown delivery profile {profile_id!r}; available: "
            f"{available_delivery_profiles()}"
        )
    return profile


def apply_delivery(profile: DeliveryProfile, text: str, params: Optional[dict]):
    """Apply a delivery profile -> (projected_text, overlaid_params).

    Pure + deterministic. For neutral this returns the text unchanged and the
    engine params unchanged.
    """
    return profile.project_text(text), profile.overlay_params(params)
