"""Engine-profile resolver (plan D5 / Wave 0 piece 8).

Loads ``config/audio_engine_profiles.yaml`` lazily (cached + exception-wrapped)
and resolves a (role, engine[, voice_bank]) request to a curated
:class:`EngineProfile` via a FAIL-CLOSED ladder that reuses the audio-engine
registry's :class:`EngineUnusable` taxonomy.

C-5 invariants honored:
  * **No module-scope IO.** The YAML is read only when ``load_resolver()`` is
    called -- never at import. A node's ``INPUT_TYPES`` must instead use
    :func:`legacy_first_engines` (a hardcoded, legacy-first, never-empty list)
    so a box-fresh install with a missing/broken YAML still builds a valid combo.
  * **Token/model checks run in generate(), not INPUT_TYPES.**
    :func:`assert_token_for_profile` / :func:`assert_model_available` touch the
    environment / disk and are called on the dispatch path only.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason, assert_usable

log = logging.getLogger("OTR")

PROFILE_SCHEMA_VERSION = "1"
_PROFILES_FILENAME = "audio_engine_profiles.yaml"

# Hardcoded, legacy-first engine order per role for INPUT_TYPES (C-5: never empty
# combo, stable across flags, no IO). The internal build default is the first
# (legacy) entry until promotion (I) flips the shipped default per role.
_LEGACY_FIRST_ENGINES: Dict[str, tuple] = {
    "char_voice": ("bark", "chatterbox", "indextts2"),
    "announcer_voice": ("kokoro", "chatterbox"),
    "music": ("musicgen", "stable_audio_music"),
}


def legacy_first_engines(role: str) -> List[str]:
    """Hardcoded legacy-first engine names for ``role`` (INPUT_TYPES helper).

    Pure + IO-free so a node combo never depends on the YAML being present
    (C-5). Returns the legacy/default engine first.
    """
    return list(_LEGACY_FIRST_ENGINES.get(role, ()))


class EngineProfile(BaseModel):
    """One curated (role, engine) profile row. Frozen + strict (extra keys are a
    config error -> surfaces as malformed_config at load)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    role: str
    engine: str
    commercial_clean: bool
    model_path: str = ""
    model_sha256: str = ""
    default_params: dict = Field(default_factory=dict)
    allowed_voice_banks: List[str] = Field(default_factory=list)
    engine_impl_version: str = "1"
    sample_rate: int = 0
    requires_hf_token: bool = False


class EngineProfileResolver:
    """Resolves casting plans against the loaded profile set (v1)."""

    VERSION = "1"

    def __init__(self, profiles: Dict[str, EngineProfile], source_sha256: str):
        self._by_id = dict(profiles)
        self.source_sha256 = source_sha256

    def profile_ids(self) -> List[str]:
        return sorted(self._by_id)

    def get(self, profile_id: str) -> EngineProfile:
        try:
            return self._by_id[profile_id]
        except KeyError:
            raise EngineUnusable(
                profile_id, "", EngineUsabilityReason.MALFORMED_CONFIG,
                f"no engine profile '{profile_id}'",
            )

    def for_role(self, role: str) -> List[EngineProfile]:
        return [p for p in self._by_id.values() if p.role == role]

    def profile_for(self, role: str, engine: str) -> Optional[EngineProfile]:
        for p in self._by_id.values():
            if p.role == role and p.engine == engine:
                return p
        return None

    def resolve_casting_plan(
        self, *, role: str, engine: str, voice_bank: Optional[str] = None
    ) -> EngineProfile:
        """Resolve (role, engine[, voice_bank]) -> EngineProfile, FAIL CLOSED.

        Ladder: ``assert_usable`` (registry: gated_by_flag / malformed_config /
        incompatible_profile) -> a profile must exist for (role, engine), else
        malformed_config -> if a voice bank is requested it must be in the
        profile's ``allowed_voice_banks``, else incompatible_profile (this is how
        a reference-cloning engine rejects the legacy bark preset bank).
        """
        validated = assert_usable(engine, role)
        profile = self.profile_for(role, validated)
        if profile is None:
            raise EngineUnusable(
                engine, role, EngineUsabilityReason.MALFORMED_CONFIG,
                f"no profile for role '{role}' + engine '{validated}'",
            )
        if voice_bank is not None and voice_bank not in profile.allowed_voice_banks:
            raise EngineUnusable(
                engine, role, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                f"voice bank '{voice_bank}' is not allowed for profile "
                f"'{profile.profile_id}' (allowed: {profile.allowed_voice_banks})",
            )
        return profile


# ---------------------------------------------------------------------------
# Lazy, cached, exception-wrapped loading (no module-scope IO -- C-5)
# ---------------------------------------------------------------------------

_RESOLVER_CACHE: Dict[str, EngineProfileResolver] = {}


def _profiles_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "config", _PROFILES_FILENAME)


def _resolver_from_text(text: str) -> EngineProfileResolver:
    """Parse profiles YAML text into a resolver (cached by content sha). Raises
    on malformed content -- the caller (``load_resolver``) wraps it."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _RESOLVER_CACHE.get(sha)
    if cached is not None:
        return cached
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("audio_engine_profiles: top level must be a mapping")
    rows = data.get("profiles")
    if not isinstance(rows, list) or not rows:
        raise ValueError("audio_engine_profiles: 'profiles' must be a non-empty list")
    by_id: Dict[str, EngineProfile] = {}
    for row in rows:
        profile = EngineProfile.model_validate(row)
        if profile.profile_id in by_id:
            raise ValueError(f"duplicate profile_id '{profile.profile_id}'")
        by_id[profile.profile_id] = profile
    resolver = EngineProfileResolver(by_id, sha)
    _RESOLVER_CACHE[sha] = resolver
    return resolver


def load_resolver() -> Optional[EngineProfileResolver]:
    """Load the profile resolver, or ``None`` if the YAML is missing/broken.

    Exception-wrapped so a bad config never crashes an INPUT_TYPES path; the
    dispatch path treats ``None`` as a fail-closed condition and raises a named
    error there instead.
    """
    try:
        with open(_profiles_path(), "r", encoding="utf-8") as fh:
            return _resolver_from_text(fh.read())
    except Exception as exc:  # noqa: BLE001 -- intentional fail-soft for INPUT_TYPES
        log.debug("[OTR] engine profiles load failed: %s", exc)
        return None


def require_resolver() -> EngineProfileResolver:
    """Load the resolver or raise malformed_config (dispatch-path use)."""
    resolver = load_resolver()
    if resolver is None:
        raise EngineUnusable(
            "", "", EngineUsabilityReason.MALFORMED_CONFIG,
            "audio_engine_profiles.yaml is missing or malformed",
        )
    return resolver


def assert_token_for_profile(profile: EngineProfile) -> None:
    """Raise MISSING_HF_TOKEN if a HF-gated profile lacks ``HF_TOKEN``.

    Reads the environment, so call this on the generate() path, NEVER in
    INPUT_TYPES (C-5).
    """
    if profile.requires_hf_token and not os.getenv("HF_TOKEN"):
        raise EngineUnusable(
            profile.engine, profile.role, EngineUsabilityReason.MISSING_HF_TOKEN,
            f"profile '{profile.profile_id}' needs HF_TOKEN (accept the model "
            f"license and set HF_TOKEN before enabling it)",
        )


def assert_model_available(profile: EngineProfile) -> None:
    """Raise MISSING_MODEL if the profile pins a model_path that is absent.

    Touches disk -> generate()-path only (C-5). A blank model_path means the
    engine resolves its own weights, so there is nothing to check.
    """
    if profile.model_path and not os.path.exists(profile.model_path):
        raise EngineUnusable(
            profile.engine, profile.role, EngineUsabilityReason.MISSING_MODEL,
            f"profile '{profile.profile_id}' model_path does not exist: "
            f"{profile.model_path}",
        )
