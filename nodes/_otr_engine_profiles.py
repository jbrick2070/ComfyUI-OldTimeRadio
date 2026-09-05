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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason, assert_usable

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR")

_PROFILES_FILENAME = "audio_engine_profiles.yaml"

# Sprint 1: validated value sets for the new declarative profile metadata.
# "cloud" (cloud-audio campaign 2026-07-03, C1): a partner-node engine invoked via
# invoke_partner_node (ElevenLabs TTS / Sonilo music) -- dispatch runs through the
# adapter, not profile.runtime, but the runtime tag must validate. "direct_api"
# is a BYO-key provider call made directly by an adapter (not a Partner node).
_VALID_RUNTIMES = {"in_graph", "oop_venv", "cloud", "direct_api"}
_VALID_LICENSE_STATES = {"", "clean", "gated", "unknown"}
# error_policy for cloud engines -- fail-loud, never a silent local fallback.
_VALID_ERROR_POLICIES = {"", "fail_loud"}

# Hardcoded engine order per role for INPUT_TYPES (C-5: never empty combo, stable
# across flags, no IO). Index 0 is the shipped default for the role: char_voice and
# music are PROMOTED to their model-backed defaults (indextts2, stable_audio_3);
# announcer stays kokoro.
_LEGACY_FIRST_ENGINES: Dict[str, tuple] = {
    # char_voice PROMOTED 2026-06-04: indextts2 (Path B oop_venv worker) is the
    # shipped default; chatterbox + dia (both Path B sidecars) + bark are
    # selectable. Index 0 stays indextts2 -> byte-identical default combo.
    # elevenlabs (cloud, dropdown-opt-in) APPENDED 2026-07-03 -- index 0 stays the
    # byte-identical default; a cloud pick is never automatic (C2).
    "char_voice": (
        "indextts2", "chatterbox", "dia", "bark", "kokoro", "elevenlabs",
        "google_tts",
    ),
    # bark APPENDED 2026-08-24 -- a second zero-setup engine for a fresh
    # install (voices baked into weights, no reference WAV to supply).
    # kokoro stays index 0 -- byte-identical default combo.
    "announcer_voice": (
        "kokoro", "chatterbox", "dia", "elevenlabs", "google_tts", "bark",
    ),
    # music PROMOTED 2026-06-03: Stable Audio 3 (ComfyUI-native, no dep conflict,
    # render-proven) is index 0 = the shipped default; musicgen kept selectable.
    # sonilo (cloud, dropdown-opt-in) APPENDED 2026-07-03; google_lyria
    # (direct-api, BYO key) APPENDED 2026-07-08 -- index 0 stays the
    # byte-identical default; external paid picks are never automatic (C2).
    "music": (
        "stable_audio_3", "musicgen", "stable_audio_music", "sonilo",
        "google_lyria",
    ),
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
    default_params: dict = Field(default_factory=dict)
    allowed_voice_banks: List[str] = Field(default_factory=list)
    engine_impl_version: str = "1"
    sample_rate: int = 0
    requires_hf_token: bool = False

    # --- Sprint 1: declarative engine metadata (additive). Defaults keep the
    # existing rows valid; the YAML populates them explicitly. These describe
    # selection intent + licensing; they do NOT change the live byte-identical
    # dispatch (which stays on the engine-level default until promotion S6). ---
    rank: int = 100                  # fallback priority; lower = tried first
    is_default: bool = False         # scope/logical default for the role
    runtime: str = "in_graph"        # in_graph | oop_venv | cloud | direct_api
    needs_ref_clip: bool = False     # reference-clip identity engines
    caps: dict = Field(default_factory=dict)
    license_state: str = ""          # blank -> derive from commercial_clean
    warn_text: str = ""              # non-blocking notice emitted when gated

    # --- Cloud-audio campaign 2026-07-03 (C1): declarative cloud-engine metadata
    # (additive; blank defaults keep every existing row valid). Populated only on
    # runtime="cloud" profiles (ElevenLabs TTS / Sonilo music). These describe the
    # partner-node invoke contract; they do NOT change the byte-identical default
    # dispatch (the cloud engine is dropdown-opt-in, never a default). ---
    partner_row: str = ""            # pinned partner_nodes.yaml key (e.g. cloud_elevenlabs_tts)
    required_param_defaults: dict = Field(default_factory=dict)
    auth_required: bool = False      # cloud engines need OTR_COMFY_API_KEY (fail-loud)
    error_policy: str = ""           # "" | fail_loud (cloud = fail_loud, no local fallback)

    # --- Cloud-audio-cache chunk 2 (2026-08-08): content-addressed replay
    # activation. When True, the per-line voice dispatcher runs FileAudioCache
    # get/put around the adapter call so drift-prone providers (Gemini TTS) pin
    # to identical bytes on re-render. Default False keeps every LOCAL profile
    # byte-identical to today's shipped path; the two Google TTS profiles opt
    # in. Read at nodes/_otr_voice_node_common._render_per_line. ---
    use_cache: bool = False

    # --- Voice identity 2026-08-18 (PBUG-20260817-09): character-stable engine
    # seeding. The per-engine external seed is normally reduced from
    # `stable_line_seed`, which includes `line_id` -- so a CLONE engine drew a
    # different seed for every line one character speaks, and the operator heard
    # a character change voice between two of his own beats. A profile that sets
    # this True seeds on the CHARACTER and his resolved reference instead, so his
    # identity survives his own dialogue.
    #
    # DEFAULT False, AND THAT IS THE SAFETY. Every profile that does not opt in
    # keeps the legacy formula byte for byte -- announcer profiles included,
    # which are deliberately left alone: an announcer is not a cloned character
    # and has no identity to hold steady across beats. Opted in today: the three
    # char_* CLONE profiles (indextts2, chatterbox, dia).
    character_stable_seed: bool = False

    @model_validator(mode="after")
    def _validate_metadata(self):
        if self.runtime not in _VALID_RUNTIMES:
            raise ValueError(
                f"profile '{self.profile_id}': runtime '{self.runtime}' not in "
                f"{sorted(_VALID_RUNTIMES)}"
            )
        if self.license_state not in _VALID_LICENSE_STATES:
            raise ValueError(
                f"profile '{self.profile_id}': license_state "
                f"'{self.license_state}' not in {sorted(_VALID_LICENSE_STATES)}"
            )
        if self.error_policy not in _VALID_ERROR_POLICIES:
            raise ValueError(
                f"profile '{self.profile_id}': error_policy "
                f"'{self.error_policy}' not in {sorted(_VALID_ERROR_POLICIES)}"
            )
        # cloud runtime demands the fail-loud contract + a partner row.
        if self.runtime == "cloud":
            if self.error_policy != "fail_loud":
                raise ValueError(
                    f"profile '{self.profile_id}': runtime=cloud requires "
                    f"error_policy=fail_loud (no silent local fallback)")
            if not self.partner_row:
                raise ValueError(
                    f"profile '{self.profile_id}': runtime=cloud requires a "
                    f"partner_row (the pinned partner_nodes.yaml key)")
        # direct_api runtime is explicit-selection-only BYO-provider access. It
        # must be fail-loud, authenticated, and must not claim a Partner row.
        if self.runtime == "direct_api":
            if self.error_policy != "fail_loud":
                raise ValueError(
                    f"profile '{self.profile_id}': runtime=direct_api requires "
                    f"error_policy=fail_loud (no silent local fallback)")
            if not self.auth_required:
                raise ValueError(
                    f"profile '{self.profile_id}': runtime=direct_api requires "
                    f"auth_required=true")
            if self.partner_row:
                raise ValueError(
                    f"profile '{self.profile_id}': runtime=direct_api requires "
                    f"partner_row='' (direct APIs are not Partner nodes)")
        return self


def effective_license_state(profile: "EngineProfile") -> str:
    """The profile's license state, deriving from ``commercial_clean`` when the
    explicit ``license_state`` is blank: clean iff commercial_clean else gated.
    """
    if profile.license_state:
        return profile.license_state
    return "clean" if profile.commercial_clean else "gated"


def gate_state(profile: "EngineProfile") -> str:
    """Three-state commercial gate for a profile: 'clean' | 'warn' | 'stop'.

    Mirrors :mod:`nodes._otr_release_gate`: clean ships silently, gated ('warn')
    renders with a non-blocking notice, unknown ('stop') is fail-closed.
    """
    state = effective_license_state(profile)
    if state == "clean":
        return "clean"
    if state == "gated":
        return "warn"
    return "stop"


def engine_warning(profile: "EngineProfile") -> str:
    """The non-blocking notice to surface for a gated profile at episode start,
    or '' when the engine is clean. A 'stop'-state engine yields no warning --
    it is refused by the resolver upstream, not warned-and-shipped.
    """
    if gate_state(profile) == "warn":
        return profile.warn_text or (
            f"{profile.engine}: known-gated (license_state=gated) -- "
            f"non-blocking notice; verify before commercial release."
        )
    return ""


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

    def rank_chain(self, role: str) -> List["EngineProfile"]:
        """LOCAL profiles serving ``role`` sorted by ``rank`` ascending (lowest
        rank = highest priority), ``profile_id`` breaking ties for stability.

        CLOUD and DIRECT-API profiles are EXCLUDED: this chain is the AUTOMATIC
        auto-selection / fallback ladder, and paid/authenticated external
        engines must never be picked automatically. They are reachable ONLY by
        explicit dropdown/profile selection, which routes through
        ``resolve_casting_plan`` (``profile_for``), not this chain.
        """
        local = [
            p for p in self.for_role(role)
            if p.runtime not in ("cloud", "direct_api")
        ]
        return sorted(local, key=lambda p: (p.rank, p.profile_id))


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
    if profile.requires_hf_token and not otr_env.get("HF_TOKEN"):
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
