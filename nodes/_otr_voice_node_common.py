"""Shared dispatch core for the v2 voice nodes (Wave 1 / 1a + 1b).

OTR_BatchCharacterVoices (1a) and OTR_AnnouncerVoice (1b) share ONE dispatch
contract: pick an engine from the registry, FAIL CLOSED, and render per line into
the existing Bark AUDIO-batch contract -- with engine teardown in ``finally``
BEFORE the ``done`` signal (I-7). The legacy batch-delegation path was retired in
the audio clean-break (1c); every audio engine is now self-contained per_line /
clip. This module holds that core so each node file declares only its role, its
INPUT_TYPES, and its output names.

The theme node (1c) is NOT built on this base: it has three AUDIO outputs and a
``clip`` interface, so it is self-contained.

Import-time is side-effect-free; engine libraries are lazy-imported INSIDE
``generate`` (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import _otr_voice_route as _ROUTE

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Voice-CLONING engines that REQUIRE a per-character reference WAV declare it via
# adapter metadata (base.AudioEngineAdapter): requires_voice_ref=True. When such an
# engine is selected but a char_voice line has no usable voice_ref (no clip ref
# assigned, or no reference clips installed on disk), the per-line path FAILS LOUD
# with a named EngineUnusable(MISSING_MODEL) -- NO bark fallback (no-fallback rip,
# operator 2026-07-03). A missing reference is a casting/install defect to fix, not
# something to paper over. Branching on metadata (not a hard-coded engine-name
# tuple) means a NEW cloning engine slots in by setting requires_voice_ref on its
# adapter, with ZERO changes here.
def _engine_requires_voice_ref(adapter) -> bool:
    """True iff ``adapter`` is a voice-cloning engine needing a reference WAV."""
    return bool(getattr(adapter, "requires_voice_ref", False))


def _resolve_ref_to_disk(ref_path):
    """Resolve a voice-bank ref_path (usually relative to the ComfyUI root, e.g.
    'models/TTS/refs/indextts2/ix_male_warm.wav') to an absolute path.

    DELEGATES to the ONE shared resolver (Lemmy chunk B). It used to be a
    SECOND, broader implementation of the same question -- this one knew about
    the migrated ``C:\\ComfyUI-Models`` root and the adapters' private copies did
    not, so this existence check could confirm a reference the worker then could
    not open. The docstring even said it "mirrors the indextts2 adapter's own
    resolution", which had stopped being true. One resolver, one answer.

    Keeps ``None`` for an empty ref, because callers here distinguish "nothing
    asked for" from "asked for and not found"."""
    if not ref_path:
        return None
    from ._otr_audio_engines.base import resolve_voice_ref_path
    return resolve_voice_ref_path(ref_path)


def _provisional_identity_fingerprint(engine, voice_ref_id):
    """Cache-key material for ONE provisional identity, or ``None`` to fail open.

    A provisional row deliberately carries no ``voice_route`` -- that field means
    "a qualified route was proved" -- so the route-shaped fingerprint above cannot
    see it, and a swapped reference under an unchanged id would replay the
    PREVIOUS render's audio while the ledger named the new identity.

    Both LOCAL identity kinds are fingerprinted by their bytes: a clone engine's
    reference WAV and kokoro's ``.pt`` voice tensor alike. Hashing only the WAV
    would leave the one identity kind that neither the qualified route nor the
    clone rows cover silently cacheable.

    ``None`` means "expected a local file and could not read it", and the caller
    turns that into NaN -- fail OPEN, rerun, and let the render path fail loudly,
    rather than quietly serving audio nobody asked for. A provider voice has no
    local bytes and is not a failure: it contributes its id and nothing else.
    """
    try:
        from ._otr_voice_bank import load_voice_bank

        bank, _sha = load_voice_bank()
    except Exception:                         # noqa: BLE001 -- unreadable bank
        return None
    entry = next((e for e in bank
                  if e.voice_ref_id == voice_ref_id and e.engine == engine), None)
    if entry is None:
        return None                           # the id names nothing -- fail open
    ref_path = str(getattr(entry, "ref_path", "") or "")
    if not ref_path or ref_path.startswith("cloud:"):
        # NEVER a network call. A provider voice is identified by its id, which
        # the caller already folded in.
        return "provider:%s" % (getattr(entry, "provider_voice_id", "") or "",)
    full = _resolve_ref_to_disk(ref_path) or ref_path
    digest = _ROUTE.sha256_of_file(full)
    return digest


def _resolve_clone_ref_path(engine, cast, episode_seed, role="char_voice"):
    """Best-effort on-disk reference WAV for a cloning engine + cast row, or None.

    preserve_ledger does not stamp a clip ref, and CastLock stamps only
    voice_ref_id (not the path), so the per-line path can arrive with no
    voice_ref_path even when a bank ref applies. Resolve it here: prefer the
    cast's voice_ref_id, else assign one deterministically by gender. Returns an
    absolute path ONLY if the file exists on disk; None -> the caller FAILS LOUD
    (no-fallback rip: a cloning engine with no resolvable ref raises, never bark).
    Never raises itself (a bad bank just yields None -> the caller's raise)."""
    try:
        from ._otr_voice_bank import (
            assign_voice_for_slot, filter_by_quality_tier, load_voice_bank,
        )

        bank, _ = load_voice_bank()
        # Whiny-fix P2c (G16): the render-time fallback consumes the SAME
        # tier-filtered pool as the caster, so audited rejects can never leak
        # through this route. Un-audited banks pass through unchanged.
        bank = tuple(filter_by_quality_tier(bank))
    except Exception:  # noqa: BLE001
        return None
    entry = None
    vrid = cast.get("voice_ref_id")
    if vrid:
        entry = next(
            (e for e in bank if e.voice_ref_id == vrid and e.engine == engine), None
        )
    if entry is None:
        # SYNONYM-CANONICALIZED (item 8, 2026-08-06). This compared the RAW
        # stored value against the bank's vocabulary, so a row recorded as
        # `woman`, `man`, `m` or `f` matched nothing and fell through to the
        # gender-agnostic path -- a correctly-gendered character silently
        # getting a voice of any gender.
        #
        # Deliberately NOT the tri-state normalize_gender: the bank has its own
        # vocabulary and carries a `neutral` reference (el_river). Collapsing
        # `neutral` into `other` would skip the one voice that fits those rows.
        # Blank also stays blank so the guard below still short-circuits.
        from ._otr_roster_gender import canonical_bank_gender
        gender = canonical_bank_gender(cast.get("gender"))
        if gender:
            try:
                entry = assign_voice_for_slot(
                    role=role, engine=engine,
                    char_id=str(cast.get("char_id") or ""), gender=gender,
                    timbre=tuple(cast.get("timbre") or ()),
                    age_band=str(cast.get("age_band") or ""),
                    episode_seed=episode_seed, allow_voice_reuse=True, bank=bank,
                )
            except Exception:  # noqa: BLE001 -- gender unservable; gender-agnostic below
                entry = None
        if entry is None:
            # Gender-agnostic last resort: an empty or out-of-bank gender (e.g.
            # the writer emitting gender='other'/'unspecified') has no same-gender
            # reference, but a clone engine must still get a REAL voice rather
            # than silently dropping to bark (PD1 + the index-only goal).
            #
            # This branch is now the SECOND reader of that draw. CastLock calls
            # the same helper and stamps its result as voice_ref_id, so a fresh
            # episode normally takes the vrid lookup above. This stays reachable
            # for rows CastLock did not stamp -- preserve_ledger, legacy ledgers
            # frozen before the stamp existed, and any render whose engine
            # differs from the one CastLock resolved. Do NOT narrow it to a
            # gender-filtered draw: by definition nothing here matches the
            # gender, and do not re-implement the choice locally either, or the
            # ledger will name a voice this path does not open.
            from ._otr_voice_bank import gender_agnostic_fallback_ref
            entry = gender_agnostic_fallback_ref(
                bank, engine=engine, char_id=str(cast.get("char_id") or ""),
                episode_seed=episode_seed, role=role,
            )
    path = _resolve_ref_to_disk(getattr(entry, "ref_path", "") or "")
    return path if (path and os.path.exists(path)) else None


def _resolve_provider_voice_id(engine, cast, episode_seed, role="char_voice"):
    """The provider voice_id for a CLOUD voice engine (e.g. elevenlabs) + cast row
    when CastLock stamped none -- i.e. the operator changed only the VOICE ENGINE
    and left the CastLock voice_bank on a LOCAL bank (so the cast carries a local
    voice_ref_id, not a provider id). Resolve a gender-matched voice from the
    ENGINE's OWN pool here, the SAME deterministic per-character CASTING the
    clone-ref path does -- so the voice engine is the SINGLE knob (parity with how
    every LOCAL engine already shares the default bank). This is casting a REAL
    per-character voice, NOT a fallback: no voice is inherited from another
    character/engine, and an engine with NO pool still yields None -> the adapter
    fails loud. Never raises."""
    # Google TTS has a stricter voice-quality/no-fallback contract: CastLock must
    # stamp the exact provider_voice_id so announcer separation and gender-aware
    # assignment have already been enforced. Do not invent one at render time.
    if engine == "google_tts":
        return None
    try:
        from ._otr_voice_bank import (
            assign_voice_for_slot, filter_by_quality_tier, load_voice_bank,
        )
        bank, _ = load_voice_bank()
        bank = tuple(filter_by_quality_tier(bank))
    except Exception:  # noqa: BLE001
        return None
    entry = None
    vrid = cast.get("voice_ref_id")
    if vrid:
        entry = next(
            (e for e in bank if e.voice_ref_id == vrid and e.engine == engine), None)
    if entry is None:
        # SYNONYM-CANONICALIZED (item 8, 2026-08-06). This compared the RAW
        # stored value against the bank's vocabulary, so a row recorded as
        # `woman`, `man`, `m` or `f` matched nothing and fell through to the
        # gender-agnostic path -- a correctly-gendered character silently
        # getting a voice of any gender.
        #
        # Deliberately NOT the tri-state normalize_gender: the bank has its own
        # vocabulary and carries a `neutral` reference (el_river). Collapsing
        # `neutral` into `other` would skip the one voice that fits those rows.
        # Blank also stays blank so the guard below still short-circuits.
        from ._otr_roster_gender import canonical_bank_gender
        gender = canonical_bank_gender(cast.get("gender"))
        if gender:
            try:
                entry = assign_voice_for_slot(
                    role=role, engine=engine,
                    char_id=str(cast.get("char_id") or ""), gender=gender,
                    timbre=tuple(cast.get("timbre") or ()),
                    age_band=str(cast.get("age_band") or ""),
                    episode_seed=episode_seed, allow_voice_reuse=True, bank=bank,
                )
            except Exception:  # noqa: BLE001 -- gender unservable; gender-agnostic below
                entry = None
        if entry is None:
            # THIRD copy of the same gender-agnostic draw, now folded onto the
            # one selector the caster and the clone-ref path already share.
            # It differed only in its seed suffix ('_provid' rather than
            # '_anyref'), which is exactly the shape that lets two code paths
            # name two different voices for one character -- the defect the
            # shared selector was extracted to end. Same pool, same ordering,
            # same char_id keying; the suffix is deliberately dropped so a
            # cloud row and its local twin resolve to the SAME bank entry.
            from ._otr_voice_bank import gender_agnostic_fallback_ref
            entry = gender_agnostic_fallback_ref(
                bank, engine=engine, char_id=str(cast.get("char_id") or ""),
                episode_seed=episode_seed, role=role,
            )
    return getattr(entry, "provider_voice_id", "") or None


# --------------------------------------------------------------------------- #
# Pure helpers (no IO at import; engines are self-contained per_line / clip)
# --------------------------------------------------------------------------- #
def coerce_int_seed(val) -> int:
    """Reduce a ledger ``episode_seed`` (often a string) to a stable int.

    Numeric strings pass through; anything else folds through the shared
    sha256->int reducer so a textual seed still yields a stable value.
    """
    from ._otr_resolved_request import _seed_to_int64

    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    s = str(val or "")
    if s.lstrip("-").isdigit():
        try:
            return int(s)
        except ValueError:
            pass
    return _seed_to_int64("episode_seed", s)


def build_engine_combo(role, fallback) -> list:
    """C-5 safe engine list for INPUT_TYPES: never empty, stable order, no IO.

    Uses the hardcoded legacy-first ``fallback`` and only consults the (pure)
    profiles helper opportunistically. Index 0 is the legacy/byte-identical
    default for the role.
    """
    engines = list(fallback)
    try:
        from ._otr_engine_profiles import legacy_first_engines

        got = legacy_first_engines(role)
        if got:
            engines = list(got)
    except Exception:  # noqa: BLE001 -- INPUT_TYPES must never crash (C-5)
        engines = list(fallback)
    if not engines:
        engines = list(fallback)
    return engines


def _voice_device_from_ledger(ledger_json, script_json="") -> str:
    """The explicit voice device from the CastLock ledger stamp
    (``meta.voice_device``, S4). Falls back to the script ledger's meta,
    then the nv50 baseline ``cuda``. NEVER probes hardware -- the stamp is
    the single source of truth; a wrong device fails loud at the adapter."""
    import json as _json

    for raw in (ledger_json, script_json):
        if not raw:
            continue
        try:
            led = _json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(led, dict):
            dev = ((led.get("meta") or {}).get("voice_device") or "")
            dev = str(dev).strip().lower()
            if dev in ("cuda", "cpu", "mps"):
                return dev
    return "cuda"


def voice_input_types(role, fallback) -> dict:
    """The shared INPUT_TYPES for a v2 voice node (1a / 1b).

    forceInput sockets carry no widget; the only serialized widget is
    ``engine`` (``stereo_policy`` is no longer surfaced -- single option
    "mono_safe"; the ``generate()`` kwarg still defaults to "mono_safe"); there
    is no ``seed``-named widget and no ``model_id`` widget (CLAUDE.md rule 6).
    """
    engines = build_engine_combo(role, fallback)
    return {
        "required": {
            "script_json": ("STRING", {
                "multiline": True,
                "default": "[]",
                "forceInput": True,
                "tooltip": (
                    "Frozen v2 ledger JSON from OTR_LedgerFreezeCascade "
                    "(node 62 slot 1). Passed VERBATIM to the legacy engine on "
                    "the byte-identical batch path."
                ),
            }),
            "engine": (engines, {
                "default": engines[0],
                "tooltip": (
                    "Voice engine for this role. The legacy engine is the "
                    "byte-identical default; opt-in engines are flag-gated "
                    "until the GPU dependency pilot promotes them. Unusable "
                    "selections fail closed with a named error at queue time."
                ),
            }),
        },
        "optional": {
            "ledger_json": ("STRING", {
                "multiline": True,
                "default": "",
                "forceInput": True,
                "tooltip": (
                    "Cast-locked ledger from OTR_CastLock (carries "
                    "voice_ref_id / episode_seed). Read on the per-line path; "
                    "the batch path ignores it and delegates the raw "
                    "script_json. No episode_seed widget (I-4)."
                ),
            }),
            "gate_in": ("STRING", {
                "multiline": True,
                "default": "",
                "forceInput": True,
                "tooltip": (
                    "Optional ordering signal. Wire an upstream 'done' here to "
                    "force this node to run after it. Optional on the first "
                    "chain node; the audio chain binds it node->node."
                ),
            }),
        },
    }


# --------------------------------------------------------------------------- #
# Cloud-audio-cache chunk 2 (2026-08-08): per-episode content-addressed cache
# helpers. All are cache-off-safe -- they only run when the resolved profile
# has ``use_cache=True`` (today: the two Google TTS profiles only). Local paths
# are byte-identical to the pre-chunk-2 render.
# --------------------------------------------------------------------------- #
def _audio_cache_dir_for(meta: dict) -> str:
    """Return the per-episode cache dir, or "" if none resolves.

    Precedence: OTR_AUDIO_CACHE_DIR env override (essential for tests) ->
    <meta.paths.audio_dir>/audio_cache/ -> "". A "" return with cache
    enabled is treated as a config error by the caller (fail loud).
    """
    override = otr_env.get("OTR_AUDIO_CACHE_DIR", "").strip()
    if override:
        resolved = os.path.abspath(os.path.expanduser(override))
        log.info("[OTR voice cache] using OTR_AUDIO_CACHE_DIR: %s", resolved)
        return resolved
    paths = (meta or {}).get("paths") or {}
    audio_dir = paths.get("audio_dir") or ""
    if audio_dir:
        cache_dir = os.path.abspath(os.path.join(audio_dir, "audio_cache"))
        # THE CACHE DIR MUST LAND IN THE OUTPUT TREE (2026-09-05). `meta` is
        # parsed from the `ledger_json` / `script_json` workflow STRING, so
        # `meta.paths.audio_dir` is caller-chosen, and the cache writer below
        # `os.makedirs` it and `os.replace`s files into it. The env override
        # above is operator configuration and is deliberately NOT confined --
        # same rule as every other env knob in this pack. A refusal returns ""
        # here, which the caller already treats as "no cache dir resolved".
        try:
            from ._otr_paths import confine_to_output_tree
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_paths import confine_to_output_tree  # type: ignore
        try:
            confine_to_output_tree(cache_dir, "meta.paths.audio_dir")
        except Exception as exc:  # noqa: BLE001 -- a refusal disables the cache, never raises
            log.warning("[OTR voice cache] audio_dir refused (%s); no cache dir", exc)
            return ""
        return cache_dir
    return ""


def _resolve_voice_ref_early(adapter, engine, cast, episode_seed, role, voice_ref):
    """Consolidate the branches at ``_render_per_line:566-591`` into ONE call
    returning ``(voice_ref_for_call, id_for_key)``.

    Used only when cache is enabled so the RESOLVED identity enters the cache
    key (r1 MF#3 rationale). The non-cache path preserves today's inline
    resolution order byte-identically.
    """
    ref_field = getattr(adapter, "voice_ref_field", "voice_ref_path")
    if _engine_requires_voice_ref(adapter) and voice_ref:
        _disk = _resolve_ref_to_disk(voice_ref)
        if not (_disk and os.path.exists(_disk)):
            voice_ref = None
    if _engine_requires_voice_ref(adapter) and not voice_ref:
        voice_ref = _resolve_clone_ref_path(engine, cast, episode_seed, role=role)
    if ref_field == "provider_voice_id" and not voice_ref:
        voice_ref = _resolve_provider_voice_id(engine, cast, episode_seed, role=role)
    id_for_key = str(voice_ref or (cast or {}).get("voice_ref_id") or "")
    return voice_ref, id_for_key


# --------------------------------------------------------------------------- #
# ONE PER-LINE RUNTIME CONTEXT (voice-identity fix 2026-08-18, PBUG-20260817-09)
#
# WHY THIS EXISTS [QA-2]. Six surfaces describe one line's render: the cache-key
# params, the engine seed, the outbound worker payload, the P-OBS receipt, the
# cap metrics and the ledger stamp. Each used to resolve its own values from its
# own source at its own moment -- and every defect this fix closes is two of
# those surfaces disagreeing. The alpha keyed at request-build time while the
# forward read the env again at generate time; the receipt described a vector
# the adapter had not yet sanitized; the seed was derived before the reference
# it should depend on had been resolved. One object, resolved once per line and
# read by all of them, cannot drift.
#
# TWO PHASES, ON PURPOSE. `begin` runs BEFORE the request is built, because the
# cache key needs the params. `resolve_seed` runs AFTER reference resolution,
# because the character seed must key on the reference the adapter will actually
# clone -- not on the blank the request may have carried [QA-5].
# --------------------------------------------------------------------------- #

#: Seed policies. ``line_v1`` is the legacy formula, preserved EXACTLY as
#: ``_seed_to_int64(engine, request.stable_line_seed)``, so every profile that
#: does not opt in renders byte-identically [QA-1]. ``char_v1`` is the new
#: character-stable derivation the char_* clone profiles opt into [QA-6].
SEED_POLICY_LINE = "line_v1"
SEED_POLICY_CHARACTER = "char_v1"


@dataclass
class _LineRuntime:
    """Every resolved per-line value that more than one surface reads."""

    params: dict = field(default_factory=dict)
    alpha: object = None
    emotion: object = None
    vector_state: str = "omitted"
    ref_identity: str = ""
    seed_policy: str = SEED_POLICY_LINE
    character_seed_enabled: bool = False
    engine_seed: int = 0

    @property
    def effective_mass(self):
        """Emotion mass the vendor will actually spend, or None if it has none."""
        return None if not self.emotion else self.emotion.get("effective_mass")

    @property
    def mass_capped(self) -> bool:
        return bool(self.emotion and self.emotion.get("mass_capped"))


def _safe_vector_state(delivery_vector) -> str:
    """``omitted`` / ``zero`` / ``nonzero`` without trusting the stamped values.

    The engine-agnostic floor for adapters that expose no emotion payload of
    their own. A delivery vector is hand-editable JSON, so this must survive a
    string, a ``None`` or a NaN where a number belongs -- the previous inline
    ``float(v or 0.0)`` did not, and raised out of the observability line on an
    out-of-contract ledger [QA-4]. THE LAW: a render degrades, never raises.
    """
    if delivery_vector is None:
        return "omitted"
    if not isinstance(delivery_vector, dict):
        return "zero"
    for value in delivery_vector.values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number > 0.0:
            return "nonzero"
    return "zero"


def _begin_line_runtime(adapter, profile, delivery_vector) -> _LineRuntime:
    """Phase one: resolve the params, the alpha and the emotion payload ONCE.

    CALL-TIME NUMERIC PARAMS JOIN THE KEY (Lemmy chunk A1). ``default_params``
    is captured at request-BUILD time, so an adapter resolving a knob at
    GENERATE time moved the render without moving the key. ``render_time_params``
    is merged last so the live value beats a stale profile default of the same
    name. Engines with no such knob contribute ``{}`` and key exactly as before.
    """
    params = dict(getattr(profile, "default_params", None) or {})
    try:
        params.update(adapter.render_time_params() or {})
    except Exception:  # noqa: BLE001 -- a duck-typed adapter without the hook
        pass           # keys exactly as it did before the hook existed

    runtime = _LineRuntime(params=params)

    # THE SEED POLICY IS RESOLVED HERE, WITH THE PARAMS, SO IT KEYS.
    # `OTR_VOICE_CHARACTER_SEED=0` forces the legacy per-line seed on a profile
    # that opted in -- the control arm of the 2x2 proof, and the operator's
    # one-line rollback. It mirrors `OTR_DELIVERY_VECTOR=0` above, which this
    # dispatch already documents as "a TRUE old path".
    #
    # FOLDED INTO THE REQUEST PARAMS ON PURPOSE. A knob that changes the RENDER
    # and not the KEY is the Lemmy chunk A1 defect: the next identical line
    # replays audio made under the other setting while its receipt describes
    # this one. Only profiles that opted into character seeding contribute the
    # param at all, so every other engine keys byte-identically to before.
    if bool(getattr(profile, "character_stable_seed", False)):
        runtime.character_seed_enabled = (
            otr_env.get("OTR_VOICE_CHARACTER_SEED", "1") != "0")
        runtime.params["voice_seed_policy"] = (
            SEED_POLICY_CHARACTER if runtime.character_seed_enabled
            else SEED_POLICY_LINE)

    # The adapter that owns an emotion blend hands back the EXACT arguments it
    # will send, so the receipt below cannot describe a blend the worker never
    # got. Adapters without the hook keep the engine-agnostic state only.
    payload_fn = getattr(adapter, "emotion_payload", None)
    if callable(payload_fn):
        try:
            runtime.emotion = payload_fn(delivery_vector)
        except Exception as exc:  # noqa: BLE001 -- observability is never fatal
            log.debug("[OTR] emotion payload unavailable: %s", exc)
    if runtime.emotion:
        runtime.alpha = runtime.emotion.get("emo_alpha")
        runtime.vector_state = runtime.emotion.get("vector_state") or "omitted"
        # The context is the single authority: the KEY records the same alpha
        # the forward spends, by construction rather than by coincidence [QA-2].
        if "emo_alpha" in runtime.params:
            runtime.params["emo_alpha"] = runtime.alpha
        if "emo_mass_cap" in runtime.params:
            runtime.params["emo_mass_cap"] = runtime.emotion.get("emo_mass_cap")
    else:
        alpha_fn = getattr(adapter, "current_emo_alpha", None)
        if callable(alpha_fn):
            try:
                runtime.alpha = alpha_fn()
            except Exception:  # noqa: BLE001
                runtime.alpha = None
        runtime.vector_state = _safe_vector_state(delivery_vector)
    return runtime


def _durable_reference_identity(voice_ref_id, voice_ref) -> str:
    """The character's reference identity, stable across boxes and renders.

    Prefers the bank's ``voice_ref_id``; falls back to the resolved file's BASE
    NAME. Never the absolute path -- the models root differs per machine (it is
    ``C:/ComfyUI-Models`` on this box, not the repo tree), and a seed derived
    from an absolute path would render one character two ways on two boxes,
    which is the very defect this seed exists to end.
    """
    identity = str(voice_ref_id or "").strip()
    if identity:
        return identity
    if voice_ref:
        return os.path.basename(str(voice_ref))
    return ""


def _resolve_engine_seed(runtime, seed_reduce, profile, engine, request,
                         char_id, episode_seed, cast_lock_revision,
                         voice_ref_id, voice_ref) -> None:
    """Phase two: the per-engine external seed, AFTER reference resolution.

    THE DEFECT, IN ONE LINE. ``stable_line_seed`` includes ``line_id``, so every
    line of one character drew a DIFFERENT engine seed. On a clone engine with a
    live emotion blend that is audible: the operator heard NAG's second beat as
    a different actor from his first. A character's identity must survive his
    own dialogue.

    ``char_v1`` keys the seed on WHO IS SPEAKING and WHAT VOICE HE WAS CAST
    WITH -- episode, cast revision, char_id, resolved reference -- so every line
    he speaks is drawn from one voice, while a different character, a re-cast or
    a different episode still moves it.

    ``line_v1`` IS PRESERVED EXACTLY [QA-1]: it is the whole legacy expression
    ``_seed_to_int64(engine, request.stable_line_seed)``, not the raw
    ``stable_line_seed``, and it is still what every profile that has not opted
    in -- every blank ``char_id``, and every leg booted with
    ``OTR_VOICE_CHARACTER_SEED=0`` -- receives, byte for byte.

    The opt-in decision itself was resolved in phase one, with the params, so
    the value that moved the render also moved the key. This reads it rather
    than asking the environment a second question that could answer differently.
    """
    legacy_seed = seed_reduce(engine, request.stable_line_seed)
    runtime.ref_identity = _durable_reference_identity(voice_ref_id, voice_ref)

    if not runtime.character_seed_enabled:
        runtime.seed_policy = SEED_POLICY_LINE
        runtime.engine_seed = legacy_seed
        return
    if not str(char_id or "").strip():
        # A line with no character cannot have a character-stable seed. The
        # legacy formula keeps it renderable AND keeps it honest -- seeding
        # every anonymous line alike would collapse them onto one voice.
        runtime.seed_policy = SEED_POLICY_LINE
        runtime.engine_seed = legacy_seed
        return

    runtime.seed_policy = SEED_POLICY_CHARACTER
    runtime.engine_seed = seed_reduce(
        "char_voice_seed_v1", engine, int(episode_seed),
        int(cast_lock_revision), str(char_id), runtime.ref_identity,
    )


def _persist_ledger_stamps(meta, stamps, log_, failed_line_ids=None) -> int:
    """Reload the on-disk ledger, stamp each line, save.

    Returns count of degraded stamps (a stamp helper False or a
    save_ledger_safe False counts). Never writes back the wire JSON --
    reload-before-save preserves prior roles' stamps (r2 MF#4).

    ``failed_line_ids`` (plan 5.3) is an optional set the caller passes in to be
    filled with the line_ids that did NOT persist. The COUNT alone cannot answer
    "did the qualified route's own receipt land?" -- and punishing a proved route
    because some unrelated line's stamp failed would throw away good, fully
    evidenced audio. Which lines failed is the question; this answers it.
    """
    from ._otr_ledger import (
        in_flight_ledger_path, save_ledger_safe, stamp_per_line_audio_meta)

    paths = (meta or {}).get("paths") or {}
    ledger_path = paths.get("ledger_path") or ""
    # THE LEDGER TO REWRITE IS THE ONE THIS RUN OPENED, NOT THE ONE THE WIRE
    # NAMED (2026-09-05). `meta` is parsed from the `ledger_json` / `script_json`
    # workflow STRING, so `meta.paths.ledger_path` is caller-chosen -- and it
    # went straight to `save_ledger_safe`, which writes a temp file beside the
    # target and `os.replace`s over it. That is an arbitrary-JSON-overwrite from
    # an unauthenticated /prompt. The in-flight singleton knows the real path by
    # construction and advances through `rename_episode`, so preferring it is
    # also more correct than trusting a value the wire could have gone stale on.
    # The wire value is kept only as the headless fallback (no singleton), and
    # then only when it is inside the output tree.
    in_flight = in_flight_ledger_path()
    if in_flight is not None:
        if ledger_path and str(in_flight) != str(ledger_path):
            log_.debug(
                "[OTR voice cache] meta.paths.ledger_path %r ignored; stamping "
                "the in-flight ledger %s", ledger_path, in_flight)
        ledger_path = str(in_flight)
    elif ledger_path:
        try:
            from ._otr_paths import confine_to_output_tree
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_paths import confine_to_output_tree  # type: ignore
        try:
            ledger_path = confine_to_output_tree(ledger_path, "meta.paths.ledger_path")
        except Exception as exc:  # noqa: BLE001 -- a refusal skips stamping, never raises here
            log_.warning("[OTR voice cache] ledger_path refused (%s); stamps skipped", exc)
            ledger_path = ""
    def _mark_all_failed() -> None:
        if failed_line_ids is not None:
            failed_line_ids.update(lid for lid, _ in stamps)

    if not ledger_path or not os.path.exists(ledger_path):
        log_.warning("[OTR voice cache] no ledger_path in meta.paths; stamps skipped")
        _mark_all_failed()
        return len(stamps)
    degraded = 0
    try:
        with open(ledger_path, "r", encoding="utf-8") as fh:
            full_ledger = json.load(fh)
        for lid, fields in stamps:
            if not stamp_per_line_audio_meta(full_ledger, lid, **fields):
                degraded += 1
                if failed_line_ids is not None:
                    failed_line_ids.add(lid)
                log_.warning("[OTR voice cache] stamp failed for line %s", lid)
        if not save_ledger_safe(Path(ledger_path), full_ledger):
            # Nothing reached disk, so every stamp failed -- including any that
            # the per-line loop above had reported as fine.
            degraded = len(stamps)
            _mark_all_failed()
            log_.warning("[OTR voice cache] save_ledger_safe returned False")
    except Exception as exc:  # noqa: BLE001
        log_.warning("[OTR voice cache] ledger stamp failed: %s", exc)
        degraded = len(stamps)
        _mark_all_failed()
    return degraded


# --------------------------------------------------------------------------- #
# Shared node base
# --------------------------------------------------------------------------- #
class OTRVoiceNodeBase:
    """Dispatch core shared by the v2 character + announcer voice nodes.

    Subclasses set the role config (``ROLE``, ``LINE_ROLES``, ``DONE_PREFIX``,
    ``LEGACY_FIRST_FALLBACK``) and declare ``INPUT_TYPES`` / ``RETURN_TYPES`` /
    ``RETURN_NAMES`` / ``FUNCTION`` / ``CATEGORY``. The node FUNCTION is
    ``generate`` (inherited).
    """

    ROLE: str = ""
    LINE_ROLES: tuple = ("character",)
    DONE_PREFIX: str = "voice"
    LEGACY_FIRST_FALLBACK: tuple = ()

    @classmethod
    def IS_CHANGED(cls, script_json="", engine="", ledger_json="", gate_in="",
                   stereo_policy="mono_safe", **_kw):
        """Cache-enabled legs rerun (NaN); local legs return a FINGERPRINT.

        The old local answer was the constant string ``"static"``, which told
        ComfyUI that a local voice leg never changes. For a qualified route that
        is false in the one way that matters: swap the reference WAV under a pin,
        or move the route to a new contract version, and ``"static"`` would serve
        the previous render's audio while the ledger claimed the new route. The
        fingerprint below is over the route identity, the active profile/runtime
        fields, and -- for ``local_wav`` -- the actual reference BYTES.

        Three rules this obeys, all of them load-bearing:

        * **A ledger with no routes fingerprints to the literal ``"static"``.**
          Not "something stable" -- the same string as before, so every shipping
          local render keeps its exact in-graph caching behaviour.
        * **NEVER a network call.** ``provider_voice`` rows contribute route id,
          provider, voice and runtime values only; nothing is fetched, and no
          cloud URI is treated as a file to hash.
        * **An unreadable expected local file returns NaN.** Failing OPEN on a
          missing reference is the Bug Bible unavailable-input rule: rerun and
          let the render path fail loudly, rather than quietly reusing audio.
        """
        try:
            from ._otr_engine_profiles import require_resolver

            profile = require_resolver().resolve_casting_plan(role=cls.ROLE, engine=engine)
        except Exception:
            return float("nan")
        if getattr(profile, "use_cache", False):
            return float("nan")

        try:
            from . import _otr_ledger_consumers as _OTRLC

            source = ledger_json if (ledger_json or "").strip() else script_json
            led = _OTRLC.load_ledger(source)
            routes = [
                e.get("voice_route") for e in (led.get("cast") or [])
                if isinstance(e, dict) and isinstance(e.get("voice_route"), dict)
                and e.get("voice_route")
            ]
            provisional_rows = [
                e for e in (led.get("cast") or [])
                if isinstance(e, dict)
                and str(e.get(_ROUTE.CAST_ROW_TIER_FIELD) or "")
                == _ROUTE.ROUTE_TIER_PROVISIONAL
            ]
        except Exception:
            # No ledger yet, or one that will not parse. This is the ORDINARY
            # case at graph-eval time (upstream has not run), and it was
            # "static" before this method learned about routes -- so it stays
            # "static". Returning NaN here would make every local voice leg
            # uncacheable in-graph, which is a performance regression dressed up
            # as caution. Route safety is enforced on the render path, which
            # fails closed regardless of what this method answers.
            routes = []
            provisional_rows = []

        # CALL-TIME NUMERIC PARAMS (Lemmy chunk A1), and they are read BEFORE
        # the no-routes shortcut because the defect has nothing to do with
        # routes. `OTR_INDEXTTS2_EMO_ALPHA` changes what a local indextts2 leg
        # renders, so answering "static" for that leg tells ComfyUI the render
        # can never change when it just did.
        #
        # EMPTY IS THE COMMON CASE and it preserves everything: an engine with
        # no such knob contributes nothing, so a no-routes ledger still returns
        # the literal "static" and every shipping local render keeps its exact
        # in-graph caching behaviour. Only an engine that actually resolves a
        # knob at render time stops being "static" -- which is the truth about
        # that engine, and the whole point.
        render_params = {}
        try:
            from ._otr_audio_engines import get_engine as _get_engine
            render_params = _get_engine(engine).render_time_params() or {}
        except Exception:  # noqa: BLE001 -- unknown/duck-typed engine
            render_params = {}

        if not routes and not render_params and not provisional_rows:
            return "static"

        parts = [
            str(getattr(profile, "profile_id", "")),
            str(getattr(profile, "engine_impl_version", "")),
            str(engine or ""),
            str(stereo_policy or ""),
        ]
        for name in sorted(render_params):
            parts.append("%s=%s" % (name, render_params[name]))
        for route in sorted(routes, key=lambda r: str(r.get("route_id") or "")):
            runtime = route.get("runtime") or {}
            parts.extend([
                str(route.get("route_id") or ""),
                str(route.get("route_contract_version") or ""),
                str(route.get("status") or ""),
                str(route.get("engine") or ""),
                str(route.get("voice_ref_id") or ""),
                str(route.get("reference_kind") or ""),
                str(route.get("qualification_record_id") or ""),
                str(runtime.get("model_id") or ""),
                str(runtime.get("engine_impl_version") or ""),
                str(runtime.get("weight_revision") or ""),
            ])
            if route.get("reference_kind") == "local_wav":
                path = str(route.get("ref_path") or "")
                full = _resolve_ref_to_disk(path) or path
                digest = _ROUTE.sha256_of_file(full) if path else None
                if digest is None:
                    return float("nan")      # fail OPEN, never reuse
                parts.append(digest)
            else:
                parts.append(str(route.get("source_ref_sha256") or ""))

        for row in sorted(provisional_rows,
                          key=lambda r: str(r.get(_ROUTE.CAST_ROW_ROUTE_ID_FIELD) or "")):
            row_engine = str(row.get("voice_engine") or engine or "")
            voice_ref_id = str(row.get("voice_ref_id") or "")
            parts.extend([
                str(row.get(_ROUTE.CAST_ROW_ROUTE_ID_FIELD) or ""),
                row_engine,
                voice_ref_id,
                str(row.get("provider_voice_id") or ""),
            ])
            identity = _provisional_identity_fingerprint(row_engine, voice_ref_id)
            if identity is None:
                return float("nan")          # fail OPEN, never reuse
            parts.append(identity)

        import hashlib as _hashlib
        return _hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # D: local-disk only -- never touch the network at validation time.
        # Engine usability + model/token presence are enforced fail-closed on
        # the generate() path (C-5/C-7), not here, so a box-fresh graph still
        # validates clean.
        return True

    # ------------------------------------------------------------------ #
    def generate(self, script_json, engine, ledger_json="", gate_in="",
                 stereo_policy="mono_safe"):
        # CANONICAL REPLAY (campaign item 0): the frozen master carries every
        # take; nothing renders here. A typed empty AUDIO batch (nodes 3 and 7
        # do not consume it on replay) and an explicit done token.
        try:
            from .production_ledger import replay_descriptor as _replay_descriptor
            from ._otr_resolved_request import empty_audio_batch as _empty_audio
            _rmeta = (json.loads(ledger_json or script_json or "{}") or {}).get("meta") or {}
        except (ValueError, TypeError, ImportError, AttributeError):
            # AttributeError: a non-dict wire (the legacy parser LIST) is not a replay
            _rmeta = {}
        if _rmeta and _replay_descriptor(_rmeta):
            log.warning("[%s] REPLAY: pass-through, no TTS", type(self).__name__)
            return (_empty_audio(), "replay: pass-through (frozen master)",
                    "replay:passthrough")
        from ._otr_audio_engines import (
            EngineUnusable, EngineUsabilityReason, assert_usable, get_engine,
        )

        render_log: list = []
        audio_out = None
        adapter = None
        sr_hint = 24000
        n_clips = 0
        try:
            engine = assert_usable(engine, self.ROLE)  # FAIL CLOSED (6-class)
            adapter = get_engine(engine)
            # S4 platform-portability (2026-07-10): thread the EXPLICIT voice
            # device (CastLock ledger stamp meta.voice_device; default cuda =
            # nv50 baseline) into the adapter as an attribute -- no signature
            # churn; adapters without a local device (cloud lanes) ignore it.
            # The old per-adapter cuda->mps->cpu waterfalls are deleted.
            adapter.requested_device = _voice_device_from_ledger(
                ledger_json, script_json)
            # Role threading (2026-08-24): a preset engine that serves more than
            # one role (Bark: char_voice + announcer_voice) resolves its curated
            # profile by role, not by a hardcoded string -- see
            # BarkEngine._resolve_stage_temps. Adapters are registry singletons
            # (get_engine returns the same instance across calls), but character
            # and announcer generate() calls are sequential, never concurrent, so
            # overwriting this attribute per call is safe.
            adapter.role = self.ROLE
            interface = getattr(adapter, "interface", "per_line")
            sr_hint = int(getattr(adapter, "sample_rate", 24000) or 24000)

            if interface == "per_line":
                audio_out, lines, n_clips = self._render_per_line(
                    adapter, engine, script_json, ledger_json, stereo_policy,
                )
                render_log.extend(lines)
            else:
                raise EngineUnusable(
                    engine, self.ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                    f"engine {engine!r} declares unsupported interface "
                    f"{interface!r} for role {self.ROLE!r}",
                )
        finally:
            # I-7: free single-engine residency in finally, BEFORE 'done'.
            self._teardown(adapter)

        if audio_out is None:
            from ._otr_resolved_request import empty_audio_batch

            audio_out = empty_audio_batch(sr_hint)
        done = self._done_signal(engine, n_clips)
        return (audio_out, "\n".join(render_log), done)

    # ------------------------------------------------------------------ #
    def _render_per_line(self, adapter, engine, script_json, ledger_json,
                         stereo_policy):
        """Per-line dispatch for cloning engines (chatterbox / indextts2).

        Builds one frozen ``ResolvedVoiceRequest`` per in-role line, prepares
        the spoken text, calls the adapter, and packs the clips into the Bark
        AUDIO-batch contract (C-4). Zero in-role lines short-circuits to the
        canonical empty batch with no model load.
        """
        from . import _otr_ledger_consumers as _OTRLC
        from ._otr_audio_engines import (
            EngineUnusable, EngineUsabilityReason, pack_audio_batch,
        )
        from ._otr_engine_profiles import (
            assert_model_available, assert_token_for_profile,
            effective_license_state, require_resolver,
        )
        from ._otr_ledger import compute_audio_sample_hash
        from ._otr_resolved_request import (
            _seed_to_int64, build_resolved_request, empty_audio_batch,
        )
        from ._otr_determinism import deterministic_inference
        from ._otr_script_prep import prepare_text as _neutral_prepare_text
        from ._otr_text_delivery import (
            delivery_mode_for_meta, resolve_line_delivery,
        )

        sr = int(getattr(adapter, "sample_rate", 24000) or 24000)
        # Prefer the cast-locked ledger; fall back to the frozen writer ledger.
        source = ledger_json if (ledger_json or "").strip() else script_json
        led = _OTRLC.load_ledger(source)
        lines = [
            ln for ln in _OTRLC.iter_lines(led, roles=set(self.LINE_ROLES))
            if (ln.get("text") or "").strip()
        ]
        if not lines:
            return empty_audio_batch(sr), [f"{self.ROLE}: 0 lines"], 0

        resolver = require_resolver()
        profile = resolver.resolve_casting_plan(role=self.ROLE, engine=engine)
        assert_token_for_profile(profile)
        assert_model_available(profile)
        sr = int(profile.sample_rate or sr)

        meta = led.get("meta") or {}

        # Cross-widget announcer-engine agreement (2026-08-24, kibitz r3
        # MUST-FIX, corrected in r4 after codex caught a false-positive).
        # `OTR_CastLock.announcer_voice_engine` and this node's own `engine`
        # widget are two INDEPENDENTLY settable controls -- nothing else
        # checks they agree. A mismatch is not just cosmetic: CastLock's bark
        # stamp CLEARS `voice_ref_id` (bark has no bank identity), so if this
        # widget still reads 'kokoro' while CastLock resolved bark, eng_kokoro
        # receives an empty voice_ref and silently falls back to its own
        # per-episode seeded pick -- a real, audible, WRONG voice speaks the
        # line while the ledger and credits still say 'bark'.
        #
        # COMPARE AGAINST `meta["announcer_voice_engine"]`, NEVER the row's
        # own `voice_engine`/`tts_model` field. `_stamp_voice_engine_
        # selection` stamps that meta key UNCONDITIONALLY for every
        # cast_voice_policy, but under `preserve_ledger` (the DEFAULT
        # policy) CastLock only re-casts the policy-CLAIMED row -- the
        # announcer row itself keeps whatever the writer's `pick_announcer()`
        # wrote, which is ALWAYS `tts_model="kokoro"` regardless of what
        # engine was actually requested. An earlier cut of this guard read
        # the row's stale field and would have raised "controls disagree" on
        # every agreeing chatterbox/dia announcer request under
        # preserve_ledger -- a real regression against existing, working
        # functionality, caught by kibitz r4 before it shipped.
        if self.ROLE == "announcer_voice":
            stamped = str(meta.get("announcer_voice_engine") or "")
            if stamped and stamped != engine:
                raise EngineUnusable(
                    engine, self.ROLE,
                    EngineUsabilityReason.MALFORMED_CONFIG,
                    f"OTR_CastLock resolved announcer_voice_engine="
                    f"{stamped!r} but this node's own 'engine' widget is "
                    f"{engine!r} -- the two announcer-engine controls "
                    f"disagree. Set both OTR_CastLock.announcer_voice_engine "
                    f"and this node's 'engine' widget to the same value.",
                )
        elif self.ROLE == "char_voice":
            # The character-side twin of the guard above (2026-09-02, kokoro-onnx
            # r1): CastLock stamps char_voice_engine and this node carries its own
            # 'engine' widget; nothing compared them, so a graph with the two set
            # differently rendered one engine while the ledger and credits named
            # the other. "auto" is stamped LITERALLY when CastLock resolved nothing
            # (a preset bank under an auto request -- cast_lock.py
            # _stamp_voice_engine_selection), so it is never a disagreement.
            stamped = str(meta.get("char_voice_engine") or "")
            if stamped and stamped != "auto" and stamped != engine:
                raise EngineUnusable(
                    engine, self.ROLE,
                    EngineUsabilityReason.MALFORMED_CONFIG,
                    f"OTR_CastLock resolved char_voice_engine={stamped!r} but "
                    f"this node's own 'engine' widget is {engine!r} -- the two "
                    f"character-engine controls disagree. Set both "
                    f"OTR_CastLock.char_voice_engine and this node's 'engine' "
                    f"widget to the same value.",
                )

        episode_seed = coerce_int_seed(meta.get("episode_seed"))
        cast_lock_revision = int(meta.get("cast_lock_revision") or 0)
        mono = (stereo_policy == "mono_safe")
        # C2 (S2 P1.3): resolve the delivery mode ONCE. LEGACY lanes speak
        # canonical text (already Phase-7-normalized in place -> byte-
        # identical spine); content-owned lanes speak the verified
        # text_for_tts stamp (stale/absent = terminal before generation).
        _delivery_mode = delivery_mode_for_meta(meta)

        prep = getattr(adapter, "prepare_text", None)
        # Per-line delivery (emotion) vector for expressive engines (indextts2
        # emo-vector, chatterbox exaggeration). Prefer a stamped vector; else
        # derive it deterministically (pure -> C7) from line text + scene tension.
        # Engines that ignore delivery (bark / kokoro / dia) stay byte-identical;
        # OTR_DELIVERY_VECTOR=0 reproduces pre-delivery (flat) renders -- and is a
        # TRUE old path: the delivery module is imported lazily only when on.
        _delivery_on = otr_env.get("OTR_DELIVERY_VECTOR", "1") != "0"
        # voice_ref routing (clean-break 1a): most engines clone from a reference
        # clip path; bark routes its discrete v2/* voice_preset through the SAME
        # positional ref slot. The adapter declares which cast field feeds the
        # slot via voice_ref_field (default "voice_ref_path"); non-bark engines
        # omit the attr and keep the clip-path behavior unchanged.
        ref_field = getattr(adapter, "voice_ref_field", "voice_ref_path")
        # Per-episode engine context (additive, engine-agnostic): an engine that
        # needs episode-level state -- e.g. kokoro picks ONE announcer voice per
        # episode seeded from episode_seed and runs its C-7 voice-file preflight
        # -- does it here, once, before the per-line loop. Engines without the
        # hook (bark / chatterbox / indextts2) are unaffected.
        begin = getattr(adapter, "begin_episode", None)
        if callable(begin):
            begin(meta)
        clips: list = []
        log_lines: list = [
            f"{self.ROLE}: rendering {len(lines)} lines on '{engine}' "
            f"(profile {profile.profile_id})"
        ]
        # --- Cloud-audio-cache chunk 2 (2026-08-08) setup ---
        # profile.use_cache is False for every LOCAL profile; the cache branch
        # is skipped and this render path is byte-identical to today's shipped
        # behavior. Two Google TTS profiles set it True. If cache is on but no
        # cache dir resolves (no OTR_AUDIO_CACHE_DIR and no meta.paths.audio_dir),
        # FAIL LOUD -- operator explicitly enabled caching, silent fallback would
        # hide the config bug.
        from ._otr_audio_cache import FileAudioCache
        cache_enabled = bool(getattr(profile, "use_cache", False))
        cache = None
        if cache_enabled:
            cache_dir = _audio_cache_dir_for(meta)
            if not cache_dir:
                raise EngineUnusable(
                    engine, self.ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                    f"engine '{engine}' profile has use_cache=True but no cache "
                    f"dir resolved (OTR_AUDIO_CACHE_DIR unset and "
                    f"meta.paths.audio_dir absent)",
                )
            cache = FileAudioCache(cache_dir)
            log_lines.append(f"{self.ROLE}: cache enabled dir={cache_dir}")
        cache_stats = {"hit": 0, "miss": 0, "degraded_write": 0, "degraded_ledger": 0}
        ledger_stamps: list = []
        # Plan 5.3 route resolution, per character, once per render. The bank is
        # loaded lazily and only when some row actually carries a voice_route --
        # today no row does, so this costs nothing on every shipping render.
        _resolved_refs: dict = {}
        _route_bank: list = []
        # The lines that rendered on a QUALIFIED ROUTE, and the ones whose
        # receipts did not land. The raise at the end compares these two sets --
        # not a bare degraded COUNT, which cannot tell a failed Lemmy receipt
        # apart from some unrelated line's failed stamp.
        _policy_line_ids: set = set()
        _failed_stamp_ids: set = set()

        def _route_bank_lookup(voice_ref_id):
            if not _route_bank:
                from ._otr_voice_bank import load_voice_bank
                _route_bank.append(load_voice_bank()[0])
            return next(
                (e for e in _route_bank[0]
                 if e.voice_ref_id == voice_ref_id and e.engine == engine),
                None,
            )

        try:
            for occ, ln in enumerate(lines):
                # C2 (S2 P1.3): resolve canonical vs delivery through the ONE
                # resolver. `text` is the DELIVERY string every downstream
                # surface (delivery vector, adapter/neutral prep, request
                # hashing) consumes. LEGACY -> delivery == canonical (byte-
                # identical); content-owned -> the verified text_for_tts stamp
                # (this raises TextDeliveryError BEFORE generation on an
                # absent/stale stamp).
                canonical_text, _delivery_text = resolve_line_delivery(ln, _delivery_mode)
                text = _delivery_text.strip()
                char_id = str(ln.get("char_id") or "")
                line_id = str(ln.get("line_id") or "")
                cast = _OTRLC.cast_lookup(led, char_id)
                # Plan 5.3: immediately after cast_lookup, prove this row's
                # route against the engine actually rendering, BEFORE either
                # request is built. A row with no voice_route resolves to
                # LEGACY_REFERENCE -- all-empty identity, so its cache_key is
                # byte-identical to what it was before the schema grew.
                #
                # Memoized per char_id: the bytes are re-hashed once per render
                # rather than once per LINE. Re-hashing at point of use is the
                # point (a receipt proved at cast time says nothing about the
                # file five minutes later); re-hashing forty times is just I/O.
                resolved_ref = _resolved_refs.get(char_id)
                if resolved_ref is None:
                    resolved_ref = _ROUTE.resolve_and_verify_reference(
                        cast, engine, bank_lookup=_route_bank_lookup,
                        repo_root=_REPO_ROOT,
                        path_resolver=_resolve_ref_to_disk)
                    _resolved_refs[char_id] = resolved_ref
                route_fields = resolved_ref.request_fields()
                voice_ref_id = cast.get("voice_ref_id")
                voice_preset = cast.get("voice_preset")
                if ref_field == "voice_ref_path":
                    voice_ref = cast.get("voice_ref_path") or cast.get("ref_path")
                else:
                    voice_ref = cast.get(ref_field)
                # A PROVED local route renders ITS OWN bytes. Without this the
                # route would prove one file and the generic resolver below would
                # hand the adapter another -- a receipt describing audio nobody
                # ever heard, which is the exact class of defect this whole
                # contract exists to end.
                if (resolved_ref.is_policy_route
                        and resolved_ref.reference_kind == "local_wav"
                        and ref_field == "voice_ref_path"):
                    voice_ref = _resolve_ref_to_disk(resolved_ref.ref_path)                         or resolved_ref.ref_path
                delivery_vector = None
                _dv_source = "off"
                if _delivery_on:
                    _tension = ln.get("scene_tension", ln.get("tension", 0.0)) or 0.0
                    try:
                        # Whiny-fix P1.1: ONE chooser -- stamped vector with the
                        # version guard (stale v1 stamps re-derive LOUDLY), else
                        # derived from PREPARED text (G13: never raw text).
                        from ._otr_delivery_vector import select_delivery_vector
                        delivery_vector, _dv_source = select_delivery_vector(
                            ln, _neutral_prepare_text(text), float(_tension))
                    except Exception as _e:  # noqa: BLE001 -- best-effort, never fatal
                        log.debug("[OTR] delivery derive failed: %s", _e)
                        delivery_vector, _dv_source = None, "error"
                prepared = prep(text, delivery_vector) if callable(prep) else _neutral_prepare_text(text)
                # NO-FALLBACK (operator 2026-07-03): a beat that cleans to EMPTY text
                # (a stage-direction-only line like "(flips the switch)") must NEVER be
                # silently rendered as silence -- that hid a writer/ledger defect. The
                # writer must not emit voiced lines with no spoken content (a pre-freeze
                # assert guards this), and the ledger carries no stage-direction voice
                # role today. If one still reaches the voice gate, FAIL LOUD naming the
                # line so it is fixed, never papered over with a silent clip. (Routing a
                # stage-direction beat to a dedicated non-voice media engine is parked in
                # docs/ROADMAP_IDEAS.md -- re-add the role to the ledger then.)
                if not str(prepared or "").strip():
                    raise ValueError(
                        f"{self.ROLE}: line={line_id or occ} char={char_id or '-'} "
                        f"cleans to EMPTY spoken text (stage-direction-only?). No "
                        f"silent-clip fallback (no-fallback rip) -- the writer must not "
                        f"emit voiced lines with no dialogue. Canonical text: "
                        f"{canonical_text!r}"
                    )
                # Cloud-audio-cache (2026-08-08): resolve voice_ref BEFORE the
                # request build so the RESOLVED identity enters the cache key
                # (elevenlabs/kokoro/etc. draw from the current bank -- keying the
                # unresolved cast row would let a bank change replay a hit with a
                # different voice). Non-cache path keeps today's inline order below,
                # byte-identical. r2 MF#3 grounding: this does NOT perturb
                # stable_line_seed because params are not in the seed.
                provider_model_id_stamp = ""
                provider_voice_stamp = ""
                # PHASE ONE of the ONE per-line runtime context [QA-2]: the
                # cache-key params, the emotion alpha and the exact outbound
                # emotion payload, resolved ONCE. Everything downstream -- the
                # request, the P-OBS receipt, the cap metrics and the seed --
                # reads this object rather than re-deriving its own answer.
                # See `_begin_line_runtime` for the call-time-params rationale.
                line_rt = _begin_line_runtime(adapter, profile, delivery_vector)
                line_params = line_rt.params
                if cache_enabled:
                    voice_ref, id_for_key = _resolve_voice_ref_early(
                        adapter, engine, cast, episode_seed, self.ROLE, voice_ref,
                    )
                    adapter_identity = adapter.identity_params(resolved_voice_ref=id_for_key)
                    provider_model_id_stamp = str(adapter_identity.get("model", ""))
                    provider_voice_stamp = str(adapter_identity.get("provider_voice", id_for_key))
                    request = build_resolved_request(
                        role=self.ROLE,
                        engine_name=engine,
                        engine_profile_id=profile.profile_id,
                        engine_impl_version=profile.engine_impl_version,
                        char_id=char_id,
                        line_id=line_id,
                        occurrence=occ,
                        prepared_text=prepared,
                        voice_ref_id=id_for_key,
                        voice_preset=voice_preset,
                        episode_seed=episode_seed,
                        cast_lock_revision=cast_lock_revision,
                        sample_rate=sr,
                        channels=1,
                        params=line_params,
                        commercial_clean=profile.commercial_clean,
                        provider_model_id=provider_model_id_stamp,
                        provider_voice_id=provider_voice_stamp,
                        **route_fields,
                    )
                else:
                    request = build_resolved_request(
                        role=self.ROLE,
                        engine_name=engine,
                        engine_profile_id=profile.profile_id,
                        engine_impl_version=profile.engine_impl_version,
                        char_id=char_id,
                        line_id=line_id,
                        occurrence=occ,
                        prepared_text=prepared,
                        voice_ref_id=voice_ref_id,
                        voice_preset=voice_preset,
                        episode_seed=episode_seed,
                        cast_lock_revision=cast_lock_revision,
                        sample_rate=sr,
                        channels=1,
                        params=line_params,
                        commercial_clean=profile.commercial_clean,
                        **route_fields,
                    )
                # THE SEED IS DERIVED BELOW, NOT HERE [QA-5]. It used to be
                # computed at this point, before the block that resolves a
                # fallback reference -- so a character-stable seed keyed here
                # would key on the BLANK the request carried rather than on the
                # voice the adapter actually clones. Reference first, seed after.
                #
                # Ref-clip resolution (no-fallback rip 2026-07-03): a voice-CLONING char
                # engine (voice_ref_field == "voice_ref_path", e.g. indextts2 /
                # chatterbox) cannot synthesize without a per-character reference WAV.
                # We resolve the cast row's OWN ref below; if none resolves, the line
                # FAILS LOUD (no bark fallback). A valid ref is left byte-identical.
                #
                # Cloud-audio-cache chunk 2 (2026-08-08): gated behind
                # `not cache_enabled` because _resolve_voice_ref_early above already
                # ran these three branches when the cache is active (r4 Sonnet 5 QA
                # SF#1). Removes wasted resolver work per line and closes the latent
                # trap where a future non-idempotent clone-engine resolver would
                # double-mutate state.
                if not cache_enabled:
                    if _engine_requires_voice_ref(adapter) and voice_ref:
                        # A non-empty but STALE ref (the file is gone) must fall through to
                        # resolution, not be shipped to the worker (which would hard-fail
                        # "ref_clip missing"). A valid ref is left untouched (the adapter
                        # resolves it) so shipped paths are byte-identical; only a
                        # missing-on-disk ref is nulled -> re-resolved or fails loud below.
                        _disk = _resolve_ref_to_disk(voice_ref)
                        if not (_disk and os.path.exists(_disk)):
                            voice_ref = None
                    if _engine_requires_voice_ref(adapter) and not voice_ref:
                        # preserve_ledger / CastLock-stamps-only-the-id: resolve an
                        # on-disk reference WAV from voice_ref_id or by gender so the
                        # clone engine clones a REAL voice for THIS character. If none
                        # resolves, we FAIL LOUD below (no-fallback rip 2026-07-03) -- a
                        # cloning engine NEVER silently renders on bark.
                        voice_ref = _resolve_clone_ref_path(engine, cast, episode_seed, role=self.ROLE)
                    # ONE-KNOB (2026-07-04): a cloud PROVIDER-voice engine (voice_ref_field
                    # == "provider_voice_id", e.g. elevenlabs) whose cast row has NO provider
                    # id -- the operator changed only the VOICE ENGINE and left CastLock's
                    # voice_bank on a LOCAL bank. Resolve a gender-matched voice_id from the
                    # ENGINE's own pool here (deterministic per-character casting, parity with
                    # the clone-ref path above), so the voice engine is the SINGLE knob and
                    # CastLock never has to be touched. A missing pool -> None -> the adapter
                    # still fails loud (no silent inherit of another voice).
                    if getattr(adapter, "voice_ref_field", "") == "provider_voice_id" and not voice_ref:
                        voice_ref = _resolve_provider_voice_id(engine, cast, episode_seed, role=self.ROLE)
                # PHASE TWO of the per-line context: G1's per-engine external
                # seed, now that the reference this character will actually be
                # cloned from is resolved on BOTH paths [QA-5]. Profiles that
                # have not opted into character-stable seeding keep the legacy
                # `_seed_to_int64(engine, request.stable_line_seed)` byte for
                # byte [QA-1]; the char_* clone profiles get one seed per
                # character per episode, so a character stops changing voice
                # between his own lines.
                _resolve_engine_seed(
                    line_rt, _seed_to_int64, profile, engine, request,
                    char_id, episode_seed, cast_lock_revision,
                    voice_ref_id, voice_ref,
                )
                engine_seed = line_rt.engine_seed
                # ---- P-OBS (whiny-fix v3.1): per-line attribution, ALWAYS -------
                # char -> voice_ref_id -> ref basename -> engine -> alpha ->
                # delivery version/state/source -> seed. Render_log line + runtime
                # log mirror; this is the observability floor every later step
                # (P0-zero, the audit, P3a durable stamping) builds on.
                # BOTH READ THE CONTEXT [QA-4]. `_vec_state` used to call
                # float() on the RAW stamped values -- before the adapter's own
                # sanitation -- so a hand-edited ledger carrying a string where
                # a number belongs raised ValueError out of the observability
                # line and killed the render. THE LAW: a render degrades, never
                # raises. The alpha and the emotion mass come from the same
                # resolution the worker payload is built from, so the receipt
                # can no longer describe a blend the engine did not use.
                _alpha = line_rt.alpha
                _vec_state = line_rt.vector_state
                _mass = line_rt.effective_mass
                try:
                    from ._otr_delivery_vector import DELIVERY_TABLE_VERSION as _DTV
                except Exception:  # noqa: BLE001
                    _DTV = "?"
                _pobs = (
                    f"{self.ROLE}: line={line_id or occ} char={char_id or '-'} -> "
                    f"voice_ref_id={voice_ref_id or '-'} "
                    f"ref={os.path.basename(voice_ref) if voice_ref else '-'} "
                    f"engine={engine} "
                    f"alpha={'n/a' if _alpha is None else _alpha} "
                    f"delivery={_DTV}:{_vec_state}({_dv_source}) "
                    f"emo_mass={'n/a' if _mass is None else _mass}"
                    f"{'(capped)' if line_rt.mass_capped else ''} "
                    f"seed={engine_seed} policy={line_rt.seed_policy} "
                    f"seed_ref={line_rt.ref_identity or '-'}"
                )
                # Cloud-audio-cache chunk 2 (2026-08-08): when cache is enabled,
                # a SECOND P-OBS emit happens further down with the terminal
                # cache=<status> token. Skip the bare emit here to avoid the
                # double-log per r4 Sonnet 5 QA MF#1. When cache is off, the
                # bare emit here IS the one and only P-OBS line (byte-identical
                # to pre-chunk behavior).
                if not cache_enabled:
                    log_lines.append(_pobs)
                    log.info("[OTR voice P-OBS] %s", _pobs)
                # NO-FALLBACK (operator 2026-07-03): a cloning engine that reached this
                # line with no usable voice reference FAILS LOUD -- it never silently
                # renders on bark. The old bark missing-ref net is retired; a missing
                # reference is a casting/install defect the operator must fix, surfaced
                # here as a NAMED EngineUnusable (MISSING_MODEL) naming the char/line.
                if _engine_requires_voice_ref(adapter) and not voice_ref:
                    raise EngineUnusable(
                        engine, self.ROLE, EngineUsabilityReason.MISSING_MODEL,
                        f"no usable voice reference for cloning engine '{engine}' on "
                        f"line={line_id or occ} char={char_id or '-'} "
                        f"(voice_ref_id={voice_ref_id or '-'}). Install the engine's "
                        f"reference WAVs or cast a voice with a resolvable ref -- there "
                        f"is NO bark fallback (no-fallback rip).",
                    )
                # G1: scope strict-determinism + seed/restore every RNG around the
                # single forward (I-2/C-2). warn_only=True keeps the process default
                # non-strict so a nondeterministic CUDA op cannot crash the opt-in
                # render on sm_120; bit_exact (warn_only=False) is gated on the F
                # pilot verifying each engine binds an external torch.Generator.
                #
                # Cloud-audio-cache chunk 2 (2026-08-08): when profile.use_cache is
                # True, look up FileAudioCache first; on a hit skip the API call
                # entirely. On a miss run the forward, cache the bytes, stamp the
                # ledger. Both branches inline "adapter.generate_voice(" under the
                # deterministic_inference CM to keep the source-grep guard at
                # tests/test_audio_determinism_wrap.py:158-163 green.
                #
                # Per-line try/finally (r4 Fable gate SF#2): ensures P-OBS emits
                # per line even when generate_voice raises, so a dying render
                # always logs which line it died on -- preserves the pre-chunk
                # observability contract ("per-line attribution, ALWAYS").
                _render_start = time.monotonic()
                cache_status = "off"
                audio = None
                try:
                    if cache_enabled and cache is not None:
                        loaded = cache.load(request)
                        if loaded is not None:
                            audio, cached_record = loaded
                            cache_status = "hit"
                            cache_stats["hit"] += 1
                    if audio is None:
                        if cache_enabled and engine == "google_tts":
                            with deterministic_inference(engine_seed, warn_only=True):
                                audio = adapter.generate_voice(
                                    prepared, voice_ref, delivery_vector, engine_seed,
                                    disable_retry=True,
                                    resolved_model=provider_model_id_stamp,
                                )
                        else:
                            with deterministic_inference(engine_seed, warn_only=True):
                                audio = adapter.generate_voice(
                                    prepared, voice_ref, delivery_vector, engine_seed,
                                )
                    # P-OBS sample-rate assert: a primary-engine clip whose rate does
                    # not match the pack rate is a real defect (the pack would crash
                    # or silently resample later) -- LOUD, never silent.
                    _got_sr = int(audio.get("sample_rate", sr) or sr)
                    if _got_sr != sr:
                        _sr_msg = (
                            f"{self.ROLE}: WARNING line={line_id or occ} clip sample "
                            f"rate {_got_sr} != pack rate {sr} (engine {engine}) -- "
                            f"investigate before trusting this episode's voice lane"
                        )
                        log_lines.append(_sr_msg)
                        log.warning("[OTR voice P-OBS] %s", _sr_msg)
                    # Cloud-audio-cache: on the miss path, cache the bytes (unless
                    # rate mismatched, in which case skip the put -- a mismatched
                    # clip cached would silently mispitch on the next hit). On any
                    # path collect the ledger stamp bundle. r4 MF#4: hit stamps
                    # render_ms=0 (valid persisted value under the new
                    # render_ms=None-is-skip signature); miss stamps the elapsed time.
                    if resolved_ref.is_policy_route:
                        _policy_line_ids.add(line_id)
                    _asample_hash = compute_audio_sample_hash(
                        audio["waveform"] if isinstance(audio, dict) else audio
                    )
                    _dur_s = (
                        float(audio["waveform"].shape[-1]) / float(_got_sr)
                        if _got_sr else 0.0
                    )
                    if not cache_enabled and resolved_ref.is_policy_route:
                        # Plan 5.3: LOCAL renders leave a receipt too. They never
                        # did -- every stamp below sat inside the cache branch --
                        # so an indextts2 episode finished with no per-line record
                        # of which engine, which route or which bytes had spoken,
                        # and Lemmy's route is a LOCAL one. A qualified route
                        # whose evidence only exists on the cloud lane is not
                        # evidence.
                        #
                        # SCOPED TO POLICY-ROUTE LINES ON PURPOSE. Stamping every
                        # local line would reload-and-resave the whole ledger file
                        # on every ordinary leg, which
                        # test_end_to_end_google_tts_cache_off_byte_identity
                        # exists to forbid -- and it is right to: two voice roles
                        # rewriting one file per render is a corruption hazard
                        # bought for telemetry nobody asked for. A proved route is
                        # different; its receipt is the entire point of proving
                        # it. Today no row carries a route, so this branch never
                        # runs and the control's invariant holds byte-for-byte.
                        ledger_stamps.append((line_id, {
                            "tts_engine": engine,
                            "voice_preset": voice_preset or "",
                            "render_ms": int(
                                (time.monotonic() - _render_start) * 1000),
                            "generated_dur_s": _dur_s,
                            "audio_sample_hash": _asample_hash,
                            "sample_rate": _got_sr,
                            "voice_route_id": resolved_ref.route_id,
                        }))
                    if cache_enabled:
                        if cache_status == "hit":
                            ledger_stamps.append((line_id, {
                                "tts_engine": engine,
                                "voice_preset": voice_preset or "",
                                "render_ms": 0,
                                "generated_dur_s": _dur_s,
                                "audio_sample_hash": _asample_hash,
                                "audio_cache_key": cached_record.cache_key,
                                "audio_sha256": cached_record.audio_sha256,
                                "provider_model_id": cached_record.provider_model_id or "",
                                "sample_rate": _got_sr,
                                "voice_route_id": resolved_ref.route_id,
                            }))
                        else:
                            _elapsed_ms = int((time.monotonic() - _render_start) * 1000)
                            if _got_sr != sr:
                                cache_status = "degraded_write"
                                cache_stats["degraded_write"] += 1
                                ledger_stamps.append((line_id, {
                                    "tts_engine": engine,
                                    "voice_preset": voice_preset or "",
                                    "render_ms": _elapsed_ms,
                                    "generated_dur_s": _dur_s,
                                    "audio_sample_hash": _asample_hash,
                                    "provider_model_id": provider_model_id_stamp,
                                    "sample_rate": _got_sr,
                                    "voice_route_id": resolved_ref.route_id,
                                }))
                            elif cache is not None:
                                try:
                                    fresh_record = cache.put(
                                        request, audio,
                                        allowed_for_release=(
                                            effective_license_state(profile) == "clean"
                                        ),
                                        actual_sample_rate=_got_sr,
                                        provider_model_id=provider_model_id_stamp,
                                    )
                                    cache_status = "miss"
                                    cache_stats["miss"] += 1
                                    ledger_stamps.append((line_id, {
                                        "tts_engine": engine,
                                        "voice_preset": voice_preset or "",
                                        "render_ms": _elapsed_ms,
                                        "generated_dur_s": _dur_s,
                                        "audio_sample_hash": _asample_hash,
                                        "audio_cache_key": request.cache_key,
                                        "audio_sha256": fresh_record.audio_sha256,
                                        "provider_model_id": provider_model_id_stamp,
                                        "sample_rate": _got_sr,
                                        "voice_route_id": resolved_ref.route_id,
                                    }))
                                except Exception as _put_err:  # noqa: BLE001
                                    log.warning("[OTR voice cache] put failed: %s", _put_err)
                                    cache_status = "degraded_write"
                                    cache_stats["degraded_write"] += 1
                                    ledger_stamps.append((line_id, {
                                        "tts_engine": engine,
                                        "voice_preset": voice_preset or "",
                                        "render_ms": _elapsed_ms,
                                        "generated_dur_s": _dur_s,
                                        "audio_sample_hash": _asample_hash,
                                        "provider_model_id": provider_model_id_stamp,
                                        "sample_rate": _got_sr,
                                        "voice_route_id": resolved_ref.route_id,
                                    }))
                finally:
                    if cache_enabled:
                        _pobs_tail = f" cache={cache_status}"
                        log_lines.append(_pobs + _pobs_tail)
                        log.info("[OTR voice P-OBS] %s%s", _pobs, _pobs_tail)
                clips.append(audio)
            packed = pack_audio_batch(clips, sample_rate=sr, mono=mono)
            n = int(packed["waveform"].shape[0]) if packed["waveform"].numel() else 0
            log_lines.append(f"{self.ROLE}: packed {n} clips at {sr} Hz")
        finally:
            # Flush the collected per-line ledger stamps whether the loop
            # completed cleanly or a mid-loop exception is propagating out.
            # Any raise from _persist_ledger_stamps' pre-try setup is credited
            # as fully degraded so telemetry never lies (r4 SF#1 defensive wrap).
            # Plan 5.3: the flush is no longer cache-only. Local renders collect
            # receipts too, and a receipt that is collected but never written is
            # not a receipt.
            if ledger_stamps:
                try:
                    cache_stats["degraded_ledger"] += _persist_ledger_stamps(
                        meta, ledger_stamps, log,
                        failed_line_ids=_failed_stamp_ids)
                except Exception as _pe:  # noqa: BLE001
                    cache_stats["degraded_ledger"] += len(ledger_stamps)
                    _failed_stamp_ids.update(lid for lid, _ in ledger_stamps)
                    log.warning(
                        "[OTR voice] ledger stamp flush failed in finally "
                        "(reporting %d stamps as degraded): %s",
                        len(ledger_stamps), _pe)
        # Cache summary emitted only on successful completion; a mid-loop
        # exception propagates past this block, correctly skipping the summary.
        if cache_enabled:
            _total = cache_stats["hit"] + cache_stats["miss"] + cache_stats["degraded_write"]
            log_lines.append(
                f"{self.ROLE}: cache summary hit={cache_stats['hit']} "
                f"miss={cache_stats['miss']} degraded_write={cache_stats['degraded_write']} "
                f"degraded_ledger={cache_stats['degraded_ledger']} "
                f"api_saved={cache_stats['hit']} of {_total}"
            )
        # Plan 5.3, and it is deliberately the harshest rule in this method:
        # failure to persist a SELECTED-ROUTE receipt fails BEFORE returning
        # audio. A qualified route exists to make a claim provable after the
        # fact; audio that shipped with its receipt silently dropped has un-made
        # that claim, and handing it back would be the quiet lie the whole
        # contract was built to prevent. Non-policy lines keep the existing
        # degraded-telemetry behaviour -- this raises only when a proved route
        # actually rendered.
        _unproved = _policy_line_ids & _failed_stamp_ids
        if _unproved:
            raise RuntimeError(
                f"{self.ROLE}: the per-line receipt did not persist for "
                f"{len(_unproved)} line(s) rendered on a QUALIFIED VOICE ROUTE "
                f"({', '.join(sorted(_unproved))}). The audio is not being "
                f"returned: a route whose evidence did not land is an "
                f"unprovable claim. Note this is scoped to the ROUTE's own "
                f"lines -- an unrelated line's failed stamp stays telemetry and "
                f"never throws away good, fully evidenced audio."
            )
        return packed, log_lines, n

    # ------------------------------------------------------------------ #
    @staticmethod
    def _teardown(adapter):
        """I-7 teardown: unload local engines + free VRAM. Best-effort, never
        raises (a teardown failure must not mask the render result).

        Cloud adapters declare ``native=False``: they have no local residency,
        so teardown must not import torch or poke CUDA just because they emitted
        a Comfy AUDIO tensor.
        """
        try:
            if adapter is not None and hasattr(adapter, "unload"):
                adapter.unload()
        except Exception:  # noqa: BLE001
            pass
        try:
            gc.collect()
            if adapter is not None and getattr(adapter, "native", True) is False:
                return
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _done_signal(self, engine, n_clips) -> str:
        """Non-empty completion sentinel for the done->gate chain (E.5).

        Bounded + informative; the next node binds its ``gate_in`` to this so it
        cannot fire until this node has returned (and torn down, per I-7).
        """
        return f"{self.DONE_PREFIX}:done:engine={engine}:clips={int(n_clips)}"
