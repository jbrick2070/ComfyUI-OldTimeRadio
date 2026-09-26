"""OTR_CastLock -- single v2 ledger authority (plan E.0-E.5, Wave 2a).

Sits AFTER OTR_LedgerFreezeCascade and is the one place the v2 cast is locked:
it validates the cast, optionally assigns voice references from the bank with the
deterministic caster, stamps ``voice_ref_id`` / ``voice_preset`` /
``cast_lock_revision`` onto the cast entries, and emits the single canonical
``ledger_json`` the v2 audio nodes (and HuMo) consume.

Byte-safety (I-1 / I-3): CastLock's ``ledger_json`` feeds the v2 nodes' per-line
path ONLY. The legacy raw-delegation path keeps reading the untouched
FreezeCascade ``script_json`` (the bark batch path delegates that verbatim), so
the legacy audio stays byte-identical even though CastLock rewrites the ledger.

Casting (I-4): the new caster runs on its own seeded RNG, disjoint from the
legacy cast RNG. ``preserve_ledger`` (default) re-casts nothing; ``auto_registry``
assigns references from the selected voice bank.

E.4: the surfaced widgets are exactly ``cast_voice_policy`` /
``allow_voice_reuse`` / ``char_voice_engine`` / ``announcer_voice_engine`` --
the bank follows the engine per role (no ``voice_bank`` widget), and 4a/4b
inherit the stamped engines. ``delivery_profile`` is no longer surfaced
(single option "neutral" in v2; the ``lock()`` kwarg still defaults to
"neutral" and is validated + stamped) -- no ``voice_engine_mode``,
``deterministic_inference`` or ``model_id`` widget.
Import-time is side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from ._otr_shared import device_options as _DEVOPTS

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _require_language_engines(meta, char_engine, announcer_engine) -> str:
    """Non-English rows admit only the engines listed on the row. English
    is unchanged -- bark / indextts / the rest still run."""
    from . import _otr_episode_languages as _EPLANG
    iso = _EPLANG.iso_from_meta(meta if isinstance(meta, dict) else {})
    if iso == _EPLANG.ENGLISH_ISO:
        return iso
    row = _EPLANG.row_from_meta(meta)
    admitted = set(row.engines or {})
    for engine in (char_engine, announcer_engine):
        eng = str(engine or "").strip()
        if not eng or eng == "auto":
            continue
        if eng not in admitted:
            raise ValueError(
                "OTR_CastLock: engine %r is not admitted on a %s episode "
                "(row engines: %s). Kokoro is the dance leader day 1."
                % (eng, row.label, sorted(admitted)))
    _EPLANG.assert_readiness_extras(row)
    return iso


# Leftover ``lock(voice_bank=...)`` kwargs still exist for old callers.
# The bank is not a CastLock widget; ``_bank_following_engine`` derives it
# from each engine profile's ``allowed_voice_banks``.
_CAST_POLICIES = ("preserve_ledger", "auto_registry")
_CHAR_VOICE_ENGINES = (
    "auto", "indextts2", "chatterbox", "dia", "bark", "kokoro",
    "cloud_elevenlabs", "google_tts")
_ANNOUNCER_VOICE_ENGINES = (
    "auto", "kokoro", "chatterbox", "dia", "cloud_elevenlabs", "bark",
    "google_tts")
# google_tts stays last in the COMBO (draft google_* profiles need it) but is
# never a CastLock default and must not be pinned by any otr_cloud_* profile.
_VOICE_ENGINE_RESOLVE_EXTRA = frozenset()
# Old graphs/profiles stored ``elevenlabs``; that id was always the Comfy
# Credits partner adapter, never a local install. Normalize so the dropdown
# label stays ``cloud_elevenlabs`` without bricking saved widgets.
_VOICE_ENGINE_ALIASES = {"elevenlabs": "cloud_elevenlabs"}
_DEFAULT_ANNOUNCER_ENGINE = "kokoro"
_DEFAULT_CHAR_ENGINE = "kokoro"


#: Cast-row fields cleared before the claimed row is re-stamped at a DIFFERENT
#: tier or engine. Every one of these is an ENGINE-SPECIFIC identity that the
#: dispatch PREFERS over resolving from the freshly stamped bank id
#: (`_otr_voice_node_common` reads `cast.get("voice_ref_path") or
#: cast.get("ref_path")` for wav engines and `cast.get(ref_field)` otherwise), so
#: a survivor from a previous lock is not stale metadata -- it is the file or the
#: cloud voice that actually gets rendered.
#:
#: `voice_route` is here because it means "a QUALIFIED route was proved". A row
#: locked once under the IndexTTS2 route and re-locked on another engine keeps it
#: and the voice node then raises ENGINE DISAGREEMENT -- a render killed by a
#: leftover.
#:
#: `voice_preset` IS DELIBERATELY NOT HERE. It is bark's identity, it is written
#: far upstream at writer time by `lemmy_row()`, and `_stamp` has never touched
#: it. The 2026-08-16 acceptance leg proved the two-stage behaviour that depends
#: on it surviving: the frozen row keeps the writer-stage Bark preset while
#: delivery resolves the qualified IndexTTS2 route. Clearing it here would delete
#: a fact this module does not own.
_STALE_IDENTITY_FIELDS = (
    "voice_route",
    "voice_ref_path",
    "ref_path",
    "provider_voice_id",
    # THE THREE RETIRED ROUTE FIELDS, kept as LITERALS and kept on this list on
    # purpose. Nothing writes them any more, but a ledger locked before
    # 2026-09-24 still carries them, and a row re-cast today should shed them
    # rather than keep a tier claim about a system that no longer exists. They
    # are spelled out rather than imported because the module that defined them
    # is gone; this is not a migration of finished episodes on disk, only a
    # clear on a row this lock is re-stamping anyway.
    "lemmy_route_tier",
    "lemmy_route_id",
    "lemmy_route_reason_code",
)


def _recurring_character_key(entry) -> str:
    """The recurring-character key for this row, or "" -- import-safe.

    Wrapped so cast_lock keeps its cold-import discipline and so a flat-import
    test harness that cannot see `config` degrades to "ordinary row" instead of
    raising inside casting.
    """
    # RELATIVE FIRST. A bare `config` is not importable when ComfyUI loads the
    # pack as a submodule -- see the note at `_announcer_pool_from_pools`.
    try:
        from ..config.cast_pools import recurring_character_key
    except ImportError:  # pragma: no cover -- flat-import harnesses
        try:
            from config.cast_pools import recurring_character_key  # type: ignore
        except ImportError:
            try:
                from cast_pools import recurring_character_key  # type: ignore
            except ImportError:
                return ""
    return recurring_character_key(entry)


def _clear_stale_voice_identity(entry: dict) -> list:
    """Clear stale engine-specific identity from ONE cast row.

    RENAMED 2026-09-24 from `_normalize_row_for_tier_switch`. There are no tiers
    to switch between any more; what survives, and is the only reason this is
    still here, is that a row re-cast onto a different engine must not keep one
    field from the old one.

    Transactional in the only sense that matters here: the keys to remove are
    computed first and then removed together, so no caller can observe a row that
    is half a chatterbox and half an ElevenLabs voice.

    SCOPED TO THE CLAIMED ROW ON PURPOSE. The defect is a tier or engine switch on
    the row this policy pins, and clearing these fields across an entire cast would
    change bytes on two hundred rows that have nothing to do with Lemmy.
    """
    if not isinstance(entry, dict):
        return []
    doomed = [f for f in _STALE_IDENTITY_FIELDS if f in entry]
    for field in doomed:
        entry.pop(field, None)
    return doomed


def _is_announcer_entry(entry: dict) -> bool:
    char_id = str(entry.get("char_id") or "").strip().lower()
    name = str(entry.get("name") or "").strip().upper()
    role = str(entry.get("speaker_role") or entry.get("role") or "").strip().lower()
    return char_id == "announcer" or name == "ANNOUNCER" or role == "announcer"


def _row_has_resolvable_voice(row) -> bool:
    """A spoken row is voiced once it has a preset OR a bank reference.

    CastLock auto_registry stamps ``voice_ref_id`` for kokoro / google_tts /
    elevenlabs and clears leftover Bark ``voice_preset``. Requiring only
    ``voice_preset`` would fail-loud on a correctly stamped kokoro row.
    """
    if not isinstance(row, dict):
        return False
    preset = str(row.get("voice_preset") or "").strip()
    ref = str(row.get("voice_ref_id") or "").strip()
    return bool(preset or ref)


def _profile_role_for_entry(entry: dict) -> str:
    """The engine-profile role a cast row draws its voice from."""
    return "announcer_voice" if _is_announcer_entry(entry) else "char_voice"


def _model_license_clean(role: str, engine: str) -> bool | None:
    """Whether ``engine``'s MODEL is commercial-clean for ``role``.

    Resolved through the engine-profile resolver, which already owns this fact:
    `char_indextts2_v1` carries `commercial_clean: false` for the bilibili Model
    Use License. Exactly ONE profile answers, looked up by the (role, engine)
    pair -- never by engine name alone, because the same engine can be curated
    differently for the announcer than for a character.

    ``None`` means no profile covers the pair. That is not a licence: the caller
    keeps the clip's own flag rather than inventing a verdict either way.
    """
    # Imported HERE, not at module scope, on purpose: `_otr_engine_profiles`
    # reaches `_otr_audio_engines`, whose `base.py` imports torch. Cast locking
    # runs on paths that must stay light, so the dependency is paid only when a
    # licence is actually being resolved -- the same lazy shape `eng_bark` uses
    # to read a profile default.
    from . import _otr_engine_profiles as profiles

    resolver = profiles.load_resolver()
    if resolver is None:
        return None
    profile = resolver.profile_for(role, str(engine or ""))
    if profile is None:
        return None
    return profiles.effective_license_state(profile) == "clean"


def _delivered_commercial_clean(entry: dict, ref) -> bool:
    """The DELIVERED audio's commercial standing -- the CLIP's licence AND the
    MODEL's, joined.

    TWO DIFFERENT LICENCES MEET HERE and only one of them used to be read. A
    voice-bank row's `commercial_clean` describes the reference CLIP (the 40
    indextts2 rows are public-domain recordings, so they say true, correctly).
    The MODEL that speaks with that clip has a licence of its own, and
    indextts2's is non-commercial. Reading the clip alone made the cast report
    print `clean=True` for audio no one could ship, and stamped that on the
    ledger.

    Either half gates the result. An unknown model licence leaves the clip's
    flag standing, so a partial install or an engine with no curated profile
    behaves exactly as it did before this join existed.
    """
    clip_clean = bool(getattr(ref, "commercial_clean", False))
    if not clip_clean:
        return False
    model_clean = _model_license_clean(
        _profile_role_for_entry(entry), getattr(ref, "engine", ""))
    if model_clean is None:
        return clip_clean
    return model_clean


def _recurring_character_bank_ref(entry, engine, bank_entries, language):
    """The bank row a recurring character is delivered with here, or ``None``.

    Returns ``(ref, miss_reason)``. ``ref`` is ``None`` on every miss and
    ``miss_reason`` is a short string for the report; a miss is ORDINARY, not an
    error, and the caller falls through to the normal draw.

    THE MATCH IS EXACT AND UNAMBIGUOUS. Exactly one bank row must carry the
    assigned id on this engine. Zero means the table names something the bank
    does not ship; more than one means the bank is ambiguous and picking either
    would be a coin flip that changes with file order. Both refuse.

    LANGUAGE IS CHECKED AFTER THE MATCH, deliberately: "the assigned voice does
    not speak this episode's language" is a different answer from "the assigned
    voice does not exist", and collapsing them would hide a broken table behind
    a language miss.

    TWO SOURCES, IN ORDER. A bank row RESERVED for this character is his own
    recording and wins -- PROVIDED ITS BYTES ARE ON THIS MACHINE; otherwise the
    shared catalogue assignment in RECURRING_CHARACTER_VOICES applies. A
    reservation that never reached its owner is the defect this order exists to
    prevent; a reservation delivered without its file is the crash that presence
    check exists to prevent.

    WHAT THIS DELIBERATELY NO LONGER CHECKS, stated because it is a real
    reduction and not an oversight: the retired route subsystem gated its
    indextts2 clone on a RUNTIME FINGERPRINT -- it hashed three files (the
    adapter, the worker, and `_otr_resolved_request.py`, that last one because
    the seed path is part of the rendering code), compared the result against
    the value frozen at the 2026-08-18 audition, and demoted the voice to an
    ordinary draw when any of them had moved since.
    This resolver has no such gate; a reserved row is delivered on the strength
    of being in the bank. That trade is intentional. The fingerprint produced
    eighteen false demotions in nineteen commits (measured, and recorded in
    `tests/test_stale_ledger_voice_guard_removed.py`), it silently substituted a
    stranger's voice as its failure mode, and the operator's standing direction
    is that a guard is legitimate only against a silent WRONG result -- which
    this one caused rather than prevented. The residual risk is real and is
    accepted: if the indextts2 adapter drifts far enough to change how that
    reference clones, nothing here will notice, and the check is the operator's
    ear on the next leg.

    No ledger field is written here and no qualification is consulted. The row
    either takes its assigned voice or takes the ordinary draw.
    """
    try:
        from ..config.cast_pools import (
            recurring_character_key, recurring_character_voice)
    except ImportError:  # pragma: no cover -- flat-import harnesses
        try:
            from config.cast_pools import (  # type: ignore
                recurring_character_key, recurring_character_voice)
        except ImportError:
            try:
                from cast_pools import (  # type: ignore
                    recurring_character_key, recurring_character_voice)
            except ImportError:
                return None, "recurring table unavailable"

    character_key = recurring_character_key(entry)
    if not character_key:
        return None, ""

    # Imported at call time, like every other voice-bank name in this module:
    # cast_lock must stay cold-import clean, and the bank pulls in the schema
    # validator.
    try:
        from ._otr_voice_bank import voice_speaks_language
    except ImportError:  # pragma: no cover -- flat-import harnesses
        from _otr_voice_bank import voice_speaks_language  # type: ignore
    if not engine:
        # A preset-only lock resolved no engine; there is no bank to assign
        # from. Bark rows reach this and keep the preset path.
        return None, "no engine resolved"

    def _deliver(ref, voice_ref_id):
        """Language is checked AFTER the match, for both sources alike."""
        if not voice_speaks_language(ref, language):
            return None, "%s does not speak %s" % (voice_ref_id, language)
        return ref, ""

    # HIS OWN RECORDING FIRST. A bank row whose `reserved_for` names this
    # character is a clone of that character's actual voice, withheld from
    # every other draw. It outranks a shared catalogue assignment because it
    # IS him rather than a stand-in, and because withholding a row from
    # everyone and then not giving it to its owner reserves it for nobody --
    # which is exactly what happened between the casting cutover and this
    # change: Lemmy's chatterbox and dia clones sat reserved while he was cast
    # on an ordinary librivox voice.
    #
    # Read off the bank, not a second table: `reserved_for` already carries
    # the owner, and duplicating those ids into RECURRING_CHARACTER_VOICES
    # would also break that table's own rule that its ids stay castable for
    # everyone else.
    owned = [e for e in (bank_entries or ())
             if e.engine == engine
             and str(getattr(e, "reserved_for", "") or "").strip().casefold()
             == character_key.strip().casefold()]
    if len(owned) > 1:
        # Ambiguous exactly like the table path below: picking either would
        # make the cast depend on bank file order.
        return None, "%s has %d reserved %s rows" % (
            character_key, len(owned), engine)
    if owned:
        row = owned[0]
        # HIS RECORDING IS ONLY HIS VOICE IF IT IS ON THIS MACHINE.
        #
        # The reserved rows name a PRIVATE clip that nothing distributes:
        # `scripts/otr_dl_indextts2_refs.py` has no entry for it and
        # `scripts/otr_provision.py` skips reserved rows deliberately, so the
        # provisioner reports green on a box where the bytes are absent. 51
        # profiles put a clone engine on the character slot, including the
        # rented-pod starter. Delivering the row regardless hands the adapter a
        # path that is not there and the voice path fails loud by design -- so a
        # cameo would turn a working episode into a dead render on every machine
        # except the one that recorded it.
        #
        # Resolved through the RENDER PATH's own resolver so this check and the
        # adapter cannot disagree about where the file lives.
        ref_path = str(getattr(row, "ref_path", "") or "")
        if ref_path and not ref_path.startswith("cloud:"):
            try:
                from ._otr_voice_node_common import _resolve_ref_to_disk
            except ImportError:  # pragma: no cover -- flat-import harnesses
                from _otr_voice_node_common import (  # type: ignore
                    _resolve_ref_to_disk)
            # RESOLVE **AND STAT**. `_resolve_ref_to_disk` answers "where would
            # this live", not "is it there" -- it returns None only for an empty
            # or remote ref and otherwise hands back a path that may not exist.
            # Checking only its truthiness (the first version of this guard) is
            # a guard that cannot fire, which is worse than none because it
            # reads as covered.
            #
            # The stat is deliberately on the resolver's OWN answer rather than
            # a second path built here: this function's docstring records that
            # the previous existence check used a broader resolver than the
            # adapters did, so it could confirm a reference the worker then
            # could not open. One resolver, then stat what it said.
            import os

            resolved = _resolve_ref_to_disk(ref_path)
            if not resolved or not os.path.exists(resolved):
                # Named, not silent: this is the one miss an operator can
                # actually act on, by fetching the clip.
                return None, (
                    "reserved row %s: its reference is not on this machine "
                    "(%s); taking the ordinary draw"
                    % (row.voice_ref_id, ref_path))
        return _deliver(row, row.voice_ref_id)

    voice_ref_id = recurring_character_voice(character_key, engine)
    if not voice_ref_id:
        return None, "no %s mapping for %s" % (engine, character_key)

    matches = [e for e in (bank_entries or ())
               if e.voice_ref_id == voice_ref_id and e.engine == engine]
    if len(matches) != 1:
        return None, "%s/%s matched %d bank rows" % (
            engine, voice_ref_id, len(matches))

    return _deliver(matches[0], voice_ref_id)

class CastLock:
    """Registered as ``OTR_CastLock``. Single v2 ledger authority."""

    DESCRIPTION = (
        "Locks character and announcer voice assignments between the script freeze "
        "cascade and audio synthesis. Maps cast members to reference audio from the voice "
        "bank, enforces language compatibility, and stamps the canonical ledger consumed "
        "by downstream audio nodes. Adjust the casting policy, toggle voice reuse when "
        "casts exceed the bank, or set default voice engines for characters and the announcer."
    )

    CATEGORY = "OldTimeRadio/v2/audio"
    FUNCTION = "lock"
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("ledger_json", "cast_lock_revision", "cast_report", "done")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # C-5: no IO. Import-time side-effect-free. delivery_profile is no longer
        # a surfaced widget (single option "neutral" in v2); lock() keeps the
        # kwarg default and still validates + stamps it.
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": (
                        "Frozen v2 ledger JSON from OTR_LedgerFreezeCascade "
                        "(node 62 slot 1). CastLock rewrites it into the "
                        "canonical ledger_json; the legacy raw path keeps "
                        "reading this untouched string."
                    ),
                }),
            },
            "optional": {
                "cast_voice_policy": (list(_CAST_POLICIES), {
                    # Matches the shipped graph.
                    "default": "auto_registry",
                    "tooltip": (
                        "Who picks the voices. auto_registry casts them from "
                        "the bank, the same way every time; preserve_ledger "
                        "keeps whatever the writer already assigned."
                    ),
                }),
                "allow_voice_reuse": ("BOOLEAN", {
                    # Matches the shipped graph. Off means a cast larger than
                    # the bank stops the render rather than doubling a voice.
                    "default": True,
                    "tooltip": (
                        "Let two characters share a voice when the bank runs "
                        "out of distinct ones. Off stops the render instead."
                    ),
                }),
                "char_voice_engine": (list(_CHAR_VOICE_ENGINES), {
                    "default": "auto",
                    "tooltip": (
                        "Which engine speaks the characters. auto keeps the "
                        "shipped Kokoro voice. Character Voices inherits this "
                        "stamp -- there is no second engine dropdown."
                    ),
                }),
                "announcer_voice_engine": (list(_ANNOUNCER_VOICE_ENGINES), {
                    "default": "auto",
                    "tooltip": (
                        "Which engine reads the announcer. auto keeps the "
                        "shipped Kokoro voice. Announcer Voice inherits this "
                        "stamp -- there is no second engine dropdown."
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional ordering signal (wire an upstream done).",
                }),
                # S5 platform-portability (2026-07-10): explicit voice device
                # (append-only; widget slot 5). Stamped as meta.voice_device
                # (S4) so every voice adapter + theme music reads ONE truth;
                # the per-adapter waterfalls are gone.
                # Core's host-detected vocabulary plus the legacy names. See
                # nodes/_otr_shared/device_options.py for why both halves exist.
                "voice_device": (_DEVOPTS.device_options(), {
                    "default": _DEVOPTS.DEFAULT_DEVICE_OPTION,
                    "tooltip": "Device for local voice and music engines. "
                               "'default' asks ComfyUI what this machine has "
                               "and RECORDS what it chose; an explicit device "
                               "is never second-guessed and still fails loud "
                               "if it is not there.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # D: local-disk only; casting/bank checks are fail-closed on the lock()
        # path, not here, so a box-fresh graph validates clean.
        return True

    # ------------------------------------------------------------------ #
    # voice_bank is no longer a widget. The kwarg stays for direct Python
    # callers and for an explicit override; Comfy graphs omit it, so the
    # default MUST be None -- a leftover "default" string would force the
    # indextts2 bank onto a Kokoro dropdown (VoiceCastingError).
    def lock(self, script_json, voice_bank=None,
             cast_voice_policy="preserve_ledger", delivery_profile="neutral",
             allow_voice_reuse=False, char_voice_engine="auto",
             announcer_voice_engine="auto", gate_in="",
             voice_device="cuda"):
        from . import _otr_ledger_consumers as _OTRLC
        from ._otr_delivery_profiles import (
            DELIVERY_PROFILE_VERSION, get_delivery_profile,
        )

        led = _OTRLC.load_ledger(script_json)
        # CANONICAL REPLAY (campaign item 0): BEFORE the freeze gate, the
        # revision increment, the Bark assignment and every model resolution.
        # The imported ledger's cast rows are the source's; nothing changes.
        from .production_ledger import replay_descriptor as _replay_descriptor
        if _replay_descriptor(led.get("meta") or {}):
            _rev = int((led.get("meta") or {}).get("cast_lock_revision") or 0)
            log.warning("[OTR_CastLock] REPLAY: cast preserved as frozen "
                        "(revision %d), no voice assignment, no model", _rev)
            return (json.dumps(led, ensure_ascii=True, separators=(",", ":")),
                    _rev, "cast_lock: replay pass-through", "cast_lock:replay")
        # Freeze-halt + VRAM-recovery gate, re-homed here from the legacy audio
        # nodes (audio clean-break). CastLock runs first in the v2 audio chain
        # (CastLock -> CharacterVoices -> Announcer -> Theme), so one gate covers
        # every downstream audio engine instead of one copy per legacy node.
        self._enforce_freeze_gate(led.get("meta") or {})
        get_delivery_profile(delivery_profile)  # fail-closed on unknown profile
        cast = led.get("cast") or []
        report: list = []

        # Cheap char_id-subset validator (E.0): duplicate char_id fails before
        # any casting / model load.
        self._assert_unique_char_ids(cast)

        meta = led.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            led["meta"] = meta
        revision = int(meta.get("cast_lock_revision") or 0) + 1
        # STEP 1 (plan 5.2): the revision is stamped BEFORE any route is
        # resolved. A qualified re-pin can raise, and when it does the ledger
        # must already carry the revision that attempted it -- a failure with no
        # revision on it is a failure nobody can locate afterwards. The rest of
        # the CastLock-owned meta is stamped after casting, as before.
        meta["cast_lock_revision"] = revision

        # Concrete engines FIRST, then banks. `auto` is not a YAML engine --
        # `voice_bank_for_engine("char_voice", "auto")` would climb rank_chain
        # and land on indextts2/default, which is the opposite of the shipped
        # Kokoro dropdown. 4a/4b inherit these stamps; they have no engine widget.
        char_voice_engine = str(char_voice_engine or "auto").strip() or "auto"
        if char_voice_engine == "auto":
            char_voice_engine = _DEFAULT_CHAR_ENGINE
        announcer_voice_engine = self._resolve_announcer_engine(
            announcer_voice_engine)

        language_iso = _require_language_engines(
            meta, char_voice_engine, announcer_voice_engine)

        # Sprint 2 (a): CastLock OWNS bark voice casting. The writer no longer
        # stamps voice_preset -- it persists cast_seed in meta.cast_contract and
        # CastLock replays the deterministic picker (byte-identical) and stamps
        # the bark voices here, then runs the relocated voice invariants (Gate 1,
        # formerly in lock_cast). Runs regardless of cast_voice_policy (the policy
        # governs the clip-engine voice bank, not bark casting).
        if language_iso == "en":
            self._assign_bark_voices(
                cast, meta, report,
                announcer_voice_engine=announcer_voice_engine,
                char_voice_engine=char_voice_engine)

        # STEP 2 (plan 5.2): resolve the bank and stamp the engine metadata ONCE,
        # for BOTH modes, before any route is looked at. It used to happen inside
        # each branch, which meant a route could not be proved against the engine
        # that was going to render it -- the agreement check needs the engine in
        # hand first.
        char_bank = self._bank_following_engine(
            "char_voice", char_voice_engine, voice_bank)
        ann_bank = self._bank_following_engine(
            "announcer_voice", announcer_voice_engine)

        bank_entries = None
        if cast_voice_policy == "auto_registry":
            from ._otr_voice_bank import load_voice_bank
            bank_entries, _bank_sha = load_voice_bank()
        target_engine, announcer_engine = self._stamp_voice_engine_selection(
            led, char_bank, ann_bank, char_voice_engine, announcer_voice_engine,
            bank_entries=bank_entries, voice_device=voice_device)

        if cast_voice_policy == "auto_registry":
            self._auto_registry(
                led, cast, char_bank, allow_voice_reuse, report,
                char_voice_engine=char_voice_engine,
                announcer_voice_engine=announcer_voice_engine,
                voice_device=voice_device,
                bank_entries=bank_entries,
                target_engine=target_engine,
                announcer_engine=announcer_engine,
                ann_bank=ann_bank,
                language=language_iso)
        else:
            # STEP 5 (plan 5.2): in preserve_ledger ONLY the claimed row changes.
            # Every other row keeps the bytes it arrived with -- that is the
            # mode's whole contract, and an explicit re-pin is not a licence to
            # re-cast the cast.
            claimed = self._apply_recurring_character_voices(
                cast, char_voice_engine, language_iso, report)
            report.append(
                f"preserve_ledger: {len(cast) - claimed} cast entries preserved "
                f"(no re-cast)"
            )

        # STEP 6 (plan 5.2): the ordinary meta stamp and durable save happen ONCE,
        # here, after casting -- unchanged. cast_lock_revision was stamped up top
        # (step 1) and is not re-written.
        meta["cast_voice_policy"] = cast_voice_policy
        meta["delivery_profile_id"] = delivery_profile
        meta["delivery_profile_version"] = DELIVERY_PROFILE_VERSION
        meta["voice_bank_id"] = char_bank

        # STEP 3 (NO-FALLBACK rip, operator 2026-07-03): node-80 OUTPUT voice
        # resolution. The cast presets have now been assigned (replay + optional
        # auto_registry); guarantee -- BEFORE the ledger leaves CastLock for the
        # TTS nodes -- that no speaker_role='character' line reaches node 81
        # (OTR_BatchCharacterVoices) without a resolvable voice. FAIL LOUD: a
        # preset-less character row or a true orphan RAISES VoiceCastingError (no
        # synthesized-identity / orphan-reassign fallback); the ONLY repair kept is
        # re-routing a mis-stamped announcer line (a routing correction, not a
        # fallback). Runs UNCONDITIONALLY, independent of cast_seed.
        for _note in self._resolve_character_voices_fail_soft(
            cast, led.get("lines") or []
        ):
            log.warning("[CastLock] announcer reroute: %s", _note)
            report.append(f"announcer reroute: {_note}")

        report.insert(0, f"cast_lock_revision={revision} policy={cast_voice_policy}")

        # S2 durable persistence (credits enrichment 2026-07-03): everything
        # above stamped the LOCAL wire ledger (``led``), not the singleton --
        # a bare get_ledger().save() would persist EMPTY/stale state. Copy the
        # final cast (voice_ref_id / voice_engine / commercial_clean per
        # entry -- the DELIVERED voices the credits roll must read) plus the
        # CastLock-owned meta keys into the singleton and save LOUDLY
        # (raises LedgerStampError on save failure; test-mode injects
        # in-memory only).
        from .production_ledger import stamp_durable
        stamp_durable(
            sections={"cast": led.get("cast") or []},
            meta_updates={
                k: meta[k] for k in (
                    "cast_lock_revision", "cast_voice_policy",
                    "delivery_profile_id", "delivery_profile_version",
                    "voice_bank_id",
                    "char_voice_engine", "announcer_voice_engine",
                ) if k in meta
            },
            source="cast_lock",
        )

        ledger_json = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
        done = f"cast_lock:done:rev={revision}:policy={cast_voice_policy}"
        return (ledger_json, int(revision), "\n".join(report), done)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _enforce_freeze_gate(meta) -> None:
        """Enforce the structural/safety freeze and recover writer VRAM.

        Current freeze failures are only genuine ledger corruption --
        STRUCTURAL, and nothing else. Spoken-safety block classes went with the
        content-guardrail rip (2026-08-05); subjective quality block classes and
        their escape hatch were retired before that. A missing verdict remains
        compatible with legacy ledgers.

        If writer teardown reported an unload failure, attempt one defensive
        unload before the audio chain claims VRAM.
        """
        verdict = (meta or {}).get("freeze_verdict")
        if verdict == "needs_full_rerun":
            bypass = otr_env.get("OTR_BYPASS_FREEZE_HALT", "0") == "1"
            if bypass:
                log.warning(
                    "[CastLock] FREEZE HALT BYPASSED (OTR_BYPASS_FREEZE_HALT=1); "
                    "casting a structurally flagged ledger for operator "
                    "diagnostics only. See BUG-LOCAL-276."
                )
            else:
                raise ValueError(
                    "OTR_CastLock: freeze cascade stamped "
                    "freeze_verdict='needs_full_rerun' for structural ledger "
                    "corruption. Refusing to cast/render. "
                    "Set OTR_BYPASS_FREEZE_HALT=1 only for operator diagnostics. "
                    "See BUG-LOCAL-276."
                )

        if (meta or {}).get("freeze_unload_ok") is False:
            log.warning(
                "[CastLock] meta.freeze_unload_ok=False -- cascade teardown "
                "reported unload_llm failure; attempting one defensive unload "
                "before the audio chain claims VRAM"
            )
            try:
                from ._otr_model_loader import unload_llm

                unload_llm()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[CastLock] defensive unload_llm raised %r; proceeding", exc
                )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assign_bark_voices(cast, meta, report,
                             announcer_voice_engine="auto",
                             char_voice_engine="bark") -> None:
        """Sprint 2 (a): stamp bark voice_preset onto the cast by REPLAYING the
        writer's deterministic picker -- ONLY when the character engine is
        Bark. A kokoro / google_tts / elevenlabs request must not re-inject
        v2/* identities the writer left empty. Also owns the ANNOUNCER's bark
        preset (2026-08-24) when Bark is the resolved announcer engine.

        Direct helper callers default ``char_voice_engine="bark"`` so replay
        parity tests keep exercising the Bark owner. ``lock()`` always passes
        the resolved concrete engine.

        The writer persists ``cast_seed`` (OS-entropy per episode) in
        ``meta.cast_contract`` and no longer stamps voice_preset itself.
        ``replay_voice_assignment`` reconstructs the exact picker sequence keyed
        on that cast_seed -- byte-identical to what the writer used to assign
        (pinned by tests/test_cast_voice_replay_parity.py) -- and we stamp it
        onto the bark (non-ANNOUNCER) rows by char_id. The relocated Gate 1 voice
        invariants then run HERE, after assignment.

        A ledger with no persisted cast_seed (legacy graph / minimal test
        fixture) cannot be replayed; character voice_preset is preserved
        untouched, and -- UNLESS the announcer is also drawing a bark preset
        this call -- the invariants are skipped too, exactly as before this
        function grew announcer support: a missing/malformed character
        voice_preset in that case is NOT this function's concern to raise on;
        `lock()`'s STEP 3 (`_resolve_character_voices_fail_soft`) is the
        actual, more specific NO-FALLBACK gate for that (raises
        `VoiceCastingError`, a different exception with a different message --
        pinned by tests/test_cast_lock.py::test_seedless_missing_preset_fails_loud).
        Running Gate 1 unconditionally would preempt that gate with the wrong
        exception type for a case this function never touched.

        The announcer draw does NOT depend on cast_seed -- it must run under
        `preserve_ledger` too (the default cast_voice_policy), where
        `_auto_registry` never runs at all -- so it is not gated on the
        character-replay branch below, and it is the one case where the
        invariants DO still need to run even with no cast_seed (this function
        just stamped a fresh preset onto the announcer row, so its own
        correctness is this function's concern).
        """
        from . import _otr_casting as _OTRCAST
        from ._otr_text_delivery import CONTENT_OWNED, delivery_mode_for_meta

        announcer_engine = CastLock._resolve_announcer_engine(
            announcer_voice_engine)
        char_engine = str(char_voice_engine or "bark").strip() or "bark"
        if char_engine == "auto":
            char_engine = _DEFAULT_CHAR_ENGINE
        content_owned = delivery_mode_for_meta(meta) == CONTENT_OWNED

        if char_engine != "bark":
            # Do not replay or preserve Bark character identity on a
            # non-Bark request. Gate 1's v2/* contract is Bark-only.
            report.append(
                f"bark voices: skipped character replay "
                f"(char_voice_engine={char_engine})"
            )
            if announcer_engine == "bark":
                CastLock._assign_bark_announcer(cast, meta, report)
                _OTRCAST._assert_unique_bark_voices(cast)
            return

        # A content-owned lane builds its OWN character-cast rows; the writer's
        # seeded picker never ran, so there is no sequence to replay. Replaying
        # anyway would fabricate a cast that was never rolled. VERIFY what the
        # lane assigned for characters -- Gate 1 still runs.
        #
        # Source banks are NOT married to TTS engines (operator 2026-09-16).
        # `pick_announcer()` may DEFAULT the announcer row to Kokoro; CastLock
        # still honors `announcer_voice_engine` on every bank. A bark request
        # stamps a v2/* preset here. Refusing used to crash a live My Story
        # Bark listen at lock() after the writer finished.
        if content_owned:
            report.append(
                "bark voices: source bank owns character cast -- "
                "voice_preset preserved (no writer replay)"
            )
            if announcer_engine == "bark":
                CastLock._assign_bark_announcer(cast, meta, report)
            _OTRCAST._assert_unique_bark_voices(cast)
            _OTRCAST._assert_voice_preset_invariant(cast)
            return

        contract = (meta or {}).get("cast_contract") or {}
        cast_seed = contract.get("cast_seed")
        if cast_seed is not None:
            num_characters = int(contract.get("num_characters_request") or 0)
            lemmy_hit = bool(contract.get("lemmy_hit"))
            voices = _OTRCAST.replay_voice_assignment(
                cast_seed=int(cast_seed), num_characters=num_characters,
                lemmy_hit=lemmy_hit,
            )
            stamped = 0
            for row in cast:
                if not isinstance(row, dict):
                    continue
                cid = row.get("char_id")
                if cid in voices:
                    row["voice_preset"] = voices[cid]
                    stamped += 1
            report.append(
                f"bark voices: replayed cast_seed -> {stamped} voice_preset(s) "
                f"stamped (CastLock owns bark casting)"
            )
        else:
            report.append(
                "bark voices: no cast_seed in meta.cast_contract -- "
                "character voice_preset preserved (no replay)"
            )

        bark_announcer = announcer_engine == "bark"
        if bark_announcer:
            CastLock._assign_bark_announcer(cast, meta, report)

        # Gate 1 (relocated from the writer's lock_cast): every bark-delivered
        # row now carries a v2/* voice_preset, and no two bark rows share one.
        # Conditional on tts_model, not on identity alone -- a non-bark
        # announcer (the common case) is exempt exactly as before. Only run
        # when this call actually stamped something (character replay, or the
        # bark announcer draw just above) -- see the docstring: a no-op call
        # (no cast_seed, non-bark announcer) must defer entirely to `lock()`'s
        # own later, more specific NO-FALLBACK gate.
        if cast_seed is not None or bark_announcer:
            _OTRCAST._assert_unique_bark_voices(cast)
            _OTRCAST._assert_voice_preset_invariant(cast)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assign_bark_announcer(cast, meta, report) -> None:
        """Draw and stamp a ``v2/*`` Bark preset onto the ANNOUNCER row.

        Dynamic exclusion, not a standing reservation (r1 ruling, 2026-08-24):
        `VOICE_PROFILES` is 6 male / 4 female, and the 4-female pool has
        already been exhausted by a 3-female character cast once before
        (FIX-3, `config/cast_pools.py`). Reserving presets for the announcer
        would recreate that exhaustion, so the announcer draws from the SAME
        ten presets, excluding only what a character in THIS episode actually
        took.

        Seeded independently from the character stream (a distinct sha1
        discriminator, not a bare int reuse of ``episode_seed`` -- two
        ``random.Random`` instances seeded with the same integer are NOT
        independent, they replay the identical sequence), so this draw never
        perturbs -- and is never correlated with -- character voice picks.

        Clears stale cross-engine identity before stamping (mirrors
        `_stamp_row`'s "clear and stamp are atomic" contract): a re-locked
        episode that was previously cast with Kokoro left `voice_ref_id` /
        `voice_engine` on the row, and the credits roll reads those AHEAD of
        `voice_preset` -- an uncleared row would credit the old engine for
        audio Bark actually rendered.
        """
        import random

        from . import _otr_casting as _OTRCAST
        from ._otr_voice_node_common import coerce_int_seed

        # Relative first (production: ComfyUI loads this as part of the
        # ComfyUI-OldTimeRadio package -- `..config` resolves from `nodes`'
        # own __package__). Absolute fallback for tests, which add the repo
        # root to sys.path directly. This is the SAME two-tier shape already
        # used at :50-55 in this file and at 10+ other cast_pools call sites
        # (_otr_casting.py, _otr_voice_bank.py, _otr_voice_route.py,
        # _otr_scifi_news_pro.py) -- this one function was missing it.
        #
        # 2026-08-25 PBUG: a bare `from config import cast_pools` here (no
        # relative-first, no fallback) worked by ACCIDENT in every proof leg
        # run tonight via otr_canonical_api_run.py, because that script's
        # working directory happens to put the repo root on sys.path -- the
        # same accident that makes tests pass. It does NOT work under a real
        # ComfyUI Desktop install, where the pack is loaded as a submodule and
        # its own root is never added to sys.path as a bare entry. Introduced
        # in f3130f6d (bark-as-announcer, 2026-08-24) and reviewed by a full
        # 4-round kibitz arc that did not catch it -- the arc grounds panel
        # claims against the tree, it does not simulate a second install.
        try:
            from ..config import cast_pools as _POOLS  # type: ignore
        except ImportError:
            try:
                from config import cast_pools as _POOLS  # type: ignore
            except ImportError:
                # Fail soft HERE, fail closed DOWNSTREAM (this file's own
                # convention -- see _lemmy_voice_policy above). The
                # announcer row is left unstamped; _assert_voice_preset_invariant,
                # run by the caller right after this returns, raises a named,
                # actionable error instead of this function's own opaque
                # ModuleNotFoundError traceback.
                report.append(
                    "bark voices: cast_pools import failed (broken install?) "
                    "-- announcer left unstamped, downstream invariant will "
                    "raise"
                )
                return

        row = next((r for r in cast
                    if isinstance(r, dict) and _is_announcer_entry(r)), None)
        if row is None:
            report.append(
                "bark voices: announcer_voice_engine=bark but no ANNOUNCER "
                "row found -- nothing to stamp"
            )
            return

        taken = {r.get("voice_preset") for r in cast
                 if isinstance(r, dict) and r.get("voice_preset")
                 and not _is_announcer_entry(r)}
        pool = _POOLS.open_voice_pool(taken)

        episode_seed = coerce_int_seed((meta or {}).get("episode_seed"))
        seed_key = hashlib.sha1(
            f"bark_announcer:{episode_seed}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(seed_key)

        slot = _OTRCAST.EnsembleSlot(
            char_id=str(row.get("char_id") or "announcer"),
            name="ANNOUNCER",
            gender=str(row.get("gender") or ""),
            timbre="",
            role="announcer",
        )
        preset = _OTRCAST.python_assign_voice_preset(
            slot, available_voices=pool, rng=rng)

        # Gender coherence (kibitz r3 MUST-FIX, codex): python_assign_voice_preset
        # deliberately falls back to the FULL pool when the requested gender's
        # column is exhausted (`candidates = gender_pool or list(available_voices)`
        # in _otr_casting.py) -- and this codebase has hit that exhaustion for
        # real (FIX-3, 3-female character casts against the 4-female pool). A
        # fallback draw can therefore hand this row a preset of the OPPOSITE
        # gender from `row["gender"]`. Trusting the stale field here would be
        # exactly the "two correlated attributes, each locally plausible,
        # globally incoherent" class Bug Bible 10.08 exists for -- so derive
        # `presentation_gender` from what was ACTUALLY drawn, never from what
        # was merely requested.
        from ._otr_voice_bank import bark_preset_gender

        delivered_gender = bark_preset_gender(preset) or str(row.get("gender") or "")

        row["voice_preset"] = preset
        row["tts_model"] = "bark"
        row["voice_engine"] = "bark"
        row["voice_ref_id"] = ""
        row["commercial_clean"] = False
        row["voice_cast_fallback"] = ""
        row["presentation_gender"] = delivered_gender
        report.append(
            f"bark voices: announcer stamped {preset} "
            f"(excluded {len(taken)} character preset(s))"
        )
        if delivered_gender and delivered_gender != str(row.get("gender") or ""):
            report.append(
                f"bark voices: announcer gender pool exhausted -- requested "
                f"{row.get('gender')!r}, delivered {delivered_gender!r} "
                f"(presentation_gender stamped from the actual preset)"
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_character_voices_fail_soft(cast, lines) -> list:
        """STEP 3 (NO-FALLBACK rip, operator 2026-07-03): guarantee every
        speaker_role='character' line reaches node-81 with a resolvable voice, or
        FAIL LOUD. The name is retained for the call site; behavior is now fail-loud.

        Two behaviors:
          * (ROUTING -- KEPT) a speaker_role=='character' LINE whose char_id is the
            announcer marker (or names the ANNOUNCER) is a MIS-STAMPED announcer
            line -> re-stamp speaker_role='announcer' so it routes to
            OTR_AnnouncerVoice. This is a routing CORRECTION, not a fallback.
          * (FAIL LOUD) a non-ANNOUNCER character cast row with no voice_preset, OR
            a character LINE whose char_id matches no voiced cast row (a true
            orphan), RAISES VoiceCastingError. The old fail-soft repairs
            (synthesize a v2/en_speaker_* identity; reassign an orphan to another
            character) are RETIRED -- a missing/orphan voice is a writer/casting
            defect the operator must fix, never papered over (no silent swap).

        Mutates line dicts in place for the announcer reroute only. Returns LOUD
        routing notes for the cast report. RAISES VoiceCastingError on any
        unvoiceable character row/line.
        """
        from ._otr_voice_bank import VoiceCastingError

        rows_by_id: dict = {}
        announcer_ids: set = {"announcer"}
        char_rows: list = []  # non-ANNOUNCER character cast rows, in order
        for row in cast or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("char_id") or "")
            rows_by_id[cid] = row
            if _is_announcer_entry(row):
                announcer_ids.add(cid)
            else:
                char_rows.append(row)

        notes: list = []

        # (1) FAIL LOUD: every non-ANNOUNCER character cast row MUST carry a
        # resolvable voice identity by now (replay and/or auto_registry).
        # voice_ref_id counts: kokoro rows clear leftover v2/* and speak from
        # the bank reference. A row with neither is a casting defect -- raise,
        # never synthesize a fallback identity.
        for row in char_rows:
            if _row_has_resolvable_voice(row):
                continue
            cid = str(row.get("char_id") or "")
            raise VoiceCastingError(
                f"character cast row {cid!r} reached CastLock with no voice_preset "
                f"or voice_ref_id -- casting must assign one. NO synthesized "
                f"fallback identity (no-fallback rip)."
            )

        voiced_char_ids = sorted(
            str(r.get("char_id") or "") for r in char_rows
            if _row_has_resolvable_voice(r)
        )

        # (2 KEPT: routing) + (3 FAIL LOUD: orphan) over character LINES.
        for ln in lines or []:
            if not isinstance(ln, dict):
                continue
            if str(ln.get("speaker_role") or "").strip().lower() != "character":
                continue
            cid = str(ln.get("char_id") or "")
            row = rows_by_id.get(cid)
            if _row_has_resolvable_voice(row):
                continue  # resolves fine
            lid = ln.get("line_id")
            is_ann = cid in announcer_ids or (
                row is not None
                and str(row.get("name") or "").strip().upper() == "ANNOUNCER"
            )
            if is_ann:
                ln["speaker_role"] = "announcer"
                notes.append(
                    f"line {lid!r} char_id={cid!r}: mis-stamped announcer line "
                    f"-> re-routed to OTR_AnnouncerVoice (announcer model)"
                )
                continue
            raise VoiceCastingError(
                f"character line {lid!r} char_id={cid!r} has no voiced cast row "
                f"(true orphan). NO orphan-reassignment fallback (no-fallback rip) "
                f"-- the writer/casting must give this line a real voiced character."
            )
        return notes

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assert_unique_char_ids(cast) -> None:
        seen = set()
        for entry in cast:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("char_id")
            if not cid:
                continue
            if cid in seen:
                raise ValueError(
                    f"OTR_CastLock: duplicate char_id {cid!r} in cast -- the "
                    f"writer cast contract is violated (fails before any cast)"
                )
            seen.add(cid)

    # ------------------------------------------------------------------ #
    def _auto_registry(self, led, cast, voice_bank, allow_voice_reuse, report,
                       char_voice_engine="auto",
                       announcer_voice_engine="auto",
                       voice_device="cuda",
                       bank_entries=None,
                       target_engine=None,
                       announcer_engine=None,
                       ann_bank=None,
                       language="en"):
        """Re-cast the registry rows.

        ``bank_entries`` / ``target_engine`` / ``announcer_engine`` /
        optional because this method is also called directly, with five
        positional arguments, and must keep resolving its own inputs when it is.
        """
        from ._otr_voice_bank import (
            CASTING_POLICY_VERSION, _SEEDED_ANNOUNCER_ENGINES, VoiceCastingError,
            announcer_voice_ref, assign_voice_for_slot,
            filter_voices_for_language, gender_agnostic_fallback_ref,
            accent_timbre_tags,
            load_voice_bank, voice_speaks_language,
            voice_ref_usage_keys,
        )
        language = str(language or "en").strip() or "en"
        from ._otr_voice_node_common import coerce_int_seed

        if bank_entries is None:
            bank_entries, _bank_sha = load_voice_bank()
        meta = led.get("meta") or {}
        if meta.get("episode_seed") is None:
            # SILENCE IS HOW THIS HID. A missing seed folds through
            # coerce_int_seed(None) to one constant, so every episode drew the
            # same announcer and the same character voices while every unit test
            # stayed green -- measured over 14 published episodes before the
            # writer began stamping it. Never fail on this; a legacy ledger must
            # still render. Just stop it being invisible.
            log.warning(
                "[OTR_CastLock] meta.episode_seed is ABSENT -- voice and "
                "announcer draws fall back to a CONSTANT seed, so this episode "
                "will cast identically to every other seedless one. Expected on "
                "pre-2026-08-05 ledgers; on a fresh render it means the writer "
                "did not stamp it.")
        episode_seed = coerce_int_seed(meta.get("episode_seed"))
        # VC chunk 3 (2026-06-22): the writer stamps per-character voice-fit
        # facts (timbre/age_band) the frozen cast ROW schema cannot carry. Match
        # the bank on those, not just gender. Legacy ledgers without the stamp
        # fall back to the (empty) entry-level fields -> behavior unchanged.
        voice_slots = meta.get("cast_voice_slots") or {}
        # The `voice_decisions` local that used to read
        # `meta.voice_cast_decision` here was removed 2026-08-28: it was
        # assigned and never used once the hybrid LLM voice-fit branch went
        # (2026-08-18). The durable ledger KEY is untouched -- it is still
        # stamped and still verified -- only this dead read is gone.
        # announcer_engine is the sentinel: the resolver never returns None for
        # it, while target_engine legitimately can be None (a preset-only bank).
        # Direct callers still pass the default "auto"; lock() already resolved.
        # voice_bank_for_engine rejects "auto" (it is not a YAML engine).
        char_voice_engine = str(char_voice_engine or "auto").strip() or "auto"
        if char_voice_engine == "auto":
            char_voice_engine = _DEFAULT_CHAR_ENGINE
        announcer_voice_engine = self._resolve_announcer_engine(
            announcer_voice_engine)
        # Direct callers still pass the old CastLock widget default "default".
        # Banks follow the concrete engine; a leftover id kokoro does not own
        # must not raise (MACHINE_MATRIX 2026-08-31).
        voice_bank = self._bank_following_engine(
            "char_voice", char_voice_engine, voice_bank)
        if announcer_engine is None:
            if ann_bank is None:
                ann_bank = self._bank_following_engine(
                    "announcer_voice", announcer_voice_engine)
            target_engine, announcer_engine = self._stamp_voice_engine_selection(
                led, voice_bank, ann_bank, char_voice_engine, announcer_voice_engine,
                bank_entries=bank_entries, voice_device=voice_device)
        if target_engine is None:
            report.append(
                f"auto_registry: voice_bank {voice_bank!r} has no character "
                f"reference engine; character voices preserved"
            )

        if target_engine and bank_entries is not None:
            lang_pool = filter_voices_for_language(
                [e for e in bank_entries if e.engine == target_engine],
                language)
            if not lang_pool:
                raise VoiceCastingError(
                    "naming pool size: no %r voices for engine %r"
                    % (language, target_engine))
            n_slots = sum(1 for e in cast if isinstance(e, dict))
            if len({e.voice_ref_id for e in lang_pool}) < n_slots:
                # Thin rows (French 1, Italian 2): reuse in-pool for a gender
                # the row CAN serve. The ladder still prefers unused voices
                # first. A gender the row cannot serve at all is a different
                # case and is handled in the draw loop below, where it borrows
                # the same gender from English rather than taking whichever
                # voice the row happens to have -- this comment used to say
                # "never borrow English", and that policy is what put a
                # woman's voice on Horatio (operator 2026-09-19).
                allow_voice_reuse = True

        announcer_ref = None
        has_announcer = any(
            isinstance(entry, dict) and _is_announcer_entry(entry)
            for entry in cast
        )
        if has_announcer and announcer_engine in _SEEDED_ANNOUNCER_ENGINES:
            announcer_ref = announcer_voice_ref(
                announcer_engine, bank=bank_entries, episode_seed=episode_seed,
                language=language)

        used: set = set()
        def _mark_used(ref) -> None:
            used.update(voice_ref_usage_keys(ref))

        # Rows this lock actually re-cast. The tier sweep at the end reports only
        # on these, because `unrouted` is a claim about a DRAW -- and a row the
        # caster never reached did not take one.
        stamped_this_lock: set = set()

        def _stamp_row(entry, ref, *, fallback: str = "") -> None:
            """Stamp a cast row, clearing the CLAIMED row's stale cross-engine
            identity in the same operation.

            CLEAR AND STAMP ARE ATOMIC ON PURPOSE, and an earlier cut of this got
            it wrong in a way a QA pass reproduced live. Clearing up front, before
            the loop, meant a claimed row could be stripped and then fall through
            one of the loop's `continue`s -- a bank with no character engine, a row
            with no usable gender -- and end the lock carrying LESS identity than
            it arrived with, while a tier field said the ordinary draw had chosen
            it. The credits roll reads `voice_engine` / `voice_ref_id` ahead of
            `voice_preset`, so a Bark episode would have credited Lemmy to an
            ElevenLabs voice nobody heard. A row that is not re-cast is now left
            exactly as it arrived, in either mode.

            Every other row passes straight through: the normalizer is scoped to
            the one row this policy claims, so two hundred unrelated rows keep
            their bytes.
            """
            if not _is_announcer_entry(entry) and _recurring_character_key(entry):
                cleared = _clear_stale_voice_identity(entry)
                if cleared:
                    report.append(
                        "  %s: cleared stale voice identity before re-stamp (%s)"
                        % (entry.get("char_id") or entry.get("name"),
                           ", ".join(cleared)))
            self._stamp(entry, ref, fallback=fallback)
            stamped_this_lock.add(id(entry))

        if (announcer_ref is not None and target_engine == announcer_engine
                and not allow_voice_reuse):
            _mark_used(announcer_ref)
        gated = 0
        for entry in cast:
            if not isinstance(entry, dict):
                continue
            char_id = str(entry.get("char_id") or "")

            # Ledger completeness: every row this caster CONSIDERS carries the
            # field, so a downstream reader never has to tell "cast normally"
            # apart from "field never written". Set BEFORE the announcer branch,
            # which has its own `continue` -- an announcer whose engine cannot
            # serve it is reported NOT cast and would otherwise be the one row
            # the caster touched without leaving a verdict. _stamp overwrites it;
            # rows that fall through keep the empty default. preserve_ledger is
            # deliberately not touched -- that mode's contract is byte-safety,
            # and a row the caster never ran on has no cast decision to report.
            entry.setdefault("voice_cast_fallback", "")

            if _is_announcer_entry(entry):
                if announcer_engine == "bark":
                    # Bark has zero bank rows (P2.2) -- the bank-ref lookup
                    # below can never serve it, and `_assign_bark_voices`
                    # already stamped the v2/* preset earlier in `lock()`,
                    # before this method ever runs. Report truthfully instead
                    # of falling into the try/except below, which would raise
                    # VoiceCastingError, get swallowed, and log a false
                    # "announcer NOT cast" for a row that was, in fact, cast.
                    report.append(
                        f"  {char_id or 'ANNOUNCER'}: announcer "
                        f"{entry.get('voice_preset')} (bark, stamped by "
                        f"_assign_bark_voices)"
                    )
                    continue
                try:
                    ref = announcer_ref or announcer_voice_ref(
                        announcer_engine, bank=bank_entries,
                        episode_seed=episode_seed, language=language)
                    _stamp_row(entry, ref)
                    announcer_clean = _delivered_commercial_clean(entry, ref)
                    gated += 0 if announcer_clean else 1
                    report.append(
                        f"  {char_id or 'ANNOUNCER'}: announcer {ref.voice_ref_id} "
                        f"({ref.engine}, clean={announcer_clean})"
                    )
                except VoiceCastingError as exc:
                    if announcer_engine == "google_tts":
                        raise
                    report.append(f"  {char_id or 'ANNOUNCER'}: announcer NOT cast -- {exc}")
                continue

            # THE RECURRING-CHARACTER ASSIGNMENT, after the announcer check and
            # before the ordinary draw. One table lookup, no tiers, no receipts:
            # a character named in RECURRING_CHARACTER_VOICES is delivered with
            # the catalogue voice that table names for this engine, and everyone
            # else takes the normal seeded selection.
            #
            # THE PIN IGNORES THE USED SET, exactly as the branch it replaced
            # did. ANNOUNCER may already hold `bm_george` -- it is a shared
            # catalogue row and both may have it. `_mark_used` runs AFTERWARDS so
            # later ordinary rows still see it as spoken for.
            _recurring_ref, _recurring_miss = _recurring_character_bank_ref(
                entry, target_engine, bank_entries, language)
            if _recurring_ref is not None:
                _stamp_row(entry, _recurring_ref, fallback="character_voice")
                _mark_used(_recurring_ref)
                gated += 0 if _delivered_commercial_clean(
                    entry, _recurring_ref) else 1
                report.append(
                    f"  {char_id}: {_recurring_ref.voice_ref_id} "
                    f"({_recurring_ref.engine}, recurring character)"
                )
                continue
            if _recurring_miss:
                report.append(
                    f"  {char_id}: recurring assignment missed "
                    f"({_recurring_miss}); taking the ordinary draw"
                )

            if target_engine is None:
                continue
            # SYNONYM-CANONICALIZED (item 8, 2026-08-06). THIS is the path that
            # actually runs: CastLock stamps voice_ref_id before any render, so
            # the render-time resolvers in _otr_voice_node_common find a stamped
            # id and never reach their own gender fallback. Fixing only those
            # left the real defect live -- a row recorded `woman` raised
            # VoiceCastingError here, was caught below, and took the
            # gender-agnostic draw. (It also fed the hybrid voice-fit branch's
            # validation until that branch was ripped on 2026-08-18; the scorer
            # is now the only consumer.)
            from ._otr_roster_gender import canonical_bank_gender
            gender = canonical_bank_gender(entry.get("gender"))
            # AN UNSTATED GENDER ON google_tts TAKES THE SEEDED DRAW (0j,
            # operator 2026-09-25: "random genders and leave it nebulous").
            # This used to refuse -- NO FALLBACK -- so a My Story character
            # whose author never said a gender stopped the render on the
            # Google lane while the same row took Kokoro's seeded,
            # gender-agnostic draw below. Provider voices are gendered, so
            # the pick is a coin the episode seed flips: deterministic, and
            # _otr_my_story still leaves the gender empty on purpose. A
            # STATED gender is still honoured with no cross-gender fallback
            # (see the re-raise in the except below).
            # THE HYBRID LLM VOICE-FIT BRANCH WAS HERE AND IS GONE (2026-08-18).
            # It read meta.voice_cast_decision, re-validated the LLM's proposed
            # voice_ref_id, and on success stamped it and `continue`d -- skipping
            # the deterministic scorer below entirely. That is why the scorer
            # handled only ~4% of production casting.
            #
            # `meta.voice_cast_decision` is still STAMPED (empty) by the writer
            # and still verified downstream, so a legacy ledger carrying real
            # decisions loads without complaint -- its proposals are simply
            # ignored now, and the scorer casts the row. That is the intended
            # behaviour, not a fallback: the LLM had no information the scorer
            # lacks. CastLock itself no longer reads the key at all (the dead
            # local above went 2026-08-28); an earlier version of this comment
            # said it did.

            # Prefer the writer's voice-fit slot (timbre/age_band); fall back to
            # any entry-level fields for legacy ledgers without the stamp.
            slot = voice_slots.get(char_id) or {}
            slot_timbre = slot.get("timbre") or entry.get("timbre") or ()
            # THE ROW'S DECLARED ACCENT IS A TIMBRE PREFERENCE. `lemmy_row()`
            # has stamped `accent` since it was written and no caster ever read
            # it -- the selector scores `timbre`, so the two never met. UNIONED,
            # not substituted: adding tags can only let a better-fitting voice
            # win, never stop a previously-matching one from matching, and the
            # selector's ladder drops the dimension entirely when nothing in
            # the bank carries it. So a character whose accent no voice can
            # speak still draws a gender-correct voice, unchanged.
            _accent_tags = accent_timbre_tags(
                entry.get("accent"), bank=bank_entries, engine=target_engine)
            if _accent_tags:
                slot_timbre = tuple(dict.fromkeys(
                    tuple(slot_timbre) + tuple(_accent_tags)))
            slot_age = str(slot.get("age_band") or entry.get("age_band") or "")
            try:
                if not gender:
                    # Cast the same real open-pool reference that the renderer
                    # would select, so the wire, ledger and credits name it.
                    # This does not invent a gender for the character.
                    raise VoiceCastingError(f"{char_id}: source gender unspecified")
                ref = assign_voice_for_slot(
                    role="char_voice",
                    engine=target_engine,
                    char_id=char_id,
                    gender=gender,
                    timbre=tuple(slot_timbre),
                    age_band=slot_age,
                    episode_seed=episode_seed,
                    casting_policy_version=CASTING_POLICY_VERSION,
                    allow_voice_reuse=allow_voice_reuse,
                    used_voice_ref_ids=used,
                    bank=bank_entries,
                    language=language,
                )
            except VoiceCastingError as exc:
                if target_engine == "google_tts" and gender:
                    raise
                # BORROW THE GENDER FROM ENGLISH BEFORE GIVING UP ON IT
                # (operator 2026-09-19, option A). Kokoro ships ONE French
                # voice, ff_siwis, and it is female; every other admitted row
                # carries at least one voice of each gender. The selector
                # fails closed on gender by design, so a French male raised
                # here and the gender-agnostic draw below -- a uniform pick
                # over the French pool -- handed him Siwis. Measured on the
                # real ledger of the first French Hamlet leg: HORATIO and
                # MARCELLUS both `gender_unservable`, both `ff_siwis`, both
                # presenting female beside a bearded still. The operator heard
                # it in thirty seconds; every structural check had passed.
                #
                # Same gender, English pool, same deterministic ladder, same
                # used-set -- so two French men draw two DIFFERENT English
                # men. The timbre is wrong-accented; the man is a man. That is
                # the trade the operator chose over a woman's voice on a
                # bearded face, and the render path takes lang_code from the
                # episode row, not the voice id, so a borrowed voice still
                # speaks with French phonemes.
                #
                # CONFINED EXPLICITLY, NOT BY ASSUMPTION. The selector raises
                # for TWO reasons -- "no gender-matching reference exists" and
                # "all matching references are already used" -- and the first
                # cut of this tier fired on both. The second one is the
                # default Spanish and Portuguese shape: three voices, three
                # cast rows, so reuse stays off; the announcer takes the one
                # woman (`ef_dora` is tagged preferred_announcer) and marks
                # her used; the next woman raises for the SECOND reason and
                # was handed an English voice. Measured on the working tree:
                # eight of eight seeds. That is not the operator's ruling --
                # it changes two languages he never heard -- and the report
                # line would have said "has no 'es' voice" while Dora sat on
                # the announcer row. So the gate is the condition itself:
                # does the language row carry ANY voice of this gender? Only
                # when it does not is the English pool consulted. A served
                # gender whose voices are merely taken keeps the pre-existing
                # path, whatever its own faults, until someone rules on it.
                # `other` still falls through -- no English row carries it.
                lang_can_serve = any(
                    e.engine == target_engine
                    and voice_speaks_language(e, language)
                    and canonical_bank_gender(getattr(e, "gender", "")) == gender
                    for e in (bank_entries or ())
                )
                # REUSE HER OWN LANGUAGE'S VOICE BEFORE TAKING A MAN'S. When
                # the row DOES carry this gender but every one of them is
                # already spoken for, the old path fell to the gender-agnostic
                # draw -- a uniform pick over the whole language pool, most of
                # which is the other gender. Measured on the default Spanish
                # shape (3 voices, 3 rows, so reuse is off): the announcer
                # takes `ef_dora`, and ANA -- a woman -- was stamped `em_alex`
                # and presented MALE. That is the operator's own complaint
                # inverted, in a language he has not heard yet.
                #
                # A woman sharing the narrator's voice is worse than two
                # distinct women and far better than a woman with a man's
                # voice, so the same-gender reuse is tried first and the
                # gender-agnostic draw stays as the last resort it was written
                # to be. The distinctness line above reports the sharing.
                reused = None
                if gender and lang_can_serve and not allow_voice_reuse:
                    try:
                        reused = assign_voice_for_slot(
                            role="char_voice", engine=target_engine,
                            char_id=char_id, gender=gender,
                            timbre=tuple(slot_timbre), age_band=slot_age,
                            episode_seed=episode_seed,
                            casting_policy_version=CASTING_POLICY_VERSION,
                            allow_voice_reuse=True,
                            used_voice_ref_ids=used,
                            bank=bank_entries, language=language,
                        )
                    except VoiceCastingError:
                        reused = None
                if reused is not None:
                    _stamp_row(entry, reused, fallback="gender_reused_in_lang")
                    _mark_used(reused)
                    gated += 0 if _delivered_commercial_clean(
                        entry, reused) else 1
                    report.append(
                        f"  {char_id}: {reused.voice_ref_id} "
                        f"({reused.engine}, every {language!r} {gender} voice "
                        f"already cast -- reused in-language rather than "
                        f"crossing gender)"
                    )
                    continue
                borrowed = None
                if gender and language != "en" and not lang_can_serve:
                    try:
                        borrowed = assign_voice_for_slot(
                            role="char_voice",
                            engine=target_engine,
                            char_id=char_id,
                            gender=gender,
                            timbre=tuple(slot_timbre),
                            age_band=slot_age,
                            episode_seed=episode_seed,
                            casting_policy_version=CASTING_POLICY_VERSION,
                            allow_voice_reuse=allow_voice_reuse,
                            used_voice_ref_ids=used,
                            bank=bank_entries,
                            language="en",
                        )
                    except VoiceCastingError:
                        borrowed = None
                if borrowed is not None:
                    _stamp_row(entry, borrowed, fallback="gender_borrowed_en")
                    _mark_used(borrowed)
                    gated += 0 if _delivered_commercial_clean(
                        entry, borrowed) else 1
                    report.append(
                        f"  {char_id}: {borrowed.voice_ref_id} "
                        f"({borrowed.engine}, gender {gender!r} has no "
                        f"{language!r} voice -- borrowed from English, "
                        f"same gender)"
                    )
                    continue
                # The bank cannot serve this row's gender -- 'other' is 20% of
                # every roll and the bank carries zero rows for it. Previously
                # the row was reported "NOT cast" and left with NO voice_ref_id,
                # and the render path then drew a gender-agnostic reference of
                # its own. The ledger therefore did not name the voice that
                # actually spoke. Stamp the SAME draw the render will make, so
                # the ledger is complete and honest. This is a ledger fix, not a
                # content gate: no refusal, no gender restriction.
                fallback_ref = gender_agnostic_fallback_ref(
                    bank_entries, engine=target_engine, char_id=char_id,
                    episode_seed=episode_seed, role="char_voice", used=used,
                    language=language,
                )
                if fallback_ref is None:
                    report.append(f"  {char_id}: NOT cast -- {exc}")
                    continue
                _stamp_row(entry, fallback_ref, fallback=(
                    "gender_unservable" if gender else "gender_unspecified"))
                _mark_used(fallback_ref)
                gated += 0 if _delivered_commercial_clean(
                    entry, fallback_ref) else 1
                reason = f"gender {gender!r} unservable" if gender else "source gender unspecified"
                report.append(
                    f"  {char_id}: {fallback_ref.voice_ref_id} "
                    f"({fallback_ref.engine}, {reason} -- "
                    f"gender-agnostic reference)"
                )
                continue
            _stamp_row(entry, ref)
            _mark_used(ref)
            drawn_clean = _delivered_commercial_clean(entry, ref)
            gated += 0 if drawn_clean else 1
            report.append(
                f"  {char_id}: {ref.voice_ref_id} ({ref.engine}, "
                f"clean={drawn_clean})"
            )

        # VOICE DISTINCTNESS, reported so a collision can never pass silently.
        # The credits roll prints "N VOICES ACCOUNTED FOR", and it counts
        # ASSIGNMENTS: three rows, three stamps, three accounted for -- while
        # all three were ff_siwis. A structural check that cannot see a
        # collision is the same blindness as a wrong-mouth scan that only
        # looks for speakers who survived parsing. This counts DISTINCT ids
        # across the character rows this lock actually stamped and says so in
        # the report the leg log carries.
        #
        # A REPORT LINE, NOT A GATE. A thin language row collides legitimately
        # -- Italian has one voice per gender, so two Italian women share one
        # -- and the standing rule is no gates on models. The line exists so
        # the collision is READ, by whoever reads the log, instead of being
        # inferred from a bearded man sounding like a woman.
        # THE ANNOUNCER IS COUNTED, and excluding him was the hole a contrarian
        # pass found in the first cut of this line. Under `allow_voice_reuse`
        # -- which every thin row turns on -- the announcer's reference is
        # deliberately NOT marked used (see the guard above), so a French
        # episode with a narrator and ONE female lead puts both on `ff_siwis`.
        # With the announcer excluded and a `> 1` floor, that shape reported
        # nothing at all: the count saw a single character row and fell
        # silent. "The narrator and the heroine are the same person" is the
        # operator's own complaint relocated, not a corner case -- announcer
        # plus one same-gender lead is a typical cast.
        #
        # Announcer-vs-character collisions are named separately from
        # character-vs-character ones because they mean different things: the
        # first is a narrator who sounds like the cast, the second is a cast
        # that sounds like itself. Both are report lines, never gates -- a thin
        # row collides legitimately and the standing rule is no gates on
        # models.
        stamped = [
            e for e in cast
            if isinstance(e, dict) and id(e) in stamped_this_lock
            and e.get("voice_ref_id")
        ]
        char_ids = [str(e.get("voice_ref_id"))
                    for e in stamped if not _is_announcer_entry(e)]
        ann_ids = {str(e.get("voice_ref_id"))
                   for e in stamped if _is_announcer_entry(e)}
        if stamped:
            all_ids = char_ids + sorted(ann_ids)
            line = ("  voice distinctness: %d distinct voice(s) across %d "
                    "stamped row(s) (%d character, %d announcer)"
                    % (len(set(all_ids)), len(all_ids), len(char_ids),
                       len(ann_ids)))
            if len(char_ids) > 1 and len(set(char_ids)) == 1:
                line += (" -- VOICE COLLISION: every character on one voice "
                         "(%s)" % char_ids[0])
            shared = ann_ids & set(char_ids)
            if shared:
                line += (" -- ANNOUNCER COLLISION: the narrator shares %s "
                         "with a character" % ", ".join(sorted(shared)))
            report.append(line)

        # LEDGER COMPLETENESS FOR THE TIER, and this sweep is why the three fields
        # can be read as an enumeration downstream. The claimed row can leave the
        # loop above by several doors -- the hybrid voice-fit, the gender-agnostic
        # fallback, the ordinary draw -- and a field written at only some of them
        # is worse than no field at all.
        #
        if gated:
            report.append(
                f"auto_registry: {gated} assigned voice(s) are known-gated "
                f"(reference clip and/or model licence is not commercial-clean) "
                f"-- non-blocking warning (I-8)"
            )


    # ------------------------------------------------------------------ #
    def _apply_recurring_character_voices(
            self, cast, engine, language, report) -> int:
        """Stamp recurring characters' assigned voices in preserve_ledger.

        Returns the number of rows changed.

        THIS MODE IS A CASTING MODE, not an old-save compatibility layer, so a
        configured assignment applies here exactly as it does in auto_registry.
        What stays true is the mode's contract: ONLY a matching recurring row is
        touched, and every other row keeps the bytes it arrived with. A row the
        table does not name is not stamped, not cleared, and not annotated.

        THE ENGINE COMES FROM THE NORMALIZED `char_voice_engine`, never from
        `target_engine`. In this mode no bank is loaded up front, so
        `target_engine` is None -- reading it would silently mean "no engine" and
        every assignment would miss while the report said nothing was claimed.
        `lock()` has already turned "auto" into a concrete engine before here.

        THE BANK LOADS LAZILY, when a registered row is present. The dormant
        case -- no recurring character in the cast at all -- stays free of bank
        I/O, which is what it was before this existed. It deliberately does NOT
        also require the catalogue table to name the engine: a character can be
        delivered by a RESERVED bank row that the table never mentions, and
        checking the table here is what hid that path from this mode until
        2026-09-24.

        THE COST THAT BUYS, stated rather than glossed: a cast that DOES name a
        recurring character now loads the bank even on an engine that can never
        deliver one -- bark, which has no bank rows and no table mapping. The
        old pre-filter skipped that read. It is one bank load, the row is
        unchanged either way, and the miss is still reported; the trade is a
        redundant read in one case against a silently unreachable code path in
        three, which is the bug this replaced.
        """
        try:
            from ..config.cast_pools import recurring_character_key
        except ImportError:  # pragma: no cover -- flat-import harnesses
            try:
                from config.cast_pools import (  # type: ignore
                    recurring_character_key)
            except ImportError:
                try:
                    from cast_pools import (  # type: ignore
                        recurring_character_key)
                except ImportError:
                    return 0

        engine = str(engine or "").strip()
        if not engine:
            return 0

        # WHICH ROWS NAME A RECURRING CHARACTER -- and nothing more than that.
        # This used to also ask the catalogue table for a voice and drop any row
        # it could not answer for, which made it a SECOND resolver: when
        # `_recurring_character_bank_ref` learned that a reserved bank row is an
        # assignment, this filter did not, so on the clone engines the row was
        # dropped here and the reserved scan never ran. One resolver decides.
        wanted = []
        for entry in cast:
            if not isinstance(entry, dict) or _is_announcer_entry(entry):
                continue
            key = recurring_character_key(entry)
            if key:
                wanted.append((entry, key))
        if not wanted:
            return 0

        try:
            from ._otr_voice_bank import load_voice_bank
        except ImportError:  # pragma: no cover -- flat-import harnesses
            from _otr_voice_bank import load_voice_bank  # type: ignore
        try:
            bank_entries = load_voice_bank()[0]
        except Exception as exc:  # noqa: BLE001 -- a bank fault is not a re-cast
            report.append(
                f"preserve_ledger: recurring assignment skipped, bank "
                f"unavailable ({exc})"
            )
            return 0

        changed = 0
        for entry, _key in wanted:
            ref, miss = _recurring_character_bank_ref(
                entry, engine, bank_entries, language)
            if ref is None:
                report.append(
                    f"  {entry.get('char_id') or entry.get('name')}: "
                    f"recurring assignment missed ({miss or 'no match'}); "
                    f"row left as it arrived"
                )
                continue
            # Clear first: this row may carry identity from a previous lock on a
            # different engine, and a leftover provider id or reference path
            # renders with the wrong voice while nothing reports it.
            _clear_stale_voice_identity(entry)
            self._stamp(entry, ref, fallback="character_voice")
            changed += 1
            report.append(
                f"  {entry.get('char_id') or entry.get('name')}: "
                f"{ref.voice_ref_id} ({ref.engine}, recurring character)"
            )
        return changed


    # ------------------------------------------------------------------ #
    def _stamp_voice_engine_selection(self, led, voice_bank, ann_bank,
                                      char_voice_engine="auto",
                                      announcer_voice_engine="auto",
                                      bank_entries=None,
                                      voice_device="cuda"):
        """Stamp requested voice-engine routing even when cast rows are preserved.

        ``auto_registry`` uses the resolved target engine for casting; preserved
        ledgers still need the explicit engine choice recorded so profiles like
        otr_cloud_lanes cannot silently drift back to a local voice route.

        S4 platform-portability (2026-07-10): ``meta["voice_device"]`` rides
        the ledger exactly like the engine stamps -- every downstream voice
        adapter (and theme music) reads the SAME explicit device; the old
        per-adapter cuda->mps->cpu waterfalls are gone.
        """
        meta = led.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            led["meta"] = meta

        # RESOLVE FIRST, THEN STAMP THE CONCRETE DEVICE (2026-09-12). The widget
        # may now say "default" or "gpu:N" -- ComfyUI's own vocabulary -- and
        # neither is a device a loader can open. `resolve_device` turns those
        # into the real name and passes an explicit choice through untouched, so
        # what lands in the ledger is always what actually ran. That keeps every
        # receipt interpretable and replay faithful; a ledger holding the word
        # "default" would mean nothing six months from now.
        #
        # The refusal below is KEPT, deliberately. The 2026-07-09 portability
        # ruling is that an unavailable device fails loud rather than silently
        # downgrading, and that still holds: only "default" is ever resolved for
        # you, and a junk value is still an error rather than a quiet fallback.
        _raw = str(voice_device or _DEVOPTS.DEFAULT_DEVICE_OPTION).strip().lower()
        if _raw not in _DEVOPTS.device_options():
            raise ValueError(
                f"OTR_CastLock: voice_device {voice_device!r} is not one of "
                f"{'/'.join(_DEVOPTS.device_options())} -- NO silent default.")
        _dev = _DEVOPTS.resolve_device(_raw, fallback="cpu")
        meta["voice_device"] = _dev

        requested = str(char_voice_engine or "auto").strip() or "auto"
        if requested == "auto":
            requested = _DEFAULT_CHAR_ENGINE
        if bank_entries is None:
            # preserve_ledger: do not re-cast and do not load the bank just to
            # write a stamp. 4a inherits this concrete engine.
            target_engine = None
            meta["char_voice_engine"] = requested
        else:
            target_engine = self._resolve_char_engine(
                voice_bank, bank_entries, requested)
            meta["char_voice_engine"] = target_engine or requested

        announcer_engine = self._resolve_announcer_engine(
            announcer_voice_engine)
        meta["announcer_voice_engine"] = announcer_engine
        return target_engine, announcer_engine

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stamp(entry, ref, *, fallback: str = "") -> None:
        """Stamp the chosen reference onto a cast entry (I-4 / I-9).

        ``fallback`` records HOW the reference was chosen. It is written on every
        stamped row, empty string for the ordinary deterministic cast, so a
        downstream reader never has to distinguish "cast normally" from "field
        was never written".
        """
        entry["voice_ref_id"] = ref.voice_ref_id
        entry["voice_engine"] = ref.engine
        entry["tts_model"] = str(getattr(ref, "engine", "") or "")
        entry["commercial_clean"] = _delivered_commercial_clean(entry, ref)
        entry["voice_cast_fallback"] = fallback
        # A kokoro / google / elevenlabs stamp must not keep a leftover Bark
        # ``v2/`` preset. Lime 20260917 left Stomp/Tiptoe/Whiskers speaking
        # kokoro while the ledger still named Bark, and the two CastLock
        # tests that pin this were red at HEAD. Bark's own identity stays:
        # when the stamped engine IS bark, ``voice_preset`` is the spoken
        # id and is left alone (Lemmy's frozen v2/* beside a bark row).
        #
        # THIS INCLUDES LEMMY'S PROVISIONAL STAMPS, and the question was
        # settled by dates (2026-09-20). A 2026-08-16 test pinned the writer
        # preset SURVIVING a chatterbox audition stamp; a 2026-09-01 portable
        # bank test pinned it CLEARED on a kokoro one; the Lime clear of
        # 2026-09-17 is the newest statement of intent. Two reviewers traced
        # every consumer: only bark's dispatch USES `voice_preset` to choose
        # a voice; kokoro reads `voice_ref_id`, the credits prefer
        # `voice_engine` / `voice_ref_id`, and no bark stage runs after a
        # non-bark provisional stamp inside one render. One more reader,
        # found by the QA pass on the pushed diff: `_otr_voice_node_common`
        # copies the field into every engine's resolved request, where it is
        # part of the audio-cache key. So a non-bark row's key changes once,
        # from the leftover `v2/...` to "", which is a single cache miss and
        # a re-render of that line, never different audio. A preset kept on
        # a row another engine speaks is a stale identity on the ledger and
        # a stale cache key, so the newest rule wins everywhere and the
        # 08-16 test was retired rather than carved around.
        if str(getattr(ref, "engine", "") or "") != "bark":
            leftover = str(entry.get("voice_preset") or "")
            if leftover.startswith("v2/"):
                entry["voice_preset"] = ""
        # presentation_gender (item 8 chunk 4, 2026-08-06): the gender the
        # DELIVERED voice presents as, taken from the reference actually chosen
        # rather than from the row's label. Stamped HERE because this is the one
        # place every stamped row passes through -- characters, the announcer,
        # the hybrid voice-fit branch and the gender-agnostic fallback alike.
        #
        # Two rows the label cannot answer for, and this is why the field exists:
        # the ANNOUNCER's reference is drawn from the episode seed and never read
        # its row's gender at all, and an `other` row is served by a draw the bank
        # makes without regard to gender. In both cases the row said one thing and
        # the audience heard another, with nothing in the ledger recording it.
        # Whatever the bank's own vocabulary says wins -- including `neutral`,
        # which is a real reference (el_river), not a bucket to round away.
        entry["presentation_gender"] = str(getattr(ref, "gender", "") or "").strip().lower()
        # C3 (cloud-audio 2026-07-03): carry the provider voice id for cloud
        # (ElevenLabs) casting -- ONLY when present, so local (ref-clip/preset)
        # cast entries stay byte-identical. The durable cast stamp copies the
        # whole cast section (production_ledger.stamp_durable), so this survives
        # to the admission gate + OTR_CreditsRoll.
        pvid = getattr(ref, "provider_voice_id", "") or ""
        if pvid:
            entry["provider_voice_id"] = pvid

    @staticmethod
    def _bank_following_engine(role, engine, requested_bank=None):
        """Engine owns the bank. A leftover or mismatched id is replaced.

        Comfy graphs no longer pass ``voice_bank``. Direct callers and
        ``_auto_registry`` tests still hand the old widget default ``default``,
        which kokoro does not own -- that pairing used to raise at CastLock
        twelve minutes into a leg. Derive the engine's first allowed bank
        instead of failing closed on a widget that no longer exists.
        """
        from ._otr_engine_profiles import require_resolver, voice_bank_for_engine
        derived = voice_bank_for_engine(role, engine)
        bank = str(requested_bank or "").strip()
        if not bank:
            return derived
        prof = require_resolver().profile_for(role, engine)
        if prof is not None and bank not in prof.allowed_voice_banks:
            return derived
        return bank

    @staticmethod
    def _normalize_voice_engine(requested_engine: str) -> str:
        requested = str(requested_engine or "auto").strip() or "auto"
        return _VOICE_ENGINE_ALIASES.get(requested, requested)

    @staticmethod
    def _resolve_announcer_engine(requested_engine="auto") -> str:
        requested = CastLock._normalize_voice_engine(requested_engine)
        if requested == "auto":
            return _DEFAULT_ANNOUNCER_ENGINE
        allowed = set(_ANNOUNCER_VOICE_ENGINES) | _VOICE_ENGINE_RESOLVE_EXTRA
        if requested not in allowed:
            from ._otr_voice_bank import VoiceCastingError
            raise VoiceCastingError(
                f"unsupported announcer_voice_engine {requested!r}; expected "
                f"one of {tuple(sorted(allowed))}")
        return requested

    @staticmethod
    def _resolve_char_engine(voice_bank, bank_entries, requested_engine="auto"):
        """First legacy-first char_voice engine whose profile allows ``voice_bank``
        AND that has reference entries in the bank. ``None`` if there is none
        (e.g. bark_legacy / kokoro_builtin -> preset engines, no refs)."""
        from ._otr_voice_bank import VoiceCastingError

        try:
            from ._otr_engine_profiles import legacy_first_engines, load_resolver

            resolver = load_resolver()
            engines_with_refs = {e.engine for e in bank_entries}
            requested = CastLock._normalize_voice_engine(requested_engine)
            if requested != "auto":
                allowed = set(_CHAR_VOICE_ENGINES) | _VOICE_ENGINE_RESOLVE_EXTRA
                if requested not in allowed:
                    raise VoiceCastingError(
                        f"unsupported char_voice_engine {requested!r}; expected "
                        f"one of {tuple(sorted(allowed))}")
                if resolver is not None:
                    prof = resolver.profile_for("char_voice", requested)
                    if prof is None:
                        raise VoiceCastingError(
                            f"no char_voice profile for engine {requested!r}")
                    if voice_bank not in prof.allowed_voice_banks:
                        raise VoiceCastingError(
                            f"voice_bank {voice_bank!r} is not allowed for "
                            f"char_voice_engine {requested!r}; allowed "
                            f"{prof.allowed_voice_banks}")
                if requested in engines_with_refs:
                    return requested
                if requested == "bark":
                    return None
                raise VoiceCastingError(
                    f"char_voice_engine {requested!r} has no reference entries "
                    f"in the active voice bank")
            for eng in legacy_first_engines("char_voice"):
                if eng not in engines_with_refs:
                    continue
                if resolver is None:
                    return eng
                prof = resolver.profile_for("char_voice", eng)
                if prof and voice_bank in prof.allowed_voice_banks:
                    return eng
        except VoiceCastingError:
            raise
        except Exception:  # noqa: BLE001
            return None
        return None
