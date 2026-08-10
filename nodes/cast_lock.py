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

E.4: the surfaced widgets are exactly ``voice_bank`` / ``cast_voice_policy`` /
``allow_voice_reuse`` / ``char_voice_engine`` / ``announcer_voice_engine`` --
``delivery_profile`` is no longer surfaced (single option "neutral" in v2; the
``lock()`` kwarg still defaults to "neutral" and is validated + stamped) -- no
``voice_engine_mode``, ``deterministic_inference`` or ``model_id`` widget.
Import-time is side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from . import _otr_voice_route as _ROUTE

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _lemmy_voice_policy():
    """The character voice policy, or ``{}`` if the pack cannot be imported.

    Fail-SOFT here and fail-CLOSED downstream: an unreadable policy yields no
    approved routes, so nothing is selected and casting proceeds exactly as it
    did before this path existed. The strictness lives where a route is actually
    claimed -- refusing to import is not the place to break a render.
    """
    try:
        from ..config import cast_pools as _POOLS  # type: ignore
    except ImportError:
        try:
            from config import cast_pools as _POOLS  # type: ignore
        except ImportError:
            return {}
    return getattr(_POOLS, "LEMMY_VOICE_POLICY", None) or {}

# Voice-bank ids the operator picks from the OTR_CastLock dropdown. The bank id
# does NOT filter the per-ref bank (assign_voice_for_slot scores by gender/timbre/
# role/age, engine-restricted); it gates which char ENGINE _resolve_char_engine
# selects, via each engine profile's allowed_voice_banks. "default_clean" routes
# the cast to the COMMERCIAL-CLEAN cloner (chatterbox MIT, then dia Apache) and
# EXCLUDES the non-commercial indextts2 -- the release-safe cast (2026-06-18
# voice-engine roundtable). "default" is unchanged (indextts2 first = quality).
_VOICE_BANKS = ("default", "default_clean", "bark_legacy", "kokoro_builtin",
                "elevenlabs_cloud", "google_tts")
_CAST_POLICIES = ("preserve_ledger", "auto_registry")
_CHAR_VOICE_ENGINES = (
    "auto", "indextts2", "chatterbox", "dia", "bark", "kokoro", "elevenlabs",
    "google_tts")
_ANNOUNCER_VOICE_ENGINES = (
    "auto", "kokoro", "chatterbox", "dia", "elevenlabs", "google_tts")
_DEFAULT_ANNOUNCER_ENGINE = "kokoro"


def _is_announcer_entry(entry: dict) -> bool:
    char_id = str(entry.get("char_id") or "").strip().lower()
    name = str(entry.get("name") or "").strip().upper()
    role = str(entry.get("speaker_role") or entry.get("role") or "").strip().lower()
    return char_id == "announcer" or name == "ANNOUNCER" or role == "announcer"


class CastLock:
    """Registered as ``OTR_CastLock``. Single v2 ledger authority."""

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
                "voice_bank": (list(_VOICE_BANKS), {
                    "default": _VOICE_BANKS[0],
                    "tooltip": (
                        "Voice reference bank scope used by auto_registry. "
                        "'default' casts from the chatterbox / indextts2 "
                        "reference banks; bark_legacy / kokoro_builtin keep the "
                        "preset-based engines."
                    ),
                }),
                "cast_voice_policy": (list(_CAST_POLICIES), {
                    "default": _CAST_POLICIES[0],
                    "tooltip": (
                        "preserve_ledger: keep the writer's voice assignments "
                        "(byte-safe default). auto_registry: assign voice "
                        "references from the bank with the deterministic caster."
                    ),
                }),
                "allow_voice_reuse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When the bank runs out of unique references, allow "
                        "reusing one already assigned (the gender floor still "
                        "holds). Off -> casting fails closed instead."
                    ),
                }),
                "char_voice_engine": (list(_CHAR_VOICE_ENGINES), {
                    "default": "auto",
                    "tooltip": (
                        "Character TTS engine CastLock stamps into the durable "
                        "cast. auto resolves from voice_bank; explicit cloud "
                        "picks fail loud if the bank is incompatible."
                    ),
                }),
                "announcer_voice_engine": (list(_ANNOUNCER_VOICE_ENGINES), {
                    "default": "auto",
                    "tooltip": (
                        "Announcer TTS engine CastLock stamps into the durable "
                        "cast. auto keeps the shipped Kokoro announcer."
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
                "voice_device": (["cuda", "cpu", "mps"], {
                    "default": "cuda",
                    "tooltip": "EXPLICIT device for local voice/music "
                               "engines (platform profile value). No "
                               "auto-detect; unavailable devices fail loud.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # D: local-disk only; casting/bank checks are fail-closed on the lock()
        # path, not here, so a box-fresh graph validates clean.
        return True

    # ------------------------------------------------------------------ #
    def lock(self, script_json, voice_bank="default",
             cast_voice_policy="preserve_ledger", delivery_profile="neutral",
             allow_voice_reuse=False, char_voice_engine="auto",
             announcer_voice_engine="auto", gate_in="",
             voice_device="cuda"):
        from . import _otr_ledger_consumers as _OTRLC
        from ._otr_delivery_profiles import (
            DELIVERY_PROFILE_VERSION, get_delivery_profile,
        )

        led = _OTRLC.load_ledger(script_json)
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

        # Sprint 2 (a): CastLock OWNS bark voice casting. The writer no longer
        # stamps voice_preset -- it persists cast_seed in meta.cast_contract and
        # CastLock replays the deterministic picker (byte-identical) and stamps
        # the bark voices here, then runs the relocated voice invariants (Gate 1,
        # formerly in lock_cast). Runs regardless of cast_voice_policy (the policy
        # governs the clip-engine voice bank, not bark casting).
        self._assign_bark_voices(cast, meta, report)

        # STEP 2 (plan 5.2): resolve the bank and stamp the engine metadata ONCE,
        # for BOTH modes, before any route is looked at. It used to happen inside
        # each branch, which meant a route could not be proved against the engine
        # that was going to render it -- the agreement check needs the engine in
        # hand first.
        #
        # preserve_ledger deliberately passes bank_entries=None: with an `auto`
        # request that mode records `char_voice_engine="auto"` rather than
        # pinning a concrete engine into a ledger it promised not to re-cast.
        # Loading the bank here would silently change that stamp.
        bank_entries = None
        if cast_voice_policy == "auto_registry":
            from ._otr_voice_bank import load_voice_bank
            bank_entries, _bank_sha = load_voice_bank()
        target_engine, announcer_engine = self._stamp_voice_engine_selection(
            led, voice_bank, char_voice_engine, announcer_voice_engine,
            bank_entries=bank_entries, voice_device=voice_device)

        # STEP 4 (plan 5.2): prove the policy route BEFORE either caster runs, so
        # a route that cannot prove itself stops the lock instead of losing a
        # race with the generic selector. Returns None when no policy claims a
        # row -- which is every render shipping today.
        # Hand over the bank auto_registry already loaded. preserve_ledger passes
        # None here by design and the claim path loads its own only if a route is
        # actually selected -- which keeps the dormant case free of bank I/O.
        policy_claim = self._resolve_policy_claim(
            voice_bank, target_engine, bank_entries=bank_entries)

        if cast_voice_policy == "auto_registry":
            self._auto_registry(
                led, cast, voice_bank, allow_voice_reuse, report,
                char_voice_engine=char_voice_engine,
                announcer_voice_engine=announcer_voice_engine,
                voice_device=voice_device,
                bank_entries=bank_entries,
                target_engine=target_engine,
                announcer_engine=announcer_engine,
                policy_claim=policy_claim)
        else:
            # STEP 5 (plan 5.2): in preserve_ledger ONLY the claimed row changes.
            # Every other row keeps the bytes it arrived with -- that is the
            # mode's whole contract, and an explicit re-pin is not a licence to
            # re-cast the cast.
            claimed = self._apply_policy_claim(cast, policy_claim, report)
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
        meta["voice_bank_id"] = voice_bank

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
            cast, led.get("lines") or [], voice_bank=voice_bank
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
            bypass = os.environ.get("OTR_BYPASS_FREEZE_HALT", "0") == "1"
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
    def _assign_bark_voices(cast, meta, report) -> None:
        """Sprint 2 (a): stamp bark voice_preset onto the cast by REPLAYING the
        writer's deterministic picker.

        The writer persists ``cast_seed`` (OS-entropy per episode) in
        ``meta.cast_contract`` and no longer stamps voice_preset itself.
        ``replay_voice_assignment`` reconstructs the exact picker sequence keyed
        on that cast_seed -- byte-identical to what the writer used to assign
        (pinned by tests/test_cast_voice_replay_parity.py) -- and we stamp it
        onto the bark (non-ANNOUNCER) rows by char_id. The relocated Gate 1 voice
        invariants then run HERE, after assignment.

        A ledger with no persisted cast_seed (legacy graph / minimal test
        fixture) cannot be replayed; its voice_preset is preserved untouched and
        the post-assignment invariant is skipped (nothing was assigned).
        """
        from . import _otr_casting as _OTRCAST
        from ._otr_text_delivery import CONTENT_OWNED, delivery_mode_for_meta

        # A content-owned lane builds its OWN cast rows and stamps their
        # voice_preset in the lane runner; the writer's seeded picker never ran,
        # so there is no sequence to replay and no num_characters_request to
        # replay it with. Replaying anyway would fabricate a cast that was never
        # rolled and overwrite the voices the lane chose. VERIFY what the lane
        # assigned instead -- the Gate 1 invariants still run, so a content-owned
        # lane can never ship duplicate or non-v2/ bark voices.
        if delivery_mode_for_meta(meta) == CONTENT_OWNED:
            report.append(
                "bark voices: content-owned lane owns its cast -- "
                "voice_preset preserved (no writer replay)"
            )
            _OTRCAST._assert_unique_bark_voices(cast)
            _OTRCAST._assert_voice_preset_invariant(cast)
            return

        contract = (meta or {}).get("cast_contract") or {}
        cast_seed = contract.get("cast_seed")
        if cast_seed is None:
            report.append(
                "bark voices: no cast_seed in meta.cast_contract -- "
                "voice_preset preserved (no replay)"
            )
            return
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
        # Gate 1 (relocated from the writer's lock_cast): every non-ANNOUNCER row
        # now carries a v2/* voice_preset, and no two bark rows share a voice.
        _OTRCAST._assert_unique_bark_voices(cast)
        _OTRCAST._assert_voice_preset_invariant(cast)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stable_index(key, n: int) -> int:
        """A deterministic index in [0, n) from a string key -- stable across
        runs regardless of PYTHONHASHSEED (Python's built-in hash() is salted).
        """
        if n <= 0:
            return 0
        digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
        return int(digest, 16) % n

    @staticmethod
    def _resolve_character_voices_fail_soft(cast, lines, voice_bank="default") -> list:
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

        # (1) FAIL LOUD: every non-ANNOUNCER character cast row MUST carry a voice
        # identity by now (replay + optional auto_registry stamped it). A row with
        # none is a casting defect -- raise, never synthesize a fallback identity.
        for row in char_rows:
            preset = row.get("voice_preset")
            if isinstance(preset, str) and preset.strip():
                continue
            cid = str(row.get("char_id") or "")
            raise VoiceCastingError(
                f"character cast row {cid!r} reached CastLock with no voice_preset "
                f"-- casting must assign one. NO synthesized fallback identity "
                f"(no-fallback rip)."
            )

        voiced_char_ids = sorted(
            str(r.get("char_id") or "") for r in char_rows
            if isinstance(r.get("voice_preset"), str)
            and str(r.get("voice_preset")).strip()
        )

        # (2 KEPT: routing) + (3 FAIL LOUD: orphan) over character LINES.
        for ln in lines or []:
            if not isinstance(ln, dict):
                continue
            if str(ln.get("speaker_role") or "").strip().lower() != "character":
                continue
            cid = str(ln.get("char_id") or "")
            row = rows_by_id.get(cid)
            preset = (row or {}).get("voice_preset")
            if isinstance(preset, str) and preset.strip():
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
                       policy_claim=None):
        """Re-cast the registry rows.

        ``bank_entries`` / ``target_engine`` / ``announcer_engine`` /
        ``policy_claim`` are OPTIONAL pre-resolved values from ``lock``, which
        now does that work once for both modes (plan 5.2 step 2). They stay
        optional because this method is also called directly, with five
        positional arguments, and must keep resolving its own inputs when it is.
        """
        from ._otr_voice_bank import (
            CASTING_POLICY_VERSION, _SEEDED_ANNOUNCER_ENGINES, VoiceCastingError,
            announcer_voice_ref, assign_voice_for_slot,
            gender_agnostic_fallback_ref, load_voice_bank, voice_ref_usage_keys,
        )
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
        # VC chunk 4 (2026-06-22): the HYBRID LLM voice-fit decision (the writer
        # proposed a voice_ref_id; Python validated it). Honour the accepted id
        # when this CastLock resolved the SAME engine and it still validates +
        # does not collide; otherwise fall closed to the deterministic scorer.
        voice_decisions = meta.get("voice_cast_decision") or {}
        # announcer_engine is the sentinel: the resolver never returns None for
        # it, while target_engine legitimately can be None (a preset-only bank).
        if announcer_engine is None:
            target_engine, announcer_engine = self._stamp_voice_engine_selection(
                led, voice_bank, char_voice_engine, announcer_voice_engine,
                bank_entries=bank_entries, voice_device=voice_device)
            if policy_claim is None:
                policy_claim = self._resolve_policy_claim(
                    voice_bank, target_engine, bank_entries=bank_entries)

        if target_engine is None:
            report.append(
                f"auto_registry: voice_bank {voice_bank!r} has no character "
                f"reference engine; character voices preserved"
            )

        announcer_ref = None
        has_announcer = any(
            isinstance(entry, dict) and _is_announcer_entry(entry)
            for entry in cast
        )
        if has_announcer and announcer_engine in _SEEDED_ANNOUNCER_ENGINES:
            announcer_ref = announcer_voice_ref(
                announcer_engine, bank=bank_entries, episode_seed=episode_seed)

        used: set = set()
        def _mark_used(ref) -> None:
            used.update(voice_ref_usage_keys(ref))

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
                try:
                    ref = announcer_ref or announcer_voice_ref(
                        announcer_engine, bank=bank_entries,
                        episode_seed=episode_seed)
                    self._stamp(entry, ref)
                    gated += 0 if ref.commercial_clean else 1
                    report.append(
                        f"  {char_id or 'ANNOUNCER'}: announcer {ref.voice_ref_id} "
                        f"({ref.engine}, clean={ref.commercial_clean})"
                    )
                except VoiceCastingError as exc:
                    if announcer_engine == "google_tts":
                        raise
                    report.append(f"  {char_id or 'ANNOUNCER'}: announcer NOT cast -- {exc}")
                continue

            # STEP 4/5 (plan 5.2): the EXPLICIT RE-PIN, ahead of both the hybrid
            # LLM voice-fit and the generic seeded selection. The claim was
            # already proved in `lock` -- bytes hashed, engine triple agreed,
            # rights checked -- so all that happens here is the stamp.
            #
            # `_mark_used` runs regardless of allow_voice_reuse. The `used` set
            # only changes other rows' draws when reuse is off, so marking
            # unconditionally is free there and correct here: a pinned reference
            # is spoken for either way.
            if policy_claim is not None and _ROUTE.cast_row_matches_policy(
                    entry, policy_claim.character_key):
                self._stamp(entry, policy_claim.bank_entry,
                            fallback="policy_route")
                entry["voice_route"] = dict(policy_claim.voice_route)
                _mark_used(policy_claim.bank_entry)
                gated += 0 if policy_claim.bank_entry.commercial_clean else 1
                report.append(
                    f"  {char_id}: {policy_claim.voice_ref_id} "
                    f"({policy_claim.engine}, QUALIFIED policy route "
                    f"{policy_claim.voice_route.get('route_id')})"
                )
                continue

            if target_engine is None:
                continue
            # SYNONYM-CANONICALIZED (item 8, 2026-08-06). THIS is the path that
            # actually runs: CastLock stamps voice_ref_id before any render, so
            # the render-time resolvers in _otr_voice_node_common find a stamped
            # id and never reach their own gender fallback. Fixing only those
            # left the real defect live -- a row recorded `woman` raised
            # VoiceCastingError here, was caught below, and took the
            # gender-agnostic draw. The same value also feeds
            # validate_voice_proposal on the hybrid voice-fit branch, so this
            # one line closes both.
            from ._otr_roster_gender import canonical_bank_gender
            gender = canonical_bank_gender(entry.get("gender"))
            if not gender:
                if target_engine == "google_tts":
                    raise VoiceCastingError(
                        f"{char_id}: google_tts character casting needs a cast "
                        f"gender to choose a gender-plausible provider voice. "
                        f"NO FALLBACK.")
                report.append(f"  {char_id}: no gender -- preserved (not re-cast)")
                continue
            # VC chunk 4: honour the HYBRID LLM voice-fit when this engine matches
            # the one the proposal was built for AND it re-validates (no stale
            # bank / no collision). Else fall closed to the deterministic scorer.
            decision = voice_decisions.get(char_id) or {}
            accepted = str(decision.get("accepted_id") or "").strip()
            if accepted and decision.get("engine") == target_engine:
                from ._otr_voice_bank import (
                    validate_voice_proposal, voice_ref_entry,
                )
                if validate_voice_proposal(
                    accepted, target_engine, gender,
                    bank=bank_entries, used_ids=used,
                ):
                    hybrid_ref = voice_ref_entry(accepted, target_engine, bank_entries)
                    if hybrid_ref is not None:
                        self._stamp(entry, hybrid_ref)
                        _mark_used(hybrid_ref)
                        gated += 0 if hybrid_ref.commercial_clean else 1
                        report.append(
                            f"  {char_id}: {hybrid_ref.voice_ref_id} "
                            f"({hybrid_ref.engine}, hybrid LLM voice-fit, "
                            f"clean={hybrid_ref.commercial_clean})"
                        )
                        continue

            # Prefer the writer's voice-fit slot (timbre/age_band); fall back to
            # any entry-level fields for legacy ledgers without the stamp.
            slot = voice_slots.get(char_id) or {}
            slot_timbre = slot.get("timbre") or entry.get("timbre") or ()
            slot_age = str(slot.get("age_band") or entry.get("age_band") or "")
            try:
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
                )
            except VoiceCastingError as exc:
                if target_engine == "google_tts":
                    raise
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
                )
                if fallback_ref is None:
                    report.append(f"  {char_id}: NOT cast -- {exc}")
                    continue
                self._stamp(entry, fallback_ref, fallback="gender_unservable")
                _mark_used(fallback_ref)
                gated += 0 if fallback_ref.commercial_clean else 1
                report.append(
                    f"  {char_id}: {fallback_ref.voice_ref_id} "
                    f"({fallback_ref.engine}, gender {gender!r} unservable -- "
                    f"gender-agnostic reference)"
                )
                continue
            self._stamp(entry, ref)
            _mark_used(ref)
            gated += 0 if ref.commercial_clean else 1
            report.append(
                f"  {char_id}: {ref.voice_ref_id} ({ref.engine}, "
                f"clean={ref.commercial_clean})"
            )

        if gated:
            report.append(
                f"auto_registry: {gated} assigned voice(s) are known-gated "
                f"(commercial_clean=false) -- non-blocking warning (I-8)"
            )

    # ------------------------------------------------------------------ #
    def _resolve_policy_claim(self, voice_bank, target_engine,
                              bank_entries=None):
        """Prove the character voice policy's selected route, or return None.

        None means NOTHING WAS SELECTED, which is every render shipping today:
        ``LEMMY_VOICE_POLICY["approved_native_routes"]`` is empty, because no
        audition has yet proved a Cockney route on any engine. The mechanism is
        built and inert, and it activates the day an operator receipt fills that
        dict -- not before.

        A route that IS selected and cannot prove itself raises
        ``VoiceRouteError``. It never falls back: casting somebody else's voice
        because the qualified one failed its own check is precisely the silent
        substitution this path exists to prevent.
        """
        policy = _lemmy_voice_policy()
        if not (policy or {}).get("approved_native_routes"):
            return None                      # dormant -- do not touch the bank

        # A policy with real routes needs a concrete engine to prove agreement
        # against, even in preserve_ledger, where `lock` deliberately leaves the
        # engine stamp as "auto". Resolve one HERE, for the proof only -- this
        # must not write meta["char_voice_engine"].
        if bank_entries is None:
            from ._otr_voice_bank import load_voice_bank
            bank_entries, _bank_sha = load_voice_bank()
        engine = target_engine
        if engine is None:
            engine = self._resolve_char_engine(voice_bank, bank_entries, "auto")

        return _ROUTE.resolve_policy_route_claim(
            policy, engine, datetime.now(timezone.utc),
            bank_entries=bank_entries, repo_root=_REPO_ROOT,
        )

    # ------------------------------------------------------------------ #
    def _apply_policy_claim(self, cast, policy_claim, report) -> int:
        """Stamp a proved claim onto its row in preserve_ledger. Returns rows changed.

        Deliberately narrow: announcer rows are never claimed, and a policy that
        matches no row at all is REPORTED rather than raised. The route proved
        itself; that the episode did not cast that character is an ordinary fact
        about the episode, not a qualification failure.
        """
        if policy_claim is None:
            return 0
        changed = 0
        for entry in cast:
            if not isinstance(entry, dict) or _is_announcer_entry(entry):
                continue
            if not _ROUTE.cast_row_matches_policy(
                    entry, policy_claim.character_key):
                continue
            self._stamp(entry, policy_claim.bank_entry, fallback="policy_route")
            entry["voice_route"] = dict(policy_claim.voice_route)
            changed += 1
            report.append(
                f"  {entry.get('char_id') or entry.get('name')}: "
                f"{policy_claim.voice_ref_id} ({policy_claim.engine}, "
                f"QUALIFIED policy route "
                f"{policy_claim.voice_route.get('route_id')})"
            )
        if not changed:
            report.append(
                f"policy route {policy_claim.voice_route.get('route_id')!r} "
                f"proved, but no cast row matches "
                f"{policy_claim.character_key!r} -- nothing re-pinned"
            )
        return changed

    # ------------------------------------------------------------------ #
    def _stamp_voice_engine_selection(self, led, voice_bank,
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

        _dev = str(voice_device or "cuda").strip().lower()
        if _dev not in ("cuda", "cpu", "mps"):
            raise ValueError(
                f"OTR_CastLock: voice_device {voice_device!r} is not one of "
                "cuda/cpu/mps -- NO silent default.")
        meta["voice_device"] = _dev

        requested = str(char_voice_engine or "auto").strip() or "auto"
        target_engine = None
        if requested == "auto" and bank_entries is None:
            meta["char_voice_engine"] = "auto"
        else:
            if bank_entries is None:
                from ._otr_voice_bank import load_voice_bank
                bank_entries, _bank_sha = load_voice_bank()
            target_engine = self._resolve_char_engine(
                voice_bank, bank_entries, requested)
            meta["char_voice_engine"] = (
                target_engine or ("auto" if requested == "auto" else requested)
            )

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
        entry["commercial_clean"] = bool(ref.commercial_clean)
        entry["voice_cast_fallback"] = fallback
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
    def _resolve_announcer_engine(requested_engine="auto") -> str:
        requested = str(requested_engine or "auto").strip()
        if requested == "auto":
            return _DEFAULT_ANNOUNCER_ENGINE
        if requested not in _ANNOUNCER_VOICE_ENGINES:
            from ._otr_voice_bank import VoiceCastingError
            raise VoiceCastingError(
                f"unsupported announcer_voice_engine {requested!r}; expected "
                f"one of {_ANNOUNCER_VOICE_ENGINES}")
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
            requested = str(requested_engine or "auto").strip()
            if requested != "auto":
                if requested not in _CHAR_VOICE_ENGINES:
                    raise VoiceCastingError(
                        f"unsupported char_voice_engine {requested!r}; expected "
                        f"one of {_CHAR_VOICE_ENGINES}")
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
