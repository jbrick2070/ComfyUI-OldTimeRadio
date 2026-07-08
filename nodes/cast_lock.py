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

log = logging.getLogger("OTR")

# Voice-bank ids the operator picks from the OTR_CastLock dropdown. The bank id
# does NOT filter the per-ref bank (assign_voice_for_slot scores by gender/timbre/
# role/age, engine-restricted); it gates which char ENGINE _resolve_char_engine
# selects, via each engine profile's allowed_voice_banks. "default_clean" routes
# the cast to the COMMERCIAL-CLEAN cloner (chatterbox MIT, then dia Apache) and
# EXCLUDES the non-commercial indextts2 -- the release-safe cast (2026-06-18
# voice-engine roundtable). "default" is unchanged (indextts2 first = quality).
_VOICE_BANKS = ("default", "default_clean", "bark_legacy", "kokoro_builtin",
                "elevenlabs_cloud")  # cloud ElevenLabs voice pool (S2), 2026-07-03
_CAST_POLICIES = ("preserve_ledger", "auto_registry")
_CHAR_VOICE_ENGINES = (
    "auto", "indextts2", "chatterbox", "dia", "bark", "kokoro", "elevenlabs")
_ANNOUNCER_VOICE_ENGINES = ("auto", "kokoro", "chatterbox", "elevenlabs")
_DEFAULT_ANNOUNCER_ENGINE = "kokoro"


def _is_announcer_entry(entry: dict) -> bool:
    name = str(entry.get("name") or "").strip().upper()
    role = str(entry.get("speaker_role") or entry.get("role") or "").strip().lower()
    return name == "ANNOUNCER" or role == "announcer"


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
             announcer_voice_engine="auto", gate_in=""):
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

        # Sprint 2 (a): CastLock OWNS bark voice casting. The writer no longer
        # stamps voice_preset -- it persists cast_seed in meta.cast_contract and
        # CastLock replays the deterministic picker (byte-identical) and stamps
        # the bark voices here, then runs the relocated voice invariants (Gate 1,
        # formerly in lock_cast). Runs regardless of cast_voice_policy (the policy
        # governs the clip-engine voice bank, not bark casting).
        self._assign_bark_voices(cast, meta, report)

        if cast_voice_policy == "auto_registry":
            self._auto_registry(
                led, cast, voice_bank, allow_voice_reuse, report,
                char_voice_engine=char_voice_engine,
                announcer_voice_engine=announcer_voice_engine)
        else:
            self._stamp_voice_engine_selection(
                led, voice_bank, char_voice_engine, announcer_voice_engine)
            report.append(
                f"preserve_ledger: {len(cast)} cast entries preserved "
                f"(no re-cast)"
            )

        # Stamp the lock identity onto meta (I-4: stamp once at cast lock).
        meta["cast_lock_revision"] = revision
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
        """Freeze-halt + VRAM-recovery gate, re-homed from the legacy audio
        nodes in the audio clean-break (BUG-LOCAL-276 / BUG-LOCAL-300 / E9).

        freeze_verdict=='needs_full_rerun' refuses to cast/render UNLESS the
        block is a subjective 'quality' verdict (renders with a warning -- the
        cast is clean, only the story-critic arc is weak) or the operator sets
        OTR_BYPASS_FREEZE_HALT=1 for sprint-time smoke iteration. A missing or
        unknown block class is treated as structural (halt);
        OTR_BARK_HALT_ON_QUALITY_BLOCK=1 restores strict halt-on-any. A missing
        verdict (legacy graphs / tests) proceeds unchanged.

        Then, if the cascade teardown reported an unload failure
        (freeze_unload_ok is False), attempt one defensive unload_llm before the
        audio chain claims VRAM -- CastLock runs before any audio engine loads,
        so this protects the 14.5 GB ceiling (I-7).
        """
        verdict = (meta or {}).get("freeze_verdict")
        if verdict == "needs_full_rerun":
            block_class = (meta or {}).get("freeze_block_class")
            bypass = os.environ.get("OTR_BYPASS_FREEZE_HALT", "0") == "1"
            strict_quality = (
                os.environ.get("OTR_BARK_HALT_ON_QUALITY_BLOCK", "0") == "1"
            )
            if bypass:
                log.warning(
                    "[CastLock] FREEZE HALT BYPASSED (OTR_BYPASS_FREEZE_HALT=1); "
                    "casting a flagged ledger. Intended for sprint-time smoke "
                    "iteration only; downstream gates may still surface issues. "
                    "See BUG-LOCAL-276."
                )
            elif block_class == "quality" and not strict_quality:
                log.warning(
                    "[CastLock] freeze_verdict='needs_full_rerun' with "
                    "freeze_block_class='quality' -- a subjective story-critic "
                    "verdict on a cast-clean, renderable ledger, not a "
                    "renderability failure. Proceeding. Set "
                    "OTR_BARK_HALT_ON_QUALITY_BLOCK=1 to halt on quality blocks "
                    "too. See BUG-LOCAL-300."
                )
            else:
                raise ValueError(
                    "OTR_CastLock: freeze cascade stamped "
                    "freeze_verdict='needs_full_rerun' (structural -- the writer "
                    "left the ledger in an unrenderable state). Refusing to "
                    "cast/render. Re-run the writer phase. Set "
                    "OTR_BYPASS_FREEZE_HALT=1 only for sprint-time smoke "
                    "iteration. See BUG-LOCAL-276."
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
            if str(row.get("name") or "").strip().upper() == "ANNOUNCER":
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
                       announcer_voice_engine="auto"):
        from ._otr_voice_bank import (
            CASTING_POLICY_VERSION, VoiceCastingError, announcer_voice_ref,
            assign_voice_for_slot, load_voice_bank,
        )
        from ._otr_voice_node_common import coerce_int_seed

        bank_entries, _bank_sha = load_voice_bank()
        meta = led.get("meta") or {}
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
        target_engine, announcer_engine = self._stamp_voice_engine_selection(
            led, voice_bank, char_voice_engine, announcer_voice_engine,
            bank_entries=bank_entries)

        if target_engine is None:
            report.append(
                f"auto_registry: voice_bank {voice_bank!r} has no character "
                f"reference engine; character voices preserved"
            )

        used: set = set()
        gated = 0
        for entry in cast:
            if not isinstance(entry, dict):
                continue
            char_id = str(entry.get("char_id") or "")

            if _is_announcer_entry(entry):
                try:
                    ref = announcer_voice_ref(announcer_engine, bank=bank_entries)
                    self._stamp(entry, ref)
                    gated += 0 if ref.commercial_clean else 1
                    report.append(
                        f"  {char_id or 'ANNOUNCER'}: announcer {ref.voice_ref_id} "
                        f"({ref.engine}, clean={ref.commercial_clean})"
                    )
                except VoiceCastingError as exc:
                    report.append(f"  {char_id or 'ANNOUNCER'}: announcer NOT cast -- {exc}")
                continue

            if target_engine is None:
                continue
            gender = str(entry.get("gender") or "").strip().lower()
            if not gender:
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
                        used.add(hybrid_ref.voice_ref_id)
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
                report.append(f"  {char_id}: NOT cast -- {exc}")
                continue
            self._stamp(entry, ref)
            used.add(ref.voice_ref_id)
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
    def _stamp_voice_engine_selection(self, led, voice_bank,
                                      char_voice_engine="auto",
                                      announcer_voice_engine="auto",
                                      bank_entries=None):
        """Stamp requested voice-engine routing even when cast rows are preserved.

        ``auto_registry`` uses the resolved target engine for casting; preserved
        ledgers still need the explicit engine choice recorded so profiles like
        cloud_all cannot silently drift back to a local voice route.
        """
        meta = led.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            led["meta"] = meta

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
    def _stamp(entry, ref) -> None:
        """Stamp the chosen reference onto a cast entry (I-4 / I-9)."""
        entry["voice_ref_id"] = ref.voice_ref_id
        entry["voice_engine"] = ref.engine
        entry["commercial_clean"] = bool(ref.commercial_clean)
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
