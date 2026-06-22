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

E.4: the widgets are exactly ``voice_bank`` / ``cast_voice_policy`` /
``delivery_profile`` / ``allow_voice_reuse`` -- no ``voice_engine_mode``,
``deterministic_inference`` or ``model_id`` widget. Import-time is
side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

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
_VOICE_BANKS = ("default", "default_clean", "bark_legacy", "kokoro_builtin")
_CAST_POLICIES = ("preserve_ledger", "auto_registry")
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
        # C-5: no IO. Delivery list is a tiny pure call; fall back hard-coded.
        try:
            from ._otr_delivery_profiles import available_delivery_profiles

            delivery = available_delivery_profiles() or ["neutral"]
        except Exception:  # noqa: BLE001
            delivery = ["neutral"]
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
                "delivery_profile": (delivery, {
                    "default": delivery[0],
                    "tooltip": "Delivery profile id (only 'neutral' ships in v2).",
                }),
                "allow_voice_reuse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When the bank runs out of unique references, allow "
                        "reusing one already assigned (the gender floor still "
                        "holds). Off -> casting fails closed instead."
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
             allow_voice_reuse=False, gate_in=""):
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
            self._auto_registry(led, cast, voice_bank, allow_voice_reuse, report)
        else:
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

        # STEP 3 (2026-06-22 story+cast fix): node-80 OUTPUT fail-closed voice
        # gate. The cast presets have now been assigned (replay + optional
        # auto_registry); enforce -- BEFORE the ledger leaves CastLock for the
        # TTS nodes -- that no character line can reach node 81
        # (OTR_BatchCharacterVoices) with a None/empty voice_preset on its cast
        # row. Runs UNCONDITIONALLY, independent of cast_seed (the existing
        # Gate 1 is skipped on the cast_seed=None early return).
        self._assert_voice_fail_closed(cast, led.get("lines") or [])

        report.insert(0, f"cast_lock_revision={revision} policy={cast_voice_policy}")
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
    def _assert_voice_fail_closed(cast, lines) -> None:
        """STEP 3 (2026-06-22 story+cast fix): node-80 OUTPUT voice gate.

        No ``speaker_role == "character"`` line may leave CastLock for
        OTR_BatchCharacterVoices (node 81) with a None/empty ``voice_preset``
        on its cast row. The relocated Gate 1
        (``_otr_casting._assert_voice_preset_invariant``) only runs INSIDE
        ``_assign_bark_voices`` AFTER a successful seed replay, so the
        ``cast_seed is None`` early-return -- and an unmatched ``char_id``
        even with a seed -- could let an empty preset propagate to TTS
        (grounding_r2 #4). This backstop runs UNCONDITIONALLY and is
        engine-agnostic (requires a non-empty preset, not specifically a
        ``v2/*`` bark id, since the character engine may be indextts2 /
        chatterbox / bark).

        Announcer lines route to node 82 (OTR_AnnouncerVoice), which resolves
        its engine directly (kokoro) and never reads a cast-row preset, so the
        announcer is intentionally excluded -- matching Gate 1. Cue rows
        (music_*/sfx) never reach character/announcer TTS.

        The deterministic picker is upstream (``replay_voice_assignment``,
        keyed on ``meta.cast_contract.cast_seed``); this gate does not
        fabricate a voice (that would risk the determinism / uniqueness
        contract). A genuine gap is a NAMED, fail-closed ``ValueError`` --
        never a silent None to TTS. A seedless production ledger with voiced
        character lines therefore raises here, by design.
        """
        preset_by_id: dict = {}
        for row in cast or []:
            if isinstance(row, dict):
                preset_by_id[str(row.get("char_id") or "")] = row.get("voice_preset")
        missing: list = []
        for ln in lines or []:
            if not isinstance(ln, dict):
                continue
            if str(ln.get("speaker_role") or "").strip().lower() != "character":
                continue
            cid = str(ln.get("char_id") or "")
            preset = preset_by_id.get(cid)
            if not (isinstance(preset, str) and preset.strip()):
                missing.append((ln.get("line_id"), cid))
        if missing:
            detail = ", ".join(
                f"line_id={lid!r} char_id={cid!r}" for lid, cid in missing
            )
            raise ValueError(
                "OTR_CastLock: voice fail-closed gate (node-80 output) -- "
                f"{len(missing)} character line(s) would reach "
                "OTR_BatchCharacterVoices with no voice_preset on the cast row: "
                f"{detail}. The deterministic picker (replay_voice_assignment, "
                "keyed on meta.cast_contract.cast_seed) did not stamp a voice "
                "for them -- a seedless ledger or an unmatched char_id. Refusing "
                "to route a None voice to TTS; re-run the writer so cast_seed and "
                "the character cast rows are consistent."
            )

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
    def _auto_registry(self, led, cast, voice_bank, allow_voice_reuse, report):
        from ._otr_voice_bank import (
            CASTING_POLICY_VERSION, VoiceCastingError, announcer_voice_ref,
            assign_voice_for_slot, load_voice_bank,
        )
        from ._otr_voice_node_common import coerce_int_seed

        bank_entries, _bank_sha = load_voice_bank()
        meta = led.get("meta") or {}
        episode_seed = coerce_int_seed(meta.get("episode_seed"))
        target_engine = self._resolve_char_engine(voice_bank, bank_entries)
        announcer_engine = _DEFAULT_ANNOUNCER_ENGINE

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
            try:
                ref = assign_voice_for_slot(
                    role="char_voice",
                    engine=target_engine,
                    char_id=char_id,
                    gender=gender,
                    timbre=tuple(entry.get("timbre") or ()),
                    age_band=str(entry.get("age_band") or ""),
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
    @staticmethod
    def _stamp(entry, ref) -> None:
        """Stamp the chosen reference onto a cast entry (I-4 / I-9)."""
        entry["voice_ref_id"] = ref.voice_ref_id
        entry["voice_engine"] = ref.engine
        entry["commercial_clean"] = bool(ref.commercial_clean)

    @staticmethod
    def _resolve_char_engine(voice_bank, bank_entries):
        """First legacy-first char_voice engine whose profile allows ``voice_bank``
        AND that has reference entries in the bank. ``None`` if there is none
        (e.g. bark_legacy / kokoro_builtin -> preset engines, no refs)."""
        try:
            from ._otr_engine_profiles import legacy_first_engines, load_resolver

            resolver = load_resolver()
            engines_with_refs = {e.engine for e in bank_entries}
            for eng in legacy_first_engines("char_voice"):
                if eng not in engines_with_refs:
                    continue
                if resolver is None:
                    return eng
                prof = resolver.profile_for("char_voice", eng)
                if prof and voice_bank in prof.allowed_voice_banks:
                    return eng
        except Exception:  # noqa: BLE001
            return None
        return None
