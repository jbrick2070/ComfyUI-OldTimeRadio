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
import logging
import os

log = logging.getLogger("OTR")

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
    'models/TTS/refs/indextts2/ix_male_warm.wav') to an absolute path. Mirrors the
    indextts2 adapter's own resolution so this existence check matches what the
    worker will actually open."""
    if not ref_path:
        return None
    if os.path.isabs(ref_path):
        return ref_path
    rp = ref_path.replace("\\", "/")
    stripped = rp[len("models/"):] if rp.startswith("models/") else rp
    candidates = []
    try:
        import folder_paths

        md = folder_paths.models_dir
        candidates.append(os.path.join(os.path.dirname(md), ref_path))  # <base>/models/...
        candidates.append(os.path.join(md, stripped))                   # <models_dir>/TTS/...
    except Exception:  # noqa: BLE001 -- non-Comfy contexts (tests / CLI)
        pass
    # Known extra models root after the Comfy Desktop 1.0.4 model-path migration.
    candidates.append(os.path.join("C:\\ComfyUI-Models", stripped))
    candidates.append(os.path.abspath(ref_path))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[0] if candidates else None


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
        # NORMALIZED (item 8, 2026-08-06). This compared the RAW stored value
        # against the bank's `male`/`female`, so a row recorded as `woman`,
        # `man`, `m` or `f` matched nothing and fell through to the
        # gender-agnostic path -- a correctly-gendered character silently
        # getting a voice of any gender. `other` still falls through, which is
        # correct: the bank carries no `other` references.
        from ._otr_roster_gender import normalize_gender
        gender = normalize_gender(cast.get("gender"))
        if gender in ("male", "female"):
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
        # NORMALIZED (item 8, 2026-08-06). This compared the RAW stored value
        # against the bank's `male`/`female`, so a row recorded as `woman`,
        # `man`, `m` or `f` matched nothing and fell through to the
        # gender-agnostic path -- a correctly-gendered character silently
        # getting a voice of any gender. `other` still falls through, which is
        # correct: the bank carries no `other` references.
        from ._otr_roster_gender import normalize_gender
        gender = normalize_gender(cast.get("gender"))
        if gender in ("male", "female"):
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
    def VALIDATE_INPUTS(cls, **kwargs):
        # D: local-disk only -- never touch the network at validation time.
        # Engine usability + model/token presence are enforced fail-closed on
        # the generate() path (C-5/C-7), not here, so a box-fresh graph still
        # validates clean.
        return True

    # ------------------------------------------------------------------ #
    def generate(self, script_json, engine, ledger_json="", gate_in="",
                 stereo_policy="mono_safe"):
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
            assert_model_available, assert_token_for_profile, require_resolver,
        )
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
        _delivery_on = os.getenv("OTR_DELIVERY_VECTOR", "1") != "0"
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
            voice_ref_id = cast.get("voice_ref_id")
            voice_preset = cast.get("voice_preset")
            if ref_field == "voice_ref_path":
                voice_ref = cast.get("voice_ref_path") or cast.get("ref_path")
            else:
                voice_ref = cast.get(ref_field)
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
                params=dict(profile.default_params or {}),
                commercial_clean=profile.commercial_clean,
            )
            # G1: per-engine external seed reduced from the stable line seed.
            engine_seed = _seed_to_int64(engine, request.stable_line_seed)
            # Ref-clip resolution (no-fallback rip 2026-07-03): a voice-CLONING char
            # engine (voice_ref_field == "voice_ref_path", e.g. indextts2 /
            # chatterbox) cannot synthesize without a per-character reference WAV.
            # We resolve the cast row's OWN ref below; if none resolves, the line
            # FAILS LOUD (no bark fallback). A valid ref is left byte-identical.
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
            # ---- P-OBS (whiny-fix v3.1): per-line attribution, ALWAYS -------
            # char -> voice_ref_id -> ref basename -> engine -> alpha ->
            # delivery version/state/source -> seed. Render_log line + runtime
            # log mirror; this is the observability floor every later step
            # (P0-zero, the audit, P3a durable stamping) builds on.
            _alpha_fn = getattr(adapter, "current_emo_alpha", None)
            _alpha = _alpha_fn() if callable(_alpha_fn) else None
            if delivery_vector is None:
                _vec_state = "omitted"
            elif not any(float(v or 0.0) > 0.0 for v in delivery_vector.values()):
                _vec_state = "zero"
            else:
                _vec_state = "nonzero"
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
                f"delivery={_DTV}:{_vec_state}({_dv_source}) seed={engine_seed}"
            )
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
            clips.append(audio)
        packed = pack_audio_batch(clips, sample_rate=sr, mono=mono)
        n = int(packed["waveform"].shape[0]) if packed["waveform"].numel() else 0
        log_lines.append(f"{self.ROLE}: packed {n} clips at {sr} Hz")
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
