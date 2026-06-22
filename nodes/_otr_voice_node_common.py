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
# adapter metadata (base.AudioEngineAdapter): requires_voice_ref=True plus
# missing_ref_fallback="bark". When such an engine is selected but a char_voice
# line has no usable voice_ref_path (preserve_ledger did not assign a clip ref, or
# no reference clips are installed on disk), the per-line path renders that line on
# the fallback engine (bark preset voices) so the episode always renders -- audio
# is king (PD1). Branching on metadata (not a hard-coded engine-name tuple) means a
# NEW cloning engine -- chatterbox / dia / xtts / cosyvoice -- slots in by setting
# those attributes on its adapter, with ZERO changes here (casting MUST-FIX #6).
def _engine_requires_voice_ref(adapter) -> bool:
    """True iff ``adapter`` is a voice-cloning engine needing a reference WAV."""
    return bool(getattr(adapter, "requires_voice_ref", False))


def _engine_missing_ref_fallback(adapter):
    """Name of the engine to render a line on when a cloning engine has no usable
    reference (e.g. ``"bark"``), or None to render no fallback (let the forward
    fail closed). Reads adapter metadata so the dispatch never hard-codes a name.
    """
    return getattr(adapter, "missing_ref_fallback", None) or None


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
    absolute path ONLY if the file exists on disk; None -> the caller's bark
    fallback renders the line. Never raises (a bad bank just means bark)."""
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
        gender = str(cast.get("gender") or "").strip().lower()
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
            # reference, but a clone engine must still get a REAL voice rather than
            # silently dropping to bark (PD1 + the index-only goal). Pick any ref
            # for this engine, deterministically keyed on char_id so C7 holds.
            import random as _random
            # Prefer a ref whose roles include the active role (so an announcer
            # render gets an announcer ref), then fall back to ANY ref for this
            # engine so a clone engine still never silently drops to bark (PD1).
            role_cands = [e for e in bank if e.engine == engine and role in e.roles]
            cands = sorted(
                role_cands or [e for e in bank if e.engine == engine],
                key=lambda e: e.voice_ref_id,
            )
            if cands:
                _seed = f"{episode_seed}_{cast.get('char_id', '')}_anyref"
                entry = _random.Random(_seed).choice(cands)
    path = _resolve_ref_to_disk(getattr(entry, "ref_path", "") or "")
    return path if (path and os.path.exists(path)) else None


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


def voice_input_types(role, fallback) -> dict:
    """The shared INPUT_TYPES for a v2 voice node (1a / 1b).

    forceInput sockets carry no widget; the only serialized widgets are
    ``engine`` and ``stereo_policy``; there is no ``seed``-named widget and no
    ``model_id`` widget (CLAUDE.md rule 6).
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
            "stereo_policy": (["mono_safe"], {
                "default": "mono_safe",
                "tooltip": (
                    "Channel policy for the per-line path. 'mono_safe' "
                    "downmixes to mono so the mono assembly chain stays "
                    "byte-identical to the legacy engines."
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
        self._bark_fallback_active = None
        try:
            engine = assert_usable(engine, self.ROLE)  # FAIL CLOSED (6-class)
            adapter = get_engine(engine)
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
            # If _render_per_line loaded a bark fallback adapter and an exception
            # skipped its normal post-loop unload, tear it down here so a failed
            # render cannot leave Bark resident (GPT-5.5 roundtable, grounded).
            _fb = getattr(self, "_bark_fallback_active", None)
            if _fb is not None:
                self._teardown(_fb)
                self._bark_fallback_active = None

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
        from ._otr_audio_engines import pack_audio_batch
        from ._otr_audio_utils import resample_audio
        from ._otr_engine_profiles import (
            assert_model_available, assert_token_for_profile, require_resolver,
        )
        from ._otr_resolved_request import (
            _seed_to_int64, build_resolved_request, empty_audio_batch,
        )
        from ._otr_determinism import deterministic_inference
        from ._otr_script_prep import prepare_text as _neutral_prepare_text

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
        # Lazy bark fallback adapter (2026-06-04): loaded only if a voice-cloning
        # char engine (indextts2 / chatterbox) hits a line with no usable
        # reference clip. Torn down after the loop.
        _bark_fb = None
        log_lines: list = [
            f"{self.ROLE}: rendering {len(lines)} lines on '{engine}' "
            f"(profile {profile.profile_id})"
        ]
        for occ, ln in enumerate(lines):
            text = (ln.get("text") or "").strip()
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
            # PD1 robustness (2026-06-22): a beat with NO spoken content -- e.g. a
            # stage-direction-only line like "(pauses, then flips the switch)" --
            # cleans to empty `prepared` text. Handing a per-line voice worker
            # empty text crashes some engines (IndexTTS2: torch.cat() over zero
            # audio chunks -> the whole render dies before publish) and yields
            # garbage in others. Emit a short SILENCE for this beat and skip the
            # engine call -- engine-agnostic, so it future-proofs every approved
            # model. The stage direction was never dialogue; the beat keeps its
            # slot and the episode ships.
            if not str(prepared or "").strip():
                _sil_msg = (
                    f"{self.ROLE}: line={line_id or occ} char={char_id or '-'} "
                    f"has no spoken content (stage-direction-only?) -> emitting "
                    f"silence, skipping the voice worker"
                )
                log_lines.append(_sil_msg)
                log.warning("[OTR voice P-OBS] %s", _sil_msg)
                import torch as _torch
                _n = max(1, int(sr * 0.30))
                clips.append({"waveform": _torch.zeros(1, _n),
                              "sample_rate": int(sr)})
                continue
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
            # Graceful ref-clip fallback (2026-06-04): a voice-CLONING char engine
            # (voice_ref_field == "voice_ref_path", e.g. indextts2 / chatterbox)
            # cannot synthesize without a per-character reference WAV. If this cast
            # row carries none (preserve_ledger does not assign clip refs, and/or
            # no reference clips are installed on disk), render THIS line with bark
            # (preset voices, no ref clip) using the cast's replayed voice_preset,
            # so the episode never hard-fails on a missing voice reference -- audio
            # is king (PD1). The clone engine engages automatically once a usable
            # voice_ref_path is present. Bark is loaded once, lazily.
            if _engine_requires_voice_ref(adapter) and voice_ref:
                # A non-empty but STALE ref (the file is gone) must fall through
                # to resolution + fallback, not be shipped to the worker (which
                # would hard-fail "ref_clip missing") -- PD1. A valid ref is left
                # untouched (the adapter resolves it) so shipped paths are
                # byte-identical; only a missing-on-disk ref is nulled.
                _disk = _resolve_ref_to_disk(voice_ref)
                if not (_disk and os.path.exists(_disk)):
                    voice_ref = None
            if _engine_requires_voice_ref(adapter) and not voice_ref:
                # preserve_ledger / CastLock-stamps-only-the-id: resolve an
                # on-disk reference WAV from voice_ref_id or by gender so the
                # clone engine clones a real voice. None -> fallback below.
                voice_ref = _resolve_clone_ref_path(engine, cast, episode_seed, role=self.ROLE)
            fb_name = _engine_missing_ref_fallback(adapter)
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
            if (self.ROLE in ("char_voice", "announcer_voice")
                    and _engine_requires_voice_ref(adapter)
                    and not voice_ref and fb_name):
                if _bark_fb is None:
                    from ._otr_audio_engines import get_engine as _get_engine
                    _bark_fb = _get_engine(fb_name)
                    _bark_fb.load()
                    self._bark_fallback_active = _bark_fb
                    log_lines.append(
                        f"{self.ROLE}: WARNING engine '{engine}' has no reference "
                        f"clip for one or more lines; rendering those on "
                        f"'{fb_name}' (preset voices). Install {engine} reference "
                        f"WAVs (or set cast_voice_policy=auto_registry) to enable it."
                    )
                bark_seed = _seed_to_int64(fb_name, request.stable_line_seed)
                _pobs_fb = (
                    f"{self.ROLE}: line={line_id or occ} char={char_id or '-'} -> "
                    f"FALLBACK engine={fb_name} preset="
                    f"{voice_preset or 'v2/en_speaker_6'} (no usable ref for "
                    f"'{engine}') delivery=omitted seed={bark_seed}"
                )
                log_lines.append(_pobs_fb)
                log.warning("[OTR voice P-OBS] %s", _pobs_fb)
                with deterministic_inference(bark_seed, warn_only=True):
                    audio = _bark_fb.generate_voice(
                        prepared, voice_preset or "v2/en_speaker_6", None, bark_seed,
                    )
                # Mixed-rate fix (BUG-LOCAL voice): bark renders at its native
                # rate (24000), but this batch packs at the primary engine's sr
                # (e.g. indextts2 22050). Downsample the fallback clip to sr so
                # pack_audio_batch's single-rate contract holds -- otherwise a
                # cast that mixes ref'd (indextts2) and ref-less (bark fallback)
                # characters crashes with "mixed sample rates [22050, 24000]".
                # Primary-engine clips are never touched (C7 bit-exact);
                # resample_audio is deterministic CPU (scipy.resample_poly, I-11).
                if int(audio.get("sample_rate", sr)) != sr:
                    audio = resample_audio(audio, sr)
                clips.append(audio)
                continue
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
        if _bark_fb is not None:
            try:
                _bark_fb.unload()
            except Exception:  # noqa: BLE001 -- teardown must not mask the render
                pass
        self._bark_fallback_active = None
        packed = pack_audio_batch(clips, sample_rate=sr, mono=mono)
        n = int(packed["waveform"].shape[0]) if packed["waveform"].numel() else 0
        log_lines.append(f"{self.ROLE}: packed {n} clips at {sr} Hz")
        return packed, log_lines, n

    # ------------------------------------------------------------------ #
    @staticmethod
    def _teardown(adapter):
        """I-7 teardown: unload the engine + free VRAM. Best-effort, never
        raises (a teardown failure must not mask the render result)."""
        try:
            if adapter is not None and hasattr(adapter, "unload"):
                adapter.unload()
        except Exception:  # noqa: BLE001
            pass
        try:
            gc.collect()
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
