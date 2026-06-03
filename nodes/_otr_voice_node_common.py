"""Shared dispatch core for the v2 voice nodes (Wave 1 / 1a + 1b).

OTR_BatchCharacterVoices (1a) and OTR_AnnouncerVoice (1b) share ONE dispatch
contract: pick an engine from the registry, FAIL CLOSED, and either delegate
verbatim to a legacy batch node (byte-identical) or render per line into the
existing Bark AUDIO-batch contract -- with engine teardown in ``finally`` BEFORE
the ``done`` signal (I-7). This module holds that core so each node file declares
only its role, its INPUT_TYPES, and its output names.

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

log = logging.getLogger("OTR")

_MANIFEST_FILENAME = "legacy_invocation_manifest.json"


# --------------------------------------------------------------------------- #
# Pure helpers (no IO at import; the manifest read happens on the dispatch path)
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


def _manifest_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "config", _MANIFEST_FILENAME)


def frozen_batch_widgets(node, role) -> list:
    """Frozen non-``script_json`` widget defaults for a legacy batch node.

    Reads ``config/legacy_invocation_manifest.json`` (generate-path IO -- never
    INPUT_TYPES) and returns the serialized widget defaults in declaration
    order, dropping any ``script_json`` widget (passed explicitly first). These
    frozen values are what the R0a render-twice baseline was captured against,
    so passing them verbatim keeps raw delegation byte-identical (re-baseline
    trigger: ``legacy_manifest_sha``).
    """
    from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason

    cls_name = type(node).__name__
    with open(_manifest_path(), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entry = None
    for spec in (manifest.get("nodes") or {}).values():
        if spec.get("class") == cls_name:
            entry = spec
            break
    if entry is None:
        raise EngineUnusable(
            cls_name, role, EngineUsabilityReason.MALFORMED_CONFIG,
            f"no legacy invocation manifest entry for class {cls_name!r}",
        )
    return [
        w.get("default")
        for w in (entry.get("widgets") or [])
        if w.get("name") != "script_json"
    ]


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
        try:
            engine = assert_usable(engine, self.ROLE)  # FAIL CLOSED (6-class)
            adapter = get_engine(engine)
            interface = getattr(adapter, "interface", "per_line")
            sr_hint = int(getattr(adapter, "sample_rate", 24000) or 24000)

            if interface == "batch":
                audio_out, line, n_clips = self._delegate_batch(
                    adapter, engine, script_json,
                )
                render_log.append(line)
            elif interface == "per_line":
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
    def _delegate_batch(self, adapter, engine, script_json):
        """Raw, byte-identical delegation to the legacy batch node (I-3).

        The legacy node is handed the EXACT ``script_json`` string object and
        the frozen manifest widget tuple -- zero transform. Returns the legacy
        AUDIO output (slot 0) unchanged, asserted against the AUDIO-batch
        contract, plus a log line and the batch size.
        """
        from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason
        from ._otr_resolved_request import assert_audio_batch_contract

        node = adapter.make_batch_node()
        func_name = getattr(node, "FUNCTION", None)
        fn = getattr(node, func_name, None) if func_name else None
        if not callable(fn):
            raise EngineUnusable(
                engine, self.ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                f"legacy batch node {type(node).__name__!r} has no callable "
                f"FUNCTION {func_name!r}",
            )
        widgets = frozen_batch_widgets(node, self.ROLE)
        # Verbatim: exact upstream string first, then the frozen widget tuple.
        result = fn(script_json, *widgets)
        audio = result[0] if isinstance(result, tuple) else result
        audio = assert_audio_batch_contract(
            audio, where=f"{type(self).__name__}.delegate",
        )
        n = int(audio["waveform"].shape[0]) if audio["waveform"].numel() else 0
        line = (
            f"{self.ROLE}: delegated to legacy '{engine}' (batch, verbatim "
            f"script_json + {len(widgets)} frozen widgets)"
        )
        return audio, line, n

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
        clips: list = []
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
            ref_clip_path = cast.get("voice_ref_path") or cast.get("ref_path")
            prepared = prep(text, None) if callable(prep) else _neutral_prepare_text(text)
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
            # G1: scope strict-determinism + seed/restore every RNG around the
            # single forward (I-2/C-2). warn_only=True keeps the process default
            # non-strict so a nondeterministic CUDA op cannot crash the opt-in
            # render on sm_120; bit_exact (warn_only=False) is gated on the F
            # pilot verifying each engine binds an external torch.Generator.
            with deterministic_inference(engine_seed, warn_only=True):
                audio = adapter.generate_voice(
                    prepared, ref_clip_path, None, engine_seed,
                )
            clips.append(audio)
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
