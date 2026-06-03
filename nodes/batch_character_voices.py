"""OTR_BatchCharacterVoices -- model-agnostic character-voice node (Wave 1 / 1a).

The generic character-voice surface for the opt-in v2 audio lane. It picks its
engine from the shared audio-engine registry (bark / chatterbox / indextts2)
instead of being hardcoded to one model, and dispatches FAIL-CLOSED:

  * ``batch`` engines (the legacy Bark default) -> RAW delegation to the existing
    OTR_BatchBarkGenerator with the EXACT upstream ``script_json`` string plus the
    frozen widget tuple from ``config/legacy_invocation_manifest.json``. Zero
    transform (no loads/dumps, no canonicalize, no stamp), so the default path
    stays byte-identical to the legacy baseline by construction (I-1, I-3).
  * ``per_line`` engines (chatterbox / indextts2, opt-in + flag-gated) -> build a
    frozen ``ResolvedVoiceRequest`` per dialogue line, prepare the spoken text,
    call the adapter, and pack the per-line clips into the existing Bark
    AUDIO-batch contract ``{"waveform": [B, C, T], "sample_rate": int}`` (C-4).

Invariants honored:
  * **C-5 INPUT_TYPES is exception-safe, never-empty, IO-free.** The engine combo
    is built from the hardcoded legacy-first list; no YAML read, no token check,
    no engine-lib import at import time or in INPUT_TYPES.
  * **C-7 fail-closed, named.** ``assert_usable`` raises a classified
    ``EngineUnusable`` at execute time; the node never silently swaps engines.
  * **I-7 single-engine residency.** Engine teardown runs in ``finally`` BEFORE
    the ``done`` signal is produced, so a downstream gate bound to ``done``
    implies this engine is unloaded.
  * **Gate chain.** ``gate_in`` (STRING forceInput, optional on this first chain
    node) creates the ordering edge; ``done`` is the non-empty completion signal
    the next node's ``gate_in`` binds to (E.5). Even a zero-line role emits
    ``done`` and the canonical empty batch.

Engine libraries are lazy-imported INSIDE the adapter ``load``/``generate`` calls,
never at module import (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import gc
import json
import logging
import os

log = logging.getLogger("OTR")

# Registry role id for character voices (matches the audio-engine registry and
# the engine-profile YAML -- NOT "character_voice").
ROLE = "char_voice"

# C-5 hardcoded, legacy-first fallback used if the profiles helper is somehow
# unavailable at INPUT_TYPES time. Order is stable across flags; index 0 is the
# byte-identical legacy default.
_LEGACY_FIRST_FALLBACK = ("bark", "chatterbox", "indextts2")

_MANIFEST_FILENAME = "legacy_invocation_manifest.json"


def _coerce_int_seed(val) -> int:
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


def _frozen_batch_widgets(node) -> list:
    """Frozen non-``script_json`` widget defaults for a legacy batch node.

    Reads ``config/legacy_invocation_manifest.json`` (generate-path IO -- never
    INPUT_TYPES) and returns the widget defaults in declaration order, dropping
    the ``script_json`` widget (passed explicitly first). These frozen values are
    what the R0a render-twice baseline was captured against, so passing them
    verbatim keeps raw delegation byte-identical (re-baseline trigger:
    ``legacy_manifest_sha``).
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
            cls_name, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
            f"no legacy invocation manifest entry for class {cls_name!r}",
        )
    return [
        w.get("default")
        for w in (entry.get("widgets") or [])
        if w.get("name") != "script_json"
    ]


class BatchCharacterVoices:
    """Generic character-voice node. Registered as ``OTR_BatchCharacterVoices``."""

    CATEGORY = "OldTimeRadio/v2/audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("character_audio", "render_log", "done")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # C-5: never empty, stable order, no IO. Use the hardcoded legacy-first
        # list; only consult the (pure) helper opportunistically.
        engines = list(_LEGACY_FIRST_FALLBACK)
        try:
            from ._otr_engine_profiles import legacy_first_engines

            got = legacy_first_engines(ROLE)
            if got:
                engines = list(got)
        except Exception:  # noqa: BLE001 -- INPUT_TYPES must never crash (C-5)
            engines = list(_LEGACY_FIRST_FALLBACK)
        if not engines:
            engines = list(_LEGACY_FIRST_FALLBACK)
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "forceInput": True,
                    "tooltip": (
                        "Frozen v2 ledger JSON from OTR_LedgerFreezeCascade "
                        "(node 62 slot 1). Passed VERBATIM to the legacy engine "
                        "on the byte-identical batch path."
                    ),
                }),
                "engine": (engines, {
                    "default": engines[0],
                    "tooltip": (
                        "Character-voice engine. Legacy Bark is the byte-"
                        "identical default; chatterbox / indextts2 are opt-in "
                        "(flag-gated) until the GPU dependency pilot promotes "
                        "them. Unusable selections fail closed with a named "
                        "error at queue time."
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
                        "voice_ref_id / episode_seed). Read on the per-line "
                        "path; the batch path ignores it and delegates the raw "
                        "script_json. No episode_seed widget (I-4)."
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Optional ordering signal. Wire an upstream 'done' here "
                        "to force this node to run after it. Optional on the "
                        "first chain node; the audio chain binds it node->node."
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
            # FAIL CLOSED (6-class taxonomy). Never silent-swaps (C-6).
            engine = assert_usable(engine, ROLE)
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
                    engine, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                    f"engine {engine!r} declares unsupported interface "
                    f"{interface!r} for role {ROLE!r}",
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

        The legacy node is handed the EXACT ``script_json`` string object and the
        frozen manifest widget tuple -- zero transform. Returns the legacy AUDIO
        output unchanged (asserted against the AUDIO-batch contract) plus a log
        line and the batch size.
        """
        from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason
        from ._otr_resolved_request import assert_audio_batch_contract

        node = adapter.make_batch_node()
        func_name = getattr(node, "FUNCTION", None)
        fn = getattr(node, func_name, None) if func_name else None
        if not callable(fn):
            raise EngineUnusable(
                engine, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                f"legacy batch node {type(node).__name__!r} has no callable "
                f"FUNCTION {func_name!r}",
            )
        widgets = _frozen_batch_widgets(node)
        # Verbatim: exact upstream string first, then the frozen widget tuple.
        result = fn(script_json, *widgets)
        audio = result[0] if isinstance(result, tuple) else result
        audio = assert_audio_batch_contract(
            audio, where="BatchCharacterVoices.delegate",
        )
        n = int(audio["waveform"].shape[0]) if audio["waveform"].numel() else 0
        line = (
            f"char_voice: delegated to legacy '{engine}' (batch, verbatim "
            f"script_json + {len(widgets)} frozen widgets)"
        )
        return audio, line, n

    # ------------------------------------------------------------------ #
    def _render_per_line(self, adapter, engine, script_json, ledger_json,
                         stereo_policy):
        """Per-line dispatch for cloning engines (chatterbox / indextts2).

        Builds one frozen ``ResolvedVoiceRequest`` per character line, prepares
        the spoken text, calls the adapter, and packs the clips into the Bark
        AUDIO-batch contract (C-4). Zero character lines short-circuits to the
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
        from ._otr_script_prep import prepare_text as _neutral_prepare_text

        sr = int(getattr(adapter, "sample_rate", 24000) or 24000)
        # Prefer the cast-locked ledger; fall back to the frozen writer ledger.
        source = ledger_json if (ledger_json or "").strip() else script_json
        led = _OTRLC.load_ledger(source)
        lines = [
            ln for ln in _OTRLC.iter_lines(led, roles={"character"})
            if (ln.get("text") or "").strip()
        ]
        if not lines:
            return empty_audio_batch(sr), ["char_voice: 0 character lines"], 0

        # Resolve the curated engine profile (fail-closed) + enforce token/model
        # presence on the generate path only (C-5).
        resolver = require_resolver()
        profile = resolver.resolve_casting_plan(role=ROLE, engine=engine)
        assert_token_for_profile(profile)
        assert_model_available(profile)
        sr = int(profile.sample_rate or sr)

        meta = led.get("meta") or {}
        episode_seed = _coerce_int_seed(meta.get("episode_seed"))
        cast_lock_revision = int(meta.get("cast_lock_revision") or 0)
        mono = (stereo_policy == "mono_safe")

        prep = getattr(adapter, "prepare_text", None)
        clips: list = []
        log_lines: list = [
            f"char_voice: rendering {len(lines)} lines on '{engine}' "
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
                role=ROLE,
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
            audio = adapter.generate_voice(prepared, ref_clip_path, None, engine_seed)
            clips.append(audio)
        packed = pack_audio_batch(clips, sample_rate=sr, mono=mono)
        n = int(packed["waveform"].shape[0]) if packed["waveform"].numel() else 0
        log_lines.append(f"char_voice: packed {n} clips at {sr} Hz")
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

    @staticmethod
    def _done_signal(engine, n_clips) -> str:
        """Non-empty completion sentinel for the done->gate chain (E.5).

        Bounded + informative; the next node binds its ``gate_in`` to this so it
        cannot fire until this node has returned (and torn down, per I-7).
        """
        return f"char_voice:done:engine={engine}:clips={int(n_clips)}"
