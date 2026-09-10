"""ComfyUI node for deterministic ledger validation, readiness, and freeze.

The node preserves the writer's accepted story: it audits structure, runs the
audio/video readiness passes, freezes the ledger and releases any local LLM
still resident. It acquires no model and rewrites no content; word length,
visual vocabulary, style, craft, and quality never affect publication.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerFreezeCascade"]


# S30 B3: DEFAULT_MODEL_ID literal DELETED. The cascade no longer
# carries a local widget default; the technical_model id arrives over
# the wire from the writer's broadcast output socket. An unwired
# socket triggers MissingModelInputError at run-time -- the recovery
# is graph-level (connect the writer's `technical_model` output),
# not a code-side fallback.


def _no_ledger_error_json(incoming_script_json: str) -> str:
    """Synthesize a parseable error-state JSON when no ledger exists.

    Always stamps the synthetic-error-state shape regardless of
    whether the incoming script_json is empty -- consumers parsing
    `meta.freeze_verdict` and `schema_version` see a consistent
    signal.

    The incoming JSON content (truncated to 200 chars) is preserved
    on `meta.freeze_disposition.skipped_reason_detail` for forensic
    inspection.
    """
    incoming = (incoming_script_json or "").strip()
    detail = ""
    if incoming and incoming != "{}":
        # Preserve a forensic snippet so soak diagnostics can see
        # what the writer DID produce, even though the ledger handle
        # was lost.
        detail = incoming[:200]
    return json.dumps({
        "schema_version": "synthetic_error_state",
        "lines": [],
        "cast": [],
        "meta": {
            "freeze_verdict": "needs_full_rerun",
            "freeze_disposition": {
                "verdict": "needs_full_rerun",
                "skipped": True,
                "skipped_reason": "no_writer_produced_ledger",
                "skipped_reason_detail": detail,
            },
        },
    }, indent=2, ensure_ascii=False)


def _episode_seed_from_ledger(ledger_json: str) -> int:
    """Derive a stable, read-only ``episode_seed`` from the FROZEN ledger JSON.

    Pure function of the locked ledger content -- NEVER stamped back into the
    ledger, so out[1] (script_json) stays byte-identical (R0a / I-2). The
    ``episode_seed_v1`` domain tag namespaces the reduction.
    """
    from ._otr_resolved_request import _seed_to_int64
    return _seed_to_int64("episode_seed_v1", ledger_json or "{}")


class OTR_LedgerFreezeCascade:
    """Finalize the accepted ledger without quality-driven reauthoring."""

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used",
        "estimated_minutes", "freeze_verdict",
        # R0a: appended at indices 5,6 -- never inserted (outputs 0-4 frozen).
        "episode_seed", "v2_ledger_json",
    )
    # OUTPUT_NODE so a stripped "story-only" workflow (validator -> writer ->
    # freeze) can terminate HERE: ComfyUI runs the freeze as a terminal and the
    # frozen ledger lands on disk with no downstream media node. In the full
    # canonical this node already always executes, so marking it terminal is
    # inert there -- downstream nodes still consume its outputs normally.
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # LFC sprint commit 12 (2026-05-11). Each enable_phase_*
        # widget gates the corresponding cascade phase. New phases
        # default OFF until soak validates them; deterministic
        # phases (7, 8, 10) default ON because they are cheap +
        # high-value.
        return {
            "required": {
                "script_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Passthrough from OTR_LedgerScriptWriter. The "
                        "cascade reads the production ledger directly "
                        "via peek_ledger(); this socket exists so the "
                        "graph wires the writer and cascade in line."
                    ),
                }),
            },
            "optional": {
                "script_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Forwarded from the writer's script_json slot. "
                        "The cascade re-serializes this from the "
                        "post-freeze ledger so downstream consumers "
                        "see the final state."
                    ),
                }),
                "news_used": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Passthrough of the writer's news_used slot. "
                        "Not touched by the cascade."
                    ),
                }),
                "estimated_minutes": ("INT", {
                    "forceInput": True,
                    "tooltip": (
                        "Passthrough of the writer's estimated_minutes "
                        "INT slot. Not touched by the cascade."
                    ),
                }),
                # S30 B3: model_id widget + 6 phase-toggle widgets
                # DELETED. The writer's broadcast `technical_model`
                # socket below is kept for graph compatibility: the
                # cascade validates only that a non-empty id arrived
                # (an unwired socket delivers the empty default) and
                # acquires nothing (the reviewer passes that once
                # consumed it are gone). Phase 3/4/4.5/5/6 toggles were all
                # defaulted OFF and the surrounding standalone LFC
                # nodes went away in B4 -- the cascade never invoked
                # those phases in any shipped workflow.
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Technical model ID from the writer. Required "
                        "for compatibility on a normal current-ledger "
                        "run; freeze does not acquire or generate with "
                        "this model. Validated via "
                        "_otr_model_inputs.require_model -- an unwired "
                        "or blank socket raises MissingModelInputError "
                        "loud."
                    ),
                }),
                "enable_phase_7_audio_readiness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "LFC Phase 7 -- audio readiness. Expands "
                        "abbreviations (Dr. -> Doctor), symbols (& -> "
                        "and), and numbers (42 -> forty-two) so TTS "
                        "produces pronounceable output. Default ON "
                        "(deterministic + cheap)."
                    ),
                }),
                "enable_phase_8_video_readiness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "LFC Phase 8 -- video readiness audit. Checks "
                        "cast portraits + voiced-line visual coverage. "
                        "Mutates nothing; stamps meta.video_readiness. "
                        "Default ON."
                    ),
                }),
                # Positional compatibility inputs retained because the canonical
                # workflow is immutable in this change. All four are ignored;
                # real render selection belongs exclusively to ShotLock.
                # Four "deprecated compatibility inputs" (render_selection,
                # render_max_n, protagonist_only, manual_line_ids) were REMOVED
                # 2026-08-28: each was accepted and immediately deleted, so the
                # UI showed four knobs that controlled nothing. They were the
                # TRAILING widget suffix, so the saved-graph migration was
                # [true,true,"all",6,false,""] -> [true,true] with no re-index
                # of survivors and no link movement (safety-gated: no link in
                # any of the 63 graphs targeted their slots).
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time as _t
        return _t.time()

    def run(
        self,
        script_text: str = "",
        script_json: str = "",
        news_used: str = "",
        estimated_minutes: int = 0,
        # S30 B3: technical_model arrives via input socket (no widget).
        # The 6 phase-toggle kwargs are deleted; the orchestrator's
        # function defaults still keep Phases 3/4/4.5/5/6 OFF.
        technical_model: str = "",
        enable_phase_7_audio_readiness: bool = True,
        enable_phase_8_video_readiness: bool = True,
    ):

        # Lazy imports to keep node-load cheap.
        from . import _otr_freeze_cascade as _LFC_ORCH
        from . import _otr_model_loader as _OTRML
        from . import _otr_model_inputs as _OTRMI
        from . import production_ledger as _PL

        # CANONICAL REPLAY (campaign item 0): the bundle's ledger is already
        # frozen (its freeze_timestamp is kept byte-identical by the import), so
        # the cascade neither validates the socket nor re-mints the receipt.
        # Same JSON on script_json and v2_ledger_json, verdict "replay".
        try:
            _rmeta = (json.loads(script_json or "{}") or {}).get("meta") or {}
        except (ValueError, TypeError, AttributeError):
            # AttributeError: a non-dict wire (the legacy parser LIST) is not a replay
            _rmeta = {}
        if _PL.replay_descriptor(_rmeta):
            log.warning("[OTR_LedgerFreezeCascade] REPLAY: pass-through, no freeze, "
                        "no model (workspace %s)", _rmeta.get("replay_workspace_id"))
            return (
                script_text or "",
                script_json or "{}",
                news_used or "",
                int(estimated_minutes or 0),
                "replay",
                _episode_seed_from_ledger(script_json or "{}"),
                script_json or "{}",
            )
        has_current = getattr(_PL, "has_current_ledger", None)
        peek = getattr(_PL, "peek_ledger", None)
        if callable(has_current) and not has_current():
            log.warning(
                "[OTR_LedgerFreezeCascade] no writer-produced ledger "
                "in this process; returning needs_full_rerun without "
                "running LLM calls."
            )
            return (
                script_text or "",
                _no_ledger_error_json(script_json),
                news_used or "",
                int(estimated_minutes or 0),
                "needs_full_rerun",
                0,
                _no_ledger_error_json(script_json),
            )
        led = (peek() if callable(peek) else _PL.get_ledger())
        if led is None:
            log.warning(
                "[OTR_LedgerFreezeCascade] ledger handle is None; "
                "returning needs_full_rerun."
            )
            return (
                script_text or "",
                _no_ledger_error_json(script_json),
                news_used or "",
                int(estimated_minutes or 0),
                "needs_full_rerun",
                0,
                _no_ledger_error_json(script_json),
            )

        # The technical id is validated for graph compatibility only:
        # require_model rejects a blank/whitespace value, which is what an
        # unwired socket delivers; it checks neither catalog membership nor
        # connectivity. The cascade is deterministic and never generates with
        # this model, so nothing is acquired here. The slot request that used to sit at this
        # point fed a callback the orchestrator has not invoked since the
        # same-story cleanup was retired (2026-08-05); removed 2026-09-10.
        _OTRMI.require_model(technical_model, slot="technical")

        log.info(
            "[OTR_LedgerFreezeCascade] running cascade on ledger %s "
            "(%d lines)",
            led.episode_id,
            len(led.data.get("lines", []) or []),
        )

        # B1 fix (commit 12.12, 2026-05-12): wrap the cascade body
        # in try/finally so unload_llm() runs even when
        # run_freeze_cascade raises (LLM OOM, pydantic crash, etc.).
        # Pre-fix the unload sat outside the try block; on cascade
        # exception VRAM stayed held and the next downstream visual
        # node (HuMo / LTX / SignalLostVideo) hit OOM on top of an
        # un-released Mistral-Nemo cache. The whole point of B14 +
        # C7 was VRAM-safe handoff.
        disp = None
        updated_script_json = script_json or "{}"
        rebuilt_script_text = script_text or ""
        unload_ok = True
        try:
            # S30 B3: Phase 3/4/4.5/5/6 toggles deleted at the
            # cascade-NODE surface. The orchestrator's defaults
            # (all OFF) carry them; B4 deletes the underlying
            # phase functions from _otr_lfc.py. The first positional
            # argument is the orchestrator's public generation
            # callback; it is never invoked, so no callable is built.
            disp = _LFC_ORCH.run_freeze_cascade(
                None,
                led,
                enable_phase_7_audio_readiness=enable_phase_7_audio_readiness,
                enable_phase_8_video_readiness=enable_phase_8_video_readiness,
            )
            log.info(
                "[OTR_LedgerFreezeCascade] freeze_verdict=%s "
                "(pre_warns=%d post_warns=%s cleanup=%s)",
                disp.verdict,
                len(disp.gap_audit_pre.warnings),
                (
                    len(disp.gap_audit_post.warnings)
                    if disp.gap_audit_post is not None
                    else "n/a"
                ),
                (
                    (disp.cleanup_receipt or {}).get("status", "n/a")
                    if isinstance(disp.cleanup_receipt, dict)
                    else "n/a"
                ),
            )

            # First serialization + script rebuild, BEFORE the final
            # unload gate in the finally block. Neither touches torch
            # tensors (assemble_script_text_from_ledger is pure
            # dict/string work; json.dumps walks the meta tree). The
            # second serialization after the finally block carries the
            # unload receipt when it succeeds.
            try:
                updated_script_json = json.dumps(
                    led.data, indent=2, ensure_ascii=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LedgerFreezeCascade] failed to serialize "
                    "post-freeze ledger to JSON (%s); falling back to "
                    "incoming script_json.", exc,
                )
                updated_script_json = script_json or "{}"

            try:
                rebuilt_script_text = (
                    _PL.assemble_script_text_from_ledger(led.data)
                    or (script_text or "")
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LedgerFreezeCascade] assemble_script_text_"
                    "from_ledger raised (%s); falling back to "
                    "incoming script_text.", exc,
                )
                rebuilt_script_text = script_text or ""
        finally:
            # B14 (commit 12.5) + B1 (commit 12.12): release any local
            # LLM still resident before downstream visual nodes load.
            # The freeze acquires nothing itself, but the writer's own
            # teardown can be skipped (OTR_WRITER_UNLOAD_AFTER_SCRIPT=0),
            # so this conditional gate stays. Wrapped in best-effort
            # try/except -- an unload failure logs at WARNING + stamps
            # meta.freeze_unload_ok=False so the next visual node can
            # branch on the stamp instead of OOM-ing on top of a leaked
            # cache. The cascade itself still returns its verdict; the
            # downstream visual nodes decide what to do about a failed
            # unload.
            try:
                _OTRML.unload_llm_if_local_resident()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LedgerFreezeCascade] unload_llm at cascade "
                    "exit raised (%s); VRAM may not be released "
                    "before downstream nodes load",
                    exc,
                )
                unload_ok = False
            # Stamp on meta so soak diagnostics see the unload
            # outcome without grepping stderr. Best-effort: a
            # malformed ledger handle should not break the return.
            try:
                if hasattr(led, "data") and isinstance(led.data, dict):
                    led.data.setdefault("meta", {})[
                        "freeze_unload_ok"
                    ] = unload_ok
            except Exception:  # noqa: BLE001
                pass

        # S34 B2 (2026-05-15): second serialization, so the
        # freeze_unload_ok stamp set in the finally block above is
        # visible to downstream JSON consumers. The first
        # serialization preceded the unload and lacks the stamp; if
        # this one fails, the first result (or the incoming JSON when
        # both failed) is returned and the stamp may not reach the
        # wire.
        try:
            updated_script_json = json.dumps(
                led.data, indent=2, ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[OTR_LedgerFreezeCascade] failed to reserialize "
                "post-unload ledger to JSON (%s); freeze_unload_ok "
                "stamp may not reach downstream consumers.", exc,
            )
            # Keep the pre-finally serialization as best-effort fallback.

        # Cascade body completed (any exception propagated out of
        # the try/finally above and ComfyUI rendered the node red,
        # which is the correct loud-failure convention -- the
        # finally still ran unload_llm so VRAM is released).
        # disp is non-None here because the cascade body returned
        # without raising.
        return (
            rebuilt_script_text,
            updated_script_json,
            news_used or "",
            int(estimated_minutes or 0),
            disp.verdict,
            _episode_seed_from_ledger(updated_script_json),
            updated_script_json,
        )
