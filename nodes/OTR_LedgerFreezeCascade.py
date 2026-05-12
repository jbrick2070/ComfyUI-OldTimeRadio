"""nodes/OTR_LedgerFreezeCascade.py — Ledger Freeze Cascade ComfyUI node.

Renamed in LFC sprint commit 2 from `OTR_LedgerScriptReviewer`. The
previous module is preserved as a back-compat shim that re-exports
this class so existing workflow JSONs referencing the old name
continue to load.

Wires AFTER OTR_LedgerScriptWriter and BEFORE SceneSequencer. Touches
the production ledger only; emits no audio, no video, no LLM weights
beyond what the writer already loaded.

Output contract (preserved from the OTR_LedgerScriptReviewer 5-slot
shape -- ADR section 9):
    script_text, script_json, news_used, estimated_minutes, freeze_verdict

The fifth slot is renamed `reviewer_verdict` -> `freeze_verdict`. The
verdict literal set is extended:

    frozen_clean
    frozen_with_warns
    frozen_with_doctor_edits
    cast_unrecoverable        (preserved)
    too_many_edits            (preserved)
    needs_full_rerun          (preserved)
    post_audit_failed         (preserved)

Status: LFC sprint commit 2 of 14 (2026-05-11).
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerFreezeCascade"]


DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"


def _no_ledger_error_json(incoming_script_json: str) -> str:
    """Synthesize a parseable error-state JSON when no ledger exists.

    Mirrors the previous OTR_LedgerScriptReviewer behaviour: any
    downstream consumer doing `json.loads(script_json)` gets back a
    dict (not `"{}"` placeholder) flagged with freeze_verdict =
    needs_full_rerun.
    """
    incoming = (incoming_script_json or "").strip()
    if incoming and incoming != "{}":
        return incoming
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
            },
            # Legacy field kept so consumers still keyed on the old
            # name see the same signal until the cascade migration
            # lands across every downstream node.
            "reviewer_verdict": "needs_full_rerun",
        },
    }, indent=2, ensure_ascii=False)


class OTR_LedgerFreezeCascade:
    """Ledger Freeze Cascade -- multi-phase post-writer cleanup.

    Inputs:
      script_text         Forwarded from OTR_LedgerScriptWriter so the
                          graph wires this node in line. Returned
                          rebuilt (from the post-freeze ledger) as the
                          first output slot.
      script_json         Forwarded JSON snapshot from the writer
                          (slot index 1). Re-serialized from the
                          post-freeze ledger in the output.
      news_used           Passthrough of the writer's news_used slot.
      estimated_minutes   Passthrough of the writer's est_minutes INT.
      model_id            HF model ID for the reviewer LLM passes
                          (Phase 1 Auditor, Phase 2 Script Doctor,
                          Phase 9 Auditor). Phase 3/4/4.5/5/6 future
                          LLM phases reuse the same loader.

    Outputs (matches writer's RETURN_TYPES for drop-in midstream
    wiring; fifth slot renamed reviewer_verdict -> freeze_verdict):
      script_text         Rebuilt from the post-freeze ledger.
      script_json         JSON snapshot of the post-freeze ledger.
      news_used           Passthrough.
      estimated_minutes   Passthrough INT.
      freeze_verdict      One of the FreezeVerdict literals.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used",
        "estimated_minutes", "freeze_verdict",
    )

    @classmethod
    def INPUT_TYPES(cls):
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
                "model_id": ("STRING", {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "HF model ID for the reviewer LLM passes "
                        "(Phase 1 / 2 / 9). Default Mistral-Nemo "
                        "matches the writer's default."
                    ),
                }),
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
        model_id: str = DEFAULT_MODEL_ID,
    ):
        # Lazy imports to keep node-load cheap.
        from . import _otr_freeze_cascade as _LFC_ORCH
        from . import _otr_model_loader as _OTRML
        from . import production_ledger as _PL

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
            )

        cache_entry = _OTRML.load_llm(model_id=model_id or DEFAULT_MODEL_ID)
        generate_fn = _OTRML.make_generate_fn(cache_entry)

        log.info(
            "[OTR_LedgerFreezeCascade] running cascade on ledger %s "
            "(%d lines)",
            led.episode_id,
            len(led.data.get("lines", []) or []),
        )
        disp = _LFC_ORCH.run_freeze_cascade(generate_fn, led)
        log.info(
            "[OTR_LedgerFreezeCascade] freeze_verdict=%s "
            "(pre_warns=%d post_warns=%s reviewer=%s)",
            disp.verdict,
            len(disp.gap_audit_pre.warnings),
            (
                len(disp.gap_audit_post.warnings)
                if disp.gap_audit_post is not None
                else "n/a"
            ),
            (
                disp.reviewer_disposition.verdict
                if disp.reviewer_disposition is not None
                else "n/a"
            ),
        )

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
            rebuilt_script_text = _PL.assemble_script_text_from_ledger(
                led.data,
            ) or (script_text or "")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[OTR_LedgerFreezeCascade] assemble_script_text_from_"
                "ledger raised (%s); falling back to incoming "
                "script_text.", exc,
            )
            rebuilt_script_text = script_text or ""

        return (
            rebuilt_script_text,
            updated_script_json,
            news_used or "",
            int(estimated_minutes or 0),
            disp.verdict,
        )
