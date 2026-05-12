"""nodes/OTR_LFCPhase5Voice.py — standalone LFC Phase 5 ComfyUI node.

See OTR_LFCPhase4Scene module docstring for the C1 rationale and
contract. Status: clean-break commit 12.11 (2026-05-12).
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


__all__ = ["OTR_LFCPhase5Voice"]


DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"


class OTR_LFCPhase5Voice:
    """LFC Phase 5 per-speaker voice drift as a standalone node."""

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script_json", "phase_5_verdict")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Passthrough from upstream. Phase 5 reads "
                        "the ledger via peek_ledger() internally."
                    ),
                }),
            },
            "optional": {
                "enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 5 per-speaker voice drift "
                        "detection + targeted rewrites. Default "
                        "OFF -- the stats heuristic is "
                        "conservative; most episodes flag zero "
                        "lines and the LLM call is skipped."
                    ),
                }),
                "model_id": ("STRING", {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "HF model ID for the batched Phase 5 LLM "
                        "call."
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
        script_json: str = "",
        enable: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        if not enable:
            return (script_json or "{}", "skipped")

        from . import _otr_model_loader as _OTRML
        from . import production_ledger as _PL
        from . import _otr_lfc_phase_5_voice_drift as _LFC_P5

        peek = getattr(_PL, "peek_ledger", None)
        led = (peek() if callable(peek) else _PL.get_ledger())
        if led is None:
            log.warning(
                "[OTR_LFCPhase5Voice] no writer ledger; skipping"
            )
            return (script_json or "{}", "no_ledger")

        cache_entry = _OTRML.load_llm(model_id=model_id or DEFAULT_MODEL_ID)
        generate_fn = _OTRML.make_generate_fn(cache_entry)
        try:
            rep = _LFC_P5.phase_5_voice_drift(
                generate_fn, led, enable=True,
            )
        finally:
            try:
                _OTRML.unload_llm()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LFCPhase5Voice] unload_llm raised: %s", exc,
                )

        try:
            new_json = json.dumps(
                led.data, indent=2, ensure_ascii=False,
            )
        except Exception:  # noqa: BLE001
            new_json = script_json or "{}"

        # G3 fix (commit 12.13): surface the explicit `failed` flag
        # before bucketing by edit count. Pre-fix a two_step
        # parse-exhaustion returned "completed_no_edits_after_flag",
        # masking the real failure mode.
        if rep.failed:
            verdict = "failed"
        elif rep.edits_applied > 0:
            verdict = "completed_with_edits"
        elif rep.flagged_lines:
            verdict = "completed_no_edits_after_flag"
        else:
            verdict = "completed_no_flagged_lines"
        return (new_json, verdict)
