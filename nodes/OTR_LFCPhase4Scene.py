"""nodes/OTR_LFCPhase4Scene.py — standalone LFC Phase 4 ComfyUI node.

Per the C1 design call in the clean-break go-forward plan
(docs/2026-05-12-lfc-clean-break.md), the three heavy LLM phases
(4, 5, 6) are also exposed as standalone ComfyUI nodes so
operators can:

  * Per-phase skip / rerun from the canvas without rerunning the
    full cascade upstream.
  * Per-phase failure visibility (red node on the canvas instead
    of a buried log warning inside the cascade).
  * Per-phase telemetry stamped distinctly on the ledger meta.

The main OTR_LedgerFreezeCascade still owns the full chain. The
standalone node is an ADDITIONAL entry point -- it reads the
current ledger handle via production_ledger.peek_ledger(), runs
Phase 4 against it, and returns a passthrough plus a per-phase
verdict. Default OFF on the `enable` widget so dropping the node
on the canvas is a no-op until the operator opts in.

Status: clean-break commit 12.11 (2026-05-12).
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


__all__ = ["OTR_LFCPhase4Scene"]


DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"


class OTR_LFCPhase4Scene:
    """LFC Phase 4 per-scene coherence as a standalone ComfyUI node."""

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script_json", "phase_4_verdict")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Passthrough from upstream. Phase 4 reads the "
                        "current ledger handle via "
                        "production_ledger.peek_ledger() internally; "
                        "this socket exists so the graph wires this "
                        "node in line."
                    ),
                }),
            },
            "optional": {
                "enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 4 per-scene coherence + audio-"
                        "first directives. Default OFF -- dropping "
                        "this node on the canvas is a no-op until "
                        "the operator opts in."
                    ),
                }),
                "model_id": ("STRING", {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "HF model ID for the Phase 4 two-step gen "
                        "call. Default Mistral-Nemo."
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
        from . import _otr_lfc_phase_4_scene_coherence as _LFC_P4

        peek = getattr(_PL, "peek_ledger", None)
        led = (peek() if callable(peek) else _PL.get_ledger())
        if led is None:
            log.warning(
                "[OTR_LFCPhase4Scene] no writer ledger; skipping"
            )
            return (script_json or "{}", "no_ledger")

        cache_entry = _OTRML.load_llm(model_id=model_id or DEFAULT_MODEL_ID)
        generate_fn = _OTRML.make_generate_fn(cache_entry)
        try:
            rep = _LFC_P4.phase_4_scene_coherence(
                generate_fn, led, enable=True,
            )
        finally:
            try:
                _OTRML.unload_llm()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LFCPhase4Scene] unload_llm raised: %s", exc,
                )

        try:
            new_json = json.dumps(
                led.data, indent=2, ensure_ascii=False,
            )
        except Exception:  # noqa: BLE001
            new_json = script_json or "{}"

        verdict = (
            "completed_with_edits" if rep.total_edits_applied > 0
            else "completed_no_edits"
        )
        if rep.scenes_failed:
            verdict = "completed_with_failures"
        return (new_json, verdict)
