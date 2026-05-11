"""nodes/OTR_LedgerScriptReviewer.py — Phase 3 ComfyUI node.

Thin wrapper around `_otr_ledger_reviewer.review_ledger`. Loads the
LLM once (via _otr_model_loader.load_llm + make_generate_fn) and
runs the three-pass cast-gated reviewer on the current production
ledger.

Wires into the workflow AFTER OTR_LedgerScriptWriter and BEFORE
SceneSequencer / batch_bark_generator. Touches only the ledger;
emits no audio, no video, no LLM weights beyond what the writer
already loaded.

Output contract matches OTR_LedgerScriptWriter so this node is a
drop-in midstream replacement: downstream consumers that read
script_text + script_json + news_used + estimated_minutes from the
writer slot can read the post-review state from the reviewer slot
without re-wiring. (Wiring-review #2.)

Status: Phase 3b of v2.0 sprint (2026-05-11). New node.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerScriptReviewer"]


DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"


class OTR_LedgerScriptReviewer:
    """Three-pass cast-gated reviewer for the production ledger.

    Inputs:
      script_text         Forwarded from OTR_LedgerScriptWriter so the
                          graph wires this node in line. Returned
                          unchanged as the first output slot.
      script_json         Forwarded JSON snapshot from the writer
                          (slot index 1). Re-serialized from the
                          post-review ledger in the output so
                          downstream consumers see the updated state.
      news_used           Passthrough of the writer's news_used slot
                          for graph continuity.
      estimated_minutes   Passthrough of the writer's est_minutes
                          INT slot.
      model_id            HF model ID to use for the three reviewer
                          LLM calls.

    Outputs (matches writer's RETURN_TYPES — wiring-review #2):
      script_text         Passthrough of the input.
      script_json         JSON of the POST-REVIEW ledger state (after
                          Pass 3 commits / restores). Downstream
                          nodes (SceneSequencer, TTS) consume this
                          instead of the writer's pre-review JSON.
      reviewer_verdict    One of six ReviewerVerdict literals:
                            clean_no_edits / improved /
                            cast_unrecoverable / too_many_edits /
                            needs_full_rerun / post_audit_failed
      estimated_minutes   INT passthrough from the writer.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    # Wiring-review #2 (2026-05-11): the first FOUR output slots
    # match OTR_LedgerScriptWriter's RETURN_TYPES exactly so any
    # workflow that consumed (script_text, script_json, news_used,
    # estimated_minutes) from the writer can re-route the same
    # wires through the reviewer without re-mapping slots. The 5th
    # slot is the new reviewer_verdict; downstream graphs that
    # don't care about it just ignore the socket.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used",
        "estimated_minutes", "reviewer_verdict",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_text": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Passthrough from OTR_LedgerScriptWriter. The "
                        "reviewer reads the production ledger directly "
                        "via peek_ledger(); this socket exists so the "
                        "graph wires the writer and reviewer in line."
                    ),
                }),
            },
            "optional": {
                "script_json": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Forwarded from the writer's script_json slot. "
                        "The reviewer re-serializes this from the "
                        "post-review ledger so downstream consumers "
                        "see the updated state."
                    ),
                }),
                "news_used": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Passthrough of the writer's news_used slot. "
                        "Not used by the reviewer; carried through so "
                        "the reviewer is a drop-in midstream replacement."
                    ),
                }),
                "estimated_minutes": ("INT", {
                    "default": 0, "min": 0, "max": 999, "step": 1,
                    "tooltip": (
                        "Passthrough of the writer's estimated_minutes "
                        "INT slot. Not modified by the reviewer."
                    ),
                }),
                "model_id": ("STRING", {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "HF model ID to use for the three reviewer LLM "
                        "calls (Pass 1 Auditor, Pass 2 Script Doctor, "
                        "Pass 3 Auditor). Default Mistral-Nemo matches "
                        "the writer's default."
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
        # Lazy imports to keep node module load cheap (no torch / no
        # pydantic at registration time; the writer has loaded those
        # already by the time this node fires).
        from . import _otr_ledger_reviewer as _OTRLR  # type: ignore
        from . import _otr_model_loader as _OTRML  # type: ignore
        from . import production_ledger as _PL  # type: ignore

        # Wiring-review #3 (2026-05-11): use has_current_ledger()
        # to fail loud instead of get_ledger()'s auto-create. The
        # reviewer must never operate on an empty placeholder.
        has_current = getattr(_PL, "has_current_ledger", None)
        peek = getattr(_PL, "peek_ledger", None)
        if callable(has_current) and not has_current():
            log.warning(
                "[OTR_LedgerScriptReviewer] no writer-produced ledger "
                "in this process; returning needs_full_rerun without "
                "running LLM calls."
            )
            return (
                script_text or "",
                script_json or "{}",
                news_used or "",
                int(estimated_minutes or 0),
                "needs_full_rerun",
            )
        led = (peek() if callable(peek) else _PL.get_ledger())
        if led is None:
            log.warning(
                "[OTR_LedgerScriptReviewer] ledger handle is None; "
                "returning needs_full_rerun."
            )
            return (
                script_text or "",
                script_json or "{}",
                news_used or "",
                int(estimated_minutes or 0),
                "needs_full_rerun",
            )

        # Load (or reuse cached) LLM via the writer's loader path.
        # The writer should have already warmed Mistral-Nemo for us;
        # load_llm short-circuits to the cache_entry when the same
        # model_id is requested.
        cache_entry = _OTRML.load_llm(
            model_id=model_id or DEFAULT_MODEL_ID,
        )
        generate_fn = _OTRML.make_generate_fn(cache_entry)

        log.info(
            "[OTR_LedgerScriptReviewer] running 3-pass reviewer on "
            "ledger %s (%d lines)",
            led.episode_id,
            len(led.data.get("lines", []) or []),
        )
        disp = _OTRLR.review_ledger(generate_fn, led)
        log.info(
            "[OTR_LedgerScriptReviewer] verdict=%s "
            "(pre=%d violations, %d repairs / doctor=%d proposed, "
            "%d applied / post=%d violations / phantom_skip=%d)",
            disp.verdict,
            disp.pre_audit_violations, disp.pre_audit_repairs_applied,
            disp.doctor_edits_proposed, disp.doctor_edits_applied,
            disp.post_audit_violations, disp.phantom_skip_count,
        )

        # Wiring-review #2 + #5 (2026-05-11): build the JSON snapshot
        # AFTER review_ledger.led.save() has assigned merged-with-disk
        # state back to led.data (Patch 5). The resulting JSON is
        # byte-aligned with disk and reflects whichever ledger
        # (original or candidate) ended up committed.
        try:
            updated_script_json = json.dumps(
                led.data, indent=2, ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[OTR_LedgerScriptReviewer] failed to serialize "
                "post-review ledger to JSON (%s); falling back to "
                "incoming script_json.", exc,
            )
            updated_script_json = script_json or "{}"

        return (
            script_text or "",
            updated_script_json,
            news_used or "",
            int(estimated_minutes or 0),
            disp.verdict,
        )
