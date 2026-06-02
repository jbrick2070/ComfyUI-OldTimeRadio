"""nodes/OTR_LedgerFreezeCascade.py — Ledger Freeze Cascade ComfyUI node.

Wires AFTER OTR_LedgerScriptWriter and BEFORE SceneSequencer. Touches
the production ledger only; emits no audio, no video, no LLM weights
beyond what the writer already loaded.

Output contract (7 slots):
    script_text, script_json, news_used, estimated_minutes, freeze_verdict,
    episode_seed, v2_ledger_json

R0a (2026-06-02): episode_seed + v2_ledger_json appended at output indices
5,6 -- never inserted, so outputs 0-4 (and the 13-consumer fan-out on out[1]
script_json) keep byte-identical raw-delegation behavior. episode_seed is
derived read-only from the frozen ledger (never stamped back), so out[1] stays
unchanged; v2_ledger_json carries the frozen ledger for OTR_CastLock (Wave 2a).

`freeze_verdict` literal set (S33 B2 trim 2026-05-15):

    frozen_clean
    frozen_with_warns
    frozen_with_doctor_edits
    too_many_edits
    needs_full_rerun

`cast_unrecoverable` and `post_audit_failed` retired in S33 B2 with
their respective rollback gates per the refined no-auditors rule.

Status: LFC v2.0-alpha (2026-05-12 clean-break).
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
                          (Phase 1 Auditor, Phase 2 Script Doctor).
                          S33 B3 (2026-05-15) retired Phase 9 per
                          the refined no-auditors rule. Phase
                          3/4/4.5/5/6 future LLM phases reuse the
                          same loader.

    Outputs (5 slots):
      script_text         Rebuilt from the post-freeze ledger.
      script_json         JSON snapshot of the post-freeze ledger.
      news_used           Passthrough from writer to SignalLostVideo.
      estimated_minutes   Passthrough INT.
      freeze_verdict      One of the FreezeVerdict literals.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used",
        "estimated_minutes", "freeze_verdict",
        # R0a: appended at indices 5,6 -- never inserted (outputs 0-4 frozen).
        "episode_seed", "v2_ledger_json",
    )

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
                # DELETED. Reviewer passes (Phase 1 / 2 / 9) consume
                # the writer's broadcast `technical_model` socket via
                # `technical_model` input below. Phase 3/4/4.5/5/6
                # toggles were all defaulted OFF and the surrounding
                # standalone LFC nodes go away in B4 -- the cascade
                # never invoked those phases in any shipped workflow.
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved technical_model id from the writer's "
                        "broadcast output (S30 B3). No local widget; "
                        "the cascade must be wired into the writer's "
                        "technical_model output socket to receive its "
                        "reviewer-LLM id. Validated at run-time via "
                        "_otr_model_inputs.require_model -- an unwired "
                        "socket raises MissingModelInputError loud."
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
                "vram_ceiling_gb": ("FLOAT", {
                    "default": 14.0,
                    "min": 4.0,
                    "max": 24.0,
                    "step": 0.5,
                    "tooltip": (
                        "VRAM ceiling (GB) stamped on meta; entry-time "
                        "check warns on over-ceiling. Per-phase "
                        "skipping is follow-up wiring once soak data "
                        "shows where the actual ceiling hits are. ADR "
                        "section 6.8 caps at 14.0 GB on the 5080 "
                        "Laptop (16 GB total, 0.5 GB margin under the "
                        "14.5 GB usable cap)."
                    ),
                }),
                # ---- Sprint 6 -- critic-to-render coupling --------
                # These four widgets steer which lines BatchHumoRender
                # renders downstream. The cascade computes the plan
                # from the post-reroll critic report + the widgets
                # and stamps `meta.render_plan`; HuMo reads + honours
                # it. All four default to behave-as-before so existing
                # workflows render unchanged unless an operator opts in.
                "render_selection": (
                    ("all", "dramatic_peaks_only"),
                    {
                        "default": "all",
                        "tooltip": (
                            "Sprint 6 -- render selection mode. 'all' "
                            "(default) keeps every character line; "
                            "'dramatic_peaks_only' reorders to the "
                            "critic's render_priority list so the most "
                            "dramatically loaded lines come first."
                        ),
                    },
                ),
                "render_max_n": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": (
                        "Sprint 6 -- cap on plan length. Default 6. "
                        "0 disables the cap (every selected line "
                        "renders). Applied after render_selection / "
                        "protagonist_only / manual_line_ids."
                    ),
                }),
                "protagonist_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Sprint 6 -- restrict the render plan to the "
                        "protagonist's lines (the character with the "
                        "most CHARACTER-role beats, ties broken by "
                        "cast-roster order). Default OFF."
                    ),
                }),
                "manual_line_ids": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Sprint 6 -- comma-separated explicit override. "
                        "When non-empty, supersedes render_selection / "
                        "flat_lines exclusion / arc_verdict gating. The "
                        "operator's hand wins -- ship exactly these "
                        "line_ids, in this order (capped by "
                        "render_max_n)."
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
        # S30 B3: technical_model arrives via input socket (no widget).
        # The 6 phase-toggle kwargs are deleted; the orchestrator's
        # function defaults still keep Phases 3/4/4.5/5/6 OFF.
        technical_model: str = "",
        enable_phase_7_audio_readiness: bool = True,
        enable_phase_8_video_readiness: bool = True,
        vram_ceiling_gb: float = 14.0,
        # Sprint 6 -- critic-to-render coupling widgets. Defaults match
        # the INPUT_TYPES defaults; the cascade stamps meta.render_plan
        # from these + the post-reroll critic report.
        render_selection: str = "all",
        render_max_n: int = 6,
        protagonist_only: bool = False,
        manual_line_ids: str = "",
    ):
        # Lazy imports to keep node-load cheap.
        from . import _otr_freeze_cascade as _LFC_ORCH
        from . import _otr_model_loader as _OTRML
        from . import _otr_model_inputs as _OTRMI
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

        # S30 B3: resolve the technical_model id via the shared
        # require_model helper. Fail loud (MissingModelInputError) if
        # the input socket is unwired -- the recovery is to connect
        # the writer's broadcast `technical_model` output.
        # LLM slot: technical -- reviewer Phase 1 (auditor) + Phase 2
        # (Script Doctor) consume this entry. Structured verdict-style
        # passes; routes to the technical slot per the S30 routing
        # table. S33 B3 (2026-05-15) retired Phase 9 (post-edit
        # auditor) per the refined no-auditors rule.
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )
        cache_entry = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )
        generate_fn = _OTRML.make_generate_fn(cache_entry)
        # LFC commit 12, ADR section 6.4: build the polish-specific
        # generate_fn off the same cache_entry so composer-tuned
        # sampling does not leak in. S28 cleanbreak: drop the
        # try/except None-fallback (producer-side legacy debris).
        # B3 keeps the call required; a factory failure surfaces as
        # a hard node error rather than silently degrading.
        polish_generate_fn = _OTRML.make_polish_generate_fn(cache_entry)

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
            # phase functions from _otr_lfc.py.
            disp = _LFC_ORCH.run_freeze_cascade(
                generate_fn,
                led,
                polish_generate_fn=polish_generate_fn,
                enable_phase_7_audio_readiness=enable_phase_7_audio_readiness,
                enable_phase_8_video_readiness=enable_phase_8_video_readiness,
                vram_ceiling_gb=float(vram_ceiling_gb),
                # Sprint 6 -- critic-to-render coupling.
                render_selection=str(render_selection or "all"),
                render_max_n=int(render_max_n or 0),
                protagonist_only=bool(protagonist_only),
                manual_line_ids=str(manual_line_ids or ""),
            )
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

            # Serialize + rebuild WHILE the model is still loaded.
            # Neither touches torch tensors (assemble_script_text_from_ledger
            # is pure dict/string work; json.dumps walks the meta tree)
            # so placement order is safe -- the model could already be
            # released here. We keep the order for cleanliness; the
            # finally-block unload is the actual VRAM-safe gate.
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
            # B14 (commit 12.5) + B1 (commit 12.12): unload Mistral-
            # Nemo before downstream visual nodes load. Wrapped in
            # best-effort try/except -- an unload failure logs at
            # WARNING + stamps meta.freeze_unload_ok=False so the
            # next visual node can branch on the stamp instead of
            # OOM-ing on top of a leaked cache. The cascade itself
            # still returns its verdict; the downstream visual
            # nodes decide what to do about a failed unload.
            try:
                _OTRML.unload_llm()
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

        # S34 B2 (2026-05-15): reserialize led.data so the
        # freeze_unload_ok stamp set in the finally block above is
        # visible to downstream JSON consumers. The earlier
        # serialization at L346 happened BEFORE the stamp; without
        # this reserialization, the comment at L374 claiming "the
        # next visual node can branch on the stamp" is false because
        # the JSON they receive doesn't contain it.
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
