"""nodes/OTR_LedgerFreezeCascade.py — Ledger Freeze Cascade ComfyUI node.

Wires AFTER OTR_LedgerScriptWriter and BEFORE SceneSequencer. Touches
the production ledger only; emits no audio, no video, no LLM weights
beyond what the writer already loaded.

Output contract (5 slots):
    script_text, script_json, news_used, estimated_minutes, freeze_verdict

`freeze_verdict` literal set:

    frozen_clean
    frozen_with_warns
    frozen_with_doctor_edits
    cast_unrecoverable
    too_many_edits
    needs_full_rerun
    post_audit_failed

Status: LFC v2.0-alpha (2026-05-12 clean-break).
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerFreezeCascade"]


DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"


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

    Outputs (5 slots):
      script_text         Rebuilt from the post-freeze ledger.
      script_json         JSON snapshot of the post-freeze ledger.
      news_used           Passthrough from writer to SignalLostVideo.
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
                "model_id": ("STRING", {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "HF model ID for the reviewer LLM passes "
                        "(Phase 1 / 2 / 9). Default Mistral-Nemo "
                        "matches the writer's default."
                    ),
                }),
                "enable_phase_3_polish": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 3 -- per-line polish. Re-runs the "
                        "composer's polish_line over any line that "
                        "still trips needs_polish AFTER the reviewer. "
                        "Default OFF until soak validates the inter-"
                        "action with the composer's inline polish pass."
                    ),
                }),
                "polish_announcer_beats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 3 -- announcer-beat handling. "
                        "When False (default), announcer beats are "
                        "skipped (they are by-design narration). "
                        "When True, the announcer-aware polish prompt "
                        "fires (per ADR section 6.1)."
                    ),
                }),
                "enable_phase_4_scene_coherence": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 4 -- per-scene coherence + audio-"
                        "first directives. Iterates scenes (music_"
                        "inter dividers) and runs a two-step LLM "
                        "call per scene; edits cap = min(3, scene_"
                        "lines // 2). Also caches meta.scene_synopses "
                        "for Phase 6 (episode arc). Default OFF until "
                        "soak validates the prompt + scene-boundary "
                        "logic."
                    ),
                }),
                "enable_phase_4_5_smart_suggestion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 4.5 -- deterministic SFX / music "
                        "synthesis. Scans dialogue for verb patterns "
                        "(start the car -> car_engine_start) and "
                        "appends auto_generated=True beats. Default "
                        "OFF for v2.0-alpha per ADR section 6.17."
                    ),
                }),
                "enable_phase_5_voice_drift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 5 -- per-speaker voice drift "
                        "detection + targeted rewrites. Stats helpers "
                        "(mean_line_length + vocab diversity) flag "
                        "lines that diverge from a character's "
                        "established voice; ONE batched LLM call "
                        "rewrites the flagged subset. Default OFF "
                        "until soak data tunes the 40 percent / 60 "
                        "percent drift thresholds."
                    ),
                }),
                "enable_phase_6_episode_arc": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "LFC Phase 6 -- episode arc audit with "
                        "Editor-note scaffold. Reads cached "
                        "meta.scene_synopses from Phase 4 (or "
                        "falls back to outline intents) and runs "
                        "ONE two-step LLM call that emits Editor "
                        "notes scoped per-scene / speaker / "
                        "episode. Edit cap = min(8, max(3, "
                        "voiced_beats // 3)) -- shared pool with "
                        "the reviewer's Doctor. Default OFF."
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
        enable_phase_3_polish: bool = False,
        polish_announcer_beats: bool = False,
        enable_phase_4_scene_coherence: bool = False,
        enable_phase_4_5_smart_suggestion: bool = False,
        enable_phase_5_voice_drift: bool = False,
        enable_phase_6_episode_arc: bool = False,
        enable_phase_7_audio_readiness: bool = True,
        enable_phase_8_video_readiness: bool = True,
        vram_ceiling_gb: float = 14.0,
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
        # LFC commit 12, ADR section 6.4: build the polish-specific
        # generate_fn off the same cache_entry so composer-tuned
        # sampling does not leak in. Best-effort: if the loader
        # doesn't yet expose make_polish_generate_fn the cascade
        # falls back to generate_fn.
        try:
            polish_generate_fn = _OTRML.make_polish_generate_fn(cache_entry)
        except Exception as exc:  # noqa: BLE001
            # B4 fix (commit 12.1): bumped from debug -> warning so a
            # real make_polish_generate_fn regression surfaces in the
            # boot log instead of silently falling back to the
            # closure-leak path.
            log.warning(
                "[OTR_LedgerFreezeCascade] make_polish_generate_fn "
                "unavailable (%s); falling back to generate_fn", exc,
            )
            polish_generate_fn = None

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
            disp = _LFC_ORCH.run_freeze_cascade(
                generate_fn,
                led,
                polish_generate_fn=polish_generate_fn,
                enable_phase_3_polish=enable_phase_3_polish,
                polish_announcer_beats=polish_announcer_beats,
                enable_phase_4_scene_coherence=enable_phase_4_scene_coherence,
                enable_phase_4_5_smart_suggestion=enable_phase_4_5_smart_suggestion,
                enable_phase_5_voice_drift=enable_phase_5_voice_drift,
                enable_phase_6_episode_arc=enable_phase_6_episode_arc,
                enable_phase_7_audio_readiness=enable_phase_7_audio_readiness,
                enable_phase_8_video_readiness=enable_phase_8_video_readiness,
                vram_ceiling_gb=float(vram_ceiling_gb),
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
        )
