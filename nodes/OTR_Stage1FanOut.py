"""nodes/OTR_Stage1FanOut.py -- Sprint 4.8 (2026-05-28).

ComfyUI wrapper around `_otr_stage1_fanout.fan_out_and_select_stage1
_plan`. Mirrors OTR_BeatSelector's shape so the two can be chained
in the canonical workflow:

    upstream news_seed STRING -> OTR_Stage1FanOut runs 3 LLM calls
      with diversity knobs and emits:
        candidate_a_json (moral_dilemma)
        candidate_b_json (bureaucratic_absurd)
        candidate_c_json (intimate_personal_cost)
        winning_plan_json (the Sprint 4 mechanical selector's pick)
        fanout_audit (winner_knob, per-knob ok, selector scores)
    -> OTR_BeatSelector receives the 3 candidate strings and emits
      its own winner (operator can use either node's winner).

PD6: the technical_model id arrives via the writer's broadcast
output (`technical_model` forceInput socket). No `model_id` widget.

PD3: this node ships DETACHED in the workflow JSON. Operator wires
it manually -- once the diagnostic Sprint 4.5 audits confirm the
diversity knobs produce structurally distinct candidates.

LLM slot: creative. Stage 1 outlines are a creative pass per
project rule 6.

UTF-8 no BOM.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger("OTR")


__all__ = ["OTR_Stage1FanOut"]


class OTR_Stage1FanOut:
    """Sprint 4 fan-out + selector ComfyUI node wrapper.

    Inputs (required):
        news_seed         STRING (multiline) -- the news premise
                          (RSS body / custom premise) the Stage 1
                          plan is built against.
        num_characters    INT (1..6) -- cast size.
        target_words      INT (50..4000) -- episode word budget.

    Inputs (optional):
        episode_title_hint STRING -- optional working title.
        technical_model    STRING (forceInput) -- writer's broadcast
                           output. PD6: no widget; unwired socket
                           triggers MissingModelInputError.

    Outputs:
        candidate_a_json   STRING -- moral_dilemma candidate plan.
        candidate_b_json   STRING -- bureaucratic_absurd candidate.
        candidate_c_json   STRING -- intimate_personal_cost candidate.
        winning_plan_json  STRING -- selector winner (== one of the
                           three on success; empty on all-fail).
        fanout_audit_json  STRING -- {ok, winner_knob, per_knob_ok,
                           selector_audit, error} for operator
                           inspection.
    """

    CATEGORY = "OldTimeRadio/v2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "candidate_a_json", "candidate_b_json", "candidate_c_json",
        "winning_plan_json", "fanout_audit_json",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "news_seed": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "News premise (RSS body or custom prose) "
                        "for Stage 1 generation. Same content as "
                        "OTR_LedgerScriptWriter sees."
                    ),
                }),
                "num_characters": ("INT", {
                    "default": 2,
                    "min": 1, "max": 6, "step": 1,
                    "tooltip": "Cast size (1..6).",
                }),
                "target_words": ("INT", {
                    "default": 350,
                    "min": 50, "max": 4000, "step": 50,
                    "tooltip": "Total episode word budget.",
                }),
            },
            "optional": {
                "episode_title_hint": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Optional working title hint. The Stage 1 "
                        "LLM may keep, replace, or ignore it."
                    ),
                }),
                "technical_model": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Resolved technical_model id from the "
                        "writer's broadcast output (PD6). No local "
                        "widget; unwired socket raises "
                        "MissingModelInputError when the node runs."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The fan-out runs three LLM calls; non-deterministic at the
        # sampler level by design. Always-changed so the operator
        # sees fresh output each queue.
        return time.time()

    def _empty_audit(self, reason: str) -> dict:
        return {
            "ok": False,
            "winner_knob": None,
            "per_knob_ok": [],
            "selector_audit": None,
            "error": reason,
        }

    def run(
        self,
        news_seed: str = "",
        num_characters: int = 2,
        target_words: int = 350,
        episode_title_hint: str = "",
        technical_model: str = "",
    ):
        seed = (news_seed or "").strip()
        if not seed:
            audit = self._empty_audit("empty news_seed -- nothing to plan")
            empty = ""
            return (empty, empty, empty, empty,
                    json.dumps(audit, ensure_ascii=False))

        # Lazy imports keep node-load cheap.
        from . import _otr_constrained_generate as _OTRCG
        from . import _otr_model_inputs as _OTRMI
        from . import _otr_model_loader as _OTRML
        from . import _otr_stage1_call as _OTRS1
        from . import _otr_stage1_fanout as _OTRFAN
        from . import _otr_stage1_plan as _OTRS1P

        # PD6: technical_model must be wired.
        resolved_technical_id = _OTRMI.require_model(
            technical_model, slot="technical",
        )

        # LLM slot: creative
        # Reason: outline generation is a creative pass per project
        # rule 6. The constrained generator binds to Stage1Plan so
        # the LLM emits validated JSON.
        cache_entry = _OTRML.request_slot(
            "technical", resolved_technical_id,
        )
        generate_fn = _OTRCG.make_constrained_generate_fn(
            cache_entry, _OTRS1P.Stage1Plan,
        )

        def _parse_fn(raw: str):
            return _OTRS1P.parse_and_validate_plan(json.loads(raw))

        user_prompt = _OTRS1.build_stage1_user_prompt(
            news_seed=seed,
            num_characters=int(num_characters),
            target_words=int(target_words),
            episode_title_hint=(episode_title_hint or "").strip(),
        )

        res = _OTRFAN.fan_out_and_select_stage1_plan(
            generate_fn=generate_fn,
            parse_fn=_parse_fn,
            base_system_prompt=_OTRS1._STAGE1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Serialize candidates per-knob (in declaration order).
        def _ser(plan: Any) -> str:
            if plan is None:
                return ""
            if hasattr(plan, "model_dump"):
                return json.dumps(
                    plan.model_dump(), ensure_ascii=False, indent=2,
                )
            return json.dumps(plan, ensure_ascii=False, indent=2)

        cand_a = _ser(res.fan_out_results[0].plan) if len(res.fan_out_results) >= 1 else ""
        cand_b = _ser(res.fan_out_results[1].plan) if len(res.fan_out_results) >= 2 else ""
        cand_c = _ser(res.fan_out_results[2].plan) if len(res.fan_out_results) >= 3 else ""

        winner_json = _ser(res.winner_plan) if res.winner_plan is not None else ""

        audit = {
            "ok": res.ok,
            "winner_knob": (
                res.fan_out_results[res.selector_audit.winner_index].knob
                if res.ok and res.selector_audit is not None
                else None
            ),
            "per_knob_ok": [
                {"knob": r.knob, "ok": r.ok, "error": (r.error or "")[:200]}
                for r in res.fan_out_results
            ],
            "selector_audit": (
                res.selector_audit.to_dict()
                if res.selector_audit is not None
                else None
            ),
            "error": res.no_valid_error_msg,
        }
        log.info(
            "[OTR_Stage1FanOut] ok=%s winner_knob=%s",
            res.ok, audit["winner_knob"],
        )
        return (cand_a, cand_b, cand_c, winner_json,
                json.dumps(audit, ensure_ascii=False))
