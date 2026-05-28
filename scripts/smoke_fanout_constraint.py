"""scripts/smoke_fanout_constraint.py -- Sprint 2.3 (2026-05-28).

End-to-end smoke test for the Sprint 4 fan-out + Sprint 5
constraint editor chain. Validates the wiring without touching a
real LLM -- every generate_fn is replaced with a deterministic
JSON-emitter. A subagent (or Jeffrey) runs this script to confirm
all five wires are intact end-to-end after any code change:

  Stage 1 fan-out -> 3 candidate Stage1Plans
  Sprint 2 validators -> structural pass/fail per candidate
  Sprint 4 selector -> winner pick + audit
  Sprint 5 constraint editor -> verdict + repair_note
  Sprint 5.2 derive_repair_prompt -> Writer revision block

Exit code 0 = all wires intact. Non-zero = a wire broke; the
script prints which step failed.

Usage:
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe \\
        scripts\\smoke_fanout_constraint.py
"""
from __future__ import annotations

import json
import sys
import traceback

# Make the package importable when run from the repo root.
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ds() -> dict:
    return {
        "dramatic_question": "Will Maeve sign the confession before the audit closes?",
        "character_a_wants": "hide the falsified ledger before the audit",
        "character_b_wants": "get Maeve's signature on the record tonight",
        "costly_choice_beat": "d003",
        "ending_change": "the falsified ledger becomes public record",
    }


def _plan_dict(*, premise_suffix: str = "") -> dict:
    """Return a structurally-valid Stage1Plan dict for the smoke test.
    Each diversity knob varies the premise so the three candidates
    are visibly distinct in the smoke output."""
    return {
        "premise": (
            f"A short test premise long enough to pass schema."
            f"{premise_suffix}"
        ),
        "arc": {
            "setup": "setup statement long enough to validate.",
            "complication": "complication statement long enough to validate.",
            "resolution": "resolution statement long enough to validate.",
        },
        "cast": [
            {
                "name": "ALICE",
                "gender": "female",
                "pronouns": "she/her",
                "voice_id": "v2/en_speaker_3",
                "persona": "weary forensic engineer with dry humor.",
                "arc_role": "reluctant insider",
            },
            {
                "name": "BOB",
                "gender": "male",
                "pronouns": "he/him",
                "voice_id": "v2/en_speaker_5",
                "persona": "ambitious grant officer evasive about funding.",
                "arc_role": "pressure source",
            },
        ],
        "beats": [
            {"beat_id": "b001", "speaker": "ANNOUNCER",
             "intent": "open the episode",
             "length_target_words": 15,
             "emotional_register": "welcoming",
             "state_before": "audit has not begun",
             "state_after": "audit window is open",
             "tension": 2},
            {"beat_id": "b002", "speaker": "ALICE",
             "intent": "discover the missing ledger",
             "length_target_words": 20,
             "emotional_register": "tight unease",
             "state_before": "ledger somewhere",
             "state_after": "ledger missing",
             "tension": 3},
            {"beat_id": "b003", "speaker": "ALICE",
             "intent": "make the costly choice to sign",
             "length_target_words": 30,
             "emotional_register": "grim resolve",
             "state_before": "alice undecided",
             "state_after": "alice has signed",
             "tension": 5},
            {"beat_id": "b004", "speaker": "ANNOUNCER",
             "intent": "close the episode",
             "length_target_words": 15,
             "emotional_register": "quiet aftermath",
             "state_before": "signature is real",
             "state_after": "record is public",
             "tension": 3},
        ],
        "running_facts": ["the audit closes at midnight"],
        "dramatic_state": _ds(),
    }


def _fail(step: str, reason: str) -> None:
    print(f"[SMOKE FAIL] {step}: {reason}")
    raise SystemExit(1)


def _step(title: str) -> None:
    print(f"\n[SMOKE] {title}")


def main() -> int:
    print("=" * 60)
    print("Sprint 2.3 smoke -- fan-out + selector + constraint")
    print("=" * 60)

    # --- 1. Sprint 4.3 fan-out + Sprint 4 selector ---------------
    _step("1/5  fan_out_and_select_stage1_plan (mocked LLM)")
    try:
        from nodes._otr_stage1_fanout import (
            ALL_DIVERSITY_KNOBS,
            DIVERSITY_KNOB_PROMPTS,
            fan_out_and_select_stage1_plan,
        )
        from nodes._otr_stage1_plan import parse_and_validate_plan

        knob_seq = list(ALL_DIVERSITY_KNOBS)
        knob_call_idx = [0]
        knob_premise = {
            knob_seq[0]: " (moral_dilemma flavor)",
            knob_seq[1]: " (bureaucratic_absurd flavor)",
            knob_seq[2]: " (intimate_personal_cost flavor)",
        }

        def fake_generate_fn(messages, *, temperature, max_new_tokens):
            sys_msg = messages[0]["content"]
            # Detect which knob fired by the prefix.
            for knob, prefix in DIVERSITY_KNOB_PROMPTS.items():
                if sys_msg.startswith(prefix) or sys_msg.startswith(prefix.strip()):
                    knob_call_idx[0] += 1
                    suffix = knob_premise.get(knob, "")
                    return json.dumps(_plan_dict(premise_suffix=suffix))
            return json.dumps(_plan_dict())

        def parse_fn(raw: str):
            return parse_and_validate_plan(json.loads(raw))

        res = fan_out_and_select_stage1_plan(
            generate_fn=fake_generate_fn,
            parse_fn=parse_fn,
            base_system_prompt="BASE_STAGE1_SYSTEM",
            user_prompt="USER",
        )
        if not res.ok:
            _fail("1/5", f"fan-out returned ok=False: {res.no_valid_error_msg}")
        if len(res.fan_out_results) != 3:
            _fail("1/5", f"expected 3 results, got {len(res.fan_out_results)}")
        if not all(r.ok for r in res.fan_out_results):
            _fail("1/5", f"per-knob ok mismatch: {[r.ok for r in res.fan_out_results]}")
        if knob_call_idx[0] != 3:
            _fail("1/5", f"expected 3 LLM calls, got {knob_call_idx[0]}")
        winner_idx = res.selector_audit.winner_index
        print(f"        winner_index={winner_idx} "
              f"score={res.selector_audit.scores_per_candidate[winner_idx].total}/5")
        for i, r in enumerate(res.fan_out_results):
            print(f"        candidate[{i}] knob={r.knob} ok={r.ok}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("1/5", "exception during fan-out + selector")

    # --- 2. Sprint 5 constraint editor (Python check) ------------
    _step("2/5  check_constraints on the winner")
    try:
        from nodes._otr_editor_constraints import (
            EditorConstraint,
            check_constraints,
        )
        verdict = check_constraints(res.winner_plan)
        if not verdict.pass_decision:
            _fail("2/5",
                  f"clean fixture failed constraint check: "
                  f"{verdict.failing_constraints}")
        print(f"        pass_decision=True (clean plan)")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("2/5", "exception during check_constraints")

    # --- 3. Sprint 5.3 LLM constraint editor (mocked) ------------
    _step("3/5  run_constraint_editor with mock LLM")
    try:
        from nodes._otr_director_brief import DirectorBrief
        from nodes._otr_editor_constraints import run_constraint_editor

        def fake_editor_fn(messages, *, temperature, max_new_tokens):
            return json.dumps({
                "pass_decision": False,
                "failing_constraints": [EditorConstraint.WRONG_SPEAKER],
                "repair_note": (
                    "beat b003 is attributed to MAEVE but cast is "
                    "ALICE/BOB; rename MAEVE -> ALICE."
                ),
            })

        brief = DirectorBrief(
            news_premise="An audit team races to verify a falsified ledger.",
            dramatic_question="Will Maeve sign in time?",
            opposed_desires="hide the ledger vs get the signature",
            arc_shape="discover -> press -> sign or refuse",
            audience_feel="tight forensic tension",
            forbidden_drift=[],
            raw_brief="raw",
        )
        editor_verdict = run_constraint_editor(
            brief,
            "ALICE: open\nMAEVE: I won't sign.",
            generate_fn=fake_editor_fn,
            cast_names=["ALICE", "BOB"],
            cycle=0,
        )
        if editor_verdict.pass_decision:
            _fail("3/5", "editor should have flagged WRONG_SPEAKER")
        if EditorConstraint.WRONG_SPEAKER not in editor_verdict.failing_constraints:
            _fail("3/5",
                  f"expected WRONG_SPEAKER in failing_constraints, got "
                  f"{editor_verdict.failing_constraints}")
        print(f"        pass_decision=False "
              f"failing={editor_verdict.failing_constraints}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("3/5", "exception during run_constraint_editor")

    # --- 4. Sprint 5.2 derive_repair_prompt ----------------------
    _step("4/5  derive_repair_prompt from the failing verdict")
    try:
        from nodes._otr_editor_constraints import derive_repair_prompt
        repair_block = derive_repair_prompt(editor_verdict)
        if "REVISE" not in repair_block:
            _fail("4/5", "repair block missing REVISE header")
        if EditorConstraint.WRONG_SPEAKER not in repair_block:
            _fail("4/5", "repair block missing WRONG_SPEAKER code")
        if "MAEVE" not in repair_block:
            _fail("4/5", "repair_note text did not land in the block")
        print(f"        block length={len(repair_block)} chars (REVISE + per-code + note)")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("4/5", "exception during derive_repair_prompt")

    # --- 5. Sprint 5.5 Writer prompt rendering -------------------
    _step("5/5  build_writer_user_prompt with constraint-editor splice")
    try:
        from nodes._otr_editor_pass import EditorVerdict
        from nodes._otr_story_room import build_writer_user_prompt

        # Mirror the adapter shape _call_constraint_editor produces.
        shaped = EditorVerdict(
            pass_decision=False,
            failing_axes=[EditorConstraint.WRONG_SPEAKER],
            per_axis_notes={},
            overall_note="MAEVE not in cast",
            cycle=0,
        )
        prompt = build_writer_user_prompt(
            brief=brief,
            cast=[{"name": "ALICE"}, {"name": "BOB"}],
            news_seed="seed",
            last_editor_verdict=shaped,
            last_draft="ALICE: open\nMAEVE: I won't sign.",
            cycle=1,
            use_constraint_editor=True,
        )
        if "REVISE" not in prompt:
            _fail("5/5", "constraint-editor splice missing REVISE block in prompt")
        if "Do NOT rewrite passing passages" not in prompt:
            _fail("5/5", "Sprint 5.5 revision directive missing")
        if "every failing axis" in prompt:
            _fail("5/5",
                  "legacy directive present under use_constraint_editor=True")
        print(f"        constraint splice OK; prompt length={len(prompt)} chars")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("5/5", "exception during build_writer_user_prompt")

    print()
    print("=" * 60)
    print("[SMOKE OK] all 5 wires intact end-to-end")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
