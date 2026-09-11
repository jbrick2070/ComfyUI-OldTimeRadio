<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Verdict

The diagnosis (frame ownership, not name collision) is correct and well evidenced. The **remedy is under-powered relative to its own risk analysis**, contains at least one internal prompt contradiction, and its ban list forecloses the only mechanisms that could make the outcome observable. Several concrete defects in the cited code are not in scope but should be.

---

### 1. The decisive defect is in P0's payload, and the plan treats it as secondary

The failing treatment is a faithful rendering of its input. `_pass_treatment` (line 510) dumps the whole interpretation verbatim into the user turn, and that interpretation says:

```
{"name":"Announcer","speaking":true,"required":true}, cast_plan.planned=3,
reason: "user explicitly requested an 'announcer' ... three speaking roles"
```

Meanwhile the treatment seam already says, in plain words, *"The cast array contains the story's speaking characters, excluding ANNOUNCER… ANNOUNCER is reserved for the frame."* **That instruction exists today and lost, twice, to a required/speaking row in the interpretation.** The plan's headline lever ("clarify existing phase ownership in P0/P1 pack instructions") is therefore partly a re-statement of an instruction with a demonstrated 0/2 record against a contradicting data payload. Strengthening prose on the side that already lost, while leaving the winning payload intact, is the weakest available intervention.

**Omission:** the plan never states *how* P0 should represent the announcer. The options are materially different and R2 cannot "nail exact text" without this decision:
- drop the announcer from `named_cast` entirely and record it under `assumptions`/`conflicts`;
- keep it with `speaking:false, required:false`;
- keep it and require `cast_plan.planned` to exclude it (here: planned should have been 2, and `reason` should have cited frame ownership).

The plan also says "No schema change expected," which may leave no field capable of expressing "named by the listener, owned by the frame" except free prose in `notes`. That is an architectural choice being made by omission. Note too that `conflicts` is empty in the actual failure, though this is exactly a conflict the seam's own rules ("If they demand something the form cannot do, record it in conflicts") should have caught — an existing-owner gap the plan does not claim.

### 2. The repair prompt will contain a direct self-contradiction

`_full_artifact_repair` (line 420) appends, unconditionally:

> "Preserve unaffected story events, relationships and **ending**"

Root's intended repair action is: *"make globalending the characters' realized conclusion."* So the second call would simultaneously be told to preserve the ending and to rewrite the ending. Compounding it, the treatment-specific instruction passed at line 510 is:

> "Reorganize the treatment into exactly %d acts… change the act grouping to fit."

which names the **wrong defect entirely** for this failure — acts were already correct (1/1). The model received: "reorganize into 1 act, preserve the ending, and by the way ANNOUNCER is reserved; give characters distinct names." That instruction set does not describe the actual required edit at all. The plan gestures at this ("Clarify generic treatment repair to correct its actual defect without unnecessarily regrouping already-correct acts") but does not identify the preserve-the-ending clause as an obstacle, and does not say whether the instruction becomes defect-conditional or whether the generic wrapper is left alone (it is shared by interpret/act/frame — changing it has blast radius the plan does not scope).

### 3. The fix makes the *worst* outcome more likely to pass validation

Root correctly argues (Actual ownership; Grounded diagnosis) that deleting only the cast row leaves an ANNOUNCER-authored `ending` and ANNOUNCER turns, and that `_pass_act` (line 588) then promotes that ending to the binding final-act target that "supersedes this act's planned ending_state." But the plan's only enforcement remains the **cast-only** check in `_make_treatment_validator` (line 494), while explicitly banning any check on `ending`/`turns` ("no broad keyword gate over prose").

Consequence: after telling the model precisely which field is inspected, the most probable partial compliance — drop the cast row, keep the frame ending — now **passes** and proceeds to P2 with exactly the misdirection the plan says must be prevented. That is a net regression in failure mode, not an improvement.

The stated constraint also conflates two different things. Checking prose for semantics is indeed out of this validator's remit. Checking whether the reserved identifier `ANNOUNCER_NAME` — a constant this module already owns and already rejects in `cast` — appears in `treatment.ending` or `acts[].turns` is neither broad nor fuzzy nor semantic; it is the same reserved-name condition applied to the same owner's other fields. Banning it is unjustified by the plan's own reasoning, and without it the plan has no observable success criterion at all.

### 4. Unverified downstream crash path

`_assemble` (line 769) builds `char_id_by_name` with `if r["name"] != ANNOUNCER_NAME`, then indexes it unguarded for every act line speaker. If a treatment loses its ANNOUNCER cast row but an act script still speaks as ANNOUNCER, the lookup is a bare `KeyError` unless `_make_act_validator` (not shown) rejects unknown speakers first. The plan asserts P3/P4 "already own the announcer" but does not confirm this guard. R1