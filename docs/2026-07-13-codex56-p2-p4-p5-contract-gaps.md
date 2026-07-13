# original_codex56sol -- P2 / P4 / P5 contract gaps (coding plan)

**Date:** 2026-07-13
**Branch:** `v2.0-alpha`
**Grounded base:** `c8f975eb` (after PBUG-20260713-10 and -11 shipped and live-reverified)
**Status:** plan for review; no code written yet.

## 0. Why this exists

Two live production kills today, same class, both fixed and live-reverified at prompt
`28fe3cdf`:

- **PBUG-20260713-10** (`3a98a6f1`): the P9 audit could cast a *blocking* vote against
  the manifest -- a Python-compiled artifact its only repair route (a spoken-line
  retake) cannot touch. Dead end at the last gate.
- **PBUG-20260713-11** (`58983363`): `clue_plan`'s one-clue-per-lost-object rule existed
  only as `Field(min_length=3)`. It was in no seam and `_repair_rules` had no `P1`
  branch, so the repair prompt could not state it. The model repeated the defect and the
  ladder exhausted.

A grounded fan-out audit then found three more instances of the same class, each one a
future dead render. This plan closes them.

**The class, stated once:** an invariant enforced in Python that is absent from BOTH the
seam prompt AND `_repair_rules(pass_id)`, or whose only available repair route cannot
physically satisfy it. When it trips, the ladder cannot state the rule and the episode
fails closed with no recovery.

**The structural cause** (verified): `schema_shape_instruction`
(`nodes/_otr_structured_call.py`) emits only top-level key names and required nested
paths -- no `min_length`, no `max_length`, no `pattern`, no `extra="forbid"`. **The model
never sees the schema.** Every pydantic constraint in every lane is therefore
model-invisible unless a human wrote it into the seam or the repair rules. Fixing that
generator is explicitly OUT of scope here (it changes every lane's prompts and needs its
own regression pass); this plan only closes the three live-risk gaps in this lane.

---

## 1. GAP A -- P5 clue ownership: the rule is stated, but enforced where no repair can act

### Grounded evidence

`_validate_score` (`_otr_original_codex56sol.py:1030-1043`) requires, for every
lost-object anchor, at least one **non-announcer** beat that BOTH carries a clue for that
object's thread AND names the anchor in its intent:

```python
for beat in score.beats
if beat.char_id != "announcer"
```

The P5 `_call` post_validator is deliberately **blinded** to this -- it passes `None`
for the grounding contract (`:2000-2002`):

```python
post_validator=lambda value: _validate_score_attempt(
    value, truth, None, story_rules,
),
```

So the P5 ladder never fires on a grounding defect. The violation surfaces afterwards at
`:2004` and routes to `_repair_score_grounding_intents`, whose only tool is the bounded
`codex56_score_anchor_patch`. That seam explicitly forbids the fix:

> "Do not change cast, scenes, shots, beat ids, **clue_ids**, arc phases, music, dialogue,
> or any unrequested field."

`_score_grounding_repair_plan` (`:1306-1312`) then hard-aborts when the only beat carrying
an object's clue is the announcer:

```python
eligible = [beat for beat in score.beats
            if beat.char_id != "announcer"
            and relevant_clues.intersection(beat.line_intent.clue_ids)]
if not eligible:
    return None          # -> raise "P5 grounding repair has no eligible ... plan"
```

**Result: hard abort, zero LLM repair calls.** The one rung that could have re-authored
the score -- the P5 typed repair, which *does* already carry the rule -- never sees the
error.

### Why the blinding was right, and what must change

The blinding is not a bug by itself. It implements Lesson 20: do not regenerate a whole
score because one intent leaf is missing an anchor word. That localized omission is
exactly what the bounded intent patch is for, and it works.

The defect is that the blinding is **too wide**. It hides two different failures:

| Failure | Owner | Bounded patch can fix it? |
|---|---|---|
| an eligible non-announcer clue beat exists, but its `intent` text omits the anchor | intent prose | **YES** -- this is the patch's whole job |
| **no** non-announcer beat carries that object's clue at all (the announcer owns it) | `clue_ids` **assignment** -- a P5 authoring decision | **NO** -- the patch seam forbids touching `clue_ids` |

Only the second is unrepairable. It must be returned to the model that owns it (P5),
while the ladder can still re-author the score.

### Fix

Split the check. Add a narrow `_validate_score_clue_ownership(score, grounding)` that
asserts only the structural half -- every lost-object anchor has at least one
non-announcer beat carrying one of its clues -- and OR it into the P5 post_validator:

```python
post_validator=lambda value: _validate_score_attempt(
    value, truth, None, story_rules,
) or _validate_score_clue_ownership(value, grounding),
```

The anchor-*naming* half stays out of the ladder, still owned by the bounded patch. The
rule is already stated in the seam and in `_repair_rules("P5")` ("each lost-object anchor
MUST appear verbatim in at least one non-announcer clue-carrying intent for its thread"),
so the repair prompt is already correct -- it simply was never invoked.

`_score_grounding_repair_plan`'s `if not eligible: return None` and its raise stay as
fail-closed defense in depth; production should no longer reach them.

**Ownership statement:** Python proves the shortfall (a set operation over the accepted
truth map). Python never reassigns a clue -- clue ownership is story.

---

## 2. GAP B -- P4 fair-play: a blocking finding kills the episode with no repair route in existence

### Grounded evidence

`_call(pass_id="P4", ...)` (`:1985`) has **no post_validator** (it takes the default
`lambda value: None`). Immediately after (`:1986-1988`):

```python
corroborated_fair_blocks = _corroborated_fair_blocks(fair, truth)
if corroborated_fair_blocks:
    raise OriginalCodex56SolContractError("fair-play audit rejected the truth map")
```

There is **no P3 retake, no P4 rerun, no repair prompt, and no `_repair_rules("P4")`
branch.** One `blocking=true` finding whose `field_path` root is a truth-map collection
and whose `item_id` resolves (`_corroborated_fair_blocks:1151-1160`) ends the episode
outright. Nothing tells the model what `blocking` costs -- compare the post-PBUG-10 P9
seam, which now spends a paragraph bounding blocking authority.

**Second, independent defect -- the seam contradicts the schema.** The seam says:

> "Report only concrete corroborable findings; **taste notes are warnings**."

But `FairPlayReport` (`:191-193`) is `accepted: bool` + `findings: list[...]` -- **there is
no `warnings` field** -- and `StrictModel` is `extra="forbid"` (`:41`). A model that obeys
the seam literally and emits `warnings` gets an `extra_forbidden` ValidationError, sent to
a typed repair with **no P4 branch**. The prompt instructs a shape the schema rejects.

**Third:** `FairPlayReport.accepted` is read nowhere. `accepted=false` with no corroborated
finding ships silently.

### The key ownership distinction (this is NOT the P9 case)

P9's rejection was aimed at the **manifest** -- Python-derived, already proven, and
unfixable by the available retake -> correct treatment was *demotion to advisory*.

P4's rejection is aimed at the **truth map** -- which is **model-authored by P3**. A
fair-play rejection is therefore *legitimate and meaningful*: fair play is the one thing
this pass exists to protect. Demoting it would gut the gate. The right answer is the
mirror image: **give the rejection the repair route it always deserved.**

### Fix

1. **Schema/prompt lockstep:** add `warnings: list[str]` to `FairPlayReport`, so the
   non-blocking notes the seam already promises have a legal home. Mirrors
   `FinalContractAudit`.
2. **Envelope validator** `_validate_fair_play_envelope(report, truth)` as the P4
   post_validator, mirroring `_validate_audit_envelope`:
   - every `blocking` finding MUST corroborate -- `field_path` root is a real truth-map
     collection and `item_id` resolves to a real id in it -- with non-empty `category`
     and `detail`. An **uncorroborated blocking finding** is ungrounded: Python cannot
     tell what it means, so it returns to typed repair and fails closed if the ladder
     exhausts (same rule as an ungrounded script finding at P9).
   - `accepted` MUST be false iff at least one blocking finding exists (honors the field
     that is currently dead).
3. **Repair route:** a corroborated blocking finding triggers exactly ONE truth-map
   retake -- `P3_rerun` against `codex56_audible_truth_map` with the findings -- then
   rebuilds the grounding contract (`_build_grounding_contract(draw, truth)`, since the
   truth map changed) and re-audits once (`P4_rerun`). If the rerun still corroborates a
   block, fail closed with the existing error. Bounded, one retake, mirrors
   `P9_retake`/`P9_rerun` exactly.
4. **Seam + `_repair_rules("P4"/"P4_rerun")`:** state the blocking authority -- blocking
   only for a concrete truth-map item named by `field_path` + `item_id`; taste, tone, and
   style notes go in `warnings`; `accepted` is false iff a blocking finding exists.

**Ownership statement:** the model owns the fair-play verdict and every word of the
retaken truth map. Python only checks that a blocking verdict names a real item, and
counts the retakes.

**Open risk to flag:** rebuilding the truth map at P3_rerun invalidates `grounding`, which
must be recomputed before P5. Any consumer captured between P4 and P5 must be re-derived.
The plan recomputes exactly one value (`grounding`); a reviewer should confirm nothing
else is captured in that window.

---

## 3. GAP C -- P2 triage: "do not block the card you select" is stated nowhere

### Grounded evidence

`_validate_triage` (`:873-877`) enforces two rules:

```python
if triage.selected_possibility_id not in ids:
    return "selected_possibility_id must exactly match one slate id"
if any(f.blocking and f.possibility_id == triage.selected_possibility_id
       for f in triage.findings):
    return "triage selected a possibility it marked blocking"
```

It IS wired as the P2 post_validator (`:1979`), so the ladder does fire -- P2 is
therefore *less* severe than P1 was. But the seam states neither rule:

> "Act as a technical contract checker. Select one possibility that is causally solvable
> through sound and complies with every safety and originality constraint. Findings must
> identify concrete fields; do not assign an originality score and do not rewrite prose."

and **`_repair_rules` has no `P2` branch**, so the repair prompt is the seam + the generic
"Return the same complete artifact, repairing only the typed contract error" + the raw
error string. The seam actively invites the contradiction: it asks for a checker that
reports findings and never says a blocking finding on the *selected* card is forbidden.

### Fix

Statements only -- no behavior change:

- **Seam:** `selected_possibility_id` must be copied verbatim from one supplied
  `possibility_id` (never a title, index, or invented id). Never mark the possibility you
  select as `blocking`: if every candidate has a concern, select the least-compromised one
  and record its concern with `blocking=false`.
- **`_repair_rules("P2")`:** the same two rules, plus `blocking` is a boolean.

---

## 4. Non-goals (explicitly out of scope)

- `schema_shape_instruction` constraint exposure -- the structural root. Touches every
  lane; needs its own chunk, regression pass, and smoke.
- `ScoreIntentPatch.replacements` cap 6 vs >=7 planned targets at 5+ lost objects, and
  `PossibilityCard.callers` cap 4. Both are **latent**: `constraint_deck.json` ships 3
  draws, all with exactly 3 lost objects, so neither is reachable in production today.
  Record as known risks; do not fix blind.
- `num_characters_advisory=2` vs the P5 seam's "3-5 cast" and `cast` `min_length=3`. A live
  contradiction, but P5 has passed it in every observed run; needs a live artifact before
  it earns a change.

## 5. Admission rule

All three gaps are **static findings from a code audit**. Per `AGENTS.md` and
`PRODUCTION_SPRINT_LESSONS.md`, a review observation never creates a PBUG or a Bug Bible
rule on its own. **Nothing here enters `PROD_BUG_LOG.md`** unless and until it fails a
live run. They are fixed and tested as dev catches.

## 6. Test plan

Focused, in `tests/test_original_codex56sol_runner.py`:

1. **P5 ownership reaches the ladder:** a score whose only clue for one lost object sits on
   the `announcer` beat is rejected by the P5 post_validator with the exact ownership
   error, and the runner shows a P5 typed-repair attempt whose prompt carries the
   non-announcer rule -- instead of today's zero-call abort.
2. **P5 naming still uses the bounded patch:** a score with an eligible non-announcer clue
   beat whose intent omits the anchor still routes to `codex56_score_anchor_patch` and does
   NOT trigger a whole-score repair (guards the Lesson-20 design against regression).
3. **P4 corroborated block retakes the truth map:** blocking finding on a real clue id ->
   exactly one `P3_rerun` + one `P4_rerun` -> episode completes; grounding is recomputed.
4. **P4 rerun still blocked fails closed:** the retaken truth map is blocked again ->
   `OriginalCodex56SolContractError`, one retake only, no loop.
5. **P4 uncorroborated block returns to typed repair:** `blocking=true` with an `item_id`
   that resolves to nothing -> post_validator error -> typed repair -> accepted.
6. **P4 warnings are legal:** a report carrying `warnings` validates (today it raises
   `extra_forbidden`), and `accepted=false` with zero blocking findings is rejected.
7. **P2 rules are stated:** seam and `_repair_rules("P2")` both carry the verbatim-id and
   never-block-the-selected-card rules; a triage that blocks its own selection routes to
   typed repair and recovers.

## 7. Gates

Focused tests -> full OTR suite -> Bug Bible -> `git diff --check` -> `py_compile` ->
commit + push to `v2.0-alpha` -> canonical 42-word `original_codex56sol` live run
(Aion 3.0 Mini creative + Mistral-Nemo technical) proving `RESULT SUCCESS`,
`obs_publish OK`, and the final OBS asset on disk.

No canonical workflow change is expected: no node, widget, link, or schema surface of
`workflows/otr_canonical.json` is touched by any item in this plan. If that turns out to
be false, the JSON change ships in the same commit as the code.
