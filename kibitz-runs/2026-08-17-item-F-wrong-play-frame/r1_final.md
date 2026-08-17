# r1 FINAL -- item F, the wrong-play announcer frame

**Driver:** Claude (Cowork), sole judge. **Date:** 2026-08-17.
**Input to r2.** Supersedes `driver_anchor.md` where they disagree.

## PROVENANCE

**r1, TWO lanes, both operator-named:** Antigravity `Gemini 3.7 Flash (High)`
(`r1_antigravity_flash37.md`) and `Gemini 3.1 Pro (High)`
(`kibitz-runs/2026-08-17-item-F-pro31/r1/antigravity.md`). Codex quota-held to
2026-08-19 20:31, excluded. **r1 only -- this is not yet a full four-round arc.**

> **THE LANE-COLLISION TRAP, now fixed and worth keeping.** The first launch ran
> both lanes and produced ONE review: `kibitz.py`'s run folder is
> `<date>-<--topic>` and `--topic` defaults to `kibitz`, so lane two overwrote
> lane one. Both runs returned rc=0 and both printed "Reviews collected:
> antigravity: OK", so nothing warned. **`--topic` is the isolation lever** --
> the Pro re-run used `--topic item-F-pro31` and landed clean, with
> `agy_model_selected.txt` reading `Gemini 3.1 Pro (High)`. One topic per lane,
> always.

## WHERE THE TWO LANES AGREE -- treat as settled

1. **The plan's "sampled independently" premise is FALSE.** Pro raised it
   unprompted as a MUST-FIX ("False Premise of Sampling"); it matches the
   driver's own trace. `setting` is a free-text LLM field
   (`_otr_outline._MacroShape`), not a draw. **Two independent arrivals.**
2. **Fix BOTH producers or the fix is undone.** Pro names the site by its meta
   key, `announcer_intro_rewrite` (grounded: the key exists in
   `OTR_LedgerScriptWriter`, with `announcer_intro_rewrite_failed` beside it).
3. **Thread `work_title` ONLY -- never the whole record.** Pro cuts full
   `source_meta` explicitly on KILL 2 grounds; Flash proposes exactly the single
   field. Convergent.
   > *Pro's CUT is aimed at something the anchor never proposed -- Shape A was
   > always one field. Recorded as agreement on the narrow shape, not as a
   > correction.*
4. **Unit tests cannot catch this.** Both say so independently, which is the
   repo's own twice-proven lesson.

## SHAPE C IS DEAD, AND BOTH LANES WERE RIGHT FOR DIFFERENT REASONS

Flash: category error -- `_otr_passage_selector` is a verbatim dialogue-window
slicer (`parse_speeches`, `eligible_windows`, `select_passage` -> `Passage`) and
touches no announcer symbol. **Grounded, upheld.**

Pro: don't wire it until you know WHY it is dead -- it may have a fatal flaw.
**Better instinct, and now answered.** `git log` on the module shows three
healthy commits ending in a REFINEMENT (*"The word count is a request, not a
gate: the selector always returns its best passage"*), it has its own QA doc
(`docs/2026-08-03-passage-selector-QA.md`), and `docs/2026-08-03-public-domain-plays-PLAN.md`
calls it *"Already implemented"*. **It is built-and-parked for the passage lane,
not abandoned as broken.** So it is unrelated, not radioactive -- which is
exactly Flash's conclusion reached from the other side.

**The driver's error stands recorded:** I tabled Shape C on a docstring sentence
containing the word "Verona". A docstring naming your symptom is not evidence the
module solves your defect.

## THE OPEN QUESTION -- STILL OPEN, AND r2 MUST ANSWER IT

Neither lane engaged the clause the anchor flagged. `SafeOpenBrief`'s docstring
says *"``cast`` is the LOCKED cast: the only proper names the announcer may
use."* That is a **hallucination fence**, not a spoiler rule. Both lanes answered
the spoiler half only:
* Flash: a bibliographic title is orientation, not a spoiler. **Accepted.**
* Pro: same, and its `[ASSUMPTION]` volunteers the driver's own caveat back --
  *"The Tragedy of Romeo and Juliet"* might telegraph an ending.

**Unanswered:** does adding `work_title` widen the proper-name allowance enough
that the model imports Illyria, Malvolio and Olivia into a frame that should
describe only THIS scene -- trading a wrong-play frame for a wrong-scene one?

**Pro's SHOULD-FIX 2 is the closest thing to an answer and is promoted here:**
assert the generated `setting` contains **no proper name outside the locked
cast**. That converts the docstring's fence from prose into a testable
invariant, and it is the only proposal that addresses the fence rather than the
spoiler. r2 should cost it and decide whether it is a test-only assertion or a
runtime one -- noting THE LAW forbids it ever REJECTING an episode.

## THE ADOPTED PLAN going into r2

| # | Change | Source | Status |
|---|---|---|---|
| 1 | `work_title: str = ""` on `_otr_line_composer.SafeOpenBrief`; render as `WORK:` in `compose_announcer_intro` when non-empty | Flash MF3 | adopted |
| 2 | Populate it from `_otr_source_identity.identity_from_meta(meta).work_title` -- ONE symbol, both lanes, never raises, empty on degraded | Flash MF4 | adopted, best idea in r1 |
| 3 | Same fix at the I.4.9 rewrite (`announcer_intro_rewrite`); **no dialogue parsing** to recover a value we already hold | both lanes | adopted |
| 4 | `work_title` on `_otr_outline.OutlineRequest`, rendered in `_build_macro_user_prompt`, so upstream `setting` stops minting the wrong place | Flash SF1 | adopted, raised to MUST |
| 5 | `fallback_safe_open` carries the work when non-empty | Flash SF3 | adopted |
| 6 | Pack seam wording: orient by WORK, forbid importing characters/houses/locations from other works | Flash MF3 + OPT1 | adopted, OPT promoted to MUST |
| 7 | Prompt-capture test on BOTH producers, both banks: `WORK: Twelfth Night` present, `Verona`/`Capulet`/`Montague` absent | Flash SF2 | adopted |
| 8 | Assert `setting` carries no proper name outside the locked cast | Pro SF2 | adopted in principle, r2 to scope |
| 9 | ONE live shakespeare leg, read for the play name | driver | required before "done"; batches into the operator's GPU session |
| -- | Shape C, and threading full `source_meta` | -- | **CUT** |

## CONSTRAINTS THAT KILL A PROPOSAL ON SIGHT

THE LAW (no audit may FAIL an episode); no content guardrails on generated
episodes; fidelity lanes invent nothing; story quality is closed -- naming the
right play is correctness, writing a better opening is not on the table; a render
must degrade, never raise.

## r2 SCOPE (coding plan)

Turn rows 1-8 into an ordered diff plan with the exact symbols, say what each
test asserts and why a green suite is not proof, and answer the open question
above. Codex joins from 2026-08-19 for the coding round proper.
