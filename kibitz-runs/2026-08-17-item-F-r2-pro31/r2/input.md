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

## FABLE GATE (added 2026-08-17 on the operator's call) -- IT KILLED A ROW AND FOUND A DEFECT

Fable was asked the ONE open question above, cold on the artifacts. It answered
it, killed an adopted row with hard evidence, and found a defect none of the
three of us had. Every claim below was re-grounded by the driver at the files.

**1. THE OPEN QUESTION IS ANSWERED: THREAD THE TITLE.** The risk is real but
small and dominated by the defect being fixed. The reasoning is the good part:
**the observed defect is a STARVATION VACUUM.** The seam orders *"Sentence 1
orients the listener: the play-world place and who is there"* and then supplies
no play, so the model must retrieve to obey. Elaboration risk is highest when a
prompt starves an instruction and lowest when it supplies material. The fix
supplies the material, and `compose_announcer_intro` already hands over three
scene-pinning fields (`SETTING`, `OPENING STATUS`, `CAST`) to keep it home. The
wrong-play frame is the DEFAULT output of the current prompt; the wrong-scene
frame requires the model to override three provided facts with one title. **A
defect that fires on the default beats a risk that requires ignoring the inputs.**

**2. ROW 8 IS DEAD -- and it would have broken the build.** Pro's "assert
`setting` contains no proper name outside the locked cast" rejects the CORRECT
setting on at least 5 of the 14 manifest rows. **Verified by the driver against
the manifest itself:** Hamlet 1.1 is *"On a cold platform at **Elsinore**"*,
Romeo and Juliet 2.2 is *"In **Capulet's** garden"*, Comedy of Errors 3.1 is
*"Antipholus of **Ephesus**"*, plus Arden and the wood near Athens -- none of
those place names are cast members, and all are the manifest's OWN synopsis
wording. Forcing "a castle platform in the north" instead of "Elsinore" is
anti-fidelity on the lane where fidelity outranks arc. **Replaced by the
inversion:** assert the frame carries no proper name belonging to a DIFFERENT
manifest row -- for a Twelfth Night episode, no `Verona`/`Capulet`/`Montague` and
no other row's `play_title`. Finite, checkable against the manifest, cannot
reject legitimate writing about the correct play, and it is precisely the check
that would have caught the shipped defect. Per THE LAW: a test/receipt assertion,
never a runtime reject.

**3. THE DEFECT NOBODY HAD -- a bare title promises the WHOLE PLAY.**
`identity_from_meta` carries `work_title` alone, so the adopted plan has the
announcer say "Tonight: Twelfth Night" and then deliver one ~300-word scene. That
is a frame/content contradiction of exactly the kind this item exists to close,
and it is **systematic -- all 14 rows, every episode -- where the wrong-play bug
was intermittent.** The fix is wording, not threading: present *a scene from* the
work.

**4. THE SPOILER-BY-TITLE CAVEAT IS MOOT** on the current manifest -- **verified:
every `play_title` is short form** (`Romeo and Juliet`, `Macbeth`, `King Lear`).
It becomes live only if a future row vendors a full Folger title. One sentence in
the r2 test plan, not a design change.

**5. THE SECOND PRODUCER IS AN ARGUMENT *FOR* SHAPE A, which neither lane made.**
`_PRODUCED_OPEN_PROMPT` orders *"use ONLY information visible in the scene-1
lines"*, so once row 3 threads the title there too, the I.4.9 rewrite regrounds
the frame in what is actually spoken and squeezes out imported furniture. The
two-producer requirement is not only leak-proofing; it actively reduces the
question-A risk.

### THE DRIVER'S OWN ADDITION, and it settles Shape B for good

**The cast-only fence ALREADY EXISTS IN THE SHIPPED SEAM AND IT DID NOT HOLD.**
Verbatim from `announcer_intro_safe_system`: *"Use ONLY the proper names in the
cast list below; invent none."* `Verona`, `Capulet` and `Montague` are proper
names, that rule already forbade them, and they shipped anyway. So:
* Fable's seam rewording (row 6) is worth shipping and **cannot be the
  guarantee** -- we have direct proof this exact instruction class fails.
* **The cross-play leak check is therefore the ENFORCEMENT, not a nicety**, and
  Shape B alone is dead on evidence rather than on principle. This is the same
  finding Bible `12.103` records -- an instructed model is still a model -- but
  here we have the shipped artifact proving it on this very prompt.

### ADOPTED PLAN, amended

| Row | Change |
|---|---|
| 6 (amended) | Ship Fable's two seam lines verbatim: *"Sentence 1 names tonight's work from the WORK line and places the listener using only the SETTING and the cast list - nothing from the play beyond this one scene."* and *"Use ONLY the WORK title and the proper names in the cast list below; invent none."* |
| 8 (**REPLACED**) | ~~no proper name outside the cast~~ -> **cross-play leak check**: the frame carries no other manifest row's `play_title` or signature place/house names. Test-only. |
| 10 (**NEW**) | The frame must present *a scene from* the work, not the work. Either render `WORK: a scene from <title>` or let the row-6 sentence carry it -- r2 picks one. Wording only, no new threading. |
| -- | `scene_label` threading: NOT raised to MUST -- new surface, and the public_domain mirror deliberately excludes its equivalent. |

## CONSTRAINTS THAT KILL A PROPOSAL ON SIGHT

THE LAW (no audit may FAIL an episode); no content guardrails on generated
episodes; fidelity lanes invent nothing; story quality is closed -- naming the
right play is correctness, writing a better opening is not on the table; a render
must degrade, never raise.

## r2 SCOPE (coding plan)

Turn rows 1-8 into an ordered diff plan with the exact symbols, say what each
test asserts and why a green suite is not proof, and answer the open question
above. Codex joins from 2026-08-19 for the coding round proper.
