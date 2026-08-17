# r1 judgment -- item F, the wrong-play announcer frame

**Driver:** Claude (Cowork), sole judge. **Date:** 2026-08-17. **HEAD:** `bcde7a81`.

## PROVENANCE -- READ THIS FIRST, IT IS NOT A FULL ARC

**r1 ONLY, and ONE reviewer lane, not two.** The operator named
`Gemini 3.1 Pro (High)` and `Gemini 3.7 Flash (High)` on 2026-08-17 and both were
launched. **Only Flash 3.7's review survives.** Both lanes were pointed at
`kibitz.py`'s own default run directory (`kibitz-runs/2026-08-17-kibitz/r1/`)
rather than an isolated per-lane directory, so the second lane's
`antigravity.md` OVERWROTE the first. `agy_model_selected.txt` still reads
`Gemini 3.7 Flash (High)` and the surviving file's mtime matches the second
launch. **Pro 3.1 produced a review and it is GONE.**

* **The trap, for whoever runs r2:** `kibitz.py --only agy` writes to a run dir
  derived from the DATE, not from `--doc`. Two lanes in one round collide
  silently -- rc=0, "Reviews collected: antigravity: OK", both times. Give each
  lane its own output directory, or run them in separate rounds.
* Codex is quota-held to 2026-08-19 20:31 and was excluded.
* So the record is: **one round, one lane.** This may not be reported as a full
  four-round arc, and the r2 window owes Pro 3.1 a re-run.

Artifacts: `driver_anchor.md`, `r1_antigravity_flash37.md`, this file, and the
two launch logs, all in this directory.

## THE PANEL CORRECTED THE DRIVER, AND IT IS THE THIRD TIME IN ONE DAY

**MUST-FIX 1 is UPHELD IN FULL: Shape C was a category error and it was mine.**
The anchor tabled "wire `_otr_passage_selector.select_passage`" as a candidate
shape on the strength of that module's own docstring sentence -- *"this is what
stops a Forest-of-Arden scene being narrated as if it were Verona."* Grounded at
the file, the reviewer is right and I was wrong:

* The module's symbols are `parse_speeches`, `eligible_windows`, `select_passage`
  returning a `Passage`, and `strip_stage_directions` -- a **verbatim
  dialogue-window slicer**, sized to a word budget.
* Its opening docstring is the operator's 2026-08-03 fidelity ruling: *"a play
  episode is 'very strict -- based on word count and random choice it hones in on
  a specific part of a play to get real specific dialogue, no paraphrasing.'"*
* The Verona sentence is therefore about **carrying the play's own words so there
  is nothing for a model to drift away from** -- DIALOGUE fidelity. It has
  nothing to do with the announcer wrapper. It touches no announcer symbol.

**The lesson generalizes and belongs in the plan: a docstring naming your
symptom is not evidence the module solves your defect.** I matched on the word
"Verona" and inherited a false candidate -- the same shape as item A's ruling
that name-matching ships false positives, arrived at from a different direction.
**Shape C is CUT.**

*(Predicted in the anchor's own section 6.5: "assume there is a third." There
was.)*

## WHAT ELSE SURVIVED GROUNDING

**MUST-FIX 4 is upheld and it is the best idea in the review.**
`_otr_source_identity.identity_from_meta(meta)` returns a `SourceIdentity`
dataclass carrying a real `work_title: str = ""` attribute; it "never raises",
and a degraded meta yields an empty string rather than `None`. Crucially it
**already normalizes BOTH adaptation lanes at one symbol** --
`prov["work_title"] = "meta.source_meta.play_title"` for shakespeare and
`"meta.source_meta.title"` for public_domain. That dissolves the family problem
the anchor raised in section 5 without a per-lane branch, and it is strictly
better than the hand-rolled lookup the anchor implied.

> **Correcting the anchor's own citation:** section 2 cited
> `identity_from_meta` as `prov["work_title"]`. That is the PROVENANCE map, not
> the value. The value is the dataclass attribute. Cited by symbol and still
> wrong about which symbol -- worth noting, because symbol-citation is a guard
> against rot, not against misreading.

**MUST-FIX 2 is upheld.** The I.4.9 rewrite must be fixed in the same change or
it silently restores the defect, and the reviewer's refinement is correct and
better than anything in the anchor: do NOT ask `derive_produced_open_brief` to
recover the title from scene-1 dialogue. The title is an immutable constant in
`meta`; parsing it back out of prose would be a non-deterministic route to a
value we already hold.

**SHOULD-FIX 1 is upheld and is arguably a MUST.** The reviewer spotted a hole
the anchor missed: even with the announcer prompt fixed, `_MacroShape.setting`
is authored by the macro LLM, which sees only the brief -- so **the hallucinated
"Verona" can be minted upstream and then handed to a correctly-fixed announcer as
`SafeOpenBrief.setting`**. Fixing only the announcer makes the frame name the
right play while still describing the wrong place. That is a second producer of
the same defect, distinct from I.4.9, and the anchor did not have it.

**SHOULD-FIX 3 is upheld on fact.** `fallback_safe_open` reads
`"Good evening. This is SIGNAL LOST. We open on {where}."` and does omit the
work entirely.

## WHAT I AM NOT ACCEPTING YET -- and it is the one thing r2 must break

**MUST-FIX 3 (the spoiler objection) is HALF answered, and the panel disposed of
the wrong half.** The reviewer argues that KILL 2 starved the *script_brief*,
which carries the outcome, whereas a bibliographic title is orientation rather
than a spoiler. That distinction is real and consistent with the docstring, and I
accept it **as far as plot leakage goes**.

**It does not touch the other clause.** `SafeOpenBrief`'s docstring also says:
*"``cast`` is the LOCKED cast: the only proper names the announcer may use."*
A work title IS a proper name, and that sentence is not about spoilers at all --
it is a hallucination fence. The reviewer never engages it. So the open question
for r2, stated precisely so it cannot be answered with the spoiler argument
again:

> Adding `work_title` widens the announcer's proper-name allowance from one
> closed set (the locked cast) to that set plus one string. Does naming
> "Twelfth Night" make the model MORE likely to import Illyria, Malvolio and
> Olivia into a frame that should describe only this scene -- i.e. does the fix
> trade a wrong-play frame for a wrong-scene frame? And if so, is the answer a
> tighter system prompt, or is it the OPTIONAL negative constraint the review
> already proposes ("do not import characters, houses, or locations from other
> plays"), promoted to a MUST?

I have no evidence either way, which is exactly why it goes to a panel and not
into code.

## ACCEPTANCE -- upheld with one hard amendment

SHOULD-FIX 2's prompt-capture harness is the right instrument: capture the user
message on BOTH producers with a Twelfth Night fixture, assert `WORK: Twelfth
Night` present and `Verona`/`Capulet`/`Montague` absent, across both banks.

**But it is not sufficient on its own, and the amendment is not optional.** This
repo has proven twice in one day that a green gate is not a working fix (the
2026-08-17 style build: every text check passed and the pixels moved 40% less;
and the PRE-FIX announcer that minted a literal photograph while every prompt
agreed). A prompt-capture test proves the string ARRIVED. It cannot prove the
announcer USED it, because the announcer is a model. **One live leg on the
shakespeare lane, read for the play name, is owed before this item is called
done** -- and per the operator's 2026-08-16 ruling that eyes-on sessions are
deferred, that leg batches into his declared GPU session rather than blocking the
code.

## DISPOSITION

| Item | Verdict |
|---|---|
| Shape C (`_otr_passage_selector`) | **CUT** -- category error, driver's own, grounded |
| Shape A (`work_title` threaded) | **ADOPTED as the path**, pending the proper-name question above |
| Shape B (prompt-only) | Not adopted alone -- persuasion, not structure (Bible `12.103`); survives as the OPTIONAL negative constraint |
| `identity_from_meta(...).work_title` as the single source | **ADOPTED** -- one symbol, both lanes |
| Fix I.4.9 in the same change | **ADOPTED**, no dialogue parsing |
| Thread `work_title` into `OutlineRequest` | **ADOPTED and raised toward MUST** -- upstream `setting` is a second producer |
| `fallback_safe_open` carries the work | **ADOPTED** |
| Prompt-capture acceptance test | **ADOPTED**, plus one live leg -- a green gate is not a fix |

**NO CODE IS WRITTEN FROM THIS ROUND.** One lane, one round, one open design
question, and the operator's standing rule is that a design fork gets the arc
before the diff.

## WHAT r2 OWES

1. Re-run **Pro 3.1** into an ISOLATED output directory (see the trap above).
2. Answer the proper-name / wrong-scene question, without re-litigating spoilers.
3. Codex from 2026-08-19 20:31, for the coding-plan round proper.
4. Only then r3 wiring and r4 convergence.
