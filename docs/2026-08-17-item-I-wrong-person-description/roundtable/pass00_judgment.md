# R1 judgment -- item I. The panel inverted the premise; the item is re-sized.

**Driver:** Claude (Cowork), panelist and sole judge. **Date:** 2026-08-17.
**Anchor written BEFORE the fan-out** (`pass00/claude_anchor.md`).

**PANEL PROVENANCE, exact.** Both agy lanes returned `RESOURCE_EXHAUSTED` (429),
so under the operator's 2026-08-17 substitution ruling the seats were filled:
* **Cloud roundtable (~$0.12):** `openai/gpt-5.6-sol` (full),
  `google/gemini-3.1-pro-preview` (**TRUNCATED at its 2000-token cap -- verdict
  plus one MUST-FIX only**), `deepseek/deepseek-v4-pro` (**FAILED**, spent
  max_tokens on hidden reasoning, zero content).
* **Substitute local reviewers:** Sonnet (mechanics), Opus (measurement), Fable
  (narrative).
Codex: quota-held to 08-19, absent. **This is r1 with five reviewers of whom one
was truncated and one failed. Not a full arc.**

---

## THE HEADLINE: THIS IS AN ACCELERATING REGRESSION, NOT A BACKGROUND RATE

Opus measured it; **I re-derived it from scratch with an independently written
detector and got the same shape**:

| month | ledgers with a pitch | affected | rate |
|---|---|---|---|
| 2026-07 | 74 | 5 | **6.8%** |
| 2026-08 | 36 | 18 | **50.0%** |

(Opus: 6.8% -> 52.8%. Two independently written detectors, same inflection.)

**~7-8x in one month.** My anchor sized this as a stable ~16% and asked which fix
to build. **That was the wrong question.** The first action is to BISECT what
changed in early August. A fix chosen now may be treating a symptom of a recent
change.

## WHAT THE PANEL PROVED I HAD WRONG -- five corrections, all grounded

1. **MY MARQUEE SYMBOL DOES NOT EXIST.** I cited
   `_otr_casting.build_description_prompt` throughout. **Verified absent.** The
   real symbol is **`_otr_casting._build_user_prompt`**. Caught independently by
   Sonnet and Opus. In a document whose banner reads *"CITE SYMBOLS, NEVER LINE
   NUMBERS"* because citations rot, the headline citation was fiction. The
   substance survives verbatim (`story_text = brief` -> `f"Story: {story_text}"`,
   `f"Name: {name}"`, no precedence anywhere).
2. **MY SECOND DETECTOR IS BOGUS AND MUST NOT BE QUOTED AGAIN.** I reported
   "18 rows / 14 ledgers" as a peer floor. Opus reproduced my *pitch* detector
   exactly (20/28 -- confidence anchor) and then showed a bare name-shape test
   returns **765 ledgers / 1,259 rows**, because the contract's own
   `<story-linked role>` slot legitimately holds Title-Case and ALL-CAPS prose:
   *"Late 40s, seasoned Marine Biologist"*, *"40s, LEAD SCIENTIST"*. **Those are
   the contract WORKING.** The real union is **31 ledgers / 43 rows**, computed.
3. **"FOURTEEN BAKED COPIES" IS ONE EPISODE.** All 14 `baked_ledger.json` files
   are `signal_lost_nightshift_erasure_20260809_115705`, twelve of them as
   `wan_cost_ladder_*` bench fixtures under `_shared\`. The freeze evidence is
   1 episode. **Retracted.**
4. **IT IS NOT PIXELS. THE WRONG NAME IS SPOKEN ALOUD.** `build_voice_card` feeds
   `character_description` into the line-composer prompt (**confirmed at the
   file**), and Opus measured the wrong name in **spoken lines in 18 of 31
   affected episodes**, captions in 10, treatment in 28. Real shipped lines:
   *"FLETCHER CORBEN: Not without Evelyn, Juliana."* (one row, two names);
   *"CLARISSE DRAKE: Then surrender the unmarked reel, Dr. Hartley."* -- **she is
   addressing herself.** Fable independently found the same class plus phantom
   names in the audible narration. **This kills post-hoc description repair: by
   the time the description is repairable the dialogue is already contaminated
   from the same carrier.**
5. **IT IS NOT THE `original` LANE ONLY.** 7 of 31 affected episodes record **no
   pitch at all** -- 6 `media_archive`, 1 `public_domain` -- where the name
   arrives via the interpreter's own brief. **Confirmed at the file:** the
   media-archive interpreter's schema literally asks for `"casting_brief":
   "source-grounded human roles and voices"`. My claim that the brief instruction
   never asks for names holds for `news_interpreter` ONLY.

## THE CUT MOVES: `casting_brief`, not the pitch

Opus measured the carrier: the wrong name is in `meta.news.casting_brief` for
**42 of 43 rows (98%)** versus the pitch at **81%**. **Only the brief reaches the
non-pitch lanes.** So options A and C are pitch-shaped and structurally cannot
fix 7 of 31 affected episodes. **B is the only candidate covering every lane.**

**And it must patch the PROMPT-LOCAL copy, never the stored field.** GPT and
Sonnet converged here, and Sonnet named the exact insertion point: the local
`casting_brief` variable in `OTR_LedgerScriptWriter` between its read
(`casting_brief = briefs.casting_brief`) and its use
(`lock_cast(casting_brief=...)`), with no write-back to `meta["news"]` in
between. **Confirmed by my own consumer sweep: `casting_brief` is read across
NINE modules**, including `video_engine`, which prints it verbatim into the
human-readable treatment. Mutating the stored field would silently rewrite that
report.

## OPTION A IS CUT -- three reviewers, one mechanism

Fable and Sonnet independently killed it, and the mechanism is decisive:
continuity runs at H.5 on the **locked cast**, and `_render_cast_block` feeds the
continuity model the **already-corrupted** `character_description`. So
`"Lucille (Nia Philbin)"` is not a cross-check -- it is the technical slot
paraphrasing the same corruption it was shown. Fable added that the pairing is
unparseable by `_speaker_matches`, self-contradictory in the same ledger (one
member both `known_by` and `hidden_from` the same fact), and **emitted in
violation of its own prompt** -- `_CONTINUITY_SYSTEM_PROMPT` says *"Use ONLY the
exact character names from the Cast block. Never invent"* (**confirmed at the
file**). My anchor's "a mapping EXISTS somewhere" was false as data. **CUT.**

## OPTION C IS REOPENED -- my dismissal cited a rule my own log excludes

Opus showed three of my four dismissal clauses fail: the cast pool is flat lists
of bare strings, so a name binds to nothing but a gender tag; `12.51` is about
seed-receipt semantics; and **`10.08` is excluded by this defect's own log
entry**, which reads *"NOT 10.08 (two correlated attributes from two Python
draws...)"*. I cited a rule to reject a fix that my own PBUG had already ruled
out as the wrong analogy. Only the gender-ladder clause has teeth, and it must be
stated correctly: `gender_of_first_name` returns `"unknown"` for off-pool names
so the repair silently stands down. **Reopened, not adopted** -- Fable's
counter stands: the pitch model's name prior is narrow (LUCILLE, WALSH, GRISWOLD
recurring across unrelated episodes), and the pool exists to prevent exactly that.

## TWO DEFECT SURFACES, AND FIXING ONE BLINDS THE DETECTOR TO THE OTHER

Fable's sharpest structural point, and it changes the acceptance test:
* **Surface 1** -- alien NAME in the identity slot, persona congruent with the
  row.
* **Surface 2** -- whole-persona TRANSPLANT, gender-crossed (a male row carrying
  female face/voice/presence prose). RICK's row is a **chimera**: Hal's age band
  welded to Lucille's persona -- so the model BLENDS under ambiguity rather than
  copying.

**Both detectors key on names. Strip the names and surface 2 goes invisible.** So
the acceptance metric is **persona congruence** -- description age/gender/voice
words against the row's Python-decided facts -- **not "no proper name found"**.

## THE LAW CARVE-OUT I DROPPED

Sonnet caught it: THE LAW's next sentence is *"Structural JSON/schema/IDs/roster/
... failures remain fail-closed because they protect a usable ledger rather than
judge prose"*, and it also sanctions *"Same-story LLM cleanup"*. So (a) whether a
wrong-person description is a ROSTER defect or a QUALITY defect decides whether a
gate may fail closed at all, and (b) the LAW-safe repair shape is an **LLM reroll
of the flagged slot**, not Python string surgery on a field tagged
`# LLM slot: creative`. **Neither is settled and both must be before E is scoped.**

**And GPT's hardest catch, confirmed at the file:** `llm_write_description`
raises `CastingFailedError` on ladder exhaustion with an explicit comment about
`lock_cast`'s `CastValidationLLMError`. **Adding a name validator to that ladder
can reduce publication** -- which now collides with the operator's same-day rule
that `otr/obs/` volume is the success signal. Prevention on every attempt;
deterministic non-empty output afterwards; never an unbounded regeneration.

## A SECOND COLLISION PATH NONE OF MY FIVE OPTIONS TOUCH

Sonnet found `_apply_llm_slot_fill`, which reassigns `row["name"]` **after**
descriptions are written -- deterministically pairing a new name with an old
description. Gated off by default (`OTR_NAME_MODE`), so almost certainly not the
reported episode, but **any fix that ships and declares item I closed leaves this
same-family bug open.** Scope item I to pool mode explicitly and open a follow-up.

## DISPOSITION -- the plan for r2

1. **BISECT early August first.** Nothing else is decided until we know what
   moved the rate 7x. This replaces "compute the union" as the opening move.
2. Strip pitch/brief names from the **prompt-local** `casting_brief` copy at the
   named insertion point. Episode-scoped literal substring replace off
   `selected_concept.cast[].name` **plus** the non-pitch lanes' brief names --
   never fuzzy, never name-shape.
3. Substitute **role nouns**, not the assigned cast names (that is A in B's
   clothing and needs the dead mapping).
4. Acceptance = **persona congruence**, plus the name check as a secondary.
5. Repair, if any, = same-story LLM reroll of the flagged slot; never a raise,
   never a publication-reducing gate.
6. **Forward-only.** The wrong name is in the audio of 18 of 31 episodes, so the
   corpus cannot be repaired by touching descriptions.
7. Corrected citations throughout: `_build_user_prompt`; union 31/43; one baked
   episode, not fourteen.

**NO CODE THIS ROUND.** Five reviewers, five driver corrections, one inverted
premise, and the opening move changed from "build a fix" to "bisect a
regression".
