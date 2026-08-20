# QA PROMPT -- item I, the wrong-person character description

**For the operator to paste into Codex (or any reviewer) by hand.**
Written 2026-08-20. Branch `v2.0-alpha`. The diff is UNPUSHED at time of
writing; if it has landed, review the commit instead of the working tree.

**How to run it yourself (the lane is not on PATH -- the hash dir changes):**

```bash
"$LOCALAPPDATA/OpenAI/Codex/bin"/*/codex.exe exec -m gpt-5.6-sol -s read-only --skip-git-repo-check -C "C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio" - < docs/2026-08-20-item-I-QA-PROMPT.md
```

---

## THE PROMPT (everything below this line is what the reviewer reads)

You are doing a **post-coding QA pass on a finished diff**. This is not a design
review -- the design already ran a four-round panel and the last round
(`kibitz-runs/2026-08-20-item-I-wrong-person/r4/judgment.md`) refused to
converge on the driver's anchor and forced a rewrite. Your job is to find what
is WRONG WITH THE CODE AS WRITTEN. Read the real files. Report MUST-FIX first.

### What the change is for

Bug Bible `11.61` (in a SEPARATE repo:
`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\BUG_BIBLE.yaml`,
entry starts line 8481 -- READ IT). Two naming authorities are handed to one
prompt with no precedence stated: an upstream creative pass invents cast names,
a downstream deterministic assigner overrides them, and the per-record
description prompt receives both. The model then writes a description about the
OTHER person, and that string is copied verbatim into the portrait prompt.

Measured on the live archive by `scripts/audit_wrong_person_census.py`:
**70 contaminated row occurrences across 42 ledger files = 46 unique rows in 30
unique episodes; 67 of 70 also contaminated `meta.visual_plan.characters[NAME].portrait_prompt`.**
Live at HEAD -- the most recent episode on the affected lane
(`signal_lost_rivers_embrace_20260817_233013`) has BOTH dramatic rows wrong.

### Files in the diff

| file | state |
|---|---|
| `nodes/_otr_name_authority.py` | NEW -- the shared detector/reconciler |
| `nodes/_otr_casting.py` | MODIFIED -- `lock_cast` reconciles, checks, recovers |
| `nodes/OTR_LedgerScriptWriter.py` | MODIFIED -- supplies identities, persists guard events |
| `scripts/audit_wrong_person_census.py` | NEW -- the archive sweep instrument |
| `tests/test_name_authority_boundary.py` | NEW -- 16 tests |

### HARD CONSTRAINTS the code must satisfy -- check each one against the source

1. **NOTHING in this mechanism may fail an episode.** Operator directive
   2026-08-20: no block, no reject, no retire, no raise out of this path. The
   check is deliberately NOT a `structured_call(post_validator=...)`, because
   validator exhaustion becomes `CastingFailedError` and `lock_cast` re-raises
   it. **Verify there is no path out of `_enforce_name_authority` that raises**,
   including inside the clean-room retry and the deterministic floor.
2. **No fuzzy repair of generated prose.** `11.61` calls it actively harmful:
   renaming the intruder in finished text leaves the other person's face and
   delivery in place, so the row still describes the wrong human and nothing
   points at it. The contaminated response must be DISCARDED WHOLE, never
   rewritten in place. Verify `find_foreign_identities` only reports.
3. **Adaptation lanes must never be redacted.** On `shakespeare` and
   `public_domain` the roster IS the source's cast, so
   `superseded_identities()` must return `[]` and those prompts must stay
   byte-identical. Fidelity to the source is the point of those lanes.
4. **A row must never be flagged for its own name.** Local models routinely
   repeat the assigned name in the subject head (`"30s, Rick Steiner, lead
   pilot"`). A detector that fires on the correct case is worse than none.
5. **Distinct placeholder per identity.** Both r4 reviewers proposed a single
   shared token (`"this character"`); the driver REJECTED that because the brief
   describes several people and collapsing them invites a blended description.
   Verify the labels stay distinct and the brief keeps its substance.

### SPECIFIC THINGS TO ATTACK

* **The empty-context trap.** `_build_user_prompt` (`_otr_casting.py:~351`)
  falls back to the raw `news_seed` slice when the brief is empty. If
  reconciliation can ever empty a brief that had content, that fallback could
  reinstate the names on a lane whose seed carries them. There is a guard for
  this -- **decide whether it actually closes the hole, on every lane.**
* **Is the roster really final where reconciliation runs?** The claim is that
  `assemble_pre_locked_rows` + `precompute_ensemble_slots` freeze every name
  before the description loop. r4 already caught that
  `_apply_llm_slot_fill` (`~:1888`) renames rows AFTER descriptions in
  `OTR_NAME_MODE=llm_slot_fill`. That mode is now RECORDED as unfenced rather
  than fixed -- **is recording it enough, or is it a latent wrong-person path?**
* **`speech_signature`.** It is a second model-owned prose field and it is
  demonstrably contaminated in the archive. Verify BOTH fields are reconciled,
  checked, AND replaced on the floor -- not just `character_description`.
* **The deterministic floor.** BUG-098 was one generic fallback producing ONE
  portrait for an entire cast. `_deterministic_identity_floor` weaves in the
  slot's role/timbre/face-pressure to stay distinguishable. **Is it actually
  distinct across a realistic 3-5 row ensemble, or does it collapse?**
* **The clean-room retry.** It passes `news_seed=""`, `prior_cast=[]`,
  `casting_brief=""`, `max_attempts=1`. Confirm that is genuinely a different
  lever rather than a resample, and that `max_attempts=1` is really honoured
  down the `structured_call` ladder.
* **The census instrument.** It is the evidence base for every number quoted
  above. Check: dedupe is computed INSIDE the tool (an earlier version quoted an
  externally-computed figure that could not be reproduced from it); `os.walk`
  errors raise rather than silently shrinking the corpus; one row with several
  intruders counts once; matching has lexical boundaries. **Confirm exit code 2
  really means "incomplete scan", not "clean".**
* **Byte-identity.** Any lane that supplies no upstream identities must produce
  an unchanged prompt. Verify `upstream_identity_names=None` is fully inert.

### KNOWN, DELIBERATE GAPS -- do not report these as new findings

* `media_archive` has the identical defect (verified: `ADRIAN CARRUTHERS`
  carrying *"Dr. Amelia Hartley"*) but names its people only in free-text prose.
  Harvesting names from that prose was MEASURED and rejected -- it fires on the
  healthy Title-Case occupation head (`"30s, skeptical Film Historian"`). It is
  filed as its own item.
* `scifi_news_pro` never calls `lock_cast` (it derives its cast from the written
  script), so it is out of reach by construction.
* The 46 historical contaminated rows are NOT repaired by this change. Stopping
  the bleeding and back-repairing the archive are separate jobs.

### WHAT I ALREADY PROVED -- re-check it rather than take my word

* Full suite green before the change: **11237 passed / 114 skipped / 1 xfailed**.
* The new tests are NOT tautological, proved by breaking the code: neutering
  layer 1 (raw brief to the prompt) turns
  `test_verify6_raw_brief_never_reaches_the_prompt_builder` RED; neutering
  layer 3 (the floor) turns
  `test_a_generator_that_always_contaminates_still_yields_a_full_cast` RED. Both
  files were restored and confirmed byte-identical by SHA-256.
* `tests/test_b7_forbidden_sweep.py` caught a real regression in the first
  draft (the identifier `alias` is an S28 extinction marker); renamed and green.
* A Sonnet 4.6 QA pass already ran against THIS prompt; its findings and the two
  the driver REJECTED (with reasons) are in
  `kibitz-runs/2026-08-20-item-I-wrong-person/r4/sonnet_qa_disposition.md`.
  Read it, then look for what it MISSED -- do not simply re-derive it.

### OUTPUT I WANT

1. **MUST-FIX** -- anything that breaks a hard constraint above, can raise out of
   the guard, can fail an episode, or is simply wrong. Cite `file:line`.
2. **SHOULD-FIX** -- correctness or clarity issues that are real but not
   blocking.
3. **TAUTOLOGY CHECK** -- name any test that would still pass with the mechanism
   it claims to test removed. Be specific about which line makes it vacuous.
4. **A DIRECT ANSWER** to: *does this actually change the shipped output, or is
   it a scorer that flags and changes nothing?* Cite the code path.

---

## ROUND 4 -- FOUR PASSES HAVE RUN. EVERY ONE FOUND SOMETHING REAL.

Dispositions in `kibitz-runs/2026-08-20-item-I-wrong-person/r4/`:
`sonnet_qa_disposition.md`, `sol_qa_disposition.md`,
`agy_qa_round3_disposition.md`. **Read them. Re-reporting a fixed defect costs
the round.**

Assume there is still something. The pattern so far: the crashes were easy, and
the expensive defects were all cases where **the guard quietly made output worse
or a check silently did nothing.**

### FIXED SINCE ROUND 3

| defect | fix |
|---|---|
| A legitimate MENTION appearing first SHADOWED a real claim behind it (`.search` returns one match) -- *"foil to Jonas's obsession. But I am Jonas!"* escaped entirely | `finditer` + `any(...)`; every occurrence examined |
| **The possessive check contained a literal backspace byte (0x08)** -- a `` mangled by a shell heredoc -- so it had NEVER fired, and it fails OPEN (flags healthy prose) | pattern repaired; a repo-wide control-character scan is now a test |
| `test_a_row_is_never_flagged_for_its_own_name` was vacuous (asserted `find(..., []) == []`) | detector now drops roster-owned identities ITSELF; the test passes the identity directly |
| The clean-room retry stripped ALL story context, so a wrong-person row became a story-detached one -- and its portrait was painted from generic filler | the retry keeps the RECONCILED brief and seed (the names are already gone from them); only `prior_cast` is dropped |
| The retry referenced an out-of-scope variable -- `NameError` on every guard fire, silently forcing the floor instead of a regeneration | fixed; caught only because the never-raises handler swallowed it |

### CURRENT MEASUREMENT

**68 occurrences in 40 files; 64 unique rows / 38 durable episodes; 40 rows / 26
REAL episodes; 65 of 68 also in the portrait prompt.** Exit 1. The figure has
fallen 70 -> 69 -> 68 across two rounds, each step removing rows that were never
contaminated.

### STILL OPEN AND DELIBERATE

`llm_slot_fill` ordering (env-gated, recorded); bench classification by path
substring; `media_archive` (no structured identities); `scifi_news_pro` (never
calls `lock_cast`); the 40 historical rows (not back-repaired).

### WHAT I WANT

1. **MUST-FIX** in the CURRENT tree, `file:line`.
2. **THE SILENT-NO-OP HUNT.** One check in this diff had never executed once.
   Find another: a regex that cannot match, a branch that cannot be reached, a
   guard whose condition is always false, a test whose assertion is unreachable.
   Grep for control characters, over-escaped patterns, and conditions that
   depend on something always empty.
3. **THE HARM HUNT.** Another input where the guard makes output WORSE than
   doing nothing. Three rounds found one each.
4. **TAUTOLOGY CHECK** on all 33 tests. Every round has found exactly one.
5. **ONE QUESTION:** `_ORDINARY_WORDS` is a hand-written denylist of ~60 words.
   What happens when a real character is named with a word that is NOT on it --
   or when a legitimate description uses a word that IS? Is a denylist the right
   mechanism here at all, or is it a list that will rot? Answer with the code
   and say what you would use instead.
