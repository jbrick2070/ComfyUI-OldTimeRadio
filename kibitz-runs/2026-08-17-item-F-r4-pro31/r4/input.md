# r3 judgment -- item F wiring, and the round that caught a real build-breaker

**Driver:** Claude (Cowork), panelist and sole judge. **Date:** 2026-08-17.

**Lanes:** `Gemini 3.7 Flash (High)`
(`kibitz-runs/2026-08-17-item-F-r3-flash37/r3/`) and `Gemini 3.1 Pro (High)`
(`kibitz-runs/2026-08-17-item-F-r3-pro31-t15/r3/`), plus a **Sonnet 5 QA pass on
the finished diff** (operator's standing rule, requested again this session).
Codex quota-held to 2026-08-19. **r1+r2+r3 of four.**

> **THE PRO LANE TOOK THREE ATTEMPTS AND THE FIRST TWO FAILURES WERE MINE TO
> READ.** Attempts 1 and 2 died with `Error: timeout waiting for response`.
> `kibitz.py --timeout` does NOT reach agy: the CLI flag is built from
> `AGY_PRINT_TIMEOUT`, which reads **`KIBITZ_AGY_PRINT_TIMEOUT`** and defaults
> to `5m`. Attempt 3 with `KIBITZ_AGY_PRINT_TIMEOUT=15m` landed first try. **Pro
> on a long doc needs the env var, not the flag** -- and the failure says
> "timeout", not "quota", so do not read it as a quota block: `agy models`
> returned rc=0 throughout.

---

## THE PANEL CAUGHT A BUILD-BREAKER THAT SONNET CLEARED

**Pro MUST-FIX 1 is CORRECT, Sonnet's category-3 clearance was WRONG, and the
corpus settled it.**

* **The driver's claim** (in the r2 judgment and the code comments): the four
  non-adaptation banks yield `work_title == ""`, so `WORK` self-omits.
* **Sonnet said** media_archive is empty in practice -- it read
  `_otr_media_archive_interpreter` and found nothing populating `source_label`.
* **Pro said** `identity_from_meta` explicitly maps media_archive's
  `source_label` onto `work_title`, so the lane WOULD get a populated title.
* **The measurement, run over the live corpus:** of **98 media_archive ledgers
  on disk, 56 carry a `source_label`** -- and the first example is literally
  **`"Now See Hear!"`**, the exact string Pro predicted.

So the shipped behaviour would have been `WORK: a scene from Now See Hear!` in
the announcer prompt and `Adapted work: Now See Hear! -- the setting must belong
to this work and to no other.` in the outline prompt, on **57% of a live lane**.
**That invents a play, which is a worse fidelity defect than the wrong-play
frame this item was opened to fix.**

**Why Sonnet missed it and Pro did not:** Sonnet reasoned from the producer
(which module writes the field) and concluded the path was dead; Pro reasoned
from the consumer (`identity_from_meta`'s own mapping). Neither is a substitute
for the artifact. **This is the repo's own rule arriving again -- a static audit
cannot prove behaviour.** A green suite would not have caught it either: no test
constructed a media_archive meta with a `source_label`.

**THE ROOT SHAPE, worth more than the fix:** `work_title` carries TWO MEANINGS
in one field -- the work being PERFORMED on the adaptation lanes, and the
PUBLICATION a post came from on media_archive. That is the same
two-authorities-in-one-value shape as the `_neg_source` lie (H-receipt) and as
PBUG-20260817-03's two naming authorities. **A consumer that means "the work
this episode performs" must gate on the LANE, never on whether the value is
non-empty.**

**Adopted, with the gate made a named constant rather than an inline tuple:**
`_otr_source_identity.ADAPTATION_SOURCE_KINDS`, a frozenset of the two
performing lanes, with the measurement recorded beside it. The writer gates
`_work_title` on it. Three regression tests pin it, including a POSITIVE control
(a shakespeare meta must still qualify) so the gate cannot be satisfied by a
predicate that is false for everything.

## SONNET'S OWN CATCH, and it was the one that mattered procedurally

**`tests/test_cross_play_frame_leak.py` was UNTRACKED.** It holds the sharpest
pin in the change -- the only test that drives the real composer against the
real manifest and asserts on the literal shipped-defect string. `git status`
showed it as `??` and a commit staging only the modified files would have left
it local-only, making the green suite unreproducible from origin. `git
check-ignore` confirms it is not ignored, merely never added. **This is the
untracked-artifact trap for the third time in one day** (`scripts/_*.py`,
`kibitz-runs/`, now a plain new test) -- the class is "new file, green locally,
absent on origin", and the only reliable guard is checking `git status` before
the commit rather than trusting `git add <dir>`.

## FLASH r3 -- one stale finding, three live ones, all verified

* **MUST-FIX 1 (the leak detector fails) is STALE, not wrong.** Flash read the
  tree mid-edit, before the detector was switched to stem matching plus
  vendored-text terms. **Lesson, and it generalizes the suite rule: do not edit
  tracked files while a PANEL is reading them either.** Its underlying point --
  possessive/plural mismatch -- was real and is now handled by `_leaks`.
* **MUST-FIX 2 UPHELD:** `fallback_safe_open` still had two
  `getattr(brief, ..., "")` survivors sitting beside a direct `work_title`
  read. Converted; the doctrine is now consistent in that function.
* **SHOULD-FIX 1 UPHELD:** `identity_from_meta(meta)` was being evaluated
  twice. Bound once, reused.
* **SHOULD-FIX 2 UPHELD:** raw `.strip()` on the title would split the prompt
  line if a title ever carried a newline. Collapsed inline rather than importing
  the composer's `clean_one_line` -- a new cross-module import on the outline
  hot path buys an import-cycle risk for one line of normalization.
  **This introduced a `NameError` for one edit cycle** (I wrote
  `clean_one_line(...)` into a module that does not import it) and was caught by
  running the builders directly before any commit.
* **OPTIONAL 2 is the r3 ANSWER and is confirmed independently:** the diff adds
  no `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES` or widget lines, so
  **`workflows/otr_canonical.json` needs no migration.** Verified by grepping
  the diff for node-surface declarations.

## PRO'S OPTIONAL 1, DEFERRED WITH A REASON

Pro notes `fallback_safe_open` re-implements the "a scene from" phrasing inline
instead of reusing `_work_line`. Correct in principle, and NOT taken here: the
two strings are deliberately different shapes -- the prompt label is
`a scene from X` (a context value) and the fallback is a full spoken sentence,
`Tonight, a scene from X.` Collapsing them into one helper would make the helper
serve a prompt label and a spoken line at once, which is the two-meanings shape
this very round is fixing. The shared piece that CAN drift is the phrase itself,
and it is already a module constant, `WORK_LINE_PREFIX`.

## DISPOSITION

Adopted from r3: the media_archive lane gate (**build-breaker**), the untracked
test staging (**would have lost the best pin**), consistent direct attribute
access, single identity binding, title sanitation, and the confirmation that no
workflow migration is owed. Deferred: the fallback/helper merge, with the reason
recorded above.

**Still true, and unchanged by three rounds:** no unit result may be called "the
wrong-play frame is fixed". The suite proves the title reaches both producers and
that no cross-play name appears in the captured prompts. **"Fixed" needs the live
leg**, batched into the operator's GPU session.

## r4 SCOPE

Convergence only: confirm no new must-fix against the finished, pushed diff, with
Codex included from 2026-08-19.
