# My Story D0 -- documentation currentization receipt

Date: 2026-09-10. Sprint 3, first half. Base HEAD `75bb405b` == origin,
branch `v2.0-alpha`. Documentation only: no runtime file, no workflow JSON, no
test fixture and no registry row changed in this chunk.

## What was done

Seven documents were audited against the live Windows files and the operator's
standing rulings, and every surviving finding was applied. The rule throughout:
current behaviour is described as current, dated history stays history, the
proposed `my_story` bank is described only as PROPOSED, and no content filter,
word target or prose-quality instruction was reintroduced.

| Document | Edits applied |
|---|---|
| `README.md` | 4 |
| `docs/EXTENDING_OTR.md` | 12 |
| `docs/SOURCE_BANK_GUIDE.md` | 17 |
| `docs/SOURCE_BANK_PREFLIGHT.md` | 21 |
| `docs/PRODUCTION_SPRINT_LESSONS.md` | 13 |
| `docs/SOAK_LEG_GUIDE.md` + `docs/README.md` | 12 |
| **total** | **79** |

## Method, and what it actually proves

Two independent passes per document, both grounded on the real Windows files
(never the Linux mount): a reader produced findings with file:line evidence and
replacement prose, then a SECOND reader tried to REFUTE each one by re-opening
the document at the cited lines and re-checking the cited code.

* 73 findings CONFIRMED, 13 AMENDED (the verifier corrected the evidence or the
  replacement), **0 REFUTED**, and the verifiers found **16 MISSED** stale
  claims the first pass had walked past. All 79 survivors were applied.
* The AMENDED ones are the reason the second pass earned its keep. Two examples:
  a finding about `docs/multimodal-story-schema/` attributed `target_words` and
  a six-bank roster to plans that never mention either (a grep collision), and a
  node-88 widget list was corrected to the real eight widgets and the real
  consumer relationship.
* Roughly a third of the corrections were the same four defect classes:
  the retired `story_rules` file, `target_words` and the 30/120/720-word ladder,
  content-filtering language that contradicts the 2026-08-03 directive, and
  "six banks" where five are runnable plus one signpost row.

**This is a static documentation comparison against live definitions. It is not
a bank preflight pass, and it proves nothing about runtime behaviour.**

## Verification performed

* All 79 edits confirmed present in the files (4 of them adapted: three
  re-wrapped to the surrounding paragraph width and re-pointed as a preceding
  supersession note rather than an in-place rewrite, one instruction prefix
  stripped). No edit was skipped.
* Encoding: every touched file is UTF-8 with **no BOM**, decodes clean, and
  carries no mojibake. Exactly one non-ASCII character was ADDED across the
  whole change -- a `§` in the soak guide's act-count table row, matching the
  `see §2` already in the row above it.
* Markdown: all code fences balanced; no ragged table in any touched file.
* Focused tests, the four suites that read these documents:
  `test_image_gen_preflight_matrix.py`, `test_multiclip_session_identity_roster.py`,
  `test_pbug_20260710_07_cast_keyed_mutation.py`, `test_source_bank_widget_2c.py`
  -- **129 passed, 54 skipped**.

## Limits, stated plainly

* No live render, no server, no GPU leg and no published episode was involved.
* Line numbers cited inside the edited prose were true at HEAD `75bb405b` and
  drift with the code; several edits say so and tell the reader to re-grep.
* `nodes/story_packs/banks.json` still carries the stale
  `custom_source_bank.guide_ref` wording ("equal to the shipped six"). That is
  runtime JSON, deliberately OUT of a Markdown-only chunk; it is corrected in
  the registration chunk (sprint 4), where the file is already being edited.
* The reviewers ran as internal subagents. Partway through the apply pass the
  subagent lane hit its monthly spend limit and stopped; the remaining 34 edits
  were applied by hand in the driver window and verified by the same checks.
  No claim is made that an external CLI lane reviewed this documentation chunk.
