# PROBLEM STATEMENT -- claims that are not performed

**Date:** 2026-08-08
**Branch / HEAD at framing:** `v2.0-alpha` @ `bf9f7fb1` (== origin)
**Baselines:** suite 9486 / 111 / 1; Bible `7a5fb88` 262 entries / 370 rows;
`build_variants --check` 45 / 0.

**Files this chunk may touch:** `scripts/validate_canonical_workflow.py`,
`nodes/_otr_comfy_backend.py`, `nodes/_otr_openrouter_backend.py`,
`tests/_helpers.py`, plus new/edited tests.
**Files this chunk MUST NOT touch** (concurrent Codex window owns LEMMY):
`config/cast_pools.py`, `nodes/_otr_dialogue_policy.py`,
`nodes/_otr_line_composer.py`, `nodes/_otr_compose_exchange.py`,
`nodes/production_ledger.py`, `nodes/_otr_voice_node_common.py`.

---

## 0. THE THEME, AND WHY THESE FOUR BELONG IN ONE CHUNK

Bible entry **BUG-12.86** was promoted 2026-08-07 for exactly this defect class:
*a receipt or prompt-context field keyed on a producer string the producer never
emits, so it reads empty/False forever.* Four instances were found in one
afternoon.

These four are the same shape one level up: a **gate**, a **comment**, a
**captured field** and a **test helper** that each announce a behaviour the code
does not perform. None is a crash. Each one lies to a reader, and two of them
lie to an automated gate, which is worse -- a green that was never earned is
more dangerous than a red.

They are batched because the fix is the same judgement every time: make the
claim true, or delete the claim. Not because they touch the same code.

## 1. A GATE THAT CAN PASS WITHOUT CHECKING (the important one)

`scripts/validate_canonical_workflow.py:105-114`. The block resolving
`NODE_CLASS_MAPPINGS` is wrapped in `except Exception`, which prints
`[validate_canonical_workflow] SKIPPED validate_workflow_contract (...)` to
stderr and then `return []`. An empty list means "no problems found", so the
caller adds nothing to its problem count and the script **exits 0**.

Why it matters beyond tidiness: this script is named as an acceptance gate in
`docs/GO_FORWARD_PLAN.md` (the queue item 8 tombstone) and was used as one
during that ship. Anyone reading its exit code is reading a skip as a pass. The
skip is not hypothetical -- it fires whenever the OTR package cannot import,
which is precisely the condition under which you most want the check to run.

Fix direction (attack this): return a real problem entry instead of `[]`, OR add
`--strict` that converts SKIPPED into a nonzero exit and use that flag wherever
the script gates. Audit the same file for any other early-return-empty path with
the same shape.

**Open question for the panel:** the docstring at `:91-93` says the skip is
DELIBERATE -- "skip this check with a warning line rather than fail -- the
link/widget/deleted-type audits below cover the structural invariants this
script guarantees." Is that reasoning sound? If the remaining audits genuinely
cover the guarantee, the defect is the SCRIPT'S ADVERTISED SCOPE, not its exit
code, and the honest fix is to stop calling it a contract validator. Decide
which, do not split the difference.

## 2. A COMMENT WHOSE EXCLUSION RULE MAY OR MAY NOT STILL HOLD -- READ THIS ONE CAREFULLY

`nodes/_otr_comfy_backend.py:84-91`. **The GO_FORWARD summary of this item is
itself suspect, and re-reading the actual lines is why.** GO_FORWARD (STILL OPEN
item 6) says the comment "claims an exclusion the list no longer performs",
citing `deepseek/deepseek-v3.2` and `x-ai/grok-4.20` as reasoning models sitting
inside the list.

What the code actually says (`:84-86`): *"CURATED (operator directive
2026-07-04): ONE recognizable, NON-REASONING model per major brand, ordered
cheapest -> premium. Reasoning models (deepseek-\*-pro, perplexity
sonar-reasoning/deep-research, \*-thinking, gpt-5.5-pro, qwen \*-max) ..."*

That is an exclusion by **SKU PATTERN**, not by capability. `deepseek-v3.2` is
not `deepseek-*-pro`; `grok-4.20` is not `*-thinking`. The list even labels its
own entry `"deepseek/deepseek-v3.2",  # DeepSeek -- cheap general chat
(NON-reasoning)`. So on its own stated terms the list may still be performing
exactly the rule it describes, and GO_FORWARD's "the comment lies" framing
conflates *"this brand's non-reasoning SKU"* with *"a model that cannot reason"*.

**The real question for the panel, then, is which of these it is:**
(a) the rule is intact and only the parenthetical needs a note that most modern
SKUs expose reasoning even when they are not reasoning-branded; (b) the rule was
always about capability, in which case it genuinely no longer holds; or (c) the
rule is now meaningless because the SKU distinction has collapsed industry-wide,
and the honest move is to delete the rule rather than restate it.

Do not "fix" this by deleting the comment before answering that. Already
established and NOT to be re-derived (GO_FORWARD item 6): all six slugs are
LIVE against the OpenRouter catalog; the list is VERSION-BEHIND, not broken;
and Comfy Cloud's partner catalog -- not OpenRouter -- is the authority for
whether Comfy actually SERVES a slug.

Re-dating the slugs is the mechanical half and is NOT in scope here.

Already established, do not re-derive (`GO_FORWARD_PLAN.md`, STILL OPEN item 6):
all six slugs are LIVE -- checked against the live OpenRouter catalog. The list
is VERSION-BEHIND, not broken, and that distinction must not be lost again.
Comfy Cloud's partner catalog is the real authority for whether Comfy SERVES a
slug; OpenRouter presence is a signal, not proof.

This chunk fixes the COMMENT -- the cheap, high-value half. Re-dating the slugs
is the mechanical half and is NOT in scope here.

## 3. A FIELD CAPTURED AND NEVER READ

`nodes/_otr_openrouter_backend.py:918` slims `reasoning.default_enabled` into
every cached catalog row. Only `mandatory` and `supported_efforts` are ever
consulted (`:324`, `:330`). The field reads as though it informs a decision and
informs nothing -- the sibling of BUG-12.86, found while grounding the slug
curation and PRE-EXISTING, not caused there.

Per the admission rule a static observation does NOT create a PBUG. Delete-or-
consume, and say which and why in the code.

## 4. DEAD TEST INFRASTRUCTURE

`tests/_helpers.py:26-118` -- `load_all_ledger_fixtures` /
`_looks_like_l3_ledger`. No callers anywhere, and none of the 5 JSON fixtures
match its `l3-` prefix filter, so even if it were called it would return
nothing. Delete-or-revive.

---

## 5. WHAT "DONE" LOOKS LIKE

* Each of the four is either made TRUE or DELETED, with the reason recorded in
  the code, never left half-stated.
* Suite at or above **9486 / 111 / 1**, zero failures; Bible 17/24/3 at
  `7a5fb88`; `build_variants --check` 45 / 0.
* `git diff <BUILD_START_HEAD literal>..HEAD -- workflows/otr_canonical.json`
  EMPTY. (A bare `git diff -- workflows/` after committing is VACUOUS.)
* Gate integrity gets a TEST: the validator's skip path must be proven to fail
  loudly, by forcing the import failure -- not asserted in prose.
* Commit granularity is the panel's call to challenge: item 1 is behavioural and
  wants its own commit; 2-4 are plausibly one "delete the claim" commit. Each
  commit gated and PUSHED on its own (CLAUDE.md section 7).
* Sonnet 5 QA on each diff; the Fable final gate on the behavioural one.

## 6. THE TRAP THIS CHUNK MUST AVOID

Deleting a claim is only correct when the behaviour is genuinely unwanted. Twice
in the last two chunks a "stale" thing turned out to be deliberate -- the
post-cap style cue (closed as INTENDED, `aae732f7`) and the banana gate's
fail-open on a missing `source_bank` (specified in three places). Before
deleting any of these four, find the commit that introduced it and say what it
was for. If it was deliberate and still wanted, the fix is to make the claim
accurate, not to remove the code.
