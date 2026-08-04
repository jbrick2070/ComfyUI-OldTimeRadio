# Consistency review: saved widget values vs live choice lists

Target for an independent panel. HEAD `1e7d8aa4`, branch `v2.0-alpha`.

## The defect that prompted this

The operator noticed the writer's model dropdown rendering RED when opening
`workflows/otr_canonical.json`. It was a real binding fault, not a UI quirk:

* the graph saved `google/gemma-4-12b-it` for `creative_writing_model` and
  `technical_model` on node 1;
* the live COMBO offers `google/gemma-4-12b-it (11.9 GB)` -- the picker gained a
  VRAM badge (operator ruling 2026-08-01: "all dropdowns should state the name
  of the model and how much VRAM it needs") and the saved graph never followed;
* index 0 of that choice list is `mistralai/Mistral-Nemo-Instruct-2407`, so the
  project's declared source of truth named one writer and could have run
  another.

Fixed by replacing the two values in place. No insert, remove or reorder, so no
widget slot moves (BUG-LOCAL-097).

## What the fix exposed, which is the more interesting half

`test_default_workflow_only_binds_mit_equivalent_rows_to_creative_slot` is a
LICENCE guard: no default-shipped workflow may bind a catalog row whose
`license_audit_status != "mit_equivalent"`. Its helper matched widget values
against curated `repo_id`s by EXACT string, while the runtime strips the badge
via `catalog._strip_label_suffix`. So the moment a value carried its correct
badge, the helper matched nothing, returned an empty binding list, and the
licence guard had nothing left to check. It failed only because someone had
written `assert bindings, "cannot verify the guardrail applies to anything"`.

The helper now normalizes the way the runtime does.

## THE QUESTIONS FOR THE PANEL

**Q1. Is this defect class present anywhere else?** A saved value that the live
server no longer offers is invisible until someone opens the graph and notices a
colour. Audit every saved COMBO in `workflows/otr_canonical.json` against a live
`/object_info`. I found exactly two unmatched and believe the rest are clean --
attack that. Note my FIRST audit produced four additional "findings" that were
artifacts of naive indexing (it did not skip non-combo widgets, so its cursor
slid); a second pass asking "is this value legal for some OTHER combo on this
node?" cleared nodes 12, 80 and 87 entirely. If you reproduce the first result,
reproduce the second before believing it.

**Q2. Where else does an EXACT-string comparison meet a value the runtime
normalizes?** This is the general shape of the licence-guard failure: two
layers, one of which strips a decoration. Find other places -- tests, validators,
profile resolution, capability checks, engine registries -- that compare a
user-facing label to an internal id without normalizing. Each is a guard that
silently stops guarding.

**Q3. Can a badge change break a binding again, and would anything catch it?**
The badge is computed from measured VRAM (`vram_badge_for`). If a measurement
changes, the label changes, and every saved graph binding that model goes stale.
Is there a mechanism that would fail loudly, or does it depend on a human
noticing a colour? If the latter, propose the cheapest durable check.

**Q4. Is the two-baseline-test pin the right response?** Two tests now assert
the badged value in full. That pins today's number. Argue for or against
asserting the badge instead via `_strip_label_suffix` equality, which would
survive a VRAM re-measurement but would no longer catch a badge going missing.

**Q5. Anything unsound in the surrounding work?** Same HEAD also carries: a
public-domain library grown from 1 to 65 sources with a random unit selector; a
provenance change moving licence text out of the spoken line into the printed
credit plus a non-commercial notice; and D1 observability on a silently skipped
image still. Flag anything inconsistent ACROSS those, especially shared
assumptions that one change made true and another still assumes false.

## Invariants a proposal must not break

* `workflows/otr_canonical.json` is the source of truth; any node/widget/link
  change lands IN it, in the same change as the code.
* `widgets_values` is POSITIONAL. Only ever append at the end.
* The licence guard must keep guarding; do not weaken it to make a test pass.
* No fallbacks; the image completion gate stays fail-closed.
* THE LAW: an audit may improve a story, never fail one for length, language,
  style or quality.
