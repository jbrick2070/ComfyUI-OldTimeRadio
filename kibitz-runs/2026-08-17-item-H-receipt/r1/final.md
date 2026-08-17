# Item H, hardened -- the plan forward after r1

**Scope reminder: r1 ONLY. r2/r3/r4 not run.** Two Antigravity calls (Flash High +
Pro 3.1), one Fable pass, one driver anchor. Codex excluded on quota. See
`../scope_receipt.md` and `judgment.md`.

## H splits, formally

**H-RECEIPT -- SHIPPED in this change.** Driver-sized, zero pixels, no recipe.
**H-FLOOR -- PARKED for the operator.** Needs a render; not a driver decision.

## What shipped (H-receipt)

1. **`negative_source_label(pack_negative, obj_negative)`** -- a new pure
   module-level helper in `otr_image_gen_dispatcher.py`, the ONE place that names
   this field. The inline ternary is gone. Extracted deliberately: an untestable
   inline expression is what let the wrong value ship, so the root fix is to give
   the four arms a named, unit-testable owner.
2. **The empty arm is now `none_contributed`**, not `engine_hygiene`. It describes
   COMPOSITION, which is all the dispatcher can verify from its own two inputs.
3. **`NEGATIVE_SOURCE_LABELS`** exported, so a test pins the vocabulary instead of
   re-typing it.
4. **Five tests** in `tests/test_visual_style_negative.py::TestNegativeSourceLabel`:
   all four arms, the empty arm asserts nothing about any engine, the vocabulary is
   closed and matches what the function can produce, the signature is
   engine-blind by construction, and falsy shapes are treated as absent.
   (Counted from the suite delta, 10819 -> 10824, not from memory -- the first
   draft of this file said six.)
5. **Stale references corrected in the same change:** the dispatcher call-site
   comment, two comment sites in `lumina_image.py`, and the enum in
   `docs/2026-08-17-one-style-authority-PLAN.md` -- which documented
   `env_override` (never shipped) and omitted `request` and the empty case.

**Why a rename rather than the panel's engine-aware "third option":** both agy
lanes proved the engine-aware version is FEASIBLE (no reordering -- `_neg_source`
is write-only telemetry, decoupled from `prompt_hash` and the banana transform)
and I folded that fact and corrected my anchor. But their conclusion re-commits the
original defect: a field named for composition asserting engine behaviour is TWO
AUTHORITIES IN ONE VALUE, which is the shape that produced the lie. The rename
also DISSOLVES the ordering coupling instead of working around it -- the answer no
longer depends on the engine, so where it is computed stops mattering.

## H-FLOOR -- the operator's decision, three options

Giving `lumina_image` a hygiene floor changes conditioning at cfg 4.0 on a live
engine. The recipes are not on the table, and green gates are not a working fix, so
this owes a render. Rejected 4/4 as a driver action; recorded as a choice:

* **(a) No floor.** Accept that an empty request negative reaches the encoder as
  `""`. Now HONESTLY reported -- the receipt no longer claims otherwise, which was
  the only urgent part.
* **(b) Copy z_image's `_HYGIENE_NEGATIVE`.** Cheapest, and the trap: z_image runs
  cfg 2.0, lumina 4.0, different model, different artifact profile. The proximity
  of this option during the receipt edit is exactly why it is written down as a
  decision rather than taken.
* **(c) A lumina-specific string.** Most correct, most work, needs an A/B.

Any of these needs one A/B at a fixed seed on the shipped path before it counts.

## Now known feasible, deliberately NOT built

* **Engine-aware hygiene telemetry as a SEPARATE field.** Feasible and cheap
  (`_neg_source` is write-only; the write sites are after engine resolution).
  Mechanism constraint, non-negotiable: engines DECLARE a floor and the dispatcher
  reads the declaration with a dual-read default (`engine_consumes_still` is the
  precedent). **Never a name match** -- item A's ruling is that name-matching would
  have shipped two false positives.
* **D-BIS finding 4: record the resolved cfg or a `negative_live` bool.** Both agy
  lanes proposed it independently and it is arguably more useful than provenance,
  because at cfg 1.0 a logged negative conditioned nothing. Adds a ledger field, so
  it waits for the operator.

## Owed to the GPU batch, not to code

**The entire `visual` ledger section has never been written to disk.** Confirmed by
scanning 4,795 JSON files under the real output base: `negative_source`,
`self_veto_resolved` and `_style_spread` are all absent together, while
`visual_style` appears in 770 and `prompt_hash` in 1,022. So D-BIS finding 5 is
worse than recorded -- no tests AND no live observation -- and the operator's "lock
them in the ledger" ask is unproven end-to-end. One render that writes the section
at all is the cheapest close, and it now writes the CORRECTED label.
