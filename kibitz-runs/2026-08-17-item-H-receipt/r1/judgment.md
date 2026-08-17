# Judgment -- item H, r1 (Claude driving, sole judge)

**Provenance, stated precisely.** r1 ONLY; r2/r3/r4 NOT run. Panel: **two
Antigravity calls** -- `Gemini 3.5 Flash (High)` (kibitz default, in
`2026-08-17-item-H-receipt/`) and `Gemini 3.1 Pro (High)` (the operator's actual
request, in `2026-08-17-item-H-receipt-pro/`) -- plus a **Fable** judgment pass on
the queue-order question spawned outside kibitz, plus this driver's anchor.
**Codex excluded: quota-held until 2026-08-19 20:31.** See `../scope_receipt.md`.

## ACCEPTED

**A1 -- my anchor's reordering claim was FALSE. Both agy lanes caught it
independently; I verified it myself.** CONFIRMED at the file: `_neg_source` is
computed at `otr_image_gen_dispatcher.py:1166`, `engine_id` is bound at `:1225`,
and `negative_source` is WRITTEN at `:1413` and `:1608` -- both AFTER engine
resolution. Pro sharpened it further and that too is CONFIRMED: `_neg_source` is
**write-only telemetry**, three references in the whole file, and it is decoupled
from `prompt_hash` and the banana transform. So there was never a "production
reordering in a loop that also computes cache keys, seeds and the banana
transform". I conflated where the string is COMPUTED with where it is USED.
Recorded as a driver error, not softened: this is the SECOND time in one day a
panel corrected an execution-order claim of mine (GO_FORWARD records the same for
the 2026-08-17 style build).

**A2 -- Option B is rejected. Unanimous 4/4** (anchor, Fable, Flash, Pro). Giving
`lumina_image` a hygiene floor changes conditioning on a live engine at cfg 4.0,
owes a render under the standing trap, and the recipes are not on the table. The
engine's own inline comment already defers it. Parked as an operator decision memo
with three options (no floor / copy z_image's / a lumina-specific string).

**A3 -- the rename is SAFE, and Fable's timing argument inverts my caution.**
CONFIRMED: `negative_source` has zero readers (written at two sites, mentioned in
one comment, that is the entire repo), zero tests, and zero on-disk rows across
4,795 JSON files under the real output base. Fable's decisive point: the field's
FIRST real population will be the operator's declared batch GPU session, so
leaving `engine_hygiene` in place means the first ledgers ever written are born
asserting a floor lumina does not have. Waiting does not preserve safety, it
guarantees the lie gets minted.

**A4 -- the enum already drifted once with zero consequence. CONFIRMED (Fable).**
`docs/2026-08-17-one-style-authority-PLAN.md:244` documents the values as
`pack | pack+request | env_override`, while what shipped is
`pack+request | pack | request | engine_hygiene` -- a value that never existed and
two that do, missing. Direct evidence nothing depends on this vocabulary. Fixed in
the same change.

## REJECTED, with reason

**R1 -- BOTH agy lanes' shared conclusion: make `negative_source` engine-aware
instead of renaming it. REJECTED.** Their FACTS are right (it is feasible, no
reordering needed) and I have folded those. Their CONCLUSION re-commits the
original defect: this field is named for COMPOSITION (`pack`, `request`) and the
bug is precisely that its fourth arm asserts ENGINE BEHAVIOUR. Making it more
engine-aware puts two authorities in one value, which is the shape that produced
the lie. Keep the composition field describing composition; record engine
behaviour in a SEPARATE post-resolution field if it is wanted. Fable reached the
same place from the other side: a name describing what the dispatcher KNOWS stays
correct even if lumina later gains a floor, so there is no second rename.

Second, decisive reason: the rename **dissolves** the ordering coupling rather
than working around it. Once the value no longer claims engine behaviour, where it
is computed stops mattering at all -- so the diff is smaller AND the class of bug
is gone, instead of being re-created one field over.

**R2 -- Antigravity Flash's suggested MECHANISM (`matching "z_image_turbo"`).
REJECTED outright.** That is a NAME MATCH, and item A's ruling forbids exactly it:
*"CHECK THE TOKENIZER, NOT THE NODE NAME... Name-matching would have shipped TWO
FALSE POSITIVES (`z_image_turbo`, `flux_gen1`)."* Pro's alternative phrasing
("query the engine") is the correct shape. If the follow-on is ever built, engines
DECLARE a floor and the dispatcher reads the declaration, with a dual-read default
for engines that have not declared -- the precedent is `engine_consumes_still`.

## VERIFY-AT-BUILD / DEFERRED

**V1 -- the whole `visual` ledger section has never been written to disk.** Both
Flash and my anchor reached this; CONFIRMED by scan (`negative_source`,
`self_veto_resolved`, `_style_spread` all absent together while `visual_style`
appears in 770 files and `prompt_hash` in 1,022). This upgrades D-BIS finding 5
from "no tests" to "no tests AND no live observation". The cheapest real close is
one render that writes the section at all -- goes into the operator's declared
batch GPU session, not into this change.

**V2 -- D-BIS finding 4 (record the cfg or a `negative_live` bool).** Both agy
lanes independently proposed it. It is genuinely adjacent and arguably more useful
than naming provenance, because at cfg 1.0 a logged negative conditioned nothing.
NOT folded here: it adds a field, and adding ledger fields while the operator is
away is the thing I am declining to do. Recorded as the natural next step.

**V3 -- Pro's `engine_effective_negative` idea.** Interesting but rests on its own
stated assumption that engines can expose their applied negative post-generation.
Unverified. Left as a note.

## What ships from r1

The rename, its comment, the two stale comment sites in `lumina_image`, the stale
enum in the one-style-authority PLAN doc, item H's body in GO_FORWARD, and a test
pinning all four arms. Zero pixel change, no recipe touched, no new ledger field.
