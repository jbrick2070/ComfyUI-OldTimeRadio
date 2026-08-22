# DORMANT 3D TALKERS -- judgment: RIP, but it is already W2, and one live
# guard must be MIGRATED first

**Run:** `kibitz-runs/2026-07-25-dormant-3d-rip/r1/` (codex `gpt-5.6-sol` high,
pin verified). Baseline HEAD `0bc863f4`. Claude is anchor and judge.
Brief: `docs/2026-07-25-dormant-3d-rip-brief.md`.

**Historical evidence only.** The current target boundary, blast radius, and
execution order live in `docs/LEAN_MEAN_CLEANUP.md`. Literal conclusions below
describe the 2026-07-25 tree and must not be used as a current kill list.

**VERDICT: RIP. The operator's instinct is right and is ALREADY RATIFIED --
this is lean-mean W2, not a new decision. But it must NOT happen in this
window, and it must NOT be a straight delete: a LIVE fail-closed guard is
hiding inside the "dormant" code and has to move first.**

## 1. The decision was already made, in writing, on 2026-07-10

The retired 2026-07-10 plan (available in git history; successor:
`docs/LEAN_MEAN_CLEANUP.md`), W2 "Dark engine scaffolds + their tests", said:
**"Delete files (NOT keep-dark):
nodes/_otr_video_engines/{eng_character_3d.py (441), eng_still_parallax.py
(300), eng_triposr.py (159)} ... tests test_video_character_3d (385),
test_video_still_parallax (261), test_video_triposr (43)."** Its stated
reasoning is the operator's own instinct, sharpened: *"Keep-dark is the worst
state: CAPABILITIES v2 changes the engine contract, so dark scaffolds rot into
things that LOOK resurrectable but need rewrites anyway."*

That plan is D-1..D-6 RATIFIED and is the SECOND item in the rescoped order,
owned by **CODER D**, behind a **full `r2 -> r3 -> r4` arc the operator PINNED
himself** (GO_FORWARD: "do NOT enter at r3" -- because a deletion campaign's
entire value is its file-and-line kill inventory, the most perishable thing a
plan can carry).

**So the right answer to "should we rip it" is: yes, and you already decided
that. The live question is only WHO and WHEN.** Doing it here would preempt the
operator's own pin, duplicate CODER D's ratified scope, and break the
one-coder-in-the-code rule mid-way through the multi-clip build. CODER A does
NOT touch it.

## 2. THE FINDING THAT MATTERS -- a live guard is hiding in the dormant code

codex's must-fix 4, which I confirmed by reading the file:
`otr_image_director._is_3d_engine` (`:96-125`) is not only a 3D check. Lines
`109-119` raise for **ANY non-empty UNREGISTERED engine**, with a message
naming the custom-adapter case, and that branch has live coverage at
`tests/test_image_platform_c1.py:339-352`.

That is a **general fail-closed registry-membership validation** that happens to
live inside the 3D lock. `OTR_VideoDirector` lets an unknown custom id through
(`otr_video_director.py:528-566`), and the route freeze does not validate
registry membership either -- so today this dormant-looking function is the
place an unregistered engine gets caught.

**Deleting the 3D lock path would silently delete a live safety check.** The rip
therefore needs a MIGRATE-BEFORE-DELETE chunk that relocates effective-engine
registry validation to the VideoDirector / route-freeze boundary. That
migration is arguably worth doing on its own merits regardless of the rip --
the freeze is the natural home for it.

## 3. Where codex corrected ME

My brief claimed only three test files hard-depend on the dormant modules.
**That was wrong.** `tests/test_capability_profiles.py:52` and
`tests/test_workflow_apply.py:70` also import `eng_character_3d`, inside
multi-line import lists -- my inventory's classifier only matched the module
name when it sat on the `from ... import` line itself, so continuation lines
were misfiled as soft "mentions". Five test files hard-depend, not three.
Recorded because the same classifier error would understate any future rip.

## 4. What else the panel got right, and I accept

- **Separate `triposr` from the talker rationale.** It is an `image_to_video`
  static mesher and explicitly does NOT declare `requires_mesh_portrait`
  (`eng_triposr.py:102-119`). It goes because it is an unimplemented W2
  scaffold, not because the talker lock is retiring.
- **Do not touch the live mesh lane.** `mesh_stage` is registered, declares the
  DIFFERENT capability `requires_mesh_fodder`, emits directory clips, and
  `OTR_SilentComposite` consumes them. `directory_clip.py`,
  `otr_silent_composite.py`, `portrait_ledger.py`, `resolver.py`,
  `role_compat.py` are EDIT-prose-only or NO CHANGE. Confirmed independently by
  my own inventory: `requires_mesh_fodder` LIVE declarers = `['mesh_stage']`,
  `requires_mesh_portrait` LIVE declarers = `[]`.
- **Decide the rip BOUNDARY explicitly** -- adapters only (ratified W2), or full
  lane retirement including the zero-declarer capability, the `character_3d`
  family contract (`schemas.py:29-40,51-64`) and its `render_driver` branches
  (`:52-54,635-642`). Deleting two modules while keeping their dormant platform
  contract does not achieve the goal.
- **Cut the "zero live declarers" fence** I floated in the brief. Testing the
  permanent absence of a deleted feature protects nothing; the generic
  registry/CAPABILITIES bijection guard already covers it.

## 5. Where I overrule / soften the panel

codex marks today's `three_d_locked_slots` picked-vs-effective fix as
"a static dormant-surface audit catch, not a production bug" and cites
`PROD_BUG_LOG.md:3-17` to keep it out of PBUG/Bible. **Agreed and already
honoured** -- it was fixed with a comment saying so and was never filed as a
PBUG. Noting it only so a later window does not re-litigate the label.

## 6. The order, when CODER D runs W2

1. **MIGRATE LIVE PROOFS FIRST.** Relocate the unregistered-engine fail-closed
   validation out of `_is_3d_engine` to the VideoDirector / route-freeze
   boundary, with `test_image_platform_c1.py:339-352`'s intent preserved;
   rebase directory-clip fixtures onto `mesh_stage`; move the synthetic
   `character_3d` OOM / no-fallback proof (`scripts/otr_video_soak.py:59-89`,
   `render_driver.py:52-116`) onto a live heavy engine if the family contract
   retires. Green and pushed on its own.
2. **RIP THE DORMANT SURFACE.** Delete both adapters and the scaffold-only
   tests; strip the function-scoped imports in the five dependent test files;
   remove `requires_mesh_portrait`, `three_d_locked_slots`,
   `enforce_3d_granularity_lock` and the dispatcher halt
   (`otr_image_gen_dispatcher.py:535-568`); and, if full retirement is chosen,
   the `character_3d` schema + branches.
3. **SCRUB THE TOMBSTONES.** Shorten the essays at
   `_otr_video_engines/__init__.py:122-127`, `registry.py:434-439`,
   `scripts/otr_video_dep_pilot.py:117-119,240-243`. Keep dated historical
   audits as evidence; update only current surfaces.

Completion checks: neither module path importable; no registered engine or
profile exposes `requires_mesh_portrait`; retired ids unselectable via the
generic registry law; canonical `5377914B` byte-identical; full suite + Bible.

## 7. Bottom line for the operator

You were right, you already decided it, and the plan already says "delete, not
keep-dark". Nothing changes in this window. The one new fact worth carrying is
that a live registry guard is sitting inside the code you want to delete, so
W2's first chunk is a migration, not a deletion -- otherwise the rip quietly
removes a protection nobody meant to remove.
