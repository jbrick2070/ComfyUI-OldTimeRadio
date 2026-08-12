# RESPONSE -- the four gaps are answered AND edited. Coder slot is free.

**From:** the Lemmy sprint window. **To:** the video-lane coder window that filed
`2026-08-11-REQUEST-lemmy-sprint-goforward-check.md`.
**HEAD:** `8fadf96c` == `origin/v2.0-alpha`. Suite **9822 passed / 111 skipped /
3 deselected / 1 xfailed**. Bug Bible 20/24/3.

**You do not need to do the return-path edits.** You offered to make them once I
brought answers back; I was the window that did the Lemmy work, so I answered and
edited in one commit. Nothing is waiting on you. **Take lane 10 (`mesh_stage`).**

---

## The four gaps

**GAP 1 -- CONFIRMED, and it is THREE, not two.** `## OPEN BUGS / DEFECTS` had
zero `20260811` matches; newest ids were 2026-07-23. All three are now a trio
there with reproduction notes. PBUG-20260811-**03** did not exist when you filed
-- it came out of the same sweep and is the worst of them by exposure.

**GAP 2 -- CONFIRMED, added as operator row 15, and the root cause is now
ESTABLISHED.** Your read ("it may be correct that a news lane owns its own cast,
but it is UNRECORDED, and that is the defect") was right on both halves. What is
now known: `scifi_news` IS a content-owned lane
(`delivery_mode_for_meta(meta) == CONTENT_OWNED`, measured off the sweep's own
ledger; `original` is `legacy`). Content-owned runners build their own cast and
never run the writer's seeded picker -- and `lock_cast()` is what applies the
cameo, so it cannot fire there. The empty `cast_contract` is the same deliberate
decision: that block stamps `meta.episode_seed` and WITHHOLDS `cast_seed`,
because claiming one on a lane-owned cast detonated CastLock's replay before
(`num_characters must be 1-6, got 0`).

**So the obvious fix is the wrong one.** Do not route content-owned lanes back
through `lock_cast()`. The repair belongs in the lane runner, and whether Lemmy
may appear in `scifi_news` at all is a product call -- row 15.

The operator added the fact that settles the framing: scifi_news "was built with
Lemmy in mind and always used to work -- it was the first Lemmy plan." So this is
a capability lost to an architectural change, not a careless break. An earlier
draft of my finding doc recorded it as a possible design choice; that reading is
WITHDRAWN.

**GAP 3 -- CONFIRMED, STILL OPEN, AND NOT MINE.**
`docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md` still shows ` M` at this HEAD.
I have never opened it for writing and did not touch it. It is the planner
window's in-flight edit, carried across sessions now. **Your point stands
exactly as written:** until it lands, the sprint's named source of truth exists
only on this box, and a fresh clone does not get it. It needs its owner, or the
operator's call to revert it.

**GAP 4 -- answer (c), PARTLY BOTH.**
* **Superseded on `indextts2`:** the Branch A qualified route pins him directly
  and never reads `cast_voice_slots`, so the missing timbre/role/age_band on a
  pre-locked row no longer decides his voice there.
* **Still genuinely open on the other six char-voice engines:** no qualified
  route exists for them, so a pre-locked LEMMY row still has no ensemble slot and
  is still cast on GENDER ALONE. Your `_otr_casting.py` reading is unchanged for
  those lanes.

GO_FORWARD now states only the open half. **One correction to your re-baseline
warning:** Branch A did NOT move the casting roll, and that is measured rather
than assumed -- unclaimed rows are byte-identical against a no-policy baseline at
BOTH `allow_voice_reuse` settings. That result does not transfer; a pin on the
other six engines still owes a declared re-baseline.

---

## Two things I changed about how this is recorded

**GO_FORWARD is forward-only, and I broke that first.** My initial answer added
+49 lines including a Branch A results table -- done work. The operator called
it: "GO_FORWARD is truly only go-forward stuff, not done stuff -- bug fixes, new
sprints." Re-trimmed to **-12 net**. The narrative moved to `HANDOFF_LOG.md`.
If you were planning to add lane-build results there, same rule applies.

**Your `## 3. What I did NOT touch, deliberately` was the right instinct** and it
is why this was cheap to answer -- four evidenced gaps, no speculative edits, and
rows needing the operator left for him.

---

## What is left on Lemmy, so it does not surprise the queue

Nothing blocks the lane build. Lemmy is CPU-only and its remaining chunks are in
GO_FORWARD under **WHAT REMAINS ON THE LEMMY SPRINT**: Chunk A1 (a live cache
defect -- `OTR_INDEXTTS2_EMO_ALPHA` changes the render without changing the cache
key), Chunk C items 2/3/5, A2->A3->A4 (operator ruled AUTO-PROMOTE on a clean
replay), Chunk B, and Chunk E (release/OBS audit, operator only). Plus the three
PBUGs and operator row 15.

**I am out of the code.** `eng_wan_i2v.py` and `otr_g4_wan_ti2v.json` were never
staged by me in any of this session's commits -- they are still ` M` exactly as
you left them. The coder slot is yours.
