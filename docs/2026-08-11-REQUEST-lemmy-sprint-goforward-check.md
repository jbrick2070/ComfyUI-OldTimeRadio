# REQUEST -- confirm GO_FORWARD carries everything the LEMMY VOICE sprint needs

**From:** the coder window that closed video lane 9 (HEAD `856fc288`).
**To:** whoever picks up the Lemmy sprint plan.
**Status:** REQUEST ONLY. Nothing in GO_FORWARD has been changed for the four
gaps below -- the operator asked to review first and return for final signoff.

---

## 0. PRIORITY, so nobody reorders the build by accident

**THE VIDEO LANE BUILD COMES FIRST.** 9 of 21 lane packets are closed and
pushed; **lane 10 (`mesh_stage`) is OPEN, fully diagnosed and uncoded**, and
lanes 11-22 follow it one at a time under the operator's build law (one lane
open at a time, close its QA before touching the next).

Lemmy is **CPU-only work** and does not contend with the video lanes for the
GPU -- but it does contend for the ONE coder slot, and the operator's standing
rule is one coder window in the code at a time, serialized through GO_FORWARD.
So this request is about keeping the Lemmy plan CORRECT AND READY, not about
starting it. Do not open Lemmy coding work ahead of the lane queue without the
operator saying so.

---

## 1. What the previous Lemmy voice sprints actually did (the state to check against)

**Phase 1 (`bec0ca79`)** -- shipped and left. Landed `accent: "cockney"` at
`config/cast_pools.py:317`, plus `dialogue_orthography`, `speech_signature` and
`nodes/_otr_dialogue_policy.py`. It shipped `LEMMY_VOICE_POLICY`
**defined-but-unwired**, which is why later work could land with zero behaviour
risk.

**r1 panel (`kibitz-runs/2026-08-08-lemmy-cockney/r1/`)** -- its `final.md`
**is STALE on its own step 1**. It opens by asking to reconcile
`accent: 'neutral'` against a Cockney description; Phase 1 already fixed that
AFTER r1 was written. Anyone resuming from r1 must SKIP its step 1. D-4 is
also partly addressed by Phase 1.

**r2 panel, both lanes (`kibitz-runs/2026-08-10-lemmy-cockney/r2/`)** -- r3 was
deliberately NOT run because r2 hit a plan-level blocker, now **operator row
14**. It also produced the qualification-receipt honesty fix:
`approved_native_routes` had listed bark as APPROVED via a bare string with no
artifact, hash, test lines or operator verdict. That dict is now EMPTY,
`canonical_route` keeps bark as a routing fact with
`qualification_receipt: None`, and `QUALIFICATION_RECEIPT_REQUIRED_FIELDS` +
`is_qualified_route()` define what a real receipt must contain. Fail-closed.

**Branch A (`46608b93`, 2026-08-10) -- SHIPPED, G1 PASSED.**
* **G0 rights closed on the record** --
  `docs/2026-08-10-G0-RIGHTS-DECISION-CARD-lemmy.md`, decided
  2026-08-10T20:37:17Z against a same-day terms snapshot. Tier left
  UNDETERMINED and marked for the evidence packet.
* **The premise correction that must not be re-inherited:** Lemmy was never
  redrawn per episode. Of 186 LEMMY rows across 1,633 ledgers, 151 are the
  bark-preset path and **33 of the remaining 35 are the SAME reference** --
  every one had `meta.episode_seed=None`, so CastLock derived an identical
  selector seed. He was **ACCIDENTALLY PINNED**. The fix was an explicit
  qualified re-pin, NOT a rewrite of the generic selector. The defect is a
  floor-EVIDENCE failure; the earlier attempt to infer a violation from the
  `_indian` filename or the `warm` timbre tag was rejected and must stay
  rejected.
* Plans 5.1 / 5.2 / 5.3 shipped (`nodes/_otr_voice_route.py`, CastLock re-pin
  ordering `e791344b`, reference resolution + receipts + fingerprint
  `fdc016ef`). `REQUEST_SCHEMA_VERSION` 2 -> 3 deliberately, measured at ZERO
  cached entries so the practical cost was nil.
* **The blinded audition returned a PASS.** The operator identified the
  incumbent as Indian WITHOUT seeing its label, and the candidate also beat the
  same-speaker control -- so IndexTTS2 carried the ACCENT through the clone, not
  merely the timbre. Shipped:
  `models/TTS/refs/indextts2/lemmy_algenib_cockney_v1.wav`, bank row
  `idx_lemmy_algenib_cockney_v1`, policy record with the verdict quoted.
* **Proven in production 2026-08-11** by a six-bank sweep: both cameo lanes cast
  Lemmy on the qualified route and published to `otr/obs/`; both fidelity lanes
  REFUSED the forced cameo and recorded `source_fidelity_exclusion` -- the half
  that needed proving, since a broken exclusion looks exactly like a working one
  at an ~11% roll.
* **Branch B stays unbuilt.** It existed only for a G1 failure, and G1 passed.

---

## 2. THE FOUR GAPS I VERIFIED -- please confirm or correct each

Each was checked against the file at HEAD `856fc288`, not assumed.

### GAP 1 -- two live PBUGs exist only as prose, not in OPEN BUGS
`PBUG-20260811-01` (forcing the cameo kills the `scifi_fable2` writer on
`scifi_news_pro`, reproduced at 30 AND 90 words, so not a word squeeze) and
`PBUG-20260811-02` (`scifi_news_pro` dies at node 92 with no materialized still
for `music_closing_001`, seen once) appear ONLY inside the Branch A narrative.
The `## OPEN BUGS / DEFECTS (live, not yet closed)` section contains **zero**
matches for `20260811`; its newest ids are from 2026-07-23. **A window working
the OPEN BUGS list will never see either.**
*Ask:* should both be promoted into OPEN BUGS with their reproduction notes?

### GAP 2 -- an "OPERATOR DECISION OWED" that is not in the operator list
The sweep found that **`scifi_news` writes an EMPTY cast contract**: the
`scifi_news_circuit` pipeline never calls `lock_cast()`, so it silently ignores
`lemmy_cameo` AND `num_characters` (asked 2, got 3) and records no `cast_seed`.
The narrative marks it **OPERATOR DECISION OWED** -- it may be correct that a
news lane owns its own cast, but it is UNRECORDED, and that is the defect.
`### WAITING ON THE OPERATOR -- the whole list, in one place` currently holds
rows **10, 12, 13, 14 only**. This is not among them, so the section's own
"whole list, in one place" claim is not true today.
*Ask:* add it as a numbered operator row?

### GAP 3 -- the authoritative plan has UNCOMMITTED changes
`docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md` is the named authoritative
plan for Branch A and it shows as **` M` (modified, uncommitted)** in the
working tree -- another window's in-flight edit, carried across at least two
sessions now. Whatever it says is not on origin, so a fresh window cloning or
pulling does not get it.
*Ask:* whose edit is this, and can it be committed or reverted? Until then the
sprint's source of truth exists only on this box.

### GAP 4 -- possible contradiction between the two Lemmy sections
`### LEMMY COCKNEY` still says, under **"WHAT IS ACTUALLY LEFT"**, that D-2 is
open: *"the per-character voice pin still does NOT reach pre-locked LEMMY"*,
with `_otr_casting.py:1815-1837` cited and the warning that changing how he is
cast MOVES the casting roll and needs a DECLARED re-baseline.
But `### LEMMY BRANCH A` reports a qualified re-pin SHIPPED and Lemmy cast on
`idx_lemmy_algenib_cockney_v1` in a live production sweep.
*Ask:* is D-2 (a) still genuinely open for the OTHER six char-voice engines,
(b) fully superseded by the Branch A route, or (c) partly both? Whichever it
is, the two sections should stop implying different things -- and if a
re-baseline is still owed, that needs to be stated as a task rather than a
warning buried in a paragraph.

---

## 3. What I did NOT touch, deliberately

I made **no edits** to the Lemmy sections, the operator list, or OPEN BUGS.
The operator asked to review first and come back for final signoff, and rows
that need his ruling are not mine to write. GO_FORWARD's Lemmy content is
otherwise rich and current -- these four are the only gaps I could evidence.

**Return path:** bring the answers back and I will make the edits in one commit
-- promote the two PBUGs, add the operator row, resolve the D-2 wording, and
either commit or flag the dirty plan doc. That is a docs-only change and can
land without disturbing the lane queue.
