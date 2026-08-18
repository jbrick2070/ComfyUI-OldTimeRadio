# Item F -- LIVE LEG RECEIPT (the only proof that counts)

**Run:** 2026-08-17, `scripts/otr_writer_bank_gate.py --banks shakespeare
--acts 1`, profile `otr_w45_still_flat`, against a freshly booted headless
server on :8000 loading `workflows/otr_canonical.json`. **PASS, exit 0,
10.6 min.** Code under test: `87dee50d`.

**Episode:** `signal_lost_shattered_histories_20260817_170934`
**Selected work:** `The Tempest` -- `folger-tempest:act1-scene2-prospero-miranda`,
*"Act 1, Scene 2 - Prospero opens the island's history"*.
**Cast:** ANNOUNCER, PROSPERO, MIRANDA.

## What the announcer actually said

> **Good evening, listeners. Tonight, we bring you Shakespeare's 'The Tempest',
> where Prospero and Miranda brace against a gathering storm.**

And the closing:

> **From the heart of tonight's tempestuous sea: Tonight's scene was drawn from
> William Shakespeare's The Tempest.**

## Why this is the acceptance

* **It names the work it actually performed.** The BEFORE behaviour on this exact
  lane was a Tempest scene framed as *Romeo and Juliet*, and a Twelfth Night
  scene announced as *"Verona ... Capulets and Montagues"*.
* **It names ONLY the locked cast** -- Prospero and Miranda, both real cast rows.
  No character imported from elsewhere in the play, which was the open question
  Fable was asked and the risk it judged real-but-small.
* **No other manifest play's title appears anywhere in the episode.** Checked
  programmatically over the full spoken text against all 14 manifest rows:
  `OTHER play titles spoken anywhere: NONE`.
* **THE SECOND PRODUCER RAN AND THE FIX HELD.** `meta.announcer_intro_rewrite`
  reads `status = "announcer_intro_rewritten"`. The I.4.9 rewrite overwrote the
  first frame and STILL named the right work -- which was the single most likely
  way for this fix to be silently undone, and the reason r1 insisted both
  producers be fixed in one change. **A leg where the rewrite did not fire would
  not have tested this.**
* **The framing fits the scene, not just the play.** "brace against a gathering
  storm" is Act 1's situation; the announcer did not reach for Act 3 or for the
  play's ending, so the KILL 2 no-spoiler contract survived the added field.

## What this does NOT prove, stated plainly

One leg, one play, one seed. It shows the mechanism works end to end on the
shipped path; it is not a rate. The cross-play detector in
`tests/test_cross_play_frame_leak.py` catches leakage from OTHER manifest works
only -- a wholly invented place belonging to no play would still pass both this
leg and that test. The honest claim is **"the announcer named the work it
performed, on a live leg, with the rewrite producer active"**, not "the class is
eliminated".

`public_domain` shares every symbol on this path and its seam was updated in the
same change, but **it was not exercised by this leg** and remains unproven live.
