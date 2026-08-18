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

## SECOND LEG -- `public_domain`, AND IT DID NOT PASS

Run at the operator's instruction on `b45c5577`:
`otr_writer_bank_gate.py --banks public_domain --acts 1`, RESULT SUCCESS,
episode `signal_lost_the_blackwood_enigma_20260817_172553`, source
**`Nonsense Novels` by Stephen Leacock**.

> **Tonight, from the cluttered confines of an office, we gather for 'The
> Adventure of the Purloined Paper', starring THE GREAT DETECTIVE and his
> reluctant confidant, the SECRETARY.**

**That work does not exist.** It is neither the source (`Nonsense Novels`) nor
the episode title (`The Blackwood Enigma`) -- a third invented string. The
closing coda was correct: *"Tonight's tale was adapted from Nonsense Novels, by
Stephen Leacock."*

**THE FIX WORKED AND THE MODEL IGNORED IT.** Replayed the shipped ledger through
the real code: `identity_from_meta(meta).work_title == "Nonsense Novels"`,
`source_kind == "public_domain"` (in `ADAPTATION_SOURCE_KINDS`), so the writer
passed it and `_work_line` rendered **`WORK: a scene from Nonsense Novels`** into
the prompt. `announcer_intro_rewrite == "announcer_intro_rewritten"`, so the
second producer wrote the line and it receives the title too. The fact was
delivered; the model named something else.

**So the honest verdict on item F is SPLIT:** shakespeare PASSES live,
public_domain FAILS live, on the same commit. Logged as **PBUG-20260817-04**, no
fix attempted -- the seam has said *"invent none"* the entire time, so guessing
at wording is the mechanism this item already proved unreliable.

**And it is the residue this receipt predicted.** The section above says the
detector catches cross-play leakage only and "a wholly invented place belonging
to no play would still pass both this leg and that test". An invented WORK title
is exactly that residue, found one leg later. **The two legs together are the
argument for the live-leg rule: one lane passing proved the mechanism, the other
failing proved the mechanism is not the whole defect.**
