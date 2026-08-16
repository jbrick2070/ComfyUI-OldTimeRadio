# Blind narrative ranking BY SOURCE BANK (2026-08-16)

**Operator question:** which BANK tells the best stories?

**Method.** Two current-era episodes per shipped bank (12 total), shuffled,
labelled `01`-`12`, bank identity withheld from the reader. Regenerate with
`scratchpad/bank_transcripts.py` (seeded 816).

**Trust check first: the reader re-paired all twelve into their six banks,
6/6 CORRECT, blind** -- by lane fingerprints it discovered itself (Folger act
citations on one pair, "adapted from public domain" on another, named
speaker tags + double-announcer button on a third). It also independently
paired the two lanes it had never been told were siblings. The rankings
below are therefore a signal, not noise.

## The ranking

| bank | avg /10 | reader's verdict |
|---|---:|---|
| `scifi_news_pro` | **7.5** | best structure; "the cleanest blueprint in the corpus" |
| `original` | 6.5 | best SINGLE episode (8/10) and the highest floor |
| `media_archive` | 3.5 | truncated closers, motives from nowhere |
| `public_domain` | 2.5 | famous words standing still |
| `shakespeare` | 2.5 | wrong-play frames, corrupted lines |
| `scifi_news` | **2.0** | "it is not a story"; worst sample scored 1/10 |

Structure score (the deterministic finder) ranked the same two banks 1st and
last, so two independent instruments agree.

## The structural finding: lanes that BUILD an arc beat lanes that BORROW one

* **`original` wins on FLOOR** because the arc is native -- the generator owns
  start/middle/end, no fidelity debt, no five-act plot to compress. Even its
  weaker sample failed only on execution, never on shape.
* **The news lanes are HIGH-VARIANCE, and the variance is SOURCE SELECTION,
  not writing.** When the item is a physical crisis with a deadline (a star
  being eaten, a hurricane, a river running red) reality supplies stakes,
  clock and imagery for free and the lane only adds two people with a
  tradeoff -- those scored 8 and 7. When the item is institutional PR or a
  trend survey, there is NO conflict inside the source, so the lane either
  invents incoherent melodrama or recites statistics -- those scored 3 and 1.
  **Same machinery, 1/10 to 8/10, decided by which article was picked.**
* **The adaptation lanes lose STRUCTURALLY:** a scene-slice from a long-arc
  work has no self-contained arc, so the pipeline pads it by re-generating
  the scene's one emotional peak.

## CORRECTNESS defects the read surfaced -- NOT quality, and all pre-TTS

These are carved out of the 2026-08-04 freeze (they are faults, not taste),
and every one is deterministically detectable before audio:

1. **THE SHAKESPEARE LANE FRAMES THE WRONG PLAY -- measured TWICE.** A
   Twelfth Night scene announced as *"Verona ... Capulets and Montagues"*;
   a Tempest scene framed as Romeo and Juliet. **This CONFIRMS the
   previously-undiagnosed GO_FORWARD item D** (`tempests_midnight_
   revelations`, "is the scene actually Macbeth?") -- it is a real, repeating
   frame/scene binding defect on the lane where fidelity outranks arc. Root
   shape: the announcer frame is sampled independently of the selected
   excerpt instead of being generated from the same metadata record.
2. **Corrupted text spoken on air:** `"I love NOW YOU."`
3. **Speaker tags / char_ids leaking into dialogue:** `THE TIME TRAVELER`,
   `MIRA REEVES`.
4. **Closers truncated mid-sentence** (media_archive, both samples):
   *"...working tirelessly to preserve and restore a lost."*
5. **Pipeline metadata read aloud** (scifi_news): *"final coda, factual
   report backed by P0 facts F01-F06."*
6. **Looped beats:** the reveal fires three times verbatim ("I am Viola",
   "In vain have I struggled") -- the per-beat writer cannot see that the
   beat already happened.

## Recorded, NOT scheduled

Per the 2026-08-04 ruling the operator prices craft work himself. Items 1-5
are correctness and are legitimately fixable; item 6 and the
source-selection gate for the news lanes are the highest-leverage of the
rest. Nothing here is queued without his say-so.

**Best episode in the corpus:** `signal_lost_both_our_thumbs_on_the_key_
20260730_222942` (`original`) -- the reader would play it for a stranger.
**Cleanest structural model to copy:** the `scifi_news_pro` two-hander with a
real cost paid.
