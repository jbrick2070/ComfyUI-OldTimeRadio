# Item I -- THE BISECT. The description producer never changed.

**Date:** 2026-08-17. **Driver:** Claude (Cowork), measured at the files.
**Ordered by** `NEXT_WINDOW_KICKOFF.md` step 2 and the r1 judgment's disposition
item 1: *"BISECT early August first. Nothing else is decided until we know what
moved the rate 7x."*

## THE ANSWER, IN ONE LINE

**Nothing in the wrong-person description path changed.** `nodes/_otr_casting.py`
(the description producer, `_build_user_prompt`) and `nodes/_otr_original_radio.py`
(the pitch producer) have **no commits at all** between 2026-07-30 and the
inflection. The rate moved because **which defective episodes reach a ledger**
changed -- twice, both times as a side effect of a correct fix elsewhere.

## THE MEASUREMENT

Detector: a roster row's `character_description` contains a name token from the
pitch cast (`meta.source_meta.selected_concept.cast[].name`) that no roster row
owns. Tokens shorter than 3 characters, stop words and honorifics are dropped,
and the ANNOUNCER row is skipped -- **v1 of this detector scored `alien='THE'`
on any pitch named "The ..." and inflated July; a detector that fires on an
English article cannot date a regression.**

| month | pitch-bearing | affected | rate |
|---|---|---|---|
| 2026-07 | 74 | 6 | **8.1%** |
| 2026-08 | 36 | 22 | **61.1%** |

**THE CONTROL, and it is what makes the date trustworthy.** A run of zeros means
nothing if those days rendered a different bank. Composition is **identical
across the boundary** -- `original` bank, 3-character casts, on both sides:

| day | pitch-bearing | affected | bank mix |
|---|---|---|---|
| 2026-07-23 | 11 | 0 | `original` x11 |
| 2026-07-28 | 22 | 0 | `original` x22 |
| 2026-07-30 | 3 | 0 | `original` x3 |
| 2026-08-01 | 4 | 0 | `original` x4 |
| **2026-08-02** | 2 | **1** | `original` x2 |
| 2026-08-03 | 7 | 4 | `original` x7 |
| 2026-08-05 | 4 | **4** | `original` x4 |
| 2026-08-09 | 6 | **6** | `original` x6 |

**40 consecutive clean pitch-bearing `original` renders, 07-23 through 08-01.**
Then it turns on.

* **Last clean:** `signal_lost_echoes_of_a_recorded_will_20260802_020020`
* **First defective:** `signal_lost_the_unraveled_secret_20260802_231339`
  -- pitch `ELIZABETH HAWTHORNE` / `THOMAS HAWTHORNE`, roster
  `WENDY VOLKOV` / `PETER STENDAHL`, and WENDY's description carries both
  alien names.

No pitch-bearing render happened in the 21 hours between them, so the window
holds 26 commits (2026-08-02 02:46 .. 22:29).

## WHAT ACTUALLY MOVED -- two commits, neither of them a regression

**1. `afe53c7c` (2026-08-02 02:46:03) -- the trigger, by survivorship.**
It touched `nodes/_otr_fable2_markup.py`: speaker matching now retries once with
a trailing role parenthetical stripped. Before it, a script whose writer restated
each speaker with its role (`Commander Vance (Space Force Tactician)` against a
roster holding `Commander Vance`) raised `UNKNOWN_SPEAKER` on every line, burned
all four repair attempts, and **the episode died in the writer and wrote no
completed ledger.** Its own commit message states the effect plainly:
> *"It let a script that previously died at the parser reach the freeze gate."*

It is the only commit in the window that touches identity handling anywhere.

**2. `de6b2ce2` (2026-08-03 14:36:34) -- the amplifier.** Turned the story
scaffold **off** for `original` (`banks.json defaults.story_scaffold`), because
*"a catalog premise injected beside the pitch fought it"*. The ledgers show the
flip to the minute: `scaffold_on=True` through `20260803_115837`, `False` from
`20260803_144159` onward -- the first render after the commit. And the rate
follows:

| segment | affected / pitch-bearing |
|---|---|
| 08-02 23:13 .. 08-03 11:58 (scaffold ON) | 3 / 9 |
| 08-03 14:41 onward (scaffold OFF) | **11 / 11** |

With the catalog premise gone, **the pitch is the only story material the
casting brief carries**, so the description producer has nothing else to lean on.

## WHAT THIS CHANGES FOR THE FIX

* **The r1 cut still stands and is now safer to build.** The kickoff warned
  *"you may be about to fix a symptom of a recent change."* You are not: the
  producer is untouched code, and the defect it emits is the same one it emitted
  in July. Strip pitch/brief names from the **prompt-local** `casting_brief`
  copy, as r1 disposed.
* **NEITHER COMMIT MAY BE REVERTED.** `afe53c7c` fixes a real episode-killer;
  `de6b2ce2` is an operator directive (`original` gets no catalog seeds). They
  exposed and then amplified a pre-existing defect. That is what a correct fix
  upstream of a hidden bug looks like.
* **The July rate was never the true rate.** It is the post-survival rate --
  8.1% of episodes that *finished*. Some unknown share of July's confused-model
  episodes died at the parser instead of publishing. So "6.8% -> 50%" should not
  be quoted as a producer regression anywhere; it is a publication-rate shift.

## THE HONEST LIMIT ON CLAIM 1

The survivorship mechanism is **consistent with all the evidence and is not
proven.** What is proven at the files: the producer did not change; the window
contains exactly one identity-handling commit; and that commit's own message
records that it let previously-dying scripts complete. What is NOT proven: that
the specific July episodes which died at the parser carried this defect -- a
failed run leaves no completed ledger to measure, which is precisely why the
effect was invisible. The falsifiable prediction, if a window wants it: episodes
affected by the wrong-person description should show a higher rate of markup
repair attempts than clean ones.
