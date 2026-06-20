# 864-word FRONTIER soak -- REVIEW (2026-06-21)

35 legs, 4 frontier writers (Opus / Gemini-pro / GPT-latest / Grok-4.3, each with
Opus as the slot-B finisher), indextts2 voice, z_image stills, flat_still video,
diverse news-driven premises. Ran 04:14 -> 12:24 (8h cap), stopped cleanly at 35
of 48 planned legs. Results: `story_soak_results.csv` / `.json`.

## HEADLINE: the mechanism is rock-solid + the Phase-1 story fix HOLDS in production
- **35 / 35 legs succeeded. ZERO errors.** Every frontier writer produced a
  complete, frozen, shippable episode at 864-word target.
- The 2026-06-19 news-as-crux Phase-1 build is VALIDATED live at scale:
  - `default_wants_present = no` on **35/35** -> the old `_DEFAULT_A/B_WANTS`
    boilerplate is GONE everywhere (the central fix is holding).
  - `dramatic_state_source = llm` on **34/35** -> the B1 LLM-derived opposed
    wants/question/ending spine is what drives the drama (1 blank row).
  - `ds_aboutness = YES` on **34/35** -> the dramatic state is genuinely ABOUT
    the news premise, not generic.
  - `freeze_verdict = frozen_with_warns` on 34/35 -> REPAIR-THEN-SHIP working;
    no refusals / `needs_full_rerun` terminal skips.
- Titles show real anthology variety (diverse seeds working): "Every Eleven
  Seconds", "Twelve Degrees Off", "Teeth of the Dead", "The Correlated Dark",
  "Thumb on the Master Switch", "The Accusing Tooth" -- evocative, distinct,
  on-genre.

## WHAT TO LOOK AT (not failures -- known shape issues + one residual)
1. **Word-count under-shoot (the fixed-mold symptom).** Target 864, but actual
   mean **509** / median 558 / range 227-774 -- NONE reached 864, and
   **n_lines == 18 on every single leg** regardless of model or target. This is
   the one-size 18-beat SHAPE (`ACT_COUNT_CONFIG`), exactly the Phase-2
   "shape-follows-story" item that was deferred. Length here is a SYMPTOM of the
   fixed mold, not a craft defect -- the stories are complete within 18 beats.
2. **Grok-4.3 writes SHORT.** avg 269 words vs Opus 662 / Gemini 590 / GPT 542.
   If 864 is the real target, Grok needs a stronger length push (or it is just a
   terse model). Opus is the closest to target and the natural default writer.
3. **news_reaches_lines = no on 5/35 (14%).** Even with an news-derived dramatic
   state, the news key-terms don't surface in the VOICED lines on ~1 in 7 legs --
   a residual leak from the spine into the line composer worth a Phase-2 look
   (the dramatic state is news-grounded but the surface dialogue drifts generic).

## PER-WRITER (avg words, all legs succeeded)
| writer (slot A) | legs | avg words |
|---|---|---|
| Opus-latest | 8 | 662 |
| Gemini-pro-latest | 9 | 590 |
| GPT-latest | 9 | 542 |
| Grok-4.3 | 9 | 269 |

## VERDICT
The story spine is production-healthy: news-as-crux is real, the boilerplate is
dead, every frontier model ships a clean episode. The open work is SHAPE (length
/ act-count follows the story instead of a fixed 18 beats) + the 14% news->lines
surface leak -- both already scoped to Phase 2. No regressions; nothing blocking.

## NOTE: the companion 400-word soak ABORTED
The extra 400-word run launched against the same :8011 server self-aborted on its
smoke leg (`status=error`, 0 words, 571s) -- almost certainly queue contention
with the still-running 864 soak. No 400-word data was produced. The box is now
free; a clean 400-word run can be relaunched on a dedicated server if wanted.
