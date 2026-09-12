# OTR Go-Forward Plan

**ONLY UNFINISHED WORK BELONGS HERE.** When work finishes its receipt moves to
[HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence folder and the row leaves this
page. A finished prerequisite earns **one clause inside the row that still needs
it** -- never a receipt, never a measurement write-up, never a struck-through or
"SHIPPED" row. The test is one question: *does a row still in this file stop
making sense without that sentence?* No -> cut it.

**THIS FILE IS ARC AND CODE. TESTING IS NOT IN IT** (operator, 2026-09-12:
*"let's just do the coding and arcs first, let's not even talk testing yet"*).
Decide first, build second -- and when both are empty, section 6 at the bottom
is what you have earned. A deferred arc is deferred coding,
and a gate on evidence a leg produces is not a valid deferral either, because
the legs run last: a row that can only be settled by live evidence is settled
WITHOUT it or cut with the reason written in. **The only things that genuinely
defer** are a row blocked on an operator ruling (section 3) and a row
deliberately cut with its reason. Apply that test whenever a row claims to be
blocked.

Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md) first;
this file does not restate them. **For what has already happened -- commits,
measurements, receipts -- read [HANDOFF_LOG](HANDOFF_LOG.md), newest entry
first.**

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. Exactness is not the goal: *"I'm not expecting anything exact."*

**CRASH-CLASS AND DURABILITY-CLASS DEFECTS ARE THE WORK** -- an uncaught
exception, a live asset written where a sweeper can delete it, an identity that
silently resolves outside its episode, and **a machine that silently renders a
configuration we have already proven wrong.** Rows below use the phrase "not
crash-class" against this definition. Aesthetic drift is closed and is not work;
see [ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md).

## 1. ARC -- settle these before writing the code

Every row here has more than one defensible answer, so it gets its round, its
measurement or its ruling BEFORE code. An arc costs a wait, not a budget.

*Empty as of 2026-09-12 (batch 2, `docs/2026-09-12-arc-batch-2/`): seven
rows settled -- four cut or closed on the code, one settled by the bench,
and three converted into named CODE rows below after a codex refutation.*

## 2. CODE -- the design is settled, build it

*Empty as of 2026-09-12. Every row that was here is either shipped (its
receipt is in [HANDOFF_LOG](HANDOFF_LOG.md)) or moved to section 3 because
only a ruling is left.*

## 3. Blocked on the operator -- each unblocks with one word

**He answered all sixteen on 2026-09-12 and left to do other things, with:**
*"i cant listen until i get back thats ok if thats the one thing waiting, dont
stall for me, keep going and do a bunch of testing, we can fix when i get back."*
So the rows below are what SURVIVED his answers. The twelve rows that closed are
gone from this file by its own rule; the receipt in HANDOFF_LOG carries them.

### Waiting on his ear, and nothing else

* **The techno and house cues, re-rendered.** He judged `3_media_archive`
  (jazz) and `4_original` (salsa) RIGHT, and `1_scifi_news` (Detroit techno)
  and `2_public_domain` (Chicago house) WRONG -- *"supposed to be techno whats
  wrong??"* and *"supposed to be house very wrong"*. Three sustained-music
  instructions were reaching a dance cue: a pad closing both electronic
  palettes, the neutral floor word "atmospheric" LEADING the row, and an
  orchestral per-cue arc ("a rising overture", "resolving to a warm held
  chord") that was appended OUTSIDE the guard already written to keep
  orchestral language off a rhythmic palette. All three are fixed and the fix
  is scoped by a new `Palette.groove_arc` flag to exactly the two banks he
  rejected -- **jazz and salsa compose byte-identically to what he approved**,
  because his ear on shipped output outranks a tidier rule (codex refuted the
  wider blast radius and was right). **Unblocks with a listen to the new
  cues in `otr/obs/`.**
* **Whether the causal story is actually right.** Codex's standing objection,
  folded rather than argued with: the operator's verdict is on OUTPUT and the
  diagnosis is a reading of the PROMPT, with no seed-matched A/B between them.
  The untested competitors are the checkpoint (post-trained SA3 ignores cfg and
  negatives) and the sampler. The comments say hypothesis, not fact. **Unblocks
  with the same listen.**
* **SETTLED 2026-09-12 -- the anime checkpoint is IN.** His verdict on
  `the_clanking_chains_20260912_153041__anim__stfl__sd15__...` was one word:
  *"perfect"*. `Counterfeit-V3.0_fp16.safetensors` is fetched, the anime pack
  names it, and the server log records `[sd15] minted still 768x432 ...
  ckpt=Counterfeit-V3.0_fp16.safetensors` with `OTR_SD15_CKPT` unset -- so the
  name came from the pack through the style resolver. **His ear has ruled;
  this row does not get re-asked and the lane is not benched again.**
  The one thing left is bookkeeping, not a question: `config/profiles/
  otr_sd15_stills.json` is still `status: "draft"` because it was written
  minutes before the leg. It wants a couple more episodes on other styles
  before it claims `shipping`, and it is NOT edited while the wave head is
  frozen.

* **HIS EAR CALIBRATED THE METRIC, 2026-09-12, and this is the most useful
  thing in this file.** Two `public_domain` episodes, the SAME bank, a
  BYTE-IDENTICAL composed prompt, opposite verdicts:

  | episode | opening cue | tempo error vs the 122 BPM asked for | his words |
  |---|---|---|---|
  | `the_clanking_chains_153041` | 0.42 / 123.0 BPM | **1.0** | *"perfect music"* |
  | `firelight_skepticism_151426` | 0.29 / 156.6 BPM | **34.6** | *"just house chords no rhythm"* |

  **TEMPO ERROR is the discriminator, and onset periodicity is not.** The two
  differ 35-fold on tempo error and only 0.42 vs 0.29 on periodicity -- a gap
  far too small to have predicted his verdict. Any future guard, bench or
  A/B on groove cues binds to |measured BPM - requested BPM|, never to a
  periodicity floor.

  **It also means the wording is exonerated twice over.** Those two cues asked
  for the same thing in the same words and one was perfect. The variable is the
  render, not the prompt -- so the fix is a recipe or a re-roll, not more
  prompt-craft.

  **A FALSIFIABLE PREDICTION, left here on purpose:**
  `shadows_on_the_catwalk_144526` measures 0.74 / 120.2, i.e. **1.8 BPM off**.
  If tempo error really is the thing his ear tracks, he should like that one
  too. If he does not, this calibration is wrong and the row reopens.

* **CUT 2026-09-12: the anime checkpoint does NOT go to the AnimateDiff lanes,
  and this row is closed, not deferred.** His words: *"lets stop chasing the
  anime things, we can leave it in there if it['s] coded, I don't want to spend
  any more time chasing before release."*

  **What SHIPS is what he already approved:** the pack checkpoint reaches the
  STILL engine only (`nodes/_otr_image_engines/sd15.py`), which he judged
  *"perfect"*. That stays exactly as it is.

  **What is CUT:** extending it to `eng_ghost_signal.py` and the other SD1.5
  motion lanes, where `GHOST_CHECKPOINT_NAME` stays the pinned base checkpoint.
  A design panel on it was STOPPED mid-run on his word. Do not restart it, do
  not "just try" the constant swap, and do not re-raise this before release --
  he noticed the limitation himself and cut it himself, which is the strongest
  form this ruling can take.

  For a future reader who wonders why it looked easy: it was not. That swap is a
  RECIPE BUMP -- `GHOST_RECIPE_RECEIPT` would have to be repointed or every
  receipt already on disk stops being interpretable -- and the lane runs a LIVE
  negative at `GHOST_CFG = 8.0` specifically because the lettering defense needs
  real unconditional conditioning, which is the documented reason an AnimateLCM
  checkpoint was refused once before.

* **KNOWN ISSUE, PARKED FOR AFTER RELEASE: music cues are a lottery, and the
  prompt is not the lever.** He reported two on 2026-09-12 -- a closing cue that
  "is like 2 seconds" and an opening that is "just house chords no rhythm,
  maybe one slight beat". Both are real. A 19-agent panel settled the cause and
  the answer is the same for both: **an under-constrained prompt hands the
  outcome to the seed.** Recorded here so nobody pays for this analysis twice.

  **The driver's first framing was WRONG and the panel corrected it.** It read
  four bad closings, all on the authored `scifi_news_pro` lane, and concluded
  "authored rows are broken". There are SIX authored closings on disk and the
  sample omitted the two that render at 100%. Worse, within the authored group
  the supposed cause runs BACKWARDS: 20 chars -> 100%, 27 -> 100%, 25 -> 48%,
  45 -> 65%, 45 -> 77%, 48 -> 35%. "Soft, contemplative piano" (48%) and "A
  soft, somber piano melody" (100%) are near-identical asks with opposite
  outcomes. And the complained episode's OWN opening is also authored
  ("Detroit techno, 128 BPM", no arc, no tail) and renders at 94% -- a control
  inside the very episode that kills the missing-arc theory.

  **What the panel PROVED cannot be the cause**, each grounded in the file:
  * There is NO length-to-duration coupling anywhere.
    `eng_stable_audio_3.py:209` computes `seconds_total = max(context_s,
    dur * 3.0)` with no prompt term; `:392` builds the latent from `dur` alone;
    sampler, scheduler, steps and denoise are env constants at `:378-385`.
  * A short prompt cannot shorten any tensor. `model_base.py:895-902` pads
    cross-attention to a fixed 256 slots plus the seconds token, all attended.
  * The one prompt-derived path, `_sa3_clip_window`'s "outro"/"opening" text
    fallback, is dead TWICE: `stable_audio_theme.py:331` always passes a real
    placement, and `StableAudio3` never reads `seconds_start` at all -- the
    receipt itself records `seconds_start_read_by_model: false`.
  * Post-processing is clean: `_ceiling_the_cue` is a peak limiter at -1.0
    dBFS, no fade, and the receipts show peak -1.03 / -1.00.

  **The real shape:** composed closings measure 71-100%, mean ~96%, because a
  composed row names instruments, mood, an idiom carrying a BPM, an arc and the
  tail, leaving the sampler almost nothing to choose. Authored closings measure
  35-100%, mean ~71% -- the same distribution with a much worse floor. It is a
  VARIANCE difference, not a deterministic one.

  **THE CHEAP TEST, already designed, ~6 minutes, no episode leg.** Call
  `generate_clip` directly on the exact shipped closing spec at 8 seeds and
  measure audible fraction. A 35-100% spread on one text confirms seed noise
  and closes it; a tight cluster near 35% means the text IS deterministic and
  the hunt moves to arm B (same text plus the arc and tail).

  **NOT RUN, on his instruction:** *"I don't want to spend any more time chasing
  before release."* This row is for after.

* **RULED 2026-09-12: no auto-fetch row for the anime checkpoint.** His words:
  *"I don't think we will set an auto fetch file because it's so specialised."*
  Correct, and it settles the last open question on that feature: a 4.24 GB
  download for one style most users never pick is the wrong default. The
  checkpoint stays OPT-IN, which is exactly what already ships -- present means
  the anime pack uses it, absent means the stock model, and nothing fails
  either way. Documented in the README as opt-in. **This feature is DONE.**

* **THE GROOVE DEFECT IS FOUND, IT IS ONE LINE, AND IT IS THE DRIVER'S OWN BUG
  FROM 2026-09-12.** A 26-agent panel refuted the driver's seed-variance theory
  by measurement and found the real cause. **Awaiting his go/no-go only because
  it touches the render path and the wave head is frozen.**

  **NOT the seed.** Same prompt, different seeds, ruled `small_base` arm: pulse
  spread 0.016 (house) / 0.065 (techno), tempo never missed, 6 of 6 locked.
  A variable with a range of 0.065 cannot produce the shipped range of 0.78.
  Renders are exactly reproducible -- 18 cross-boot pairs agree on every
  recorded digit of seven metrics -- so a retry-on-tempo guard would be
  strictly wasteful: re-rolling the seed at a fixed prompt buys ~0.065 of pulse
  for a whole extra render.

  **The driver's premise was FALSE.** It claimed the two contrasting episodes
  had byte-identical prompts. They have different prompt hashes. What differs is
  the brief-mined MOOD WORDS, and they sit IN FRONT of the tempo instruction:

  | lead words | tempo | pulse | his verdict |
  |---|---|---|---|
  | suspenseful, **driving**, dark | 120.2 | 0.74 | -- |
  | tension, ominous, **frantic** | 123.0 | 0.42 | *"perfect music"* |
  | ominous, suspenseful, eerie | 156.6 | 0.29 | *"no beats, maybe one slight beat"* |

  The two that locked lead with a MOTION word. The one that failed is pure
  texture. `nodes/_otr_music_prompt.py::compose_music_prompt` appends
  `mood_terms` BEFORE `palette.idiom`, so three atmosphere words stand in front
  of "Chicago house at 122 BPM".

  **This is the exact failure the module's own docstring already names** for the
  neutral "atmospheric" default -- and the 2026-09-12 fix only skipped the
  DEFAULT, leaving a real brief's mood words leading on a groove bank. Half a
  fix.

  **THE FIX, one line:** on a `groove_arc` palette, append `palette.idiom`
  FIRST and the mood terms after it, so the genre and its BPM lead. Nothing else
  changes; sustained banks keep today's order exactly.

  **ALSO FOUND, a real bug in a bench tool:** `scripts/music_model_bench.py:295`
  derives the seed from the index into the FILTERED family list while the
  filename records only the family and a k-index, so the same filename means
  different seeds between a smoke run and a full run. That invalidates the
  determinism control cited in
  `docs/2026-09-12-music-model-bench/driver_anchor.md:94-96`. The conclusion
  there survives on other evidence; the cited proof does not.

### Still genuinely open, and not his call

* **`purple_cloud` cannot be vendored from pg11229 and that is now measured,
  not assumed.** The edition carries NO chapter divisions of any kind -- its
  only all-caps headings are the title, "INTRODUCTION" and "THE END.", and the
  body is broken solely by rows of asterisks. So `("chapters", 10, 11)` is
  unsatisfiable and no chapter pair would resolve. It wants either a different
  Gutenberg edition or a new chunk kind that slices on an explicit prose
  landmark. It is left REFUSING on purpose: the near-miss on `ghost_ship` the
  same day (a wrong anchor produced a clean 9,134-word "OK" line that had
  silently swallowed an entirely different story) is the argument against
  anchoring by feel. `ghost_ship` and `beleaguered_city` ARE vendored; the
  manifest went 65 -> 67 and lost nothing.

### Held deliberately, revisit when the thing they wait on lands

* **The 8 GB ship set** stays `draft` until the wave reports its physical 8 GB
  legs. **He ruled hold**, and it could not honestly be anything else yet.
* **`defaults.scene_coherence_check`** stays inert on every bank. **He ruled
  "not yet"**; story quality is closed and the offline corpus measurement was
  never run.
* **An IP-Adapter on the AnimateDiff lanes.** **He ruled hold.** It is a new
  dependency for the registry story and a recipe change on hard-won recipes;
  after the wave, if at all.
* **The ROCm recruitment post.** Written and pushed and waiting for him --
  `ROCM_MISSION_IMPOSSIBLE.md`, the hero still, two drafts in
  `docs/rocm-recruitment-post-draft.md`. **He said he will post it himself.**
  A window must never post it.

## 4. Constraints specific to this plan

Only the ones not already in CLAUDE.md or the standing rulings.

- Full listener source, no RSS. Cast count is flexible and records requested vs
  actual; the house announcer is excluded from dramatic cast.
- **WE DO NOT CHASE ACT COUNT** (operator, 2026-09-11), the same rule as word
  count: the value is a request, and a run delivers the closest performable
  episode.
- Model checking and a fixed attempt budget only -- no separate chunker, no
  recursive loop.
- An exhausted optional correction still yields a usable ledger; no predictive
  word or duration gate.
- Byline and attribution rules differ for My Story, Original and the adaptation
  banks.
- No replay, migration or re-render project: a saved input means fresh
  generation.

## 5. Parked

Parked and tombstoned items are preserved in
[GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md), not here -- by this file's own rule
they are not work. That includes the unqualified installed-family and GGUF opt-in
combinations, the H3 policy receipts, the cfg promotion comparisons, the AMD
scoped pod and platform acceptance, the cloud billing opt-in routing, the
operator-parked casting/adaptation ideas, OTR-Lite after v2, and the release
runway.

## 6. And when all of this is done -- it is time to TEST. Hurrah.

When sections 1 and 2 are empty, the waiting is over and the fun part starts.
Freeze ONE hash, write it into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md), and turn four
machines loose on it at once -- the 5080, the 4060, the Mac and a RunPod box,
each running the real canonical workflow, all reporting home.

Everything they owe is already written and waiting in
[COVERAGE_OWED](2026-09-11-four-machine-test-wave/COVERAGE_OWED.md) beside the
two lane documents. Nothing needs planning when the day comes; it needs starting.

Until then: **do not freeze a head, do not book a leg, and do not settle a row up
there by rendering something.** Two heads were cut early on 2026-09-11 and both
had to be withdrawn. An arc that can only be answered by live evidence is
answered without it, or cut with the reason written in.

**Then the next morning begins in `otr/obs/`, not in the editor** -- count what
landed against what was promised, read the four phone-homes, and triage anything
crash-class first. That is the good problem to have.
