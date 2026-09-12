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
* **The anime checkpoint, on screen.** `Counterfeit-V3.0_fp16.safetensors` is
  fetched (4.24 GB, in `C:\ComfyUI-Models\checkpoints`) and the anime pack now
  names it. A pack's checkpoint is a PREFERENCE and never a gate: a box without
  the file falls through to the env override and then the shipped default, so
  nothing greys out and no render can fail on it. **Unblocks with your eye on
  an anime-style still minted through it** -- one style, two checkpoints.

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
