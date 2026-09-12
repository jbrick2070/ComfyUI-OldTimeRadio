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

* **Post the ROCm recruitment, or hand it back.** The pack is written and
  pushed: `ROCM_MISSION_IMPOSSIBLE.md` at the repo root (the install, the two
  profile ids, one headless command, what success looks like, what to send
  back), a hero still at `docs/images/rocm_mission_hero.jpg`, both AMD
  profiles confirmed pure-PyTorch with `build_variants --check` clean, and two
  drafts of the post itself in `docs/rocm-recruitment-post-draft.md`. **A
  window must never post it.** **Unblocks with posted, or with your edits.**

* **Does a declared bank genre outrank an AUTHORED music row?** `scifi_news_pro`
  writes its own ledger music rows and they bypass the composed genre row, so
  that bank's cue asked for a TR-909 and "tense strings" in the same prompt and
  came back a pad. *"Sci-fi news is Detroit techno"* reads as yes, but authored
  rows are a deliberate feature of that lane and overriding them is a behaviour
  change on a path nobody has studied. **Unblocks with yes or no.**
* **Listen to the four bank genres.** Eight clips, labelled by bank, in
  `output/otr/obs/bank_genres/` (`1_scifi_news_DETROIT_TECHNO_*` through
  `4_original_SALSA_*`). **Unblocks with which ones are right.**
* **Listen for the tape scratch.** `the_jars_secret_20260912_024354` (base
  checkpoint, cfg 4) against `the_borrowed_voice_20260912_023252` (post-trained,
  cfg 7) -- the odd tape-loop scratch at ~0:47 should be gone from the first.
  **Unblocks with gone or still there.**
* **Listen for the loop on the SUSTAINED lane.** A Shakespeare leg, which is now
  the only bank still receiving the anti-loop negative -- and that negative has
  never been heard working, because it was inert on the old checkpoint. The
  original loop complaint was never actually tested. **Unblocks with looping or
  not.**
* **Small or medium.** Eighteen paired clips per arm in
  `output/otr/obs/music_model_bench/` (same family and seed number across
  arms; the README.txt inside says what each is). On the bench
  (`docs/2026-09-12-music-model-bench/`) medium base rendered NO pulse on
  sustained cues where small base has one, was mixed on the genre lanes,
  and never came back hot -- at 9.22 GB on disk, +3.4 GB of VRAM, +2.4 s a
  cue, and unproven on 8 GB. The engine keeps preferring small base until
  you say; "medium" is one tuple, one fetch lane and a 4060 proof.
  **Unblocks with small or medium.**
* **A registry version carrying this week's work.** `pyproject.toml` is still
  `2.0.0-alpha.30`, published 2026-09-11; seven commits have touched `nodes/`
  since, including the guidance fix, the render-killing build-breaker and the
  bank genres. Editing that file auto-fires a publish and the registry push is
  yours. **Unblocks with "publish alpha.31".**
* **A per-style SD1.5 checkpoint** (your idea, 2026-09-11 night: *"maybe we
  should be loading different SD1.5 models per visual pack -- an anime SD1.5
  would really pop"*). Today the still engine (`nodes/_otr_image_engines/sd15.py`)
  loads ONE checkpoint for every style, `v1-5-pruned-emaonly-fp16.safetensors`,
  and `OTR_SD15_CKPT` overrides it globally; the style catalog
  (`nodes/_otr_visual_styles.py`) has no per-style model field, and the anime
  style contributes only the words "anime style" to the prompt. The build is
  small and settled in shape: a `checkpoint` key per style in the catalog, the
  engine resolving it through the one resolver (style first, then the env, then
  the default), the file under `C:\ComfyUI-Models\checkpoints`, a variant note
  for 8 GB cards. **Unblocks with the checkpoint name you want for anime** (and
  any other style you want re-pointed); the measurement is one style, two
  checkpoints, your eye.
* **An IP-Adapter on the AnimateDiff lanes** (your question, 2026-09-11 night:
  *"wondering if we should be using an IP adapter so AnimateDiff can take ref
  images"*). Grounded: the pack has **three registered** AnimateDiff SD1.5 lanes
  -- v3 haunted, Lightning, and the still-conditioned lab lane, which already
  takes the beat's own still as its init image (the plain v2 and v3 ids are
  tombstoned in `RETIRED_ENGINE_IDS` since 2026-08-23). AnimateLCM was
  deliberately refused because the main recipe runs `GHOST_CFG = 8.0` with a LIVE
  negative and the Ghost lettering defense needs real unconditional conditioning
  (`eng_ghost_signal.py:101-106`); only the Lightning lane runs cfg 1.0. There is
  NO IP-Adapter anywhere in the pack. An IP-Adapter drives STYLE from a reference
  image (CLIP-vision, not the init latent) and would need the IP-Adapter model
  files plus a node ComfyUI does not ship natively (the IPAdapter-plus pack), so
  it is a new dependency decision for the registry story and a video-lane recipe
  change -- recipes are hard-won: one lane, A/B against itself, your eye. The
  other three tricks in the tutorial you pasted the pack already does: the ghost
  lanes lean on grain and degradation to hide morphing, and the title / timecode
  / captions are burned in post, not generated. **Unblocks with: which lane to
  try it on and where the reference image comes from** (the episode's own style
  card, the beat still, or a fixed mood board).
* **The 8 GB ship set.** Promote the 4060 profiles out of `draft` after the wave
  reports its physical 8 GB legs. **Unblocks with promote or hold.**
* **Arm `defaults.scene_coherence_check` on any bank?** The vacuity fix
  shipped 2026-08-28 but stays opt-in and inert everywhere
  (`nodes/_otr_scene_guard.py:19`, "default False -> INERT"), and the offline
  corpus measurement was never run. **Unblocks with a bank name, or not yet.**
* **Ghost names: scrub the brief after cast lock, or propagate the pitch's
  names?** Pitch-invented names never reach `lock_cast`, so a bio can open on
  a name the locked cast does not use -- the Fogbound Rails bio still opens
  "Lizzie Gray". **Unblocks with scrub or propagate.**
* **Does `media_archive` get the same no-premise-scaffold treatment as
  `original`?** The five-bank beat test caught it drawing an unrelated
  premise scaffold over its own catalogued item; the scaffold-off rule has
  only ever been stated for `original`. **Unblocks with yes or no.**
* **Spend the one Gutenberg fetch to vendor the three refusing works?**
  `ghost_ship`, `purple_cloud` and `beleaguered_city` still fail the
  vendoring parser. Operator opt-in, not schedulable inside an offline
  sprint. **Unblocks with go or skip.**
* **Give `style_tail_policy` a third token, or rule the `ltx_radio_face`
  path exempt?** `build_radio_host_prompt`'s `ltx_radio_mouth` branch returns
  early and skips the tail the `ltx_audio_in` bookend declares
  (`otr_meta_brief_image_prompt.py:196,253,286`). Unruled, the exemption
  stands. **Unblocks with third token or exempt.**
* **Does a 24 GB machine class get its own row?** No entry in
  `config/machine_classes.json`; rentals file under the 16 GB class today.
  Low priority. **Unblocks with yes or no.**
* **Run the Bible fan-out on this week's fixes.** PBUG-20260911-06 and -07
  and PBUG-20260912-01 through -05 are each live-verified with a published
  artifact and fixed, and none carries a Bible id yet. (The older
  "awaiting fan-out" strings in the log are stale status text, not work:
  their promotions already exist in the Bible.) **Unblocks with go.**

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
