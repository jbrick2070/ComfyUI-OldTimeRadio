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

| Row | The fork, and what settles it |
|---|---|
| **Which SA3 recipe ships** | **Re-measure; the old numbers are void.** Every arm was measured on the post-trained checkpoint, where cfg and the negative prompt do nothing, so the comparison attributed to guidance what was really sampler and steps -- and it scored them with loopiness, which a burst pushes DOWN. The live fork is narrow: on the BASE checkpoint at cfg 4.0, `lcm`/`simple` at 50 steps against `dpmpp_3m_sde_gpu`/`exponential` at 100. Half the render time is worth an A/B. Several renders per arm through `scripts/otr_music_ab.py` (canonical only) -- one leg per arm cannot answer it; the variance between two cues inside one arm has twice beaten the difference between arms. Read burst count beside loopiness, never loopiness alone. The 2026-09-12 music-model bench (`docs/2026-09-12-music-model-bench/`) held this recipe constant across checkpoints, so the recipe is a confound this row owns; run the A/B on whichever checkpoint the section-3 "small or medium" ruling picks. |
| **Why house and techno do not groove** | **The loop complaint and the groove complaint are now two different lanes, and neither has been heard yet.** Since the genres shipped, `negative_for()` gives the four declared banks the rhythm-friendly negative on purpose -- the anti-loop wording now guards only Shakespeare and an undeclared bank, and it became a live lever at all only when the base checkpoint was preferred. So the sustained lane has never once been heard with its own anti-loop negative actually working, and the genre lane is being asked FOR a repeating figure. This row is the second half. Four 1-act legs put a genre on each bank: salsa and jazz produced real grooves (139 and 83 onsets a minute), techno and house came back nearer pads. The sci-fi news lane is explained -- it AUTHORS its own music rows, which bypass the composed genre row, so the cue asked for a TR-909 and "tense strings" in one prompt. **Chicago house has no such excuse: its row was COMPOSED and it still came back a pad.** n=1 per bank, and the conditioning window is already excluded (measured, 10 seeds per arm: 1x, 2x and 3x all find a beat 80-90% of the time at the requested tempo). Take it to the bench before touching the palette. **Bench, 2026-09-12** (`docs/2026-09-12-music-model-bench/`): the composed house and techno opening cues, three seeds each on both base checkpoints, all twelve committed to the asked tempo (house 120-123, techno 129; pulse 0.5-0.8) on a stock-node graph -- the n=1 pads did not reproduce. What remains is whether the canonical path differs, which is a listen (section 3), not a palette change. |
| **Cast-count vocabulary disagrees across three surfaces** | `_otr_casting.py` stamps the already-clamped value as `num_characters_request`; the writer builds an `EpisodeBudget` from the UNCLAMPED request; `_otr_episode_budget.py` labels that `cast_size`. For a request of 8 those surfaces say six and eight. Decide which surface owns "the count that happened". **Do not naively retarget the replay field:** `cast_lock.py:629` consumes `num_characters_request` to replay casting. Not crash-class. |
| **2.4 routing/canvas -- allowlist the lane at all?** | The 193-frame ceiling **cannot be asserted as production-proven** -- it is a lab-warm isolation number, and this lane has a live receipt of production peaks exceeding lab peaks. The engine already rejected capping its own declaration, and said why: `ltx_audio_in` is absent from `frame_contract.PLANNING_CAP_ENGINES`, so a ceiling there reaches nothing (`eng_ltx_av.py:1664-1672`). **Decide it WITHOUT a leg** -- this file's own rule: an item that can only be settled by live evidence is settled without it or cut with the reason, because the legs run last. The two defensible answers are: allowlist the lane now and let a leg PROVE the partition afterwards, or cut it and record that 193 stays an unenforced lab number with the engine comment as the honest statement. If the answer is yes it is three artifacts, and any one alone narrows nothing: add `ltx_audio_in` to `PLANNING_CAP_ENGINES`, add `video.max_render_frames` to `otr_16gb_ltx_audio_in.json` (it has no such key today; the three capped profiles carry it at `video.max_render_frames = 81`), then regenerate the variants (`build_variants.py --all`, confirm `--check`) -- `_otr_workflow_apply.py:553` flattens that key ONLY when present, so an un-regenerated variant carries the old ceiling silently -- and prove the multi-clip partition. **Not render-inert:** `assert_coverage_plans` refuses any `ltx_audio_in` ledger planned before the change and rendered after, so in-flight episodes need replanning at cutover. |
| **2.4 model-root audit tail** | The four-owner model-root merge, or nothing -- and "nothing" is a real answer. Do NOT rip `comfy_models_dir()` / `resolve_hf_model_path()` on a caller count: they are a dead CHAIN, the archive PARKED the convention, and a FOURTH spelling lives in `nodes/_otr_image_engines/flux2_klein.py:214-220`. |
| **3.5 per-beat reload** | **Direction undercut by prior art.** Moving this same encoder off-GPU was already tried and REVERTED on live-measured evidence that it did not move the peak (PBUG-20260616-01 / BUG-07.17). The generic scope hook also closes on engine-CHANGE, not every beat, so "even consecutive same-engine beats start cold" was wrong. Per-lane encoder device defaults differ. Decide whether it is worth doing at all before any caching work. Not OOM-proven -- wall-clock cost. |
| **2.4 voice/credits** | **The instrument does not fit the defect.** `high_band_edge_ratio` detects edge squeal; PBUG-20260902-03 documents a SUSTAINED TONE, and a synthetic reproduction of the exact documented frequencies scored ~0 against the real function. Closing this means designing and empirically qualifying a NEW whole-clip speech-shape scorer that does not exist in the tree. Real work, for a non-crash defect -- weigh against the bar before starting. |
| **OpenRouter catalog copy-forward** | Decide whether an existing warm in-pack cache is copied forward on first run after the path move, or close it as accepted. A cold cache is already a designed safe state (empty catalog, logged fallback, never a raise, one refresh run restores it), so this is a nicety. |

## 2. CODE -- the design is settled, build it

| Row | What to write |
|---|---|
| **Bark output guard** (from the 2.4 voice/credits arc, codex refutation 2026-09-12) | `docs/PROD_BUG_LOG.md` PBUG-20260902-03 is STATUS FIX-OPEN and names its own fix (`:10298`): score each bark generation for speech shape -- the fraction of one-second windows whose dominant frequency sits in 70-400 Hz with spectral flatness under 0.2 -- re-roll with `seed + 1` up to twice when it fails, log every re-roll at WARNING with the score, keep the best-scoring take if all three fail; the ledger field is always filled. The record calls its absence *"a silent wrong render"*, which is the bar. Bounded to `eng_bark`'s generate path; kokoro stays the default; the two archived artifacts in the PBUG are the calibration set (the 7-second noise floor and the 2.5 kHz tone must score near zero, the two speech takes near one). `high_band_edge_ratio` is the wrong instrument and is left alone. |
| **Cast-time preflight resolves a DIFFERENT Ghost kernel than the render** | The preflight's temporary shot is `shot_id = beat_id` (`otr_shot_lock.py:1893`); the durable row is `shot_<beat_id>`, and the render driver looks the ordinal up by exact `shot_id` (`:3269`), so the preflight always resolves at ordinal 0 while `resolve_crux_kernel` cycles the PLACE by ordinal -- the same object composes "in the archive" at preflight and "in the yard" on the row. Beat text lookup matches `shot_id` or `beat_id`, not `line_id`, so the preflight can also see no dialogue where the row sees the line. Not crash-class: the preflight is an admission check and the kernel value never makes it raise. Hand the preflight the prospective plan's canonical identity and ordinal, and resolve dialogue through `source_line_ids`. Then a behavioural test comparing resolver inputs between the two -- source-string assertions cannot establish equivalence. |
| **Four tests fail in isolation** | `tests/test_unified_memory_weight_floor.py` and `tests/test_ghost_signal_lightning_lane.py` carry four tests that pass in a full run and fail alone -- re-verified against the pushed head, so this is test hygiene, not a regression. It matters because it makes any focused subset run untrustworthy as evidence. Find the shared state and pin it, or mark them as requiring the full run. |
| **ROCm tester recruitment** (operator, 2026-09-11 night) | *"We need at some point to make a post on r/ROCm ... let's create a best-case-scenario JSON for them first ... a ROCM_MISSION_IMPOSSIBLE.md on the repo to tempt the palate of our wannabe tester, with a flattering image."* **The variant half already exists** -- `otr_amd16_rocm` and `otr_amd8_rocm` are generated draft variants with launch recipes, sage/bnb/fp8 already off, and both are labelled UNVERIFIED on hardware, which is precisely what the tester is being recruited to change. So: (1) confirm those two pin the pure-PyTorch engine set (no sageattention / flash-attn / cuda-malloc / bitsandbytes / CUDA-only GGUF kernels; the still floors, SA3 or musicgen, Kokoro/Bark) and regenerate through `scripts/build_variants.py` if not -- never a hand-edited JSON; (2) `ROCM_MISSION_IMPOSSIBLE.md` at the repo root: clone + install, the variant to load, the one headless command, what success looks like (an mp4 in `otr/obs/`), what to report back (ledger, server log, `rocm-smi`, torch version, first traceback), what they get (credit), plus what a cheap ROCm rental needs (Linux, one MI-series or RDNA3 card, 16-24 GB, ROCm 6.x, ~50 GB disk); (3) a hero still under `docs/images/`; (4) a draft r/ROCm post for HIM to paste -- never posted by a window. Check the side-quest chip before starting. |
| **3.4 clean install, durability tail** | The crash half is closed, so what remains is the non-crash durability point: work from the full r1 review (`kibitz-runs/2026-09-11-arc-cleaninstall/r1/codex.md`), keep the existing download scope, narrow the early-tool-check proposal. Low priority. |
| **Name the unruled product questions** | Section 3's last bullet points at a 2026-09-01 catch-all. Enumerate the live sub-questions inside it and the Bible fan-out candidates, then move each to section 3 as its own one-line bullet. A ruling needs something specific to land on. |

## 3. Blocked on the operator -- each unblocks with one word

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
* **Unruled product choices and Bible fan-out candidates.** Waiting on section
  2's row to name the live sub-questions; a ruling needs something specific.

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
