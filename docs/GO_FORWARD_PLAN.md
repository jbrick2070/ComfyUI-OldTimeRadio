# OTR Go-Forward Plan

**THE ONE RULE FOR THIS FILE: ONLY UNFINISHED WORK BELONGS HERE.** When work
finishes, its receipt moves to [HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence
folder and the row leaves this page.

**THE ONE EXCEPTION (operator, 2026-09-11): finished work is named ONLY when a
REMAINING row in this plan depends on it.** *"We don't call out done work unless it's
a dependency for future work in the go-forward plan itself."* So:

* A prerequisite that is already satisfied gets **one clause inside the row that needs
  it** -- "the deterministic tier is in, so what remains is X" -- because a reader who
  cannot tell a satisfied prerequisite from an unstarted one will either redo it or
  block on it. That clause is the row's *starting line*, not its history.
* Everything else that is done is **gone from this page**: no receipts, no "SHIPPED"
  rows, no struck-through rows, no measurement write-ups, no records of what was
  refused and why, no arc summaries.
* **The test, and it is one question: does a row still in this file stop making sense
  without that sentence?** No -> cut it. It is a receipt wearing context's clothes,
  and this file collects them faster than it collects work.

Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md) first; this
file does not restate them. **For what has already happened -- commits, measurements,
receipts -- read [HANDOFF_LOG](HANDOFF_LOG.md), newest entry first.**

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. Exactness is not the goal: *"I'm not expecting anything exact."*

**CRASH-CLASS AND DURABILITY-CLASS DEFECTS ARE THE WORK** -- an uncaught exception, a
live asset written where a sweeper can delete it, an identity that silently resolves
outside its episode. Rows below use the phrase "not crash-class" against this
definition. Aesthetic drift is closed and is not work; see
[ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md).

## 1. The sequence

**ARCS -> CODING -> SHAKESPEARE -> TESTING, ABSOLUTELY LAST (operator directive
2026-09-11 -- hard, and it governs everything else in this file).**

| phase | what it means | done when |
|---|---|---|
| 1. ARCS | Every row whose design has more than one defensible answer gets its round BEFORE any code. An arc IS coding; it costs a wait, not a budget. | No row in section 5 is waiting on a design decision. |
| 2. CODING | Build what the arcs settled, plus every row that never needed one. | Section 5 is empty of buildable work. |
| 3. TESTING | Freeze ONE hash, then four machines qualify it. | Section 4's coverage is owed against a real head. |

**A DEFERRED ARC IS DEFERRED CODING**, and a gate on evidence the wave produces is not
a valid deferral either, because the wave runs last. An item that can only be decided
by live evidence is decided WITHOUT it or explicitly cut with the reason in its row.

**What genuinely defers is a two-item list:** a row blocked on an operator ruling, and
a row deliberately cut from this round with the reason written in it. Nothing else.
Apply that test whenever a row claims to be blocked.

**The head is NOT frozen.** It is frozen only when phases 1-3 are empty; the hash then
goes into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md) and the four lanes start.
Two heads were cut early on 2026-09-11 and withdrawn; do not cut a third early.

**After the wave, the next day begins in `otr/obs/`, not in the editor.** Count what
landed against the legs promised, read the four phone-homes, and triage any crash-class
failure FIRST. Only when that triage is empty does new work start.

## 2. The wave

Copy-paste prompts, one block per machine, self-contained by design:
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md) (5080, RunPod) and
[NATIVE_PLAN_4060_AND_MAC.md](2026-09-11-four-machine-test-wave/NATIVE_PLAN_4060_AND_MAC.md)
(4060, Mac). Lane assignments, per-lane legs and the rules every lane follows live
THERE and are not repeated here.

Not covered by the wave: **listening** -- the operator's own ear. No agent in it can
ingest audio, and reading TTS text or checking a waveform is not listening.

## 3. Open defects with an owner

| Defect | Next action |
|---|---|
| Caption burn kills an episode over its TITLE CARD | The classification is in place -- a probe-confirmed capability gap now raises `CaptionCapabilityGapError` -- so what remains is only the POLICY: should a probe-confirmed gap pass the clean master through, or keep refusing? **Genuinely two-sided, so it takes an arc.** Degrading the wrong class ships untitled episodes forever in silence; refusing throws away a finished episode over a cosmetic card at the end of a ~30-minute render. When it changes, `tests/test_caption_failures_are_classified.py::test_a_PLANNED_TITLE_still_refuses_on_a_capability_gap` is the test to rewrite deliberately. |
| `google_image` hard-fails where a sanctioned gap exists | Set `is_model_refusal = True` on the empty-image-block `GoogleAPIRequestShapeError` path in `eng_google_image.py` so `dispatch_images` routes it into `skip_evidence_by_oid` / `STATUS_SANCTIONED_GAP`, as `Ideogram4RefusalError` already does. **First settle the contradiction:** that file's own comment rules the opposite -- *"absent a structured refusal code, a completed-but-empty response is UNKNOWN, never an inferred refusal"* -- so this is a decision, not a wiring fix. Cloud lane; reaches no leg in a local wave. |
| Cast-count vocabulary disagrees across three surfaces | `_otr_casting.py` stamps the already-clamped value as `num_characters_request`; the writer builds an `EpisodeBudget` from the UNCLAMPED request; `_otr_episode_budget.py` labels that `cast_size`. For a request of 8 those surfaces say six and eight. Decide which surface owns "the count that happened". **Do not naively retarget the replay field:** `cast_lock.py:629` consumes `num_characters_request` to replay casting. Not crash-class. |
| Cast-time preflight resolves a DIFFERENT Ghost kernel than the render | The preflight's temporary shot is `shot_id = beat_id` (`otr_shot_lock.py:~1856`); the durable row is `shot_<beat_id>`. `render_driver` looks the ordinal up by exact `shot_id` (`~3280`), so the preflight always resolves at ordinal 0, and `resolve_crux_kernel` cycles the PLACE by ordinal -- the same object composes "in the archive" at preflight and "in the yard" on the row. Beat text lookup matches `shot_id` or `beat_id`, not `line_id`, so the preflight can also see no dialogue where the row sees the line. **Predates Half B and affected the deterministic tier identically**; found by the Half B round-2 contrarian. | Not crash-class: the preflight is an admission check (does the request build?) and the kernel value never makes it raise -- the durable row's kernel is what renders. Fix is to hand the preflight the prospective plan's canonical identity and ordinal, and resolve dialogue through `source_line_ids`. Then a behavioural test comparing resolver inputs between the two -- source-string assertions cannot establish equivalence. |
| OpenRouter catalog copy-forward | Decide whether an existing warm in-pack cache is copied forward on first run after the path move, or close it as accepted. A cold cache is already a designed safe state (empty catalog, logged fallback, never a raise, one refresh run restores it), so this is a nicety. |

## 4. Qualification coverage still owed

Every qualifying run loads `workflows/otr_canonical.json` through
`scripts/otr_canonical_api_run.py` -- no `--workflow` override, no replay substitute,
no `partial_execution_targets`. Assets go to canonical episode paths; final publication
must exist in `otr/obs`. Coverage that is simply one of the wave's own legs lives in the
lane documents, not here; these are the ones the wave does not already carry.

| Coverage | Required evidence |
|---|---|
| Six-act repeatability | Three fresh full canonical runs. Record observed loads/reuse; claim clean boot and resident reuse only where the runtime actually supports it. |
| Source and cast variety | One-act monologue, three-act ensemble, detailed six-act. Requested vs actual cast, supplied/neutral byline, breaks on and off. After the wave, check what its legs already covered and run only the gap. |
| Listening | Opening/middle/ending on at least two publications including a six-act. Operator's ear only. |
| 2.2 Ghost CUDA | Live five-act forced-Ghost CUDA publication with stored prompt/admission/reuse inspection. Needs a CUDA host. **Now also the first live measurement of Half B:** count `kernel_source` across the episode's shots -- `authored_subject` (the model ranked it), `key_object_in_beat` (the dialogue named it), `key_object` (odometer). The deterministic tier's ceiling was 26.3%; how far the ranked tier lifts it is the number nobody has. Report the `Ghost Half B: N/M ranked subject(s) admitted` log line too -- admission rate says whether the model is choosing from the list or inventing. |
| 5.7 chunked music | Full canonical on `otr_8gb_fastwan` (**that is the real profile id -- "fastwan_8gb" appears nowhere on disk**, and it is status `draft`) with long opening/closing cues, to prove the existing chunked-music repair. One leg, not a design row. |
| Inherited regression debt | **Load-bearing, not a footnote** -- it is what makes any "suite is green" claim mean something. Re-ground 51 OTR and 10 Bible failures against baseline IDs **and normalized payloads**. The Bible strict metadata validator separately has 149 unchanged issues. Never quarantine silently; never call the suite all-green. |

**The six-act fixture** is specified in full in PROMPTS.md, which restates it on purpose
so it can be pasted standalone. For every attempt record source fields,
code/canonical/graph hashes, actual model/quantization/profile, prompt ID, elapsed time,
memory and loads, all repair attempts, requested vs actual acts and cast, ledger seals
and final paths. Preserve terminal failure evidence before asserting anything. Keep the
full denominator.

## 5. The remaining work

### 5A. CODE IT -- the design is settled

| Row | What to write |
|---|---|
| ROCm tester recruitment (operator, 2026-09-11 night) | *"We need at some point to make a post on r/ROCm sooner than later, to see if they know a good way to rent a cheap cloud ROCm box, or if someone wants to volunteer testing -- but let's create a best-case-scenario JSON for them first ... a ROCM_MISSION_IMPOSSIBLE.md on the repo to tempt the palate of our wannabe tester, with a flattering image."* In order: (1) a generated variant of the canonical workflow pinned to the engines most likely to run on ROCm (pure PyTorch; no sageattention / flash-attn / cuda-malloc / bitsandbytes / CUDA-only GGUF kernels; the still floors, SA3 or musicgen, Kokoro/Bark) via `scripts/build_variants.py` -- never a hand-edited JSON; (2) `ROCM_MISSION_IMPOSSIBLE.md` at the repo root: clone + install (the registry serves alpha.24, git gives the tree), the variant to load, the one headless command, what success looks like (an mp4 in `otr/obs/`), what to report back (ledger, server log, `rocm-smi`, torch version, first traceback), what they get (credit), plus what a cheap ROCm rental needs (Linux, one MI-series or RDNA3 card, 16-24 GB, ROCm 6.x, ~50 GB disk); (3) a hero still under `docs/images/`; (4) a draft r/ROCm post for HIM to paste (never posted by a window). A side-quest chip was spawned for this the same night; if it was not started, this row is the work. |
| 2.4 audit tail, model-root | The four-owner model-root merge, or nothing. Do NOT rip `comfy_models_dir()` / `resolve_hf_model_path()` on a caller count: they are a dead CHAIN, the archive PARKED the convention, and a FOURTH spelling lives in `flux2_klein.py:209-215`. |

### 5B. NEEDS A MEASUREMENT OR A DECISION, not another round

| Row | What it actually needs |
|---|---|
| 2.4 routing/canvas | The 193-frame ceiling **cannot be asserted as production-proven** -- it is a lab-warm isolation number, and this lane has a live receipt of production peaks exceeding lab peaks. Needs a canonical-path qualification leg first, then a TWO-file diff (`frame_contract.PLANNING_CAP_ENGINES` **and** `otr_16gb_ltx_audio_in.json` `video.max_render_frames`) -- changing one alone narrows nothing. **Not render-inert:** `assert_coverage_plans` refuses any `ltx_audio_in` ledger planned before the change and rendered after, so in-flight episodes need replanning at cutover. |
| 3.5 per-beat reload | **Direction undercut by prior art.** Moving this same encoder off-GPU was already tried and REVERTED on live-measured evidence that it did not move the peak (PBUG-20260616-01 / BUG-07.17). The generic scope hook also closes on engine-CHANGE, not every beat, so "even consecutive same-engine beats start cold" was wrong. Per-lane encoder device defaults differ. Decide whether it is worth doing at all before any caching work. Not OOM-proven -- wall-clock cost. |
| 2.4 voice/credits | **The instrument does not fit the defect.** `high_band_edge_ratio` detects edge squeal; PBUG-20260902-03 documents a SUSTAINED TONE, and a synthetic reproduction of the exact documented frequencies scored ~0 against the real function. Closing this means designing and empirically qualifying a NEW whole-clip speech-shape scorer that does not exist in the tree. Real work, for a non-crash defect -- weigh against the bar before starting. |
| 3.4 clean install | **The crash half is CLOSED (2026-09-11, `6223b972` + `9b879207`):** a box with ffmpeg and no ffprobe now measures media through PyAV at every probe (boundary fallback; the composite, mux, scopes and blend routed through it), and a canonical Shakespeare leg on the 5080 with ffprobe made unresolvable reached obs (`laughter_in_the_shadows_20260911_214126`). The wave's Mac and RunPod paste blocks need no `ffdl install` step for ffprobe. What remains is the non-crash durability point: work from the full r1 review (`kibitz-runs/2026-09-11-arc-cleaninstall/r1/codex.md`), keep the existing download scope, narrow the early-tool-check proposal. Low priority. |

## 6. Blocked on the operator -- each unblocks with one word

* **A per-style SD1.5 checkpoint (your idea, 2026-09-11 night: "maybe we should
  be loading different SD1.5 models per visual pack -- an anime SD1.5 would really
  pop").** Today the still engine (`nodes/_otr_image_engines/sd15.py`) loads ONE
  checkpoint for every style, `v1-5-pruned-emaonly-fp16.safetensors`, and
  `OTR_SD15_CKPT` overrides it globally; the style catalog
  (`nodes/_otr_visual_styles.py`) has no per-style model field, and the anime
  style contributes only the words "anime style" to the prompt. The build is
  small and settled in shape: a `checkpoint` key per style in the catalog, the
  engine resolving it through the one resolver (style first, then the env, then
  the default), the file under `C:\ComfyUI-Models\checkpoints`, a variant note for
  8 GB cards (SD1.5 fits). **Unblocks with the checkpoint name you want for anime**
  (and any other style you want re-pointed); the measurement is one style, two
  checkpoints, your eye.
* **An IP-Adapter on the AnimateDiff lanes (your question, 2026-09-11 night:
  "wondering if we should be using an IP adapter so AnimateDiff can take ref
  images").** Grounded: the pack has five AnimateDiff SD1.5 lanes (the ghost-signal
  family in `nodes/_otr_video_engines/eng_ghost_signal*.py`: v2, v3, v3 haunted,
  Lightning, and the still-conditioned lab lane, which already takes the beat's
  own still as its init image); AnimateLCM was deliberately refused by the ghost
  recipe (documented at `eng_ghost_signal.py:102`, the recipe runs cfg 1.0 with
  its own scheduler pair); and there is NO IP-Adapter anywhere in the pack. An
  IP-Adapter drives STYLE from a reference image (CLIP-vision, not the init
  latent) and would need the IP-Adapter model files plus a node that ComfyUI
  does not ship natively (the IPAdapter-plus pack), so it is a new dependency
  decision for the registry story, and a video-lane recipe change (recipes are
  hard-won: one lane, A/B against itself, your eye). The other three tricks in
  the tutorial you pasted the pack already does: the ghost lanes lean on grain
  and degradation to hide morphing, they run low cfg, and the title / timecode /
  captions are burned in post by the caption burn, not generated. **Unblocks
  with: which lane to try it on and where the reference image comes from (the
  episode's own style card, the beat still, or a fixed mood board).**
* **5.1-5.2 release after physical 8 GB proof.** After the wave reports the 4060's
  physical 8 GB legs, rule on promoting the 8 GB ship set.
* **Unruled product choices and Bible fan-out candidates.** The pointer is to a
  2026-09-01 catch-all; name the live sub-questions explicitly before asking, so a
  ruling lands on something specific.

## 7. Constraints specific to this plan

Only the ones not already in CLAUDE.md or the standing rulings.

- Full listener source, no RSS. Cast count is flexible and records requested vs actual;
  the house announcer is excluded from dramatic cast.
- **WE DO NOT CHASE ACT COUNT** (operator, 2026-09-11), the same rule as word count: the
  value is a request, and a run delivers the closest performable episode.
- Model checking and a fixed attempt budget only -- no separate chunker, no recursive
  loop.
- An exhausted optional correction still yields a usable ledger; no predictive word or
  duration gate.
- Byline and attribution rules differ for My Story, Original and the adaptation banks.
- No replay, migration or re-render project: a saved input means fresh generation.

## 8. Parked

Parked and tombstoned items are preserved in
[GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md), not here -- by this file's own rule they
are not work. That includes the unqualified installed-family and GGUF opt-in
combinations, the H3 policy receipts, the cfg promotion comparisons, the AMD scoped pod
and platform acceptance, the cloud billing opt-in routing, the operator-parked
casting/adaptation ideas, OTR-Lite after v2, and the release runway.
