# ARC batch 2 -- seven section-1 rows, settled

Driver: Claude Fable 5.1 (Cowork, 5080), sole judge. Written 2026-09-12 after
the music-model arc landed (`595201ec`). Operator: *"keep arcing and coding so
we can get to testing."* Every row below is settled WITHOUT a canonical leg;
the one measurement is a stock-node bench arm, which the plan's own rule
allows. Testing stays closed.

Reviewer for the batch: one codex contrarian over this whole document,
briefed to refute; dispositions in section 9.

## 1. Which SA3 recipe ships -- the shipped recipe stays

**The fork.** On the base checkpoint at cfg 4.0: `lcm` / `simple` at 50 steps
(Comfy-Org's own SA3 base template recipe, half the render time) against the
shipped `dpmpp_3m_sde_gpu` / `exponential` at 100 steps (the Stable Audio 1.0
template's pair). The row asked for several renders per arm, read burst count
beside loopiness.

**The bench** (`recipe_bench.py` beside this file; per-render JSON in
`bench_recipe.json` and `bench_recipe_with_repeat.json`; 36 MP3s plus a
README under `output/otr/obs/music_recipe_bench/`). Same six prompt families,
same three seeds and same measures as the music-model bench, ONE checkpoint
(`stable_audio_3_small_music_base`, the fetch default) at the engine's base
guidance 4.0; the sampler recipe is the only variable. Against a fresh
server boot; the dpmpp arm reproduced this morning's music-model-bench
numbers to the third decimal, so the bench is deterministic across boots.

| family | arm | bursts | loopiness | repeat | pulse | tempo | peak dBFS | RMS dBFS | wall s |
|---|---|---|---|---|---|---|---|---|---|
| organ_cathedral | dpmpp100 | 0 | 0.78 | 0.47 | 0.50 | 108 | -3.3 | -16.0 | 7.5 |
| organ_cathedral | lcm50 | 0 | 0.79 | 0.43 | 0.48 | 108 | -6.4 | -21.4 | 4.5 |
| organ_toccata | dpmpp100 | 0 | 0.60 | 0.18 | 0.31 | 108 | -2.8 | -16.4 | 7.6 |
| organ_toccata | lcm50 | 0 | 0.81 | 0.30 | 0.42 | 114 | -11.2 | -27.0 | 4.5 |
| shakespeare_opening | dpmpp100 | 0 | 0.80 | 0.23 | 0.44 | 115 | -1.9 | -14.9 | 7.1 |
| shakespeare_opening | lcm50 | 0 | 0.84 | 0.14 | 0.51 | 122 | -13.8 | -26.9 | 3.6 |
| shakespeare_closing | dpmpp100 | 0 | 0.59 | 0.17 | 0.32 | 115 | -3.8 | -17.5 | 7.5 |
| shakespeare_closing | lcm50 | 0 | 0.84 | 0.00 | 0.45 | 115 | -8.8 | -23.9 | 3.0 |
| house_opening | dpmpp100 | 2 | 0.52 | 0.44 | 0.78 | 121 | -1.1 | -16.7 | 7.1 |
| house_opening | lcm50 | 9 | 0.86 | 0.80 | 0.82 | 121 | -1.5 | -20.6 | 4.1 |
| techno_opening | dpmpp100 | 1 | 0.77 | 0.51 | 0.63 | 129 | 0.0 | -14.1 | 7.5 |
| techno_opening | lcm50 | 7 | 0.77 | 0.64 | 0.66 | 129 | -1.1 | -17.2 | 3.6 |

Per arm: dpmpp100 3 bursts, mean 8.3 s (7.5 s once the model is warm);
lcm50 16 bursts, mean 3.9 s. (The dpmpp wall figure includes one 24 s
first-render model load.)

**Read the way the music-model anchor's section 4 says to.**

* **Level, and it is the decider.** The lcm arm renders the sustained lane
  8-13 dB quieter -- consort opening at -14 dBFS peak / -27 dBFS RMS against
  -2 / -15 -- and the organ pieces 3-8 dB quieter. The pipeline's music bus is
  a ceiling with no gain-up (`stable_audio_theme._ceiling_the_cue`), so a
  cue that leaves the decoder at -27 dBFS RMS ships at -27 dBFS RMS under the
  dialogue. That is a durability point: the bed level would depend on the
  sampler, not the mix.
* **The sustained lane gets MORE beat-like on lcm**, not less: pulse 0.31-0.68
  against dpmpp's 0.13-0.46 across the four consort cues, and loopiness up
  on every sustained family (0.84 / 0.84 against 0.80 / 0.59). The closing
  cue's `repeat` of 0.00 on all three lcm seeds is an envelope with no local
  maximum between 1 and 6 s at all -- a swell with no phrase, which at -24
  dBFS RMS is close to a fade.
* **On the genre lanes lcm commits to tempo just as well** (121 / 129 on
  every seed) with slightly higher pulse (0.82 / 0.66 against 0.78 / 0.63),
  and it runs cooler on techno (no seed at full scale against dpmpp's three).
  But it carries **five times the bursts** (16 against 3, all on the
  rhythmic lanes), and a burst count that jumps 5x on the same prompts is a
  brighter, noisier top end whatever the hi-hat caveat says about the
  detector.
* **What lcm would buy:** about 3.6 s a cue, three cues an episode, so
  roughly eleven seconds on a ~24 minute run.

**Decision: the shipped recipe stays** (`OTR_SA3_SAMPLER=dpmpp_3m_sde_gpu`,
`OTR_SA3_SCHEDULER=exponential`, `OTR_SA3_STEPS=100`, cfg from the checkpoint).
Eleven seconds an episode does not pay for beds that are 8-13 dB quieter,
more periodic on the one lane the anti-loop negative guards, and burstier on
the genre lanes. The row's "half the render time is worth an A/B" was right
to ask and the answer is no. The env knobs stay, so the harness can still run
the control arm. No code changes; the row leaves the plan. If the operator
picks medium in the section-3 ruling, this bench is one command to re-run on
that file (`--ckpt` in `recipe_bench.py`), and the expectation is the same.

**Why a server-side stock bench and not the canonical harness.** The row
asked for `otr_music_ab.py` because an earlier IN-PROCESS bench could not
reproduce a shipped cue (correlation 0.92, nine dB hotter). This bench
renders against the running server through the same node classes the engine
drives, with the composer's own prompts, and it reproduces itself exactly
across boots. It is still not the canonical path, and it qualifies nothing;
it settles a design fork, which is what the plan says a bench arm may do
while testing is closed.

## 2. Cast-count vocabulary -- cut; the count that happened is the replay key

**Facts, grounded** (Sonnet reader, spot-checked by the driver at
`_otr_episode_budget.py:243-256` and `OTR_LedgerScriptWriter.py:4420-4424`):

* `lock_cast` clamps an over-request to `_LEGACY_MAX_SPEAKING_CAST = 6`
  (`nodes/_otr_casting.py:1176`, `:1980-1987`) and stamps the clamped value
  as `num_characters_request` (`:2362`).
* The writer hands `compute_episode_budget` the RAW widget value
  `resolved["num_characters"]` (`OTR_LedgerScriptWriter.py:4420-4424`),
  bounded only 1-10 by `_otr_writer_inputs.py:64`.
* `compute_episode_budget` uses that number for exactly two things: a
  `>= 1` check and the `cast_size` label (`_otr_episode_budget.py:243-256`).
  Nothing else in the budget derives from it.
* **`EpisodeBudget.cast_size` has no production reader.** Repo-wide the only
  reference is the dataclass unit test (`tests/test_phase2a_episode_budget.py:67`);
  `EpisodeBudget` is never serialised into a ledger. The consumed fields are
  `act_count`, `arc_phases`, `per_phase_beats`, `music_inter_count`.
* `cast_lock.py:629` rebuilds the voice-assignment RNG from
  `num_characters_request`; it must keep receiving the CLAMPED count or a
  replay re-casts a different sequence -- that would be the durability defect.

**Decision: cut.** The surface that owns "the count that happened" is the
cast contract's `num_characters_request` -- clamped, stamped, and the replay
key -- and it is already right. The disagreeing field is a write-only label
on dead data, which is vocabulary, not a defect under the bar. If a reader
of `cast_size` is ever written, the one-line honest fix is to pass
`cast_meta["num_characters_effective"]` at `OTR_LedgerScriptWriter.py:4420-4424`
in the same change; never retarget the replay key.

## 3. 2.4 routing/canvas allowlist -- cut; the engine comment is the record

**Facts, grounded** (Sonnet reader):

* `PLANNING_CAP_ENGINES = ("ltx_8gb", "fastwan_8gb", "wan_ti2v")`
  (`frame_contract.py:319`); `ltx_audio_in` is absent, and for an engine
  outside the tuple `frame_contract.py:384` returns the contract unchanged,
  so today's state is inert by construction.
* `eng_ltx_av.py:1664-1672` already records that capping the declaration at
  193 was considered and rejected; `:1639-1646` calls 193 a lab-warm number
  at 1024x576 and notes the live receipt of production peaks above lab.
* `otr_16gb_ltx_audio_in.json:59-62` carries no `video.max_render_frames`.
  **The row miscounted the capped profiles: FIVE carry `= 81`, not three**
  (`otr_8gb_fastwan`, `otr_8gb_wan`, `otr_g4_fastwan`, `otr_g4_wan_ti2v`,
  `otr_w45_fastwan`), and `otr_w45_wan_ti2v` uses a capped engine with no
  ceiling at all. The two other `ltx_audio_in` profiles (`otr_g4_...`,
  `otr_w45_...`) also carry none.
* `_otr_workflow_apply.py:551-554` flattens the key only when present;
  `render_driver.py:5844-5876` (`assert_coverage_plans`) re-derives the
  expected partition from the CURRENT allowlist, so an in-flight
  `ltx_audio_in` ledger would be refused at cutover.
* Multi-clip for this lane is new runtime territory, not a config flip:
  `coverage_plan.py:500-506` refuses `JOIN_CHAIN` for adapters without the
  strict continuity declaration, and whether `eng_ltx_av`'s own segment
  render re-conditions still and audio slice for segments 2..N has never
  been exercised.

**Decision: cut.** Allowlisting would route an audio-conditioned lane
through an unexercised multi-segment path and force replanning of in-flight
episodes, to enforce a ceiling nobody has proven, and the only way to prove
it is a leg. Cutting leaves NO crash-class exposure: the lane renders today
exactly as it did yesterday, and the engine comment is the honest statement
the row's own cut branch asked for. 193 stays an unenforced lab number.

## 5. 3.5 per-beat reload -- cut under the bar; if ever wanted, it is CODE

**Facts, grounded** (Sonnet reader):

* PBUG-20260616-01 / BUG-07.17 (`PROD_BUG_LOG.md:412-421`): the Gemma-3
  encoder was moved to CPU in `b0925c37` and reverted in `1e5d66f4` because
  the live-measured peak did not move.
* `_begin_engine_scope` is idempotent across consecutive same-engine beats
  (`render_driver.py:5271-5299`), so the SCOPE closes on engine change --
  the row's correction holds for the hook. **But it is misleading for the
  row's own subject engines:** `eng_ltx_video` and `eng_ltx_av` rebuild
  their unet + encoder + VAE + LoRA graph from disk every beat through
  `motion_common.py:1359`, so consecutive same-engine beats DO start cold
  there. The cost is real; it is also unmeasured -- the only load-time
  number in the tree (~63 s) is `eng_ltx25`'s, a different engine and file.
* Per-lane encoder device defaults differ: `eng_ltx_video.py:840` defaults
  to GPU, `eng_ltx_av.py:910` and `:1079` to CPU.
* No OOM or crash entry in the production log is tied to this.

**Decision: cut.** Wall-clock only, unmeasured, no crash-class or
durability-class defect, and its former justification (feeding an orphan
occupancy row) no longer exists in the plan. If it is ever wanted it is not
an arc: extend `eng_ltx25`'s `begin_encoder_scope` / `end_encoder_scope`
pattern (`eng_ltx25.py:806-930`) to the two engines, text encoder only, and
measure seconds-per-beat before and after.

## 6. 2.4 voice/credits -- cut as an accepted, documented residual

**Facts, grounded** (Sonnet reader):

* `high_band_edge_ratio` (`_otr_bark_lib.py:684-727`) measures the fraction
  of energy above 4 kHz at the clip EDGES -- the B1 squeal. PBUG-20260902-03
  (`PROD_BUG_LOG.md:10233-10308`) documents a steady ~1.4 kHz tone over
  seconds 10-21 of a line, flatness 0.04, *"the server log shows nothing
  abnormal"*. No whole-clip speech-shape scorer exists in the tree, and the
  edge metric is not wired into the live generation path at all.
* The primary exposure is already closed by an operator-shipped default:
  kokoro is the canonical voice for both slots since `00d4b72b`. Bark
  remains selectable and is forced in `otr_4060_floor`, `otr_rot_tts_bark`
  and `otr_bark_announcer_acceptance`, so the residual is real and confined.
* No recurrence is recorded since the PBUG's own follow-up.

**Decision: cut as accepted.** Not crash-class; the default moved off the
engine that produced it; the PBUG entry already carries the full scorer spec
(`:10298-10306`) for whoever reopens it. The bark-forced variants carry the
documented risk by name, which is the honest state.

## 7. OpenRouter catalog copy-forward -- closed as accepted

**Facts, grounded** (Sonnet reader):

* Old path `<repo>/models/openrouter_models.json`
  (`_otr_openrouter_backend.py:830`, a real leftover file on this box, 345
  models, gitignored); new path
  `<output>/otr/episodes/_shared/cache/openrouter/openrouter_models.json`
  (`:790-791`, `:825`). The move is `41fabf8b`, 2026-09-11, whose message
  says *"NO COPY-FORWARD, deliberately and not as a corner cut."*
* `_catalog_cache_path` (`:787-830`) and `load_catalog_cache` (`:845-869`)
  are fully guarded and return the empty catalog (`:833-841`) on a missing
  or corrupt file; `resolve_context_window` (`:892-935`) logs and falls
  back; `refresh_catalog_cache` (`:1073-1100`) rebuilds it and
  `scripts/otr_openrouter_refresh.py` ships (`.comfyignore:154`).

**Decision: close as accepted.** Only a git-checkout dev box ever had a warm
in-pack cache to lose; a cold cache never raises and one refresh restores
it. A copy-forward would add a write to a read-only resolver for a
one-time convenience on two machines.

## 4. 2.4 model-root audit tail -- "nothing" is the answer, and here is why

**Facts, grounded** (Sonnet reader; the folder alias verified by the driver):

* `comfy_models_dir()` (`nodes/_otr_paths.py:157`) has exactly one caller,
  `resolve_hf_model_path()` (`:775`, calling it at `:819`), which has none.
  A dead chain, as the row says; the archive parked it deliberately
  (`GO_FORWARD_ARCHIVE.md:4292-4297`, "do not open a third env in this
  diff") and commit `afe3bfed` refused a zero-caller rip.
* The fourth spelling is live: `flux2_klein.py:214-220` reads
  `OTR_COMFYUI_MODELS_ROOT` then `COMFYUI_MODELS_ROOT`, else resolves
  through `folder_paths.get_full_path("unet", ...)`; the engine is
  registered (`:233-234`), used from `assert_usable()` (`:377`) and the
  widget (`:278`), with a 35/0 live record (`PROD_BUG_LOG.md:7144`).
* **On this box the two reachable spellings agree.** Neither env var is set
  anywhere (launchers, shells); `_models_root()` (`_otr_gguf_backend.py:822`)
  therefore lands on `C:\ComfyUI-Models`, and ComfyUI's `folder_paths.py:111-112`
  maps `unet` to `diffusion_models`, whose list (`:31` plus the headless yaml)
  is `C:/ComfyUI-Models/diffusion_models` and `.../unet` -- the same tree.
* **Where they could disagree, the failure is loud.** `assert_usable()`
  (`flux2_klein.py:371-389`) checks `os.path.isfile` and raises a named
  `EngineUnusable` / `MISSING_MODEL` before any graph runs -- "install ... or
  set the variable to its full path". Not a silent wrong render.

**Decision: nothing, recorded.** No two spellings can silently resolve to
different directories on the machine that renders; the dead pair changes no
observable behaviour whether merged or left; and a merge risks the one
regression class `_models_root()`'s own docstring warns about (changing which
existing directory wins on the rendering box). Reopen only if a 4060, pod or
clean-install session actually observes a divergence -- and then it is the
four-round arc the earlier anchor scoped, not a rip.

## 8. Why house and techno do not groove -- settled by the bench; the rest is a listen

The row asked to "take it to the bench before touching the palette". The
music-model bench (`docs/2026-09-12-music-model-bench/`) and the recipe
bench above rendered the composed house and techno opening cues nine times
each on the base checkpoint at cfg 4.0 (three seeds, three recipes /
checkpoints): every one committed to the asked tempo (house 120-123, techno
129) with pulse 0.6-0.85. The n=1 canonical pads did not reproduce on a
stock-node graph. The palette is not touched. What remains is whether the
canonical path differs from the bench on those two banks, and that is the
operator's listen already in section 3 ("Listen to the four bank genres" --
and he has since said the techno sounds good). The row leaves the plan.

## 9. Reviewer, verdicts, and three decisions revised

One codex contrarian over this document (sections 1-8 as written above),
briefed to refute all seven rows. Verdicts: recipe STANDS, cast-count
STANDS, LTX allowlist STANDS, per-beat reload STANDS; **model root REFUTED,
voice/credits REFUTED, OpenRouter REFUTED.** The driver grounded each
refutation against the file it cited; all three hold, and the decisions in
sections 4, 6 and 7 are revised here rather than rewritten in place, so the
record shows what the readers missed.

| row | codex said | grounded? | revised decision |
|---|---|---|---|
| 2.4 voice/credits | `PROD_BUG_LOG.md:10294` marks the bark output guard FIX-OPEN and calls its absence "a silent wrong render"; the fix is specified at `:10298` | yes -- the STATUS line reads *"ROOT-PINNED (engine roll), FIX-OPEN"* and item 2 says *"a legitimate guard by the standing rule -- the alternative is a silent wrong render"* | **Not cut. Becomes a CODE row:** the bark output guard exactly as the PBUG specifies (fraction of one-second windows with dominant frequency in 70-400 Hz and flatness under 0.2; re-roll `seed + 1` up to twice on a fail; keep the best take; WARNING per re-roll with the score; the ledger field is always filled). Bounded to `eng_bark`'s generate path; kokoro remains the default. The section-6 reader was right about the metric mismatch and wrong to weigh a documented silent-wrong-render as residual. |
| OpenRouter copy-forward | `_otr_openrouter_backend.py:892-930`: a cold cache falls back to 8,192 and the docstring records that clamp cutting a script off mid-JSON | yes -- the fallback is loud in the log and silent in the render; a FRESH install of the cloud lane is cold until `scripts/otr_openrouter_refresh.py` is run by hand | **Not closed. Becomes a CODE row, and not the copy-forward:** a copy-forward helps two dev boxes and no one else. On a slug miss, `resolve_context_window` refreshes the catalog inline once (the lane is about to call OpenRouter anyway), re-reads, and only then falls back to 8,192 -- guarded, never raises, one refresh per process. The old in-pack cache becomes irrelevant. |
| 2.4 model-root | `flux2_klein.py:199-220`: the order is explicit `OTR_FLUX2_KLEIN_CKPT`, then `folder_paths`, THEN the two env roots -- the reverse of `_models_root()` (env first) | yes -- section 4 stated the order backwards. With an env root set AND a second copy visible to `folder_paths`, the image engine loads one file and the GGUF writers another, silently | **Not "nothing". Becomes a small CODE row:** in `_resolve_unet_path`, consult the env roots before `folder_paths` but only when the file EXISTS there (so a box that pins an env root for other engines and holds this one under `folder_paths` keeps loading), matching the shared resolver's precedence. On this box neither env var is set, so the path is provably unchanged -- the measurement the 0B rule asks for is one print. One test pins the order. |

The four STANDS rows leave the plan as decided. Section 8 stands. Three
rows move from section 1 to section 2 as named CODE work with a written
design; section 1 is empty after this batch.
