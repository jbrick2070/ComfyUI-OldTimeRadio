# SPEC: vram-recipe-lab findings into OTR

Date: 2026-08-09
Status: **r4-COMPLETE, LANE-1 CLEARED (2026-08-10).** r4 convergence (Sonnet 5,
`kibitz-runs/2026-08-10-video-plan/r4/`) verdict: yes-with-fixes; its eight
UNLANDED RULINGS and the highest-severity residual defects are landed in this
revision. Remaining known-open items are listed at the end of this header and
are NOT lane-1 blockers. The independent final-QA pass
(`docs/2026-08-10-FINAL-QA-video-build-corpus.md`) re-ran against this
revision on 2026-08-10 and returned **VERDICT: start lane 1 = YES, zero
blockers**, verifying all seven landed fixes (a-g) by file:line against
source, confirming the stale 107/111 sweep, all four lane-1 preconditions,
the scaffolding schedule, the superseded build order, and a clean test-suite
run. Its 21-lane-packet plan is ADOPTED as the operative build order.

**STANDING DEFAULTS FOR UNATTENDED BUILDS (operator asleep, 2026-08-10).**
The three open questions have safe defaults so no window ever guesses or
stalls overnight. A window may adopt these WITHOUT asking; the operator may
overrule any of them later at zero cost:
- Q1 H3 commit granularity: SPLIT video and audio-in into separate lane
  commits ONLY IF each half ends with a green suite on its own; otherwise
  ship them as one commit. Green-at-every-chunk wins over granularity.
- Q2 H3 multi-clip mouth warning: LEAVE AS IS (warn + loud `long_takes` +
  visible jump cut). Turning a warning into a refusal is a behavior change
  and needs an explicit operator ruling.
- Q3 WAN TI2V envelope row: do NOT qualify it from lab numbers. Ship the row
  DISQUALIFIED (status quo, safe) so admission honestly reports "not
  enforced" for that lane; re-derivation from OTR-wrapper measurements is its
  own later task.

Known-open, scheduled to their own lanes: the WAN TI2V envelope row is
unsupported as written and must be re-derived from OTR-wrapper numbers before
the S2 qualification commit (lanes 4-5); the 9 cheap lanes' defects ride the
QA plan's lanes 10-18; the H3 split-vs-single commit granularity needs one
confirmation; whether H3's multi-clip mouth WARNING should be promoted to a
refusal is an operator decision.

Prior status: **r3-APPLIED, r4 PENDING - NOT IMPLEMENTATION-READY.**
Kibitz `kibitz-runs/2026-08-09-video-plan/` (r1 Codex + driver anchor; r2
Codex + two Fable subagent lanes incl. a blast-radius/V-1 audit; plus a
dedicated fps-fix investigation) and `kibitz-runs/2026-08-10-video-plan/`
(r3 Codex + a Fable wiring/sequencing lane; judged in r3/judgment.md and its
rulings applied here - hence the "r3 proved" references below).
**r4 convergence has NOT run**, so this document authorizes no implementation
prompt; the 2026-08-10 final-QA pass correctly stopped at that gate
(`docs/2026-08-10-FINAL-QA-video-build-corpus.md`). Every file:line below was
verified against source by at least one lane. Human verdicts, the diet parity
ruling, and operator doctrine are final and recorded inline.

Build cycle (operator, 2026-08-09): spec -> code -> wire -> regress -> commit ->
ship, chunk by chunk, until EVERY chunk is in - both H3 paths and all video
lanes - and only then the end-to-end episode test. Per this repo's CLAUDE.md
section 7 Codex commits AND pushes each green chunk to `v2.0-alpha` and verifies
HEAD == origin, no 0-byte files, no BOM, AST parse on touched .py.

Lab baseline: vram-recipe-lab `4d87cfa` (PROMOTION_BRIEF, HUMO_BAKEOFF,
HUMO_DIET, WAN_RETENTION_FINDINGS + receipts). Nothing is a lab-gate pass
unless its receipt says so.

Standing scope: 14.5 GiB ceiling on 16 GiB hardware; 100% local/offline;
sequential execution; V-1 audio spine UNIVERSAL (no exceptions ship).

Operator doctrine: equal-partner dropdown, ONE master canvas JSON, per-lane
profile+launcher pairs, lanes are INDEPENDENT (an H3 episode runs sage-free;
other lanes keep Sage - do not design mixed-boot episodes); casting is
per-profile, never a new selection subsystem; lab-first for any new model load;
24 fps engines are fine but MUST deliver on the 25 fps timeline.

---

## S1. wan_i2v canvas declaration; KEEP wan_i2v

Commit the working-tree `render_canvas = (832, 480)` on `WanI2VEngine`
(`eng_wan_i2v.py:242`) together with the comment block at `:235-241` (both are
part of the same uncommitted 8-line hunk; the comment's "re-measure before a
keep/retire ruling" is now satisfied and must be rewritten, not preserved).
Ruling: KEEP wan_i2v (exonerated: warm 13.93 / cold 14.05 GiB at the declared
canvas); wan_ti2v remains the default WAN lane (0.10 GiB cold margin is too
thin to displace it).

`git add` BY NAME - the tree already carries unrelated modified/untracked
profile files. `declared_render_canvas` is applied last and overrules ledger
and env (`render_driver.py:227-267`), and wan_i2v has no drift guard while
ltx_video has two: add the wan_i2v equivalents of
`tests/test_engine_contract_roster.py:500-534` and `:568-595` in this commit.
All three shipped wan_i2v profiles are already 832x480, so profile lanes are
unaffected; the exposure is the coverage-matrix path that routes an engine via
`role_overrides` without copying a canvas (`render_driver.py:244-252`).

**BLOCKER found by the lane audit - wan_i2v cannot start today.** `_ckpt_path()`
defaults to `<comfy_root>/models/checkpoints/wan2.2-i2v.safetensors` and
`_installed()` is a bare `os.path.exists` that never consults `folder_paths`
(`eng_wan_i2v.py:245-250`) - unlike the sibling (`eng_wan_ti2v.py:331-339`).
That file does not exist on this box; the installed weight is
`C:\ComfyUI-Models\diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- different name, different category. **CORRECTED (r3 ruling #11; landed r4)** - an earlier draft claimed "no boot
script sets OTR_WAN_I2V_CKPT", which is FALSE: `_otr_soak_server_launch.cmd:93`
already pins it to the correct installed weight and sets
`OTR_ENABLE_WAN_I2V=1`. The REAL gap is that the canonical path never takes
that branch (`_otr_w45_boot.ps1:42-48` enables TI2V but not I2V, and
`otr_headless_canonical.ps1:182` boots the cmd without a lane token). So on
the canonical lane `assert_usable` raises `EngineUnusable(MISSING_MODEL)`
before the first forward. THIS COMMIT must also: add the `folder_paths`
fallback mirroring `eng_wan_ti2v.py:331-339`, and pin the env/profile so the
lane resolves. Without it the S1 smoke cannot run and the KEEP ruling is
untestable.

Acceptance: AST parse; no INPUT_TYPES change; the two new drift tests; a
`render_canvas` pin for wan_i2v (every other declaring lane has one -
`tests/test_fastwan_8gb.py:78-83`, `tests/test_ltx_8gb_canonical_canvas.py:77-82`
- and wan_i2v has none, nor any `tests/test_wan_i2v.py` at all); one wan_i2v
smoke selected via a profile `role_overrides` (the engine has
`default_roles = ()` `:214` and ships dark), NOT a node-87 edit.

## S2. VRAM admission: qualify + recalibrate (the guard is already wired)

**Grounded current state.** Two check points exist:
1. STATIC path - `compute_real_frame_budget` raises `MotionBudgetError`
   (definition `motion_common.py:332-336`; raises at `:456/:463/:475`), called
   from `eng_wan_ti2v.py:801-802`. ENFORCING today, using a row its own
   comment disqualifies (`motion_common.py:339-366`).
2. COVERAGE-PLAN path - `_assert_beat_affordable`
   (`render_driver.py:3191-3268`, calling `assert_frame_affordable` at `:3263`)
   - WIRED but inert because `QUALIFIED_COST_ROWS = frozenset()`
   (`motion_common.py:367`).
**Honest coverage note:** only wan_ti2v calls the static path. Single-clip
renders on ltx_av, humo, wan_i2v, ltx_video, ltx_8gb and fastwan_8gb get NO
admission check before or after this work. Do not claim otherwise.

**Work items.**
a. **Recalibrate wan_ti2v and qualify it IN ONE COMMIT.** Qualification without
   recalibration grounds the WAN roster: `tests/test_vram_admission_boundary.py:69-71`
   proves the shipped row refuses BOTH 93 and 177 frames at 14500 MB free - the
   planner's actual output. `fastwan_8gb` mirrors the same row from its own
   module (`eng_fastwan_8gb.py:295-296`) and must be recalibrated in the same
   change or it goes silently stale.
b. **Keep the affine signature; add absolute envelopes as a SEPARATE keyed
   table.** `FRAME_COST_MODEL` is consumed as a 2-tuple (`motion_common.py:319,
   431-433`), exposed as a staticmethod (`:509`), mutated at import by another
   engine, and six assertions in `tests/test_clip_fill.py` are computed from it.
   An absolute envelope has per_frame == 0, which lands exactly on the
   documented ZERO-SLOPE HOLE (`motion_common.py:440-482`) where only the fixed
   overhead guards. So: affine stays for the engines that fit it; absolute
   config-keyed envelopes are a separate mechanism with their own exact-length
   check (`compute_real_frame_budget` hard-applies 4n+1 quantization at `:426`
   and cannot serve H3's 17k+5 grid).
c. **The LTX affine FIT is CUT.** The HQ ladder is non-monotonic against the
   model's pixel scaling (`:433`): H2 832x480x193 = 7.93 GiB EXCEEDS H3
   1024x576x193 = 7.36 GiB at 1.48x the pixels. Use conservative absolute
   envelopes for LTX. Do not smooth data to make a line fit.
d. **NO retention surcharge.** `_assert_beat_affordable` reads live free VRAM
   at admission (`:3249`), so retained VRAM is already absent from `free_mb`;
   charging it again double-counts and produces false refusals. One accounting
   equation - live-free comparison - with operands and units in the receipt.
e. **Make the admission record observable.** `vram_admission` is written at
   `render_driver.py:3697` and read by NOTHING. TWO plumbing sites, not one
   (r3 ruling #6; landed r4): the manifest row builder at
   `render_driver.py:4861-4912` (which copies recipe/quant/render_canvas/
   vram_peak_mb/native_frame_count/extension_mode/segments but NOT
   vram_admission) AND `_build_render_engines_payload`
   (`nodes/otr_video_render_batch.py:128-175`). Adding a read only at the
   second site persists nulls. Carry it through both, and note only
   coverage-planned beats produce the key.
f. **Store MiB, display GiB** (`free_vram_mb()` divides by 1024^2, `:300-314`).

Envelope data (candidates; qualification happens through OTR's real
`prepare()` + `render_clip()` lifecycle, never from lab numbers alone):

| Engine / config (boot lane) | Peak | Provenance |
|---|---:|---|
| LTX AV GGUF Q3, 832x480x97 | 7.2-7.5 GiB | lab warm |
| LTX AV HQ, 1024x576x193 | 7.36 GiB | lab warm |
| WAN TI2V 5B Q5, 832x480 | 12.5-13.2 GiB | lab warm |
| WAN I2V 14B, 832x480 declared | 13.93 warm / 14.05 cold | exoneration receipt |
| HuMo 1.7B, default boot | 15.12-15.23 GiB | OTR-side, not lab-gated |
| HuMo 1.7B, diet boot | 12.84 GiB warm | lab warm + human parity |
| HuMo 14B FP8, default boot | 14.98 GiB | OTR-side; OVER ceiling |
| HuMo 14B FP8, diet boot, 832x480x97 (LANDSCAPE) | **13.06 GiB warm** / 13.17 cold | lab warm PASS, 1.44 GiB headroom (ENVELOPE_LADDERS.md job A) |
| HuMo 14B FP8, diet boot, 480x832x97 (portrait) | 13.22 GiB warm / 13.14 cold | lab warm PASS |
| H3 Ref2VA, sage-free+no-pinned | 6.51-6.71 GiB | lab cold gate |
| wan_ti2v chained 177+25 | 12.43 peak / +5.11 retained | retention receipts |
| fastwan_8gb chained 177+25 | 12.57 peak / +5.33 retained | retention receipts |
| ltx_video chained 169+169 | 14.59 peak / +3.06 retained | diagnostic, not a pass |

Speed context (profile authors): LTX distilled 2B is 29.5x faster than WAN
TI2V at identical 832x480x193 (13.8 s vs 407.5 s warm).

Acceptance: the recalibrated row ADMITS `_prebuilt([177, 177, 93])` at the
measured free-VRAM level (not only that it can refuse); an over-budget plan is
refused before any model load; an unqualified engine's receipt says so in
words, on disk; `tests/test_vram_admission_boundary.py:42-82` updated
deliberately (those tests are the tripwire that says no row is qualified -
changing them IS the change).

## S3. LTX AV HQ: canvas declaration, not a recipe token

`OTR_LTX_AV_RECIPE` is a GRAPH-TOPOLOGY namespace (`_VALID_RECIPES`
`eng_ltx_av.py:239-240`; `_recipe_config` returns only sampler/LoRA/scheduler
fields `:294-330` and raises on unknown `:331`), and neither canvas nor frame
count is reachable from it. Ship instead: `render_canvas = (1024, 576)` on the
HQ lane (both axes /32-legal) plus a named profile supplying 1024x576 and 193
frames, with the measured envelope (warm 7.36 GiB / 585.3 s) documented at the
declaration. H1/H2 alternates CUT. No graph topology changes - OTR's LTX graph
is ahead of the lab's (`:212-221` LoRA/scheduler exclusivity, `:109` tiled
decode seam fix, deliberate base-pass audio masking with refine re-concat).

## S4. MiniMax H3: two internal adapters, counted at 25 fps

**Two internal engines** (one implementation module, thin adapters):
`minimax_h3_video` and `minimax_h3_audio_in`. One internal engine cannot carry
both modes - capability is static per adapter (`registry.py:138-214`) - and two
public ids must not map to one internal id (`public_engines.py:70-72` assert;
precedent: `ltx_video` / `ltx_audio_in`). These are ENGINE ADAPTERS, not new
ComfyUI node classes; but registering them DOES change
`OTR_VideoDirector.INPUT_TYPES` via `_video_model_combo()`
(`otr_video_director.py:158-179` -> `:238-250`), so full widget discipline and
S10 apply.

### S4.1 The 24 -> 25 fps contract (THE critical item)

H3 generates at 24 fps; the canonical timeline is 25. Delivering 192 model
frames as 192 canvas frames plays them as 7.68 s - a **320 ms accumulating**
mouth drift over an 8 s beat, silent, no assert. The in-repo precedent is Veo:
generates at 24, declares its menu in CANVAS frames at 25
(`eng_google_veo_video.py:60-61, :516-521`), pinned by
`tests/test_engine_contract_roster.py:204-234`. Veo's cloud path gets the
conversion free from a duration-preserving ffmpeg resample in
`canonicalize_video` (`_otr_shared/cloud_media_canonical.py:387-388`). H3 is
LOCAL, and the local encoder only LABELS the rate (`-r` before `-i pipe:0`,
`wrapper_bridge.py:620`) - it cannot resample. So the adapter converts in
numpy immediately before the encoder.

Declared constants (file-local, fully declared):

```
MODEL_FPS = 24 ; CANVAS_FPS = 25 ; target_fps = 25
H3_GRID          = 124..362 step 17    (17k+5, k in [7,21])  # MODEL frames
SUPPORTED_FRAMES = (129,146,164,182,200,217,235,253,270,288,306,323,341,359,377)
canvas_frames(L) = (L * 25) // 24      # floor - never claim more content than rendered
MODEL_FRAMES_FOR = {canvas: model}     # render-time inverse
frame_contract   = FrameContract(discrete_frames=SUPPORTED_FRAMES,
                                 native_fps=25, allow_tail_trim=True, ...)
```

**Grid bounds CORRECTED 2026-08-09**, provenance restated precisely after the
2026-08-10 QA: the INSTALLED node
(`ComfyUI/comfy_extras/nodes_minimax_h3.py:90,116`) declares `length` as
`min=5, max=3600, step=17` with the tooltip "trained range is ~124-362,
longer is untested". So 124-362 is the TRAINED range we deliberately adopt as
our contract, NOT a node-enforced limit - the node would accept more. We
choose the conservative bound; any future widening is an experiment with its
own receipts, not a config tweak. (The old 107-345 draft came from the
problem statement's rounded "4-15 s" and is void. The f107 lab receipt is a
measured BELOW-RANGE failure, not evidence for the minimum.) Note the node's
canvas defaults are 1344x768 - the music study's canvas - while OTR renders
832x480; envelopes must be keyed by canvas. Both endpoints sit on the lattice (124 = 17*7+5,
362 = 17*21+5). Consequence: the smallest legal beat is **129 canvas frames
(5.16 s)**, not 111 - beats shorter than that render 129 and trim, so the
short-beat waste is larger than first estimated and the trim ratio MUST be
logged loud. Pin the ladder literal against the node's real min/max at build,
never against a doc's rounded seconds.

Conversion helper (pure, integer arithmetic, nearest source frame):
`idx[j] = min(L-1, (j*2*MODEL_FPS + CANVAS_FPS) // (2*CANVAS_FPS))` applied to
the uint8 batch between `images_to_uint8` and
`encode_frames_to_silent_mp4(frames, out, self.target_fps)` - the exact seam at
`eng_ltx_av.py:1088-1094`. Verified for every rung: `max(idx) == L-1`, indices
monotonic, first and last model frames preserved, no frame out of range.
Residual error is bounded at +/- 1/48 s (20.8 ms) and **does not accumulate**.

Worked cases (RECOMPUTED on the corrected 124-362 grid; the earlier 107/111
examples are void): an 8 s beat targets 200 canvas frames ->
`MODEL_FRAMES_FOR[200] = 192` -> 192/24 = exactly 8.000 s, WAV slice exactly
8.000 s, zero pad, 8 duplicated frames on delivery. A 94-frame beat: the
smallest menu value >= 94 is 129 -> `render_frames=129, trim_tail=35` ->
model 124; the WAV render-window is 129/25 = 5.160 s (3.760 s speech +
1.400 s tail pad) and the model window is 124/24 = 5.1667 s, so the adapter
pads 6.7 ms - the general rule is `0 <= L/24 - C/25 < 1/25` at every rung.
The WAV pad is OWNED by the adapter's audio-staging step (pure ffmpeg apad on
its own staged copy, receipted); silent truncation is forbidden. Multi-clip
partition literals (e.g. a 442-frame beat) MUST be re-derived at build by
running the real `partition_beat` against the corrected menu and pinned as
test literals - do not carry the old-grid partitions forward.
NOTE: the corrected floor also voids
`docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md:42-43` ("could not render
the default episode") - with `allow_tail_trim=True` a 94-frame beat renders
129 and trims; the waste ratio (35/129 = 27% here) is logged loud.

**Continuity, DECIDED per adapter (was "decide at build"):**
- `minimax_h3_video`: `CONTINUITY_STRICT_FIRST_FRAME` - the lab proved
  exact-last-frame continuation (3 clips / 15.4 s, seam SSIM 0.816/0.900);
  chained beats route JOIN_CHAIN and inherit the terminal frame.
- `minimax_h3_audio_in`: `CONTINUITY_SOFT_REFERENCE` - the HuMo shape:
  identity carried by the reference portrait, multi-segment beats route
  JOIN_JUMP with a fresh still per segment. Rationale: lip beats restart at
  cuts anyway, and the portrait (not the previous frame) is the identity
  authority.
- `minimax_h3_music`: `CONTINUITY_NONE` - standalone clips, never chained.

Also fix in this commit: `frame_contract.py:108` documents Veo's menu as
`(96, 144, 192)`, the exact value `tests/test_engine_contract_roster.py:228`
asserts is WRONG. It is the first thing an adapter author reads. Correct it to
the 25 fps values.

### S4.2 Mouth policy - engine-id carve-out, landed WITH the adapter

`_is_character_face_beat` hardcodes `engine_id == "ltx_audio_in"`
(`render_driver.py:1546-1550`); a new audio-in engine answers False while
still owing a mouth, so `mouth_owner_for_beat` RAISES from the plan-time call
site (`otr_shot_lock.py:1025`, `mouth_policy.py:104-117`). Extend `:1548` to a
membership test including `minimax_h3_audio_in`, with a shot_lock test
asserting the chosen answer for `role=character_video`.
**Do NOT mint a new family** to dodge `AUDIO_IN_FAMILIES`: a family outside
`content_oracle.MOTION_FAMILIES` (`:33-38`) makes frozen H3 clips
motion-EXEMPT (loud -> silent), and `schemas.py:177-181` rejects any
family_hint not in `FAMILIES`, failing every H3 request. H3 declares
`audio_conditioned_video`.
Second-order effects to verify in the same commit: that predicate also drives
`_uses_ambient_master_audio` (`:1575-1583`), prompt/still selection
(`:2600-2670`), and the one-face-per-episode cap (`mouth_policy.py:226-232`).
`tests/test_wire_w7_mouth_ownership.py` parameterizes over families and will
NOT exercise an engine-id carve-out - add a case.

### S4.2b Prompt templates - a named deliverable, not an assumption

The MiniMax H3 nodes condition through TAG GRAMMAR in the prompt text itself
(`<Picture 1>`, `<Audio 1>` - the node description says "use the same tags
when prompting", and the lab's lip-sync retraction proved the tags alone are
not enough: the prompt must also INSTRUCT the performance). OTR's prompt
router knows nothing about this grammar today. Each adapter therefore ships
with a prompt-template wrapper as an explicit deliverable:

- `minimax_h3_audio_in` (talking beats): wrap the router's scene/dialogue
  prompt with (a) the `<Picture 1>` identity clause, (b) the `<Audio 1>` lip
  ACTION clause ("speaking directly to the camera with lip movements matching
  <Audio 1> precisely" - the phrasing that made lip-sync work at all), and
  (c) medium/close shot vocabulary (the lab's retraction note: close-ups sync
  better than wide shots). Expectation stays honest: dub-grade, ruled.
- `minimax_h3_video` (scene/continuation beats): `<Picture 1>` clause when a
  reference is supplied; plain scene description otherwise.
- `minimax_h3_music` (runner script): scene description + a score request
  derived from the beat's dramatic intent, drawn from a small phrase bank in
  the operator's register family ("tense orchestral drama score", "warm
  lighthearted orchestral score", ...) - never boilerplate, never audio
  constraints. Budget variant omits the score clause entirely (prompt-only).
- ALL H3 templates avoid the restraint vocabulary ("restrained", "slow",
  "stable", "subtle") - the measured cause of near-still output across the
  motion ladder and the mime prompts. This is a rule about OUR prompt text,
  not a generation filter.

Acceptance: template unit tests assert tag presence, action clause presence
for audio-in character beats, and restraint-vocabulary absence.

### S4.3 Seed - profile, not adapter

The sampler seed is derived per segment from the request hash
(`render_driver.py:2998-3024`) specifically to kill the snap-back-to-same-pose
defect, and is stamped into `seed_bundle` + `observability.video_seed`. An
adapter that fixes 43 internally re-arms that defect AND makes both stamps
false. Adapters honour `seed_bundle.request_seed`; seed 43 is expressed as the
workhorse PROFILE's existing `seed_policy` (`seed_mode: "fixed"`,
`request_seed: 43`; schema `capability_profiles.py:118-123`), which applies
episode-wide.

**BUT THE VALUE CANNOT REACH THE ADAPTER TODAY (r3 ruling #5; landed r4).**
`otr_shot_lock.py:1494` hardcodes `"request_seed": 0` and drops the mode, and
`render_driver.py:2996-3024` then derives the sampler seed from the request
hash regardless. So the profile ruling is inert without a wiring fix. REQUIRED
WORK, in the H3 commit: thread the seed contract (mode + request_seed) from
the frozen director policy through ShotLock into each shot, and define fixed
mode's per-segment semantics as segment 0 = the fixed seed, successors =
deterministic derivatives (the existing `%s#seg%d` shape) so fixed mode does
NOT re-arm the snap-back-to-same-pose defect the per-segment derivation exists
to prevent. Test the request contents AND the observability stamps.

### S4.4 Remaining adapter contract

- `still_plan` declaration is MANDATORY and audited
  (`_otr_shared/still_plan_helpers.py:8-24`; `tests/test_still_plan_audit.py`).
- Declare continuity EXPLICITLY on BOTH adapters. Default `CONTINUITY_NONE`
  (`frame_contract.py:136`) routes to a jump cut and refuses chaining
  (`coverage_plan.py:195-196, :500-506`) - which would waste the lab's proven
  exact-last-frame continuation (seam SSIM 0.816/0.900).
- Sage: `assert_sage_not_patched` (`motion_common.py:82-98`, precedent
  `eng_ltx_video.py:611`) - NOT the wan_i2v pattern, which ESCALATES to a
  sidecar rather than refusing. Sage silently turns H3 output to noise
  (Comfy-Org/ComfyUI#15263) and the per-model KJ probe FAILED on sm_120.
- Boot: sage-free + `--disable-pinned-memory` (host RAM 61.3 -> ~26 GiB).
- V-1: call `wan_shared.validate_silent_clip_contract` on H3's OWN emitted
  clip inside `canonicalize` (`wan_shared.py:251, :273-277`). `has_audio: False`
  is a hand-written literal in every adapter and nothing probes the per-beat
  FILE - for a joint-AV model that gap ships a receipt that lies. Reproduce
  BOTH LTX mechanisms: drop the audio latent in-graph AND `-an` at encode.
  Do NOT cite `test_audio_byte_identical` as V-1 coverage - it only re-hashes a
  stored fixture.
- Import law: no top-level torch/model imports, no weight loading at import,
  UTF-8/no-BOM/ASCII source. Registry imports are silently swallowed
  (`_otr_video_engines/__init__.py:64-66`), so a NAMED unavailability reason
  must live in `assert_usable`, never at import.
- NOT in `PLANNING_CAP_ENGINES` (`frame_contract.py:328`): decide explicitly
  whether H3 reads `profile_max_render_frames` at all - a self-cap without
  allowlist membership reproduces the fastwan live failure recorded at
  `frame_contract.py:311-321`. If H3 declares `safe_render_frames`, the
  tripwire at `tests/test_engine_contract_roster.py:294` updates in the same
  commit.
- License: MiniMax grant (local, offline, operator hardware, no hosted
  service, no redistribution), conditioned on the operator's request-email
  commitments - file a redacted authorization artifact with S9.

## S5. (folded into S4.4 - Sage guardrail ships with the adapters)

## S6. Dropdown naming

Convention `<model><version>_<vramtier>gb_<capability>` per
`_otr_shared/public_engines.py`; additive only (internal ids never rename);
labels are tooltip/doc only; the aspect suffix stays separate
(`"wan22_8gb_video (16:9)"`); the bijection assert must keep passing.

**Naming RULED by the operator 2026-08-09.** The `<vramtier>gb` token is
RETIRED: it encoded "target card this lane was built for", which has drifted
badly from measured usage (`wan_8gb` really consumes 12.5-13.2 GiB and cannot
run on an 8 GB card at production canvas, while the new H3 lanes are the
LIGHTEST local engines at 6.5 GiB and would have carried a `16gb` token). It is
replaced by a coarse **`low` / `high`** marker so a user self-selects by their
own hardware. Coarse is deliberate: the measured values cluster with an empty
gap between 7.5 and 12.5 GiB, so the split needs no judgment, and the marker
survives measurement drift (the HuMo diet moved 2.4 GiB in one afternoon and
never changed bucket).

Convention: `<model><version>_<low|high>_<capability>`.

| Internal id | Public id | Measured warm | Longest 1 render |
|---|---|---:|---:|
| `minimax_h3_video` | `h3_low_video` | 6.5-6.7 GiB | 15.1 s |
| `minimax_h3_audio_in` | `h3_low_audio_in` | 6.5-6.7 GiB | 15.1 s |
| `ltx_audio_in` | `ltx23_low_audio_in` | 7.2-7.5 GiB | 19.9 s |
| `ltx_8gb` | `ltx098_low_video` | **UNMEASURED** | 6.4 s |
| `ltx_video` | `ltx23_high_video` | **UNMEASURED** | 6.8 s |
| `wan_ti2v` | `wan22_high_video` | 12.5-13.2 GiB | 7.1 s |
| `fastwan_8gb` | `wan22_high_fast` | ~12.6 GiB | 7.1 s |
| `wan_i2v` | `wan21_high_i2v` | 13.9 warm / 14.1 cold | 7.1 s |
| `humo_1.7B` | `humo17_high_audio_in_portrait` | 12.84 GiB (diet boot) | 7.1 s |
| `humo` | `humo14_high_audio_in_portrait` | 13.22 GiB (diet boot) | 3.9 s |
| `humo_14B_169` | `humo14_high_audio_in_wide` | 13.06 GiB (diet boot) | 3.9 s |

Operator naming refinements (2026-08-10): audio-conditioned lanes STATE
`audio_in` in the public id (HuMo is audio-driven and now says so, matching
`h3_low_audio_in` / `ltx23_low_audio_in`), and portrait-aspect lanes STATE
`portrait` (the bare `humo14_high_face` hid that it renders 480x832). The
label suffix still carries the exact ratio; the id carries the words.

`ltx098_low_video` and `ltx23_high_video` markers are **PROVISIONAL** - neither
lane has ever been measured on this box. Add both to a lab ladder and label
them from receipts before the naming commit ships; do not guess a marker into a
user-facing name. (ltx098 renders at 512x288 on a 2B model so `low` is very
likely right, and ltx_video's only datapoint is a 14.59 GiB CHAINED diagnostic,
which is not a single-render number.)

**ALIAS MECHANISM - MOVE, NEVER ADD (r3 ruling #3; landed r4).** Old public
ids do NOT stay in `_PUBLIC_ENGINES`: two public ids mapping to one internal
id collapses `_INTERNAL_TO_PUBLIC` and trips the MODULE-SCOPE bijection
assert (`public_engines.py:68-72`) at IMPORT time. Because
`otr_video_director.py:52` and the shared profile/driver modules import
`public_engines` unguarded, and the pack wraps each node import in its own
try/except, the practical blast radius is most of OTR silently vanishing
from the node menu with scattered logged exceptions - not one clean lane
failure. So: RELOCATE each old public id into `_LEGACY_ENGINE_ALIASES`
(old-public -> internal: `wan_8gb`->`wan_ti2v`,
`ltx23_16gb_audio_in`->`ltx_audio_in`, `ltx23_16gb_video`->`ltx_video`;
`ltx_8gb` and `fastwan_8gb` need no row - they equal internal ids and pass
through `resolve_engine_id` step 3). `_PUBLIC_ENGINES` holds ONLY live menu
ids. Internal ids never rename. Ship a standing test asserting the final
bijection BEFORE the naming work so CI catches a mistake, not ComfyUI boot.
CUT: speculative extra tier rows and any HuMo-diet public row - the boot
contract, not the name, carries the diet.

Acceptance: `exact_menu_option_for(<id>)` returns exactly one option for each
new id (`otr_video_director.py:182-195`) - `resolve_engine_id` is
one-directional and cannot "round-trip"; node-87 strings are GENERATED by
`exact_menu_option_for`, never hand-typed; `tests/test_public_engines.py:42`
(exact dict equality) updates in the same commit.

## S7. WAN retention response - instrumentation FIRST

1. **Instrument the post-close boundary.** Two explicit snapshot functions
   with a JSON-safe schema, recording `mem_get_info`, torch
   allocated/reserved, ComfyUI loaded-model identities, and live segment
   references. `BeatSession.close()` has only engine/prepared state and
   swallows teardown failures (`beat_session.py:293-315`); the live objects
   belong to `render_beat_coverage` (`render_driver.py:3442-3638`). Telemetry
   failure records `{available: false, error: <named reason>}` and never masks
   a render/teardown error. Persist via `_build_render_engines_payload`
   (`otr_video_render_batch.py:128-175`) - the only path from a clip dict to
   disk. Attribution unavailable => step 2 may not change policy.
2. **Release, justified by 1, as a decision gate with three pre-enumerated
   branches:** allocator cache -> a NARROW sibling of the residue freer doing
   only gc + `soft_empty_cache(force=True)` + `empty_cache`
   (`_otr_vram_levers.py:219-242`); Python references -> narrow the proven
   owner; model residency -> no action. **Do NOT reuse
   `free_otr_pipeline_residue` for the same-engine case**: it detaches EVERY
   tracked patcher with no engine filter (`_otr_vram_levers.py:180-210`), and
   its call site's own contract says same-engine SKIPS it to preserve reuse
   (`render_driver.py:3797-3799`); reusing it evicts the engine the fast path
   exists to keep and contradicts
   `tests/test_cs3_inter_beat_reclaim.py:60-79`. Note there is also a SECOND,
   unconditional pre-render invocation at `:3777-3782`. Shape: failed
   qualified admission -> one surgical release -> re-read free VRAM -> retry
   admission once -> raise unchanged if still unaffordable.
   Struck from this spec (wrong code, both verified): `motion_common.py:612-635`
   is `teardown`, carrying the "never unload_all_models" contract and the
   lease-release `finally`; `render_driver.py:3466-3638` holds only
   strings/dicts - frames are already on disk.
3. **Planner warning on order sensitivity** (measured: identical WAN shots
   refuse long-first, complete small-first). Warn in the shot loop
   (`render_driver.py:3788-3810`); NEVER auto-reorder.

Acceptance: the historical topology (200f chained beat first, then 65f)
completes; warm same-engine throughput bounded NUMERICALLY (state workload,
warm-up count, repetitions, statistic) - "not materially regressed" is not a
criterion; the CPU suite must not silently skip the new branch
(`free_vram_mb()` returns None off-GPU, so a headroom-gated branch is
unreachable in CI while tests stay green - add an injectable free-VRAM value).

## S8. Boot contracts - use the profile `launch` block that already exists

`profile.launch` already carries `{sage_attention, extra_args, env}`
(`capability_profiles.py:125-132`), emitted by `scripts/build_variants.py:176-181`.
**CORRECTED (r3 ruling #2; landed r4) - `extra_args` IS A DEAD CHANNEL.**
`build_variants.py:180,211` writes `extra_args` only into a markdown
documentation string; no launcher ever turns it into argv, and
`--disable-pinned-memory` appears in ZERO non-doc files repo-wide. The live
channel is `launch.env`, consumed at `_otr_soak_server_launch.cmd:120`
(`OTR_HEADLESS_RESERVE_VRAM_GB` -> `--reserve-vram`). Therefore:
`humo_diet` = `env: {"OTR_HEADLESS_RESERVE_VRAM_GB": "2.921",
"OTR_HEADLESS_DISABLE_PINNED": "1"}`; `h3` = `sage_attention: false` +
`env: {"OTR_HEADLESS_DISABLE_PINNED": "1"}`. **New work item:** add the
matching cmd hook (`if defined OTR_HEADLESS_DISABLE_PINNED set
_OTR_PINNED=--disable-pinned-memory`, appended at the command-line assembly
around `:158-166`) - without it, a "configured" diet boot silently clamps
nothing. Enforcement must probe the RUNNING server
(`comfy.cli_args.args.reserve_vram` / `.disable_pinned_memory` plus
`sageattention_patched`), never the profile text, so a dead channel cannot
also produce a falsely-passing check. The key set is closed-validated (`:258-280`) - adding a
`boot_contract` key directly breaks all ~20 existing profiles; if a NAME is
wanted, introduce `_LAUNCH_OPTIONAL_KEYS` registered in
`_SECTION_OPTIONAL_KEYS` (`:166`). Boot contract is NEVER a director widget
(positional `widgets_values`, BUG-LOCAL-097,
`tests/test_canonical_widget_input_parity.py:92-99`).

Enforcement moves EARLY: `assert_usable` currently runs inside the render
phase (`render_driver.py:3164-3167`), after writer, TTS, master freeze and
every still. Boot + Sage checks belong in the ShotLock preflight beside
`mouth_owner_for_beat` (`otr_shot_lock.py:1000-1030`), with the render-time
check retained as defence in depth. Only adapters that never shipped under
`default` (the two H3 adapters) may REQUIRE a contract; `humo_1.7B` declares
`default` AND `humo_diet` compatible - requiring the diet boot would regress a
shipping lane. Launchers stamp resolved argv + contract into the receipt.
Offline determinism: the installed manager gates startup traffic on persistent
`network_mode == "offline"` (`ComfyUI-Manager/glob/manager_server.py:1920-1962`),
not on any launcher field OTR handles today - set and verify it fail-closed
before import; a self-stamped "no traffic ran" claim is not proof.

## S8b. Lane-readiness blockers (from the 2026-08-09 per-lane audit)

Found by auditing each local lane end-to-end. These are pre-existing defects,
not consequences of this plan, and each one would surface as a failed run.

1. **`wan_i2v` cannot start** - checkpoint path + no `folder_paths` fallback.
   Fixed in S1 (see there).
2. **`config/profiles/otr_8gb_wan.json:56` pins `max_render_frames: 17`** while
   `wan_ti2v` is NOW a `PLANNING_CAP_ENGINE` (`frame_contract.py:328`, added
   2026-08-02) and the adapter-side ping-pong that made that harmless was
   ripped the same day (`eng_wan_ti2v.py:1014-1024`). `effective_frame_contract`
   (`frame_contract.py:386-417`) therefore narrows the PLANNER, turning every
   beat on that profile into a chain of 0.68-second segments - the exact "pile
   of 17-frame renders" its own comment warns against (`:284-286`). The
   uncommitted `otr_g4_wan_ti2v.json` bump to 81 is the right shape;
   `otr_8gb_wan.json` (and `launch.env.OTR_WAN_TI2V_MAX_FRAMES = "17"` at `:82`)
   was not brought along. Fix in the same commit family as S2.
3. **`humo_1.7B` cannot refuse a short render.** `safe_render_frames = None`
   (`eng_humo.py:1028`) skips the exact-fit guard at `:840`, so an over-ladder
   beat emits 177 frames stamped `extension_mode: "none"` with
   `native_frame_count == frame_count` - indistinguishable from an honest clip
   on any path reaching `render_shot` without a stamped coverage plan. Give the
   tier a declared cap or make the guard unconditional.
4. **`humo_14B_169` carries a 3.07x request/render canvas disagreement** -
   request rewritten to 1472x832 (`render_driver.py:2501-2509`) while the graph
   renders `WIDE_DIMS = (832,480)`. PRECISION (2026-08-10 QA): `_native_dims`
   (`eng_humo.py:625-635`) resolves via `humo_dims_for_aspect` (the literal
   lives at `_otr_shared/aspect.py:31`) but ALSO honors
   `OTR_HUMO_WIDTH`/`OTR_HUMO_HEIGHT` overrides - so 832x480 is the default,
   NOT a fixed runtime guarantee. The declaration fix must therefore either
   agree with those env vars or the lane must state that the overrides are
   unsupported once a canvas is declared. Harmless ONLY
   because `_aspect_plan` output is never read by `_build_graph`; it becomes a
   real error the moment admission, still sizing, or composite scaling trusts
   `request.canvas`. Since P1 may cast this engine as hero, declare
   `render_canvas = (832, 480)` on it (no HuMo lane declares one today).
5. **The `sidecar_optional -> sidecar_required` Sage escalation is dead code
   with live tests.** `resolve_isolation()` has no production caller (only
   `tests/test_video_motion.py:161-170`, `test_video_motion_common_additive.py:39-48`),
   there is no video sidecar runner in `nodes/`, and every g4/w45 profile boots
   `sage_attention: true`. So the WAN lanes' documented Sage mitigation does
   not exist. Either delete the claim or build the escape; do not leave a
   docstring describing protection that is not there.
   OPEN (needs a live answer, not a code read): whether `launch.sage_attention`
   reaches an actual `--use-sage-attention` argument at all - the only consumers
   found are the schema validator and a docs generator, and
   `_otr_soak_server_launch.cmd:159-163` passes no attention flag.
6. **HuMo manifest rows ship nulls** - `eng_humo.py:900-902, :983-997` return no
   `vram_peak_mb`/`recipe`/`quant`/`render_canvas`, so `render_shot` falls back
   to an instantaneous VRAM read (`render_driver.py:3298-3302`). The WAN lanes
   already fixed this (`eng_wan_ti2v.py:1171-1194`); mirror it, because S2's
   envelope work depends on trustworthy per-beat peaks.
7. **Stale comments on the two lanes most damaged by stale numbers:**
   `eng_humo.py:1133-1147` still explains the 14B_169 cap as "= 49 HERE" (it
   became 97 on 2026-08-02), and `eng_wan_ti2v.py:825-833` still says WAN is
   "deliberately excluded from PLANNING_CAP_ENGINES" (false since 2026-08-02).

### LTX / procedural lanes (second audit)

8. **`launch.sage_attention` is a DEAD profile field - RESOLVED 2026-08-09.**
   Both 16 GB LTX lanes hard-refuse under Sage (`eng_ltx_av.py:536`,
   `eng_ltx_video.py:611` -> `assert_sage_not_patched`), and every shipped LTX
   profile declares `sage_attention: true` - which reads as those lanes being
   grounded. Verified: NO boot script passes `--use-sage-attention` (grep over
   `scripts/*.cmd|ps1|bat` returns nothing), and the field's only consumers are
   the schema validator (`capability_profiles.py:126`) and a docs generator
   (`build_variants.py:212`). So the lanes are SAFE under OTR's own headless
   launchers and would refuse only under an externally Sage-enabled server.
   ACTION: either wire the field to the launcher or delete it - a profile
   field that looks like a boot control and controls nothing is how the next
   reader gets this wrong. Same class of defect as item 5 (the dead sidecar
   escalation). Note `sageattention_patched` also trips on mere module
   residency outside ComfyUI (`motion_common.py:52-79`).
9. **`OTR_LTX_AV_RESERVE_VRAM_GB` can silently DELETE `ltx_audio_in` from the
   registry.** `eng_ltx_av.py:177` is a bare module-scope `float()` - the one
   env read `_env_num` (`:59-90`) was not applied to. A malformed value raises
   at import, is swallowed by the guarded import
   (`_otr_video_engines/__init__.py:91-93`), and the lane vanishes from the
   dropdown with nothing in the log (reproduced live: registry 27 -> 26).
   `tests/test_ltx_av_env_import_safety.py:33-42` claims to cover every
   module-scope env read and omits exactly this one.
10. **The ia2v stage-A base latent is not /32-legal.** `eng_ltx_av.py:819`
    halves the canvas to 416x240; `240 % 32 == 16`. `assert_ltx_dims` checks
    only the full canvas (`:1040`), `render_driver.py:2541` asserts "(all /32)"
    incorrectly, and `tests/test_ltx_av_ia2v_canonical.py:62-63` PINS the
    illegal value. Upstream silently rounds (`av_dims.py:11`), so the x2
    upsample may hand refine a latent whose height no longer matches 480.
11. **Nine of twelve lanes have a dead profile canvas channel.** mesh_stage,
    the four viz lanes and the four still lanes declare no `render_canvas`, so
    their `otr_w45_*.json` `render.canvas_w/h = 832x480` is never read and they
    all render at the 1472x832 landscape default. `otr_w45_ltx_8gb.json` /
    `otr_g4_ltx_8gb.json` additionally disagree with a LIVE declaration
    (832x480 vs the declared 512x288) and sit outside the one-profile drift
    guard.
12. **The four still lanes never check ffmpeg at preflight**
    (`cheap_families.py:123-126` returns unconditionally; the viz lanes gate it
    at both boundaries), and three of four (`still_motion`, `still_flat`,
    `still_pan`) emit a DARK LAVFI FLOOR on a missing scene still
    (`:213-215`) instead of refusing - the historical black-beat defect, still
    reachable. Only `still_word` sets `_require_still` (`:397`).
13. **`ltx_8gb` is an LTX-Video 0.9.8 engine with NO Sage gate** and no node-
    class gate in `assert_usable` (`:1061-1086`) - the family BUG-070 was
    written for. No test asserts the gate should exist, so its absence is
    structurally invisible.
14. **`ltx_video` renders 169 frames for every beat** (`min_frames ==
    max_frames == 169`, `:483-490`; `_ltx_frame_length` raises any ask to the
    decode floor `:225-249`). A 50-frame beat costs a 169-frame render - ~3.4x
    the GPU work, untracked. The floor was measured at 1472x832, not at the
    832x480 the lane now declares.
15. **`still_plan` is declared on every lane and read by NOTHING in
    production** (audit-only). The `routing_state.enable_ltx_i2v` token
    `ltx_video`'s rows key on (`still_plan_helpers.py:111`) exists in no code;
    the real gate is `OTR_ENABLE_LTX_I2V` (`eng_ltx_video.py:670`).
16. **`mesh_stage` gates the hy3d checkpoint but not the hy3d graph** - the ten
    node classes resolve only inside `load()`, so preflight passes and the
    render dies. **`viz_mxc_mandala` requires pycairo**, which appears in no
    requirements file.

## S8c. LANE PREFLIGHT MATRIX (operator directive 2026-08-10): the checklist
becomes a test suite

The transplant plan's five "working right" criteria and the lessons-ledger
checks are formalized as ONE parametrized suite -
`tests/test_lane_preflight_matrix.py` - that runs over
`all_engine_names()` and asserts, per lane, everything the matrix page
claims. A lane is IN the matrix when its row is green here; a future lane
(or a future card in the 5070-Ti-and-down ladder) inherits the whole
checklist for free. Per-lane assertions (CPU-safe, no renders):

1. WEIGHTS: every declared weight resolves via `folder_paths` (or the
   engine's documented env pin) - no bare `os.path.exists` on a hardcoded
   default (lesson L1, the wan_i2v killer).
2. CANVAS: `render_canvas` declared for every GPU lane that renders at a
   fixed size; /32-legal on both axes; equals the size the graph actually
   emits (lesson L2, the humo_14B_169 3.07x disagreement); any profile
   canvas either matches the declaration or the profile channel is
   documented dead.
3. CONTRACT: `native_fps == target_fps == 25`; discrete menus declared in
   frames not seconds; menu arithmetic legal (H3: every value = floor of a
   17k+5 model length x 25/24, boundaries pinned); continuity explicitly
   declared, never defaulted (lesson L3).
4. ADMISSION HONESTY: the lane either has a QUALIFIED cost row/envelope key
   or its receipts say "admission NOT enforced" in words - never a silently
   unguarded lane that looks guarded.
5. V-1 SELF-PROBE: the adapter's canonicalize path calls
   `validate_silent_clip_contract` (or, for the future music runner, its
   explicit keeps-audio exemption is named) - `has_audio` literals alone do
   not pass (lesson L4).
6. GUARDS: Sage-sensitive lanes call `assert_sage_not_patched` in
   assert_usable; required boot contract declared where applicable; missing
   nodes/weights produce a NAMED unavailability from assert_usable, never a
   swallowed import.
7. SURFACE: exactly one live public menu id resolves per internal id
   (exact_menu_option_for), legacy aliases resolve without appearing in
   menus, ENGINE_MATRIX.md row regenerated, still_plan declared and
   audit-clean.

Rows that legitimately cannot pass a criterion (cheap lanes with no canvas,
unbounded contracts, no VRAM) declare a named exemption in the suite, not a
skip - the matrix page's per-row claims and this suite must agree exactly.
Ships EARLY (with the evidence manifest) so lane 1 onward is graded by it;
each lane's build is done when its preflight row flips green AND its smoke
render passes.

## S9. Evidence manifest (first commit)

Immutable manifest in OTR embedding the measured VALUES - engine, recipe,
canvas, frames, boot lane, peak, units, conditions - alongside digests and lab
commit `4d87cfa`. A SHA-256 of a file that is not shipped proves nothing to a
reader without it. Include the human-verdict provenance and a redacted license
authorization artifact (never personal data).

## S10. Canonical-workflow ownership

Any commit changing engine selections, dropdown rows, or director widgets
enumerates its exact node-87 `widgets_values` changes IN that commit, then
runs `OTR_WorkflowValidator`, JSON round-trip, link audit, and the live
INPUT_TYPES/widget-order audit (CLAUDE.md section 0;
`docs/PRODUCTION_SPRINT_LESSONS.md:117-130`). Adding combo OPTIONS does not
change widget count - the S6 rows are safe on that axis - but the strings must
be generated, not typed. `docs/ENGINE_MATRIX.md` is a generated drift gate
(`tests/test_engine_matrix_doc.py:43-49`): regenerate it in the same commit as
any registration. `registry.CAPABILITIES` must equal `all_engine_names()`
(`tests/test_capability_profiles.py:384`).

---

## P1. Casting - per profile

Human verdict 2026-08-09: 14B FP8 WINS; H3 seed 43 OK second; 1.7B acceptable;
H3 seed 42 fails to track. Diet: PARITY - 1.7B gate-legal at 12.84 GiB.
Casting is per PROFILE (`role_overrides.character_visual` ->
`OTR_VideoDirector.character_video_model`, `config/profiles/widget_mapping.json:23-30`).
Name the exact profile files and spell every override in the owning commit.

- Hero profile: **CLOSED 2026-08-10 - human parity RULED on the 14B diet A/B
  ("look the same to me").** Cast is final: `humo_14B_169` under `humo_diet`,
  13.06 GiB warm, zero perceived quality cost. Original unblock trail: HuMo 14B under the diet
  boot fits: **13.06 GiB warm at the canonical 832x480 landscape canvas**
  (1.44 GiB headroom; cold 13.17; portrait warm 13.22). The diet takes 14B
  from 14.98 -> 13.06 with the graph unchanged, exactly as it did for 1.7B.
  Cast `humo_14B_169` (landscape, `eng_humo.py:1121`) under boot contract
  `humo_diet` - NOT `humo` (portrait, `:231`), because the canonical canvas is
  landscape and the landscape leg measured better. Remaining 14B constraint is
  CONTENT, not VRAM: the 97-frame cap = 3.88 s per render, so hero beats
  longer than that chain as JUMP segments (soft_reference continuity), one
  fresh still each. Human A/B parity for the 14B diet clip is still pending
  (`outputs/humo_14b_diet_ab_production_vs_reserve2p921_warm.mp4`).
  This also retires S8b item 4 for the hero lane specifically: job A measured
  at 832x480 explicitly, so declare `render_canvas = (832, 480)` on
  `humo_14B_169` to make the request match the render it was measured at.
- Workhorse profile: `minimax_h3_audio_in`, `seed_policy` fixed/43, boot
  contract `h3`.
- LONG-BEAT profile (NOT "fallback" - `fallback_engine = None  # NO FALLBACKS`
  is law): `humo_1.7B` under `humo_diet`.

## P2. Motion ladder - RULED, nothing ships
All four rungs read as stills; no lever promoted; stillness attributed to
over-constrained prompting; a looser-prompt experiment is logged as a future
lab question.

## P3. Mime - FAIL on the tested artifact; capability REOPENED by new evidence

The ruled FAIL stands for `h3_mime_i2v_ledger_music_closing_8s`. But on
2026-08-09 evening the operator recalled good music from a different clip and
identified it from a frame contact sheet: **`outputs/h3_r2v_best_out_00001_.mp4`
(2026-08-08) was rendered from a portrait + a scene-only prompt with NO audio
input, and carries a full generated soundtrack.** It was filed as an r2v
quality test, so its audio was never auditioned. The rejected clip is the only
mime-labelled render whose prompt DIRECTED the audio - the style retired the
same day. Hypothesis under test: directing the audio degrades it.

Status for THIS build: unchanged. V-1 stays universal, engine audio is
discarded, NO architecture work, the H3 adapters ship audio-suppressed. The
capability question is a LAB question (`H3_UNCONDITIONED_MUSIC` mission:
seed repeatability, prompt-style A/B/C, duration scaling), gated behind
Jeffrey's ears.

**PROVEN RELIABLE 2026-08-10 (human-ruled, follow-up study):** the
score-request path passed end to end - effectively 5-of-5 usable scores
across seeds, five distinct usable moods ("music is good in all - if that's
what the script beat calls for"), duration survival to 11.5 s with MORE
movement at length. Machine bounds: ~8.2 GiB absolute at f124, 11.06 at
f192; f277 hit 14.72 (over gate) - lengths above f192 need headroom work.

Consequence for the build (**RESOLVED 2026-08-10, operator delegated the
wiring call**): the music lane ships THIS build as a STANDALONE RUNNER - a
dedicated script (the `_otr_single_engine_smoke.py` shape) taking
portrait/prompt/target-seconds, snapping to the H3 grid, rendering ONE
self-scored mp4 into a durable deliverables directory with its own receipt.
It is NOT registered in the video registry this build - r3 proved a
registered engine cannot be kept out of the episode dropdowns
(role_compat grants every role the full input vocabulary; the dropdown IS
the registry) - so the dropdown entry follows in a later build behind a real
standalone-only boundary. The runner never invokes TTS or music generation; the clip's invented score
is its audio. Normal H3 lanes (video/audio_in) DROP the audio latent
in-graph, LTX-style - the unused audio is never even decoded, so nothing is
rendered-then-discarded.

**Runner options (operator request 2026-08-10, hybrid experiments):**
- `--stem-wav`: also export the native invented score as a sidecar .wav
  (one cheap audio-VAE decode; the stem becomes reusable material).
- `--mux-tts <wav>`: produce an ADDITIONAL review copy with a supplied real
  TTS voice mixed OVER the invented music (music ducked under speech,
  ffmpeg-side; both originals preserved). Lab precedent:
  `h3_r2v_otr_source_mix_with_generated_stem_AUDITION_ONLY.mp4`.
  These outputs are runner deliverables and never enter episode assembly;
  if a hybrid mix ever graduates toward episodes, it re-enters through a
  fresh ear gate and its own spec, not through this flag.

**DESTINATION ARCHITECTURE RULED (operator, 2026-08-10): "the mime video
dropdown OVERRULES any TTS or music dropdowns."** This resolves the long-
deferred option-(a)-vs-(b) fork to **(a)**: mime becomes a first-class
dropdown engine, and selecting it for a beat means that beat renders NO TTS
and NO music - the video's invented score IS the beat's audio, entering the
master pre-freeze so V-1's identity assert still holds (the frozen master
simply CONTAINS the mime window; the memory-banked insertion point is the
segment list at scene_sequencer.py:1296-1320, before
`_stamp_master_audio_identity`).

Honest cost, and why it is a FOLLOW-UP spec rather than a line item here:
audio-first ordering means the master WAV freezes BEFORE video renders -
but a mime beat's audio does not exist until its video renders. So option
(a) requires a phase inversion for mime beats: their duration comes from
the script target (not TTS samples), their video renders EARLY, and their
stem joins assembly pre-freeze. That is sacred-path surgery with its own
kibitz arc - the mime scoping doc P3 always pointed at, now with its answer
decided in advance. Sequence: THIS build ships the runner (immediate
capability + the test bed that proves stem quality and the mux behavior);
the dropdown-overrule wiring is the next spec after the transplant, built
on the runner's proven pieces. The registry's standalone/overrule boundary
(r3 MUST-FIX 1's option b) gets built THERE, where it has a real consumer. Prompt-builder rule (operator): the score
request is derived from the beat's dramatic intent, in the "tense orchestral
drama" register family, never boilerplate. Naming: the operator keeps "mime"
(his word, his product); label clarifies: proposed `h3_low_mime` /
"MiniMax H3 - Mime (self-scored, body-acted, no lip-sync)". Enhancement
question still queued in the lab: the prompt-only origin test (whether
DROPPING picture conditioning frees even better music).

---

## Build order - SUPERSEDED by the 21-lane-packet plan (r4 adjudication)

The 9-step sequence below is HISTORICAL, retained for its per-item detail.
The operative build order is the 21 lane packets in
`docs/2026-08-10-FINAL-QA-video-build-corpus.md`, adopted at r4 because it is
a strict refinement of this one, not a competitor: every lane's RELATIVE
order is identical, but it (a) refuses to GROUP lanes - which is what the
operator's build law literally says ("build one lane at a time and test it...
so a bug we hit on one gets PREVENTED on the next"), while this sequence
grouped the HuMo pair, the WAN pair and the LTX trio; (b) schedules the 9
cheap lanes neither document scheduled (mesh_stage, 4 viz, 4 still - exactly
half the roster); and (c) does per-lane naming instead of one big-bang naming
commit, which shrinks the blast radius of an alias-move mistake from nine
engines to one.

Two caveats from r4, needing explicit confirmation rather than silent
adoption: the QA plan splits H3 video and H3 audio-in into SEPARATE lane
commits (finer than S4's "registration changes INPUT_TYPES so they cannot be
split" and finer than any judgment authorized - plausible, since the mouth
carve-out is audio-in-only, but unblessed); and the QA doc's own
contradiction #3 misreads the transplant plan's H3 ordering (claims it builds
H3 first; the plan says LAST), so that document gets the same grounding
treatment as every other reviewer's output.

**Scaffolding that must exist BEFORE lane 1 runs its prescribed loop (r4):**
`docs/LANE_BUILD_LESSONS.md` and `tests/test_lane_preflight_matrix.py` do not
exist yet, and the per-lane loop's first two steps read them. They ship with
the evidence-manifest commit, and S9 is built INCREMENTALLY per lane rather
than as one immutable whole (r4: several receipts cited from lab `4d87cfa`
are absent, and wan_i2v's own exoneration receipt is not among them).

## Historical commit sequence (each chunk: code -> wire -> regress -> commit -> push)

1. S9 evidence manifest.
2. S1 wan_i2v canvas + comment rewrite + two drift tests.
3. S8 boot contracts via `profile.launch` + ShotLock preflight enforcement +
   receipt stamping (acceptance scoped to `humo_diet` only - H3 does not exist
   yet).
4. Cost-row QUALIFICATION RUNS through OTR's real lifecycle (lab job B supplies
   candidates; rows ship DISQUALIFIED until their run exists), then S2
   recalibration + qualification + `vram_admission` plumbing, one commit,
   including `eng_fastwan_8gb.py:296`.
5. S7.1 instrumentation.
6. S7.2 release (branch chosen by 5's report) + S7.3 planner warning.
7. S4 H3 adapters: two engines, fps conversion, mouth carve-out at
   `render_driver.py:1548`, still_plan, continuity, Sage refusal, V-1 self-
   probe, `frame_contract.py:108` doc fix; PLUS S6 public ids, regenerated
   ENGINE_MATRIX.md, and S10 node-87 enumeration - ONE commit (registration
   changes INPUT_TYPES, so they cannot be split).
8. S3 LTX HQ canvas + profile.
9. P1 profiles once the 14B measurement lands and the operator confirms.

Regression after every commit touching INPUT_TYPES, widgets_values, or
workflow JSONs: AST parse, dead-ref grep, cross-import check, widget count vs
widgets_values audit, plus the full Windows regression suite and Bug Bible run.

## Known test impact (fold into each commit, do not discover at CI)

WILL FAIL BY DESIGN: `test_public_engines.py:42`;
`test_engine_matrix_doc.py:43-49`; `test_vram_admission_boundary.py:42-82`;
`test_clip_fill.py` (x6, computed from the old row); `test_fastwan_8gb.py:71-72`;
`test_cs3_inter_beat_reclaim.py:60-79`;
`test_engine_contract_roster.py:294` (only if H3 declares safe_render_frames).
WATCH FOR FALSE PASSES: `test_cs3` (CPU stubs never satisfy a headroom gate);
`test_engine_contract_roster.py:126-137` (early-returns on falsy target_fps -
the 24/25 defect has NO test until S4.1 adds one);
`test_wire_w7_mouth_ownership.py` (family-parameterized, misses an engine-id
carve-out); `test_audio_byte_identical.py` (fixture self-consistency only).

## Acceptance

1. Per-leg canonical smokes: wan_i2v at declared canvas; LTX HQ; each H3
   adapter under the `h3` boot; HuMo 1.7B under `humo_diet`; wrong-boot
   refusal for H3 under `default`; a beat SHORTER than H3's minimum (107) to
   exercise trim, with the trim ratio logged loud. (Corrected r4: "shorter
   than H3's minimum" means shorter than **129 canvas frames**, not the void
   107.)
2. S7 historical topology: 200f chained beat first, then 65f - completes.
3. New H3 fps tests per S4.1 (grid legality, index-map bounds, the
   `abs(j/25 - idx[j]/24) <= 1/48` drift assertion, and the two pinned
   partitions).
4. ONE end-to-end episode, run LAST, with every chunk in and the full roster
   in the dropdown: RESULT SUCCESS, canonical episode assets, obs_publish OK,
   including at least one same-engine chained beat and one H3 character beat.
