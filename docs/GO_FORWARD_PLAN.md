# OTR GO-FORWARD PLAN -- THE SINGLE SOURCE OF TRUTH

> **This file is canonical.** The forward order, runway, open tickets, current step, hard
> rules, and sprint lanes all live HERE (one-doc rule, operator-directed 2026-06-12).
> `docs/VIDEO_BUILD_HANDOFF.md` and `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` section 0
> are now THIN POINTERS to this file. The `otr-build-tracker` artifact is the visual
> DASHBOARD (gauge + lanes) and mirrors this doc -- it is not the source of truth.
> Dated `docs/<date>-*` folders are EVIDENCE records (roundtables, problem statements), not
> plans. When this doc and any other disagree, THIS doc wins.
>
> **Last updated:** 2026-06-12 (planner HARDENING pass on the Wan coder prompt: grounded the on-disk reality -- 16gb engine is a low-noise-only PLACEHOLDER graph (no high-noise expert on disk), 8gb tier is TWO fetches (5B + wan2.2 VAE), sage-sidecar + assert-in-trace + render-phase-NVML caveats added. Hardened kickoff ready. NEXT = the Wan smoke, coder window. HEAD ae94970, 22 commits unpushed; docs edits this session uncommitted.)
> **Branch:** `v2.0-alpha`. **HEAD:** see git (commit pending this session's harness fixes; do NOT
> push unprompted). Update the "Last updated / HEAD" line and the relevant section on every tick.

---

## 1. CURRENT STEP

**ALL THREE 0-E ENGINES ARE GREEN** (ltx_orbit + still_parallax + mesh_stage each PASS on GPU with
the engine IN THE TRACE as final_engine -- autonomous run 2026-06-12 night). The 3D quick-smoke goal
is DONE.

**ACTIVE THREAD = VIDEO MOTION ENGINE SELECTION (operator-driven).** Today's LTX is too static; a
full investigation (fast isolated LTX smoke `scripts/otr_ltx_motion_smoke.py` + MAD metric
`scripts/otr_ltx_mad.py` + a 3-model roundtable, all this session) proved the motion gap is the
MODEL: LTX-2.3 22B = real motion but does NOT fit the 14.5GB ceiling; the old 2B v0.9 we run fits
but warps. **DECISION (operator): move the video engine to the Wan 2.2 family, two sizes mapped to
the 8gb/16gb profile tiers, lip-sync kept SEPARATE (LatentSync/HuMo).**

**NEXT CONCRETE STEP -- the Wan 2.2 video-engine SMOKE (coder window):** see
`docs/2026-06-12-ltx23-motion/WAN_VIDEO_CODER_PROMPT.md` (the ready-to-paste kickoff). Smoke-first,
EYEBALL-gated (visual, not MAD):
- **16GB tier = Wan 2.2 I2V-14B** -- LESS DONE THAN THE OLD NOTE CLAIMED (planner hardening
  2026-06-12): `eng_wan_i2v.py` exists but its `_build_graph` is a SINGLE-UNETLoader + single-KSampler
  PLACEHOLDER marked "ASSUMED/VERIFY-ON-GPU" -- it does NOT implement the two-expert HIGH/LOW MoE
  sigma-split its own docstring promises. ONLY the LOW-noise expert is on disk
  (`wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`, 13.3GB); the high-noise expert is NOT.
  -> SMOKE Path A (low-noise-only, zero download) first; Path B (true two-expert A14B = fetch high-noise
  + rebuild w/ ModelSamplingSD3 + KSamplerAdvanced split) only if A's motion disappoints. The engine
  already does `free_after_use keep={unet,vae,terminal}` (umt5 TE freed pre-sampler, CS-4-style) but
  it is UNVERIFIED on GPU. Fetch GGUF Q5 only if the fp8 render-phase peak busts 14.5GB.
- **8GB tier = Wan 2.2 TI2V-5B GGUF Q6** -- TWO fetches (NOT one): the 5B model AND the new Wan2.2
  high-compression VAE (only `wan_2.1_vae` is on disk and it won't drive the 5B). Dense 5B (not a MoE);
  different node class; wire as a 2nd selectable Wan engine, smoke one b-roll clip.
ACCEPTANCE: operator eyeballs both clips (real camera motion + still preserved + NO warp); audio
spine untouched (Wan video is silent; mux byte-identical); smoke must ASSERT wan-in-trace (no
fallback-PASS, CS-1) + record render-phase NVML separately from whole-run (CS-2); only THEN map to
tiers + wire the episode (CS-3 = Wan+HuMo co-stage risk). NOTE: `UnetLoaderGGUF` installed; Wan S2V
sampler nodes are NOT; `umt5_xxl_fp8` + `wan_2.1_vae` on disk; SageAttention is currently importable
(BUG-070: `resolve_isolation` escalates wan to a cu128 sidecar -- decide uninstall-sage vs sidecar
before the smoke). All models under `C:\ComfyUI-Models`. Hardened kickoff:
`docs/2026-06-12-ltx23-motion/WAN_VIDEO_CODER_PROMPT.md`.

---
### (historical -- superseded 2026-06-12 night; all three engines now GREEN)
**Item 4 -- the dropdown coverage sweep -- is DEFERRED to an on-demand overnight run** (operator
chose to code first; GPU freed 2026-06-12). Fire it via the one-click `otr-overnight-sweep`
scheduled task ("Run now" at bedtime): it boots a fresh headless ComfyUI on :8000 with whatever
code is current, runs all 27 runnable legs on the `humo_1.7B` default, and the `otr-sweep-monitor`
task writes `scripts/sweep_monitor_digest.md` every 30 min and creates `scripts/_otr_0e_gpu_go.txt`
on a clean 27/27 PASS (else it HOLDS and reports the failures).

**Track-3 (s1) is COMPLETE** (planner audit 2026-06-12 -- image-routing must-fixes + builder
migration + cache-key split all verified landed; no open code). The next forward-order code (s2 =
3D spike lane) is operator-gated.

**IN FLIGHT (detached, decoupled from the 27-leg sweep -- operator-directed 2026-06-12):** quick 3D
smoke -- one 30-word character-slot test per 0-E engine, EASIEST -> HARDEST (`ltx_orbit` ->
`still_parallax` -> `mesh_stage`), via `scripts/otr_3d_quick_tests.ps1` on a fresh :8000. Results +
verdicts land in `scripts/otr_3d_quick_digest.md` (marker `scripts/.otr_3d_quick_active` exists
while running). **Round 1 (10:43) = 3/3 SOAK_FAIL, ALL HARNESS-SIDE -- fixed 12:00-12:16, clean
re-run launched 12:16:43** (server boot ~7.5 min w/ full custom nodes, then ~10-17 min/leg).
Round-1 root causes + fixes (all landed, uncommitted-or-committed per git):
1. Sweep checked the OLD output tree (Documents) while the hand-rolled server boot wrote to the
   install root -> orphan-report reject. FIX: `otr_coverage_sweep.py` resolves the server output
   dynamically (env override, else newest `otr/episodes` mtime, LOUD).
2. The ps1 hand-rolled the server boot -- no 0-E enable flags (every 3D engine fell back
   `gated_by_flag`), no TEMP repoint (`otr_floor_*` leaked into %TEMP% = hygiene fail), wrong
   output dir. FIX: ps1 now boots via the CANONICAL `_otr_soak_server_launch.cmd` and stages
   `OTR_ENABLE_LTX_ORBIT/STILL_PARALLAX/MESH_STAGE=1` through the launcher's
   `_marathon_extra_env.cmd` seam (removed post-health).
3. Headless boots loaded ONLY the install-root custom_nodes (OldTimeRadio junction) -- NO
   ComfyUI-LTXVideo wrapper -> `WrapperNodeMissing: LTXVImgToVideoConditionOnly` -> ltx_video
   fell to the floor on EVERY leg (this also would have broken the 27-leg overnight sweep).
   FIX: launcher passes `--extra-model-paths-config scripts\_otr_headless_model_paths.yaml`
   (headless-safe copy of the Desktop yaml; the Desktop one's desktop_extensions points at the
   dead v1 install path and crashes main.py prestartup).
4. ps1 `foreach($pid ...)` used the read-only automatic $pid -> instant death when a :8000
   listener existed. FIX: renamed $srvPid. (Plus a one-line for-loop comment bug.)
KNOWN-GOOD evidence from round 1: episode renders end-to-end, obs final playable, audio
byte-identical; all failures were env/harness. ACCEPTANCE for round 2: engine under test appears
in the trace (not fallback-only) + capstone gates. NEXT WINDOW: read the digest; if a leg fails
ENGINE-side now, fix that engine and re-run just its leg (server must be up; PYTHONPATH to the
install root; `python scripts/otr_coverage_sweep.py --only other_beats_visual_<engine>`).
NOTE: suite 4141/0 + Bug Bible green after the sweep-resolver change.

---

## 2. HARD RULES (invariants -- apply every session)

- The forward order is section 3 (below). Do NOT start/resume/"continue" any OTHER sprint --
  NOT story-spine, NOT story-pipeline, NOT the broader audio stack, NOT any other ROADMAP item.
  Those are PARKED (section 8).
- The audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. The ONLY sanctioned audio work is the character-voice
  "whiny" fix (`docs/2026-06-10-character-voice-whiny-fix__problem-statement.md`) -- UPSTREAM TTS
  only.
- EVERY session (planner AND coder) UPDATES this doc + the `otr-build-tracker` dashboard (content;
  preserve the gauge + lanes styling). Never tell a window "don't touch the tracker".
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100% local/offline;
  determinism (seed-keyed); every in-render fallback LOUD; UTF-8 no BOM; SFW; V-12 dependency
  isolation; no new widgets in the static workflow shell (V-11).
- GIT POLICY (operator 2026-06-10): ONE branch `v2.0-alpha`; commit AND push together per green
  chunk; the operator eyeball gates TAGS/promotions only; after every push verify HEAD==origin /
  no 0-byte / no BOM / AST parse on touched .py.
- v2.0 PRODUCTION / `main` is GATED until all operator work is done; a `v2.0-alpha-stable` tag on
  `v2.0-alpha` is fine; prod/main is NOT.
- COORDINATION (operator 2026-06-11): ONE coder window in the repo's code at a time; the 0-E
  Phase B agent and any coder window serialize via the GO file. Never two coders in overlapping
  files.
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) are gated behind `OTR_C7=1` (fix @847e2de).
  Production runs must log `cast RNG seed=... (OS entropy)`; "override" in the log means a stale
  env var is pinning the seed -- do NOT set `OTR_C7` for normal runs.

---

## 3. FORWARD ORDER (do in this sequence; the GATE detail is in section 5 of the 3D plan spec)

1. **Punch list** -- captions + LTX radio open + procgen rolling credits baked INTO the production
   JSON, proven by a render FROM it; operator look-QA. *(GATE A)*
2. **latentsync-100% + the demos** -- the `OTR_LSYNC_BASE_ENGINE=still_kenburns` fix + the two-demo
   set AND the mixed showcase episode. *(GATE A)*
3. **Switchable foundation S0 -> S1 -> S2** -- profiles + registry enable-set + the ONE applier that
   DELETES the hand-coded patch lists (the drift cause) + the 3 code-defect fixes. *(GATE B)*
4. **Dropdown coverage sweep** -- every announcer/music/cast engine option renders a 30-word FULL
   episode on the S2 applier; no crashes, credits + subtitles present. *(GATE A acceptance, powered
   by GATE B's applier)* -- **CURRENTLY HERE; deferred to the overnight task (section 1).**
5. **THEN the 3D sprints** -- begin with the 3D plan's image-routing must-fixes (section 3 of that
   spec -- now LANDED), then the `character_3d` family.
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README. *(closing
   phase)*

**0-E PARALLEL TRACK (additive 3D easy on-ramp, operator-ordered 2026-06-11):** `ltx_orbit`,
`still_parallax`, `mesh_stage` -- three no-toolchain LOCAL engines. CPU side SHIPPED @ `1daaa6a`
(suite 4096/0; selectable-not-default; LICENSE_RECORD gates default-on). Phase A COMPLETE
2026-06-11 (Blender 4.5.10 pinned + cube self-test PASS; hy3d ckpt + DA-V2-S fetched sha-verified;
4100/0 @ `124e90c`). Phase B (E-1 probe, E-6 renders, per-engine sweep legs) HELD on the GO file
the overnight sweep creates.

**AUDIO PARALLEL TRACK (own window, never blocks the video serial order):** the character-voice
"whiny" fix -- own plan v3.1 (`docs/2026-06-10-character-voice-whiny-fix__problem-statement.md`,
@`9181fda`). Land P-OBS + P0-zero + the cheap ref/delivery fixes BEFORE the operator's video
look-QA so the demos sound right. Frozen audio spine untouched (upstream TTS only).
(Operator note 2026-06-12: whiny voice may have self-resolved -- verify before scheduling work.)

---

## 4. RUNWAY TO DONE (sprint count -- update on every tick)

"Done" = the platform WIRED into real episodes (real per-beat video + byte-identical mux + the
legacy procgen-only path gone) + all video models verified live + the first 1-2 3D models
rendering. The future-state 3D playground is BEYOND done (separate project).

~6-9 coder-window sprints remain:
- **(s1) Track-3 remainder** -- W7-pre slice, ImageDirector fail-closed, builder migration, cache
  keys. **DONE 2026-06-12 (planner audit, code-verified):** all landed -- image-routing must-fixes
  (68/1 green), schema-valid `build_request` (init_w/init_h extras gone, observability on the real
  field), and the slice/curve cache-key split. No open code in s1.
- **(s2)** S-3D-0 + T1 + T2a (the lane-killer spike + template + wrap smoke).
- **(s3-s4)** T3 corpus + T2b KEYSTONE (timeboxed ~1 week, the big GO/NO-GO).
- **(s5)** T4 driver + alpha + LOOK gate.
- **(s6-s7)** W7 production wiring + soak = "v1-usable".
- **(s8-s9)** closing S3-S6 distribution.

**TWO SHORTCUT FORKS:** S-3D-0 NO-GO (wheels fail + operator declines the cu128 toolkit) OR T2b
keystone NO-GO -> contingency = HuMo-2D stays, `character_3d` defers -> done collapses to ~2-3
sprints (0-E engines + closing phase). 0-E ships the visible 3D win independent of the long lane,
so the keystone carries no demo pressure.

**Done definitions stay split:** "v1-usable" (one engine, one real episode) vs "B-parity ship"
(>=2 engines binds at SHIP, not first light).

---

## 5. LIVE STATUS + OPEN TICKETS

**Gauge: ~90% to done.** Lane status (the tracker dashboard mirrors this):
- **Lane 1 -- Platform built + B-shipped:** DONE (M0-M5; model-agnostic engine platform + HuMo-2D
  proven).
- **Lane 2 -- Wired into real episodes:** DONE (full smoke renders real beats headless, mux audio
  byte-identical to master).
- **Lane 3 -- Video models verified live:** ~60%. CS-4 RESOLVED (humo_1.7B default @ `955f134`);
  LTX GPU-verified; Flux live; LK look fixes shipped @`8115c72`. Remaining = the coverage-sweep
  remainder (overnight) + Wan/latentsync legs.
- **Lane 4 -- First 1-2 3D models rendering:** ~65%. 0-E CPU chain SHIPPED + Phase A complete;
  remaining = E-1 probe + E-6 renders (held on the GO file) + look-QA + license sign-off.

**Open tickets:**
- **CS-4 -- RESOLVED-BY-REROUTE 2026-06-11** (default char tier -> `humo_1.7B` @ `955f134`; 14B =
  opt-in, OPERATOR-DEPRIORITIZED). Mechanism: the umt5 TE stays 5,248 MB resident through HuMo
  sampling -- fine for the 1.7B stack, fatal for the 16.5 GB 14B. NO code regression. ACCEPTANCE
  PASSED: humo_1.7B leg = PASS 38 min, histogram {ltx_video:3, humo_1.7B:3}, audio byte-identical,
  render-phase peak 10,305 MB. `CS-4-open` (lazy): targeted post-encode TE detach for the 14B
  opt-in lane. Evidence: `docs/2026-06-11-coverage-sweep-triage__tickets.md`.
- **CS-1** -- the two latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was
  fallback-only); BOTH re-run in the sweep.
- **CS-2** -- machine NVML pins ~16 GB on every leg vs the 14.5 ceiling while driver-phase
  attribution reads 3.1-3.5 GB; needs phase attribution (the 1.7B leg's 10,305 MB render-phase
  peak is a partial answer).
- **CS-3** -- wan_i2v legs need Wan AND HuMo in one episode; if they always co-stage, wan options
  are 16gb-tier-incompatible as wired -- the supervised wan batch decides.
- **TRACK-3 (s1):** the 3D image-routing MUST-FIXES are **LANDED + green 2026-06-12**
  (`video_policy_json` required+forceInput+fail-closed; `enforce_3d_granularity_lock` raises;
  `_is_3d_engine` reads the real `requires_mesh_portrait` capability; that field is real on
  `AdapterDescriptor` + `VideoProfileRow`; the dispatcher per_beat HALT). Tests:
  `test_image_platform_c1.py` + `test_otr_workflow_validator.py` = 68 passed / 1 skipped. Doc
  corrected: 3D plan section 3 banner. **s1 builder migration + cache keys ALSO landed**
  (`build_request` emits a schema-valid `VideoRequest` -- the init_w/init_h extras are gone,
  observability rides the real field; the slice/curve cache-key split is shipped). **Track-3 (s1)
  is COMPLETE -- no open code.** The next forward-order code (s2 = S-3D-0 + T1 + T2a) is the 3D
  spike lane, GATED on the operator green light + the coverage sweep.
- **LK-1** (LTX look restoration) -- BUG-LOCAL-113 (FLUX colour bleed) + 113b (LTX ksampler 30-step
  default) FIXED @`8115c72`/`e3edce9`. Stills confirmed good.
- **0-E on-ramp** -- tickets E-1..E-7; gated on the sweep GO file; coder-window ready.
- **OH (output-tree consolidation)** -- OH-0..3 done; **OH-4** (14-entry / ~8.2 GB live->attic
  migration) STAGED, AWAITS operator "go OH-4". Contract:
  `docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md`.
- **Operator gates (unchanged):** ComfyUI Desktop relaunch (look-QA), fresh-render acceptance,
  latentsync demo set + mixed showcase, whiny-voice P0 matrix + reel, S-3D-0 green light,
  `v2.0-alpha-stable` tag decision.

---

## 6. WHERE WE ARE (factual; recent first)

- **2026-06-12 (~23:00, LTX motion -> Wan 2.2 DECISION, Opus -- one long session):**
  - **3 engines GREEN** (above). Plus shipped this session: boot fix (`748955b` UTF-8 + cmd quoting),
    ltx_orbit API drift (`bcd811b`), stills-first reclaim (`e9743cc`, no-op on DynamicVRAM), motion
    prompts (`db14f9e`), every-i2v-beat-gets-a-still + LOUD missing-still trace + music-open->inter +
    video-seed stamp (`e635596`), auto render-launcher + watchdog (`fcae8e0`/`748...`), mesh_stage
    audio false-fail fix (`f43db04`), soak NameError (`0901035`), Blender env hydrate (`439e481`),
    mesh_cache under output (`f2dca88`), SaveGLB --disable-metadata (`69cb5ea`), mesh_cache under
    episodes/_shared (`9d4a6b1`).
  - **LTX MOTION INVESTIGATION (operator-driven, then PARKED in favor of Wan):** built a FAST isolated
    LTX smoke (`scripts/otr_ltx_motion_smoke.py`, ~10-30s/clip, SaveWEBM) + MAD metric
    (`scripts/otr_ltx_mad.py`). Decoded the 6/1+Goofer recipe (sampler `euler_cfg_pp` not `euler`;
    distilled SamplerCustomAdvanced + 8 sigmas + cfg 1.0; cond_strength 0.75). Made the engine sampler
    swappable (`OTR_LTX_SAMPLER_NAME` @`23aca22`). RESULT: on the v0.9 2B model NOTHING reproduces the
    motion (euler 0.59 MAD freeze; cfg_pp 0.88 pan; cfg_pp@257 4.2 but WARPS per operator eyeball;
    distilled chain 0.6-0.9; t2v 0.7). Root cause = the MODEL: **ComfyUI-Goofer runs LTX-2.3 22B-distilled
    + gemma + a camera LoRA**, not v0.9. Clips in `docs/ltx_motion_clips/`, write-up
    `docs/2026-06-12-ltx-motion-sweep.md`.
  - **ROUNDTABLE** (gpt-5.5 + gemini-3.1-pro + grok-4.3 + Claude judge, ~$0.14,
    `docs/2026-06-12-ltx23-motion/roundtable/`): UNANIMOUS the 22B-distilled fp8 (23.5GB) does NOT fit
    the 14.5GB ceiling; the camera LoRA is model-matched (can't help v0.9). MAD != the visual gate.
  - **DECISION (operator):** video engine -> **Wan 2.2** family, lip-sync kept SEPARATE
    (LatentSync/HuMo). 16gb = Wan 2.2 I2V-14B (already wired `eng_wan_i2v.py` + fp8 on disk); 8gb =
    Wan 2.2 TI2V-5B GGUF (fetch). Ready-to-paste coder kickoff:
    `docs/2026-06-12-ltx23-motion/WAN_VIDEO_CODER_PROMPT.md` (the S2V variant was superseded by the
    cleaner motion/lip-sync separation).
  - **22 commits on `v2.0-alpha`, HEAD `ae94970`, NONE pushed** (operator gate). Server idle, GPU clean.

- **2026-06-12 (~20:15, ALL THREE 0-E ENGINES GREEN, Opus -- autonomous run):** every 0-E engine
  now PASSES on GPU with the engine IN THE TRACE as final_engine:
  - **ltx_orbit** PASS `{ltx_orbit:3, ltx_video:3}`; render-phase 10.9GB.
  - **still_parallax** PASS `{still_parallax:3, ltx_video:3}`; render-phase 11.1GB (a soak NameError
    `master_sha` masked an earlier green -- fixed @`0901035`).
  - **mesh_stage** PASS `{mesh_stage:2, ltx_video:3}`; render-phase 10.8GB; hy3d-2mv mesh -> SaveGLB
    -> Blender orbit all live. SIX fixes to get there: soak 2-master audio false-fail (`f43db04`),
    soak NameError (`0901035`), hydrate OTR_BLENDER_EXE in the launcher (`439e481`), mesh_cache under
    the CONFIGURED output for SaveGLB (`f2dca88`), `--disable-metadata` for core SaveGLB's cls.hidden
    (`69cb5ea`), mesh_cache under otr/episodes/_shared for the output-tree contract (`9d4a6b1`).
  - TOOLING: `scripts/otr_run_leg.ps1` is the ONE render-launch path (reset->boot->leg->auto-armed
    watchdog `scripts/otr_render_watchdog.ps1`, writes `<leglog>.watchdog` RUNNING/DONE/DEAD; declares
    DEAD on a 5-min stall or down queue). `OTR_SMOKE_WORDS` env added for a longer episode.
  - The watchdog caught a transient writer flake (style inventor recovered 4/5 descriptors) in 0 min
    instead of a wasted 24-min wait -- re-ran clean.
  - A longer (90-word) look-QA episode is rendering for the b003-middle-beat + motion-prompt eyeball.
  - ~16 commits this session on `v2.0-alpha`, NONE pushed (operator gate).

- **2026-06-12 (~15:05, ltx_orbit leg = FULL PASS, Opus):** the clean GPU re-run of
  `other_beats_visual_ltx_orbit` **PASSED 1/1** (24 min; the real LTX render is ~3.8min/clip vs the
  instant fallback). Histogram `{ltx_video:3, ltx_orbit:3}` over 6 beats -- ZERO still_kenburns
  fallback; the API fix is fully validated. `V-3 render-phase VRAM peak OK: 10916MB <= 14848MB`
  (the 16009MB whole-run spike was the Flux still phase, informational). Audio byte-identical OK;
  playable obs final published. **KEY REFRAME: the VRAM ceiling was NOT the real blocker** -- the
  earlier 15093/15245MB "render-phase" peaks were the FALLBACK path (still models resident while
  still_kenburns ran in-phase); with the real engine the render-phase peak is a comfy 10.9GB. So the
  V-3 measurement/eviction debate is MOOT for ltx_orbit. `e9743cc`'s reclaim detached 0 on DynamicVRAM
  but its gc+soft_empty_cache may have helped trim the render-phase baseline; harmless either way.
  Idle-after-run VRAM sat ~60% (9850MB = ComfyUI staging/cache of the resident server, NOT a leak) ->
  killed per the new directive -> 2053MB. OPEN: re-run still_parallax (renders real; likely PASSes the
  same way) + mesh_stage (still has the AUDIO-NOT-BYTE-IDENTICAL dig, task 6).
  - **ltx_orbit API fix PROVEN ON GPU.** Clean re-run leg: the server log shows a LIVE 30-step LTX
    KSampler @ ~7.6s/it -- ltx_orbit now renders REAL LTX video instead of the instant still_kenburns
    fallback (the leg runs ~3-4min/clip longer for exactly this reason). The `positive`-kwarg drift is
    fixed end-to-end @`bcd811b`.
  - **VRAM eviction (e9743cc) is a NO-OP on this ComfyUI -- DynamicVRAM.** The pre-render reclaim
    FIRED (`LOUD VRAM reclaim (pre-render: all stills minted...): detached 0 resident model(s)`) but
    detached ZERO: ComfyUI 0.24.1 uses DynamicVRAM ("Model FluxClipModel_ prepared for dynamic VRAM
    loading. 4777MB Staged"), so models are STAGED, not in `current_loaded_models` -- the BUG-291
    `reclaim_idle_models` detach pattern finds nothing. Flux's ~4.7GB stays staged; peak still 16009MB.
  - **The peak is NOT fatal (operator catch).** 16.3GB card, the render COMPLETES at 16009MB -- no OOM.
    The 14.848GB V-3 gate is a SOFT policy gate that reads the whole-machine NVML pin (incl. the
    DynamicVRAM-staged still models), NOT the ~3-3.5GB the render engine actually uses (CS-2). So the
    real fix is the V-3 MEASUREMENT (attribute the render-phase engine VRAM / discount staged-but-idle
    still models), OR a DynamicVRAM-aware free, OR adjust the ceiling -- NOT the detach eviction.
    DECISION PENDING. e9743cc left in place (harmless, LOUD, helps a non-DynamicVRAM box) but it does
    NOT move this gate.
  - NET: ltx_orbit's engine bug is FIXED; the remaining "fail" on every leg is the V-3 VRAM gate,
    which is a measurement/policy question, not a render failure. mesh_stage's AUDIO-NOT-BYTE-IDENTICAL
    is still its own separate dig (task open).

- **2026-06-12 (~14:05, 3D quick-smoke ALL 3 LEGS RAN, Opus):** harness fully working; all three
  legs produced verdicts (none PASS yet -- real engine-side findings):
  - **ltx_orbit** = was fallback-only (histogram all still_kenburns). Root cause: the installed
    `LTXVImgToVideoConditionOnly` was rewritten (now `vae,image,latent,strength`->LATENT, no
    positive/negative) -> OTR's old kwargs raised "unexpected keyword argument 'positive'".
    FIXED @`bcd811b` (rewired `_build_graph_i2v`; motion contract preserved; test updated). NOT yet
    re-run on GPU to confirm engine-in-trace.
  - **still_parallax** = REAL ENGINE RUNS (histogram still_kenburns:3 + still_parallax:3 / 6 beats);
    fails ONLY the VRAM ceiling (`render-phase peak 15245MB > 14848MB`).
  - **mesh_stage** = fallback-only (fell to still_parallax) AND `AUDIO NOT BYTE-IDENTICAL`
    (final 58b8cb2c != master b8e36186) -- a frozen-spine violation specific to the mesh_stage path;
    needs its own dig.
  - **COMMON BLOCKER = the 14.848GB VRAM ceiling** -- every leg peaks ~15.2-15.9GB because the
    Flux/portrait/HuMo still-phase models are NOT evicted before the 3D/video render phase. This is
    the operator's "stills first, then 3D" insight = CS-2. The fix (evict-before-render / phase
    ordering, OR correct the V-3 NVML phase attribution) is the next decision; it unblocks ALL legs.
  - **BASELINE DRIFT FOUND:** `sageattention` is now importable in the venv (`find_spec` True). This
    pre-dates this session and FAILS 5 sage/dep-gating tests (test_video_motion ltx/ltx_orbit/wan
    usable-flag + test_video_dep_pilot x2) -- NOT regressions from the ltx fix (confirmed by stash
    test on clean HEAD). It also reopens BUG-070 risk for LTX (Sage process-aborts LTX). Suite is
    therefore 4136/5 on this machine, NOT the handoff's "4141/0". NOT baselined into EXPECTED_FAILED
    (would hide the BUG-070 risk) -- operator decides (uninstall sageattention vs accept).
  - Commits this session: `748955b` (boot fix) + `bcd811b` (ltx i2v fix), both on `v2.0-alpha`, NOT
    pushed.

- **2026-06-12 (~13:27, ROUND 2 root-cause + fix, Opus):** the round-2 quick-smoke had ABORTED
  ("SERVER DID NOT COME UP" 12:28). Two NEW bugs, both now fixed + verified:
  (1) **UTF-8 boot crash** -- the detached headless launcher inherited the cp1252 console codec, so
  OTR `prestartup_script.py` crashed the instant it printed an emoji (UnicodeEncodeError on U+2705/
  U+2713) -> boot died ~13s, exit 1. ComfyUI Desktop used to set UTF-8 for us; the v2 install move
  dropped it. FIX: `set PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8` in `_otr_soak_server_launch.cmd`.
  (2) **Start-Process quote-mangling** -- the ps1 booted via `cmd.exe /c "<launcher>" "<log>"`, which
  hits cmd's two-quoted-token stripping rule (4 quotes -> outer pair eaten -> mangled path -> launcher
  never ran -> ZERO log output). FIX: run the `.cmd` directly as `-FilePath`; also hardened the kill
  (CIM CommandLine, not `Get-Process .Path` which is blank for unreadable processes). VERIFIED: clean
  boot reaches healthy in ~20s (NOT the 7-8 min the prior handoff feared -- it was crashing, not slow);
  `system_stats`/`OTR_VideoDirector`/`LTXVImgToVideoConditionOnly` all 200; 0 import failures; 0 charmap.
  Harness now runs the legs (ltx_orbit first). Touched: `scripts/_otr_soak_server_launch.cmd`,
  `scripts/otr_3d_quick_tests.ps1`. NOTE: this UTF-8 fix ALSO unblocks the 27-leg overnight sweep
  (same launcher) -- every leg would have died the same way. Commit pending; do NOT push unprompted.

- **2026-06-12 (midday, harness-fix session -> Opus):** 3D quick-smoke round 1 = 3/3 SOAK_FAIL,
  all harness-side (see section 1 for the 4 root causes + fixes: sweep output resolver,
  canonical-launcher boot + 0-E flags via env seam, headless model-paths yaml for the wrapper
  custom nodes, $pid PS bug). Suite 4141/0 + Bug Bible green. Clean re-run launched 12:16:43
  detached; monitor `scripts/otr_3d_quick_digest.md`. The `--extra-model-paths-config` launcher fix
  also UNBLOCKS the 27-leg overnight sweep's ltx/latentsync legs (they would all have floored on
  WrapperNodeMissing). Touched: `scripts/otr_coverage_sweep.py`, `scripts/otr_3d_quick_tests.ps1`,
  `scripts/_otr_soak_server_launch.cmd`, NEW `scripts/_otr_headless_model_paths.yaml` (+ throwaway
  helper ps1s under scripts/). Commit pending; do NOT push unprompted (operator kickoff rule).

- **2026-06-12 (cont.):** Track-3 (s1) verified COMPLETE (no open code). Consolidated the whole
  go-forward plan into THIS file (single source of truth); demoted VIDEO_BUILD_HANDOFF.md + 3D plan
  section 0 to pointers; re-pointed the tracker. Consolidated the handoff skills into ONE installed
  `otr-handoff` skill (old `otr-build-handoff` + `otr-video-handoff` deleted). Launched the decoupled
  3D quick-smoke (see section 1 IN FLIGHT). Pushed docs @ `ef49e09`. Handed off to a fresh window.
- **2026-06-12 (earlier):** coverage sweep launched live on a fresh :8000 boot, then DEFERRED
  per operator -> GPU freed for coding. Built the one-click overnight path: `otr_overnight_sweep_launch.ps1`
  + the `otr-overnight-sweep` (manual) + `otr-sweep-monitor` (30-min, marker-guarded) tasks. Synced
  repo to `847e2de`. Verified + doc-corrected the Track-3 section-3 image-routing must-fixes (LANDED,
  68/1). NOTE: ComfyUI Desktop install moved to `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py`
  (Desktop v2 standalone); venv = `C:\Users\jeffr\Documents\ComfyUI\.venv` (py3.12.11, torch
  2.10.0+cu130); custom_nodes/ComfyUI-OldTimeRadio is a JUNCTION to the Documents repo.
- **2026-06-11:** CS-4 resolved (1.7B default); 0-E Phase A complete; coverage-sweep triage; LK-1
  problem statement; OH consolidation tickets.
- **Earlier:** GATE B S0-S2 complete; Track 3 (GATE B) CLOSED; A-ship soak GREEN x2; B-ship DONE via
  HuMo-2D rescope (`character_3d` 3D-mesh path DEFERRED to a future opt-in engine).

---

## 7. POINTERS (evidence + tooling -- not plans)

- Tracker dashboard (visual gauge + lanes): `otr-build-tracker` artifact (OneDrive\Documents\Claude\Artifacts).
- Overnight sweep: `scripts/otr_overnight_sweep_launch.ps1`; tasks `otr-overnight-sweep` + `otr-sweep-monitor`;
  digest `scripts/sweep_monitor_digest.md`; GO file `scripts/_otr_0e_gpu_go.txt`.
- 3D spec (detail behind forward-order item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Bug Bible (survival guide repo): `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`
  (`BUG_BIBLE.yaml` + `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale; LTX-AV lane (own plan,
gated); switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes (T/G/W) until S-3D-0 + the
operator green light.
