# OTR GO-FORWARD PLAN -- THE SINGLE SOURCE OF TRUTH

> **This file is canonical.** The forward order, runway, open tickets, current step, hard
> rules, and sprint lanes all live HERE (one-doc rule, operator-directed 2026-06-12).
> `docs/VIDEO_BUILD_HANDOFF.md` and `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` section 0
> are now THIN POINTERS to this file. The `otr-build-tracker` artifact is the visual
> DASHBOARD (gauge + lanes) and mirrors this doc -- it is not the source of truth.
> Dated `docs/<date>-*` folders are EVIDENCE records (roundtables, problem statements), not
> plans. When this doc and any other disagree, THIS doc wins.
>
> **Last updated:** 2026-06-12 (planner session, cont.). **Branch:** `v2.0-alpha`. **HEAD:** `ef49e09`
> (HEAD==origin). Update the "Last updated / HEAD" line and the relevant section on every tick.

---

## 1. CURRENT STEP

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
while running). NEXT WINDOW: read that digest for pass/fail per engine; fix the hardest ones that
fail and re-run that engine's leg (`python scripts/otr_coverage_sweep.py --only
other_beats_visual_<engine>`).

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
