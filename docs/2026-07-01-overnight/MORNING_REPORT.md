# OTR Overnight Report -- 2026-07-01

Operator was asleep at start, then woke mid-session and requested two custom episodes
(see "Operator-requested episodes"). Autonomous window otherwise ran Phases A/B/C.

Branch `v2.0-alpha`. All commits pushed; HEAD == origin at each push.

---

## 1. SHIPPED (commits, all pushed to v2.0-alpha)

| Commit | What |
|---|---|
| `dfacea49` | **E4 dropdown labels** -- registry-derived `" (audio-reactive, no scene image)"` descriptor for the abstract visualizers (viz_green / viz_mxc_cpu / viz_mxc_mandala). Auto-derived from `family=="abstract" && accepts_still is False` (no hand-maintained map, no drift). Round-trip-safe (`_engine_id_from_pick` strips it). No workflow-JSON change (node-87 stores bare ids). +2 tests. |
| `4caeaa90` | docs(GO_FORWARD): E4 status + HEAD pointer refresh. |
| `ddbbe8c2` | **fix(S-F smoke)**: corrected `otr_state_dir` import in `scripts/otr_visual_smoke.py` (`nodes._otr_paths`, not the non-existent `nodes._otr_video_engines._otr_paths`) so the default `bake` path works without `--input`. Root-cause bug in the shipped S-F CLI. |

**Green after every change:** full suite **5906 passed / 35 skipped / 0 failed** (~87s); Bug Bible **16 passed / 7 skipped / 3 xfailed**; B7 forbidden-sweep **5/5**. Pre-change baseline was 5904/0 (my +2 tests account for the delta).

---

## 2. PHASE A -- QA-triage of the shipped BUILD-READY QUEUE (CPU/deterministic) -- ALL VERIFIED GREEN

- **Targeted suites (200 tests):** test_video_viz_mandala (13), test_video_still_parallax (15), test_clip_fill (27), test_video_mesh_stage (41), test_3d_image_streams (14), test_video_role_eligibility_matrix (75), test_slot_matrix_soak (9) -- **200 passed**.
- **viz_green rename + engine retirement (registry probe):** 15 engines registered; **none** of station_card / abstract / still_parallax / visualizer still registered; viz_green + viz_mxc_mandala present. Radio-is-host proxy correct (announcer_visual/music_visual True, character_video False).
- **HuMo radio-is-host routing:** test_video_render_driver_perbeat_audio + test_video_humo + test_speaker_role -- **101 passed** (TestRadioIsHostGuard: HuMo on announcer/music redirects to ltx_audio_in; character_video untouched).
- **clip-fill face-exclusion:** test_clip_fill 27/27 (HuMo/audio_driven_face rows excluded from loop-fill).
- **mesh radio-fodder + adaptive-camera math:** covered in test_video_mesh_stage (41) + test_3d_image_streams (14).
- **Mandala CPU render eyeball (real artifact):** rendered viz_mxc_mandala at production canvas 1472x832 x50 frames against a real master WAV (tests/fixtures/baseline_v1.5.wav) -> valid h264/yuv420p, nb_frames=50, mid-frame non-black ratio 0.19 (not a black floor), reactive (audio_used True). File: `docs/2026-07-01-overnight/mandala_preview_1472x832.mp4`.

GPU-only Phase A checks (mandala in-slot, HuMo bookend live, mesh headroom proof-frame) were folded into the live episode renders below.

---

## 3. PHASE B -- E4 dropdown-label polish

Shipped the safe, registry-derived half (the "audio-reactive, no scene image" descriptor). See commit `dfacea49`.

**DEFERRED (morning-operator call, NOT cosmetic):** spelling out WHICH LTX/Wan/image model+recipe each label carries. There is no clean registry field for it (recipe/quant resolve at render time via env like `OTR_LTX_AV_RECIPE`); doing it needs a design decision -- add a per-engine registry descriptor vs. a hand-maintained map that would violate the no-drift contract. E1/E2 (no-fallback scaffolding) and C1 (audio_motion_profile) left untouched per instruction (load-bearing / bigger-tail; morning-operator calls).

---

## 4. OPERATOR-REQUESTED EPISODES (Jeffrey, awake mid-session)

Mechanism used: `OTR_FORCE_ENGINE_MAP=*=<engine>` set at server boot (FLOOR lane) forces EVERY beat to one engine -- the documented "all-X episode" knob. Verified live (LOUD per-beat override + engine-specific render lines in the log).

### 4a. 800-word ALL-MANDALA (viz_mxc_mandala) -- DELIVERED
- obs final: `output/otr/obs/signal_lost_stellar_tension_20260701_011217_silent_procgen_blended_final.mp4` (119.9 MB, 145.8s, h264+aac, playable).
- Copy for eyeball: `docs/2026-07-01-overnight/EPISODE_800w_all_mandala_stellar_tension.mp4`.
- Every visual beat rendered as viz_mxc_mandala (confirmed: 13+ `viz_mxc_mandala 1472x832 xNNN frames (audio=True)` clips). Cast MIRA CROSS + ROBINSON MALONE; indextts2 + kokoro voices.
- **NOTE (writer, not engine):** the first two 800-word attempts hard-failed at OTR_CastLock with `freeze_verdict='needs_full_rerun'` (BUG-LOCAL-276 -- the story quality gate rejects a structurally-incomplete long ledger). The third (num_characters=2, act_count=auto) passed `frozen_with_warns` but the story CONVERGED at ~260 spoken words, so the episode is ~2.5 min, not a full 8 min. The freeze gate is flaky at long target lengths -- a genuine writer-convergence limit, flagged for operator follow-up.

### 4b. 420-word ALL-RAINBOW non-mandala (viz_mxc_cpu) -- DELIVERED
- Interpretation: "rainbow non mandala" = viz_mxc_cpu (the numpy/PIL rainbow scope visualizer), distinct from viz_mxc_mandala and from green viz_green.
- obs final: `output/otr/obs/signal_lost_heart_of_the_mainframe_20260701_014200_silent_procgen_blended_final.mp4` (130.9s, h264+aac, playable).
- Copy for eyeball: `docs/2026-07-01-overnight/EPISODE_420w_rainbow_scope_heart_of_the_mainframe.mp4`.
- All 18 beats rendered as viz_mxc_cpu (confirmed: 19 `viz_mxc_cpu 1472x832 xNNN frames (audio=True)` clips; SilentComposite assembled 18 beats -> 3273 frames, 0 held_last_frame; captions LINT-clean). Passed the freeze gate first try (`frozen_with_doctor_edits`) -- 420w is more reliable than 800w. Story converged ~200 words (~2:11). Cast a Lead AI Researcher + a second voice; a fitting meta story ("heart of the mainframe", AI's aesthetic limitations).
- Prompt executed in 25:22 total.

---

## 5. NOTES / FINDINGS FOR OPERATOR

- **S-F fixture stale-capture:** the existing `node_episode_input.json` capture (6/30 19:36) had its referenced assets already cleaned off disk (master WAV + per-beat assets gone), so a bake from it would fail the LOUD preflight. The S-F-accelerated all-engines soak therefore needs a FRESH clean reference episode rendered first. The two episodes above are fresh full renders (not S-F replays).
- **Full-matrix S-F soak (the original Phase C fallback work) was SUPERSEDED** by the operator's two episode requests, which took priority once Jeffrey woke. The S-F import fix (`ddbbe8c2`) unblocks the default bake path for a future soak.
- **Wasteful pre-render:** both forced-engine episodes still minted Flux character stills during image-gen (the override applies at video-render time, after image-gen runs on the default plan). Harmless but slow; a future optimization could short-circuit still-gen when a force-map maps a role to an accepts_still=False engine.
- **Box hygiene:** every headless reset used a SELECTIVE CIM kill (main.py + port 8000) -- never a blanket python kill; MCP pythons untouched throughout. GPU returned to ~1.1-1.8 GB baseline between boots.

---

## 6. RESUME KICKOFF (paste as message #1 of the next window)

```
Run the otr-handoff skill to resume. HEAD should be ddbbe8c2 (+ any docs commits) == origin/v2.0-alpha. Since 2026-07-01 overnight: E4 audio-reactive descriptor SHIPPED (dfacea49); S-F smoke import fixed (ddbbe8c2); Phase A shipped-work all verified green (suite 5906/0 + BugBible 16/7/3 + B7 5/5). Two operator-requested episodes delivered (800w all-mandala + 420w all-viz_mxc_cpu) via OTR_FORCE_ENGINE_MAP=*=<engine>. OPEN: (1) 800-word episodes trip the writer freeze gate (needs_full_rerun, BUG-LOCAL-276) and converge short (~260w) even when they pass -- writer-convergence at long lengths needs investigation; (2) E4 "which model" label spell-out DEFERRED (needs a registry data source, no hand-maintained map); (3) E1/E2 no-fallback scaffolding + C1 audio_motion_profile still NOT started (load-bearing/bigger-tail); (4) the full S-F all-engines x all-slots soak still un-run (needs a fresh baked bundle). Rules unchanged: forward order = GO_FORWARD section 3; audio spine FROZEN; single resident heavy <=14.5GB; determinism; LOUD fallbacks; UTF-8 no BOM; SFW; commit+push per green chunk; prod/main GATED.
```
