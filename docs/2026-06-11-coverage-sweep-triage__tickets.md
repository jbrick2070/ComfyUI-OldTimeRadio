# Dropdown Coverage Sweep — Triage Record (section-0 item 4)

**Date:** 2026-06-11 (sweep launched ~04:00, detached). **Status: INTERIM — run 2 in flight.**
Runner: `scripts/otr_coverage_sweep.py` (S2 applier-powered). Summary:
`scripts/coverage_sweep_summary.json`; per-leg: `scripts/_otr_soak_capstone_results/sweep_*.json`.
Triage policy: failures become TICKETS, not immediate fixes (operator directive).

## Run history (the sweep is now THREE runs; verdicts must cite their run)

- **Run 0 (overnight, ~04:00–08:29):** OLD pre-restart server code (Desktop-era process,
  stale-server shim active). Legs 1–6 PASS; died with the server mid-leg-7 (~08:29).
  Summary preserved: `coverage_sweep_summary_part1.json`.
- **Run 1 (~09:10–11:00, VOID — environment-poisoned):** fresh headless server, tonight's
  code, but launched with the OTR_ENABLE_WAN_I2V superset env. Wan TE (6.4 GB) + WanVAE
  staged ALONGSIDE HuMo-14B (16.5 GB) -> ~23 GB staged on the 16 GB card -> dynamic-offload
  THRASH (sampler steps 220s -> 1,788s/it; proven-good r5d baseline on the same install =
  7.2s/it). Leg 7 TIMEOUT@42min + music ltx/visualizer TIMEOUTs are ARTIFACTS of the env,
  not engine verdicts — all run-1 verdicts VOID (see CS-3). Killed ~11:00.
- **Run 2 (clean, in flight):** fresh headless server, tonight's code, proven env
  (HUMO=1 + LTX=1, NO Wan). 9 legs queued: still_kenburns, music ltx_video / visualizer /
  abstract, other_beats flux_still / humo / humo_1.7B / latentsync, announcer latentsync
  RE-RUN (CS-1). Log: `coverage_sweep_resume2.log`; per-batch summaries
  `coverage_sweep_summary_r2_*.json`. The 2 wan_i2v legs are DEFERRED to a final
  supervised batch (CS-3 coexistence question).

## Leg matrix (16 runnable + 4 skipped-disabled)

| # | Leg (slot / engine) | Verdict | Elapsed | Target engine actually rendered? |
|---|---------------------|---------|---------|----------------------------------|
| 1 | announcer / ltx_video | PASS | 1875s | YES (ltx_video:3) |
| 2 | announcer / station_card | PASS | 1928s | YES (station_card:2) |
| 3 | announcer / flux_still | PASS | 2444s | YES (flux_still:2) |
| 4 | announcer / humo | PASS | 2853s | YES (humo:5) |
| 5 | announcer / humo_1.7B | PASS | 2856s | YES (humo_1.7B:2) |
| 6 | announcer / latentsync | PASS | 2351s | **NO — fell back, see TICKET CS-1** |
| 7 | announcer / still_kenburns | running (started 08:00) | — | — |
| 8-11 | music / ltx_video, visualizer, abstract, wan_i2v | queued | — | — |
| 12-16 | other_beats / flux_still, humo, humo_1.7B, latentsync, wan_i2v | queued | — | — |
| — | announcer+other_beats / hunyuan3d_talk, trellis_talk (4 legs) | SKIPPED_DISABLED | — | missing_toolchain (cu128), expected |

All 6 completed legs: playable final mp4 in episodes/ + obs/, 6 beats,
**audio byte-identical** (pcm_sha256 == master_pcm_sha256 on every leg),
render-phase driver VRAM 3.1–3.5 GB.

## TICKETS (no fixes in this window)

### CS-1 — latentsync legs prove the fallback, NOT latentsync (coverage gap)
- **Evidence:** leg 6 trace: `shot_b001` and `shot_b005` attempts
  `["latentsync","still_kenburns"]` → final_engine `still_kenburns` on BOTH.
  The leg's engine histogram contains zero latentsync clips. Verdict is PASS
  because the sweep gates on render/audio/hygiene with `expect_engine=""`
  (by design, runner line ~11) — so PASS ≠ engine coverage here.
- **Probable cause:** the live server is STALE (pre-restart code) — the
  GATE A latentsync-100% fix (`OTR_LSYNC_BASE_ENGINE` base-clip synthesis,
  commit 4e164d9) is on disk but not loaded, so latentsync still runs the old
  face-lottery and falls LOUD to floor (fallback chain behaved correctly).
- **Action:** after the ComfyUI RESTART, re-run BOTH latentsync legs
  (announcer + other_beats) and require latentsync in the trace/histogram.
  Leg 16 (other_beats/latentsync) in tonight's run will likely show the same
  pattern — treat its PASS the same way.
- **Optional v2 hardening:** add an `expect_engine` assert mode to the runner
  so a coverage leg FAILS LOUD if the target engine never lands.

### CS-2 — machine NVML peak ~15.9–16.1 GB on every leg vs the 14.5 GB ceiling
- **Evidence:** `whole_run_nvml_machine_mb` 15,956–16,104 on all 6 legs
  (16 GB card near-saturated); peak holds flat for long stretches per
  `coverage_sweep.log`. Render-phase DRIVER attribution is only 3.1–3.5 GB,
  so the pin happens OUTSIDE the video-driver phase (suspects: in-graph
  FLUX portrait gen / LTX open / TTS phase), and machine-wide NVML includes
  non-OTR processes.
- **Question for triage:** is this the known/accepted in-graph peak (the
  2026-06-03 full-workflow run also peaked 15.4 GB > ceiling), or a
  ceiling-invariant violation needing per-phase attribution? Needs a
  phase-tagged NVML sample pass or ledger phase timestamps cross-checked
  against the sample log. No crash/OOM on any leg.
- **Action:** operator decision after restart + fresh render; if the ceiling
  invariant is meant to bind the WHOLE pipeline (not just the resident heavy
  engine), the in-graph image/video nodes need the same reclaim treatment.

### CS-4 — NEW-CODE HuMo THRASH (CRITICAL PATH: blocks the sweep AND operator look-QA)
- **Evidence:** every server running POST-overnight code thrashes HuMo sampling:
  run 1 (superset env) 220->1,788 s/it; run 2 (CLEAN env, idle machine: CPU 2%,
  20.7 GB RAM free) 153->452+ s/it escalating, leg TIMEOUT at 90 min. Baselines:
  run 0 (OLD-code process, same install, same launch recipe) ran humo-character legs
  31-48 min all night at the 14-18 s/it class; r5d acceptance ditto. WanTE/WanVAE/
  WAN21_HuMo staging lines are NORMAL HuMo components (HuMo is Wan-based) — run 1's
  CS-3 "superset env" attribution was WRONG as the root cause (the wan_i2v coexistence
  QUESTION below stays open).
- **Mechanism (hypothesis, evidence-bounded):** comfy dynamic-offload weight paging per
  step = something extra stays RESIDENT when HuMo samples on the new code. The escalating
  s/it + machine NVML pinned ~16.2 GB fit per-step paging with fragmentation. NOT the
  bounded teardown wait (3x2s) and NOT assert_vram_within_ceiling (raises, never loops).
- **Suspect commits (bisect set, newest first):** 1ef6786 (63->87 gate_in wire EDITED THE
  MASTER JSON — execution-order/topology change: could keep the image lane resident into
  video beats), da49ada (ST-3 stills dispatcher slots + materialization), 5b73001 (S1
  dynamic ceiling env-at-dispatch), 0a3af91 (S2 applier), Track-3 a0f1441/80ce175/1571e0f,
  0-E a05bda/3b535c7/1daaa6a (cold-import-only darks — unlikely).
- **Repro:** the thrash-era headless server is UP on :8000 (queue cleared) for direct A/B;
  30-word smoke (scripts/queue_smoke.py) with a humo character beat reproduces in one leg.
- **Blocks:** sweep run 2 remainder (every leg has humo character beats), wan batch,
  0-E Phase B, the operator fresh-render look-QA (Desktop relaunch loads the SAME code).
- **Action:** coder-window regression hunt (kickoff issued 2026-06-11); sweep resumes
  after the fix lands.
- **OPERATOR DIRECTIVE (2026-06-11 late, supersedes the blocking framing):** "the 6/5
  HuMo was the working HuMo; the bigger HuMo doesn't work -- chasing it is the wrong
  tree." Record-squared: the 6/5-era default WAS humo_1.7B (BUG-265 Option C, 2026-05-24;
  ~3.9 GB, huge margin); the 14B verified ONCE on 06-09 at 13.8 GB vs the 14.5 ceiling =
  0.7 GB margin, and CS-4 is that margin dying under the new code's residency delta.
  THEREFORE: (a) **default flips back to humo_1.7B** (14B returns to opt-in via the
  existing tier loader + auto-downgrade chain -- restores the BUG-265 Option C policy);
  (b) the CS-4 hunt's FIRST test = the same smoke on humo_1.7B on new code -- if 1.7B is
  full-speed, the regression is a small-residency delta that only kills the 14B margin,
  CS-4 DOWNGRADES from everything-blocker to an open ticket, and the sweep/wan/0-E
  Phase B/look-QA queue resumes on the 1.7B default immediately; (c) the sweep's
  humo(14B) dropdown leg gets marked OPERATOR-DEPRIORITIZED (option stays selectable +
  opt-in, never default) unless the root cause turns out to be a one-line fix.

#### CS-4 RESOLUTION (coder window, 2026-06-11 evening) — evidence + ship record
- **No code regression found in the bisect window.** Idle-box repro at HEAD `ca3faee`
  (same install ComfyUI 0.24.1 / torch 2.10.0+cu130 / NORMAL_VRAM, same launch recipe,
  fresh server, FIRST humo block, `--verbose DEBUG`): humo-14B sampled
  **46.2 / 62.8 / 62.4 / 88.5 / 119.2 s/it** — the r5d sane class (52.4–57.7 flat),
  nowhere near the thrash class (153->1,788). The window's whole nodes/+workflows diff
  (6a1b716~1..230fe4e) is CPU-side: prompt gear-scrub, env-gated LTX-I2V (default OFF),
  dynamic-ceiling assert/settle-wait consumers, validator stamp/env-export, the applier.
  The 63->87 gate wire is one STRING link + 3 padded widgets — no residency mechanism.
  Load stories byte-similar across eras (`WAN21_HuMo 16531MB Staged. 1053 patches`,
  same staging order, `0 models unloaded` both). Per-process VRAM mid-thrash (leftover
  render): the comfy python ALONE held 15.5 GB; external processes ~0.7 GB total.
- **Mechanism (NAMED via the operator's BUG-291/265 history pointer + DEBUG VBAR dumps):**
  at WAN21_HuMo-14B sampling start, the aimdo VBAR ledger shows **WanTEModel (umt5-xxl)
  still 5,248 MB Actual-Resident** (+WanVAE 192; FLUX/LTX/Mochi/Whisper VBARs all freed).
  That is the eng_humo "fully resident, no free_after_use" design (correct for the 1.7B
  stack, which fits whole: ~3.3+5.2+0.2 GB < ceiling) applied to a 16,531 MB model on a
  16,303 MB card: the 14B samples against a ~10 GB budget and dynamic-pages ~6 GB EVERY
  step (comfy 0.24.1 `ModelMMAP` + `budget_deficit`), escalating with fragmentation
  (46->119 s/it idle; deficit 96->172). Box contention (runs 1-2 overlapped the 0-E
  agent's 4.93 GB download + sha256 + repeated full-suite runs + Blender) turns that
  paging into disk-class reads -> 153->1,788 s/it. Run 0 + r5d ran on a quiet box —
  and r5d's "sane" 52-57 s/it was ALREADY the TE-starved class: the documented healthy
  12-14 s/it (BUG-LOCAL-265/291 verify numbers) was always the **1.7B's** class. The
  14B was never in it; its 06-09 "13.8/14.5" verification measured peak FIT, not
  sustained step health.
- **Operator CHECK-FIRST answers (BUG-291 re-open audit):** (a) the legacy
  `OTR_BatchFluxPortraitRender` EXIT-eviction fix site was REMOVED with the legacy
  render-path teardown (CW-1 era) — no "[OTR_BatchFluxPortraitRender] EXIT eviction" /
  "PHASE-C-VRAM-PROBE" lines exist in ANY current-era log (r5d included; greps clean);
  (b) the executor-cached-FLUX class is NOT the live mechanism — FLUX's VBAR reads 0 MB
  resident at HuMo entry in the idle DEBUG repro (the new image lane + dynamic loader do
  evict it); the 63->87 wire / ST-3 dispatcher / 87->88 link did NOT re-open
  FLUX-into-HuMo; (c) the BUG-291-CLASS item that IS live = the umt5 TE pinned through
  14B sampling (MRU `vbar_prioritize` right before sampling keeps it above HuMo in
  eviction priority). Candidate 14B-lane fix for CS-4-open: a TARGETED post-encode TE
  detach at the encode->sample seam (BUG-291 detach ladder), guarded against the
  BUG-265 "inter-node eviction fragmented the allocator into OOM" note — NOT shipped
  now (14B deprioritized).
- **FIRST TEST (operator amendment) PASSED:** the same 30w character smoke on
  **humo_1.7B at current HEAD: 9.9 s/it FLAT** (20-step block, zero escalation, zero
  paging — `WAN21_HuMo 3320MB Staged. 0 patches`), better than the 12-14 documented
  class, same session/box as the 14B's 46->119. CS-4 DOWNGRADES to CS-4-open per the
  amendment; the deep bisect is skipped.
- **Shipped (this session):** the operator policy flip — default character tier =
  **humo_1.7B** in `config/profiles/16gb_full.json` (role + slot overrides),
  `workflows/otr_scifi_16gb_full.json` (node-87 `other_beats_video_model`, node-92
  `engine`), `nodes/otr_video_render_batch.py` defaults, profile pin test updated.
  14B stays registered/selectable/opt-in; fallback chain unchanged. Launcher gained
  an optional `%3 DEBUG` arg (`scripts/_otr_soak_server_launch.cmd`).
- **CS-4-open (lazy, nothing queues behind it):** one unhurried bisect later for the
  small residency delta that ate the 14B margin on post-overnight code — the same delta
  class will eventually matter for wan/ltx peaks. 14B sweep legs: OPERATOR-DEPRIORITIZED.
- **Ops guardrail (recorded):** no heavy parallel agent work (suites, multi-GB
  downloads/hashing, Blender) while a timed GPU leg renders — it poisons s/it and can
  thrash margin-edge models.
- **Bookkeeping:** the runner rewrites `scripts/coverage_sweep_summary.json`
  unconditionally; a no-server CLI mistake this session clobbered the (already-VOID)
  run-2 interim copy — real verdicts live in `coverage_sweep_summary_part1.json` /
  `_r2_*.json`.

### CS-3 — wan_i2v + humo coexistence on 16 GB (RESCOPED by CS-4)
**Original "Wan global-enable poisons every leg" attribution RETRACTED** — the staged
WanTE/WanVAE belong to HuMo itself, and run 2 thrashed without the superset env (see
CS-4). What REMAINS open: a music_visual=wan_i2v episode needs Wan-i2v AND HuMo in one
render; whether both fit the 16 GB tier sequentially (free-after-use) is unproven until
the wan legs run post-CS-4-fix. Supervised wan batch stays deferred.

### CS-3-old — original text (superseded, kept for the record)
- **Evidence (run 1):** with `OTR_ENABLE_WAN_I2V=1` set server-wide, WanTEModel (6,419 MB)
  + WanVAE (241 MB) staged alongside WAN21_HuMo (16,531 MB) on the 16 GB card; sampler
  thrashed 220 -> 1,788 s/it (r5d same-install baseline: 7.2 s/it). The launch lanes were
  EXCLUSIVE for this reason; the resume launch's superset env caused it (operator-side
  miss, not an engine defect).
- **The real question for the wan_i2v DROPDOWN options:** a music_visual=wan_i2v episode
  needs Wan (music beats) AND HuMo (character beats) in ONE render. If enabling both
  always co-stages them, the wan dropdown options are 16 GB-incompatible as wired —
  that is exactly what the sweep exists to surface. NEEDS: a supervised 2-leg wan batch
  watching staging behavior; if thrash recurs -> ticket the fix (defer/lease Wan TE
  staging until dispatch, or pin wan-leg characters to humo_1.7B, or mark wan
  enable-set-incompatible with humo on the 16gb tier).
- **Status:** wan legs deferred from run 2; supervised batch after the 9 clean legs.

## Non-ticket observations
- Fallback LOUDness verified: the latentsync fallback is visible in the
  per-shot `attempts` trace — the AS-2/CW-7 LOUD-fallback contract held.
- Leg cadence ~31–48 min; 16 legs ≈ 10–12 h total wall clock.
- Reddish stills / repeated script in these renders = EXPECTED (stale server
  + sweep cache reuse by design; operator look-QA note ~05:45). The
  acceptance test stays: ONE fresh full-json render on saved defaults after
  the restart.

## Gate status after triage (section 0)
Item 4 acceptance = all 16 legs render a full playable episode without
crashing, credits + subtitles present. INTERIM: 6/16 PASS on soak gates;
CS-1 means latentsync coverage is NOT yet proven and re-runs post-restart
are required before item 4 can be called DONE.
