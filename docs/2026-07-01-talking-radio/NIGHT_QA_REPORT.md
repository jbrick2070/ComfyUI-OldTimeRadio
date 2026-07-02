# Talking-Radio Night -- QA REPORT (2026-07-02 ~06:15)

Scope: talking-radio contract build (Sub-plans B + C) + the operator overnight
batch, run unattended 00:30-06:15. Contract:
`kibitz-runs/2026-07-01-talking-radio/r1/final.md`. Companion docs: `EYEBALL.md`
(probe verdict, operator gate), `MORNING_REPORT.md` (watcher snapshot),
`night_results.jsonl` (machine-readable legs).

---

## 1. Executive summary

- **10/10 legs SUCCESS so far** (2 probe, 2 humo, 6x 120w all-ltx face1);
  the 2x 120w face0 comparison legs were still rendering at report time.
  Zero failed legs; 2 freeze-gate flakes auto-retried to success.
- **Sub-plan B shipped and visually verified.** The corrected appliance-mouth
  still is ON-DESIGN: glass tuning-dial eyes, huge rubbery cartoon lips,
  woven-grille face, bakelite cabinet, **no human** (`batch_face1_mouth_still.png`).
- **Sub-plan C verdict: NO-GO on lip-sync** per the pre-registered criterion,
  now corroborated on 3 additional 120w corrected-mouth legs. The mouth
  ARTICULATES (opens/closes, camera push-ins -- cinematic) but is NOT locked
  to the audio transients. Operator eyeball still gates (see EYEBALL.md
  Verdict section for your decision options).
- **Two production bugs found live and fixed at root, same night:**
  BUG-LOCAL-415 (orphaned env hook poisoned every headless boot; launcher now
  consume-once; promoted to Bug Bible 12.47) and the ltx_radio_mouth
  human-face leak (image pipeline has no negative channel; positive prompt
  material-anchored).

## 2. Leg results (night_results.jsonl)

| # | leg | words | engine | status | att | obs final (otr\\obs) | size |
|---|-----|-------|--------|--------|-----|----------------------|------|
| 1 | probeA_face0 | 30 | ltx_audio_in | success | 1 | ..._recorded_mysteries_005445... | 46.6 MB |
| 2 | probeB_face1 | 30 | ltx_audio_in | success | 2 | ..._jazz_code_cracker_012739... | 37.4 MB |
| 3 | humo169_50w | 50 | humo_1.7B_169 | success | 1 | ..._times_ticking_out_015257... | 27.0 MB |
| 4 | humo169_100w | 100 | humo_1.7B_169 | success | 1 | ..._jazz_starfish_025542... | 19.7 MB |
| 5 | batch_face1_01 | 120 | ltx_audio_in | success | 1 | ..._lanterns_secret_034244... | 41.2 MB |
| 6 | batch_face1_02 | 120 | ltx_audio_in | success | 1 | ..._neon_truth_040827... | 34.8 MB |
| 7 | batch_face1_03 | 120 | ltx_audio_in | success | 1 | ..._clutching_secrets_043359... | 63.6 MB |
| 8 | batch_face1_04 | 120 | ltx_audio_in | success | 2 | ..._flickering_screens_050310... | 41.8 MB |
| 9 | batch_face1_05 | 120 | ltx_audio_in | success | 1 | ..._shadows_of_the_stage_052756... | 42.4 MB |
| 10 | batch_face1_06 | 120 | ltx_audio_in | success | 1 | ..._broken_circuit_055350... | 28.5 MB |
| 11 | batch_face0_01 | 120 | ltx_audio_in | RUNNING at report time | - | - | - |
| 12 | batch_face0_02 | 120 | ltx_audio_in | queued | - | - | - |

Engine histograms: every completed ltx leg = 100% `ltx_audio_in` (6/6 clips);
both humo legs = 100% `humo_1.7B_169` (6/6). Every obs final Test-Path-verified
by the driver before "success".

## 3. Lip-sync criterion -- probe + batch corroboration

Pre-registered bar (EYEBALL.md): r1 >= 0.35 AND r1 - r0 >= 0.15 on the b001
announcer bookend window (mouth-region motion energy vs audio onset envelope).

| leg (b001 window) | still | r |
|-------------------|-------|---|
| probeA_face0 (control) | faceless scene | 0.009 |
| probeB_face1 | pre-fix human face (strongest mouth prior) | 0.047 |
| batch_face1_01 | corrected appliance mouth | -0.030 |
| batch_face1_02 | corrected appliance mouth | 0.095 |
| batch_face1_03 | corrected appliance mouth | -0.024 |

Mean face1 r across 4 legs ~= 0.02 -- indistinguishable from the control.
**Conclusion: ltx_audio_in (distilled-1.1 Q3_K_M, distilled_native, 8-step)
animates the mouth expressively but does NOT couple it to the conditioning
audio.** Frames (`qa_face1_01_t11.png`, `qa_face1_01_t15.png`,
`probeB_t10p5/t12/t15/t19p5.png`) show genuine open/close articulation +
camera push-ins -- the "old dubbed film" look, not sync. Re-probe knob if
wanted: `OTR_LTX_AV_UNET=ltx-2.3-22b-dev-Q3_K_M.gguf` (sharp_lora recipe,
~1.4x step cost) on the same harness.

## 4. Visual QA

- `batch_face1_mouth_still.png` -- corrected mint (hash 9ad17156, episodes 5-10):
  PASS on design intent; no human face; brief-driven radio form present.
  Minor: a small distant human silhouette in the background of the lanterns
  still (scene dressing, not a face; acceptable -- note for a future
  "empty background" tightening if it recurs).
- `probe_face1_mouth_still.png` -- PRE-fix mint (hash 7e5e76c0, probe B only):
  the human-face leak, kept as the defect exhibit.
- 120w bookends in motion: announcer line delivered over a wide-open mouth
  (t=11), push-in with mouth part-closed + teeth (t=15) -- cinematic, SFW,
  on-design.
- humo169_50w open ("Time's Ticking Out"): retro-futurist title open renders
  clean; HuMo radio-host portrait path exercised under OTR_ENABLE_HUMO_HOSTS=1.

## 5. Defects found this run

| id | severity | state | summary |
|----|----------|-------|---------|
| BUG-LOCAL-415 | HIGH (env integrity) | **FIXED at root** (55e35468) + Bug Bible 12.47 (guide 8911c43) | Crash-orphaned `_marathon_extra_env.cmd` silently forced `*=humo` + HUMO_HOSTS onto EVERY headless boot; probe A attempt 1 rendered all-HuMo. Launcher now consume-once (echo LOUD, call, delete). Live-verified: all subsequent boots clean, histograms match config. |
| human-face leak | HIGH (content/SFW-adjacent) | **FIXED at root** (d87f8fc5) | Image dispatcher has NO negative channel -> RADIO_CONSOLE_NEG inert -> "full soft lips" minted a literal screaming human face in the radio. Positive prompt material-anchored; corrected mint verified on 6 batch legs. |
| writer freeze-gate flake | known (BUG-LOCAL-276 family) | mitigated (auto-retry) | 2/12 legs tripped `needs_full_rerun`; both recovered on retry 1. No new action. |
| humo s/it drift | LOW (perf, observation) | OPEN -- logged only | humo_1.7B_169 clips drifted 7->27 s/it across beats in the 50w leg (BUG-414 "no-OOM crawl" family smell); the leg finished at ~59.5 min of a 60 min cap. Recommendation: humo legs get >= 5400s caps; investigate residency between beats if it recurs. |
| watcher stills/ glob miss | TRIVIAL (throwaway tooling) | noted | Watcher globbed `images\`; batch episodes emit radio-face stills under `stills\`. Snap done manually; watcher's other duties unaffected. |

## 6. Gates + git state

- Full suite **5922 passed / 0 failed** (x2 runs during the night), Bug Bible
  16/0, B7 in-suite green, `test_audio_byte_identical` green.
- Pushed to `v2.0-alpha`, HEAD == origin at every step: `d48a9d76` (B),
  `55e35468` (C infra + BUG-415), `d87f8fc5` (material anchor), `3f575ed1`
  (C results). Survival-guide `main`: `8911c43` (Bible 12.47, Three-File
  Contract). Post-push verify each time: no 0-byte, no BOM, AST parse.
- No workflow-JSON change (all flag/env-gated; no node/widget change) --
  per contract.

## 7. Operator decisions outstanding

1. **GO/NO-GO on the talking-radio direction** (EYEBALL.md Verdict): criterion
   says NO-GO on true lip-sync; your eyeball may still take the articulating
   mouth as a creative LOOK for the A/B toggle. Watch: jazz_code_cracker
   (human-face probe, historical), lanterns_secret / neon_truth /
   clutching_secrets etc. (corrected mouth), recorded_mysteries (control).
2. **Sub-plan A** stays unbuilt unless you override (contract-gated on C).
3. Optional **dev-unet re-probe** (one env var) if you want a second data
   point on sync before closing the question.
4. humo lane cap bump (5400s+) if you want more 100w humo episodes.

-- assembled by the night session; batch_face0 legs append to
night_results.jsonl + MORNING_REPORT.md automatically when they land.
