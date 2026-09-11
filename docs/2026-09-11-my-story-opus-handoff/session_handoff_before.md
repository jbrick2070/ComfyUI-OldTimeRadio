# SESSION HANDOFF -- 2026-07-02 night (talking-radio S4x + cloud-S3 core window)

Fresh window: read CLAUDE.md + docs/GO_FORWARD_PLAN.md first. Branch v2.0-alpha,
HEAD == origin @ 13de2a60 (verify). Suite 6059/0 + Bug Bible 16/0 at last push.

## What SHIPPED this session (all pushed)
- f9eed360 cloud-S0 c4: invoke_partner_node bridge + watchdog + gated smokes
  (scripts/otr_cloud_s0_smoke.py; operator env still unset -> smokes exit 3).
- 820f6df3 S4: ia2v character beats init on the cast PORTRAIT (A/B: scene 0.57
  vs portrait 2.86 lag 0). PROOF7_VERDICT.md in docs/2026-07-02-canonical-ia2v.
- a415ad18 S4b+S4c: talking portraits mint FACE-FORWARD+warm (era/grade tails
  skipped; `talking` map VideoDirector->ImageDirector->MetaBrief) + radio-face
  A/B RETIRED into default-on under the ia2v register (fail-LOUD missing face).
- a9440980 cloud-S3 CORE (operator evening GO): eng_cloud_video.py 4 dark rows
  + REAL canonicalize_video (audio strip + post-strip proof). wan sends the
  exact pinned static set (OTR_CLOUD_WAN_MODEL env); seedance = honest dark row
  until the S1 V3-expansion pin; kling pair fully pinned.
- 9eb7e29d/d8f34835 scripts/otr_ia2v_server_boot.cmd (the bare soak launcher
  has NO engine env; FLAT unet name -- ltx2\ prefix breaks the resolve).
- Story-writer director-note leak fix REVERTED out of production and PARKED in
  UpstreamStoryLab docs/GO_FORWARD_PLAN.md "DEFERRED STORY-LLM FIXES" (7df7c80
  in THAT repo). Operator hold: NO story-LLM changes before the transplant.

## IN FLIGHT right now
- **proof9b** (the S4b/c verdict episode): launched ~18:12 via the env'd cmd
  (OTR_ENABLE_LTX_AV=1, dev unet flat name, ZIMAGE trio), driver log
  %TEMP%\proof9_driver.log, server log C:\Users\jeffr\Documents\ComfyUI\
  comfyui_8000.log. SCORE on land: mux each slice wav onto its raw clip
  (clips are SILENT by design) then scripts/otr_talking_radio_probe_eval.py
  (same-file-twice trick); slice map from the "sliced ... (beat bNNN)" server
  log lines; clips at output\otr\episodes\pending_*\clips\. Bar: speech >=2.0
  (music exempt). Expect: face-forward portraits + radio-face bookends
  (S4c fired on b000, log-proven).
- SOAKS ALREADY QUEUED (2026-07-02 ~18:28): proof9b breached the ceiling
  (14659 > 14500; desktop-session baseline crept to ~2.9GB), so the server was
  rebooted with OTR_LTX_AV_RENDER_CANVAS=768x416 (server-env, THIS boot only)
  and proof9c relaunched at that canvas; a 120w soak (%TEMP%\soak1_120w.log)
  and a 30w soak (%TEMP%\soak2_30w.log) are chained behind it in one detached
  cmd. Morning window: score proof9c + QA both soaks (metric is roughly
  scale-invariant; note the 768x416 canvas in any comparison), and consider a
  permanent baseline-aware canvas step-down if the desktop keeps squatting.

## Ops gotchas burned tonight (do not relose)
- ComfyUI main.py RE-EXECS itself under a uv-python CHILD: killing the venv
  parent leaves the child serving :8000 with STALE ENV. Kill by PORT OWNER
  (Get-NetTCPConnection -LocalPort 8000 -> OwningProcess) + CIM sweep.
- Desktop backend relaunch squats ~4GB VRAM -> 832x448 full-pipeline breaches
  the 14.5GB ceiling. Baseline must be <=~2.5GB before a proof/soak.
- The full pytest suite POLLUTES repo-root otr_runtime.log while a render is
  in flight (same box/file) -- read timestamps before declaring a stall.
- Set-Content -Encoding UTF8 writes BOM (PS5.1). Use [IO.File]::WriteAllText.

## NEXT (priority order)
1. proof9b verdict (+ side-by-side for the operator; b005 announcer dip watch).
2. Soaks x2 unattended; morning QA vs the new bars.
3. S5 (task #16, operator ratified): port the two-stage HQ recipe (upsample +
   refine, guide chain; NO audio latent) to eng_ltx_video; dev-unet family
   auto recipe like eng_ltx_av._detect_recipe; 2 LTX rows, NO ltx_lowvram.
   Measured silent-vs-audio-in VRAM/time A/B on first live clip.
4. Cloud: operator env for live smokes (leg1/leg2) -> S1 stills lane (V3
   expansion pin unblocks seedance + wan prompt) -> S3 FULL (reactive
   auto-defaults + ShotLock stamps + fallback chains).
5. Parked: director-note scrub (transplant repo), OTR_LTX_RADIO_FACE env now
   only meaningful on single-pass recipes.
