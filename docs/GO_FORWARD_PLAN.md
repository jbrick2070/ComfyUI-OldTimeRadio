# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> **>>> CURRENT STEP -- 2026-06-27 PARALLEL LANES (two coder windows). Last updated 2026-06-27; origin/v2.0-alpha
> HEAD `08540ceb` == local. prod/main + tags GATED.**
>
> **LANE A -- STORY-QUALITY BUILD (NEW; CPU-only, JSON-FREE).** Roundtable R1->R4 COMPLETE + GO. Coder spec:
> `docs/2026-06-27-story-quality/STORY_QUALITY_BUILD_PLAN.md` (build order 3.1 dignity -> 3.2 anchor/one-breath ->
> 3.3 stage-leak -> 3.4 cliche -> 3.6 cost-scrub -> 3.5 coda-counters -> 3.7 register; all pure-python in the
> writer/composer set + scripts/story_quality_scan.py). Touches NO workflow JSON, NO GPU. Step 0 = read-only baseline
> scan. Open operator decision Q7 (reroll budget <=4 default vs <=3). Docs UNCOMMITTED (git ?? docs/2026-06-27-story-quality/).
>
> **LANE B -- LTX-AV QUALITY BAKEOFF winner-wire (EXISTING step; GPU + workflow JSON).** Bakeoff RAN (13 legs/12 OK);
> awaiting operator winner pick from the clips, THEN wire into otr_silent_composite (scaler) + eng_ltx_av (decode
> temporal_size) + render_driver (canvas). Detail block immediately below. OWNS the GPU/5080 + :8000 + the workflow
> JSON otr_scifi_16gb_full.json.
>
> **PARALLEL-WORK RULES (both lanes):** (1) workflow JSON otr_scifi_16gb_full.json = LANE B ONLY; Lane A stays
> JSON-free. (2) ONLY shared python file = nodes/OTR_LedgerScriptWriter.py (Lane A v2-flag thread ~L2636/L4287; Lane B
> engine config) -- land Lane A's tiny v2-thread edit FIRST, or assign the file to one window + rebase. (3) GPU/5080 +
> :8000 = Lane B; Lane A is CPU-only (regression suite + read-only scan). (4) Separate git WORKTREES, both -> v2.0-alpha,
> small frequent pushes + rebase; stagger the full-suite/Bug-Bible runs. (5) EVERY window updates this doc + the
> otr-build-tracker.
>
> **>>> LANE B DETAIL -- 2026-06-27 LTX-AV QUALITY BAKEOFF *RAN* -- AWAITING OPERATOR WINNER PICK. origin/v2.0-alpha
> HEAD `7c3e2f26` == local. The isolated PLAN-v5 bakeoff is BUILT + SWEPT. prod/main + tags GATED.**
> - **BUILT (NO-GPU):** scripts/build_ltx_av_q_bakeoff_workflow.py (distilled_native silent builder @512x288x153 ->
>   scripts/otr_ltx_av_q_bakeoff_distilled_native.json; SaveImage frames, NO LoRA/ModelSamplingLTXV/LTXVScheduler) +
>   scripts/run_ltx_av_q_bakeoff.py (boot-per-leg; FAIL-LOUD resolved-API-prompt manifest; stages 0-3; per-leg gates;
>   silent encode via the PRODUCTION wrapper_bridge.encode_frames_to_silent_mp4, imported by file path). Committed
>   `24fedea7` + `7c3e2f26`; suite green vs the 5 pre-existing `267a53e` workflow-pin fails; Bug Bible 16/7/3.
> - **THE RAIL WORKED:** `--dry-validate` caught a real bug pre-GPU (manifest read `fps` but the live LTXVConditioning
>   names it `frame_rate`) -> fixed (`7c3e2f26`) -> all manifests PASS live. The fail-loud manifest passed on every one
>   of the 13 sweep attempts (the #1 risk -- measuring the legacy SHARP/LoRA graph -- is guarded).
> - **SWEEP RAN headless (13 attempts, 12 OK; boot-per-leg, one GPU=one render). Clips + per-leg JSON/MD + manifests in
>   `otr/episodes/_bakeoff_ltxq/`. Results table: `docs/2026-06-27-ltx-av-qa/` + `ltxq_bakeoff_results.md`:**
>   - (1) **TEMPORAL SEAM / "flash" = FIXED by WHOLE-CLIP DECODE** (VAEDecodeTiled temporal_size 4096 / overlap 8):
>     seam p99 `0.2353` (jump 2.07x local median = the visible flash) -> `0.0` (no seam), at the SAME s/it (5.37 vs 6.04)
>     and peak VRAM (14338 vs 14337 MB) as L0 -- and it MATCHES the shipped sister `eng_ltx_video` decode, so it wires
>     byte-for-byte. Best TILED alternative = 128/32 (seam ratio 0.57, VRAM 14272).
>   - (2) **SOFTNESS = FREE +14.7% sharpness** by swapping the composite scaler bilinear -> lanczos+unsharp
>     (otr_silent_composite `_seg_vf`; Laplacian 27.17 vs 23.69 at the common 1472x832; ZERO GPU cost).
>   - (3) **CANVAS bump = optional:** 640x384 fits (14476 MB, 8.98 s/it, but freezes=5 appeared); 704x384 OVER the
>     14.5 GB ceiling at reserve 4 (aborted at 14534) but FITS at reserve 5 (13355 MB, 8.75 s/it). Native-Laplacian
>     across canvases is resolution-confounded (a smaller frame scores higher per-pixel) -> judge canvas sharpness BY EYE.
>   - (4) **i2v 0.75-vs-0.62 + native-vs-respaced sigmas:** objective metrics do NOT separate them (seam 0, freezes 0)
>     -> EYEBALL-ONLY (the stutter + the distilled re-spaced-sigma look).
>   - All legs: scene-cuts 0; VRAM <= ceiling on every OK leg; the VRAM ceiling abort + reserve step-up retry chain both
>     fired correctly (704@reserve4 aborted at +34 MB, 704@reserve5 passed); s/it abort never needed (no spill).
> - **CONVERGED WINNER (3 review voices + Claude judge -- LOCKED 2026-06-27):** Codex r1/r2 (pasted) +
>   AntiGravity/Gemini (pasted) + a fresh LOCAL Codex r4 (roundtables/2026-06-27-ltx-av-wiring/r4/, codex exec
>   read-only; agy CLI installed but needs interactive sign-in) all = GO-WITH-FIXES; Claude grounded every claim.
>   FINAL = **DECODE temporal_size 128 / overlap 32** (tiled, seam imperceptible ratio 0.57, ~228MB headroom vs
>   whole-clip's 27-162MB knife-edge) + **scaler lanczos + unsharp 0.4** (the unsharp is the sharpener; resampler
>   ~irrelevant: lanczos +1.3% vs unsharp0.4 +8.9% vs unsharp0.8 +14.7%) + **canvas 512x288**. Whole-clip 4096/8 =
>   documented manual max-quality option only. The decode+scaler fix is RECIPE-AGNOSTIC; distilled_native-as-default
>   stays the separately-deferred operator decision. Hardened plan: roundtables/2026-06-27-ltx-av-wiring/r4/final.md.
>   Open eyeball: unsharp 0.4 vs 0.8 (halo on faces) -- clips S0_lanczos_unsharp04/08 in _bakeoff_ltxq/. Stutter
>   (i2v 0.62) was NOT validated by the bakeoff (freezedetect read 0 at baseline) -- remains an eyeball item.
> - **PANEL QA PROMPT for the candidate models (operator-requested):** `docs/2026-06-27-ltx-av-qa/PANEL_QA_PROMPT.md`
>   (copy-paste; full results table + the perf-vs-quality ask).
> - **NEXT (operator-gated):** operator (+ /roundtable panel) picks the winner from the clips; THEN wire it into
>   otr_silent_composite (scaler) + eng_ltx_av (decode temporal_size) + render_driver (canvas, only if bumped) in the
>   SAME change, re-validate (OTR_WorkflowValidator + round-trip + link/widget audit), suite + Bug Bible, commit+push
>   v2.0-alpha. Do NOT touch eng_humo.py / eng_wan_ti2v.py. prod/main + tags GATED.
>
> **>>> EXECUTED SPEC (PLAN v5) -- 2026-06-27 LTX-AV QUALITY BAKEOFF (isolated smoke; RAN). origin/v2.0-alpha
> HEAD `230dd1b8` (build started here). The overnight 420w soak exposed the ltx_audio_in clips as LOW QUALITY + micro-stutter; root-caused
> via a 3-AI QA (Claude + Gemini + Codex), all code-grounded + converged. prod/main + tags GATED.**
> - **DIAGNOSIS (converged):** (1) QUALITY = the `render_driver.py:1116` clamp `OTR_LTX_AV_RENDER_CANVAS=512x288`
>   (engine native = 832x480, eng_ltx_av.py:55-56); `OTR_SilentComposite` (otr_silent_composite.py:323-325) then
>   upscales ~8.3x AREA to the 1472x832 canvas -> softness. Root = the tiny native render, NOT the upscaler. (2)
>   STUTTER = LTX init-image boundary HOLD + low-motion tail -- NOT ping-pong (eng_ltx_av renders `next_8n1` full
>   length @ :605-627, never calls `wrapper_bridge.extend_frames_to_target`; Codex verified 153/153 unique frames,
>   zero dups, palindrome test FAILED). freezedetect hits only on the 153f clip (~0.16s start, 0.12s end). (3)
>   "FLASH" = NOT hard cuts (0 scene-cuts, max scene score 0.003); it's TEMPORAL-VAE-TILING seams --
>   `VAEDecodeTiled temporal_size=64/overlap=8` (eng_ltx_av.py:556-559) = a 56-frame stride, and the p99-luma jumps
>   land EXACTLY at frames 55->56 / 111->112 (Codex). Sigma-boiling from the bunched `LTX_DISTILLED_SIGMAS`
>   (eng_ltx_av.py:148; 5/8 steps >0.97 then a hard 0.42->0.0 drop) is a SECONDARY suspect (Gemini).
> - **HARD CONSTRAINTS:** VRAM knife-edge (AV stack cycles ~24GB through the 16GB card; 512x288 already peaks
>   13688MB <= 14500). 832x480 = 2.71x pixels -> LIKELY over budget unless another lever changes -> use MEASURED
>   tiers + a reserve bump, hard-fail >14.5GB. For distilled_native, `OTR_LTX_AV_STEPS` ALONE does NOTHING -- the
>   path uses FIXED manual sigmas (eng_ltx_av.py:538-544); changing effective steps REQUIRES a new sigma tuple.
> - **PLAN = isolated LTX-AV quality bakeoff** (clone workflows/ltx_av_bakeoff_gguf.json + scripts/run_ltx_av_bakeoff.py;
>   FIXED still+audio+seed+prompt, distilled_native Q3_K_M, wide aspect; vary ONE lever per leg IN THE STANDALONE JSON,
>   carry the winner forward; each leg logs peak VRAM + s/it + wall, HARD-fail >14.5GB, writes
>   `otr\episodes\_bakeoff_ltxq\<leg>.mp4` for side-by-side QA):
>   L0 baseline (512x288 / native-sigmas / overlap8 / i2v0.75); L1 `temporal_overlap` 8->16->32 (the flash);
>   L2 `i2v_strength` 0.75->0.62 (the boundary stutter); L3 a RE-SPACED 8-step sigma tuple A/B vs native (gamble --
>   distilled is calibrated to its schedule); L4 canvas 640x384 then 704x384 + reserve 4->6-7GB (the sharpness;
>   832x480 ONLY if 704 fits). Operator + AIs eyeball clips; pick the best clean combo <=14.5GB; THEN wire the winner
>   into render_driver/eng_ltx_av defaults + the boot/profile (SAME change as any code), re-validate, suite + Bug
>   Bible, commit+push v2.0-alpha.
> - **Lever locations (code-grounded):** canvas `render_driver.py:1116`; reserve `eng_ltx_av.py:90`; sigmas
>   `eng_ltx_av.py:148`; i2v_strength `eng_ltx_av.py:152-154/174-181` (`_recipe_config`); VAE temporal
>   `eng_ltx_av.py:556-559`. The isolated standalone JSON varies these DIRECTLY -- NO production-code change until the
>   winner is wired. One GPU = one render (serialize). z_image soak bug already FIXED (OTR_ZIMAGE_UNET=nvfp4 in the
>   boot, verified live); LTX-AV default unet set to distilled-1.1 Q3_K_M.
> - **>>> PLAN v2 (post-finalization roundtable; Gemini R4 = NO-GO on the v1 greedy sweep, fixes accepted). The v1
>   L0-L4 greedy carry-forward is SUPERSEDED -- the levers INTERACT (sigmas<->i2v_strength; canvas<->temporal/spatial
>   tile), so a greedy winner is false. Revised:**
>   - **PHASE 0 (FREE, do FIRST, no GPU re-render): composite scaler upgrade.** otr_silent_composite.py:323-325 upscales
>     512->1472 with ffmpeg default (bilinear) -> softness. Re-composite an EXISTING bakeoff clip with `flags=lanczos`
>     + mild `unsharp` vs the current scaler; eyeball + sharpness proxy. ZERO VRAM/render cost; may recover most of the
>     softness WITHOUT raising the render canvas (could make Phase B unnecessary).
>   - **PHASE A (cheap, baseline 512x288, FACTORIAL not greedy):** A1 `temporal_overlap` 8->16->32 (+ optional
>     temporal_size 64->128) -- metric YDIF max at frames 55/111 < 2.0. A2 a 2x2 grid {native vs re-spaced sigmas} x
>     {i2v 0.75 vs 0.62} -- metric freezedetect total frozen <0.1s + eyeball coherence (the re-spaced sigma MUST NOT
>     break the distilled look). Pick the best (overlap, sigmas, i2v) on the baseline canvas.
>   - **PHASE B (expensive, LAST, OPTIONAL fallback): canvas bump** ONLY if Phase 0+A aren't enough. 640x384 FIRST with
>     `tile_size` scaled to the new canvas + reserve raised CAUTIOUSLY; ABORT if s/it collapses toward the 223 spill
>     regime (the reserve-VRAM global is brittle per docs/2026-06-26-ltx-av-vram-headroom). 704x384 only if 640 holds.
>   - **Per leg:** peak VRAM + s/it + wall + YDIF-seam + freezedetect (objective gates), hard-fail >14.5GB, labeled clip
>     `otr\episodes\_bakeoff_ltxq\<leg>.mp4`. SAFEST single ship = Phase 0 scaler (zero risk) then the Phase-A winner.
>     Wire winner(s) across otr_silent_composite (scaler) + eng_ltx_av (temporal/sigma/i2v) + render_driver (canvas),
>     re-validate, suite + Bug Bible, commit+push v2.0-alpha.
> - **>>> PLAN v5 (LOCKED; folds BOTH v4 votes -- Gemini + Codex, both GO-with-fixes. Deltas over v2 above):**
>   - WHOLE-CLIP DECODE = ONE leg `4096/8` (matches the shipped sister `eng_ltx_video.py:761-767`, so a win wires
>     byte-for-byte), RUN LAST; `256/8` DROPPED as a separate leg (identical at >=153f -- decode chunk =
>     min(temporal_size, frames)); 256 is the contingency value ONLY if 4096 mis-allocates on the UNET-resident
>     ltx_audio_in path (eng_ltx_av.py:613-622 keeps the 22B resident through run_graph).
>   - STAGE-3 RESERVE starts at PRODUCTION 4GB (eng_ltx_av.py:90); step to 5-6 ONLY on a spill/OOM; NEVER accept a
>     crawling "pass" -- the `s/it>30 OR >2.5x-baseline` abort is LOAD-BEARING (a 704x384 sampler peak vs the resident
>     22B can hit the 223 s/it spill BEFORE decode).
>   - SHARPNESS baseline = "L0 conformed with the STAGE-0 WINNING scaler" (>=15% Laplacian variance; Stage 0 + Stage 3
>     legs ONLY).
>   - SEAM gate: `p99-luma jump <2.5 AND <=1.5x LOCAL median` = clean pass; if NO tiled candidate passes, rank by
>     seam-reduction-vs-L0 and require >=75% reduction (never strand the bakeoff with no winner); whole-clip legs
>     record "no seam"; seam indices DYNAMIC from (temporal_size-overlap).
>   - MANIFEST adds OUTPUT POLICY: the bakeoff renders SILENT to MATCH production (`encode_frames_to_silent_mp4`,
>     eng_ltx_av.py:625-627) -- NOT the legacy builder's CreateVideo/SaveVideo audio-mux
>     (build_ltx_av_bakeoff_workflow.py:213-218); + unique run-id/pre-delete, raw + conformed paths, mtime/size/SHA,
>     encoder settings, audio-ABSENT assertion. Plus the resolved-API-PROMPT asserts (canvas/length/sigmas-text/
>     temporal+spatial-tile/i2v/unet=distilled/LoRA+ModelSamplingLTXV+LTXVScheduler ABSENT/cfg=1.0/euler_cfg_pp/
>     CPU-encoder/seed/fps/prompt+asset SHA/boot reserve+baseline VRAM/conform -vf hash). Fail loud on any mismatch.
>   - HIGHEST RISK (both votes, unchanged): the harness measuring the WRONG graph (legacy builder is SHARP/LoRA
>     @832x480x105) -- the fail-loud resolved-API-prompt manifest is the load-bearing safety rail.
>
> **>>> PRIOR STEP -- 2026-06-26 LATEST (LTX distilled BAKEOFF + recipe-follows-model SHIPPED & GPU-PROVEN;
> Wan TI2V-5B low-VRAM bakeoff NO-GPU PREP STARTED -- Step-1 architecture gate = GO. origin/v2.0-alpha HEAD
> `9218b38e` == local. prod/main + tags GATED.)**
> **LTX A2V -- DONE this session (the prior step's "immediate next" is now complete):**
> - **Isolated dev-vs-distilled bakeoff RAN** (workflows/ltx_av_bakeoff_gguf.json + scripts/run_ltx_av_bakeoff.py,
>   committed `bd455b35`): dev Q3_K_M+SHARP 13.0 s/it / 15.5 GB / 231s; distilled-1.1 Q2_K 7.4 / 13.3 / 80s (soft
>   faces); Q3_K_M 9.4 / 15.1 / 95s (dev-match); Q4_K_M 32.5 / 16.1 / 281s (SPILLS, out). 3-way panel
>   (Claude+Gemini+Codex) CONVERGED: **distilled-1.1 Q3_K_M = daily driver; dev Q3_K_M+SHARP = hero/final;
>   Q4_K_M OUT; Q2_K emergency-VRAM only.**
> - **Recipe FOLLOWS THE MODEL shipped** (`fd9edc28` + log line `9218b38e`): eng_ltx_av binary OTR_LTX_AV_SHARP ->
>   tri-state resolver `_recipe()` -- auto-detects sharp_lora(ltx-2.3-22b-dev-)/distilled_native(ltx-2.3-22b-distilled-1.1-)
>   from the unet basename; `OTR_LTX_AV_RECIPE` override (auto|sharp_lora|distilled_native|m0_base); FAIL-LOUD on
>   ambiguous unet / bad override / retired OTR_LTX_AV_SHARP (NO FALLBACKS). distilled_native = sharp recipe MINUS the
>   LoRA (unet->guider directly, no ModelSamplingLTXV, fixed distilled sigmas + euler_cfg_pp + cfg 1.0). One
>   _recipe_config struct feeds all 4 sites + the dynamic _keep_set + sigmas-injection (Gemini's 2 lifecycle catches).
>   A2V ONLY; eng_ltx_video FROZEN. No widget change (env/code-driven; canonical JSON untouched). +11 tests; Bug Bible
>   green; suite has the SAME 5 pre-existing 267a53e workflow-pin fails (proven independent by stash+rerun).
> - **LIVE smoke PASS:** a fresh-cut 30w run + a 3-slot (announcer+music+character) run rendered full episodes; node-92
>   ok=true; distilled_native VERIFIED through eng_ltx_av.render_clip via ZERO LoraLoaderModelOnly + ZERO
>   ModelSamplingLTXV in the server log. Flip daily<->hero = swap OTR_LTX_AV_UNET only.
> - **DEFAULT-WIRING DEFERRED (operator call):** the engine still defaults to dev (sharp_lora); making distilled_native
>   the DEFAULT belongs with the 16GB/8GB/non-RTX/Mac/AMD TIER+SHIP decision (the recipe is env-driven, so each tier
>   profile just sets OTR_LTX_AV_UNET and the recipe auto-follows -- no per-platform code fork). LTX-AV is a CUDA/16GB
>   feature by design (fails closed without NVML); 8GB/Mac/AMD route around it to the lighter lanes.
> **WAN 2.2 TI2V-5B low-VRAM GGUF BAKEOFF (image-to-video ONLY) -- NO-GPU PREP STARTED. Scope: touch eng_wan_ti2v.py
> + a NEW isolated Wan bakeoff harness ONLY; do NOT touch eng_humo.py (HuMo done/out of scope) or eng_ltx_av.py.
> SERIALIZE the GPU render behind the in-flight LTX 3-slot smoke (one GPU=one render; one coder in code at a time).**
> - **STEP 1 ARCHITECTURE GATE = GO (read-only inspection of eng_wan_ti2v.py, 2026-06-26):** "is QuantStack TI2V-5B
>   GGUF architecturally compatible with OTR's wan_ti2v assumptions?" -> YES. OTR ALREADY loads the 5B via
>   `UnetLoaderGGUF` (`_loader_mode()` returns gguf by default; OTR_WAN_TI2V_LOADER pins it; the GGUF resolves from
>   `diffusion_models` OR `unet`). The 5B graph (schema-verified vs live /object_info 2026-06-18) is: UnetLoaderGGUF ->
>   ModelSamplingSD3(shift 5.0) -> KSampler(euler/simple, steps 30, cfg 5.0, denoise 1.0); CLIPLoaderGGUF(umt5,
>   type=wan) -> CLIPTextEncode pos/neg DIRECT; VAELoader(`wan2.2_vae.safetensors` REQUIRED -- M8 allow-list, fails
>   closed on the 2.1 VAE) -> `Wan22ImageToVideoLatent`(vae, start_image, w/h/length) -> KSampler -> VAEDecodeTiled.
>   **Swapping a GGUF quant = change OTR_WAN_TI2V_UNET_NAME ONLY -- NO node substitution beyond UnetLoaderGGUF + path.
>   So the FAILURE CRITERION is NOT triggered: GO to build the harness.** (Confirm ComfyUI-GGUF supports WAN22 at build.)
> - **NEXT no-GPU:** Step 2 confirm QuantStack/Wan2.2-TI2V-5B-GGUF + that wan2.2_vae + the umt5 TE on disk match (reuse
>   only if they match) + ComfyUI-GGUF WAN22 currency; Step 3 download current-baseline + Q2_K(1.85GB) + Q3_K_M(2.55GB)
>   to C:\ComfyUI-Models (skip Q3_K_S; Q4_0 only if Q3 wins-but-soft); Step 4 build the ISOLATED standalone Wan TI2V
>   bakeoff JSON (minimal graph above, swappable unet only -- mirror the LTX bakeoff harness) + a runner logging
>   file/format + s/it + total + peak VRAM + res/frames/scheduler/offload + clip -> otr\episodes\_bakeoff_wan\<quant>.mp4.
>   Step 5 (GPU, ONLY after the LTX smoke frees the card) run headless + report; winner wires as a wan_ti2v option in
>   the SAME commit as the canonical workflow change (CLAUDE.md S0; re-validate) + tests; suite + Bug Bible green;
>   commit+push v2.0-alpha. Operator /roundtable manually as needed.
> - **2026-06-27 HARNESS BUILT (NO-GPU; Steps 2-4 in code). Operator ran their own convergence roundtable
>   (AntiGravity); Claude grounded every claim vs the real files + judged.** Two ISOLATED files (HuMo + LTX engines
>   + eng_wan_ti2v.py UNTOUCHED -- the harness submits a RAW core-node API graph over HTTP, no engine import):
>   - `scripts/otr_wan_ti2v_bakeoff_gguf.json` (next to the runner, NOT under workflows/ -- that dir's guardrail/zod
>     tests require canonical LiteGraph shape; this is an intentional API-prompt harness input) -- format ON PURPOSE (named inputs) to sidestep the
>     LiteGraph positional-`widgets_values` trap; mirrors `eng_wan_ti2v._build_graph` (UnetLoaderGGUF ->
>     ModelSamplingSD3 shift5 -> KSampler euler/simple/30/cfg5/denoise1; CLIPLoaderGGUF umt5 type=wan ->
>     CLIPTextEncode pos/neg; VAELoader wan2.2_vae -> Wan22ImageToVideoLatent 832x480x49 -> VAEDecodeTiled
>     256/64/16/8 -> CreateVideo 24fps -> SaveVideo). Quant swap = node-1 `unet_name` ONLY.
>   - `scripts/run_wan_ti2v_bakeoff.py` -- BOOT-PER-LEG (NOT the LTX single-session loop): a leg = quant x clamp
>     tier, each its own reset_box + boot + one /prompt + teardown. Clamp tiers vram_full/8gb/6gb ride the
>     launcher's NEW optional `OTR_HEADLESS_RESERVE_VRAM_GB` -> `--reserve-vram (total-target)` clause
>     (`_otr_soak_server_launch.cmd`, additive + default-OFF = byte-identical for every other lane). Logs s/it +
>     peak VRAM + **peak SYSTEM RAM** + sysmem-spill hint; `--dry-validate` = cheap no-GPU node-currency + quant-enum
>     preflight; QUANTS baseline Q5_K_M + Q3_K_M + Q2_K (missing-on-disk = SKIP, so Step 3 download gates legs).
>   - WHY boot-per-leg (convergence catch): `--reserve-vram` is startup-only (can't mutate live over HTTP) AND Comfy
>     deliberately does NOT force-unload between prompts (wrapper_bridge `_soft_free`, BUG-265 anti-frag), so a
>     single-session loop carries the prior quant's UNET/TE resident + contaminates the next peak. Static-green
>     (AST + JSON refint + wan2.2-VAE/clip-type=wan/euler/temporal_overlap<size + no-BOM).
>   - **NEXT (GPU now free 2026-06-27): suite + Bug Bible; commit+push v2.0-alpha; then Step 3 download Q3_K_M
>     (2.55GB)/Q2_K (1.85GB) -> C:\ComfyUI-Models\unet; then `--dry-validate`; then the GPU sweep.** Overnight
>     priority per operator = the 420w big-chunky-story soak (below), so the Wan GPU sweep waits for the card again.
>
> **>>> PRIOR STEP -- 2026-06-26 LATE (LTX-AV VRAM SPILL FIXED + GPU-VALIDATED; bug cataloged; cosmetic JSON tidy.
> origin/v2.0-alpha HEAD `53418748` == local (reserve tuned 3->4GB after a live 72->5.6 s/it pass). NEXT = (1) ISOLATED
> distilled-1.1 vs dev Q3_K_M SPEED+QUALITY A/B (quants Q2_K/Q3_K_M/Q4_K_M + distilled companions DOWNLOADING now to
> C:\ComfyUI-Models\{unet\distilled-1.1, vae, text_encoders}); THEN (2) 400-700w VISUALIZER-ONLY story-quality review +
> a /roundtable. prod/main + tags GATED.)**
> **Operator pain this session: all-`ltx_audio_in` episodes rendered char/bookend beats at a 6.84-or-223 s/it LOTTERY
> (one beat 54s, the next ~30 MIN), stalling at "Model Initializing" -- NOT an OOM. Roundtable-converged
> (docs/2026-06-26-ltx-av-vram-headroom/, GPT-5.5+Gemini-3.1-pro+DeepSeek-v4-pro ~$0.38) -> TWO root causes:**
> - **`ae8ec55e` VideoVAE graph-pin split:** the single `videovae` in `eng_ltx_av._build_graph` fed BOTH i2v
>   (pre-sampler) AND decode (post-sampler); run_graph free_after_use frees only after the LAST consumer -> 1.38GB
>   pinned through the denoise loop. Split into `videovae_enc`(i2v)+`videovae_dec`(decode) -> reclaimed before the
>   sampler. Internal render graph only (no workflow-JSON change); +split-lock test.
> - **`bd5ffd23` activation reserve (load-bearing):** the operator's DESKTOP APPS eat ~5GB VRAM, so the 10.5GB 22B
>   unet FULL-loaded with ~0 activation room -> the audio-conditioned sampler spilled to system RAM (sysmem
>   fallback). `eng_ltx_av._ltx_av_vram_reserve()` holds `OTR_LTX_AV_RESERVE_VRAM_GB` (default 3) free via comfy
>   `EXTRA_RESERVED_VRAM` around run_graph (restored in finally; exception-safe; works GUI+headless). GGUF unet
>   honors partial load (ltx_video b001 already did). +4 reserve tests.
> - **GPU-VALIDATED: live 30w all-ltx_audio_in headless -> 6 consecutive beats STEADY ~11 s/it (~90s/clip) vs the
>   old 223 = ~19x faster + DETERMINISTIC (no lottery). LTX/motion 126/126; Bug Bible 16/7/3; only the 5 pre-existing
>   267a53e workflow-pin fails. OPERATOR: RESTART ComfyUI to load it (py module cache); lower
>   OTR_LTX_AV_RESERVE_VRAM_GB for more speed / less margin; closing desktop apps frees headroom.**
> - **`fa1ca903` BUG-LOCAL-414 + Bible `07.16`** (Three-File Contract; survival-guide pushed `4df8e00`): the spill is
>   a NEW instance of a RECURRING VRAM-pressure-slow-crawl-without-OOM family (Bible 11.07/07.01/07.03), cataloged.
> - **`d183544d` cosmetic tidy:** node-87 `music_video_model` bare `ltx_audio_in` -> `ltx_audio_in (16:9)` (GUI shows
>   the suffix like the sibling rows; resolves identically; validated round-trip + no-BOM).
> - **EARLIER THIS SESSION (the LTX-consolidation start + after):** leak-floor-v2 flipped DEFAULT-ON; the news-close
>   `inject_central_object_into_brief` made a NO-OP (keeps the brief NEWS summary, drops the tacked-on object).
> - **OPEN TICKET (caught, NOT fixed):** `normalize_length` is FAILING -- the writer LLM emits edit items keyed
>   `type` not the schema's `action`, so it exhausts the ladder + bails (non-fatal). It is WHY the 30w tests ballooned
>   to ~110w (length control off). SIBLING of BUG-303 (`index`->`beat_index`): a 1-line `BeatEdit.__otr_field_aliases__`
>   add (`action`<-`type`). Fix it as part of the story-quality pass.
> - **`53418748` reserve tuned 3->4GB + GPU-CONTENTION learned (live):** 3GB still let the unet FULL-load (usable
>   10958 > the 10537 unet) -> 72 s/it marginal spill; 4GB forces a real PARTIAL load (~540MB offloaded) -> no spill.
>   THEN the GUI still ran 42 s/it at 53% GPU-util while HEADLESS hit 5.6 s/it at ~full util -- the gap is DESKTOP GPU
>   CONTENTION, not the model. On this Legion (HYBRID display, iGPU drives the panel) Snagit/StreamDeck/Chrome/Edge
>   still hold 5080 graphics ctxs, and a per-app "Power Saving (Intel)" only applies on app RESTART. RECIPE for fast
>   GUI renders: 4GB reserve (shipped) + route those apps to the iGPU AND RESTART them; GUARANTEED-fast = render
>   HEADLESS. The model + fix are VINDICATED -- no Q2_K downgrade is needed for speed.
> - **DISTILLED-1.1 SPEED+QUALITY A/B (DOWNLOADING, then isolated test -- THE immediate next step):** operator wants
>   to know if the NATIVE distilled-1.1 model beats the dev+SHARP-LoRA path on QUALITY (faces/lip-sync). Downloading
>   `unsloth/LTX-2.3-GGUF` distilled-1.1 Q2_K(7.94)/Q3_K_M(10.63)/Q4_K_M(14.19) + distilled audio/video VAE +
>   embeddings_connectors (Gemma-3 TE reused from dev) -> `C:\ComfyUI-Models\{unet\distilled-1.1, vae, text_encoders}`.
>   DELIVERABLE = an ISOLATED BAKEOFF (operator-requested): a minimal STANDALONE ComfyUI JSON -- just the LTX-AV graph
>   (GGUF unet + TE + video/audio VAE -> ONE ltx_audio_in clip from a FIXED still + audio; the unet file is the only
>   swap), NOT the full OTR pipeline -- plus a runner that renders the SAME still+audio through EACH candidate (dev
>   Q3_K_M baseline + distilled-1.1 Q2_K/Q3_K_M/Q4_K_M) HEADLESS and logs per-quant **s/it + peak VRAM (efficiency)**
>   and writes each clip to a clearly-LABELED path (e.g. otr\episodes\_bakeoff\<quant>.mp4) for **operator side-by-side
>   quality comparison**. For distilled, **DROP the SHARP distilled-LoRA** (the model is
>   already distilled -> double-distill = mush), use native distilled sigmas/cfg + the distilled companions; compare
>   s/it AND faces vs the dev Q3_K_M baseline. Operator JUDGES the look (quality is the question; speed is solved).
>   If distilled-1.1 wins -> swap `OTR_LTX_AV_UNET` + companions + drop the LoRA in `eng_ltx_av`.
> - **AFTER the A/B (operator-chosen story-quality lane): 400-700w VISUALIZER-ONLY runs** -- skip the slow LTX video (use
>   visualizer/procgen for video = fast) to make real readable episodes for Claude's PERSONAL read (is the story
>   actually improving from leak-floor-v2 / news-close / specificity anchors / StoryCritic?), THEN a /roundtable to
>   harden the next lever. The 30w runs were SPEED tests -- too short to judge story quality.**
> - **PARALLEL LANE -- SANCTIONED, PREP-ONLY (Wan 2.2 TI2V-5B low-VRAM bakeoff):** an OPTIMIZATION, not a rescue
>   (OTR's wan_ti2v already runs ~9.7GB OBSERVED peak -- workflow-specific, not a model floor; Comfy docs say native
>   5B "fits well on 8GB"). A SEPARATE window may do NETWORK/DISK/HARNESS PREP ONLY; the GPU render WAITS for the LTX
>   bakeoff to free the card (one GPU = one render; one coder in the code at a time). Scope: (a) FIRST inspect
>   `eng_wan_ti2v` -- native `Load Diffusion Model` (models/diffusion_models) vs `UnetLoaderGGUF` (models/unet)? the
>   GGUF path differs; phrase it "is QuantStack TI2V-5B GGUF architecturally compatible with OTR's wan_ti2v
>   assumptions", do NOT assume. (b) target `QuantStack/Wan2.2-TI2V-5B-GGUF` (direct conv of Wan-AI/Wan2.2-TI2V-5B;
>   TI2V text+image-to-video; NOT I2V-A14B / T2V-A14B / S2V-14B). (c) the 5B needs **`wan2.2_vae.safetensors`** (NOT
>   the 14B-only `wan_2.1_vae`) -- EXPLICIT check; reuse OTR's UMT5 TE/VAE ONLY if they match. (d) bake off ONLY
>   current-baseline vs Q2_K(1.85GB) vs Q3_K_M(2.55GB) first (skip Q3_K_S = tiny diff on 16GB; add Q4_0(3.03GB) only
>   if Q3 wins-but-soft). Log file/format + s/it + total + peak VRAM + res/frames/scheduler/offload + TE/VAE-on-GPU +
>   clip -> `otr\episodes\_bakeoff_wan\<quant>.mp4`. **FAILURE CRITERION:** if a GGUF quant needs node substitutions
>   BEYOND `UnetLoaderGGUF` + the model path -> DOCUMENT and STOP before wiring (don't let the lane sprawl). WAN22
>   gotcha: ComfyUI + ComfyUI-GGUF must support `WAN22` (confirm; OTR runs wan_ti2v so likely OK IF already GGUF).
> - **TIER COVERAGE AUDIT (operator clarification 2026-06-26 -- the GOAL is COVERAGE, NOT auto-selection):** the point
>   is to confirm the model-agnostic platform has a PROVEN/viable engine OPTION at each VRAM class (the operator PICKS
>   per hardware; NO auto-selector is being built). Map by CAPABILITY x VRAM: **audio-driven SCENE / A2V** = LTX 2.3
>   audio-in (22B, 12-16GB; distilled-1.1 bakeoff = smallest-viable; under-8GB is the OPEN GAP -- LTX is just big).
>   **image-to-video** = Wan TI2V-5B (native 8-12GB / Q3 mid / Q2_K under-8 via the Wan bakeoff; t2v-only = no-anchor
>   draft/compat floor). **audio-driven FACE (lip-sync)** = HuMo -- ALREADY PROVEN, no work: HuMo-1.7B = the LOW-VRAM
>   face option (~3-4GB peak), HuMo-14B = the high-quality opt-in. **draft/floor (no model, any VRAM)** = visualizer /
>   still_parallax (procgen). COVERAGE is solid EXCEPT under-8GB A2V; the LTX + Wan bakeoffs just fill the
>   smallest-viable cells. This is a CHECKLIST of options-per-tier, NOT a runtime auto-picker.
>
> **>>> PRIOR STEP -- 2026-06-26 (CODER: LTX audio-in CONSOLIDATION COMPLETE -- Chunk 1 + Chunk 2 SHIPPED (one
> ltx_audio_in engine; ltx_av_talk/ltx_av_music DELETED), overnight soak RUNNING. origin/v2.0-alpha HEAD
> `a30f5945` == local. prod/main + tags GATED.)**
> **Operator ask: "remove the two legacy [ltx_av_talk/ltx_av_music] and ensure the still logic is most robust" + run
> the 420w overnight soak. Hardened via a 2-round LIVE roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, ~$0.61,
> docs/2026-06-26-ltx-consolidation/): the panel caught that ONE engine serving BOTH character + scene beats
> mis-routes the character beats 3 ways -> fix = ROLE-driven routing, not family-driven.**
> - **SHIPPED `e8fa941d` -- Chunk 1 (robust still routing); full suite green vs the 5 pre-existing 267a53e
>   workflow-pin fails + Bug Bible 16/7/3 + no-BOM + AST-clean:** `ltx_audio_in` now renders end-to-end. render_driver
>   routes it like flux/flat_still (the beat's WIDE scene still -- scene_open radio bookend for announcer/music,
>   scene_character for character beats) -> fixes the `requires init_image (got '')` crash on b000_music_open.
>   Role-driven: one shared `_is_character_face_beat` classifier (role primary); `_uses_ambient_master_audio` excludes
>   char-face beats (clean own voice); char beats get the gear-free char fallback; 512x288 AV canvas clamp. Other
>   engines (ltx_video/wan_i2v/flux_still/humo) UNCHANGED. +11 tests/test_ltx_audio_in_routing.py.
> - **OVERNIGHT SOAK RUNNING (operator "launch the soak yourself"):** scripts/_otr_overnight_420_soak.py launched
>   detached vs a fresh :8000 boot (scripts/_otr_overnight_420_boot.cmd). Recipe: bookends=ltx_audio_in,
>   other-beats=still_parallax (img=z_image_turbo -- CHANGED from still_kenburns so the turbo intro stills actually
>   SHOW, not a procedural floor), voice=indextts2, 420w act=3; locals=mistral-nemo+gemma-4-e2b, frontier=grok-4.3,
>   creativity rotation, $20 frontier cap, 9.5h wall. Loads the REAL workflows/otr_scifi_16gb_full.json; finals ->
>   otr/obs. Morning summary scheduled. VRAM CAVEAT: box was at 4.6 GB (operator active) at launch -> early legs may
>   trip the 14.5 GB ceiling; frees as the box quiesces overnight.
> - **SHIPPED `a30f5945` -- Chunk 2 (legacy removal); full suite green vs the SAME 5 pre-existing 267a53e fails
>   (zero new) + Bug Bible 16/7/3 + no-BOM + AST-clean:** DELETED LtxAvTalkEngine + LtxAvMusicEngine (eng_ltx_av.py);
>   ltx_audio_in is the ONE audio-in engine + the music/announcer default. Repointed registry CAPABILITIES +
>   VALIDATED_ENGINES, scripts/otr_video_dep_pilot.py, render_driver name-maps (SYNTH_FALLBACKS / ENGINE_FAMILY /
>   _LTX_OPEN_ENGINES / 512x288 canvas clamp), config/profiles/16gb_full.json, and canonical workflow node-87 (x2)
>   -- all to ltx_audio_in (JSON re-validated, 21 files). 14 test files updated. ZERO live ltx_av_talk/ltx_av_music
>   refs remain (only removal-assertions).
> - **NEXT = operator look-QA on the overnight soak output (morning summary -> obs finals) + any remaining
>   engine-track work (section 3). The 5 pre-existing 267a53e workflow-pin fails are UNRELATED to this work and
>   still open (test_capability_profiles/test_workflow_apply x2/test_workflow_live/test_full_workflow_v2).**
>
> **>>> PRIOR STEP -- 2026-06-25 LATE EVENING (CODER: BOTH build-ready specs BUILT + SHIPPED + PUSHED, all green
> per chunk. origin/v2.0-alpha HEAD `db9d8ea7` == local. NEXT = the forward-order engine track (section 3 -- Wan
> eyeball / coverage sweep, mostly operator-gated GPU) OR an operator pick. prod/main + tags GATED.)**
> **SHIPPED -- 7 commits, each: full suite green vs the 5 pre-existing `267a53e` workflow-pin fails (verified
> pre-existing by stash+rerun) + Bug Bible 16/7/3 + AST-clean + no-BOM + commit AND push to v2.0-alpha:**
> - `95bbcbd2` **leak-floor-v2** (SPEC 1, DEFAULT-OFF/dark via `OTR_ENABLE_LEAK_FLOOR_V2` + `OTR_STRICT_LOCAL_CLEAN`,
>   byte-identical off): `_otr_line_hygiene` EntityPolicy/Defect/VerificationResult + `verify_and_repair_line` + 4
>   rules -- rule1 `strip_participle_before_quote` (NEW sibling; the `Gasping,` leak is `_leading_stage_strip`'s
>   line-271 lowercase guard NOT `_NARRATION_VERBS`, which is NOT widened), rule2 `scrub_roster_vocative` (ALL-CAPS
>   full-name vocative drop), rule3 `has_malformed_internal_quote`, rule4 news-bleed at `build_allowed_roster`
>   (`build_banned_source_proper_nouns` + `banned_terms`; rejects President Trump, NASA/CERN stay). Writer builds
>   the per-episode EntityPolicy + threads it; `_otr_ledger_scrub` gated rule-1 freeze backstop; +16 acceptance
>   tests (4 leaks + 3 negatives, 0 leak + 0 FP).
> - `5d140b82` **operator-flagged audio fix** (NOT in the specs -- Jeffrey reported the announcer SPEAKING
>   "Central object, if useful: <obj>"): `inject_central_object_into_brief` was concatenating the prompt-scaffold
>   label into the SPOKEN `news_close_brief`. Fixed in the brief builder -> a clean capitalized final-image
>   sentence; removed the unused `_CENTRAL_OBJECT_LEAD`; test guards the label can't return.
> - **SPEC 2 -- story-ledger DRIFT, 5 chunks IN ORDER:**
>   - `d1bb9e7d` **C1 kill the StoryCritic fail-OPEN**: `ArcVerdict += "unverified"`; `clean()` returns it (NOT
>     "strong") on every failure path; freeze DERIVES `meta.story_critic_status {ran,validated,failure}` from the
>     verdict (no `run_story_critic` signature change -> no stub churn); A3 floor downgrades strong/unverified +
>     reroll-targets -> "uneven"; finalize maps an unverified critic frozen_clean -> frozen_with_warns (observable,
>     never a hard block).
>   - `fbc829ac` **C2 cross-stage consistency guard (THE core)**: NEW `_otr_ledger_consistency.py` pure
>     `assert_ledger_consistency` off a `SOURCE_OF_TRUTH_MATRIX` (sound_palette<-contract.sound_world, title/premise/
>     setting/time_of_day<-outline->canon, style<-slug, cast, beat ids). GROUNDING CORRECTION: CastLock is a
>     DOWNSTREAM node (re-locks the FROZEN ledger) -> NOT in scope at freeze; the check runs in the WRITER pre-emit
>     where contract/outline/canon are real objects (castlock=None, cast source = ledger.cast). Audio-safe non-
>     strict (LOUD warn + `meta.consistency_status`, never raises); CI enforcement = `tests/test_ledger_canon_parity.py`
>     (14, incl. the golden sound_palette regression).
>   - `e6ad1b0d` **C3 CI drift guards** (test-only): widget-ORDER-vs-live-INPUT_TYPES check (BUG-LOCAL-097, reuses
>     the validator's widget classifiers; validated clean on the canonical 22 nodes) + a vintage-`l3-2026-05-14`
>     schema-compat fixture (schema-version pin + no false consistency defect on an old ledger).
>   - `9f725f4a` **C4 freeze WARN taxonomy**: deterministic 3-tier `classify_freeze_warning` /
>     `build_freeze_warn_taxonomy` (structural_error blocks / story_accuracy_warning ships non-clean+visible /
>     cosmetic clean-with-warns); stops mislabeling a shipped continuity/unverified/consistency finding "structural".
>   - `db9d8ea7` **C5 whole-episode critic context + cut StanceIssue**: NEW `_critic_story_bearing_lines`
>     (character + announcer) rendered READ-ONLY in the prompt + the beat_intent doctor-drift instruction;
>     reroll_targets kept character-only via a NEW post_validator guard (the spec's "already rejects" was untrue);
>     CUT the TELEMETRY-ONLY `StanceIssue` (model + field + prompt + `test_d2_antagonist_stance.py`).
> **OPEN VERIFY (operator, GPU):** leak-floor-v2 is DEFAULT-OFF -- a live 320w validation per writer lane gates its
> promotion to default-ON. The consistency guard + WARN taxonomy run live on every render (audio-safe, never block);
> eyeball `meta.consistency_status` / `meta.freeze_warn_taxonomy` on the next soak.
>
> **>>> PRIOR STEP -- 2026-06-25 EVENING (CODER HANDOFF: two build-ready specs CONVERGED + waiting to be BUILT.
> origin/v2.0-alpha HEAD `62c9f4f0` == local. The planner window produced the specs; a CODER window builds them.)**
> **NEXT CODING WORK -- build these two, IN ORDER; each chunk: regression suite + Bug Bible green, AST/no-BOM,
> commit AND push to v2.0-alpha per green chunk. Both are CONTENT/TEST-ONLY -- NO workflow-JSON node/widget churn;
> byte-identical audio spine untouched; ledger schema frozen except ONE `ArcVerdict` enum-value add.**
> 1. **leak-floor-v2** -- spec `docs/2026-06-25-leaking-words/pass02_plan.md` (+ judgment pass02_judgment.md).
>    4 narrow DETERMINISTIC leak rules (capitalised-participle+quote extract; caps-cast-vocative scrub via a new
>    `scrub_roster_vocative`; double-quote-only malformed check) + news-bleed fixed AT `build_allowed_roster`
>    (filter the real-person/political class out of `key_terms` so the EXISTING Phase-0 gate rejects "President
>    Trump"; NASA/CERN stay). Ships DEFAULT-OFF/dark (audio-affecting, per `_otr_config.py` 95/107). Acceptance:
>    `tests/test_leak_floor_v2.py` = the 4 observed shipped leaks (positive) + 3 negatives (first-name vocative,
>    legit org noun, non-stage `-ing` opening); 0 leak + 0 FP. GROUNDED ROOT-CAUSE: the `Gasping,` leak is the
>    `_leading_stage_strip` LOWERCASE-start guard (line 271), NOT the verb whitelist -- do NOT widen `_NARRATION_VERBS`.
> 2. **story-ledger DRIFT** -- spec `docs/2026-06-25-story-ledger-integrity/roundtable/CODER_BUILD_SPEC.md`
>    (R1-converged, panel+code-grounded). 5 chunks IN ORDER: (1) kill the StoryCritic FAIL-OPEN -- add `"unverified"`
>    to `ArcVerdict` (line 238), `clean()` returns it NOT "strong" (line 261; exhaust ~445-455), stamp
>    `meta.story_critic_status`, freeze maps unverified->non-clean, A3 floor (~567-590) downgrades strong/unverified+
>    reroll-targets to "uneven"; (2) THE core -- a PURE `assert_ledger_consistency` + `tests/test_ledger_canon_parity.py`
>    reflecting `StoryContract`/`CastLock` (the sound_palette-class guard; no LLM); (3) CI drift guards -- ADD a
>    widget-order-vs-live-`INPUT_TYPES` check to `tests/test_workflow_json_guardrails.py` (it only checks typing today)
>    + a vintage-ledger schema-compat fixture; (4) freeze WARN taxonomy (structural blocks / accuracy-warn ships
>    non-clean / cosmetic clean-with-warns -- stop calling a shipped arc failure "structural"); (5) critic gets
>    READ-ONLY whole-story context (it filters `speaker_role=="character"` at line ~394) + outline `beat_intent` in the
>    prompt + CUT `StanceIssue` (~150-166, telemetry-only dead-end). Guards DETERMINISTIC, never LLM; multi-LLM binary
>    voting is CUT.
> **>>> PRIOR STEP -- 2026-06-25 (SCHEMA-ADHERENCE SPRINT COMPLETE + LIVE-VALIDATED: Lever 1 SHIPPED; Lever 2
> DROPPED via G1; C4 DEFERRED. + 8-episode cross-engine render validation; wan_ti2v 5B confirmed non-crashing @ 9.7GB peak.)**
> **E2E GPU VALIDATION (operator-requested, 2026-06-25): a 320w all-visualizer episode on the canonical JSON
> rendered end-to-end on BOTH writer lanes -> OBS final + `audio_byte_identical OK`, zero tracebacks. LOCAL
> (mistral-nemo): `signal_lost_frozen_awakening` 77.8MB, 21:03. FRONTIER (`~openai/gpt-latest` via
> `openrouter:slot-a` -- the exact path the Opus bug exhausted): `signal_lost_thumb_on_the_relay` 80.7MB, 17:42,
> ~83k tokens (<$1). Lever-1 fired LIVE: C2 clamp coerced GPT's over-long `intent` fields; `normalize_length[openrouter:slot-a]`
> PARSED GPT's nested BeatEdit (no Field-required exhaustion) then skipped the structural rung straight to typed
> repair (C3) -- NO StructuredCallFailedError, the original 90k-burn failure mode is GONE. Record:
> `docs/2026-06-25-schema-adherence/e2e/E2E_RESULTS.md`.**
> A coder session built the model-agnostic tolerance from `docs/2026-06-25-schema-adherence/roundtable/pass04_plan.md`,
> REFINED by two LIVE grounding roundtables whose converged resolutions are the actual build spec (pass04's literal
> C1/C4 were SUPERSEDED where grounding contradicted them -- the operator chose "roundtable the fork" both times):
> - **NESTED-ALIAS FORK** (`docs/2026-06-25-schema-adherence/nested-fork/`, GPT-5.5+Gemini-3.1+DeepSeek-v4, ~$0.13,
>   CONVERGED): the proven Opus `normalize_length` failure is a NESTED `BeatEdit` (action emitted under `lever`),
>   but pass04 C1's `_normalize_field_keys` was TOP-LEVEL-ONLY -> could not reach it. Resolution: a shared
>   `apply_field_aliases` helper keyed on a `__otr_field_aliases__` ClassVar, run via each schema's own
>   `mode="before"` validator (pydantic recursion reaches nested models), byte-identical on canonical input.
>   This SUPERSEDES the separate except-arm normalizer.
> - **C4-SCOPE FORK** (`docs/2026-06-25-schema-adherence/c4-scope/`, same panel, ~$0.10, CONVERGED): C4's "schema
>   REQUIRED -> edit all 15 callers across 9 modules" is CUT (unanimous). The proven failure is already fixed by
>   C0/C1, so the repair turn is dead code -> C4 (schema-in-repair) DEFERRED with a ready recipe (an OPTIONAL
>   `_build_schema_snippet` shim INSIDE `structured_call`, which already holds the schema; wire from a REAL captured
>   failure, never speculatively).
> **SHIPPED -- 2 green chunks, each: full suite green vs the 5 pre-existing `267a53e` workflow-pin fails + Bug Bible
> 16/7/3 + AST/no-BOM + commit AND push to v2.0-alpha:**
> - `516644eb` (C0+C1+C2+C5+C6): `apply_field_aliases` + `validate_tolerant_data`/`parse_validate_tolerant` shared
>   core in `_otr_structured_call.py`; `BeatEdit.__otr_field_aliases__` {beat_index<-index, merge_with_index<-merge_with,
>   action<-lever} via the shared before-validator (replaces the bespoke BUG-LOCAL-303 remap); +16 conformance tests.
>   The proven nested failure now validates on attempt 1; canonical happy path byte-identical.
> - `d4ca6cd4` (C3 ladder): structural rung is JSON-syntax-only (a `ValidationError`/`PostValidationError` skips
>   straight to typed repair -- the Opus ~90k-token burn fix); except arms narrowed to `(JSONDecodeError,
>   ValidationError, PostValidationError)` so a plain `ValueError` propagates; dynamic attempt counter; 4->3-attempt
>   docs; 5 ladder tests updated to the new contract + 3 conformance ladder tests.
> NO workflow-JSON / node / widget change (env+code only -- the schema is internal pydantic, not a node widget).
> The whole sprint touched ONLY `_otr_structured_call.py` + `_otr_radio_editor.py`'s schema + tests. prod/main +
> tags GATED.
> **>>> G1 DONE (offline, no-GPU) -> DROP the binary dialogue/stage-direction lane (Lever 2).** Over 638 shipped
> ledgers / 5,513 character lines, the existing two-tier deterministic detectors (`split_stage_business` +
> `detect_stage_business_for_reroll`, whitelist `_NARRATION_VERBS`) leave a GENUINE residual of ~0: a deliberately
> over-broad verb-agnostic heuristic flags 841 (15.3%) candidates, but a 40-line sample is 0/40 genuine -- all
> false positives (names ending in `s` like "Reeves", ordinary 3rd-person dialogue verbs, and in-character SPOKEN
> commands like "Initiating final descent" an LLM lane would WRONGLY strip). The deterministic detectors already
> catch the real stage business; the lane is unnecessary, G2 is moot. Results + the reusable measure
> (`scripts/_otr_g1_abstain_residual.py`): `docs/2026-06-25-schema-adherence/binary/G1_RESULTS.md`.
> **SCHEMA-ADHERENCE SPRINT COMPLETE: Lever 1 (tolerance) SHIPPED; Lever 2 (binary) DROPPED via G1; C4 DEFERRED
> with a ready recipe.** C4 reopens only on a real captured non-alias schema drift.
> **>>> LEAKING-WORDS ROUNDTABLE (2026-06-25, R1+R2 CONVERGED, GPT-5.5+Gemini-3.1+DeepSeek-v4, ~$0.42) -- build-ready
> plan `leak-floor-v2` at `docs/2026-06-25-leaking-words/pass02_plan.md` (coder ticket, NOT yet built).** The panel
> GROUNDING CORRECTED the prior "cheap fix is ADD the verb to `_NARRATION_VERBS`" note: that is WRONG -- the shipped
> `Gasping, "..."` leak is the `_leading_stage_strip` LOWERCASE-start guard (`_otr_line_hygiene.py` line 271), which
> never consults the verb whitelist, so widening `_NARRATION_VERBS` does nothing for it. Converged fix = ONE
> mandatory deterministic offline verifier over 4 leak classes via NARROW structural rules (capitalised-participle+
> quote extract; full-name caps-vocative scrub; double-quote-only malformed check); news-bleed ("President Trump")
> fixed at `build_allowed_roster` by filtering the real-person class out of `key_terms` so the EXISTING Phase-0 gate
> rejects it (NASA/CERN stay); LLM cleaner CUT for v1 (unanimous). Ships DEFAULT-OFF/dark (audio-affecting per the
> `_otr_config.py` 95/107 pattern) + a regression corpus of today's 4 leaks. Judgment: pass02_judgment.md.
> **>>> CROSS-ENGINE RENDER VALIDATION (2026-06-25, operator-requested): 8 episodes rendered end-to-end -> OBS finals,
> all `audio_byte_identical OK`, all with the sound_palette fix populated.** 4 all-visualizer (mistral/GPT/gemma/grok)
> + 2 LTX+flux2_klein (mistral `time_slipping_away` 95.7MB, GPT `botswanas_empty_chair` 185MB) + 2 wan_ti2v(5B)+
> flux2_klein (mistral `illuminating_doubt` 295.9MB; GPT in flight). **wan_ti2v 5B VRAM peak 9.7GB << 14.5GB ceiling
> = the structural reason it renders where the 14B OOMs.** sound_palette FIX shipped `2baba3a4` (sound_world stamped
> into the contract meta + derived into `episode_canon.sound_palette`). Story analysis: frontier(GPT)>local; the
> machinery improved vs 2026-06-24 (every episode now carries a StoryContract). Docs pushed `41d49e24`.
> **NEXT = leak-floor-v2 (coder, build-ready) OR the forward-order engine track (section 3 -- Wan eyeball / coverage
> sweep, mostly operator-gated GPU) -- operator's pick.**
>
> **>>> PRIOR STEP (the planner handoff that kicked off this build) -- 2026-06-25 (SCHEMA-ADHERENCE BUILD HANDOFF -> a fresh CODER window. Docs only this
> session; design roundtable CONVERGED, NOTHING coded yet. origin/v2.0-alpha HEAD `89e9f8bf` == local.)**
> Operator constraint that drove this: *"I don't control what model people run; I don't want to force people
> into local vs remote -- it's their choice."* So `structured_call` must parse what ANY user-chosen model
> (local Ollama OR remote OpenRouter) emits, without breaking the byte-identical local happy path.
> MOTIVATING EVIDENCE (this session): a frontier-writer probe (Claude Opus via OpenRouter) DECISIVELY broke
> the local prose ceiling (real noir medical-ethics drama vs the local console-standoff), but tripped the
> `normalize_length` structured schema -- the model emitted a `lever` field + omitted a required field ->
> exhausted the retry ladder -> soft-failed + burned ~90k tokens. That friction hits ANY arbitrary model, so
> a model-agnostic robustness pass is the unlock for letting people pick their own writer.
> **A LIVE 2026-06-25 roundtable campaign CONVERGED two complementary, model-agnostic levers (total ~$0.79):**
> - **LEVER 1 = TOLERANCE -- BUILD-READY (the main ticket).** `docs/2026-06-25-schema-adherence/roundtable/
>   pass04_plan.md` (4 rounds R1 arc -> R2 coding -> R3 wiring -> R4 convergence; GPT-5.5 + Gemini-3.1-pro +
>   DeepSeek-v4-pro panel, Claude grounded judge+panelist; ~$0.66; pass00->pass04 trail + judgments
>   alongside). C0-C6, all with a **STRICT-FIRST INVARIANT** (try the exact current parse first; tolerance
>   fires ONLY on failure => the local happy path stays byte-identical): deterministic key-normalizer for
>   alias drift; SKIP the structural retry on a pydantic `ValidationError` (it never helps, it just burns
>   tokens -- the Opus failure mode); CALL-SITE-WIRED schema repair (the repair factory is injected at the
>   call site, NOT imported into the core -> avoids the circular import); a shared `validate_tolerant_data`
>   core reused by both `structured_call` and the binary lane; incremental per-pass opt-in via a ClassVar so
>   you harden one schema at a time, never a big-bang flip. Touches the `structured_call` CORE
>   (`_otr_structured_call.py`) -> a careful dedicated coder session, NOT auto-built. Problem framing:
>   `docs/2026-06-25-schema-adherence/PROBLEM_STATEMENT.md`.
> - **LEVER 2 = BINARY DECOMPOSITION -- CONVERGED but GATED (a thin lane on top of Lever 1).** Operator
>   instinct ("LLMs are reliable at binary decisions like split A dialogue | B stage-direction -- can we use
>   that to keep the ledger intact?") is SOUND; a 1-round addendum (anchor + all 3 panel converged, ~$0.13)
>   pruned + sharpened it: `docs/2026-06-25-schema-adherence/binary/pass01_plan.md` (+ `pass01_judgment.md`).
>   SCOPE = ONE classifier only (dialogue vs stage-direction); CUT the other 3 ideas (edit/no-op,
>   speaker-membership, normalize_length -- already handled / wrong domain / O(N) latency trap). Reframes:
>   per-SPAN not per-line (mixed lines can't be whole-line classified); a NEW explicit `HIT|CLEAN|ABSTAIN`
>   tri-state in the PURE `_otr_line_hygiene` layer (today's `False`/empty conflates clean vs uncertain);
>   `binary_decide` lives in the LLM layer (NOT pure hygiene) as a thin wrapper over the Lever-1 core with a
>   1-field `Literal["A","B"]` schema (A/B not yes/no -> refusal-safe); strict single-decisive-token parse ->
>   else None -> exact deterministic fallback; SHADOW-MODE first (log verdict vs deterministic outcome,
>   mutate nothing) until an offline fixture suite proves accuracy across local + >=1 remote.
>   **BUILD GATES before writing the lane: G1 = MEASURE the abstain residual** (an offline, no-GPU count --
>   what fraction of spans the existing two-tier deterministic detector `split_stage_business` +
>   `detect_stage_business_for_reroll` already handles; if ~0, the lane is unnecessary -> DROP it);
>   **G2 = byte-identity of abstain** (`segment_double_quotes` folds curly->straight, so "unchanged" isn't
>   literally byte-identical -- retain the original string + golden test). G1 is the cheap first move.
> **ORDER FOR THE CODER:** build Lever-1 pass04 C0-C6 first (each chunk: full suite + Bug Bible green vs the
> 5 pre-existing `267a53e` workflow-pin fails; strict-first byte-identity golden on the local happy path;
> `test_audio_byte_identical` green; commit AND push per green chunk). THEN run G1; build the binary lane
> ONLY if G1 shows a non-trivial residual, gated on G2. NO workflow-JSON change (env/code only). prod/main +
> tags GATED.
>
> **>>> PRIOR STEP -- 2026-06-25 (ANNOUNCER REDESIGN + KILL 2 + KILL 4 ALL BUILT + SHIPPED + PUSHED +
> LIVE RE-SOAK PASSED ON BOTH LOCAL WRITERS. origin/v2.0-alpha HEAD `b7bf7fc3`; now `89e9f8bf` w/ re-soak docs.)**
> Built C1-C4 from `docs/2026-06-24-announcer-refine/roundtable/pass04_plan.md` + `CODE_MAP.md` +
> `coda-segue/roundtable/pass03_plan.md`, all behind the `story_scaffold` widget (byte-identical off; NO
> workflow-JSON change); +65 unit tests; per chunk full suite green vs the 5 pre-existing `267a53e`
> workflow-pin fails + Bug Bible 16/7/3; commit AND push per green chunk.
> - C1 `14704f98` KILL 2 StoryContract: new frozen `StoryContract` + `build_story_contract` in
>   `_otr_style_catalog` (cast_seed-keyed select_style from script_brief/news_seed); `grammar ==
>   render_style_grammar(slug)` -- its first+only caller (the literal "zero callers" fix).
>   `OutlineRequest.style_grammar`/`story_engine` (default ""); macro prompt renders the grammar block
>   (sound_world lives HERE only), phase+beat thread story_engine. Writer hoists `_style_grammar_on` to one
>   gate, builds the contract pre-outline, swaps the late select_style source to the contract under flag,
>   stamps `meta.story_contract`.
> - C2 `69125683` announcer OPEN by INPUT STARVATION: new frozen `SafeOpenBrief` + `_ANNOUNCER_INTRO_SYSTEM_SAFE`
>   + `fallback_safe_open`; `compose_announcer_intro(...,story_scaffold,safe_open_brief)` builds the open from
>   the safe brief ONLY (script_brief never read -> the outcome can't leak). Writer captures the SafeOpenBrief
>   after generate_outline + BEFORE build_sq_data mutates the setup beat; `open_safe_fallback` telemetry.
> - C3 `e58fba40` dynamic NEWS CODA: new `compose_news_coda` (LLM writes a short bridge from `outline.premise`
>   + the safe intro tone, never the outcome; real `news_close_brief` appended deterministically;
>   `validate_news_coda_bridge`; sha256(cast_seed) rotating-pool floor). Writer early-branches to it under
>   flag+brief; else `compose_announcer_outro` UNTOUCHED (off byte-identical) + `dataclasses.replace` marker.
>   Supersedes pass04 STEP F; drops the climax-line decoupling.
> - C4 `b7bf7fc3` KILL 4: role-keyed enrichment (setup/pressure/personal_stake + every climax class;
>   consequence omitted) + truncation reserve/clamp (`max(0,...)`) so a long intent can't cut the climax
>   clause; the 2 prior roles stay byte-identical.
> **LIVE RE-SOAK PASS (2026-06-25, canonical JSON, LTX lane, toggle ON per-prompt):** mistral ON = full
> end-to-end (contract retirement_home_ghost_story/bittersweet_parting; safe-open no-spoiler;
> `open_safe_fallback=False`; coda delivered the real comet news via the floor; `ungrounded_crisis 0/149`;
> freeze frozen_with_doctor_edits; **`audio_byte_identical OK`**; OBS final
> `signal_lost_comets_trail_..._final.mp4` 51.8 MB; 25:25). gemma ON (after pulling its Ollama model) =
> contract final_message_before_silence/quiet_acceptance; safe-open ok; **dynamic coda bridge VALIDATED**
> (`news_coda_fallback=False`); `ungrounded_crisis 0/181`; froze clean. Full results:
> `docs/2026-06-24-announcer-refine/RESOAK_RESULTS.md`. Minor PRE-EXISTING notes: a `"Central object, if
> useful."` artifact leaks from the news-brief central-object injection into the coda's appended fact (fix in
> the brief builder, not the coda); gemma prose stays techno-tense (model ceiling). KILL 3 (climax POSITION)
> still DEFERRED. prod/main + tags GATED.
>
> **>>> PRIOR STEP -- 2026-06-24 (ANNOUNCER/KILL-2 DESIGN ROUNDTABLE CONVERGED -> HAND TO A CODER WINDOW.
> Docs only this session; NOTHING coded yet. origin/v2.0-alpha HEAD `b717980d` == local.)**
> The refine-before-code roundtable ran LIVE, 4 rounds (R1 arc -> R2 implementability -> R3 wiring -> R4
> convergence), panel GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, Claude grounded judge+panelist; total spend
> ~$0.51; CONVERGED at R4 (all 3 verdicts yes-with-fixes = spec-precision only, no new architecture).
> **BUILD-READY TICKET: `docs/2026-06-24-announcer-refine/roundtable/pass04_plan.md`** (self-contained;
> pass00->pass04 trail + per-round judgments + claude_anchor + raw panel reviews alongside) + **`CODE_MAP.md`**
> in the same dir (exact file:line + GREP ANCHORS for every edit point + reuse/new-symbol inventory -- the
> coder edits by anchor, no reasoning needed; lines are as of HEAD b717980d, grep the anchor if drifted).
> **THE BUILD (all behind `story_scaffold`, byte-identical off, NO workflow-JSON change; 4 commits):**
> - C1 KILL 2 StoryContract: build pre-outline (cast_seed-keyed), inject the style GRAMMAR at the MACRO prompt
>   (CONSUMES render_style_grammar -> fixes its zero-callers) + story_engine at phase/beat; carry
>   `OutlineRequest.style_grammar`/`story_engine`; REPLACE (don't delete) the late select_style @ :3224 with
>   contract.slug under flag; meta.story_contract dict. Scoped HONESTLY as a STRUCTURAL steer (sound_world
>   stays OUT of dialogue line prompts -- stage-direction-leak risk; per-line teeth = the existing conflict_object).
> - C2 ANNOUNCER OPEN: deterministic no-spoiler by INPUT STARVATION -- sever script_brief; build a SafeOpenBrief
>   (setting/time_of_day/opening_status_quo/cast/era) CAPTURED right after generate_outline + BEFORE build_sq_data
>   mutates the setup-beat intent (the R3 sequence-bug catch); fallback never reads script_brief. (Spoiler belt
>   DEFERRED -- starvation is the guarantee.)
> - C3 NEWS CODA (REFINED 2026-06-24 by a 3-round coda-segue roundtable, ~$0.30, CONVERGED): a DYNAMIC
>   news-aware segue, NOT a fixed tag (operator call). New `compose_news_coda`: the LLM writes ONLY a short
>   BRIDGE clause (specific to tonight's tale via `outline.premise` + the safe `intro_text`, NEVER the
>   outcome/news facts); the real `news_close_brief` is APPENDED deterministically -> the weak model can't
>   blend (it never writes the fact). sha256(cast_seed) rotating-pool fallback floor; coda-specific
>   `validate_news_coda_bridge` (length cap + generic-opener blacklist); empty-brief -> normal fictional outro
>   (never fabricate). `compose_announcer_outro` UNTOUCHED (off-path byte-identical). REPLACES the fixed
>   `NEWS_CODA_LEAD_IN`; DROPS the climax-line decoupling (coda never touches the fictional climax). Spec:
>   `docs/2026-06-24-announcer-refine/coda-segue/roundtable/pass03_plan.md`.
> - C4 KILL 4: role-keyed enrichment (setup/pressure/personal_stake + every CLIMAX_CLASS_ROLES member;
>   consequence omitted, not stubbed) + the truncation reserve/clamp (max(0,...) -- fixes a negative-slice bug).
> Per chunk: full suite + Bug Bible green vs the 5 pre-existing 267a53e fails; run()-level OFF-flag golden
> (open/outro/ledger-meta) + test_audio_byte_identical green; commit AND push to v2.0-alpha. After C1-C4 ->
> LIVE re-soak (gemma + mistral) via the story_scaffold toggle. KILL 3 (climax POSITION) still DEFERRED.
> 10-item verify-at-build checklist in pass04_plan.md. OPERATOR creative call (not a blocker): the
> NEWS_CODA_LEAD_IN wording. prod/main + tags GATED.
>
> **>>> PRIOR STEP -- 2026-06-24 (REFINE-BEFORE-CODE: a design /roundtable in a FRESH window on the
> announcer redesign + news-coda + the KILL-2 approach, THEN code. origin/v2.0-alpha HEAD `47189349`.)**
> KILL 1 is SHIPPED + re-soaked (the 6-leg story_scaffold bake-off landed 6 OBS episodes; ON wins -- 0
> ungrounded machinery + varied endings vs OFF console-standoff; verdict in
> `docs/2026-06-24-assumption-audit/roundtable/` + the bake-off doc). Operator wants the creative/design
> refinements hardened via a roundtable BEFORE coding KILL 2. Run the `/roundtable` skill (standard repo
> defaults: LIVE, 4 rounds, GPT+Gemini+DeepSeek panel, Claude grounded judge+panelist) grounded against the
> real files, on: **`docs/2026-06-24-assumption-audit/ANNOUNCER_DESIGN.md`** (3 announcer jobs: scene-setting
> OPEN no-spoilers / character CLOSE drama / NEWS CODA teaching beat) + **`STORY_ENGINE_ROADMAP.md`** + the
> KILL-2 StoryContract section of **`roundtable/pass04_plan.md`**. REFINEMENT ASKS: (1) news-coda lead-in
> phrasing (operator leans a light fixed tag like "The real story:"); (2) OPEN structure + length (operator
> leans a tight 1-2 sentence cold open: time/place/characters/where-they-are-now, NO climax/outcome/twist);
> (3) the KILL-2 injection approach -- does rendering sound_world/story_engine into every body beat actually
> move the weak local writer, or is it the same single-prior trap that needs a deterministic gate too?;
> (4) KILL 3 = SPINE-DRIVEN climax position (last beat is fine when the spine calls for it; remove the FORCE,
> don't mandate a move). Operator thesis: the show TEACHES -- drama is delivery, the NEWS is the payload,
> explicit at the end. GROUNDING WIN to confirm: the close already pulls the real news via `news_close_brief`
> -> `compose_announcer_outro`, so the coda is largely WIRED (KILL 5 reframed "suppress" -> "frame + protect
> the character climax"). After convergence -> hand to a CODER window: KILL 2 + announcer redesign + KILL 4
> (behind the story_scaffold flag, byte-identical off; suite+Bug Bible per chunk; LIVE re-soak; commit+push
> per green chunk). prod/main GATED.
>
> **>>> PRIOR STEP -- 2026-06-24 (KILL 1 BUILT + SHIPPED + PUSHED + LIVE 309w OBS SMOKE PASSED.
> origin/v2.0-alpha HEAD `adb47483` == local. NEXT = the operator-gated KILL-1 RE-SOAK; do NOT start
> KILL 2 until that re-soak is clean.)**
> KILL 1 = the deterministic BODY-OUTPUT gate -- validate the SHIPPED character line, not `beat.intent`.
> Behind the existing story-quality lever (default ON; `OTR_ENABLE_STYLE_GRAMMAR=0` = byte-identical
> kill-switch); NO workflow-JSON change. Commit `adb47483` (+28 tests, `tests/test_body_output_gate.py`):
> - `_otr_story_quality_l12`: `validate_composed_grounding(text, sq_entry, grounded, *, max_ungrounded=0,
>   require_conflict_object_on_roles=CLIMAX_CLASS_ROLES|{PRESSURE})` -> (ok, split reasons) +
>   `ungrounded_crisis_tokens` + `line_references_object`/`_content_tokens`/`_strip_possessive`;
>   `count_ungrounded_crisis` now delegates to the token list.
> - `LineRequest.grounded_nouns` (frozenset) threaded through `_otr_reroll.build_reroll_line_request`
>   (from `meta.grounded_nouns`). Writer computes the palette ONCE
>   (`premise_noun_palette(roster, news_seed, premise, *premise_texts)`), stamps `meta.grounded_nouns`,
>   runs the gate IN-LOOP after the common `cleaned` assignment (exchange 4163 + compose 4206) and BEFORE
>   `last_lines.append`/`update_line_text` (covers the use_exchange bypass); character beats only. ONE
>   guarded reroll via the existing `reroll_hint` with SPLIT hints (ungrounded_crisis -> offending tokens;
>   missing_conflict_object -> grounded object); ship-reroll-if-valid-else-original; stamps
>   `meta.story_quality.{body_gate_rerolls, body_gate_failed, body_gate_ungrounded_crisis}`,
>   `grounding_reroll_failed` on exception.
> - Composer: de-license the generic-roles line (1162) + the news-facts rider (1442) when
>   `req.conflict_object` is set; byte-identical when empty.
> ACCEPTANCE MET: full suite green vs the 5 pre-existing `267a53e` workflow-pin fails (re-verified
> pre-existing by stash+rerun -- KILL 1 touches ZERO workflow JSON); Bug Bible 16/7/3; the 4 touched .py
> AST-clean, no BOM. **LIVE 309w OBS SMOKE (canonical `otr_scifi_16gb_full.json`, fresh LTX-lane boot):
> full episode ran end-to-end in 31:09 -> OBS final
> `otr/obs/signal_lost_corks_dance_20260624_170035_silent_procgen_blended_final.mp4` (70.3 MB);
> `audio_byte_identical OK`; zero tracebacks. The body gate FIRED LIVE: style=overnight_jazz_host_mystery,
> ending_tag=revelation; body_gate_rerolls=1 (validated+shipped), body_gate_failed=10 (kept-original LOUD),
> shipped-body ungrounded_crisis density = 2 ("button" x2 across 16 char beats -- the console standoff is
> effectively gone). The 10 "failed" are mostly missing_conflict_object on the weak local mistral-nemo (the
> line is premise-grounded but does not echo the seed-keyed object) -- the model-ceiling reality the plan
> flagged (a model-capability gate is the deferred belt-and-suspenders). Server killed (selective CIM),
> :8000 FREE, VRAM ~1.4GB. Results: `docs/2026-06-24-assumption-audit/KILL1_SMOKE_RESULTS.md`.**
> **>>> ALSO SHIPPED 2026-06-24 (operator-requested): the `story_scaffold` UI toggle (`3f053ba4`).**
> A user-facing dropdown on the writer node -- auto / on / off -- the single control over the whole
> bundled scaffold. `off` = a radio drama from the news seed (base prompt only, no style catalog /
> climax-shape grammar / grounding gate -- the writer's own story); `on` = the news story shaped by one
> of the ~100 radio-drama styles; `auto` (default) = follow `OTR_ENABLE_STYLE_GRAMMAR` / its default
> (ON). Resolved via `_apply_story_scaffold_env` at the top of `run()` (sets the env so EVERY downstream
> read -- incl the outline announcer-close gate -- is consistent; auto restores the import-time baseline,
> no cross-prompt leak). Appended at writer widget slot 24 in the canonical JSON (same change); guards
> 24->25; +10 tests. This makes the scaffold-vs-freeform A/B a UI flip.
> **>>> BAKE-OFF RUN 2026-06-24 (operator-requested, autonomous): 6 FULL episodes to OBS, 3 ON vs 3 OFF,
> recipe = visualizer character beats + LTX-AV audio bookends + flux2_klein bookend images, canonical JSON,
> 200w. ALL 6 LANDED in `otr/obs`. WINNER = ON (scaffold).** ON episodes (Breath of Warning / Blade's Dawn /
> Broadcast Dilemma) all shipped **0 ungrounded crisis nouns** with varied climax styles + endings
> (bittersweet / revelation / ironic); OFF (Keys to Control / Unmasked Data / Flame Before Time) was
> variable -- "Keys to Control" collapsed to the console standoff ("Lockdown mission control... count
> three"), and every OFF closed by restating the news outcome. Both modes still `arc=uneven` (mistral-nemo
> ceiling -- the scaffold fixes sameness + grounding, not raw prose). KILL 5 (close-by-ending_tag) is still
> the gap: one ON close drifted to the news outcome. Two legs needed a recovery rerun (NOT KILL-1/toggle
> bugs): off#3 = `build_news_briefs` key_term-not-in-source hard-fail (fresh news passed +
> `news_briefs_required=False`); on#3 = freeze `needs_full_rerun` (BUG-LOCAL-276, shipped via
> `OTR_BYPASS_FREEZE_HALT=1`). Unattended-batch recipe: boot with `OTR_BYPASS_FREEZE_HALT=1` +
> `news_briefs_required=False`. Results: `docs/2026-06-24-assumption-audit/BAKEOFF_RESULTS.md`. Box reset,
> :8000 FREE.
> **>>> NEXT (operator-gated): the KILL-1 LIVE RE-SOAK (gemma + mistral, 320w) measuring crisis-noun
> density in the SHIPPED body lines (not intent), now driveable via the `story_scaffold` toggle (on vs
> off). Do NOT start KILL 2 until the re-soak is clean.**
> Then KILL 2 -> KILL 4/5 per `docs/2026-06-24-assumption-audit/roundtable/pass04_plan.md`. KILL 3
> climax-position + model-capability gate + render profiles + `_PERSONAL_COST` rows DEFERRED.
> prod/main + tags GATED.
>
> **>>> PRIOR STEP (the build target) -- 2026-06-24 (STORY-ENGINE ASSUMPTION-AUDIT CONVERGED -> BUILD the fixes in a fresh
> CODER window. Plan is grounded + build-ready; NOTHING from it is coded yet. origin/v2.0-alpha HEAD
> `0c8bd191` == local.)**
> Two live story-engine runs proved the style grammar fixes the CLIMAX + announcer CLOSE but NOT the body:
> gemma collapsed into the exact console standoff (red levers / blowing fuel cells) it targets, and the
> style label was ignored (memory-erasure-clinic style -> NASA story). A 4-round LIVE assumption-attack
> roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude grounded judge, ~$2.29, CONVERGED) found
> the two root seams + a self-correction, all grounded to file:line. **BUILD-READY plan:
> `docs/2026-06-24-assumption-audit/roundtable/pass04_plan.md`** (R1->R4 trail + judgments alongside).
> **THE BUILD (in order; all behind the existing story-quality flag => byte-identical when off; NO workflow
> JSON change):**
> - **KILL 1 (do FIRST -- the load-bearing fix): deterministic body-output gate.** Validate the SHIPPED
>   character line, not just `beat.intent`. IN-LOOP, after the common `cleaned` assignment (writer 4163
>   exchange / 4206 compose) and BEFORE `last_lines.append` (4222) -- covers the use_exchange bypass. New
>   `_otr_story_quality_l12.validate_composed_grounding(text, sq_entry, grounded, max_ungrounded=0,
>   require_conflict_object_on_roles=CLIMAX_CLASS_ROLES|{BEAT_ROLE_PRESSURE})` (object match by head-noun /
>   `_TOKEN_RE` token overlap, casefold, strip possessive). Add `grounded_nouns` to `LineRequest`, computed
>   once from `premise_noun_palette(roster, news_seed, outline.premise, *premise_texts(meta))` + thread thru
>   `_otr_reroll.build_reroll_line_request`. ONE guarded reroll; SPLIT hints (ungrounded_crisis -> offending
>   tokens only; missing_conflict_object -> the grounded conflict_object); ship-reroll-if-valid-else-original
>   deterministically; stamp `meta.story_quality.{body_gate_rerolls,body_gate_failed,body_gate_ungrounded_
>   crisis}`. De-license composer line 1162 ("mission control...fine") + 1442 ("Ground...in the news facts")
>   by style. **This is the change that stops gemma's standoff -- ship it alone first + re-soak.**
> - **KILL 2: one StoryContract, selected pre-outline, injected into the whole body.** `render_style_grammar`
>   has ZERO callers today (style never injected -> only ending_tag survives). New frozen `StoryContract` +
>   `build_story_contract` in `_otr_style_catalog`; build ONCE after cast-lock (`cast_seed` as seed) + news
>   interpretation, BEFORE `OutlineRequest`, from `script_brief or news_seed`; REUSE in F2 (delete the late
>   `select_style(outline.premise,...)`). Add style fields to `OutlineRequest` + render in
>   `_build_macro/phase/beat_user_prompt`; add to `LineRequest` + render for EVERY character beat (not just
>   climax). ADD `meta.story_contract`; DO NOT overwrite `resolved["style"]`/`meta.style`/`visual_plan.style`
>   (they feed build_news_briefs + cast) -- defer the collapse.
> - **KILL 4 + KILL 5: un-starve the body + fix the close.** build_sq_data enrich is gated to PERSONAL_STAKE/
>   IRREVERSIBLE_CHOICE (~694) -> role-keyed map for setup/pressure/personal_stake + every CLIMAX_CLASS_ROLES
>   member (CUT consequence -- unreachable); fix the 200-char truncation order. Announcer close: my C5 gate is
>   MOOT -- `compose_announcer_outro` (writer 4230) overwrites it with an F3 "State this outcome plainly"
>   branch (composer 2819); add `ending_tag` param, force resolved=False + non-resolving fallback for
>   {unresolved_final_sound,revelation,quiet_acceptance}.
> - **DEFERRED: KILL 3 climax POSITION** (breaks validator + ending_template target + outro last-line
>   assumption -- its own build after KILL 1/2), model-capability gate, render profiles, _PERSONAL_COST rows.
> ACCEPTANCE: per-chunk full suite + Bug Bible green vs the 5 pre-existing 267a53e fails; byte-identical when
> the story-quality flag is off (new fields default empty, no meta.story_contract); then a LIVE re-soak
> (gemma + mistral, 320w) confirming the body no longer collapses to machinery (crisis-noun density in the
> SHIPPED body lines, not just intent). 13-item verify-at-build checklist in pass04_plan.md. commit+push per
> green chunk; prod/main GATED.
>
> **>>> PRIOR STEP -- 2026-06-24 (STORY-GRAMMAR build: chunks 1-6 ALL SHIPPED + PUSHED. The ONLY thing
> left is the operator-gated LIVE LLM behavioral A/B soak. origin/v2.0-alpha HEAD `4c9793b2` == local.)**
> Chunks 4-6 (the WIRING) are DONE -- the style grammar is live behind `OTR_ENABLE_STYLE_GRAMMAR` (env-only,
> DEFAULT OFF, byte-identical when off, NO workflow-JSON change; bundled with `OTR_STORY_QUALITY_L12`).
> 3 commits `762b20d7` (C4) -> `e86adb59` (C5) -> `4c9793b2` (C6), +26 tests; full suite green vs the 5
> pre-existing `267a53e` workflow-pin fails (verified pre-existing by stash+rerun -- this sprint touches ZERO
> workflow JSON); `test_audio_byte_identical` GREEN; Bug Bible 16/7/3. Results doc:
> `docs/2026-06-24-ending-mode/STORY_GRAMMAR_C456_RESULTS.md`.
> - C4 `762b20d7` `_otr_line_composer`: `LineRequest.ending_template` field + `Ending:` render (only when
>   populated; empty => byte-identical golden); `build_sq_data(climax_role=)` threaded into
>   `assign_beat_roles`; `_otr_config.style_grammar_enabled()`. Inert at this commit.
> - C5 `e86adb59` WIRING live: `_otr_outline._assemble_outline` announcer close gated via a DIRECT
>   `os.environ.get("OTR_ENABLE_STYLE_GRAMMAR")` read (OFF=exact pre-grammar string; ON=non-outcome close,
>   <=200 chars, NEVER removed -> validator #7 safe). `OTR_LedgerScriptWriter` F2: when on, after
>   `generate_outline`, `slug = select_style(outline.premise, meta, cast_seed)` -> ending_tag = climax_role
>   -> `build_sq_data` (bundled w/ L12) -> climax-class beat id + `ending_template_for(slug)` -> the
>   LineRequest for THAT beat only -> `meta.story_quality {style_slug, ending_tag, final_beat_crisis_nouns}`.
>   Fails soft (climax falls back to irreversible_choice, ending template dropped; never breaks audio).
> - C6 `4c9793b2` `tests/test_story_grammar_wiring.py` (26): default-OFF byte-identity, flag-ON ending render,
>   announcer non-outcome + close-not-removed, climax_role threading, selector determinism.
> **DETERMINISTIC A/B PROVEN (no GPU; 12 premises x 250 seeds = 3000 eps):** lever OFF = irreversible_choice
> 100% (the forced console standoff); lever ON = irreversible_choice **2.1%**, **97.9% non-doomsday** (target
> >=80%), all 9 ending classes used, 98/100 styles, even spread (revelation 19% / reversal 15% / unresolved 13%
> / reconciliation 13% / bittersweet 12% / quiet_acceptance 11% / confession 9% / ironic 7%). The climax SHAPE
> is no longer uniform. select_style is sha256(cast_seed)-keyed (C7-safe).
> **>>> LIVE LLM A/B -- RUN 2026-06-24 (operator freed :8000; box reset S4 before+after). Results:
> `docs/2026-06-24-ending-mode/LIVE_AB_RESULTS.md`.** Full canonical JSON, 320w, N=2/leg (mistral+gemma);
> OFF=clean vs ON=`OTR_ENABLE_STYLE_GRAMMAR=1`+`OTR_STORY_QUALITY_L12=1`. (Reduced from N=6: full episodes run
> ~16-20 min each here -- writer/critic/reroll + per-line indextts2 + per-beat flux portraits dominate; the box
> can't do 12 full episodes in one sitting.) FINDINGS: (1) WIRING proven live end-to-end on BOTH writers
> (`story-grammar ON: style=... ending_tag=... climax_beat=b017` logs + ledger `meta.story_quality` stamp);
> OFF ledgers carry only base keys (inert). (2) ANNOUNCER CLOSE = clear win: OFF states the news outcome ("Puma
> safely in orbit / new era", "splashes down safely / new era for SpaceX") -> ON non-outcome image ("satellite
> stood tall, truth took flight"; "lights dim over the galactic bulge"). (3) FINAL BEAT: writer obeys to model
> strength -- mistral landed the REVERSAL crisply ("I've already taken steps, Rick. The truth has its own
> jammer."), gemma went oblique on RECONCILIATION (machinery-free but loose). (4) crisis-noun density did NOT
> discriminate (0 both legs -- the writers never reached for console/lever vocab even OFF on these space
> premises; the discriminators are the close + the final-beat SHAPE, not crisis count). (5) NO render bugs --
> full workflow ran clean OFF (LTX-AV bookends) and ON (visualizer -> OBS final `signal_lost_ink_stir`), zero
> tracebacks. Workflow JSON UNCHANGED (env-only).
> **>>> DEFAULT FLIPPED ON 2026-06-24 (operator: "if it makes a better story, default it on -- I don't want a
> lever people have to find and flip"). `57279156`.** `STYLE_GRAMMAR_DEFAULT=True` in `_otr_config`
> (`style_grammar_enabled()` returns the default when `OTR_ENABLE_STYLE_GRAMMAR` is unset/empty;
> `OTR_ENABLE_STYLE_GRAMMAR=0`/false/no/off is the kill-switch -> exact pre-grammar byte-identical). The outline
> announcer-close gate now reads the shared config reader (same default). Bundled L12 build runs via the writer's
> `l12_enabled() OR style_grammar_on` condition. Tests updated for default-ON + the `=0` kill-switch byte-identity.
> Suite green vs the 5 pre-existing `267a53e` fails; Bug Bible 16/7/3. **Every episode now ships with the style
> grammar (varied climax class off the console standoff + non-outcome announcer close).**
> **>>> NEW CURRENT STEP (open follow-ups):** (1) **audio-byte-identity baseline re-capture** -- the GPU-gated
> `test_audio_byte_identical` (OTR_REGRESSION_RUNTIME=1, skipped in CI) was captured pre-grammar; either
> re-capture it on the new default OR boot the regression server with `OTR_ENABLE_STYLE_GRAMMAR=0` to match the
> stored SHA. (2) Optional **wider overnight N** to harden the read (N=2/leg this session). (3) Optional:
> **strengthen the final-beat ending instruction** for the weak writer (gemma landed reconciliation loosely).
> prod/main + tags GATED.
>
> **>>> PRIOR STEP -- 2026-06-24 (OVERNIGHT CODER; STORY-ARCHITECTURE INCREMENT-1 BUILT + SMOKED.
> origin/v2.0-alpha HEAD `bbd0943f` == local; one code commit on top of the operator's 3 docs commits.)**
> Built the three CPU tickets, all DEFAULT-OFF / byte-identical, +58 tests, then proved them LIVE:
> - **T4 staging penalty** (`bbd0943f`, `_otr_story_select`): `_otr_staging_penalty` + `score_outline`/
>   `select_best_outline` optional `penalty=None` (None => byte-identical, audited all callers); env gate
>   `OTR_ENABLE_STAGING_PENALTY`; folds an on-mic-climax penalty into the best-of-N comparator.
> - **T1 pitch room** (`_otr_pitch_room.py` NEW, the PRIMARY lever): `run_pitch_room` -> 3 forcibly-divergent
>   premises (DOMAIN_PALETTE + genre + archetype seeds, seeded shuffle) -> local greenlight (frontier opt-in,
>   fail-closed) -> `dataclasses.replace(outline_req, script_brief=...)`. Gated `OTR_ENABLE_PITCH_ROOM`
>   (default OFF), wired AFTER news briefs / BEFORE generate_outline, skipped on refine sub-passes.
> - **T2 critic adapter + escalation** (`_otr_story_select` + `_otr_freeze_cascade`):
>   `critic_report_to_refine_signals` (arc_verdict/reroll_targets -> failing_axes/regeneration_hint; the
>   StoryCriticReport has NO failing_axes -- the grounding catch), `build_escalation_signal` gated
>   `OTR_ENABLE_CRITIC_ESCALATION` (default OFF => `{}` byte-identical), refine loop prefers the critic
>   adapter when a report is present (falls back to grade weakness today => byte-identical), `keep_best_index`.
>   GROUNDING CATCH: `enable_critic_escalation` was never a wired widget (Stage-7 shadow critic removed
>   2026-05-29) -> implemented as an ENV flag, no JSON change.
> **LIVE SMOKE PASS** (`docs/2026-06-23-story-architecture/SMOKE_RESULTS.md`): one episode on the REAL
> canonical JSON, LTX/HuMo-free, levers ON (`OTR_ENABLE_PITCH_ROOM=1` + `OTR_ENABLE_CRITIC_ESCALATION=1` +
> bypass). Pitch room greenlit a divergent Mars-geologist premise (3 distinct conflict types); critic
> `arc_verdict=uneven` -> adapter `failing_axes=['emotional_arc']` -> escalation `scope=episode`; full pipeline
> composed -> graded -> rendered -> **OBS final published** (`signal_lost_akiras_resolution_..._blended_final.mp4`,
> 48.6 MB), no crash. **T0 CEILING** (`docs/2026-06-23-story-architecture/CEILING_PROBE.md`): real grades from
> the live refine soak -- gemma-4-12b ~65, mistral-nemo ~42, **NONE reach B(75)**, refine lift 0. Pitch room
> fixes SAMENESS, not the prose grade. Recommendation = accept-B-relabel for the local lane + offer
> frontier-greenlight (cheap, env `OTR_ENABLE_FRONTIER_GREENLIGHT` + `OTR_GREENLIGHT_MODEL`) + frontier-writer
> only if A+ prose is paid for. **Did NOT auto-enable frontier.**
> **OPEN for the operator (morning):** flip the Increment-1 flags ON for an N=3 sameness/grade eyeball re-soak,
> then promote defaults if it looks good; **T0 frontier decision**; **T3 use_exchange** = DEFERRED (a writer
> BOOLEAN widget, not on CREATIVE_WHITELIST so headless can't patch it -- needs a dedicated single-variable
> N=3 GPU run asserting VRAM<=14.5 + zero slot drift, then a config-only JSON flip). **5 pre-existing suite
> failures** (16gb-profile / workflow-structure / audio-wiring pins) are from the 2026-06-23 HuMo-free UI-save
> `267a53e` -- verified pre-existing (stash + rerun), NOT this sprint; they need a profile/fixture re-pin
> (operator's workflow domain). Suite otherwise green; Bug Bible 16/7/3. Box reset, :8000 FREE. prod/main GATED.
>
> **>>> PRIOR STEP -- 2026-06-23 night (OPERATOR-DIRECTED; STORY-ARCHITECTURE INCREMENT-1 SPRINT,
> code-ready, hand off to a fresh OVERNIGHT CODER window). HEAD `ece57e8` == origin/v2.0-alpha.**
> Operator pivoted to an A+-story push: a fresh window ran a LIVE 4-round roundtable (GPT-5.5 +
> Gemini-3.1-pro + DeepSeek-v4-pro, Claude grounded judge, ~$0.29) that CONVERGED. The quality
> apparatus already exists + is wired (critic 5B / targeted reroll 5C / structural escalation /
> grouped-exchange / best-of-N outline / keep-best refine) -- it REMOVES flaws but cannot MANUFACTURE a
> good story because it never varies the PREMISE. Root cause (triangulated): beat-planner / premise
> sameness ("every premise = a console standoff, climax off-stage"). **Detail spec =
> `docs/2026-06-23-story-architecture/SUBAGENT_SPRINT.md`** (built from `SPEC.md`). Increment-1 tickets,
> all DEFAULT-OFF / byte-identical when off: **T0 ceiling-probe (GATE, GPU -> writes CEILING_PROBE.md,
> operator decides frontier-vs-accept-B; do NOT auto-enable frontier) -> T3 flip use_exchange (GPU N=3,
> independent) -> T4 staging penalty (`_otr_staging_penalty`, `score_outline` optional `penalty=None`
> byte-identical) -> T1 pitch-room+greenlight (`_otr_pitch_room.py`, the PRIMARY lever, gated behind
> `OTR_ENABLE_PITCH_ROOM`, `dataclasses.replace(outline_req, script_brief=...)`) -> T2 critic-axes drive
> the refine loop**. GROUNDING CATCHES baked into the sprint: conflict palette symbol = `DOMAIN_PALETTE`
> (NOT the SPEC's assumed `load_conflict_palette()`); `BEAT_ROLE_IRREVERSIBLE_CHOICE` confirmed for T4;
> **`StoryCriticReport` exposes `arc_verdict`/`reroll_targets`/`flat_lines`/`continuity_issues`/
> `render_priority`, NOT `failing_axes`/`regeneration_hint` -- T2 MUST build an adapter.** Increment 2
> (premise re-pitch / `EscalationScope.PREMISE`) + the prose->ledger parser are DEFERRED. Every ticket:
> edit the canonical JSON in the same commit, suite + Bug Bible green, commit+push per green chunk,
> default-OFF byte-identical; operator flips flags + eyeballs an N=3 re-soak in the morning. prod/main GATED.
> **FORWARD PLAN (operator 2026-06-23 night):** a couple more nights of sci-fi (original-fiction)
> Increment-1 polish, THEN pivot to the **open WRITER-ENGINE / multi-source platform** as the next MAJOR
> sprint -- seed doc `docs/2026-06-23-story-architecture/SOURCE_BANKS.md` (mirrors the video platform;
> Outline=`Beat[]` is the universal seam; 5 engines: `science_explainer` do-first [the STEM win, lightest
> -- one prompt swap; sidesteps the story-quality bar], `news`, `archive_seed`, `pd_adapt` [Poe-class PD
> adaptation -- the ceiling-buster], `frontier`/`human`). Reframes the goal from "make the original-drama
> engine A+" to "swap to engines that never had the problem"; deserves its OWN roundtable -> SPEC -> sprint.
>
> **WHERE WE ARE (this session, 2026-06-23 night):** (1) v1 refine loop SHIPPED + made **keep-best**
> (`fedbce0`) -- ships the highest-grade pass, not the last (gemma pass1=72 was being thrown away for
> pass4=65); telemetry re-saves the winner ledger LAST so the downstream latest-ledger handoff ships it.
> (2) `refine_target_grade` dropdown shipped (`a0ab962`) + CREATIVE_WHITELIST fix (`9f2bd56`). (3)
> Canonical workflow is now **HuMo-free** (`267a53e`): operator UI-save promoted -- `other_beats_video_model`
> humo_1.7B -> `visualizer (16:9)`, bookends -> `ltx_av_talk (16:9)`, `unique_per_beat`; fixes the
> no-fallbacks crash on gated HuMo so OBS publishes (106 workflow guards green, writer keeps 24 widgets).
> (4) A grade-lift validation **soak is RUNNING** (gemma+mistral rotation, refine bar=B, 200-320w,
> HuMo-free render + OBS); driver `scripts/_otr_refine_soak.py` (throwaway; monotonic timer + /history
> completion + OBS-set diff). **Auto-summary scheduled 12:20 AM** (`otr-refine-soak-summary`) reads
> `docs/2026-06-23-refine-loop/grade_lift_soak_summary.json` + resets the box. EARLY data: grades 42-72,
> mostly < B(75), revision non-monotonic -- consistent with the model-ceiling read (why T0 is the gate).
> (5) Story-arch SPEC + sprint + roundtable artifacts committed (`ece57e8`). Suite 5238/34 + Bug Bible
> green throughout. OPEN: the soak's morning summary; the overnight Increment-1 sprint.
>
> **HANDOFF -- 2026-06-23 (CODER; v1 ITERATIVE STORY-REVISION LOOP -- ALL CHUNKS 0-4 BUILT + SHIPPED +
> PUSHED + LIVE-PROVEN. origin/v2.0-alpha HEAD `a0ab962` == local.)** Built the BUILD-READY plan below
> (`docs/2026-06-23-refine-loop/roundtable/pass04_plan_FINAL.md` + the C3/C4 staging kickoff). Each chunk
> full-suite + Bug-Bible green; DEFAULT-OFF => byte-identical (single compose call, no `refine_loop` key, no
> best-of-N suppression when off). Suite 5238/34, Bug Bible 16/7/3.
> - **Chunk 0** -- fixed the SHIPPED v0 `diversity_hint` DEAD-code: Path C now renders the structural-variation
>   overlay in `_build_macro_user_prompt` + `_build_beat_user_prompt` (was only in test-only `_build_user_prompt`).
> - **Chunk 1** -- `OutlineRequest.prior_critique`/`prior_macro` + MACRO/BEAT **REVISE** overlays (revise the prior
>   story, don't regen from scratch) + `critique_to_hint` (sanitized, injection-guarded, <=200 words).
> - **Chunk 2** -- `grade_story`/`StoryGrade`/`extract_spoken_text_for_grade` (low-temp 0.1/0.0 structured grade
>   0-100 of the COMPOSED spoken text vs premise; floor-fallback never-fail).
> - **Chunks 3-4** -- the loop. `run()` stays the compose BODY (AST-guard-safe) + a GATE at top: when
>   `resolve_refine_passes(...).effective_passes >= 2` it delegates to `_refine_loop`, which RE-INVOKES
>   `self.run(_refine_active=True, _refine_prior_macro=..., _refine_prior_critique=..., _refine_forced_cast_seed=...)`
>   N times (best-of-N bypassed inside; cast_seed forced so only the STORY varies), grades each pass, and ships the
>   **LAST** revision (winner = `candidates[-1]`) -- which aligns with the downstream latest-ledger handoff (the
>   freeze/audio/video resolve the last-saved ledger, NOT the writer return). Merged `meta.story_quality.refine_loop`
>   telemetry. `resolve_refine_passes`: widget map `{C+:68,B:75,B+:80,A:90}`, env `OTR_STORY_REFINE_BAR`/`_PASSES`
>   override, remote provider => 1 pass, hard cap `_REFINE_MAX_PASSES=5`.
> - **WIDGET** -- `refine_target_grade` dropdown (Off/C+/B/B+/A, default **Off**) appended to writer INPUT_TYPES +
>   slot 23 of `workflows/otr_scifi_16gb_full.json` (BUG-LOCAL-097, same commit `a0ab962`); the 4 widget-layout
>   guards updated 23->24 (migration / order / round-trip schema + node fixtures).
> **LIVE-PROVEN (65-word FLOOR smoke, `OTR_STORY_REFINE_BAR=90 PASSES=2`):** pass0 grade=65 -> REVISE -> pass1
> grade=65 -> `cap_reached_below_bar` -> shipped the last revision -> freeze landed `frozen_with_doctor_edits`,
> ledger saved clean (NO save-failures) -> episode `signal_lost_seize_the_signal_*.mp4` (69.1 MB, 75.9 s) +
> `refine_loop` telemetry in the shipped ledger. **Build fix:** ship-last + dropped the loser-dir cleanup that
> RACED the freeze cascade (`[Ledger] save failed ...tmp` warnings). **HONEST-FLOOR NOTE:** the weak local model
> (mistral-nemo) did NOT lift the grade on revision in the smoke (65->65) -- the loop MECHANICS are proven;
> grade-LIFT is model/target-dependent (B ~75 reachable per the plan, A ~90 may never hit -> ships last revision).
> **>>> NEW CURRENT STEP = operator GRADE-LIFT validation soak (GPU):** run `OTR_STORY_REFINE_BAR=B` (or via the
> dropdown) over M episodes and measure the grade delta pass0->final + the cap-reached rate; operator-judged
> go/no-go (no implicit threshold). Default-OFF ships byte-identical; prod/main + tags GATED.
>
> **HANDOFF -- 2026-06-23 (PLANNER/ROUNDTABLE; v1 ITERATIVE STORY-REVISION LOOP -- 4-ROUND ROUNDTABLE
> CONVERGED, BUILD-READY PLAN. Docs only, NOT code -- await operator GO. HEAD unchanged `e2425b8`.)**
> Operator asked for a recursive story-refine loop ("keep improving until B+/B, never stops"). Ran a LIVE
> 4-round roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, Claude grounded judge+panelist, ~$0.52,
> R4 DeepSeek errored on reasoning-token exhaustion -- GPT+Gemini converged). CONVERGED on a DETERMINISTIC,
> local-only, default-OFF ITERATIVE REVISION loop inside `OTR_LedgerScriptWriter.run()` -- NOT best-of-N;
> NOT write-from-scratch: it REVISES the existing story (spine only when needed, seeded by the prior story +
> a graded weakness) until a target grade OR a hard cap (5), then ships keep-best. Operator LOCKED: target =
> a node dropdown Off/C+/B/B+/A (default Off, B recommended); always a REWRITE not a from-scratch regen.
> **ROUNDTABLE CAUGHT A SHIPPED v0 BUG: best-of-N `diversity_hint` is DEAD -- rendered only in
> `_build_user_prompt` (back-compat/test-only); production is Path C (`_build_macro/beat_user_prompt`), so v0
> candidates vary only by RNG seed (why the 30-word smoke tied all 3 candidates). best-of-N still functions
> mechanically; the structural STEERING is inert.** Build-ready plan (6 chunks 0-5; chunk 0 = the v0
> diversity_hint Path-C wiring fix): `docs/2026-06-23-refine-loop/roundtable/pass04_plan_FINAL.md` (+ pass00-03
> plans/judgments + OPERATOR_INPUTS.md). **NEW CURRENT STEP = operator GO to BUILD the v1 refine loop** (or do
> the standalone v0 `diversity_hint` fix first -- low-risk, makes shipped best-of-N actually steer). v1 ships
> default-OFF; an L1/L2-style grade-lift soak is post-build validation. prod/main + tags GATED.
>
> **HANDOFF -- 2026-06-23 (CODER; BEST-OF-N STRUCTURAL STORY-REFINE SELECTOR -- v0 LOCAL + OPTIONAL REMOTE,
> ALL 4 CHUNKS BUILT + SHIPPED + PUSHED. origin/v2.0-alpha HEAD `4593bc5` == local.)** Built
> `docs/2026-06-23-multipass-refine/roundtable/pass04_plan_FINAL.md` in order; each chunk full-suite + Bug-Bible
> green, ZERO `otr_scifi_16gb_full.json` change (verified), flag DEFAULT-OFF => byte-identical (single
> `generate_outline` call + no `meta.story_quality.best_of_n` key asserted):
> - **C1 `4dc631a`** -- `OutlineRequest.diversity_hint: str = ""`; `_otr_outline._build_user_prompt` renders a
>   structural-variation overlay ONLY when non-empty (empty => byte-identical prompt, asserted). Suite 5161/34.
> - **C2 `0473f67`** -- new `nodes/_otr_story_select.py`: `StoryScore` + PURE `score_outline(outline, meta, roster)`
>   on the RAW beat intents (NO `build_sq_data` -- that mutates intent + swaps the generic crisis nouns and would
>   ZERO `ungrounded_crisis_density`, the R3 catch). Metrics `ungrounded_crisis_density` / `distinct_conflict_nouns`
>   / `premise_grounding`, reusing public `count_ungrounded_crisis`/`premise_noun_palette`/`premise_texts`. 5170/34.
> - **C3 `43c7143`** -- `select_best_outline(...)` (LOCAL torch import; `sha256("{cast_seed}:outline:{i}")`-keyed
>   per-candidate seed + structural `diversity_hint` for i>=1; per-candidate try/except `OutlineFailedError`
>   LOUD+continue; keep-best `min(density asc, -distinct, -grounding, index)`; deterministic never-fail i=0
>   fallback then LOUD-fail) + `resolve_best_of_n` (flag parse + provider gate) wired into writer `run()`.
>   `OTR_STORY_BEST_OF_N` unset/0/1=off, >=2 => local max 6; remote (`openrouter:`/`comfy:`) => N=1. Selector
>   called ONLY when effective_n>=2 (else the byte-identical single path). `build_sq_data` still runs EXACTLY ONCE
>   downstream on the winner. Telemetry MERGED into `meta.story_quality.best_of_n`. Suite 5187/34.
> - **C4 `4593bc5`** -- opt-in remote best-of-N: `OTR_STORY_BEST_OF_N_ALLOW_REMOTE` (default OFF) +
>   `_REMOTE_BEST_OF_N_MAX=3` + fail-closed `remote_cost_guard` checked BEFORE the first paid call (clamp to N=1
>   on per-run token-budget breach OR worst-case >= the $20 autonomy ceiling, LOUD) + provider/per-candidate
>   `cost_usd` telemetry (OpenRouter `resolved_models_snapshot` probe). Suite 5200/34; Bug Bible 16/7/3 each chunk.
> **>>> NEW CURRENT STEP = step 5 VALIDATION soak (operator GPU, AFTER the build):** run local
> `OTR_STORY_BEST_OF_N=3` vs baseline over M episodes and measure the cross-episode SAMENESS drop
> (`meta.story_quality.ungrounded_crisis` density + distinct `conflict_object`/`type` counts). The PREREQUISITE
> go/no-go is operator-judged + WRITTEN after reviewing the soak table (no implicit threshold). If the
> outline-layer metric does NOT discriminate, escalate to the v1 post-compose "B+" grade (DEFERRED, separate
> project). Remote best-of-N stays opt-in + cost-guarded. prod/main + tags GATED.
>
> **HANDOFF -- 2026-06-23 (CODER; STORY-QUALITY LIFT L5a + L1/L2 + L3 + L4 ALL BUILT + SHIPPED + PUSHED.
> origin/v2.0-alpha HEAD `41aed49` == local.)** Built `pass04_plan_FINAL.md` in order, each chunk full-suite +
> Bug-Bible green then commit AND push, ZERO `otr_scifi_16gb_full.json` change (verified each commit), all SQ
> flags DEFAULT-OFF => byte-identical:
> - **L5a `9fd4de6`** -- trustworthy measurement: `compute_edit_cap -> max(3,min(12,ceil(voiced_beats*0.6)))`
>   (6->4, 18->11, 19->12) so dense gemma-12b prose is no longer `too_many_edits`-halted BEFORE the critic;
>   scrub telemetry now MERGES into `meta.story_quality` (setdefault/update, counts from the SAVED rows) instead
>   of a blind overwrite; downstream already tolerates a missing `story_critic_report` on terminal verdicts.
> - **L1+L2 scaffolding `6174bf8`** -- new stdlib leaf `nodes/_otr_story_quality_l12.py`: `select_domain`
>   keyword map + 13-domain conflict palette, seed-keyed `conflict_object`/`type`, whole-token crisis-noun
>   grounding (mutates `beat.intent` ONLY), `beat_role` dramatic-function sequence
>   (setup/pressure/personal_stake/irreversible_choice-LAST/consequence) + validator, deterministic fallback
>   content, `build_sq_data` entrypoint. `OTR_STORY_QUALITY_L12` env flag (default OFF). `LineRequest` gains
>   `beat_role`/`conflict_object`/`conflict_type=""`; composer DRAMATIC FRAME renders them ONLY when non-empty.
>   Writer builds the writer-side `dict[beat_id]` after the outline (never raises into the writer) + threads the
>   3 fields. Flag OFF => empty dict, no intent mutation, byte-identical prompt (asserted).
> - **L1+L2 render-on telemetry `201e080`** -- writer stamps `meta.story_quality {l12_domain, conflict_objects[],
>   conflict_types[]}` (the cross-episode SAMENESS measure); scrub stamps `ungrounded_crisis={matches,total}` over
>   the SHIPPED spoken text (gated on `meta.story_quality_l12_enabled`).
> - **L3 + L4 `41aed49`** -- L3 `OTR_COMPOSER_ACTION_STRIP`: `strip_action_marker` removes model-marked
>   `ACTION: ...` from the shipped line right after compose/polish + records a SEPARATE `action_strip:` flag;
>   gated prompt instruction added. L4 `OTR_TRANSCRIPT_SANITIZER`: `sanitize_transcript_text` strips
>   prompt-leak/director-note + balances a stray wrapper quote on FINAL text in the scrub before freeze/TTS/hash;
>   `detect_mojibake` is VERIFY-ONLY. Both audio-affecting, DEFAULT-OFF => `test_audio_byte_identical` GREEN, no
>   golden re-baseline needed yet. Suite 5155/34, Bug Bible 16/7/3.
> **LIVE 30-WORD FULL RENDER (FLOOR lane, canonical `otr_scifi_16gb_full.json`, prompt `b7e5eda3`):** the
> STORY+AUDIO path -- everything the LIFT touches -- ran CLEAN end to end with the new code: outline -> compose ->
> critic (`arc_verdict=uneven`, GRADED not halted) -> reroll -> scrub -> freeze (`frozen_with_doctor_edits`,
> reviewer=improved; the L5a edit-cap is live) -> audio master 45.16s -> episode mp4 59.2 MB
> (`signal_lost_lunar_shadows_20260623_103340`, story "Lunar Shadows", a lunar-mission premise). The OBS
> broadcast-finalization pass then raised `RenderError: shot shot_b002 engine 'humo_1.7B' ... gated_by_flag`
> (HuMo is OFF under the FLOOR boot lane + the 2026-06-16 no-fallbacks rule raises LOUD) -- ORTHOGONAL to the
> story-quality work; the obs blended_final did not land. L12/L3/L4 were OFF for this run (byte-identical
> baseline). **RE-RAN on the DEFAULT (HuMo) lane (prompt `acd1d99f`): full success end-to-end -- story
> "Lunar Descent", froze `frozen_with_doctor_edits`, audio 52.98s, episode mp4 66.4 MB, and the OBS final
> LANDED: `otr/obs/signal_lost_lunar_descent_20260623_105311_silent_procgen_blended_final.mp4` (49 MB),
> "Prompt executed in 29:01", no traceback.** Confirms the FLOOR/HuMo gate was the boot lane, not a code bug.
> Box reset, :8000 FREE.
> **ALSO THIS SESSION (operator follow-up): a 4-round LIVE roundtable on a NEW idea CONVERGED ($1.21,
> docs only, NOT pushed as code).** Operator floated a never-hard-fail story-refine loop ("is this a good
> story, 10th-grade B+? how to improve?" -> rewrite spine in line with the seed -> re-slug -> loop) run ONLY
> on local writers (free passes). GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude grounded judge hardened
> it; R3 caught real design bugs (scoring via `build_sq_data` zeroes the metric; no `episode_seed` exists; the
> RNG was never wired to the LLM). CONVERGED to a DETERMINISTIC, local-only, structural **best-of-N OUTLINE
> selector** (NOT a QA-reroll gate), explicitly GATED behind the L1/L2 measurement prerequisite that can still
> CUT it; v1 holistic post-compose "B+" loop DEFERRED. Build-ready plan:
> `docs/2026-06-23-multipass-refine/roundtable/pass04_plan_FINAL.md`.
> **>>> NEW CURRENT STEP = operator MEASUREMENT soak + golden re-baseline (GPU-gated):** (1) re-soak a small
> matrix with `OTR_STORY_QUALITY_L12=1` to measure cross-episode sameness (`meta.story_quality.ungrounded_crisis`
> + distinct `conflict_object`/`type` counts) vs the R3 baseline; (2) enable L3/L4 (`OTR_COMPOSER_ACTION_STRIP`,
> `OTR_TRANSCRIPT_SANITIZER`) WITH a deliberate `test_audio_byte_identical` golden re-baseline each; (3) to get a
> clean OBS final under a quick smoke, boot the DEFAULT (HuMo) lane, not FLOOR. DEFERRED (operator call): L5b
> gemma-12b default (bake-off), L6 best-of-N (CUT from v0). prod/main + tags GATED.
>
> **HANDOFF -- 2026-06-23 (PLANNER/SCOPING; STORY-QUALITY QUALITY-REVIEW done + a 5-pass live roundtable
> CONVERGED. Docs only, NOT pushed -- operator gates. HEAD unchanged `550679d`.)** Reviewed all 18 R3 flag-ON
> soak episodes (`docs/2026-06-23-story-quality-review/STORY_REVIEW.md`) + ran two live roundtables
> (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Grok-4.3, Claude code-grounded judge, ~$0.43): roundtable A =
> panel critique of the stories (`roundtable/passA_STORY_CRITIQUE_SYNTHESIS.md`); roundtable B = R1->R4
> improvement campaign, CONVERGED at R4 (`roundtable/pass04_plan_FINAL.md`). **VERDICT:** the defect is
> CROSS-EPISODE SAMENESS -- every premise collapses into the same "console standoff" (people fight over a
> lever/key/console while a gauge climbs + a countdown runs; the climax happens OFF-stage and the announcer
> narrates the news outcome). ALL FOUR models, independently + cold, put the ROOT CAUSE in the BEAT PLANNER
> (NOT the writer model, NOT the line composer) and ALL independently said a flag-and-reroll QA gate WON'T work
> -- the strongest confirmation of the operator's instinct. Local gemma-12b OUT-WROTE frontier grok (candidate
> (c) contradicted). **CONVERGED LEVER (build-ready, NO QA gate):** L1+L2 UPSTREAM structural core, ship together
> (Python-chosen premise-specific `conflict_object`/`conflict_type` + deterministic crisis-noun substitution;
> phase = dramatic FUNCTION via a `beat_role` sequence with an ON-STAGE climax beat) + L3 composer `ACTION:`-marker
> strip + L4 regex sanitizer + L5a-FIRST (fix the edit-cap that silently terminates+never-grades the BEST writer's
> dense prose, + the telemetry undercount). DEFERRED (operator call): L5b gemma-12b default (bake-off), L6
> best-of-N (CUT from v0 -- a select-gate that can't fix structural sameness). Operator answer =
> `docs/2026-06-23-story-quality-review/SCOPING_VERDICT.md`. **>>> NEW CURRENT STEP = operator GO to BUILD
> `pass04_plan_FINAL.md` in a coder window** (order: L5a -> L1/L2 scaffolding flag-OFF byte-identical -> render-ON
> + small re-soak to measure sameness -> L3/L4 golden re-baseline). R3 spine stays shipped + DEFAULT-ON. NO
> production code written this window. prod/main + tags GATED.
>
> **HANDOFF -- 2026-06-23 (CODER; R3 spine flipped DEFAULT-ON + soak harvested. origin/v2.0-alpha
> HEAD `550679d` == local.)** After the build (full trail in the block below), the operator read the
> overnight soak (16 episodes -> `otr/obs`, all flag-ON, written by mistral/gemma12b/grok @ 883w) and
> **chose to leave the spine ON**: `STORY_QUALITY_V2_DEFAULT` flipped to True (`550679d`); the writer now
> honors the default when `OTR_STORY_QUALITY_V2` is unset, and `OTR_STORY_QUALITY_V2=0` is the kill-switch.
> Suite 5109/34, Bug Bible 16/7/3, zero workflow-JSON change. **Honest verdict from the soak: the spine is
> STABLE + HARMLESS but NOT a measurable quality lift on weak local writers** -- L1+L7 fired 0x across all
> 17 scanned ledgers; L2 was eligible everywhere (40/40 high-tension beats had subtext) and DEMONSTRABLY
> rewrote the prompt (verified live: objective withheld + deflection injected) yet the 12B/grok writers
> IGNORED it and still produced imperative command-shouting ("Do it now!", "Initiate worldwide comms
> silence"); arc verdicts (uneven/mid_collapse/flat) match the flag-OFF baseline. Root cause = weak-writer
> prompt-adherence, NOT a wiring bug. The soak server is KILLED, :8000 FREE.
> **>>> NEW CURRENT STEP = QUALITY-REVIEW WINDOW (operator-requested): review the 16 soak episodes in
> `otr/obs` for craft, then SCOPE the REAL story-quality lever.** Stop-and-orient first (no code until GO).
> The data says the next lever must NOT be another instruction-gate the weak model can ignore -- candidates:
> (a) a **bare-imperative-flatness reroll gate** (line is a bald command with no subtext -> reroll, a HARD
> deterministic catch, not lexical-objective overlap like L1), (b) **L4 best-of-N** (generate N candidates,
> score flatness, keep the least-flat -- model-agnostic, no compliance needed), (c) lean on a **stronger
> frontier writer** for the prose (grok reasoning-low already helps; try a bigger model). Telemetry
> (`meta.story_quality`) + the obs episodes are the corpus. The R3 code stays shipped + ON. DEFERRED (R1):
> L3 oblique-premise. prod/main + tags GATED.

> **HANDOFF -- 2026-06-22 (CODER; STORY-QUALITY R3 SPINE BUILT + SHIPPED + PUSHED, then an 883-word
> OVERNIGHT SOAK launched. origin/v2.0-alpha HEAD `096ef64` == local.)** Built `pass04_plan_FINAL.md`
> in order, each chunk full-suite + Bug-Bible green then commit+push, ZERO `otr_scifi_16gb_full.json`
> change (no-drift verified every commit), flag default-OFF => byte-identical:
> - **C0 `67e229e`** -- ledger plumbing: `set_lines` preserves `compose_flags`+`arc_phase` (were
>   silently dropped); `update_line_text(compose_flags_append=)` + reroll persists minted flags; shared
>   `append_compose_flag` in `_otr_ledger_scrub`; new `_otr_config.py` (`STORY_QUALITY_V2_DEFAULT=False`,
>   `OBJECTIVE_DEFLECTION_TENSION_MIN=4`, `story_quality_v2_enabled` reader). +7 tests.
> - **L2 `30516f8`** -- authoring contract: under `meta.story_quality_v2_enabled`, a character beat with
>   `beat_tension>=4` AND subtext WITHHOLDS its literal Objective + injects the deflection directive;
>   `LineRequest.story_quality_v2_enabled`; writer stamps meta from `OTR_STORY_QUALITY_V2` env ONLY when
>   enabled (no new key when off); reroll rebuilder threads it. +13 tests.
> - **L1 `b11321f`** -- objective-literal floor: `flag_objective_literal` (NARROW content-word overlap on a
>   SHORT line) wired in `compose_line` alongside cliche/on-the-nose (composer's own <=1 reroll, NOT the
>   critic loop); stamps `objective_literal_retry`. +14 tests.
> - **L7 `302e8ca`** -- dialogue|action split (subsumes L6): `split_stage_business` extracts a leaked
>   verb-led action (balanced-quote class, dialogue guaranteed well-formed); `scrub_ledger` records
>   `action_split:{json}` on `compose_flags` + STILL runs the strip (AUGMENT, the judge-flagged risk);
>   `get_line_action` reader; idempotent; LOUD. +19 tests.
> - **telemetry + L5 `bd2a0d3`** -- `meta.story_quality {l1_rerolls,l7_splits,l7_split_failures}` (gated,
>   aggregated from compose_flags); OpenRouter frontier default `reasoning_effort=low` (re-measure HALVED
>   flatness) + `DEFAULT_OUTPUT_TOKENS_CAP` 8192->16384 as the 0-line guard; explicit env still wins. +4 tests.
> - **`096ef64`** -- fix: the L2/L7 writer meta-stamp hit `UnboundLocalError` (run() has a LOCAL `import os`
>   -> `os` is function-local; the stamp used it before that import) -- crashed EVERY episode at execution
>   (the suite doesn't run the heavy node). Fixed with a local import at the stamp. Suite 5109/34, Bug Bible 16/7/3.
> **>>> NEW CURRENT STEP = read the OVERNIGHT SOAK results (operator-launched 2026-06-22 ~23:01, 9h).** A
> `scripts/_otr_r3_overnight_launch.ps1` (gitignored) booted a headless :8000 server and `_otr_night_soak.py`
> with: **883-word episodes** (new `OTR_NIGHT_WORDS` knob), writer mix **mistral/gemma12b/grok** (new
> `OTR_NIGHT_WRITERS` knob -- gpt/deepseek slugs were `value_not_in_list` on this box), creativity tiers,
> **LTX-AV bookends + z-image-turbo character stills**, **OTR_STORY_QUALITY_V2=1** (all R3 levers ON),
> **OPENROUTER_REASONING_EFFORT=low** (grok), bark voice, bypass-freeze ON for unattended reliability. Logs:
> `docs/2026-06-22-story-quality-r3/overnight/`. Episode #0 confirmed composing clean (18-beat outline, zero
> errors). **DECISION (2026-06-23, soak read, 17 flag-ON episodes): DO NOT promote `OTR_STORY_QUALITY_V2` --
> keep it OFF (as shipped).** Evidence: L1+L7 fired 0x across all 17 eps; L2 was eligible everywhere (40/40
> high-tension beats had subtext) yet the high-tension lines are STILL imperative-flat ("Do it now!",
> "Initiate worldwide comms silence", "Locking out the self-destruct"); arc verdicts (uneven/mid_collapse/
> flat) match the flag-OFF baseline. The spine is STABLE + SAFE (15+ clean eps, 0 crashes, telemetry +
> plumbing work, byte-identical off) but inert as a quality lift. Root cause: the weakness is
> imperative-flatness / prompt-adherence -- NOT lexical-objective echo (L1's narrow matcher misses it) and
> NOT a quoted-trailing-action shape (L7 has nothing to split); L2's soft directive doesn't move a 12B
> local model off command-shouting. **NEXT (real lever, next sprint): a bare-imperative-flatness reroll gate
> (command with no subtext -> reroll, NOT lexical overlap) and/or L4 best-of-N with a flatness scorer; or
> lean on stronger frontier writers.** R3 code stays shipped (default-off). DEFERRED (R1): L3 oblique-premise,
> L4 best-of-N. prod/main + tags GATED.

> **HANDOFF -- 2026-06-22 (PLANNER; STORY-QUALITY R3 arc-spine->dialogue -- 4-ROUND ROUNDTABLE CONVERGED
> + reasoning-ON re-measure. Docs only, NOT pushed -- operator gates. HEAD unchanged `d8978da`.)**
> Drove the operator's R3 ask ("improve the LLMs that make the story") two ways at once.
> **(1) reasoning-ON re-measure (LOCAL harness `_tmp_reasoning_remeasure.py`, untracked):** the prior soak
> ran grok with `reasoning_effort=none` (the 0-line workaround) and grok was the FLATTEST (14/18). Re-ran
> grok-4.3 `reasoning_effort=low` + output cap 8192->16384 (so reasoning doesn't starve the story), 2 legs,
> no-bypass FLOOR. RESULT: flatness HALVED + CONSISTENT -- leg1 8/18 (arc uneven, 256w), leg2 6/18 (175w).
> Reasoning-on is a real near-free WIN; NOT a silver bullet (still uneven + compressed). -> land
> `OPENROUTER_REASONING_EFFORT=low` as a frontier-writer config default (with the cap bump as the 0-line guard).
> **(2) 4-round roundtable (Claude code-grounded judge + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, ~$0.534):**
> hardened a model-agnostic dialogue-craft spine. Artifacts `docs/2026-06-22-story-quality-r3/roundtable/`
> (pass00..pass04 + pass0N_judgment + per-model reviews). The judge code-VERIFIED two silent data-loss bugs the
> panel found (`set_lines` drops `compose_flags`/`arc_phase` 967-984; `update_line_text` no-flag-path 892-918 ->
> targeted-reroll drops row flags) and corrected a corrupting pass03 design (L7 must AUGMENT the scrub, not
> REPLACE it).
> **>>> NEW CURRENT STEP = BUILD `docs/2026-06-22-story-quality-r3/roundtable/pass04_plan_FINAL.md` in a SEPARATE
> coder window.** 3-lever spine behind a default-OFF `meta.story_quality_v2_enabled` (flag off = byte-identical):
>   - **C0 plumbing (gates all):** `set_lines` preserve `compose_flags`+`arc_phase`; `update_line_text(...,
>     compose_flags_append=)`; reroll passes the appended flags; shared `_append_compose_flag`.
>   - **L2 authoring contract:** in `_otr_line_composer._build_user_prompt`, gate (character AND beat_tension>=4
>     AND beat_subtext) -> SUPPRESS the `Objective:` emit + inject "the line is the deflection; say what they say
>     INSTEAD" (same gate in `build_reroll_line_request`).
>   - **L1 objective-literal gate:** `flag_objective_literal` at composer 2061-2063 (Tier-2, <=1 recompose, NOT
>     the critic loop); append `objective_literal_retry` to `LineResult.compose_flags` (composer is a pure leaf).
>   - **L7 dialogue|action split (operator's idea; subsumes L6):** in `scrub_ledger` (raw dict; read flag from
>     `led.meta`), run `split_stage_business(text)->(dialogue,action,reason)` BEFORE the existing strip, store
>     `action_split:{json}` on the row `compose_flags`, then STILL run the strip + preserve its accounting. New
>     episodes only via the flag; `get_line_action` reader; LOUD on failure.
>   - **telemetry** (gated, aggregate-from-flags) + **post-build LIVE craft smoke** (flag-on vs off on local
>     mistral: L2 indirection must NOT raise the critic's incoherence count -- the one risk tests can't catch).
> Build order C0 -> L2 -> L1 -> L7 -> telemetry, suite+Bug-Bible+push per chunk; ZERO `otr_scifi_16gb_full.json`
> change (no-drift assert). DEFERRED (R1): L3 oblique-premise (operator creative call), L4 best-of-N (needs a
> trusted flatness scorer). **BOX:** a reasoning-low FLOOR server is RESIDENT on :8000 (re-measure leg-2
> finishing) -- reset per CLAUDE.md S4 before any headless build run. prod/main + any tag GATED.
>
> **HANDOFF -- 2026-06-22 (CODER; STORY-QUALITY LIFT D1+D3+D2 SHIPPED + LIVE-VALIDATED + DIAGNOSTIC SOAK
> ANALYZED. origin/v2.0-alpha HEAD `e052812` for code/docs; soak driver is LOCAL/untracked.)**
> THIS SESSION, in order: built D1+D3+D2 (`6031b97`/`2e8597f`/`a37cc2d`), live no-bypass re-smoke caught +
> fixed a D3 over-coercion (`ffe2324`: announcer stored as a cast slot `name=ANNOUNCER` must be excluded
> from cast_ids), then ran a 3h NO-BYPASS writer-rotation DIAGNOSTIC SOAK (620w, all-visualizer FLOOR,
> indextts2) on the canonical JSON, rotating mistral / gemma-4-E2B / E4B / 12b / grok (technical=local
> mistral). **RESULT: D1/D3/D2 hold on EVERY writer -- 0 stage-direction leaks, 0 role coercions, 0
> bad-role rows, 0 critic stance issues; freeze gate shipped every completed episode no-bypass.** Full
> analysis: `docs/2026-06-22-story-quality-lift/SOAK_STORY_ANALYSIS.md` (12 episodes, ground-truth
> per-writer attribution from each ledger's meta).
> **>>> NEW CURRENT STEP = STORY-QUALITY R3 (writer CRAFT, the operator's next ask -- "improve the LLMs
> that make the story").** The soak's #1 universal weakness: EVERY writer collapses to terse
> imperative command-shouting under pressure ("Override the protocols!", "Lockdown now!", "Transmit the
> coordinates!") -- that is what the critic flat_lines flag (grok 14/18, mistral 7/18). Proposed levers
> (build-ready detail in SOAK_STORY_ANALYSIS.md sec D): (1) **imperative-flatness reroll gate** (bare
> command w/ no subtext -> reroll "play the pressure indirectly"; pairs with the existing on-the-nose/
> cliche gates; model-agnostic, lifts only the weak end -- E2B's lone `strong` arc proves good lines
> survive); (2) **extend D1 to narrative-sentence-in-dialogue** (gemma-12b leaked full 3rd-person
> narration into character lines -- "a jagged cry tears from her throat as she claws..." -- a class
> distinct from the verb-led-after-quote D1 already covers); (3) **length adherence** (writers compress
> to 61-344w vs the 620 target); (4) **grok reasoning tuning** (try `reasoning_effort=low` vs none;
> no-reasoning may be flattening it); (5) **E4B robustness** (one `dramatic_state_source=fallback` + a
> caps cast-name leaked mid-dialogue). RECOMMEND a roundtable to converge R3 before building (per CLAUDE.md).
> **BOX:** soak STOPPED, :8000 free. Soak harness `_tmp_gemma_diag_loop.py` (LOCAL, untracked; 75-min
> wall + queue-idle serialization) is reusable for the R3 re-measure. prod/main + any tag GATED.

> **NO-BYPASS RE-SMOKE VALIDATION -- 2026-06-22 (CODER, headless, operator-authorized). origin/v2.0-alpha
> HEAD `ffe2324`.** Reset box (S4) -> booted a FRESH FLOOR server WITHOUT `OTR_BYPASS_FREEZE_HALT` -> ran
> `_otr_combo_soak.py` on the REAL `otr_scifi_16gb_full.json` (FLOOR visualizers, 3 chars, bark char-voice).
> **The no-bypass freeze gate ran for real and FROZE** (critic + scoped reroll loop executed; verdict
> `frozen_with_doctor_edits`, `cleanup_locked=True` -- no false halt). RESULTS:
> - **D3 PROVEN LIVE + a real bug caught & fixed.** Run 1 surfaced a D3 over-coercion: the announcer is often
>   stored as an ordinary cast slot (`char_id=c01, name=ANNOUNCER, kokoro`), not the `"announcer"` sentinel, so
>   `cast_ids_from_ledger` wrongly counted it -> the pre-freeze sweep re-roled the announcer intro to character
>   -> routed to the bark char engine -> `EngineUnusable` crash. FIXED at root (`ffe2324`): exclude any cast
>   row whose name is ANNOUNCER (mirrors the reviewer roster convention) + 2 regression tests. Run 2 (fixed):
>   the announcer intro/outro stayed `announcer` (kokoro), AND a genuinely mis-stamped character (`b004` c02
>   flagged expected=announcer) was correctly REJECTED + coerced to character by the reviewer guard. Zero
>   announcer-on-a-non-announcer-cast-id rows; zero announcer wrongly coerced.
> - **D1 PROVEN:** every spoken line in both episodes was clean -- no trailing/embedded/undelimited stage
>   direction reached the frozen ledger text; the floor left legitimate narration alone.
> - **D2 PROVEN:** `meta.dramatic_state.character_b_wants` populated; `StanceIssue` telemetry field present +
>   empty on a coherent short arc (correct); the beat-prompt stance rider is live.
> - **Audio rendered cleanly** in run 2 (bark chars got valid `v2/*` presets; no crash); episode
>   `signal_lost_experiment_in_unity_20260622_171830` rendered to a 67 MB mp4 + finished its visualizer pass.
> Full suite **5073 pass/34 skip**, Bug Bible 16/7/3. **BOX STATE: a fresh no-bypass FLOOR server is RESIDENT
> on :8000 finishing the validation episode -> obs (reset per S4 before the next headless run).** Logs:
> `docs/2026-06-22-story-quality-lift/nobypass_server.log` + `nobypass_smoke2.log`.

> **HANDOFF -- 2026-06-22 (CODER; STORY-QUALITY LIFT D1 + D3 + D2 ALL SHIPPED + PUSHED. origin/v2.0-alpha
> HEAD `a37cc2d` == local.) Built `docs/2026-06-22-story-quality-lift/roundtable/pass04_plan_FINAL.md` in
> dependency order; each chunk full suite + Bug Bible green, then commit AND push. ZERO workflow-JSON change
> across all three (hash-verified each commit); ledger schema `l3-2026-05-14` FIXED (per-line signals ride
> `compose_flags`); audio byte-identical (no in-scope audio change); UTF-8 no BOM; AST-parsed; HEAD==origin
> each push.**
> - **D1 `6031b97` -- bare stage-direction leak (after/between/without quotes).** `_otr_line_hygiene`:
>   shared `segment_double_quotes` (curly->straight normalize, odd-count = unbalanced), `is_third_person_action_clause`
>   (rejects ONLY 1st/2nd-person; verb-led), extended `_NARRATION_VERBS` (adjusts/clutches/taps/tightens/
>   overrides/dances/dancing), `detect_stage_business_for_reroll` (Tier-2, reason codes), `strip_quote_anchored_stage_direction`
>   (Tier-3 floor, balanced-quote class only, aborts on malformed). Tier-1 prompt strengthen at
>   `_otr_line_composer._build_user_prompt`; Tier-2 reroll moved INTO `compose_line_draft` (ONE guard, <=1
>   reroll/line) -- old `compose_line` bare-stage block now S3-only; Tier-3 floor wired in
>   `_otr_ledger_scrub._strip_stage_directions` (order: delimited -> quote-anchored -> leading-bare) + per-line
>   `compose_flags` breadcrumb `stage_dir_stripped:<reason>` + byte-identical GOLDEN no-op gate over a new
>   `tests/fixtures/clean_strong_ledger.json`. Corpus VERIFIED: b005/b010/b012 floor-stripped well-formed +
>   idempotent; b015/b017 left for Tier-2 (detector flags them); zero false positives on the negatives. +22 tests.
> - **D3 `2e8597f` -- b011 role mis-stamp coercion.** `production_ledger.coerce_speaker_role_for_char_id` +
>   `cast_ids_from_ledger` (cast ids minus announcer/music/sfx sentinels). Applied at (a) the reviewer
>   `role_mismatch` repair guard `_otr_ledger_reviewer.py` (REJECT expected="announcer" on a cast char_id --
>   the culprit), (b) `set_lines`, (c) the MANDATORY pre-freeze SWEEP in `run_freeze_cascade` (final mutation
>   step, after the reviewer/reroll/render-plan, BEFORE Phase 7 role-dependent readiness + Phase 10 freeze --
>   note: CastLock runs DOWNSTREAM of the freeze, so there is no cast_lock call IN the cascade; the reviewer is
>   the real mutation phase, per VERIFY item 1). Audit: `compose_flags` breadcrumb + `meta["role_coercions"]`;
>   CI invariant gated `OTR_TEST_MODE` (char_id in cast_ids => character; announcer role never on a cast id).
>   COERCE-NEVER-CRASH. +12 tests.
> - **D2 `a37cc2d` -- antagonist stance (generation lever + telemetry, auto-repair CUT).** R3 grounding
>   confirmed the DramaticState is derived AFTER the outline (writer L2865 vs outline L2684), so the lever is a
>   JSON-free stance-consistency rider in `_otr_outline._build_beat_user_prompt` (no OutlineRequest field -> no
>   dead code, no risky reorder). `StanceIssue` added to the critic report as TELEMETRY ONLY (defaulted, round-
>   trips; NOT in `FailedDimension`, NOT a `RerollTarget`, no freeze gate) + critic system-prompt SECTION 7.
>   Telemetry-does-not-gate proven via a cascade test (verdict + reroll unchanged vs clean). +8 tests.
> **D4 (abrupt UN escalation) = OUT OF SCOPE (R1 unanimous cut). Full suite 5071 passed/34 skipped; Bug Bible
> 16/7/3.**
> **>>> NEXT (CURRENT STEP) -- OPERATOR-GATED END-TO-END VALIDATION:** run the manual no-bypass BASELINE
> re-smoke on the REAL `otr_scifi_16gb_full.json` WITHOUT `OTR_BYPASS_FREEZE_HALT` (it resets the resident
> FLOOR smoke server on :8000 / interrupts OBS, so the OPERATOR triggers it -- CLAUDE.md S4 reset first). Watch
> for: zero stage-direction leak in spoken text (the freeze floor + Tier-2 reroll), zero `announcer`-on-cast-
> char-id rows (the coercion sweep), and `meta.stance_issues` populated when the antagonist flip-flops. Still
> open (separate, unchanged): the cloner golden-recapture decision + VC build-item 5 (robustness acceptance
> test). prod/main + any tag GATED.

> **ROUNDTABLE -- 2026-06-22 (PLANNER; STORY-QUALITY LIFT 4-round roundtable CONVERGED; docs only, NOT
> pushed -- operator gates). Panel = GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude code-grounded
> anchor/judge; ~$0.48 total. BUILD-READY coder kickoff: `docs/2026-06-22-story-quality-lift/roundtable/
> pass04_plan_FINAL.md` (R1 arc -> R2 coding -> R3 wiring -> R4 spec-lock; pass0N_judgment.md trail; raw
> reviews in pass0N/).** Grounded against the real "Chandra's Echo" frozen ledger + code seams -> THREE
> buildable defects + one cut, ZERO workflow-JSON change, schema FIXED (per-line signals ride
> `compose_flags` -- there is NO per-line meta dict), audio byte-identical, CI asserts gated OTR_TEST_MODE:
> - **D1 (TOP) bare stage-direction leak** (b005/b010/b012 trailing-after-quote, b015 embedded-malformed,
>   b017 undelimited): shared double-quote helper (normalize curly->straight, odd-count abort) + extended
>   `_NARRATION_VERBS` (adjusts/clutches/taps/tightens/overrides/dances) + `is_third_person_action_clause`
>   (rejects ONLY 1st/2nd-person; 3rd-person permitted) wired Tier1 STRENGTHEN (`_otr_line_composer._build_user_prompt`
>   1307-1315) / Tier2 reroll IN `compose_line_draft` (not the too-late compose_line site) / Tier3
>   deterministic floor (`_otr_ledger_scrub._strip_stage_directions`, balanced-quote class only; floor cannot
>   route back to reroll) + a byte-identical golden no-op gate.
> - **D3 b011 role mis-stamp** (char_id=c02 cast char + speaker_role=announcer): `coerce_speaker_role_for_char_id`
>   at the role_mismatch repair guard (the culprit) + set_lines + a MANDATORY pre-freeze sweep (final cascade
>   mutation step, after cast_lock, before freeze hash; cast_ids = ledger.cast.keys() - announcer/music
>   sentinels); audit via compose_flags + meta["role_coercions"]. NOT at init (char_id derived FROM role there).
> - **D2 antagonist stance** (c02 Manfred reverses w/o a turn): GENERATION lever in `_otr_outline._build_beat_user_prompt`
>   (reference the pinned `DramaticState.character_b_wants`) + critic `StanceIssue` TELEMETRY ONLY. Auto-repair
>   via needs_full_rerun CUT -- R3 PROVED no cross-run channel survives a rerun with JSON frozen (writer is
>   upstream + ignores the verdict; new_ledger wipes meta; regeneration_hint read by nobody).
> - **D4 abrupt UN escalation = OUT OF SCOPE** (R1 unanimous cut; symptom of D2 + weak model).
> **>>> NEXT = operator GO to BUILD in a SEPARATE coder window: order = manual no-bypass baseline re-smoke ->
> D1 -> D3 -> D2; then re-smoke WITHOUT OTR_BYPASS_FREEZE_HALT.** No production code written this session
> (planner window). HEAD unchanged `223877a`; box still has the FLOOR smoke server resident on :8000 (reset
> per CLAUDE.md S4 before any headless run).

> **HANDOFF -- 2026-06-22 (CODER; VC chunks 2-4 SHIPPED + LIVE-SMOKE VERIFIED + PUSHED. origin/v2.0-alpha
> HEAD `4e3c691` == local -- operator cleared the push gate; chunks 2-4 + smoke docs are on origin.)**
> Built + green chunks 2-4 (details in the block below), then ran a LIVE headless smoke of the canonical
> `otr_scifi_16gb_full.json` (FLOOR/all-visualizer, local mistral-nemo, 320 words, 3 chars, indextts2,
> OTR_BYPASS_FREEZE_HALT=1) -> episode "Chandra's Echo". **VC VERIFIED LIVE:** `meta.cast_voice_slots`
> stamped (timbre/age/role); the hybrid voice-fit PROPOSED + ACCEPTED real voice_ref_ids (c03 vz_donor_glenn,
> c04 vz_donor_marshal_indian; c02 gender=other fell closed `no_cards`); indextts2 rendered all lines, no
> crash, deterministic. Audio master 135.5s OK. **STORY GRADE = C+ (~6/10)** (floor for local mistral +
> bypass, not the ceiling).
> **>>> NEW CURRENT STEP (operator-directed): STORY-QUALITY LIFT -- get the C+ up, driven by the live-smoke
> evidence in `docs/2026-06-22-voice-casting-arch/SMOKE_CHANDRAS_ECHO_FINDINGS.md`.** TOP FIX = the
> stage-direction LEAK into spoken text (trailing/embedded directions AFTER a closing quote survive the
> pre-freeze scrub -> indextts2 SPEAKS them; concrete lines b005/b010/b012/b015/b017 captured as the corpus).
> Extend the detector + freeze floor + compose-line reroll to this post-quote pattern. Then re-smoke WITHOUT
> the bypass (the STORY+CAST FIX STEPs 1-6 shipped today were meant to make the freeze gate trustworthy) to
> see the critic actually gate/reroll. Secondary: incoherent antagonist arc (Manfred flip-flops), a b011
> character-line mis-stamped `announcer`, abrupt UN escalation. **STILL OPEN (separate):** the cloner
> golden-recapture decision (Chunk 3 timbre-matching + hybrid voice-fit re-baseline the indextts2 audio
> golden -- operator gates the recapture); VC build-item 5 (robustness acceptance test + diagnostic count).
> **BOX STATE:** a FLOOR smoke server (my new code + OTR_BYPASS_FREEZE_HALT=1) is RESIDENT on :8000 -- the
> next headless run RESETS first (CLAUDE.md S4, selective CIM kill).

> **LATEST SESSION -- 2026-06-22 (CODER; VC CHUNKS 2-4 SHIPPED -- committed LOCAL, NOT pushed
> (operator gates the push). HEAD `eb43856`, 3 commits ahead of origin/v2.0-alpha `620bdd7`. Full suite
> 5029 passed/34 skipped (was 5001/34; +28 new tests), Bug Bible 16/7/3, audio-byte-identical structural
> green (runtime gate is operator-gated). ZERO workflow-JSON change across all three chunks.)** The
> voice-casting architecture is now BUILT end to end (pass01_plan.md build-order items 1/2/4):
>   - **Chunk 2 `995206e` -- two-lane identity.** `_otr_voice_bank.bark_preset_gender` +
>     `same_gender_voice_ref_for_preset` (deterministic `v2/en_speaker_* -> same-gender clone
>     voice_ref_id`, gender read from cast_pools.VOICE_PROFILES). STEP-3 fail-soft repair now also stamps
>     a same-gender clone identity on a REPAIRED character row (scoped to the repair path -> normal golden
>     untouched). `CastingResponse.voice_preset` cap 80 -> 255 for verbose two-lane ids. +12 tests.
>   - **Chunk 3 `d6aa9ea` -- meta.cast_voice_slots.** `EnsembleSlot` gains `age_band` (default "adult").
>     `lock_cast` stamps `meta.cast_voice_slots[char_id]={gender,timbre[list],role,age_band,
>     speech_signature,description_digest(sha1[:12])}`; the writer carries it into the frozen ledger;
>     `CastLock._auto_registry` reads timbre/age FROM the slot (not just gender) into `assign_voice_for_slot`.
>     **BYTE-IDENTITY NOTE:** the canonical workflow runs CastLock at `auto_registry`, so feeding real
>     timbre into the scorer is a DETERMINISTIC re-baseline of the cloner (indextts2) audio golden ->
>     operator golden recapture needed (already flagged). Bark replay path is UNCHANGED (replay-parity test
>     green; pool-mode `python_assign_voice_preset` still gets age_band=None). +10 tests.
>   - **Chunk 4 `eb43856` -- HYBRID LLM voice-fit (the operator's CORE ask).** The LLM PROPOSES a
>     voice_ref_id from the engine's same-gender cards; Python VALIDATES (in-library + engine + gender +
>     no-collision) and FALLS CLOSED to the deterministic scorer. Decision rides
>     `meta.voice_cast_decision[char_id]` (policy_version/bank_sha/engine/prompt_version/seed/candidate_ids/
>     proposed_id/accepted_id/fallback_reason). CastLock honours the accepted id when its resolved engine
>     matches. **DELIBERATE DIVERGENCE FROM pass01_plan.md "fold into llm_write_description, NO new call":**
>     grounding found `character_description` feeds the line composer's voice card -> the dialogue -> the
>     AUDIO, so reusing that call would re-baseline dialogue audio as collateral. Implemented as a SEPARATE
>     bounded voice-fit call (`llm_propose_voice_ref`) that outputs ONLY a voice_ref_id, isolating the change
>     to the operator's intended lever. Default-ON; `OTR_HYBRID_VOICE_FIT=0` is the byte-identical escape (no
>     extra call; a no-LLM/failure run is byte-identical to pre-chunk-4). +11 tests.
> **>>> NEXT (CURRENT STEP):** (a) OPERATOR push gate for `995206e..eb43856` to origin/v2.0-alpha + golden
> recapture decision (the cloner audio re-baselines with timbre-matching + hybrid voice-fit ON); (b) build-
> order item 5 = the robustness ACCEPTANCE test + non-blocking stage-direction-only diagnostic count (the
> last VC item); (c) operator may also supply PD LibriVox titles for the still-open bank gaps (female-elder,
> child/teen, other/androgynous, male-light) for `scripts/otr_ingest_pd_voices.py`. Plan +
> grounding-corrected build placement: `docs/2026-06-22-voice-casting-arch/`.

> **LATEST SESSION -- 2026-06-22 (CODER; PD VOICE-LIBRARY INGESTION -- shipped + PUSHED. origin/v2.0-alpha
> HEAD `3cc8de6` == local. Suite 5001/34, Bug Bible 16/7/3.)** Continues VC Chunk 1's remediation ("each
> approved voice model needs a SOLID library"). Built `scripts/otr_ingest_pd_voices.py` -- ingests
> public-domain voice refs as ONE distinct voice per SPEAKER (download -> ffmpeg normalize mono/24k +
> band-pass + loudnorm -> concat+cap 25s -> sha256 -> mirror across the cloner engines vz_/cb_/dia_ -> merge
> + re-validate the bank; idempotent; one bad URL never stops the run). **Bank 137 -> 149 (+4 distinct PD
> voices x 3 engines):** Linda Johnson (F adult, LJSpeech), Mark F. Smith (M adult, Around-the-World; M
> elder, Mysterious Island), Phil Chenevert (M adult, Search the Sky). The ref WAVs live at
> `C:\ComfyUI-Models\TTS\refs\indextts2\` (model assets, NOT git). **Phil's operator-supplied URLs 404'd**
> (`search_the_sky_01_pohl_kornbluth.mp3` doesn't exist); fixed by reading the archive.org item metadata API
> -> the real files are `searchthesky_NN_pohl.mp3` (read by Phil Chenevert, PD mark 1.0). Also fixed a brittle
> test: `test_tts_engine_sidecars` asserted the pool count `== 36` -> `>= 36` (the bank grows as PD voices are
> added). Commit `3cc8de6` (tool + bank + test). Coverage now indextts2 M=15/F=23, chatterbox M=16/F=23, dia
> M=15/F=23, kokoro M=13/F=15 -- all four engines clear the >=5-adult/gender floor. **GAPS STILL OPEN
> (operator: supply PD LibriVox titles + I'll ingest): female-elder = 0, no child/teen, no androgynous/other,
> still male-light.** ZERO workflow-JSON change.
> **>>> NEXT (CURRENT STEP) UNCHANGED = VC Chunks 2-4** (the build plan in
> `docs/2026-06-22-voice-casting-arch/roundtable/pass01_plan.md`): these touch the VOICE-ASSIGNMENT
> DETERMINISM path (audio byte-identity-sensitive) -- build them with the byte-identical golden front-of-mind.
> Chunk 2 = two-lane identity refine + v2<->ref map; Chunk 3 = `meta.cast_voice_slots` stamp (so CastLock
> matches on timbre/age, not just gender -- note `EnsembleSlot` has timbre/role but NO age_band yet); Chunk 4 =
> the HYBRID LLM voice-fit (operator's core ask) folded into `llm_write_description` + the validator +
> `meta.voice_cast_decision`, $0 deterministic seeded fallback.

> **LATEST SESSION -- 2026-06-22 (CODER+ROUNDTABLE marathon; everything PUSHED. origin/v2.0-alpha HEAD
> `deb4e01`.) Two workstreams this session, both green per chunk (suite 5001/34, Bug Bible 16/7/3):**
>
> **(1) STORY+CAST FIX -- COMPLETE (STEPs 1-6 + 3 follow-up fixes), all PUSHED.** STEP 1 role source /
> STEP 2 cast-audit scope (roundtable Option A) / STEP 3 voice fail-soft / STEP 4 scoped reroll
> convergence + repair-then-ship / STEP 5 flat rubric + failed_dimension / STEP 6 escalating beat_tension
> + per-line dramatic-frame meta stamp. Then 3 LIVE-SMOKE-driven fixes: **STEP 3 voice gate is FAIL-SOFT**
> (never crash; resolve a voice per selected model -- `ff2c0c3`), **per-line SILENCE guard** for a
> stage-direction-only line (IndexTTS2 torch.cat crash -- `9a4f0a7`), and the **ROOT-CAUSE recompose** of a
> stage-direction-only character line into real dialogue (`e62081f`). All DEFAULT-ON, ZERO workflow-JSON
> change. Live smoke proved the tension ramp 1..5 + failed_dimension + frozen_with_doctor_edits WITHOUT the
> bypass; the crash was a separate pre-existing TTS bug, now fixed + a green re-smoke shipped.
>
> **(2) VOICE-CASTING ARCHITECTURE -- roundtable R1 CONVERGED + Chunk 1 shipped.** Operator opened a NEW
> workstream: every approved voice model needs a SOLID library + the LLM should make the best gender/voice
> casting call. R1 (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude judge, ~$0.21,
> `docs/2026-06-22-voice-casting-arch/`) converged on a HYBRID (LLM proposes a voice_ref_id from the
> engine's voice cards; Python validates + falls closed to the seeded scorer), a TWO-LANE identity
> (voice_ref_id for cloners / voice_preset for bark), a library coverage bar, and a non-blocking robustness
> diagnostic. The panel CORRECTED two of my groundings: gender+voice are PURE PYTHON today (Sprint 3D, not
> LLM), and the STEP-3 v2/* fallback is fine (cloners resolve via `_resolve_clone_ref_path` gender/any-ref).
> **Chunk 1 SHIPPED (`deb4e01`):** `_otr_voice_bank.compute_bank_coverage` + a floor test -- each approved
> engine has >=5 adult voices/gender (passes); the report SURFACES the gaps: cloners are MALE-LIGHT (~13-14 M
> vs 22 F), bank is adult-only (3 elder, all male), NO `other`/androgynous voices, NO announcer refs in the
> bank (announcer uses kokoro presets). **Operator remediation: add male + elder-female + other refs.**
> **>>> NEXT (CURRENT STEP) = VC Chunks 2-4 (the build plan in pass01_plan.md), a FOCUSED next push --
> these touch the VOICE-ASSIGNMENT DETERMINISM path (audio byte-identity-sensitive), so build them with the
> byte-identical golden front-of-mind:** Chunk 2 two-lane identity refine + v2<->ref map; Chunk 3
> `meta.cast_voice_slots` stamp (so CastLock's `_auto_registry` matches on timbre/age, not just gender --
> note `EnsembleSlot` has timbre/role but NO age_band yet); Chunk 4 the HYBRID LLM voice-fit folded into
> `llm_write_description` + the validator + `meta.voice_cast_decision`. Default-on, $0 deterministic fallback.

> **LATEST SESSION -- 2026-06-22 (CODER+ROUNDTABLE; STORY+CAST FIX **STEPs 5+6 SHIPPED + LIVE-SMOKED**,
> committed local NOT pushed. HEAD `f8a8645`, 4 commits ahead of origin/v2.0-alpha `da98144` (STEPs 1-4
> already pushed at da98144).) The
> STORY+CAST FIX is now COMPLETE (all 6 steps). Suite 4990 passed/34 skipped; Bug Bible 16/7/3; ZERO
> workflow-JSON change across all six steps -- every fix is unconditional internal node code, DEFAULT-ON in
> the canonical otr_scifi_16gb_full.json (nothing to enable).**
>   - **STEP 6 `7b27bd2`** -- escalating beat_tension: `compute_beat_tension_ramp` (ordinal ramp over
>     character beats, peak at the final beat) + the writer stamps a per-line dramatic frame on
>     `meta.line_dramatic_frame` (objective/obstacle/turn/subtext/tension/dramatic_question/next_turn). One
>     stamp serves three things: delivers tension to the composer, makes the target visible to the critic
>     (`_render_lines_for_doctor` renders `target_tension=N/5`), AND fixes the grounded reroll-reconstruction
>     gap (`build_reroll_line_request` rebuilt only arc_phase, losing the frame). +11 tests.
>   - **STEP 5 `165fbec`** -- flat rubric + failed_dimension: critic SECTION 3 now uses the 5-dimension
>     rubric (judged against beat_intent + appropriateness to target_tension, "be sparing"); SECTION 5 emits
>     only when a line fails BOTH advancement AND a dimension; `RerollTarget.failed_dimension` (optional enum,
>     back-compat default), folded into the reroll hint as a lightweight prefix (no re-craft mapper). +6 tests.
>   - **Roundtable R1 (~$0.21)** converged the design (`docs/2026-06-22-story-tension-rubric/`); GPT-5.5
>     surfaced the reroll-reconstruction gap. STEP6_GROUNDING.md proved beat_tension was UNWIRED.
>   - **LIVE SMOKE (canonical JSON, default settings, bypass OFF):** the ledger
>     `pending_20260622_100116` carries `line_dramatic_frame` (14 entries) with tensions escalating
>     1->1->2->2->2->3->3->3->3->4->4->4->5->5; the critic emitted `reroll_targets` with `failed_dimension`
>     (b005=pressure, b010=obstacle, b017=decision); `freeze_verdict=frozen_with_doctor_edits` (SHIPPED, no
>     bypass), reroll_diverged=True, cycle_count=2. ALL six steps are default-ON end to end.
> **>>> NEXT:** push the 8 commits (operator gate) + an optional broader re-soak read; then the S3 forward
> order (3D item 5 / distribution item 6) resumes -- all PARKED + untouched by this fix.

> **LATEST SESSION -- 2026-06-22 (CODER+ROUNDTABLE; STORY+CAST FIX STEPs 1-4 SHIPPED, committed local
> NOT pushed -- operator gates the push. HEAD `736d0d6`, 4 commits ahead of origin/v2.0-alpha `a55da87`.
> Full suite 4973 passed/34 skipped; Bug Bible 16/7/3; ZERO workflow-JSON change across all four steps.)**
> Built `docs/2026-06-22-story-cast-roundtable/roundtable/pass04_plan_FINAL.md` STEPs 1-4, each chunk
> suite+Bug-Bible green then commit:
>   1. **STEP 1 `3be38e1`** -- role_mismatch: dropped the `or row.get("tts_model")` fallback in
>      `_render_cast_contract_table` (engine name read as a role) + guaranteed `set_lines` stamps an explicit
>      speaker_role (the streaming partial-ledger path left it empty). speaker_role is the ONLY role source.
>   2. **STEP 2 `3992b68`** -- the literal "migrate music/sfx -> new cue_type + add archetype field" was a
>      schema-breaking miss (those fields exist NOWHERE; music/sfx are load-bearing line-row speaker_roles on
>      the FROZEN schema). A focused R1 roundtable (GPT-5.5+Gemini-3.1-pro+DeepSeek-v4-pro+Claude judge, ~$0.33,
>      `docs/2026-06-22-story-cast-step2-schema/`) UNANIMOUSLY adopted Option A (prompt-boundary only): derive a
>      real cast role (announcer/character) for the auditor instead of '', and audit only spoken rows.
>   3. **STEP 3 `0fa014e`** -- voice fail-closed gate at OTR_CastLock (node-80) OUTPUT: no character line reaches
>      node 81 with voice_preset=None (line-driven, engine-agnostic, NAMED raise; announcer/cue excluded).
>      cast_seed canonical-key VERIFIED (writer meta.cast_contract.cast_seed == CastLock read).
>   4. **STEP 4 `736d0d6`** -- scoped critic + correct reroll convergence: run_story_critic gains scope_line_ids
>      (None=whole-episode); the reroll loop re-scores SCOPED to patched+neighbor lines, converges on the
>      invariant (targeted clear; neighbors join next scope; halt only on cycle-cap OR outstanding-count
>      INCREASE -- not strict-decrease) and KEEPS repairs (repair-then-ship; the freeze cascade A2 path ships
>      residual needs_full_rerun). A final whole-episode pass refreshes meta.story_critic_report for render
>      coupling. +9 tests.
> **>>> NEXT (CURRENT STEP) = RE-SOAK the minimal matrix** (1 small e.g. gemma-12b + 1 frontier e.g. grok,
> ONE word tier) on the REAL `otr_scifi_16gb_full.json`, **OTR_BYPASS_FREEZE_HALT OFF**, after a box reset
> (CLAUDE.md S4 -- the night-soak server may still hold :8000 with the bypass on; resetting interrupts the
> live OBS broadcast, so OPERATOR-GATED). Acceptance: >=70% frozen_clean, 0 cast-contract violations, no
> voice_preset=None. Then REMOVE the OTR_BYPASS_FREEZE_HALT stopgap once STEP 4 converges; THEN ground STEP 6
> (read the beat/outline planner; do NOT add SceneArcContext -- LineRequest already carries the arc fields).
> STEP 5 (flat rubric + failed_dimension) pairs with STEP 6.

> **LATEST SESSION -- 2026-06-22 (PLANNER+CODER+ROUNDTABLE, marathon; HEAD `358accd` == origin/v2.0-alpha
> after the indextts2 realpath fix; the night-soak driver + roundtable docs are LOCAL/uncommitted).**
> **>>> NEW CURRENT STEP (build): the STORY + CAST FIX -- `docs/2026-06-22-story-cast-roundtable/roundtable/
> pass04_plan_FINAL.md`. Build STEPs 1-4 FIRST (small, contained, test-backed), re-soak the minimal matrix,
> THEN ground STEP 6.** A 4-round live roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Claude
> code-grounded judge, ~$0.15) hardened it; the judge+grounding caught that THREE panel "fixes" were ALREADY
> implemented (prose/metadata decouple; the reroll `hint`; targeted patching), so the build targets the 6 REAL
> defects, sequenced:
>   1. **role_mismatch = ONE line** -- `nodes/_otr_ledger_reviewer.py:500` `role = row.get("speaker_role") or
>      row.get("tts_model")` reads the TTS engine name as a role -> drop the `tts_model` fallback + guarantee
>      speaker_role is set per row.
>   2. **migrate cast schema BEFORE validating** (legacy music/sfx role values -> `cue_type`; archetype in its
>      own field; speaker_role only for spoken rows).
>   3. **voice fail-closed at OTR_CastLock (node 80) OUTPUT** -- no `voice_preset=None` reaches TTS 81/82
>      (`cast_lock.py:272` silent cast_seed=None skip + unmatched char_id are the sources).
>   4. **whack-a-mole critic** -- `run_story_critic` re-scores the WHOLE episode each reroll
>      (`_otr_freeze_cascade.py:754` whole-episode; `_otr_reroll.py:621` reroll-loop) -> add
>      `scope_line_ids` + the CORRECTED convergence invariant (targeted ids must CLEAR; newly-failed neighbors
>      join next scope; halt only on cycle-cap OR GLOBAL-count increase -- NOT "strict decrease", which
>      false-halts when fixing N surfaces N+1).
>   5. **flat rubric** -- shared 5-dimension definition in the critic PROMPT + a `failed_dimension` enum;
>      critic-output + the `_otr_reroll.py` hint parser updated TOGETHER. Rubric-guided LLM judgment, NOT a
>      deterministic code test.
>   6. **the arc lever is BEAT-PLANNING, not line-writing** -- `LineRequest` ALREADY carries
>      beat_objective/obstacle/turn/beat_tension/next_turn/outline_spine, so do NOT add a SceneArcContext;
>      GROUND-FIRST: read the beat/outline planner + check whether beat_tension escalates, THEN decide.
>   ALL fixes are INTERNAL node code -- ZERO workflow-JSON change (add a no-drift regression check). Acceptance:
>   >=70% frozen_clean on the minimal matrix (1 small + 1 frontier, 1 tier), 0 cast violations, no
>   voice_preset=None, and REMOVE the `OTR_BYPASS_FREEZE_HALT` stopgap once STEP 4 converges. Full plan + all 4
>   passes / anchors / judgments / raw panel reviews: `docs/2026-06-22-story-cast-roundtable/roundtable/`.
> **ALSO THIS SESSION (after the capability-routing block below):**
> - **indextts2/chatterbox/dia realpath fix SHIPPED+PUSHED (`358accd`):** under the Desktop-v2 junction install,
>   `abspath(__file__)` pointed `_COMFY_ROOT` at the install root so the Path-B sidecar venv/weights "weren't
>   found" -> indextts2 dead headless. Fixed with `realpath` (resolves the junction; no-op on a non-junction
>   checkout; env overrides still win). Suite 4957/34 + Bug Bible 16/7/3 green.
> - **NIGHT BROADCAST SOAK (ran ~overnight):** new `scripts/_otr_night_soak.py` (LOCAL) -- 10h, alternating full
>   / visualizer-only, rotating writers (mistral/gemma-12b local + grok-4.3/gpt-5.5-pro/deepseek-v4-pro frontier
>   via OpenRouter slot-a, technical=local mistral -- the operator's cost-smart split), word tiers
>   420/560/700/864, indextts2 voice + kokoro announcer, ltx_av_music bookends + z_image_turbo flat_still char
>   beats, fresh cast/news. **17 episodes published to obs.** REQUIRED the `OTR_BYPASS_FREEZE_HALT=1` server-boot
>   STOPGAP (the story-critic freeze gate -- the exact thing the STORY+CAST FIX repairs) + `act_count='auto'`
>   (the workflow enforces a word-scaled act minimum; act=1 is rejected). Boot = FLOOR lane + OTR_ENABLE_ZIMAGE +
>   OTR_ZIMAGE_UNET=z_image_turbo_nvfp4.safetensors + OTR_ENABLE_LTX_AV + OTR_ENABLE_OPENROUTER + OPENROUTER_MODEL_A.
> - **WAN 14B REMOVE (morning task, NOT done):** operator REVERSED the wan promotion -- the wan2.2 i2v 14B fp8
>   thrashes on 16GB (re-stages 13.6GB per chunk via the OTR VRAM wrapper) + a `b000_music_open` frame-count
>   explosion hung a 100%-wan eyeball ~40min on ONE beat. Plan: clean NATIVE image->video wan smoke (no OTR VRAM
>   wrapper) -> source/confirm a <14GB Wan 2.2 (5B `wan_ti2v` exists) -> remove the 14B as a selectable engine
>   from `workflows/otr_scifi_16gb_full.json`. The capability-routing fix (`cf5fbb3`) is engine-agnostic + STAYS.

> **UPDATE 2026-06-22 (capability-routing SHIPPED, `cf5fbb3`):** the capability-routing campaign (R1-R3)
> is now BUILT + PUSHED + GREEN, not just converged. `nodes/_otr_shared/role_compat.py::engine_fits_role`
> DROPS the per-engine `roles` whitelist gate -> eligibility is PURELY capability
> (`required_inputs <= role_available_inputs`); the `roles` attrs are now dead (deferred cleanup). wan_i2v
> now drives the announcer (and any b-roll engine fits any role whose inputs it satisfies); HuMo/LTX-AV
> stay audio-gated BY CAPABILITY. PROVABLY non-regressive SUPERSET (`tests/test_capability_routing.py`
> 10-case matrix + 6 existing role-fit tests updated to capability-only). Aspect = self-consistent per
> pick (`otr_video_director._role_aspects` derives the still aspect from the selected engine's
> render_aspect). Suite 4957/34 + Bug Bible 16/7/3. NEXT (operator eyeball): re-render the 100% Wan to SEE
> wan drive the announcer end-to-end; optional dead-`roles`-attr cleanup.
> `docs/2026-06-22-capability-routing/FINAL_PLAN.md`.

> **LATEST SESSION -- 2026-06-22 (CODER+ROUNDTABLE, very long; PER-SEGMENT RMS LOUDNESS LEVELING SHIPPED +
> PUSHED + EAR-VALIDATED; CAPABILITY-ROUTING roundtable R1 CONVERGED, not built).**
> **(1) AUDIO LEVELING (baked-in default):** even out shot-to-shot dialogue levels. 4-round live roundtable
> (gpt-5.5+gemini-3.1-pro+deepseek-v4-pro+grok-4.3 + Claude judge, ~$0.52) -> implemented in
> `nodes/scene_sequencer.py`: per-dialogue-clip RMS leveling (`_level_dialogue_clip` -> `_loudness_normalize_clip`,
> target -16 dBFS, peak-safe cap + noise gate + float64 RMS; the 3 dialogue call sites :747/:753/:775; SFX
> :726 stays peak). **RMS is the BAKED-IN DEFAULT** (`OTR_SEGMENT_LOUDNORM` defaults `rms`; `=peak` escape
> hatch) per operator "don't hide behind a switch." Episode master UNCHANGED (makeup +4 dB) + target ~= what
> peak-norm produced, so the per-line BALANCE evens WITHOUT shifting overall loudness. **Voice-MODEL-AGNOSTIC**
> (downstream of TTS; leveled bark chars + Kokoro announcer in one render). Commits `82fbd4e` (CLAUDE.md S8
> roundtable defaults) + `8ea03e9` (leveling) PUSHED to v2.0-alpha; full suite **4947 pass/34 skip** + Bug Bible
> 16/7/3 green; proven LIVE (`[loudnorm]` preflight, target -16) + operator EAR-VALIDATED ("swift sounds ok").
> Docs `docs/2026-06-22-loudness-normalization/` (FINAL_PLAN + pass00-04 + OPERATOR_NOTES +
> `tools/measure_dialogue_rms.py`; old-path dialogue measured -13..-24 dBFS, mean -17.3). **UNCOMMITTED:**
> `tests/_run_baseline.py` workflow-path bug fix (two-`..`->one; had silently broken the gated
> `OTR_REGRESSION_RUNTIME` audio capture). **Byte-identical GOLDEN re-baseline DEFERRED** (default-peak keeps the
> suite green; the rms golden is a future headless capture -- must force bark, indextts2 isn't headless-installed).
> **(2) CAPABILITY-ROUTING (NEW; roundtable R1 CONVERGED; NOT built):** live wall during a 100%-Wan eyeball --
> `wan_i2v` rejected from the announcer slot by the `roles` whitelist DESPITE being input-compatible (the
> announcer supplies `init_image`). Operator directive: declare capabilities ONCE per model, model-agnostic
> downstream (HuMo/LTX-AV = audio-in specials, the rest = still+prompt -> all slots); **HARD: strict SUPERSET,
> do NOT break working models.** R1 CONVERGED (`docs/2026-06-22-capability-routing/roundtable/pass01_plan.md`):
> make `roles` an OPTIONAL override (empty -> pure capability so wan unblocks; set -> enforced, no over-match) +
> decouple `default_roles` from eligibility + confirm the descriptor `roles` source in `otr_video_director.py` +
> handle ASPECT (wide vs portrait) + unify the render-gate source (FAMILY_REQUIRED_INPUTS <- required_inputs) + a
> GENERATED before/after eligibility test + eligibility != auto-selection + VIDEO-only v1 (defer image).
> **NEXT = capability-routing R2 (coding) -> R3 (wiring) -> R4 + build; the audio leveling is DONE.** (Wan
> roadmap item 3: wan_i2v fits b-roll music/beats, not the audio announcer -- the no-silent-swap safety fired
> correctly; the capability fix makes wan a b-roll announcer.)

> **LATEST SESSION -- 2026-06-22 (CODER, marathon; HEAD `08010ec` == origin/v2.0-alpha; full suite 4934
> pass/34 skip, Bug Bible 16/7/3, audio byte-identical 9/0). THREE sprints SHIPPED + PUSHED end to end, each
> chunk suite+Bug-Bible green -> commit+push to v2.0-alpha:**
> 1. **BARK ARTIFACT (upstream-TTS only, spine frozen):** B0 fixture+spine-verify `ffe5e82` -> B1 speech-only
>    dialogue mode + first-line [clears throat] gate `467fb05` -> B2 thread the dropped per-line seed
>    (determinism) `9e68ad5` -> B3 chunk-split hardening `e1b8196` -> QA >4kHz high-band edge metric + scan
>    `e93b54e`. Cleaned-bark re-soak (mesh 3D bookends + z_image_turbo char stills) rendered "Spinning
>    Contamination" to obs; metric before/after = pre-B1 1/14 flagged vs cleaned 0/6. min_eos_p kept 0.1;
>    panel-cut the per-chunk trim + FFT reroll loop.
> 2. **STAGE-DIRECTION LEAK (3-pass roundtable, ~$0.51):** bare undelimited stage directions ("twirls his pen
>    nervously Look...") were leaking into the frozen ledger text -> bark spoke them + captions showed them.
>    Chunk 1 pure scrub+detector+62-case corpus `8c40182` -> Chunk 2 scan + PRECISION GATE (489 ledgers, 20
>    would-mutate, 0 false positives) `e2dd95a` -> Chunk 4 freeze floor in `_strip_stage_directions` `2278bd2`
>    -> Chunk 5 prompt + the S1 music-3681 patch `9142b2f` -> Chunk 3 reroll in compose_line `6ce724d`. Fresh
>    render shipped a CLEAN ledger; the fixed freeze strips the exact screenshot lines.
> 3. **STORY-QUALITY R2 -- ALL 8 LEVERS + Final-QA scan (the operator-requested craft lift):** S1 `5396c05`
>    (music-text suppression) -> S2 `118088f` (announcer close = concrete image, not thesis) -> S3 `1887803`
>    (cliche + flat stage-business reroll) -> C0 `687f766` (action-verb beat intents + wants_are_default) ->
>    C3 `981db60` (contrasting speech_signatures) -> C4+C5 `3e19906` (escalation prompt + on-the-nose reroll)
>    -> C1+C2 `38c2c10` (specificity anchors + central object, 3-pass roundtable-converged ~$0.25) -> Final QA
>    `08010ec` (story_quality_scan extended with the lever metrics). ALL content-only, ledger schema FIXED
>    (free-form meta), NO workflow-JSON change, model-agnostic, deterministic.
> **OPERATOR VERIFY-AT-BUILD (carries forward):** the live `test_audio_byte_identical` baseline (indextts2)
> may need RECAPTURE -- the R2 prompt levers intentionally change generated dialogue (the clean fixture is a
> no-op, but a fixed-seed baseline close that newly trips a gate would shift); run the live gate + a re-soak
> read at leisure. **NEXT = OPERATOR DECISION (no sprint started this hand-off): pick the next forward-order
> item** (the S3 forward order -- 3D item 5, distribution item 6 -- + the carried per-segment LUFS/RMS plan
> are all still PARKED + untouched).

> **PRIOR SESSION -- 2026-06-22 (CODER, very long session; HEAD `90ddfca` == origin/v2.0-alpha; full suite
> 4740 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green). HANDOFF FOR A FRESH CODER WINDOW: the
> NEW CURRENT STEP is the STORY-QUALITY R2 coding sprint -- build-ready in `docs/2026-06-22-story-quality-r2/
> SPRINT_PLAN.md` (see CURRENT STEP, section 1).** SHIPPED + PUSHED this session, each suite+Bug-Bible green:
> - **3D image streams (all 7 chunks)** done + GPU-verified end-to-end earlier in the session (clay-blob fixed;
>   the block below). Then **mesh-quality v1.1+v1.2**: tighter fodder (`ea03203`), gradient sculpt surface
>   (`f8a48b9`), and the lighting fix `e485258` -- the meshes were WHITE MARBLE because WORKBENCH MATCAP
>   ignores the vertex albedo; switched to STUDIO lighting so the gradient + 3D form render (CPU-Blender
>   verified). Both 30w all-3D GPU smokes published to obs.
> - **OpenRouter writer fix** (`d249743`): the nightly frontier writers were 0-lining via `finish_reason=length`
>   (reasoning `~latest` models burning the output budget). Fix = `OPENROUTER_REASONING_EFFORT=none` (set in the
>   server boot + the User env for manual runs) + a one-line "reasoning_effort ACTIVE" log so it's EVIDENT on any
>   run. Roundtable-converged. Confirmed live (remote preflight wrote a full story; near-every leg now writes).
> - **ltx_av_music render crash** (`fadbe60`) + **beat-accurate audio** (`90ddfca`): a music beat with no
>   per-line timing FamilyInputGap-crashed ltx_av_music (ShotRow has no start_s/dur_s). Fix = synthesize a
>   BOUNDED master slice (target_frame_count/fps at the beat's cumulative position, anchored on preceding beats'
>   real line timing -> respects the opening-music offset); audio_driven_face excluded. Roundtable-converged,
>   +4 tests. NOTE: both render_driver fixes are IN-PROCESS -> live on the NEXT server boot, not the still-
>   running soak.
> - **Nightly 10h anthology soak** (local tooling `scripts/_otr_nightly_anthology_soak.py` +
>   `_otr_nightly_soak_boot_launch.ps1`, gitignored): 384w max-creativity, rotating frontier OpenRouter writers,
>   mixed video/3D bookends, rotating approved image models on character beats, rotating voices, dynamic news,
>   720w visualizer story-tests every 4th leg; smoke-first + remote preflight + soak-only local fallback. RAN
>   ~13h, feeding `otr/obs` for the operator's OBS->YouTube broadcast loop.
> - **STORY-QUALITY R2 CAMPAIGN -- CONVERGED, BUILD-READY (the NEXT step).** Grounded in the 13h soak's REAL
>   stories (opus = genuinely good; weak/local = generic + cliche + stage-business + music/non-dialogue text
>   bleeding into the caption track). 3 roundtable passes (pass01 structural + pass02 coding) + Claude's own
>   creative pass (pass01b) + seam-location -> `SPRINT_PLAN.md`: 8 ordered green chunks (S1 music/non-dialogue
>   caption suppression; S2 announcer-close final-image; S3 cliche/stage-business reject gate; C0 non-default
>   wants + action-verb in outline Stage 3; C1 specificity anchors; C2 central story-object; C3 contrasting
>   voice signatures; C4+C5 escalation+subtext). HARD: ledger schema FIXED (new fields ride free-form `meta`,
>   NO Pydantic fields), NO workflow-JSON change, reuse the EXISTING Sprint-5C `reroll_hint` loop +
>   `speech_signature` (already wired in `_otr_line_composer.compose_line_draft`), model-agnostic (every gate is
>   one opus passes -> lifts the weak end, never rewrites opus), craft-ONLY (no word/beat count). Located seams +
>   judgments in `docs/2026-06-22-story-quality-r2/`. >>> THE CODER WINDOW BUILDS SPRINT_PLAN.md, S1 FIRST.
> - OPEN (operator call): restart the soak so the ltx_av + reasoning fixes go live now, vs let it ride to the
>   10h cap and they apply next boot.

> **LATEST SESSION -- 2026-06-21 (CODER; 3D IMAGE STREAMS -- ALL 7 CHUNKS SHIPPED + PUSHED + GPU-VERIFIED
> END-TO-END; HEAD `555788e` == origin/v2.0-alpha; full suite 4733 pass/33 skip, Bug Bible 16/7/3, audio
> byte-identical green per chunk).** The clay-blob root cause is FIXED: `mesh_stage` no longer meshes the
> per-beat cinematic scene still (the whole environment) -- it now meshes a CLEAN isolated subject over a
> generated background plate. Seven ordered green chunks, each suite+Bug-Bible -> commit AND push ->
> HEAD==origin verified: **(1)** `b11759f` `requires_mesh_fodder=True` capability flag on MeshStageEngine
> (routing gate; capability-not-engine-name). **(2)** `a5be9b3` `build_request_from_shot` routes fodder
> engines to a clean `mesh_fodder` still BEFORE the `_SCENE_INIT_FAMILIES` override (skips it for fodder
> engines; LOUD `missing_mesh_fodder`, never meshes the environment); engine map read post-`OTR_FORCE_ENGINE_MAP`.
> **(3)** `af779bd` the prompt FORK -- `OTR_ImageDirector` forwards `mesh_fodder_roles` (per-role engine
> capability), `OTR_MetaBriefImagePromptGen` mints `mesh_fodder` (isolated subject) + `scene_background_plate`
> (subject-free world) with checked-in pos/neg scaffolds instead of one cinematic scene still. **(4)** `0ccb6ff`
> kind-specific indices -- `_portrait_index` kind-filtered (fodder can't leak to HuMo), `_still_index`
> prioritizes `scene_background_plate`, dispatcher carries `mesh_subject_id` onto the ledger row. **(5)** `84962bd`
> mesh cache keys on `mesh_subject_id` (char_id|object_id), not the per-beat still hash; render_driver stamps it.
> **(6)** `a2df9bf` announcer/music subject policy (announcer meshes the announcer via char_id; no-char beats
> mesh a stable `obj_<beat>` object; LOUD on bare `uncast`). **(7)** `555788e` opaque straight-alpha source-over
> composite (mesh over plate, `format=rgb`) -- kills the double-exposure ghost; blend kept as
> `OTR_MESH_COMPOSITE_STYLE=blend` opt-in; real-ffmpeg opacity regression. All content-only / capability-gated;
> ZERO workflow-JSON change; ledger schema `l3-2026-05-14` untouched. **GPU SMOKE (all-slots mesh_stage, 30w,
> real `otr_scifi_16gb_full.json`, FLOOR lane, bark voice):** SUCCESS in 20:17. The dispatcher minted 5
> `mesh_fodder` + 5 `scene_background_plate` objects (every beat -- music/announcer/character), NO cinematic
> scene stills; render_driver logged `meshing CLEAN fodder ... scene still is background plate only` for all 5
> beats; cache keys were the real subjects (`obj_b000_music_open`, `c01`, `c02`, `c03`) -- NEVER `uncast`; the
> VRAM barrier fired before each Blender spawn; `obs_publish OK`. Final = `otr/obs/
> signal_lost_colony_found_20260621_002009_silent_procgen_blended_final.mp4` (20.8 MB, 64.5s). Logs/proof in
> `docs/2026-06-21-3d-image-streams/` (smoke_server.log + the obs mp4 copy). KNOWN COSMETIC NIT (non-blocking):
> the render_driver "meshing CLEAN fodder" log prints `subject=?` for a no-char beat (the cache id is correct,
> `obj_<beat>`); a one-line log polish if it bugs you. **NEXT = 3D v1.5 (DEFERRED, panel-agreed): Cycles +
> 3-point lighting + multi-view texture bake** (the lit/textured tier, on top of clean fodder).

> **LATEST SESSION -- 2026-06-20 (CODER; BUG 1 SHIPPED + GPU-VERIFIED end-to-end; HEAD `9f03abd` ==
> origin/v2.0-alpha; suite 4633 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green).** TOP-PRIORITY
> VISUAL FIX **BUG 1 DONE** -- landscape character beats now show the CHARACTER full-frame 16:9, not the
> radio booth. Two commits: **`7e765b7`** (a) render_driver: flat_still/flux_still are render_aspect=wide ->
> REVERTED the `8bc5381` portrait-conditioning branch; they condition on the beat's scene still, never the
> 832x1216 vertical portrait (missing-still clears init_image so it can't leak); (b) image phase mints a
> per-beat `scene_character` still leading with the character, radio-booth tail dropped. **`9f03abd`** the
> ROOT CAUSE the live render exposed: `"character"` (the CANONICAL writer speaker_role) was MISSING from
> `SPEAKER_TO_VIDEO_ROLE` -> character beats fell through to `background_abstract` and got POOLED as
> other-beats (the deleted 8bc5381 branch had masked it) -> added `"character"->CHARACTER_VIDEO`; PLUS
> BEAT-AWARE stills (operator: per-shot/beat, regardless of image model, video lane too) -- each character
> beat's still is LLM-composed from that beat's `beat_intent`/`traits`/`text` (temp=0, mirrors the portrait
> path; person-guard + gear-scrub + era/grade tail + no-text clause; deterministic per-character fallback).
> Content-only, ZERO workflow-JSON change. **GPU VERIFY (z_image_turbo, FLOOR lane, all-roles flat_still,
> 60w/act1/2chars):** the 3 character beats b002/b003/b004 resolved `kind=scene_character role=character_video`
> (NO other_pool), 3 DISTINCT prompt hashes `source=char_scene_llm`, render_driver conditioned each on its
> scene_character still ("portrait never used"), `audio_byte_identical OK`, `obs_publish OK`. Frames pulled
> from the FINAL obs mp4 at each beat timestamp show the CHARACTER full-frame 16:9 with the matching SDH
> caption -> beat-synced end-to-end. Before/after + final-frame proofs in `docs/2026-06-20-visual-fixes/`.
> GOTCHAS: (1) z_image_turbo needs `OTR_ENABLE_ZIMAGE=1` + `OTR_ZIMAGE_UNET=z_image_turbo_nvfp4.safetensors`
> at server boot (only the nvfp4 quant is on disk, not the bf16 the engine defaults to). (2) the repeated
> "GULLIVER REEVES" cast is the `OTR_C7=1` byte-identity mode pinning OTR_CAST_SEED/STYLE_SEED=42 (my verify
> runs) -- production (no C7) rolls a fresh cast per episode. **ALSO THIS SESSION: BUG 2 DONE (verify-first,
> no change -- the animated title card + rolling credits already work) + BUG 3 DONE (commit `9f76937`: the
> forensic model/engine/SYSTEM dossier now scrolls on the credits via `_build_hud_dossier`; content-only, no
> JSON; +7 tests). ALSO BUG 4 DONE (commit `13017ec`: the ALWAYS-ON audio bars are a SEPARATE overlay layer,
> default ON, decoupled from the scene-aware floor -- new `audio_bars` widget wired IN the workflow JSON
> same-commit [node 93], a separate bars pass lighten-blends the green strip at 0.60 with captions ON TOP;
> +6 tests). HEAD `13017ec` == origin, suite 4646 pass, Bug Bible 16/7/3, JSON intact+extended. The 2026-06-20
> ALL FOUR visual fixes (character aspect / title card / credits dossier / audio bars) are SHIPPED. **NEXT =
> the 3D PoC build (`docs/2026-06-20-mesh-stage-texturing/` pass03+pass04 + the 4-item PIN list; chunks
> 6+7+3 + GPU smoke).**

> **LATEST SESSION -- 2026-06-20 NIGHT (CODER, autonomous "go all night"; HEAD `8de1057` == origin/v2.0-alpha;
> suite 4625 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green).** BUILT + PUSHED the CORE of the
> 3D textured-hero PoC (chunks 2+4+5 paired so producer+consumer land together -- no dead wiring), commit
> `8de1057`: **(2) single-view VERTEX-COLOR PROJECTION** in `scripts/otr_mesh_stage_blender.py` -- `--portrait`
> via `bpy.data.images.load`, a per-vertex point-domain `otr_proj` color attribute painted from the front Y/Z
> projection, set active+render so WORKBENCH `color_type='VERTEX'` draws it; GLB stays geometry-only
> (MESHER_VERSION NOT bumped); bounded hero arc (`_build_turntable` start_angle+arc clamped to MAX_ARC=45deg,
> `frames==1`->one keyframe); selftest projects a deterministic gradient + fails nonzero unless a NON-UNIFORM
> attr exists; pure helpers (`arc_keyframes`/`project_uv`/`sample_image`/`clamp_arc_degrees`) CPU-tested.
> `eng_mesh_stage.build_blender_cmd` plumbs `--portrait`/`--start-angle`/`--arc-degrees` (appended only when set
> -> legacy invocation byte-identical); `render_clip` passes the resolved still. **(4) C1 STAMP**
> (`render_driver.build_clip_manifest`): mesh_stage directory rows get `bg_still_path` from the per-beat scene
> still (the existing per-beat coverage already mints one via the per-role image engine -> image-model AGNOSTIC),
> fail-closed `os.path.isfile` + LOUD warn on absence. **(5) C1 COMPOSITE** (`otr_silent_composite.py`):
> `bg_still_path` carried through `plan_timeline_segments` + a still-aware `-loop 1` bg branch in
> `_encode_segment_from_dir`; ZERO new graph links/widgets (rides the existing 92->84 manifest channel); every
> non-mesh beat omits the field -> floor/black bg byte-identical. +13 unit tests. **REAL-BLENDER VALIDATION
> (CPU-only, did NOT touch the running soak's GPU):** the stage selftest ran through Blender 4.5.10
> (`C:\ComfyUI-Models\tools\blender-4.5.10\blender.exe`) WORKBENCH -> exit 0, 3 RGBA frames of DIFFERING sizes
> (47k/49k/50k = the arc moves + the projection is non-uniform); proof frame
> `docs/2026-06-20-mesh-stage-texturing/selftest_proof_frame_0001.png` shows the cube TEXTURED with the gradient
> (green/teal/blue/magenta), NOT flat gray -> projection works visually in real Blender. **DELIBERATELY NOT
> LANDED BLIND tonight (need the GPU render loop the soak occupies):** Chunk 6 (costly_choice_beat trigger ->
> route that beat to mesh_stage; the no-portrait->still_parallax fallback is ALREADY in place), Chunk 7 (JSON
> wiring on node 87/88), Chunk 3 (2D-ellipse contact shadow, wants GPU-render tuning), and the GPU smoke
> acceptance. Full detail: `docs/2026-06-20-mesh-stage-texturing/BUILD_RESULTS.md`. **SOAK (watched):** the
> 864-word frontier all-night soak on :8011 is HEALTHY -- smoke_0001 PASSED (status=success, 480 words; indextts2
> did NOT crash the server), now grinding the 48-leg matrix (leg_0002+). NEXT = (a) keep watching/triaging the
> soak, (b) when it frees the GPU, run the mesh_stage GPU smoke then land chunks 6+7 (+3) VALIDATED.

> **LATEST SESSION -- 2026-06-20 (CODER + 3D ROUNDTABLE CAMPAIGN; HEAD `f99af26` == origin/v2.0-alpha;
> suite 4616 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green).** SHIPPED + PUSHED (each
> suite+BugBible green): **forensic episode treatment** -- `_write_story_treatment` now emits a full
> spec sheet (LLM config: both slot models + creativity/temp/words; STORY SPINE: news premise + opposed
> wants; per-role RENDER ENGINES video+image; SYSTEM: CPU/RAM/GPU/CUDA/torch; new `nodes/_otr_sys_specs.py`)
> (`e8f3094`); **resolved OpenRouter concrete-model + cost capture** per run for historical accuracy
> (`f03db0a`, `_record_resolved`/`resolved_models_snapshot`, cleared per episode in reset_run_budget);
> **Kokoro char-voice DEEP pool** -- the bank had ONE kokoro entry (announcer-only) so chars collapsed to
> one voice; registered the full English Kokoro-82M set (DOWNLOADED the 15 missing -> 28 on disk, 15F/13M)
> as char_voice entries so CastLock assigns DISTINCT gender-matched voices like Bark (`dee7d5a`+`f99af26`;
> also fixes the missing `bf_emma` announcer voice). **3D ROUNDTABLE CAMPAIGN CONVERGED (docs only, NO 3D
> code yet)** -- `docs/2026-06-20-mesh-stage-texturing/roundtable/`: the "plaster of paris" blob is NOT
> WorldMirror (not wired into OTR at all) -- it's **`mesh_stage`** rendering a geometry-only Hunyuan3D-2mv
> GLB through Blender WORKBENCH matcap (flat gray `(0.78,0.78,0.78)` when the mesh has no vertex colors).
> R1 architecture -> R2 coding-plan (image-AGNOSTIC stills) -> R3 hardened+build-pin-list -> R4 wiring all
> converged via the live panel (GPT-5.5+Gemini-3.1-pro+Grok-4.3, ~$0.41). **PoC = a single-view-TEXTURED
> mesh_stage hero beat (vertex-color projection in Blender) over a GENERATED background, bounded camera
> arc, ledger-driven (costly_choice_beat), no-portrait->still_parallax; CAMERA-ONLY motion (lip-sync/rig =
> the deferred character_3d lane, OUT).** Wiring = C1 (per-clip `bg_still_path` in the manifest; ZERO new
> links/widgets); the exact 3-function composite patch + a 4-item build-time PIN list are in
> `pass03_plan.md` (build) + `pass04_plan.md` (wiring). Build is sprint-ready + OPERATOR-GATED.
> **NEW OPEN TICKET -- audio-reactive bottom BARS don't show:** painted by OTR_SceneAwareScopes as the
> FLOOR (under the clip) + suppressed for portrait/un-probeable/credits. Fix = a dedicated ALWAYS-ON bars
> overlay at the POST stage, decoupled from scene-aware suppression, captions stay above. Plan:
> `docs/2026-06-20-audio-bars-fix/BARS_FIX_PLAN.md`. **RUNNING NOW (next window WATCHES it):** an
> ALL-NIGHT 864-word FRONTIER story soak on :8011 -- 4 frontier creative (Gemini/Grok/GPT/Opus) + Opus
> tech, indextts2 voice, z_image character stills (flat_still), DIVERSE seeds (a different news story per
> leg via the new `OTR_SOAK_DIVERSE_SEED=1` driver knob), 48 legs/8h cap; output
> `docs/2026-06-21-allnight-864-frontier/`. **RISK: indextts2 is the headless-fragile sidecar -- a prior
> leg crashed the server; the smoke-leg-first guard aborts cleanly if it fails -> FALLBACK = kokoro (now
> a deep 28-voice pool).** Opus-tech `~latest` throws transient OpenRouterModelGoneError(404) -> the
 writer gracefully SKIPS those slots (transport fail-soft), stories still write. **SOAK DONE -- 35 legs/8h
> (results docs/2026-06-21-allnight-864-frontier/story_soak_results.csv); GPU now FREE.**
> **>>> NEW OPERATOR DIRECTIVE + TOP-PRIORITY VISUAL FIXES (2026-06-20, docs/2026-06-20-visual-fixes/
> VISUAL_FIXES_PLAN.md):** DIRECTIVE -- ONLY HuMo(portrait)+maybe 3D use the VERTICAL portrait; EVERY other
> path uses LANDSCAPE 16:9. **BUG 1 (do FIRST):** character-beat stills look like bookends/scene shots, not
> characters -- commit `8bc5381` wrongly conditions flat_still/landscape engines on the 832x1216 portrait
> (pillarboxed -> radio-booth floor sides = the "radio booth images"). FIX = (a) render_driver: landscape
> engines NEVER use the vertical portrait (gate on `render_aspect`, portrait ONLY for HuMo/3D); (b) image
> phase MINTS a 16:9 CHARACTER shot for character beats (image-model agnostic) so the character shows
> full-frame. **BUG 2 (verify-first):** the soak log shows rolling credits + procgen DID render
> (`BUG-410 credits scroll restored 6441 frames`, PostUpscaleProcgenBlend ran) -- so verify the animated
> START/END title card on a real obs render BEFORE changing; if missing/covered in still-only, make
> titles+credits an ALWAYS-ON procgen overlay (engine-independent). **BUG 3:** burn the image/audio/LLM
> model detail ONTO the rolling credits (the forensic detail I added is only in the _treatment.txt
> sidecar). **BUG 4:** audio bars always-on overlay (docs/2026-06-20-audio-bars-fix/). BUGs 2/3/4 share the
> always-on procgen-overlay layer. **NEXT = (a) the VISUAL FIXES above (BUG 1 first, GPU-verify each), (b)
> BUILD the 3D PoC (pass03+pass04 + PIN list), (c) watch nothing -- soak is done.** S3 forward order (3D =
> item 5) UNCHANGED.

> **LATEST SESSION -- 2026-06-19 (CODER; STORY-QUALITY PHASE 1 BUILT + PUSHED + RE-SOAK PASS;
> HEAD `ee83a88` == origin/v2.0-alpha; full suite 4600 passed/33 skipped, Bug Bible 16/7/3,
> test_audio_byte_identical green).** Built the roundtable-converged Phase-1 plan
> (`docs/2026-06-19-story-quality-analysis/roundtable/R2/pass02_plan.md`) -- the news->story fix
> -- in 6 dependency-ordered chunks, each suite+BugBible+no-BOM+AST green, committed AND pushed
> per chunk: **Chunk 1 A1** (`a1b05bb`, neutralize the dead `editor_constraints` diagnostic;
> run_story_qa already gated-off); **Chunk 3 A4** (`96db9aa`, new pure `nodes/_otr_anti_loop.py`
> -- deterministic near-dup + 'What if...?' loop detection over voiced lines, announcer taglines
> exempt, wired into the spine as Stage 3.6 recomposing CHARACTER targets); **Chunk 2 A3/A2**
> (`9a792b6`, run the mechanical floor UNCONDITIONALLY after the critic + UNION into
> reroll_targets since run_story_critic returns clean() identically on failure; replace the
> reroll-exhaustion needs_full_rerun terminal-skip with REPAIR-THEN-SHIP -- log residuals to
> `meta.a2_ship_through`, fall through to the normal freeze, never refuse; cast_lock's structural
> PD1 halt left intact); **Chunk 4 B1 THE SPINE** (`17d2e39`, new `nodes/_otr_dramatic_state_llm.py`
> -- derive the opposed wants+question+ending from `meta["news"]` at the writer call site via
> structured_call on the resident technical slot, post-validator requiring >=1 news term, a
> deterministic news-templated fallback opposed BY CONSTRUCTION, and a turning-slot key_terms
> floor; replaces the `_DEFAULT_A/B_WANTS` boilerplate; the pure `_otr_dramatic_state` stays
> pure); **Chunk 5 A5** (`db710a0`, new pure `build_line_dramatic_fields` adapter maps the
> news-driven slot contract -> the composer's beat_objective/obstacle/turn/subtext on the
> compose_line path; the LIVE exchange composer already consumes the same contracts); **Chunk 6
> A6** (`ee83a88`, new pure `nodes/_otr_line_hygiene.py` -- scrub parentheticals + self-vocative,
> RECOMPOSE truncation: character->spine seam, announcer open/close->the dedicated announcer
> composer; spine Stage 3.7). 36 new unit tests across the 6 chunks. **All content-only inside the
> FIXED ledger {cast,lines,meta}; ZERO JSON edits (the prod workflow is unchanged -- the new code
> runs inside the existing wired OTR_LedgerScriptWriter + OTR_LedgerFreezeCascade nodes).**
> **RE-SOAK PASS** (headless :8011, real `otr_scifi_16gb_full.json`, bark, visualizer, bypass
> OFF): leg-1 (mistral-nemo) `dramatic_state_source=llm`, wants genuinely OPPOSED and about the
> news (UCLA academic integrity vs startup commercialization at LABEST 2026), **`_DEFAULT_*`
> ABSENT**, `freeze_verdict=frozen_with_doctor_edits` (SHIPPED, no refuse), `a2_ship_through`
> stamped (A2 proven). Details: `docs/2026-06-19-story-quality-analysis/RESOAK_RESULTS.md`.
> **DEFERRED to Phase 2 (NOT done):** shape-follows-story; the use_exchange exchange composer
> enhancements; reconstruction-test validator; best-of-N; physical deletion of the neutralized QA
> modules. **NEXT:** Phase 2 (per pass02 scope) + the operator close-read of the remaining small-
> local re-soak legs (gemma-2-2b/4-E2B/4-12b). The S3 build forward order (sections 1/3) is
> UNCHANGED by this work.


> **LATEST SESSION -- 2026-06-19 (STORY-QUALITY ROUNDTABLE R1 CONVERGED; docs only, NO code; HEAD `af26492`
> == origin; build forward order sections 1/3 UNCHANGED).** Ran R1 of the story-quality campaign end to end.
> (1) Acted as the FIRST PANELIST with a fresh head -> `roundtable/pass01_opus-grounded.md` (A-E), grounding
> every claim against the REAL Windows code + the soak corpus (Desktop Commander, not the lagging mount).
> CONFIRMED: critic anti-correlation (leg_0002 best -> `needs_full_rerun`; leg_0031 worst loop ->
> `frozen_with_doctor_edits`); `_DEFAULT_A/B_WANTS` boilerplate (the news reaches only the *question*, not the
> wants); uniform 18-line mold (`act_count="auto"` -> `default_act_count`=3 for ALL >=300w; ledger has NO act
> field); lexical-only `post_assembly_keyterm_check`; ancient-DNA->aliens drift (leg_0013); 12B usable (leg_0011).
> (2) Operator supplied 6 external panel replies (`docs/story_passes/pass_1/`: chatgpt/claude/gemini-deep/grok/
> perplexity/deepseek) -- **the panels did NOT see the workflow JSON**. Read them + the private plan and synthesized
> `roundtable/pass01_architecture.md` as the grounded judge (KEEP/CUT; hallucinations cut). **Operator steers
> folded in: minimal-change/activate-not-overhaul; 'acts' are NOT a limiter; best story authentic to news; FIX
> stories not FAIL them; model-agnostic story path (per-model only for technically-unique reqs); rip-out/repurpose
> any QA-that-only-scores-or-fails (a grounded §5b inventory).** Grounded wiring the panels couldn't: story stage
> = 2 nodes (`OTR_LedgerScriptWriter` + `OTR_LedgerFreezeCascade`), critic/spine/QA/contracts all INTERNAL to the
> writer; saved `use_exchange=false` + `act_count="auto"`; `slot_drama_contract` already derives per-line
> obligations (pulls news key_terms) but is unconsumed on the single-line path; the critic `clean()`-fallback
> silently returns "strong"+no-targets = SKIPS repair (the "passes the worst" bug); `_otr_story_brief.py` is the
> VISUAL reflection brief, NOT the news brief (corrects gemini/perplexity). CONVERGED PLAN = 3 fix-moves
> (news-derived opposed wants at `derive_dramatic_state_from_meta` / call site :2780 -> deliver via the EXISTING
> contract+composer path -> critic-as-repair-driver, never a gate) + bounded shape (secondary) + the §5b
> rip-out/repurpose. **R2 DONE/CONVERGED (this session):** ran a live 2-pass roundtable on the coding plan
> (Gemini-3.1-pro + GPT-5.5 + DeepSeek-v4-pro, grounded vs the real seams; ~$0.30 total) ->
> `roundtable/R2/pass02_plan.md` is the build-ready Phase-1 plan (6 chunks A1-A6 + B1, all content-only,
> zero-JSON-targeted, audio byte-identical). Key roundtable-grounded fixes: B1's LLM call goes at the WRITER
> CALL SITE (the pure helper stays pure) + a post-validator (>=1 news term across wants/question/ending);
> CUT `stake`/`hook` (DramaticState has neither); run A4 mechanical checks UNCONDITIONALLY + union (the critic
> returns clean() identically on failure); inject the news-failure turning-slot detail INTO `key_terms`
> (validate_contract needs it); repair-then-ship ordered selection (never refuse); announcer truncation ->
> the dedicated announcer composer; neutralize the dead QA in Phase 1, defer physical deletion. **NEXT = R3
> wiring/build (operator-gated) -> R4 Comfy -> R5 polish (campaign DONE only at polish).** Build forward order
> (sections 1/3) UNCHANGED.

> **LATEST SESSION -- 2026-06-19 (ANALYSIS window; story-quality SIDE analysis, NOT a build; HEAD `af26492`
> == origin; docs only, NO code; build forward order sections 1/3 UNCHANGED).** Ran the operator-requested
> story-quality analysis off the overnight soak. Fanned out 6 READ-ONLY subagents + an Opus close-read of
> the corpus extremes, all grounded vs the REAL Windows files (Desktop Commander, not the lagging mount).
> **FINDINGS:** (1) the craft architecture (opposed wants, per-slot turn contract, exchange composer, QA)
> is BUILT but ships DORMANT -- production runs only format hygiene + a fixed 3-act/18-beat mold for any
> episode >=300 words (`_otr_episode_budget.ACT_COUNT_CONFIG`); (2) the story critic is ~ANTI-correlated
> with craft -- it gave the BEST leg (0002, blind 35/35) `needs_full_rerun` and "passed" the WORST (0031,
> 7/35, a verbatim loop) `frozen_with_doctor_edits` (verified vs `_server8011.log`); (3) word-count
> variance (34-287%) is a SYMPTOM of the fixed mold, NEVER a target; (4) 26% of legs wrote 0 words = a
> throughput/harness failure, not craft; (5) **THE NEWS IS NOT THE CRUX** -- the news brief DOES run
> (`news_briefs_required` default True, so premises are news-derived), but the news is demoted to soft
> context: the central conflict (the opposed wants in `DramaticState`) is hardcoded `_DEFAULT_A/B_WANTS`
> that IGNORE the news, the brief becomes a 15-word "theme is flavor, not structure" string, and the only
> news check anywhere is a lexical ">=2 key-terms appeared" audit (`post_assembly_keyterm_check`).
> **>>> NEW OPERATOR HARD CONSTRAINT (2026-06-19, NON-NEGOTIABLE): the news story MUST be the CRUX of the
> drama** -- folded into the deliverables as constraint 4A (content-only; exact fix seam:
> `derive_dramatic_state_from_meta` + call site `OTR_LedgerScriptWriter.py:2779`, drive the opposed wants
> from `meta["news"]`; the ledger `{cast,lines,meta}` wire format stays FROZEN; audio spine byte-identical).
> **DELIVERABLES (docs only, `docs/2026-06-19-story-quality-analysis/`):** `PROBLEM_STATEMENT.md` (code-
> illustrated, story-quality framed, news-as-crux a hard constraint) + `roundtable/PANEL_PROMPT_R1.md`
> (self-contained, paste-ready blind-panel prompt) + `_PRIVATE_OPUS_PLAN_DO_NOT_PASTE_TO_PANEL.md` (held OUT
> of the panel) + `README.md`. **CAMPAIGN MECHANICS (operator-set):** the OPERATOR runs ALL panelists himself
> (latest models, his pick; GPT-5.5 self-run) from the paste-ready file -- ZERO OpenRouter spend by Claude.
> **>>> NEW OPERATOR DIRECTIVE (2026-06-19, "surprise") -- THE NEXT WINDOW STARTS HERE:** the fresh window
> opens by acting as the **FIRST PANELIST** -- with a FRESH HEAD, read `roundtable/PANEL_PROMPT_R1.md` and
> ground every claim against BOTH the REAL code AND the REAL soak corpus (the 20+ leg scripts in
> `docs/2026-06-18-overnight-story-soak/scripts/`; it is the ONE reviewer that CAN check the repo + the data,
> unlike the external blind models), and write its own grounded panelist answer to
> `roundtable/pass01_opus-grounded.md` BEFORE opening the private plan or synthesizing. **GROUNDING BAR
> (operator): the goal is a genuinely BETTER STORY -- do NOT propose rewrites for their own sake; a lever
> ships ONLY if soak+code evidence shows it actually makes the STORY better. If it wouldn't improve the
> story, don't propose it.** (This AMENDS the brief's "Opus is never a listed panelist" line -- operator
> override.) THEN: operator runs the external blind panels -> Opus reads all + the private
> plan and synthesizes `roundtable/pass01_architecture.md` -> Rounds 2 (coding) / 3 (wiring) / 4 (Comfy
> workflow) / 5 (polish). The `/otr-handoff` for this analysis is DONE only at polish. The build forward
> order (sections 1/3) is UNCHANGED and resumes after the campaign.

> **LATEST SESSION -- 2026-06-19 (PLANNER/SOAK; HEAD `af26492` == origin; suite green, Bug Bible 16/7/3).
> SHIPPED + PUSHED earlier this session (each suite-green): bark runaway-clip cap + trailing-pad trim
> (`cc6bacc`, the caption-fit fix -- bark was over-generating ~12s for 16 words) + the `default_clean`
> commercial-clean cast voice bank (`af26492`, routes the char engine to chatterbox/dia, excludes the
> non-commercial indextts2). Also ran the voice-engine roundtable (converged) + banked the curated voice
> cast. THEN ran a 10-HOUR OVERNIGHT STORY-QUALITY SOAK (operator-requested data-gathering run, NOT a
> build): 35 legs, 13 writer LLMs (7 OpenRouter ~latest + 6 local), 320/400/500-word targets, news-driven
> premises, all-visualizer, dedicated :8011 server. Output: `docs/2026-06-18-overnight-story-soak/`
> (story_soak_results.csv/.json + 28 per-leg script .txt). Soak ended cleanly at the 10h cap.
> **>>> THIS HAND-OFF OPENS A SIDE ANALYSIS WINDOW** (does NOT change the build forward order): a
> STORY-QUALITY ANALYSIS that fans out read-only subagents over the soak corpus + the ledgers + the
> story-gen architecture, forms its own judgment, and produces a PROBLEM STATEMENT for `/roundtable`.
> **OBJECTIVE = STORY QUALITY (craft), NOT word count (operator 2026-06-19: "I don't want to chase word
> count, I want to chase story quality -- hard for a computer to decide but do your best").** The analysis
> LEADS with a qualitative close-read of the 28 scripts (+ an LLM-judge rubric + the pipeline critic).
> Supporting signals (diagnostics, NOT targets): n_lines pinned at ~18 regardless of length = one-size
> story SHAPE + the caption-fit root; story critic 0/19 clean; word-count adherence wild (34-287%).
> Full brief + the subagent plan + the HARD ledger-intact constraint (the `{cast,lines,meta}` schema is
> FROZEN downstream; audio is the first consumer):
> `docs/2026-06-19-story-quality-analysis/STORY_QUALITY_ANALYSIS_BRIEF.md`.
> **The handoff completes via a MULTI-ROUND ROUNDTABLE CAMPAIGN (operator design, brief section 5A):** a
> broad, CODE-ILLUSTRATED problem statement -> exactly 3 TOP-TIER panelists judging architecture BLIND
> (Opus holds its own best plan privately + is the sole judge/synthesizer), then a SEQUENCE of rounds --
> R1 architecture/creative approach, then coding plan, wiring, Comfy workflow, polish. The `/otr-handoff`
> is DONE only when the campaign runs through polish + the final converged plan is recorded. The S3 build
> forward order
> (3D item 5, distribution item 6) + the carried per-segment LUFS/RMS plan are UNCHANGED by this analysis.

> **LATEST SESSION -- 2026-06-18 (CODER; LUMINA-IMAGE 2.0 PROMOTED TO A VALIDATED PEER IMAGE ENGINE,
> GPU-VERIFIED; 2 commits on `v2.0-alpha`, PUSHED; HEAD `631d0b0` == origin; suite green [2 pre-existing
> test_model_catalog_scan environmental reds unchanged], Bug Bible 16/7/3, audio byte-identical green).**
> Operator ask: "code the luminous / low-VRAM image engine -- the scaffolding should already be there."
> AUDIT FINDING: the scaffolding was MORE than there -- `lumina_image` was already a COMPLETE, registered,
> cold-import-clean adapter (real `render_image`, fail-closed `assert_usable`, CAPABILITIES row, 7 CPU tests,
> a dep-pilot row, weights on disk). The ONLY gap was the tested-only dropdown gate: it was hidden pending a
> GPU smoke. So this was a verify-and-promote, not a build.
> **GPU SMOKE (RTX 5080, fresh FLOOR-lane headless server):** the EXACT engine recipe (UNETLoader
> lumina_2_model_bf16 + CLIPLoader[type=lumina2] gemma_2_2b TE + VAELoader lumina2_ae ->
> ModelSamplingAuraFlow -> KSampler -> VAEDecode) minted a real 1216x832 still in 23.5 s -- `model_type FLOW`,
> TE 4986 MB + diffusion 4977 MB staged SEQUENTIALLY, resident peak ~12.2 GB << 14.5 GB ceiling (and the
> engine's `render_image` reclaims after decode). The minted PNG is a real varied image (std 40, not flat).
> Box reset clean (GPU baseline 1.69 GB). **Commit `ed560a0`:** add `lumina_image` to `registry.VALIDATED_ENGINES`
> (now lists in the OTR_ImageDirector per-role dropdown via `validated_engine_names`); measured-VRAM note on
> the CAPABILITIES row; `test_tested_only_dropdown_gate` moves lumina out of `_HIDDEN_IMAGE` into the validated
> set. **Commit `631d0b0`:** version `scripts/_otr_lumina_image_smoke.py` (the reusable image-engine GPU smoke
> -- the counterpart to the video-only `_otr_single_engine_smoke.py`; routes the still through the `_otr_paths`
> authority so the output-tree contract stays clean). The validated IMAGE set is now flux_gen1 (default) +
> z_image_turbo + flux2_klein + lumina_image. **>>> NEXT = the S3 forward order** (3D item 5, distribution
> item 6) + the carried per-segment LUFS/RMS normalization plan (frozen-spine, roundtable-first), UNCHANGED by
> this image-lane add.

> **LATEST SESSION -- 2026-06-18 LATE (CODER; VIDEO CLIP-FILL / DYNAMIC-VRAM SHIPPED + GPU-VERIFIED;
> + the QUEUED green bottom-bars overlay SHIPPED; 4 commits on `v2.0-alpha`, NOT pushed (operator gates);
> suite 4554/0, Bug Bible 16/7/3).**
> The wan_ti2v static-episode bug is FIXED at root (no shims) -- spec was
> `docs/2026-06-18-video-clip-fill/CLIP_FILL_HANDOFF.md`; results in `GPU_VERIFY_RESULTS.md`.
> **Commit `c9bf0ab` (clip-fill, 5 pieces):** (1) `motion_common.compute_real_frame_budget` + `free_vram_mb`
> PREDICT the VRAM-affordable 4n+1 frame count from a zero-cost `torch.cuda.mem_get_info` read + a cost model
> (wan seed 7000MB + 185MB/frame @1472x832, per-frame scales with pixel area; budget = min(free,ceiling)*0.85,
> capped at the beat target, floored at the motion floor) -- NEVER react-to-OOM. (2) `eng_wan_ti2v._floor_length`
> calls it (the hard 17-frame cap is gone; `OTR_WAN_TI2V_MAX_FRAMES` is now an absolute hard cap). (3)
> `wrapper_bridge.extend_frames_to_target` seamless ping-pong-extends the short render to the beat target in
> `render_clip` (LTX boomerang untouched -> byte-identical). (4) `render_driver.persist_episode_clips` +
> `_otr_paths.otr_clips_dir` move final clips out of the swept `_shared/tmp` into durable `episodes/<ep>/clips/`
> (wired in OTR_VideoRenderBatch before the manifest). (5) `otr_silent_composite._warn_clip_underrun` LOUD-warns
> (never raises) when a real clip is far shorter than its target. **Commit `8ce9004` (pre-existing red fix):**
> `test_story_brief_c5a1::test_schema_caps_string_v2_fields` still expected reject+retry; structured_call now
> CLAMPS an over-long capped field in place (4c0e943) -- updated the assertion (this was failing at HEAD `4d8c2ba`,
> unrelated to clip-fill). **GPU SMOKE (5080, all-roles wan_ti2v 60w act=1 bark): SUCCESS.** 6/6 beats clip-filled
> with ADAPTIVE budgets (29->238, 25->290, ...; the budget shrank 29->25 across beats = live VRAM read working),
> every render peak 8610-9988 MB << 14500 ceiling (no OOM), persisted 6 clips to `episodes/<ep>/clips/`, a beat clip
> is 290 frames/11.6s with 3 distinct frames at t=2/3.5/5s (motion fills the beat, freeze GONE),
> `audio_byte_identical OK`, obs_publish OK.
> **BARS OVERLAY ALSO SHIPPED (commit `13ceeae`):** the QUEUED optional green audio-reactive bottom-bars overlay
> (operator-requested "after the gpu smoke") -- a composite-stage add to `OTR_SceneAwareScopes` (NOT a new engine):
> new `landscape_bars` widget (off DEFAULT|bottom, appended last per BUG-LOCAL-097, wired into node 94 of the real
> JSON same-commit); `bottom` paints a GREEN-ONLY freq strip (shared `scope_draw.freq_bars_green`) along the bottom
> of any LANDSCAPE clip, ABOVE the lower-15% caption safe-area (captions never occluded); `off` = byte-identical.
> Verified on the REAL persisted wan clips: off->all suppressed, bottom->all landscape clips get bars; strip ends at
> y=918 = caption-safe-area top. 7 tests. **>>> NEXT = the S3 forward order** (3D item 5, distribution item 6) +
> the carried per-segment LUFS/RMS normalization plan (frozen-spine, roundtable-first). Box reset clean (GPU baseline).
> Push is operator-gated (4 commits: `8ce9004`+`c9bf0ab`+`11d0e9a`+`13ceeae`).

> **LATEST SESSION -- 2026-06-18 EVENING (resilience + image curation; HEAD `3384418` == origin; suite green,
> Bug Bible 16/7/3). ACTIVE NEXT STEP = the VIDEO CLIP-FILL / DYNAMIC-VRAM build (spec:
> `docs/2026-06-18-video-clip-fill/CLIP_FILL_HANDOFF.md`).**
> SHIPPED + PUSHED to `v2.0-alpha` this session (each suite-green): **(a)** OpenRouter resilience -- a DELETED model
> (gone-404) now falls over to the slot's remote fallback ONCE, loudly, instead of aborting the episode
> (`OpenRouterModelGoneError`); **(b)** fixed the `KeyError 'openrouter:slot-a'` that aborted a run when the CREATIVE
> writer was a remote slot (creative_prompt_router `.get` -> modern default for non-curated ids); **(c)** OpenRouter
> A/B writer slots now show TEXT-output models ONLY (cache captures `architecture.output_modalities`; image gens like
> gemini-3-pro-image hidden); **(d)** structured_call CLAMPS an over-long capped field (e.g. outline `time_of_day`
> max 40) to its max_length instead of aborting; **(e)** the style inventor PADS a weak-model near-miss (4-of-5
> descriptors) up to 5 from a deterministic stock pool instead of aborting (operator: "no loud fail"); **(f)**
> wan_ti2v is now eligible for ALL roles (incl. announcer) + fixed flat_still's missing CAPABILITIES row; **(g)**
> DROPPED the chroma_hd image engine (`3384418`, operator values call: Chroma is a de-restricted/uncensored FLUX
> finetune -- OTR will not ship that path). All graceful-degradation "floors" follow the 3-source roundtable
> (do NOT use an LLM to repair LLM output; Python deterministic floor; LTX path byte-identical).
> **IMAGE LANE:** the 3 most-downloaded 16GB ComfyUI image models (FLUX.1-dev=flux_gen1, Z-Image-Turbo=z_image_turbo,
> FLUX.2-Klein=flux2_klein) are ALL already wired + validated. `lumina_image` is the easy SFW add NEXT -- code is
> complete (real render_image) and all its files are ALREADY on disk (`lumina_2_model_bf16` + `gemma_2_2b_fp16` +
> `lumina2_ae`), just needs `OTR_ENABLE_LUMINA=1` + a GPU smoke + promotion to `VALIDATED_ENGINES`.
> [DONE 2026-06-18: `lumina_image` GPU-smoked + promoted to `VALIDATED_ENGINES` -- commits `ed560a0`/`631d0b0`;
> see the LATEST SESSION block at the top. The validated IMAGE set is now flux_gen1 + z_image_turbo +
> flux2_klein + lumina_image.]
> **>>> ACTIVE NEXT = the video CLIP-FILL bug:** a wan_ti2v episode renders STATIC (only the procgen overlay moves).
> Root cause (code-grounded): `eng_wan_ti2v._floor_length` clamps every clip to a hard 17-frame "8GB floor"
> (`_TI2V_FLOOR_MAX_FRAMES=17`), ignoring the shot calculator's audio-derived per-beat `target_frame_count` (~280);
> the composite then holds-last-frame -> 0.68s motion + ~9s freeze. ALSO clips write to swept `_shared/tmp` (no
> persistent `clips/` folder). The roundtable-converged fix (5 pieces: dynamic-VRAM PREDICT frame budget on
> MotionEngineBase, wan honors it, loop/ping-pong-extend to target, persist clips to `episodes/<ep>/clips/`, composite
> LOUD underrun guard) is fully specified in `docs/2026-06-18-video-clip-fill/CLIP_FILL_HANDOFF.md`. BUILD THAT NEXT.
> **QUEUED FOLLOW-UP (gated on the visualizer engine being committed + pushed green):** the optional GREEN
> audio-reactive BARS overlay (old-school bottom-of-screen, over ANY engine's final video) -- a COMPOSITE-STAGE add
> to `OTR_SceneAwareScopes` + `OTR_PostUpscaleProcgenBlend` (NOT a new engine). One new `landscape_bars =
> off(DEFAULT)|bottom` widget (off = byte-identical; APPEND at end of widgets_values, BUG-LOCAL-097); landscape clips
> return a new `("bars",x,y,w,h)` bottom-strip mode instead of None; paint green-only via the shared
> `_otr_shared/scope_draw.py` (DRY); HARD: captions MUST layer ABOVE the bars (caption burn ~Node 58 is the LAST
> composite step + keep bars below the lower ~12-15% caption safe-area + a test asserting caption-after-bars); wire
> the widget into otr_scifi_16gb_full.json same-commit + re-validate. Full spec:
> `docs/2026-06-17-scope-visualizer-engine/CODER_KICKOFF_BARS_OVERLAY.md`.

> **LATEST SESSION -- 2026-06-18 (CODER; flux2_klein VERIFIED + PROMOTED + coverage-arch accepts_still SHIPPED;
> HEAD `64c14bb` == origin; suite 4500/33, Bug Bible 16/7/3). flux2_klein is DONE.**
> Two commits pushed to `v2.0-alpha`:
> **(1) `2cb25fb` coverage-arch + flux2_klein TE fix.** Root-caused the flux2_klein `mat1/mat2 (512x15360 @
> 7680x3072)` error: klein-4B uses the **Qwen-3-4B** encoder (7680-wide), NOT flux2-dev's Mistral (15360, 2x).
> Found the klein-matched TE via the official ComfyUI `image_flux2_klein_text_to_image` template; downloaded
> `qwen_3_4b.safetensors` (8 GB, `Comfy-Org/flux2-klein`) into `C:\ComfyUI-Models\text_encoders`; engine default
> TE -> qwen. Built the **coverage architecture** (operator: "all video/3D accept whatever image is selected, one
> place, no per-model whitelist"): `accepts_still` capability -- `MotionEngineBase` default True (every motion lane
> takes the selected still), `ltx_av_music`/`visualizer` opt-out False -- read centrally by `engine_consumes_still`
> in the image dispatcher (dual-read fallback to init_image-in-required_inputs; bare except -> LOUD). This makes
> silent `ltx_video` consume the flux2 still (it was skipped before). 2-model roundtable (gpt-4.1 + gemini-pro-latest
> via DIRECT APIs -- OpenRouter launcher stalled; keys in HKCU User env), grounded in
> `docs/2026-06-18-coverage-arch-wiring/`. **(2) `64c14bb` PROMOTION + creativity dial.** GPU-VERIFIED on the 5080:
> a full flux2->silent-LTX episode (all image roles=flux2_klein, all video=ltx_video, 30w, **maximum-chaos**
> creativity, bark) minted ALL 6 stills end-to-end -- `[OTR.image.flux2_klein] minted still 832x480/1472x832
> steps=20 guidance=4.00`, qwen TE staged 7671 MB, NO dim error; the LTX clips then rendered FROM those flux2 stills
> (the operator's "flux2 images on LTX"). Added `flux2_klein` to image `VALIDATED_ENGINES` (opt-in via
> `OTR_ENABLE_FLUX2_KLEIN`), out of `_HIDDEN_IMAGE`. Added `creativity` to `CREATIVE_WHITELIST` so a soak can set the
> writer dial. Box reset clean between runs.
> **>>> NEXT = the coverage-arch FOLLOW-UPS (deferred, additive; docs/2026-06-18-coverage-arch-wiring/pass01_plan.md):**
> optional_inputs for role_compat honesty (verify-at-build); optionally set accepts_still=True on the static-still
> cheap families (flux_still/station_card/still_kenburns) so they also show the SELECTED image; full Decision-3/5
> (central usable(), retire requires_mesh_portrait onto still_kind). THEN the S3 forward order (3D item 5,
> distribution item 6) + the carried per-segment LUFS/RMS plan.

> **LATEST SESSION -- 2026-06-18 (CODER; flux2_klein BUILT + image-pick DEDUP + coverage-arch roundtable;
> HEAD `c854406` == origin; suite green; Bug Bible 16/7/3). PRIORITY NEXT STEP = finish flux2_klein (klein-4B).**
> Four committed + PUSHED to `v2.0-alpha`:
> **(1) flux2_klein engine BUILT** (`36dc01a`): FLUX.2 [klein] 4B is a real image engine (official ComfyUI flux2
> recipe -- UnetLoaderGGUF + CLIPLoader[type=flux2] + FluxGuidance->BasicGuider + Flux2Scheduler +
> SamplerCustomAdvanced + VAEDecode). Apache-2.0 (commercial_clean=True). Graduated out of the stub PEERS matrix
> into its own suite `tests/test_flux2_klein_engine.py`. Stays HIDDEN/opt-in until a green GPU render promotes it.
> **(2) Image-MODEL selection collapsed to ONE place** (`b8bb388`, operator "only in one place not two"):
> removed the 3 duplicate image dropdowns from OTR_ImageDirector; it now sources picks SOLELY from
> `video_policy["image_models"]` (OTR_VideoDirector is the single home). Edited the real source-of-truth JSON node 88
> in place + widget_mapping.json (image keys -> VideoDirector only). Added `OTR_COMBO_*_IMG` env to the soak harness.
> **(3) Coverage-architecture roundtable** (`50200fd`, operator "all video accepts all images, approval in one
> place"): grounded pass01 synthesis in `docs/2026-06-18-image-video-coverage-arch/` -- add `accepts_still`/
> `still_kind` capability fields to video adapters (NOT a new type), ONE central `image_engines.registry.usable()`
> approval surface the directors+dispatcher share, dual-read migration (required_inputs wins for existing names),
> unify the 3D mesh-portrait lock onto `still_kind`. (4/6 panel models errored on token budget; gemini+grok gave
> grounded critiques -- re-run with raised max-tokens for the rest.)
> **(4) flux2_klein GPU VERIFY -- FAILED, diagnosed** (`c854406`): a full flux2+LTX episode (announcer/beats=
> ltx_av_talk, music=ltx_av_music, all image=flux2_klein, 30w act=1, bark) found two issues: (a) FIXED -- weights
> were in `Documents\ComfyUI\models` but the headless server scans `C:\ComfyUI-Models` (moved all 3 there);
> (b) OPEN -- the sampler raises `mat1/mat2 (512x15360 @ 7680x3072)`: the klein-**4B** GGUF is paired with the WRONG
> text encoder (I reused flux2-**dev**'s `mistral_3_small_flux2_fp4_mixed`, whose conditioning is 2x too wide for
> the 4B UNet). klein-4B needs its OWN ComfyUI-matched encoder. FULL RESUME RECIPE +
> the proven combo-soak invocation: `docs/2026-06-18-flux2-klein/VERIFY_ON_5080.md`. Box reset clean (GPU baseline).

> **TASK 2 PROGRESS (`4a92ed6`; HEAD afa6bf1 code): visualizer-all-roles soak via `_otr_combo_soak.py` (forces bark,
> clearing the headless-audio gap) found + fixed FOUR visualizer integration bugs -- all pushed + suite-green:
> `d460797` assert_usable no longer pre-gates audio_ref; `bad1bba` render_driver feeds b000 the master-audio slice;
> `c5c14c9` idle scopes on silent beats; `afa6bf1` 0-frame beats default to 1s. The visualizer rendered 21 real
> clips across the soaks and is now robust to every beat type (real / silent / zero-frame). A single status=success
> episode is NOT yet captured -- the confirming soak died in the WRITER's style-inventor (a transient LLM flake,
> UNRELATED to the visualizer). NEXT: one more visualizer-all-roles soak (try 30-60w to dodge the writer flake) ->
> status=success -> ADD `"visualizer"` to `registry.VALIDATED_ENGINES` (+ default-ON). Then sweep the other 8 video
> + 2 image validated engines via `_otr_combo_soak.py` (force bark). Full detail: SMOKE_SWEEP_RESULTS.md. Box reset
> clean (GPU baseline; temp visualizer extra-env deleted).**
>
> **TASK 2 EARLIER (`68b0e31`): full-episode GPU smoke sweep is BLOCKED on a headless-AUDIO env gap.** A
> visualizer-all-roles attempt aborted in the AUDIO phase (`RuntimeError: IndexTTS2 Path B not installed` -- the
> default char voice's isolated venv is absent on the ComfyUI-Installs box) BEFORE any video engine ran; this blocks
> EVERY full-episode leg. `queue_smoke.py` renders the as-saved indextts2; only the SOAK HARNESS forces
> `OTR_SOAK_CHAR_VOICE=bark`. UNBLOCK: (a) install indextts2 headless (`scripts\_otr_indextts2_install.ps1`) then
> sweep via queue_smoke, OR (b) sweep via `scripts/_otr_combo_soak.py` (forces bark). Visualizer render path is
> unit-proven (17 tests + a real ffmpeg render); promotion to VALIDATED_ENGINES stays gated on a green full-episode
> E2E + audio byte-identical. Verified sets: VIDEO 8 + IMAGE 2 (see SMOKE_SWEEP_RESULTS.md). Box reset clean; the
> temporary visualizer force-map `_marathon_extra_env.cmd` was DELETED so future boots are clean.
>
> **LATEST SESSION -- 2026-06-18 (CODER, autonomous overnight; HEAD `236db0e` == origin; suite 4496 pass / 33 skip,
> Bug Bible 16/7/3):** A long multi-thread session, all committed + PUSHED to `v2.0-alpha`:
> **(1) Still-aspect correctness + HuMo dropdown labels** (`55d9dad`). **(2) Bark whiny-voice fix S1-S4** + preset
> audition (`215738c`,`6a20c9f`). **(3) wan_ti2v promotion + render_single aspect fix** (`ca3e06c`). **(4) wan_ti2v
> 8GB-floor hardening, 3-pass roundtable-converged** (`00da690` Chunk1: 17-frame floor+clamp / euler-only sampler
> whitelist / range-checked resolver / VAE allow-list / 0.1s probe; `1587c31` Chunk2: CLIP off fp8 -> GGUF umt5 [fp8
> is Mac-MPS-broken, #9255] + tiled VAE; docs `d528084`). **(5) VISUALIZER ENGINE v1 SHIPPED** (`236db0e`, Task 1):
> the resurrected full-colour procedural CRT scope engine (engine_id `visualizer`) -- low-VRAM ffmpeg-only per-beat
> picture (ring/particles/grid/waveform/bars/CRT post), torch-free draw routines COPIED into
> `nodes/_otr_shared/scope_draw.py` (zero coupling to the floor node / SceneAwareScopes overlay), adapter
> `eng_visualizer.py` (family=abstract, has_audio=False, NO fallbacks). PLAN-PREMISE FIX: `visualizer` was already a
> cheap floor stub (`VisualizerFamily`); the new engine SUPERSEDES it (stub deleted, removed from FLOOR_NAMES +
> cheap-render tests, dep-pilot row added). 17 tests + a real CPU ffmpeg render.
> **>>> NEXT = TASK 2 (operator-requested, GPU/overnight): the 120-word verified-model smoke sweep.** Enumerate the
> validated VIDEO set (`registry.validated_engine_names()` = ltx_video, ltx_av_music, ltx_av_talk, humo, humo_1.7B,
> humo_1.7B_169, humo_14B_169, wan_ti2v) + the validated IMAGE set (OTR_ImageDirector: flux_gen1, z_image_turbo)
> PROGRAMMATICALLY; run a 120w full-pipeline smoke per engine on the REAL `otr_scifi_16gb_full.json` (random
> OS-entropy seed), RESET the box before EACH (CLAUDE.md sec 4), INCLUDE one `visualizer`-all-roles run, verify
> `test_audio_byte_identical`, write `docs/2026-06-17-scope-visualizer-engine/SMOKE_SWEEP_RESULTS.md` (engine,
> pass/fail, time, VRAM peak vs ceiling). **THEN: if the visualizer all-roles E2E is green, ADD `"visualizer"` to
> `registry.VALIDATED_ENGINES`** (+ decide default-ON) so it lists in the dropdowns. NOTE: wan_ti2v's recipe changed
> in the floor hardening (euler/17-frame/GGUF-umt5/tiled) -- its sweep leg double-checks the hardened floor renders
> + fits. Fail LOUD on any regression; reset per CLAUDE.md sec 4 between legs. Box currently free (GPU baseline).
>
> **LATEST SESSION -- 2026-06-18 (CODER; BARK WHINY-VOICE FIX S1-S4 SHIPPED; HEAD `6a20c9f`; NOT pushed yet;
> suite 4462 pass / 33 skip, Bug Bible 16/7/3):** The sanctioned upstream-TTS audio work
> (`docs/2026-06-17-bark-voice/BARK_VOICE_UPDATE_PLAN.md`) is DONE across two commits. **`215738c` (S1+S2+S4):**
> the thin/whiny Bark timbre came from a single flat `temperature` over-randomizing the acoustic stages. Bark is a
> 3-stage pipeline (semantic -> coarse -> fine); the fix sets each stage independently (semantic 0.7 warm for
> content commitment; coarse/fine 0.5 to firm up the acoustics) and routes them via the transformers
> `BarkModel.generate` PREFIXED-kwargs contract (`semantic_temperature`/`coarse_temperature`/`fine_temperature` --
> prefixed wins, routes to that sub-model; NO generation_config mutation -> no cross-voice leak; transformers 5.5.0
> probed). `_generate_single_line` gained keyword-only stage temps (back-compat: legacy `temperature=` still works,
> each None stage inherits it -> the orchestrator health probe is untouched) + a pure `_stage_temps_for_line`
> helper that applies the intl (0.55) and first-line (0.6 en / 0.5 intl) CAPS to the SEMANTIC stage only (a
> ceiling, never a floor; coarse/fine keep profile values). `eng_bark` resolves the 3 temps from char_bark_v1 via a
> precise alias ladder (honors explicit 0.0; class-attr fallback 0.7/0.5/0.5). **S4:** verified EpisodeAssembler
> already peak-normalizes per segment + a final -1.0 dBFS master limiter post-crossfade, so `normalize_bark_output`
> stays false (a Bark-side normalize would double-hit); residual "low" is RMS/LUFS, deferred. **`6a20c9f` (S3):**
> rendered the audition on the 5080 across v2/en_speaker_0..9 -- en_speaker_5 (LUFS -14.3) + en_speaker_0 (-16.8)
> are the fullest/least-thin by ~10 dB; added `recommended_speakers=[5,0,6]` to char_bark_v1 as DATA ONLY (no
> auto-filter consumer; casting enforcement is a future sprint). `scripts/bark_preset_audition.py` is the reusable
> GPU harness (CSV + manifest committed; WAVs local-only). **NOTES:** no unit-level byte-identical fixture to
> re-baseline (indextts2 is the char_voice default; a forced-Bark re-baseline is an operator GPU action). The
> render verdict ("warmer/fuller, not whiny") still wants the operator's EARS on the audition WAVs in
> `%TEMP%\bark_audition\`. Comfy Desktop was killed for the audition -- RESTART it to load the new Bark code. NOT
> pushed (operator gates). The S3 forward order (Wan smoke / 3D / distribution) is UNCHANGED.
>
> **LATEST SESSION -- 2026-06-18 (CODER; STILL-ASPECT CORRECTNESS + HUMO LABELS SHIPPED + PUSHED; HEAD `55d9dad`
> == origin; suite 4443 pass / 32 skip, Bug Bible 16/7/3):** The current-step plan
> (`docs/2026-06-17-still-aspect/STILL_ASPECT_AND_LABELS_PLAN.md`) is DONE in one commit `55d9dad`.
> **GOAL A (still-aspect):** every registered video/3D engine now declares an EXPLICIT `render_aspect` in
> {portrait, wide} -- the portrait fallback is no longer load-bearing. ROOT-CAUSE FIX: `ltx_video` declared none ->
> silently defaulted portrait -> the 16:9 render decapitated heads; now `render_aspect="wide"` (class attr only,
> the frozen sampler/sigma chain untouched). Also wide: the cheap-families base (abstract/still_kenburns/
> station_card/visualizer/flux_still + still_parallax/mesh_stage via the base), `triposr`, `wan_i2v`, `wan_ti2v`.
> Stay portrait: `humo`, `humo_1.7B`. Stay wide: `humo_1.7B_169`, `humo_14B_169`, `ltx_av_*`. The three
> `requires_mesh_portrait` 3D talkers (`triposg_talk`/`hunyuan3d_talk`/`trellis_talk`) get `render_aspect="portrait"`
> (they feed the mesher a portrait still even when the final video is 16:9 -- the one exception). **GOAL B (labels):**
> the OTR_VideoDirector dropdown shows an aspect-DERIVED label (`humo (portrait)` vs `humo_1.7B_169 (16:9)`); the
> suffix is generated from `render_aspect` so it never drifts. The saved/looked-up VALUE stays the BARE engine id:
> `direct()` parses the token before the first `" ("`, and a bare legacy value + the ADD_CUSTOM sentinel pass
> through unchanged. The applier `_is_engine_director_admissible` strips the same suffix before the registry check
> so a fresh labelled save also applies. **NO workflow-JSON change** -- the bare saved values resolve via the
> back-compat parse (a trial JSON relabel broke 4 production-pin tests; reverted -- the JSON stays bare by design).
> New `tests/test_still_aspect_and_labels.py` (13) + the tested-only dropdown gate updated to parse labels.
> **>>> NEXT STEP = the BARK "whiny/low" voice fix** (`docs/2026-06-17-bark-voice/BARK_VOICE_UPDATE_PLAN.md`, the
> sanctioned upstream-TTS audio work). The S3 forward order (Wan smoke / 3D / distribution) is unchanged.

## 1. CURRENT STEP

**>>> ACTIVE (2026-06-25 LATE, CODER) = leak-floor-v2 + story-ledger DRIFT BOTH BUILT + SHIPPED + PUSHED.**
origin/v2.0-alpha HEAD `db9d8ea7` == local. 7 commits, each suite-green vs the 5 pre-existing `267a53e` fails +
Bug Bible 16/7/3 + AST/no-BOM: leak-floor-v2 `95bbcbd2` (DEFAULT-OFF/dark); the operator-flagged central-object
announcer leak `5d140b82`; story-ledger DRIFT C1 `d1bb9e7d` (kill critic fail-open / ArcVerdict unverified) -> C2
`fbc829ac` (pure `assert_ledger_consistency` + parity test, grounding-corrected to run in the WRITER since CastLock
is downstream) -> C3 `e6ad1b0d` (widget-ORDER + vintage-ledger CI guards) -> C4 `9f725f4a` (freeze WARN taxonomy) ->
C5 `db9d8ea7` (whole-episode critic context + cut StanceIssue). Full narrative in the CURRENT STEP block at the very
top of this file. **NEXT = the forward-order engine track (section 3) or operator pick.** OPEN VERIFY (operator,
GPU): promote leak-floor-v2 to default-ON after a live 320w validation per writer lane; eyeball
`meta.consistency_status` + `meta.freeze_warn_taxonomy` on the next soak. prod/main + tags GATED.

**>>> SUPERSEDED (2026-06-25, CODER) = SCHEMA-ADHERENCE -- LEVER-1 LOAD-BEARING SHIPPED (`516644eb`, `d4ca6cd4`);
C4 DEFERRED; Lever 2 DROPPED via G1; SPRINT COMPLETE. The full build narrative is in the CURRENT STEP block at the very top.
TL;DR: Lever 1 = TOLERANCE built from `docs/2026-06-25-schema-adherence/roundtable/pass04_plan.md`, refined by
two live grounding roundtables (nested-fork/ + c4-scope/). C0+C1+C2+C5+C6 (`516644eb`) + C3 ladder (`d4ca6cd4`)
SHIPPED + pushed: the proven nested Opus `normalize_length` failure validates on attempt 1 via the shared
`apply_field_aliases`/`__otr_field_aliases__` before-validator; structural rung is JSON-syntax-only (token-burn
fix); canonical happy path byte-identical; NO workflow-JSON change. C4 (schema-in-repair) DEFERRED -- the proven
failure is fixed so it would test dead code; recipe ready in c4-scope/. Lever 2 = BINARY decomposition still
DROPPED via G1 (genuine residual ~0 over 638 ledgers / 5,513 lines; the 15.3% broad-flag was 0/40 genuine on
inspection -- names, 3rd-person dialogue verbs, in-character spoken commands; `binary/G1_RESULTS.md`). Each chunk
shipped suite-green vs the 5 pre-existing `267a53e` fails + Bug Bible 16/7/3. **SCHEMA-ADHERENCE SPRINT COMPLETE;
NEXT = the forward-order engine track (section 3) or operator pick. prod/main + tags GATED.**

**>>> SUPERSEDED (2026-06-22, PLANNER) = STORY-QUALITY LIFT -- the 4-round roundtable CONVERGED; build-ready
coder kickoff `docs/2026-06-22-story-quality-lift/roundtable/pass04_plan_FINAL.md`. NEXT = operator GO to
BUILD in a coder window: manual no-bypass baseline re-smoke -> D1 leak -> D3 role -> D2 stance; D4
escalation OUT OF SCOPE. Full defect->fix map + spend in the ROUNDTABLE block at the very top of this file.**

**>>> SUPERSEDED (history): VOICE-CASTING ARCHITECTURE, VC Chunks 2-4 (2026-06-22; HEAD `3cc8de6`). Roundtable R1
CONVERGED + Chunk 1 (library coverage gate) + the PD voice-library ingestion (bank 137 -> 149, 4 distinct PD
voices) are SHIPPED + pushed; suite 5001/34, Bug Bible 16/7/3.** Build plan:
`docs/2026-06-22-voice-casting-arch/roundtable/pass01_plan.md`. The remaining chunks touch the
VOICE-ASSIGNMENT DETERMINISM path (audio byte-identity-sensitive) -- build with the byte-identical golden
front-of-mind, default-on, $0 deterministic seeded fallback:
  - **Chunk 2** -- two-lane identity refine (`voice_ref_id` cloners / `voice_preset` bark) + the v2<->ref map.
  - **Chunk 3** -- stamp `meta.cast_voice_slots` so CastLock's `_auto_registry` matches on timbre/age, not just
    gender (NOTE: `EnsembleSlot` has timbre/role but NO age_band yet -- add it or carry age in the stamp).
  - **Chunk 4** (operator's CORE ask) -- the HYBRID LLM voice-fit: the LLM proposes a `voice_ref_id` from the
    selected engine's voice cards; Python VALIDATES (gender/engine/exists) + falls CLOSED to the seeded
    `assign_voice_for_slot` scorer; stamp `meta.voice_cast_decision`. Acceptance: same cast => same voices
    (deterministic), no `voice_preset=None`, byte-identical golden holds (or recapture deliberately).
  OPEN remediation (operator supplies PD LibriVox titles -> I ingest via `scripts/otr_ingest_pd_voices.py`):
  female-elder = 0, no child/teen, no androgynous/other, cloners still male-light. Acceptance per chunk: full
  suite + Bug Bible green, ZERO workflow-JSON change. Do NOT auto-start -- wait for GO.

**>>> SUPERSEDED (kept for history): "NO ACTIVE SPRINT (2026-06-22 hand-off)" @ `08010ec` -- the STORY+CAST
FIX and the voice-casting workstream were started after it.** Carried verify item still open: the live
`test_audio_byte_identical` baseline (indextts2) may need recapture (the R2/story levers intentionally change
generated dialogue; the clean fixture is a no-op). The S3 forward order is UNCHANGED + PARKED behind the
voice-casting work: 3D (item 5, `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`), distribution (item 6), the
carried per-segment LUFS/RMS plan, and the Wan-14B remove.

**>>> STORY-QUALITY R2 -- COMPLETE (2026-06-22). All 8 craft levers + the Final-QA scan shipped + pushed
to v2.0-alpha, each suite+Bug-Bible green; suite 4934 pass/34 skip.** S1 music-text suppression (`5396c05`)
-> S2 announcer-close = concrete final image, not thesis (`118088f`) -> S3 cliche + flat stage-business
reroll gate (`1887803`) -> C0 action-verb beat intents + wants_are_default classifier (`687f766`) -> C3
contrasting speech_signatures at CastLock (`981db60`) -> C4+C5 per-act escalation + on-the-nose reroll
(`3e19906`) -> C1+C2 specificity anchors + central story-object, 3-pass roundtable-converged (`38c2c10`) ->
Final QA: `scripts/story_quality_scan.py` extended with the lever metrics (thesis/cliche/stage-business/
on-the-nose/leading-direction counts + wants_default + anchors/central-object presence + voice distinctness).
HARD held throughout: ledger schema `l3-2026-05-14` FIXED (new values ride free-form `meta`), content-only,
NO workflow-JSON change, model-agnostic (every gate is one a strong/opus line passes -> lifts the weak end),
reuse the existing reroll loop. The whole craft lift is in `_otr_line_hygiene` / `_otr_line_composer` /
`_otr_outline` / `_otr_casting` / `_otr_dramatic_state` / `_otr_specificity` + the writer wiring -- no node/
JSON change. Campaign docs in `docs/2026-06-22-story-quality-r2/` + `docs/2026-06-22-story-quality-c1c2/`.
OPERATOR VERIFY-AT-BUILD: the live `test_audio_byte_identical` baseline (indextts2) may need recapture since
the prompt levers intentionally change generated dialogue -- the clean fixture is a no-op, but a fixed-seed
baseline close that newly trips a gate would change; run the live gate + re-soak read at your convenience.

**>>> STAGE-DIRECTION LEAK SPRINT -- COMPLETE (2026-06-22). All 5 chunks shipped + pushed to v2.0-alpha;
3-pass roundtable-converged (architecture/coding/wiring, ~$0.51).** Bare undelimited stage directions
("twirls his pen nervously Look, Pinky...") were leaking into the frozen ledger text -> Bark spoke them +
SDH captions showed them (every existing scrub is delimited-only). Fix = a fallback ladder, all
content-only (ledger schema frozen, audio byte-identical, no workflow-JSON, model-agnostic): Chunk 1
`8c40182` pure `scrub_leading_stage_direction` + `detect_leading_stage_business` in `_otr_line_hygiene`
(narrow conjunction-of-guards; +62-case corpus) -> Chunk 2 `e2dd95a` `scripts/stage_direction_scan.py` +
PRECISION GATE (489 ledgers; 20 would_mutate, ALL true positives, 0 false positives) -> Chunk 4 `2278bd2`
the FREEZE floor in `_otr_ledger_scrub._strip_stage_directions` (the deterministic guarantee; preserves the
Tuple[str,bool] contract, emits CODE_STAGE_DIRECTION, restamps word_count) -> Chunk 5 `9142b2f` composer
prompt hardening + the S1 music patch (`OTR_LedgerScriptWriter` 3681: `(beat.sfx_cue or beat.intent or "")`
-> `(beat.sfx_cue or "")`, which had been silently undoing S1) -> Chunk 3 `6ce724d` the PRIMARY reroll wiring
in `_otr_line_composer.compose_line` (detect on the raw draft -> one reroll via the existing recursive-repair
pattern, hint concatenated, guard-capped, freeze floor as backstop). FINAL QA: the fixed freeze strips the
exact screenshot lines (b004/b006/b007); a fresh max-chaos render shipped a CLEAN ledger (0 leaks); suite
4861 pass/34 skip, audio byte-identical 9/0, Bug Bible 16/7/3. Campaign + judgments in
`docs/2026-06-22-stage-direction-leak/`. **>>> RESUMES: STORY-QUALITY R2 -- S1 SHIPPED (`5396c05`); NEXT =
S2 (announcer-close = concrete final image, not thesis).**

**>>> BARK ARTIFACT SPRINT -- COMPLETE (2026-06-22). All chunks shipped + pushed to v2.0-alpha; cleaned-bark
re-soak GPU-verified end-to-end + published to obs.** B0 (fixture+spine verify, `ffe5e82`) -> B1 (speech-only
dialogue mode + first-line [clears throat] gate, `467fb05`) -> B2 (thread the dropped per-line seed ->
determinism, `9e68ad5`) -> B3 (chunk-split hardening, `e1b8196`) -> QA (deterministic >4kHz high-band edge
metric + scan tool, `e93b54e`). Upstream-TTS only; audio spine FROZEN (byte-identical baseline is indextts2,
not bark -- verified); min_eos_p kept at 0.1; no per-chunk trim / no FFT reroll loop (panel cut); NO
workflow-JSON change. Suite 4782 pass/34 skip, Bug Bible 16/7/3. **Re-soak (operator config: cleaned bark +
mesh_stage 3D bookends + z_image_turbo char stills, real otr_scifi_16gb_full.json):** rendered "Spinning
Contamination" end-to-end in 21:13 -> `otr/obs/...spinning_contamination..._final.mp4` (20.3 MB). Metric
before/after on real bark lines: BEFORE (pre-B1) 1/14 char lines flagged (max 0.694); AFTER (cleaned) 0/6
flagged, all edges <= 0.055. Details: `docs/2026-06-22-bark-artifact/{B0_FIXTURE,QA_RESULTS}.md`. Box reset
clean. **>>> RESUMES: STORY-QUALITY R2 -- S1 SHIPPED (`5396c05`); NEXT = S2 (announcer-close = concrete final
image, not thesis).** (Bark sprint was a sanctioned interleave; R2 S2-C5 remain.)

**>>> STORY-QUALITY R2 -- craft lift. S1 SHIPPED (2026-06-22); NEXT = S2 (announcer-close =
concrete final image, not thesis).** Latest green: full suite 4749 pass/33 skip, Bug Bible 16/7/3, audio
byte-identical 9/0. **S1 DONE** (commit pending push to v2.0-alpha): music/non-dialogue beats no longer
bleed their placeholder intent into the spoken/caption transcript. Root cause was
`production_ledger.init_lines_from_outline` stamping a non-voiced row's `text` from `sfx_cue OR intent` ->
music_inter (no cue) leaked "Musical interlude bridging <phase>..." into the writer's `script_text`
(`assemble_script_text_from_ledger` -> `[SFX: ...]`). Fix: (a) new canonical `is_spoken_role()` in
`_otr_ledger_scrub`; (b) non-voiced rows take ONLY a genuine `sfx_cue` as text (intent fallback dropped --
music rows -> `text=""`; real sfx cues preserved; `beat_intent` still carries the intent for visual/music
consumers); (c) neutralized the `_otr_outline._assemble_outline` music_inter intent. SDH captions already
filtered non-speech roles, so this closes the transcript path. +9 tests (`tests/test_s1_music_suppression.py`
+ updated `test_phase2b_progressive_ledger.py`). Content-only; ZERO workflow-JSON change; ledger schema
`l3-2026-05-14` untouched. **>>> NEXT = S2.** (Below: the original full-sprint brief, S2-C5 unchanged.)

HEAD baseline at sprint start `90ddfca` == origin/v2.0-alpha; suite 4740 pass/33 skip, Bug Bible 16/7/3. The plan is ROUNDTABLE-HARDENED (3 passes) + Claude-creative-pass + seams
LOCATED and BUILD-READY: **`docs/2026-06-22-story-quality-r2/SPRINT_PLAN.md`** (+ `pass01_judgment.md` /
`pass01b_creative_levers.md` / `pass02_coding_judgment.md`; raw panel reviews in `roundtable/`). Goal = a
genuinely BETTER STORY (operator: NOT word count). Grounded findings: opus is genuinely good; the weak/local
end is generic + cliche + meandering stage-business + the music/non-dialogue beats BLEED their placeholder
into the spoken/caption track ("Musical interlude bridging..." in every episode). **Build = 8 ordered green
chunks, each suite+Bug-Bible -> commit AND push:** S1 suppress music/non-dialogue beats from the
spoken/caption track (`_otr_ledger_scrub._SPOKEN_ROLES` + the text-materialization point; neutralize the
`_otr_outline._assemble_outline` music_inter intent); S2 announcer-close = concrete FINAL IMAGE not thesis
(close intent + a shared banned-thesis regex scan -> reroll via the announcer composer); S3 cliche +
stage-business reject gate in `_otr_line_hygiene` (FLAGS ONLY) feeding the EXISTING Sprint-5C `reroll_hint`
loop in `_otr_line_composer.compose_line_draft`; C0 non-default-wants classifier in `_otr_dramatic_state` +
ACTION-VERB-under-pressure in the OUTLINE Stage-3 beat prompt (`_build_beat_user_prompt`, NOT the line
composer -- intents are written there); C1 specificity anchors -> `meta["specificity_anchors"]` + a
generic-line gate; C2 central story-object -> `meta["central_object"]` (derive BEFORE the S2 close); C3
CONTRASTING `speech_signature`s at CastLock (`_otr_casting`) + promote the already-wired signature clause to
a HARD per-line constraint; C4+C5 per-act escalation contract + a turn-beat subtext nudge. HARD (panel,
code-verified): ledger schema `l3-2026-05-14` FIXED -> new fields ride FREE-FORM `meta`, NEVER new Pydantic
fields; NO workflow-JSON / node / widget change; reuse the EXISTING reroll_hint loop + speech_signature (no
new reroll infra); MODEL-AGNOSTIC (every gate is one opus PASSES -> lifts the weak end, never rewrites opus);
craft-ONLY (no word/beat/budget change); audio byte-identical; UTF-8 no BOM; SFW. FINAL QA = extend
`scripts/story_quality_scan.py` (4 structural counts + craft signals) + a small re-soak (2-3 weak-local + 1
frontier leg) with the opus-no-regress gate. START WITH S1 (kills the universal music-bleed wart).

**>>> (DONE 2026-06-22) 3D IMAGE STREAMS -- all 7 chunks shipped + GPU-verified; mesh-quality v1.1/v1.2
(STUDIO-lighting gradient, not white marble) shipped + CPU-verified.** Spec was
`docs/2026-06-21-3d-image-streams/roundtable/pass01_plan.md`; results in the LATEST SESSION blocks above.
DEFERRED to 3D v1.5: Cycles + 3-point lighting + multi-view texture bake.

**>>> SHIPPED THIS SESSION (2026-06-21, all pushed, suite+Bug-Bible green per chunk):**
- **STORY-ENGINE v1 (F1-F8) + Sprint-0 harness** -- the whole story-quality sprint (`ecd0cde`..`d9b25a0`);
  see the block below. DONE.
- **3D `mesh_stage` promoted to all 3 slots** (`6b4fd97`) -- added to `VALIDATED_ENGINES`, selectable
  (announcer/music/cast), verified live; a 30w all-slots-3D episode rendered end-to-end (6 beats mesh_stage,
  2.76 GB peak). [This is what exposed the scene-still-fodder bug above.]
- **Credits: per-slot IMAGE model** (`8a7f06b`) -- dispatcher stamps `meta.image_engines.by_role`; the credits
  dossier RENDER ENGINES section now shows video AND image engine per slot. (Treatment-card `FLUX ???` is a
  separate surface, not yet wired -- optional follow-up.)
- **latentsync REMOVED from the forward order** (`156d573`) -- ripped out, no engine/JSON refs; punch-list +
  Wan 2.2 marked operator-APPROVED.

**>>> (HISTORY) STORY-ENGINE IMPROVEMENTS v1 -- ALL CODE SHIPPED + PUSHED (HEAD `d9b25a0` ==
origin/v2.0-alpha; full suite 4717 pass/33 skip, Bug Bible 16/7/3). NEXT = the GPU MEASUREMENT + SOAK, then
the 3D PoC.** Every feature F1-F8 + the Sprint-0 harness landed as its own green chunk, each suite+Bug-Bible
green, committed AND pushed (HEAD==origin verified per chunk; no 0-byte, no BOM, AST-parse):
- **Sprint 0** (`ecd0cde`): `scripts/story_quality_scan.py` (length_ratio / length_pass_fired / episode_valid /
  outro_hedge_vs_resolved / narration_self_address; reads the on-disk `*_ledger.json`) + the SHARED
  `is_resolved_ending_change` + `HEDGE_LIST` in `_otr_dramatic_state` (one source of truth with the F3 repair).
- **T1.1 F1** (`bdeccb6`): dropped the literal "about 20-30 words" hard-cap from the line rider (the 0.70
  length_ratio root cause); None/zero-safe per-line token budget.
- **T1.4 F6** (`bf5d71a`): split the rider -- indirect-performance UNCONDITIONAL on character beats,
  situation-change GATED to turn beats.
- **T1.2 F2** (`44aecd3`): costly choice bound to a CHARACTER beat (character-only candidate list +
  contract-build guard so `must_turn` never lands on announcer/music -- fixes the slot-drama audit).
- **T1.3 F3** (`f703ea4`): ending-aware outro -- threads `dramatic_state.ending_change` + the final character
  line; recompose ONCE on a resolved-then-hedged close; deterministic resolved fallback (no hedge).
- **T2.1 F4** (`79c0040`): pins speaker gender/pronouns in the line prompt from `cast[].gender` (no schema change).
- **T2.2 F5** (`feb6fa8`): `speech_signature` -- the description LLM emits a <=5-word register, backfilled to
  "plain spoken", rendered in the voice card.
- **T2.3 F7** (`958f331`): narration/self-address detector in `_otr_line_hygiene` (shared with the scan) +
  spine recompose seam (1 attempt, LOUD, fallback) + output-format constraint.
- **T3.1 F8** (`d9b25a0`): seeded `meta.arc_shape` variety (additive) + shape templates + shape-branched
  post-validator so non-confrontation shapes (investigation / slow_dread) no longer stall.
All content-only inside node 1 `OTR_LedgerScriptWriter` + its internal modules; ZERO `otr_scifi_16gb_full.json`
edits; ledger schema `l3-2026-05-14` untouched; node imports verified clean in the venv. **REMAINING (GPU):**
(1) the 12-leg/864-word measurement (baseline + after via `story_quality_scan.py`, write `SPRINT_BASELINE.md`);
(2) the operator-requested 500-word soak (indextts2 voice, LTX-audio bookends, flux2_klein char-beat stills,
max creativity, cheap OpenRouter frontier writer). KNOWN SMALL GAP: F8's macro-prompt arc_shape context was
deferred (the dramatic-state path carries arc_shape + the meta stamp -- acceptance "distribution not
single-valued / no rejections" is met). The 3D PoC (`docs/2026-06-20-mesh-stage-texturing/`) is AFTER the soak.

**>>> SUPERSEDED 2026-06-20 (DONE) -- TOP-PRIORITY VISUAL FIXES (all four shipped + GPU-verified):**
Spec: `docs/2026-06-20-visual-fixes/VISUAL_FIXES_PLAN.md` + the 2026-06-20-NIGHT top block. HEAD `ce507e8` ==
origin/v2.0-alpha; GPU FREE (35-leg 864 frontier soak done; 3D-PoC mesh_stage GPU smoke PASSED `60a2f4f`).
DIRECTIVE (hard): ONLY HuMo (portrait) + maybe the 3D lane use the VERTICAL portrait; EVERY other path uses
LANDSCAPE 16:9 images/video. Animated start/end procgen title cards + rolling credits are NON-NEGOTIABLE
(engine-independent).
- **#1 BUG 1 -- DONE 2026-06-20 (commits `7e765b7`+`9f03abd`, GPU-verified end-to-end with z_image_turbo).**
  Character beats now show the CHARACTER full-frame 16:9 (beat-aware, per-beat, distinct), not the radio booth.
  Root cause was deeper than the 8bc5381 branch: `"character"` (the canonical writer speaker_role) was missing
  from `SPEAKER_TO_VIDEO_ROLE`, so character beats were pooled as background_abstract; added the mapping +
  scene_character stills (LLM beat-aware) + render_driver gating on render_aspect. Final-obs frames show each
  character beat's still synced to its SDH caption; audio byte-identical. See the LATEST SESSION block + the
  proofs in `docs/2026-06-20-visual-fixes/`. (HuMo still uses the vertical portrait by design.)
- **#2 BUG 2 -- DONE 2026-06-20 (verify-first, NO change needed).** Eyeballed a real obs final: the animated
  START title card resolves the episode title via a signal-decode glitch (scramble@0.3s -> "GLIMPSE THROUGH
  GLASS"@4s) and the END rolling credits render the whole post-roll. Working as designed -> no code/JSON change.
  Proofs `docs/2026-06-20-visual-fixes/B2_*.png`.
- **#3 BUG 3 -- DONE 2026-06-20 (commit `9f76937`).** The forensic model/engine/system dossier now SCROLLS on the
  rolling credits (not just the `_treatment.txt` sidecar): new `_build_hud_dossier(led)` -> 5 sections
  (WRITER/LLM CONFIG, RESOLVED OPENROUTER, STORY SPINE, RENDER ENGINES, SYSTEM = CPU/RAM/GPU/VRAM/CUDA/torch),
  drawn at the top of the HUD scroll ahead of the transcript. Content-only, NO JSON change; +7 tests; verified by
  an offline pure-PIL HUD frame (`docs/2026-06-20-visual-fixes/BUG3_credits_dossier.png`).
- **#4 BUG 4 -- DONE 2026-06-20 (commit `13017ec`).** The ALWAYS-ON audio-reactive bottom bars are now a
  SEPARATE overlay layer (DEFAULT ON; manual `off`), decoupled from the scene-aware floor so they show no
  matter what clip is above/below (landscape AND portrait). New `audio_bars` widget on
  OTR_PostUpscaleProcgenBlend (appended LAST, BUG-LOCAL-097) wired IN `otr_scifi_16gb_full.json` SAME-commit
  (node 93 +1 value, validated). Implemented as a SEPARATE second pass (the fragile procgen/scopes blend
  untouched): main blend defers captions -> a bars pass PIL-renders the green `freq_bars_green` strip above
  the caption safe-area, lighten-blends at 0.60, then burns captions ON TOP. `off` = byte-identical;
  audio `-c:a copy` throughout. +6 tests (incl. a real-ffmpeg green-paint proof); verified on a real obs
  final (green bars react in the bottom strip, captions above).
- **THEN:** the 3D PoC build (`docs/2026-06-20-mesh-stage-texturing/roundtable/pass03_plan.md` + `pass04_plan.md`
  + the 4-item PIN list; chunks 6+7+3 + GPU smoke).

**>>> SUPERSEDED 2026-06-20 (historical; the story-quality roundtable is PARKED, section 8) -- 2026-06-19 STORY-QUALITY side-campaign:**
the STORY-QUALITY roundtable campaign. **R1 DONE 2026-06-19:** the grounded FIRST-PANELIST pass
(`roundtable/pass01_opus-grounded.md`) + the 5 external panels (`docs/story_passes/pass_1/`) + the converged
judge synthesis (`roundtable/pass01_architecture.md`) are written. The converged plan = 3 fix-moves
(news-derived opposed wants at `derive_dramatic_state_from_meta`/:2780 -> deliver via the EXISTING
contract+composer path -> critic-as-repair-driver, never a gate) + bounded shape (secondary) + a §5b
rip-out/repurpose of QA-that-only-scores-or-fails; bar = a genuinely BETTER STORY, FIX-not-FAIL,
model-agnostic, activate-not-overhaul, 'acts' are not a limiter. **R2 CONVERGED via a live 2-pass roundtable
(roundtable/R2/pass02_plan.md = build-ready Phase-1 plan, A1-A6 + B1); NEXT = R3 wiring/build (operator-gated)**
-> R4 Comfy -> R5 polish (campaign DONE only at polish). Materials:
`docs/2026-06-19-story-quality-analysis/` -- START with
`roundtable/PANEL_PROMPT_R1.md`; do NOT open `_PRIVATE_OPUS_PLAN_DO_NOT_PASTE_TO_PANEL.md` until AFTER the
grounded panelist pass. Carried into every round: the story must be genuinely ABOUT the news + it must work on
a SMALL LOCAL LLM + the ledger `{cast,lines,meta}` stays FIXED (content-only) + audio byte-identical + the bar
is a genuinely BETTER STORY (no rewrite-for-its-own-sake). The news-SELECTION front-end (RSS -> brief) is a
GIVEN INPUT, out of scope -- the work is everything AFTER the brief. The BUILD current-step below is UNCHANGED
and resumes after the campaign.

**>>> CURRENT STEP DONE (2026-06-18): flux2_klein VERIFIED + PROMOTED; coverage-arch accepts_still SHIPPED.**
- The TE mismatch is FIXED: klein-4B uses the **Qwen-3-4B** encoder (7680-wide; `qwen_3_4b.safetensors` from
  `Comfy-Org/flux2-klein`), NOT flux2-dev's Mistral (15360). The official ComfyUI `image_flux2_klein_text_to_image`
  template was the ground truth (CLIPLoader = qwen_3_4b, type flux2). A full flux2->silent-LTX 30w episode minted
  ALL 6 stills green (`minted still 832x480/1472x832`), no dim error; flux2_klein is now in image
  `VALIDATED_ENGINES` (opt-in via OTR_ENABLE_FLUX2_KLEIN). See `docs/2026-06-18-flux2-klein/VERIFY_ON_5080.md` +
  the LATEST SESSION block above.
- The coverage architecture is LIVE (the operator's "all video/3D accept the selected image, one place, no
  per-model whitelist"): `accepts_still` on MotionEngineBase (default True) read centrally by
  `engine_consumes_still`; silent `ltx_video` now consumes the selected image. Design + roundtable:
  `docs/2026-06-18-coverage-arch-wiring/pass01_plan.md`.
- **>>> NEXT = coverage-arch FOLLOW-UPS (deferred, additive):** (a) **IMAGE-PHASE still coverage -- DONE for
  per-beat (`3d27bcf`) + POOL_N_LOOP DONE (`838730e`):** `derive_scene_still_targets` emits a scene_beat target
  for EVERY beat (was announcer/music-only -> b002/b003/b004 hit LTX-I2V MISSING-STILL), AND under `pool_n_loop`
  the OTHER-BEATS {background_abstract, scene_broll} now SHARE N pool stills (`other_pool_0..N-1`; ShotLock stamps
  `still_pool_key=other_pool_{i mod N}`; render_driver prefers it). `pool_n_loop`+4 is now the
  otr_scifi_16gb_full.json + widget default (in sync). announcer/music/character_video stay per-beat; HuMo keeps
  scene stills (OOM-fallback insurance). Roundtable-converged plan +
  judgment: `docs/2026-06-18-pool-loop-stills/`. Suite 4502/0. (b) `optional_inputs` so role_compat sees an OPTIONAL init_image
  (verify-at-build); (c) optionally `accepts_still=True` on the static-still cheap families
  (flux_still/station_card/still_kenburns); (d) full Decision-3/5 (central `image_engines.registry.usable()`,
  retire `requires_mesh_portrait` onto `still_kind`). THEN the carried step below (per-segment LUFS/RMS) + the S3
  forward order (3D item 5, distribution item 6).

**>>> CARRIED STEP = plan per-segment LUFS/RMS voice normalization (operator-requested; FROZEN-SPINE change, needs a plan + re-baseline).**
- **`wan_ti2v` PROMOTED 2026-06-18 (`ca3e06c`):** forced-lane GPU smoke PASSED on the 5080 via the real adapter
  (render_single -> render_clip, executor thread) -- i2v rendered 33 frames from a 16:9 still at 832x480, engine
  vram_used ~8.2 GB, independent NVML peak 13,078 MiB (< the 14.5 GB cap), twice. Added to
  `registry.VALIDATED_ENGINES` (now lists in the tested-only dropdown). It's the low-VRAM tier filler for non-face
  beats -- the weakest video engine on quality but the one that fits where 14B HuMo / 22B LTX won't; NO audio.
  Same commit fixed `render_single` to render in each engine's native aspect (was always portrait -> wide engines
  letterboxed a 16:9 still = the "postage stamp"). The recorded sub-8GB tier = IMAGE `z_image_turbo` (`dcf078c`) +
  VIDEO `wan_ti2v` + characters on HuMo-2D; 3D `triposr` dark/deferred.
- **NEXT = per-segment LUFS/RMS normalization (operator instinct 2026-06-18).** The "low/thin" Bark perception is
  loudness, not peak; the assembler currently peak-normalizes each segment to 0.85 + a final -1.0 dBFS master
  limiter (no Bark-side normalize -- correct). A per-segment LUFS/RMS target would even out perceived loudness.
  **CAUTIONS (write the plan around these):** this is the FROZEN audio spine (`scene_sequencer.py`
  EpisodeAssembler `_normalize_clip`), NOT upstream TTS -> it breaks `test_audio_byte_identical` (deliberate golden
  re-baseline, operator-gated GPU render) and hits EVERY voice, not just Bark. Design = LUFS/RMS target WITH a
  max-gain clamp + a noise-floor gate (do NOT fully flatten dynamics -> pumping). Deterministic (no RNG). Recommend
  a roundtable before coding (frozen-spine + dynamics trade-off).
- **After that:** the S3 forward order -- 3D sprints (item 5) then switchable distribution (item 6). Operator-gated
  GATE-A items (punch-list audit [APPROVED 2026-06-21], coverage sweep GREEN) run in parallel on look-QA.
  (latentsync demos REMOVED 2026-06-21 -- ripped out, no live engine.)
- **Open operator eyeballs (carried):** the Bark "warmer / not-whiny" verdict (audition WAVs in
  `%TEMP%\bark_audition\`); the wan_ti2v wide smoke clip (shared this session) + the Wan 14B-vs-5B WEBM eyeball.

---

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (operator, hard):** `workflows/otr_scifi_16gb_full.json` IS the
  production workflow. (1) ANY node / wiring / widget change MUST be made IN that file in the SAME
  change as the code -- code that is not wired into this JSON is DORMANT and does nothing (the §4D
  miss, 2026-06-13: node + blend input shipped + tested but unwired -> ran dead in production). After
  editing, re-validate via `OTR_WorkflowValidator` + a JSON round-trip + the link/widget audit.
  (2) EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot (the sandbox mount lags file writes; always
  read/write the Windows path + verify via Desktop Commander).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8) -- not story-spine, not
  story-pipeline, not the broader audio stack, not other ROADMAP items.
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream character-voice
  "whiny" fix.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100% local/offline; determinism
  seed-keyed (per-seed within a render, NOT run-to-run); every in-render fallback LOUD; UTF-8 no BOM;
  SFW; V-12 dep isolation; no new widgets in the static workflow shell (V-11).
- GIT (operator 2026-06-10): ONE branch `v2.0-alpha`; commit AND push together per green chunk; the
  operator eyeball gates TAGS/promotions only; after a push verify HEAD==origin / no 0-byte / no BOM /
  AST parse on touched .py. prod/`main` is GATED until operator work is done (a `v2.0-alpha-stable`
  tag on `v2.0-alpha` is fine).
- EVERY session updates this doc + the `otr-build-tracker` dashboard (content; keep the gauge+lanes
  styling).
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs must log
  `cast RNG seed=... (OS entropy)`. Do NOT set `OTR_C7` for normal runs.

---

## 3. FORWARD ORDER (do in sequence)

> **Two tracks, parallel.** Item 1 (punch-list audit) is OPERATOR-GATED (look-QA -- section 5); the
> ENGINE track (items 3-4, Wan + sweep GREEN) proceeds NOW. "In sequence" applies WITHIN a track, not
> across the operator gate.

1. **Punch list (GATE A) -- OPERATOR APPROVED 2026-06-21, proceed.** Captions DONE (node 86
   `OTR_CaptionBurn` in `otr_scifi_16gb_full.json`, profile resolves `burn_captions=True`). REMAINING:
   node-level audit of LTX radio-open + procgen rolling credits -- baked into the headless path but maybe
   NOT into the saved JSON; prove a render FROM the JSON has them, then operator look-QA.
2. **latentsync -- REMOVED 2026-06-21 (operator: "we ripped it out").** Verified: NO engine file under
   `nodes/_otr_video_engines/`, 0 references in `otr_scifi_16gb_full.json`, only a few stray comment/env
   strings remain (`OTR_LSYNC_BASE_ENGINE`). Not a live lane -- dropped from the forward order. (A trivial
   code-comment scrub of the stray strings can ride any future cleanup; not a roadmap item.)
3. **Wan 2.2 video engine (section 4) -- OPERATOR APPROVED 2026-06-21 ("100% approved"): proceed with the
   eyeball + acceptance.** BOTH engines BUILT + validated live (2026-06-14, `bcbe05a`):
   wan_i2v (14B, post mixin-refactor) + the new wan_ti2v (5B/GGUF 8GB tier). REMAINING = the operator
   WEBM EYEBALL (14B vs 5B) + the optional formal full-episode `--acceptance` GREEN exit (slow
   wan-music-bed leg, run attended) + the M9 CS-3 instrumented proof. Code-complete; gates are the
   operator's.
4. **Coverage sweep GREEN (GATE A acceptance).** Re-run the permutation matrix after the soak fixes.
   Matrix (additive, not cross-product): a visual-engine leg-set (varies each of music/announcer/
   other_beats), a writer-LLM leg-set (varies node-1 `creative_writing_model`/`technical_model`), and a
   curated voice-variation leg-set (2-3 refs per voice engine). Unique story per leg (OS entropy, no
   seed pins). **Wan is a CORE/BLOCKING engine** -- the sweep is NOT green until `wan_i2v` (and
   `wan_ti2v`) pass, so it stays RED until item 3 lands; that is expected. This re-run also answers the
   one open R2 question: whether `humo_1.7B` renders NATIVE char beats at 70w once its enable flag is on
   (the soak floored it only because the flag was off). **GATE-A precondition: harden the
   sweep FIRST (section 4A M1-M4) -- DONE 2026-06-13: the M1-M5 acceptance gate landed
   (`scripts/otr_coverage_sweep.py --acceptance`), so a silent fallback / empty-results
   run / missing VRAM measurement now scores RED, not GREEN.**
   **S6 harness reality:** `otr_coverage_sweep.py` enumerates ONLY the visual-engine
   leg-set today (the dropdown rotation). The writer-LLM leg-set (node-1
   `creative_writing_model`/`technical_model`) and the curated voice-variation leg-set
   are NOT yet wired into a runnable harness -- TODO: point them at a real driver
   (e.g. a `run_combo_matrix.py`) or run them as separate parametrized soak legs.
   "Coverage sweep GREEN" today means the visual-engine set only.

   **SOAK READINESS AUDIT (2026-06-13).** Walked the registry + harness. Conclusion:
   **clear to run a wan_i2v-only soak today** (no wan_ti2v hard prereq for validation).
   Verified live: `wan_i2v` enumerates `ok`/runnable under `16gb_full` (legs
   `music_visual=wan_i2v` + `other_beats_visual=wan_i2v`) -- the old "add wan_i2v to the
   enable-set" note is STALE/resolved. 27 legs enumerate; the only skips are
   `hunyuan3d_talk`/`trellis_talk` (missing cu128 toolchain, expected darks). Wan models
   on disk + `OTR_ENABLE_WAN_I2V=1` env known. **Two limitations to know:**
   (i) `--acceptance` exit is RED-by-construction until `wan_ti2v` is built (M2 requires
   BOTH Wan engines) -- expected; read the per-leg verdicts in `coverage_sweep_summary.json`,
   the wan_i2v leg PASS/FAIL is the meaningful signal.
   (ii) **The M1 no-fallback (CS-1) gate is bound to `--acceptance`** (`forbid_fallback=
   args.acceptance`); the capstone CLI does not expose it. So re-running the NON-Wan
   permutation soak (the set that originally false-greened) WITH the M1 fix active and a
   clean GREEN/RED exit needs either `wan_ti2v` built OR a small **`--strict-fallback`**
   flag that decouples M1 from the Wan-engine requirement (~10 lines; RECOMMENDED, optional
   -- operator's call). Until then: `--acceptance --only wan` exercises M1 on the wan_i2v
   legs (overall RED expected), and a non-acceptance sweep runs but with M1 OFF
   (informational). No half-built code, no missing capability rows beyond the deferred
   `wan_ti2v`, no broken tests (the 2 `test_model_catalog_scan` reds are pre-existing /
   environmental, tracked separately).
5. **3D sprints.** s2 = S-3D-0 spike + T1 template + T2a wrap smoke; then the `character_3d` family
   (image-routing must-fixes already landed). Detail in the 3D plan (pointers).
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing phase).

**0-E parallel track:** `ltx_orbit`/`still_parallax`/`mesh_stage` CPU side shipped + all three GPU-green;
Phase B (E-1 probe, E-6 renders, per-engine sweep legs) HELD on the `scripts/_otr_0e_gpu_go.txt` GO file.

**Audio parallel track (own window, never blocks video):** the character-voice "whiny" fix (upstream TTS
only; frozen spine untouched). Operator note: may have self-resolved -- verify before scheduling work.

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 video engines, eyeball-gated, b-roll/camera motion only (lip-sync stays SEPARATE
on LatentSync/HuMo). Core Comfy Wan nodes, NOT the KJ wrapper (KJ drags in SageAttention + a numpy<2 pin
this box violates). Phase 1 + the 5 code-gap fixes are DONE (`2fbc2f3`); the full grounded spec is in
that commit + git history of this file.

- **Phase 2 -- 16GB engine leg.** Drive `eng_wan_i2v.render_clip` via the real path
  (`scripts/otr_run_leg.ps1` / `coverage_sweep --only ...`). ASSERT `wan_i2v` is the final_engine in the
  trace (FAIL LOUD on fallback, CS-1) + render-phase NVML <= 14.5 GB + byte-identical audio mux + silent
  mp4 (h264/yuv420p/bt709, fps 25, `has_audio` False). Kill/reset the Phase-1 server first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF (Q6/Q5_K_M) + the wan2.2 VAE into
  `C:\ComfyUI-Models\` (record HF repo + sha256 + license, fail-closed). Define a NEW `wan_ti2v` engine
  (own flag/model/VAE env, registry registration, `_node_candidates` incl. the 5B latent node, loader
  mode, `canonicalize`, profile hook + tests) -- do NOT alias `WanI2VEngine`.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B, same still + prompt) in
  `docs/2026-06-12-ltx23-motion/wan_clips/`. Bar = real camera motion, still preserved, no warp.
  **S3 motion risk to watch:** the wired I2V-14B fp8 is a SINGLE low-noise expert (the
  two-expert HIGH/LOW MoE handoff, Path B, is NOT wired -- see `eng_wan_i2v` header). If
  the "real camera motion" bar FAILS (motion too subtle / static), the Path B two-expert
  HIGH/LOW handoff is the mitigation, not a knob tweak. Call this out at the eyeball.
- **Risk CS-3 (reframed):** sequential-residency, NOT co-residency -- see section 4A M9
  and the section-5 CS-3 entry. The supervised Wan batch proves the inter-beat reclaim,
  it does not "decide if they co-stage."

---

## 4A. WAN + GATE-A SWEEP HARDENING (roundtable 2026-06-13, grounded vs HEAD 134f8e2)

Folded from a 3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4) + Claude's
grounding against the real code. Full judgment + raw reviews:
`docs/2026-06-13-goforward-wan-hardening/`. These gate item 3 (Wan) and item 4 (sweep
GREEN). MUST-FIX -- until M1-M4 land, a GREEN sweep is meaningless:

> **STATUS 2026-06-13 (autonomous build) -- LANDING LEDGER:**
> - **M1 + M4** `9b2294b` -- no-runtime-fallback gate + VRAM fail-closed (12 tests).
> - **M2 + M3 + M5** `0ab55bc` -- sweep `--acceptance`: empty/required-engine exit
>   code + Wan enable-flag / OTR_TEST_MODE / --exclude preflight (17 tests).
> - **M6** `ec91a3c` -- `assert_usable` preflights UNET + umt5 CLIP + VAE (8 tests).
> - **M7** `f71edaa` -- render_clip ffprobe-PROVES the silent-clip contract (13 tests).
> - **S1 + S5** `dfe9ab5` -- wan_i2v vram_estimate 14500 + real wan2.2-i2v asset id.
> - **S7 + S10** `f3a529f` -- per-shot/seed init staging + Pillow-required fail-loud.
> - **S3 / S6 / S8** -- folded into this doc (MoE eyeball risk, sweep-harness reality,
>   the exact acceptance invocation below).
>
> **M8 + S2 -- LANDED 2026-06-14 (`bcbe05a`).** The `wan_ti2v` engine is built: its 5B core
> node class (`Wan22ImageToVideoLatent`) was captured from a live `/object_info` first; M8 raises
> `EngineUnusable` when the resolved VAE basename is empty or is the 2.1 VAE; S2 added the
> `medium`/8000 CAPABILITIES row (registry-consistency invariant holds -- the row + the registered
> engine landed together). Validated live (5B bare-graph smoke PASS). **STILL OPEN:** **M9** (CS-3
> sequential residency) + **S4** (leg isolation/reclaim) + **S9** (post-reset verify) are live-GPU
> proof obligations -- partial evidence only. A full multi-leg `--acceptance` GREEN exit is gated on
> the slow wan-music-bed leg (run it attended/selectively), not on missing code.
>
> **S8 -- exact acceptance invocation** (ComfyUI venv python; live server on :8000;
> `OTR_TEST_MODE` UNSET; `OTR_ENABLE_WAN_I2V=1` (+ `OTR_ENABLE_WAN_TI2V=1` once built);
> Wan UNET + umt5 CLIP + VAE on disk):
> `python scripts\otr_coverage_sweep.py --acceptance --only wan`
> (`--only wan` matches the `sweep_<slot>_wan_i2v` / `_wan_ti2v` legs; drop `--only`
> for the full visual set. `--exclude` of a core Wan engine is REFUSED in acceptance.)

- **M1 -- the sweep is BLIND to silent fallback.** `otr_coverage_sweep.py` runs every
  leg with `expect_engine=""`, which `_otr_soak_capstone.py:464` treats as
  informational (no assert), so a leg that silently falls back to `still_kenburns`
  scores PASS (this is exactly CS-1). FIX (NOT per-leg `expect_engine=engine` -- that
  false-fails a slot that gets 0 beats at 30w): in acceptance mode assert ZERO runtime
  fallbacks across the whole trace -- fail any shot where `final_engine != attempts[0]`
  -- with an opt-out only for known-degrade experiment legs. (Verify the trace field is
  a stable requested-id, not an alias.)
- **M2 -- the sweep returns GREEN on EMPTY results.** `return 0 if passed ==
  len(results)` makes `0 == 0` pass when `--only`/`--exclude` filter everything out or
  `wan_ti2v` is unregistered. FIX: fail on empty results; for GATE-A, fail unless BOTH
  `wan_i2v` AND `wan_ti2v` are present in results with PASS.
- **M3 -- acceptance preflight (closes the R2 trap).** `availability()` is pure
  profile-fit and never reads `OTR_ENABLE_WAN_I2V`, so a gated-off Wan leg enumerates
  "run", `assert_usable` fails it closed, it falls back, and (pre-M1) passes -- the same
  `gated_by_flag` mechanism that floored HuMo-1.7B (commit 5231d31). FIX: the acceptance
  run preflights `OTR_ENABLE_WAN_I2V=1` (+ future `OTR_ENABLE_WAN_TI2V=1`) and the model
  files, and FORBIDS `--exclude` of the core Wan engines.
- **M4 -- the V-3 VRAM gate fails OPEN.** `driver_peak = int(report.get("vram_peak_mb")
  or -1)` then fails only if `> ceiling`, so a missing/0/negative measurement (`-1`)
  PASSES -- the `<=14.5GB` invariant can read GREEN with no measurement. FIX: fail
  closed when `vram_peak_mb` is absent or `<= 0`.
- **M5 -- the Wan render-phase VRAM assert is skipped under `OTR_TEST_MODE`** (`if not
  os.environ.get("OTR_TEST_MODE"): ... assert_peak_within_ceiling`). Phase-2 acceptance
  MUST run with `OTR_TEST_MODE` UNSET; the harness preflight fails if it is set.
- **M6 -- `assert_usable` preflights only the ckpt.** The umt5 CLIP + the VAE are
  required graph loaders. FIX: verify UNET+CLIP+VAE present + matching the sha/license
  manifest before any forward (offline / no-runtime-fetch invariant).
- **M7 -- the Phase-2 clip contract is SELF-DECLARED, not asserted.** `_clip_from_raw`
  hardcodes `has_audio=False`/h264/yuv420p/bt709/fps25 in a dict; the soak only inspects
  the obs final's audio. FIX: ffprobe the emitted silent Wan mp4 (or a real-path test)
  to PROVE those fields before mux.
- **M8 -- `wan_ti2v` VAE fail-closed.** `eng_wan_i2v` defaults the VAE to
  `wan_2.1_vae.safetensors`; the 5B needs the Wan2.2 VAE. Give `wan_ti2v` its own VAE
  env; raise `EngineUnusable` if the resolved VAE basename is empty OR equals the 2.1
  basename. Do NOT inherit `_loader_names()` unchanged.
- **M9 -- CS-3 = sequential residency (see section 5).** Prove per-beat peak <= 14.5GB +
  the inter-beat reclaim drains the prior heavy engine (incl. the retained Wan unet
  patcher) before the next loads; that is the real risk, not co-residency. Unblocks
  Phase-2 scoping.

SHOULD-FIX: **S1** raise `CAPABILITIES["wan_i2v"].vram_estimate_mb` 14000 -> the measured
Phase-2 peak (or 14500); the 14499 smoke figure was WITHOUT `free_after_use`, which is
load-bearing -- document it as mandatory. **S2** add a concrete `wan_ti2v` CAPABILITIES
row (`medium` / ~8000 DRAFT -- the 5B VAE decode may push higher, verify on the 8GB
probe / `["wan2.2-ti2v-5b"]`). **S3** surface the single-expert (low-noise) MoE motion
risk on the eyeball gate -- Path B two-expert HIGH/LOW handoff is the mitigation if the
"real camera motion" bar fails. **S4** sweep leg isolation -- reclaim/restart between
legs that swap heavy engines (one resident server, no teardown -> residue corrupts the
next leg's peak; ties to CS-2 + the CLAUDE.md reset directive). **S5** fix the stale
`["wan2.1-i2v"]` label -> the real Wan2.2 I2V asset id. **S6** point item-4's writer-LLM
+ voice-variation leg-sets at their real harness (run_combo_matrix.py?) or mark TODO --
`otr_coverage_sweep.py` enumerates ONLY the visual-engine set today. **S7** stage the
init image under a shot/seed/uuid name (`otr_wan_init_WxH.png` is fixed -> same-dim
renders overwrite; low risk, driver is sequential). **S8** spell `scripts/otr_coverage_sweep.py`
+ the exact `--only` Wan substring + required env. **S9** Phase-2 post-reset verify
(PID/start-time changed, Sage NOT active, `OTR_TEST_MODE` unset, env visible) before
submitting. **S10** `_materialize_init_image`: require Pillow + fail loud (the no-Pillow
path leans on `WanImageToVideo` cover-resize -- N9 risk).

CUTS (panel consensus -- do NOT over-engineer): no broad VRAM-budget-aware scheduler to
close CS-3 (the reclaim assertion suffices; wait for a measured failure); do NOT subclass
all of `WanI2VEngine` for `wan_ti2v` (share only pure dims/aspect/materialize/canonicalize
helpers; keep loaders + node candidates + graph SEPARATE); keep the GATE-A sweep ADDITIVE,
not a visual x writer x voice cross-product. VERIFY-AT-BUILD: capture TI2V-5B's exact core
node class from `/object_info` before coding (the "5B latent node" is underspecified).

---

## 4B. WAN PHASE 1 -- DONE (pointer)

Phase 1 PROVEN: a real Wan b-roll clip (wan_i2v 14B fp8 in-process, ~14.5 GB; commits `2fbc2f3` +
`8eaf058`). Phase 2 is the ACTIVE next step (section 1); remaining Wan work = sections 4 + 4A. The
overnight-soak companion findings (R1 GPU-proven, R2 harness fix unexercised, R3 landed) live in git +
`scripts/FABLE_SOAK_REVIEW.md`; the not-done remainder (R2 verify) is in section 5.

---

## 5. OPEN TICKETS

- **SCHEMA-ADHERENCE (2026-06-25 -- LEVER-1 LOAD-BEARING SHIPPED; see the CURRENT STEP block at the top):**
  LEVER 1 tolerance (`pass04_plan.md` C0-C6, refined by the nested-fork + c4-scope roundtables) SHIPPED in 2
  green chunks `516644eb` (C0+C1+C2+C5+C6: `apply_field_aliases`/`__otr_field_aliases__` before-validator +
  `validate_tolerant_data` core; proven nested Opus `normalize_length` failure fixed) + `d4ca6cd4` (C3:
  JSON-syntax-only structural rung). C4 (schema-in-repair) DEFERRED -- proven failure already fixed, would test
  dead code; OPTIONAL `_build_schema_snippet`-shim recipe ready in c4-scope/, reopen on a real captured drift.
  LEVER 2 binary lane `docs/2026-06-25-schema-adherence/binary/pass01_plan.md` still GATED on **G1** (offline
  abstain-residual count -- the cheap first move; may DROP the lane) + **G2** (byte-identity of abstain).
  **G1 DONE -> Lever 2 (binary lane) DROPPED (genuine residual ~0; `binary/G1_RESULTS.md`); SCHEMA-ADHERENCE
  SPRINT COMPLETE.** NO workflow-JSON change.
- **LOOK-QA BUGS (NEW 2026-06-14 eve — operator look-QA pass; all in `BUG_LOG_2026-06.md`):**
  - **BUG-408 default MUSIC sounds non-musical (SA3).** **IMPLEMENTED 2026-06-14 (`3a4f71d`).** Path B:
    SA3-shaped prompt + real negative + per-cue `seconds_start` within a 30s `seconds_total` context (latent
    stays `dur` → length+determinism unchanged), env-overridable sampler knobs. Suite 4261/0. **OPERATOR-GATED:**
    restart Desktop, A/B listen (tune `OTR_SA3_CFG/STEPS/CONTEXT_S`), then RE-BASELINE the `test_audio_byte_identical`
    golden (intended music-bytes change). Plan: `docs/2026-06-14-sa3-music-improvement/roundtable/pass01_plan.md`.
  - **BUG-409 title card scrambles the WHOLE window** — **FIXED 2026-06-14 (`9e0b658`).** New
    `_title_reveal_progress` resolves the reveal in the first ~40% of the window then holds solid (env
    `OTR_TITLE_REVEAL_FRACTION`); close card stays bounded to the main video (no credits overlap). Suite 4259/0.
  - **BUG-410 closing ROLLING CREDITS** — **CLOSED 2026-06-14 (operator-verified on flux_still).** Credits
    scroll over the held last clip to the end again (silent after the theme). Detail in `BUG_LOG_2026-06.md`
    + `docs/2026-06-14-credits-tail-fix/`. (HuMo backdrop not yet eyeballed — low risk, engine-agnostic path.)
  - **BUG-411 flux BOOKEND / image lost its "lush" cinematic tint (NEXT — HANDOFF FOCUS).** The 6/5 image
    pipeline (`visual/batch_flux_render.py` + `flux_prompt_extractor`) was WHOLLY REWRITTEN into
    `_otr_image_engines/flux_gen1.py` + `otr_meta_brief_image_prompt.py` (pure insertions after `e4cb3ac`).
    Model/steps/cfg/sampler IDENTICAL (flux1-dev-fp8, 20, 1.0, euler/simple), but the rewrite DROPPED the look
    levers: **(1) FluxGuidance = 3.5** (flux_gen1 has NO FluxGuidance node — biggest factor), **(2) the
    cinematic style suffix** `"cinematic, 35mm film, anamorphic lens, volumetric lighting, heavy vignette,
    muted color grade, sharp focus"`, **(3) the radio broadcast-distress suffix** + retrofuturistic radio
    fallback (`35mm film grain ... dim amber and cyan rim lighting`), **(4) bookend seed 4242**, **(5) portrait
    style line**. 6/5 workflow widgets inspected + confirmed (no other hidden hardcodes). FIX = restore those in
    the new pipeline (FluxGuidance node @ ~3.5 + the suffixes + seed). Full forensic in `BUG_LOG_2026-06.md`
    BUG-411. CODER-READY (the next window's task).
  These are GATE-A look-QA items (operator-gated track), parallel to the engine forward order — NOT a
  reordering of section 3.

- **IMPROVED 3D INPUT -- BLOCKED on a PATH DECISION (operator look-QA 2026-06-14; GROUNDED this session).**
  The 3D rotating output looked like a "blobby plaster-of-paris" block. GROUNDING (checked logs + disk):
  the ONLY 3D system actually installed/active is **HunyuanWorld-Mirror / WorldMirror 2.0**
  (`custom_nodes/ComfyUI-HunyuanWorld-Mirror`, model `C:\ComfyUI-Models\WorldMirror-V2\HY-WorldMirror-2.0`)
  -- NOT Blender, NOT OTR's deferred character_3d/TripoSG. Recent episode ledgers used NO 3D engine; the
  server log only shows HWM loading (no episode rendered 3D). **WorldMirror is a MULTI-VIEW SCENE
  reconstructor** (image SEQUENCE -> point cloud / Gaussian splat): per its docs 1 frame = "depth/normals
  only"; good 3D needs **8-24 FEATURE-RICH frames, orbital/forward parallax, well-lit, 50-70% overlap**. A
  single flat/low-feature image -> the plaster blob. So the earlier "clean / object-free single image"
  idea is the OPPOSITE of what WorldMirror needs -- object-free helps only single-image-to-OBJECT-mesh
  tools (TripoSG / Hunyuan3D-2 / TRELLIS), which are NOT installed. **OPEN DECISION (operator, next window)
  -- the prompt strategy is opposite per path:** (A) WorldMirror scene/world -> improved input = GENERATE
  an orbit/multi-view sequence + rich-textured scene prompts (NOT a plain bg); or (B) single-image ->
  object mesh -> INSTALL TripoSG/Hunyuan3D-2 + clean isolated-subject prompts. Do NOT draft the improved
  3D prompts (and do NOT wire character_3d) until the path is picked. A roundtable can harden the chosen
  path's prompt set. (Note: the live roundtable launcher stalled this session -- the panel blocked with no
  output; budget a retry or a smaller panel.) Example obs finals to eyeball the EPISODE look (these do NOT
  contain 3D): `output\otr\obs\signal_lost_plunging_depths_20260614_185229_silent_procgen_blended_final.mp4`
  (a pre-fix render -- shows the closing FREEZE + the skinny flux_still portrait, both now fixed at HEAD).
- **HuMo full-frame TEST (operator 2026-06-14 -- future experiment, NOT now).** Operator wants to
  eventually SEE HuMo rendered full-frame (not the 480x832 portrait pillarbox). For now portrait stays
  HuMo's REQUIREMENT -- BUG-407 shipped "full frame everything EXCEPT HuMo". Future: a HuMo full-frame /
  16:9 smoke to evaluate whether the talking-head holds at a wider aspect before changing the default.
- **Look-QA the 5 overnight 120-word episodes (NEW 2026-06-14).** The default-lane soak ran 5/5 SUCCESS
  (LTX + humo_1.7B); the episode outputs (`...\output\otr\episodes` + obs finals) are NOT yet eyeballed.
  Check audio sync, burned captions, procgen scopes/credits, character look. This is the operator's
  "analyze the soak" item; verdicts in `scripts/_otr_120word_soak_summary.json`.
- **Wan WEBM EYEBALL -- DONE 2026-06-14 (operator + Claude live smoke).** RESULT: **Wan i2v 14B
  DRIFTS** -- holds the input still ~1 frame then re-interprets the scene into its own subject (a
  generic tube close-up). NOT fixable by easy input knobs: cfg3.5->2.0 + a locked-tripod prompt STILL
  drifts; cfg1.5 COLLAPSES into incoherent abstraction. **LTX (2B v0.9) HOLDS** the composition with
  subtle motion in all 3 modes tested (ksampler 30-step, distilled, AND 1216x704 hires -- hires
  answers the "low-res" note). => **RECOMMEND: Wan i2v 14B -> BACK-BURNER for the music/announcer
  OPENER role** (keep selectable; revisit only with Path B two-expert handoff, GO_FORWARD 4A S3); LTX
  stays the opener engine; **PROMOTE LTX-REGR (below) to the active thread.** Evidence:
  `docs/2026-06-14-wan-ti2v/EYEBALL_FINDINGS.md` + `eyeball_frames/COMPARISON_montage.png`. AWAITS
  operator confirm on the re-prioritization.

- **Non-Wan soak = ENOUGH (operator call 2026-06-13).** The non-Wan permutation coverage sweep
  (`--strict-fallback --exclude wan/latentsync/triposg`) has run sufficiently; do NOT keep grinding it.
  The non-lip-sync FLOORS (`still_kenburns` / `still_parallax` / Ken-Burns / `station_card`) render fine
  and are acceptable for the 8GB tier, but they are NOT the target experience -- the operator wants real
  audio-driven lip-sync, not a still with motion. Focus the remaining runway on **getting the Wan lane
  bug-free** (section 1 + 4 + 4A). A new sweep, if ever needed, should add `--exclude-engine humo` (the
  exact-match flag added `ca10b63`: skips the 14B `humo` that TIMES OUT per CS-4, KEEPS `humo_1.7B`).
- **LTX-REGR — SUPERSEDED 2026-06-15 by the LTX 22B-GGUF splice** (`docs/2026-06-15-ltx-splice/SPLICE_PLAN.md`).
  LTX-REGR's recommended fix was to bake the **2B** v0_9 recipe into `eng_ltx_video.py`; the splice instead swaps
  `LtxVideoEngine` to the **22B GGUF** mini recipe (verified-working). **Do NOT do the 2B bake.** Original entry kept
  below for history only:
  **LTX-REGR (operator 2026-06-13; PROMOTED to active 2026-06-14 pending operator confirm)** -- LTX
  clips no longer animate like the **2026-05-30..06-05** era (motion lost / too static). `BUG-LOCAL-113b`
  (`8115c72`: ksampler 30-step euler cfg3.0 as the LTX default, distilled 8-step = the
  `OTR_LTX_SAMPLER=distilled` rollback) was the prior fix, but the operator STILL sees the regression.
  **2026-06-14 eyeball update:** the Wan-vs-LTX smoke proved LTX HOLDS the still composition cleanly
  (good) -- so the open question is narrowed to **MOTION AMOUNT** (5/30-6/5 read as more dynamic; the
  current ksampler/distilled holds are subtle). With Wan i2v back-burnered for openers, this is the
  recommended NEXT thread. Probe = an LTX **--strength / sampler-mode / step-count / cfg / frame-cap**
  sweep (otr_ltx_motion_smoke.py exposes all of them; --strength is the prime motion lever, 1.0=max
  freedom) against the 5/30 baseline + the 169 decode floor from look-QA round 5.
  **FORENSIC DONE 2026-06-14 (BUG-LOCAL-412, `BUG_LOG_2026-06.md`):** diffed the GOOD 5/09 `l001` + 5/28
  `b001` LTX bookends vs the current engine (ledgers + the DELETED `batch_ltx_render.py` @ `70d379b^` + the
  old workflow JSON widgets). The good recipe = **v0_9 / sampler `euler_cfg_pp` / 8 distilled steps / cfg
  1.0 / 832×480 / I2V strength 0.75 / `loop_via_reverse` boomerang / audio-length**; the cleanbreak
  `70d379b` DELETED that node and `eng_ltx_video.py` shipped **ksampler / `euler` / 30-step / cfg 3.0 /
  768×512-or-1472×832 / strength 1.0 / 169-cap / no boomerang** (the code comment itself admits `euler_cfg_pp`
  is the documented dynamic-motion sampler but the default was left on `euler`). The old WORKFLOW JSON baked
  in NOTHING but seed/method/cap — the recipe lived in code. **ENV-TESTABLE A/B FIRST (no code change):** at
  832×480 set `OTR_LTX_SAMPLER=distilled` + `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`,
  re-render a bookend, A/B vs `l001`/`b001`; if it matches, bake those defaults + the boomerang + audio-length
  back into `eng_ltx_video.py` (coder chunk; no JSON change implicated).
- **CS-1** -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was fallback-only);
  re-verify in the sweep. (Non-Wan -> deprioritized per the operator's "non-Wan soak = enough" call.)
- **CS-2** -- machine NVML pins ~16 GB per leg vs the 14.5 ceiling while driver-phase attribution reads
  ~3 GB; needs phase attribution (the 1.7B leg's 10,305 MB render-phase peak is a partial answer).
- **CS-3 (reframed 2026-06-13)** -- NOT a co-residency budget: wan_i2v (~14GB) +
  humo_1.7B (~7GB) cannot co-reside under 14.5GB by construction, so they must render
  SEQUENTIALLY. The real proof obligation = per-beat NVML peak <= 14.5GB AND the
  inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`, BUG-291) fully drains the
  prior heavy engine -- incl. the retained Wan unet patcher -- before the next beat
  loads. A mixed Wan+HuMo episode is the test. This UNBLOCKS Phase-2 scoping (no
  open "decision" needed). See section 4A M9.
- **CS-4-open** (deprioritized) -- targeted post-encode umt5-TE detach for the OPT-IN 14B HuMo lane so it
  fits 14.5 GB. The default char tier is `humo_1.7B` (`955f134`); the 14B is opt-in.
- **R2 verify** -- confirm `humo_1.7B` renders native char beats at 70w with its enable flag ON (the
  soak floored it only via `gated_by_flag`); answered by the item-4 re-run.
- **README "what to expect per video model" (operator 2026-06-14).** Once the opener model bake-off
  settles (interactive render bench artifact `otr-render-bench` + `docs/2026-06-14-wan-ti2v/
  EYEBALL_FINDINGS.md`), add a user-facing "what to expect from each video engine" section to the
  README (newbie audience -- folds into S6/closing): Wan i2v 14B = drifts off the still (b-roll only,
  NOT openers); LTX = holds composition + subtle motion (opener default); TI2V-5B = 8GB tier, lower-res.
  Source the verdicts from the operator's bench ratings (export button).
- **Ship defaults (release)** -- proposed: announcer + character = `flux_still`, music = `visualizer`
  (selectable: station_card, still_parallax, abstract — `ltx_orbit` ripped 2026-06-15 in the LTX splice Phase 0). Keep HuMo/latentsync/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish** (minor) -- output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR`
  (fail LOUD on mismatch); run the OH-3 janitor sweep at server boot; widen the heartbeat cadence.
- **OH-4** -- the 14-entry / ~8.2 GB live->attic migration STAGED, awaits operator "go OH-4"
  (`docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md`).
- **0-E Phase B** -- tickets E-1..E-7, gated on the sweep GO file; coder-window ready.
- **Operator gates** -- ComfyUI Desktop relaunch (look-QA), fresh-render acceptance, whiny-voice P0 matrix
  + reel, S-3D-0 green light, `v2.0-alpha-stable` tag decision. (latentsync demos REMOVED 2026-06-21.)

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
path gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9:
S-3D-0 spike -> T2b keystone GO/NO-GO (timeboxed ~1wk) -> T4 driver + LOOK gate -> W7 production wiring +
soak ("v1-usable") -> S3-S6 distribution. SHORTCUT FORK: S-3D-0 or keystone NO-GO -> `character_3d`
defers (HuMo-2D stays) -> collapses to ~2-3 sprints (0-E + closing). Done splits: "v1-usable" (one
engine, one real episode) vs "B-parity ship" (>=2 engines bind at SHIP).

---

## 7. POINTERS (evidence + tooling -- not plans)

- Tracker dashboard: `otr-build-tracker` artifact (OneDrive\Documents\Claude\Artifacts).
- Soak review (R1/R2/R3 detail + roundtable): `scripts/FABLE_SOAK_REVIEW.md`.
- Wan/sweep hardening (grounded QA + 3-model roundtable judgment, 2026-06-13):
  `docs/2026-06-13-goforward-wan-hardening/` (pass00 plan+QA, pass01/pass01b raw
  reviews, pass01_judgment.md).
- Overnight sweep: `scripts/otr_overnight_sweep_launch.ps1`; tasks `otr-overnight-sweep` +
  `otr-sweep-monitor`; digest `scripts/sweep_monitor_digest.md`; GO file `scripts/_otr_0e_gpu_go.txt`.
- 3D spec (forward item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Bug log (this repo): ACTIVE = `BUG_LOG_2026-06.md` (epoch BUG-LOCAL-400+, started 2026-06-14);
  ARCHIVE = `BUG_LOG.md` (BUG-LOCAL-001..~305, through 2026-06-12, reference only).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale;
switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes until S-3D-0 + the operator green light.
(**LTX-AV audio-input lane MOVED OUT of PARKED 2026-06-17** -- operator revived it as the CURRENT STEP
(section 1): M1+M3 are shipped, the remaining work is recipe-align + M4 GPU smoke. It already uses the good
Q3_K_M GGUF unet; do NOT rebuild from scratch.)

(**STORY-ENGINE quality roundtable (2026-06-21) -- PARKED side campaign.** A 4-pass live roundtable converged a
sprint-ready plan for 8 content-only story-engine fixes (length tail / costly-choice binding / ending-aware outro
/ gender-pronouns / speech register / narration hygiene / arc-shape variety; F9 reorder + F10 anti-repeat list
deferred). Docs: `docs/2026-06-21-allnight-864-frontier/` -- `SPRINT_READY_PLAN.md` + `STORY_ENGINE_KICKOFF.md` +
`roundtable/pass0{1,2,3,4}_judgment.md`. All content-only inside the FIXED ledger, ZERO workflow-JSON edits
(verified vs the real consumers). NOT active -- the visual fixes (section 1) + the forward order win. Resume only
on an explicit operator green light.)
