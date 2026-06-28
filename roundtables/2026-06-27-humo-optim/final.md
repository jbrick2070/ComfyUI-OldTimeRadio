# HuMo STANDALONE bakeoff -- FINAL coder-ready spec (r1->r4 CONVERGED)

Local kibitz: Codex (gpt-5.5/high, read-only) + AntiGravity (gemini-3.5-pro, file-handoff) x r1->r4,
Claude grounded anchor + sole judge. r4 CONVERGED ("yes-with-fixes", same small set). Every accepted
claim verified vs the real files. DIAGNOSTIC ONLY -- edits NOTHING in production (eng_humo.py /
otr_scifi_16gb_full.json / OTR pack __init__.py). Answers: does the 5/21 14B fit <=13.5 GB safely
(promote later) or stay 1.7B? Operator eyeballs clips + metrics; all production changes DEFERRED.

## NEW FILES (only these; nothing in the OTR pack)
1. `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\otr_bakeoff_helper\__init__.py` -- a SIBLING custom-node
   package (sibling to ComfyUI-OldTimeRadio, since the repo itself lives under ...\ComfyUI\custom_nodes\).
   Registers ONE node `OTR_BakeoffReclaim`:
   - PASSTHROUGH on the LATENT edge: input LATENT (from WanHuMoImageToVideo output slot 2) -> output
     LATENT -> KSampler.latent_image. Executing it forces the eviction to happen AFTER conditioning/
     image-encode and BEFORE the sampler.
   - ENCODER-ONLY evict (NOT wrapper_bridge.reclaim_idle_models -- that detaches EVERY loaded model
     incl. the 14B unet, wrapper_bridge.py:248-301). Detach ONLY the umt5 CLIP + whisper audio-encoder
     model objects; KEEP the unet/LoRA/ModelSamplingSD3 + VAE resident. Log + assert the sampler model
     object SURVIVED before returning.
   - `IS_CHANGED` returns a unique value every call (e.g. time.time()/NaN) so the executor never
     cache-skips it (esp. in the same-session sentinel leg). Emit a unique manifest marker per prompt.
2. `scripts/build_humo_bakeoff_workflow.py` -- emits the per-leg STANDALONE graph JSON. Reuses
   `HuMoEngine._build_graph` READ-ONLY for node templates, then TRANSLATES the in-process run_graph spec
   (`"class"` aliases + `Wire(src,slot)` objects, eng_humo.py:215-283) into ComfyUI API `/prompt` JSON
   (`class_type` + list links `[src,slot]`; cf. render_humo_batch.py:568-665). Terminal = `SaveImage`.
   For the two-stage leg, splice `OTR_BakeoffReclaim` onto the latent edge. `--dry-validate` asserts
   every node has a valid `class_type` + inputs + a SaveImage terminal, and that OTR_BakeoffReclaim /
   WanHuMoImageToVideo classes are registered -- renders nothing.
3. `scripts/run_humo_bakeoff.py` -- boot-per-leg headless runner (clone run_ltx_av_q_bakeoff.py). Cold-
   import wrapper_bridge by file path (no `nodes` package side effects). External VRAM peak via pynvml/
   nvidia-smi (verify pynvml in the venv). Fail-loud asserting manifest. Side-by-side clips.

## LEGS (fixed still c02_466a19906ccb.png + audio c02_b002_line.wav + seed)
(i)  humo_14B_169 single-graph, 6-step distill, shift 8, cfg 1.0 = 5/21 BASELINE (today's path).
(ii) humo_14B_169 TWO-STAGE (OTR_BakeoffReclaim on the latent edge), same settings = CANDIDATE.
(iii)humo_1.7B_169 (16:9 -- NOT portrait humo_1.7B) = current-ship CONTROL.
(iv) SENTINEL, no reboot: run ONE real `ltx_audio_in` render in the SAME session (assert the LTX-AV
     classes/model names actually ran -- LTX-AV uses Gemma + LTX VAEs, NOT Whisper; eng_ltx_av.py:454-465),
     THEN run leg (ii) -> the production-true peak under cross-engine residency.
CUT: the no-LoRA ~25-step leg.

## METRICS + MANIFEST (fail-loud)
- VRAM: external render-window peak. HARD-GATE the heavy engine at 13500 MB; separately REPORT the
  14500 MB box ceiling (do NOT inherit the LTX 14500 default as the gate).
- SPEED: s/it + wall-clock per leg.
- QUALITY: side-by-side clips -> `otr/episodes/_bakeoff_humo/<leg>.mp4` (operator eyeball, primary) +
  BLUE-CAST DELTA in pure PIL+numpy (mean R/G/B of frames vs the source still; tests de-blue, no face
  libs). Face-detect / mouth-landmark / lip-SSIM SOFT-GATED: preflight cv2/dlib/mediapipe/ffprobe, mark
  "skipped" if absent (NOT in requirements) -- NEVER gating, NEVER failing the run.
- MANIFEST ASSERTS (not just records): resolved unet filename / LoRA present-absent / ModelSamplingSD3
  shift=8 / steps / cfg / seed / dims / 4n+1 frame length / terminal / output path + the engine id that
  ACTUALLY ran. STARTUP: assert the leg's checkpoint files EXIST on disk (HuMo-14B unet, lightx2v LoRA,
  umt5 CLIP, wan_2.1 VAE, whisper audio-enc) -- fail loud, never silently skip a leg into a false "DONE".

## OUTPUT (byte-match production)
Graph terminal `SaveImage` (lossless PNG batch) -> harness compiles via production
`wrapper_bridge.encode_frames_to_silent_mp4` (silent). ffprobe-verify each output: NO audio stream,
expected frame count, lands under `otr/episodes/_bakeoff_humo/`. Never tmp.

## BOOT / RESET (CLAUDE.md S4)
Boot the server WITHOUT `FLOOR` (FLOOR clears OTR_ENABLE_HUMO -> HuMoEngine.assert_usable fails closed);
manifest OTR_ENABLE_HUMO + OTR_HUMO_* + resolved class names. `reset_box` SELECTIVELY CIM-kills by
command line (excl. current PID): prior `run_humo_bakeoff.py` trees + the `cmd.exe` shell running
`_otr_soak_server_launch.cmd` + the Comfy server + port 8000 owners -- NEVER a blanket python kill.
Confirm port 8000 empty + nvidia-smi at desktop baseline before each boot.

## TEST (this build's own tests; production suite stays green since production is untouched)
- A CPU unit test for `OTR_BakeoffReclaim`: IS_CHANGED is always-dirty; the node is a latent passthrough;
  the encoder-only evict keeps a stand-in "unet" object and drops the "clip"/"audioenc" stand-ins.
- `build_humo_bakeoff_workflow.py --dry-validate`: every leg JSON valid class_type/inputs/SaveImage; the
  reclaim node spliced on the latent edge for leg (ii) only.
- Run the full OTR suite + Bug Bible to PROVE production is unchanged (no regressions), then the live
  bakeoff. Commit+push per green chunk to v2.0-alpha. prod/main + tags GATED.

## Judgment log
ACCEPTED (grounded): sibling otr_bakeoff_helper at the absolute custom_nodes path; OTR_BakeoffReclaim =
latent-passthrough + ENCODER-ONLY evict + assert-unet-survives + always-dirty IS_CHANGED; _build_graph
-> /prompt API translator; control humo_1.7B_169; sentinel = real ltx_audio_in render (LTX-AV != Whisper);
gate 13500 (report 14500); SaveImage -> production silent encoder + ffprobe verify; checkpoint-exists +
fail-loud asserting manifest; no-FLOOR boot; orphan/cmd.exe selective reset; blue-cast PIL+numpy; face
metrics soft-gated/non-gating; --dry-validate; pynvml check. CUT (both): no-LoRA 25-step; two-/prompt
conditioning round-trip. DEFERRED to a later operator-gated round: the production two-stage split +
VramPeakProbe in eng_humo + the 16gb_full profile 1.7B->14B flip (only IF the bakeoff shows <=13.5 GB).
