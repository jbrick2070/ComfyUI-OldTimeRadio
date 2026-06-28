VERDICT: yes-with-fixes
The plan has converged but requires minor clarifications regarding caching, node wiring signature, model existence checks, and graph format translation to ensure the build does not block.

MUST-FIX BEFORE BUILD:
1. [r4 questions #1] [caching / execution order] ComfyUI caches node outputs. If the `OTR_BakeoffReclaim` node does not implement an `IS_CHANGED` classmethod, ComfyUI may cache-skip it on subsequent runs or during the sentinel leg (iv) where graphs are run in the same session. 
   FIX: Implement `IS_CHANGED` on `OTR_BakeoffReclaim` in `custom_nodes/otr_bakeoff_helper/__init__.py` to always return a dynamic value (e.g., `import time; return time.time()` or `float("nan")`) so that it is never skipped by the executor.
2. [The near-final plan (r3)] [wiring signature ambiguity] The plan states `OTR_BakeoffReclaim` is wired "BETWEEN conditioning and KSampler". KSampler takes two separate conditionings (`positive` and `negative`) plus a `latent_image`. Intercepting conditioning could mean multiple nodes or confusing wiring.
   FIX: Explicitly specify that the `OTR_BakeoffReclaim` node operates as a passthrough for either `LATENT` (intercepting the `latent_image` connection from `WanHuMoImageToVideo` output slot 2 to `KSampler`'s `latent_image` input slot) or a single `CONDITIONING` (intercepting the `positive` conditioning connection from `WanHuMoImageToVideo` output slot 0 to `KSampler`'s `positive` input slot).
3. [The near-final plan (r3)] [fail-loud assertions] The script `run_humo_bakeoff.py` must not silently skip legs if critical checkpoint weights are missing on disk. Silently skipping legs would result in a false-positive success report showing "DONE" but verifying nothing.
   FIX: Add an assertion at startup of `run_humo_bakeoff.py` that verifies the candidate (ii) and baseline (i) checkpoint files actually exist on disk (under the directories configured by `folder_paths` or env overrides), failing loud immediately if they are missing.
4. [The near-final plan (r3)] [graph JSON translation] `build_humo_bakeoff_workflow.py` reuses `HuMoEngine._build_graph` read-only. However, `_build_graph` returns an in-process dictionary format (using `Wire` objects) while the headless `/prompt` endpoint consumes the ComfyUI API prompt JSON format.
   FIX: Specify that `build_humo_bakeoff_workflow.py` must include a lightweight translator that converts the in-process graph returned by `_build_graph` into the standard ComfyUI API prompt JSON format (mapping `Wire(src, slot)` to `[src, slot]`).

SHOULD-FIX:
1. [BOOT/RESET] [process tree cleanup] The selective `reset_box` implementation must cleanly terminate any running instances of `run_humo_bakeoff.py` and its shell process trees (`cmd.exe` executing `_otr_soak_server_launch.cmd`) without matching or killing the current PID.
   FIX: In the PowerShell script inside `run_humo_bakeoff.py`, walk process trees to find parent `cmd.exe` processes executing `_otr_soak_server_launch.cmd` and kill them along with child `python.exe` processes.

OPTIONAL / NICE-TO-HAVE:
- Include a `--dry-validate` mode check in `run_humo_bakeoff.py` that verifies node schemas of both LTX-AV and HuMo node classes (`WanHuMoImageToVideo`, `UnetLoaderGGUF`, `OTR_BakeoffReclaim`) are correctly registered prior to launching any heavy GPU jobs.

CUT THESE:
- None. The plan is lean, has cut the no-LoRA 25-step leg, and is focused exclusively on the stand-alone bakeoff goals.

VERIFY-AT-BUILD checklist:
1. [ASSUMPTION] Verify the following model filenames are present on the GPU box or mapped via `extra_model_paths.yaml` (ref: `nodes/_otr_video_engines/eng_humo.py` lines 191-200):
   - `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` (UNET)
   - `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` (LoRA)
   - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (CLIP)
   - `wan_2.1_vae.safetensors` (VAE)
   - `whisper_large_v3_fp16.safetensors` (Audio Encoder)
2. [ASSUMPTION] Verify that `workflows/ltx_av_bakeoff_gguf.json` (as built by `scripts/build_ltx_av_bakeoff_workflow.py`) successfully converts to the ComfyUI API format using `_run_baseline._workflow_to_api_prompt` during leg (iv) sentinel execution.
3. Verify that `pynvml` is installed in the ComfyUI venv to allow external VRAM peak checking (`run_humo_bakeoff.py` line 167).
