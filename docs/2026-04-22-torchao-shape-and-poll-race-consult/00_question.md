# Round-robin question


TWO RELATED BUGS in OTR's visual sidecar (chained FLUX -> LTX -> Wan2.1
pipeline inside a single multiprocessing.spawn subprocess).

===== BUG 1: torchao FP8 FLUX shape mismatch =====

Every shot in flux_anchor raises immediately on FluxPipeline(prompt, ...)
call:
  RuntimeError: The size of tensor a (4096) must match the size of
  tensor b (10240) at non-singleton dimension 1

First shot 9.2s (CUDA JIT warmup), subsequent shots 0.1-0.3s (short-
circuit). 4096 = 64x64 image patches at 1024x1024. 10240 unclear -- not
a clean multiple of 4096.

Checkpoint: `diffusers/FLUX.1-dev-torchao-fp8` (pre-quantized, ~17 GB,
pickled .bin). Loaded via:
    FluxPipeline.from_pretrained(path, use_safetensors=False,
                                  local_files_only=True)
    pipe.enable_model_cpu_offload()

Torchao emits warnings DURING the load that hint at the cause:
    UserWarning: Stored version is not the same as current default
    version of the config: stored_version=1, current_default_version=2,
    please check the deprecation warning
    UserWarning: Models quantized with version 1 of
    Float8DynamicActivationFloat8WeightConfig is deprecated and will no
    longer be supported in a future release, please upgrade torchao and
    quantize again

See https://github.com/pytorch/ao/issues/2649.

An existing shim (`flux_anchor._install_torchao_compat_shim`) aliases
legacy torchao helper NAMES (`float8_dynamic_activation_float8_weight`,
`int4_weight_only`, `uintx_weight_only`, etc.) to current *Config
classes -- this was BUG-LOCAL-051/053/054/055. It fixes unpickling but
NOT the tensor LAYOUT that v2 expects.

ALTERNATIVE PATH THAT WORKS: ComfyUI-native `CheckpointLoaderSimple`
with Comfy-Org's `flux1-dev-fp8.safetensors` (17.25 GB, single file,
raw FP8 tensors, no pickle). Rendered a clean 1024x1024 20-step image
in 44.62s tonight, peak 11350 MB VRAM. Zero torchao involvement at
load time.

QUESTIONS FOR BUG 1:
1a. Is the 4096 vs 10240 shape mismatch caused by the torchao v1/v2
    config layout difference, or is something else going on?
1b. Can we pin torchao to a version still using v1 layout, OR set a
    flag to force v1 compatibility, OR do we MUST either re-quantize
    or swap checkpoint?
1c. If swap checkpoint: cheapest path to get FluxPipeline.from_pretrained
    working against Comfy-Org's single-file `flux1-dev-fp8.safetensors`
    (which is NOT a Diffusers folder layout)?
1d. Alternatively -- should we bypass FluxPipeline entirely and use
    safetensors.safe_open + manual FP8 cast + custom denoise loop in
    the sidecar subprocess? What's the minimal code path?

===== BUG 2: VisualPoll returns READY before sidecar actually done =====

Observed log sequence (after BUG 1 fires and flux_anchor errors 9/9):
  [flux_anchor] _render_real done: rendered=0 oom=0 errored=9   <- writes STATUS_READY
  [VisualRenderer] NO ASSETS FOUND x9  + procedural fallback     <- poll returned READY
  Prompt executed in 172.73 seconds                              <- workflow DECLARED DONE
  [sidecar:xxx] [flux_anchor] pipe released                      <- finally block runs AFTER
  [sidecar:xxx] [video_stack] barrier after ltx_motion           <- STILL RUNNING
  Wan2.1 pipeline loading...                                     <- STILL RUNNING
  [sidecar:xxx] <sidecar exited rc=0>                            <- finally done

Root cause candidate: flux_anchor writes STATUS_READY on its stage
completion even when rendered=0 errored=9. VisualPoll polls STATUS.json
every ~2s and unblocks on FIRST status=READY seen. So poll returns
before video_stack's later STATUS_READY (and before ltx_motion and
wan21_loop even start loading their pipelines).

Sidecar PID is written to `sidecar_pid.txt` in the job input dir.

QUESTIONS FOR BUG 2:
2a. Best fix: (i) flux_anchor never writes terminal status (always
    STATUS_RUNNING when it's one stage in a chain), (ii) poll checks
    sidecar_pid.txt is-alive AND status=READY before unblocking, or
    (iii) video_stack wraps stages so only ONE STATUS_READY is ever
    written at the end?
2b. If (ii) -- any Windows gotchas for cross-process PID liveness check?
    (torch.cuda IPC context issues, spawned-subprocess-vs-fork, etc.)

===== INPUT CONSTRAINTS =====

- Windows, not WSL. subprocess spawned via multiprocessing.get_context("spawn").
- 16 GB VRAM ceiling, 14.5 GB working target. No cloud, no paid services.
- Audio path CANNOT be perturbed. If fix threatens Bark/Kokoro/MusicGen -> reject.
- BF16 FLUX checkpoint deleted 2026-04-19 (53 GB reclaimed). Re-downloading
  would take 30+ min at Jeffrey's bandwidth.
- bitsandbytes 0.49.2 installed.

Answer in the four-section format for EACH bug (8 sections total, prefix
with "BUG 1 -" and "BUG 2 -").
