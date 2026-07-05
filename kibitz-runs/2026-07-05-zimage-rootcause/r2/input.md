# z_image_turbo render failure -- root cause + proposed fix

## Symptom (anime bake-off, 2026-07-05)
z_image_turbo leg FAILED at OTR_ImageGenDispatcher (node 91):
`ImageRenderError: c01: image render with 'z_image_turbo' failed
(GraphExecutionError: node 'unet' (load_unet) raised FileNotFoundError:
Model in folder 'diffusion_models' with filename
'z_image_turbo_bf16.safetensors' not found). NO FALLBACK.`
flux_gen1 SUCCEEDED on the same run. (qwen_image also failed -- genuinely missing
`qwen_2.5_vl_7b_fp8_scaled.safetensors`; out of scope here, real download.)

## Root cause (grounded against the real tree + disk)
1. Installed on this Blackwell (5080) box: `z_image_turbo_nvfp4.safetensors`
   (4.2 GB) in `C:\ComfyUI-Models\diffusion_models`. The bf16 file is NOT present.
2. `nodes/_otr_image_engines/z_image_turbo.py`:
   - L69 `_DEFAULT_UNET = "z_image_turbo_bf16.safetensors"`.
   - L127 render graph: `"unet_name": os.path.basename(os.environ.get(MODEL_ENV,
     "") or _DEFAULT_UNET)` -- env `OTR_ZIMAGE_UNET` OR the bf16 default.
   - My headless boot (direct `main.py`, and `_otr_soak_server_launch.cmd`) do
     NOT set `OTR_ZIMAGE_UNET`; several OTHER boot scripts DO set it to nvfp4
     (otr_ia2v_server_boot.cmd / _otr_overnight_420_boot.cmd / _otr_run_smoke.cmd
     / a few .ps1). So env-unset -> bf16 default -> not on disk -> deep failure.
3. LATENT CONTRACT BUG (the "more concerning" thread): engine-level
   `ZImageTurboEngine.assert_usable` (L224-251) RAISES `EngineUnusable
   (MISSING_MODEL)` when `OTR_ZIMAGE_UNET` is UNSET -- i.e. it is designed to
   GREY the engine out. But `render_image` FALLS BACK to the bf16 default. The
   two paths DISAGREE: if the usability gate were consulted, z_image would be
   unusable (env unset) and never selected; because the dispatcher does NOT call
   the engine's assert_usable before render, the render proceeds on a default
   that isn't installed and dies deep. Any in-stack engine with an env-gated
   `assert_usable` + a lenient `render_image` default has the same landmine.

## Proposed fix (root cause, no band-aid per CLAUDE.md)
A. Add a SHARED `_resolve_unet_name()` used by BOTH assert_usable and
   render_image so their contracts can never diverge again:
   - env `OTR_ZIMAGE_UNET` (basename) wins if set;
   - else `_DEFAULT_UNET` if present in `folder_paths.get_filename_list
     ("diffusion_models")`;
   - else AUTO-DISCOVER an installed `z_image_turbo*.safetensors` (rank
     nvfp4 > fp8 > bf16 > any) with a LOUD log;
   - else None.
   - Degrade cleanly when `folder_paths` is unimportable (test venv) -> return
     the env/default name (so unit tests keep their current shape).
B. `render_image` L127 -> the resolved name (fallback to `_DEFAULT_UNET` basename
   only so the loader still raises a CLEAR error if TRULY nothing is installed).
C. `assert_usable` raises MISSING_MODEL ONLY when `_resolve_unet_name()` finds
   nothing installed -- it no longer hard-requires the specific env var (the box
   HAS a usable z_image; requiring the env to be hand-set was the bug).

Result: z_image_turbo "just works" with whatever quant is downloaded (nvfp4 here),
env override still honored, and assert_usable + render_image share ONE truth.

## Open questions for the panel (kibitz codex + Sonnet-5 fan-out)
1. Is shared auto-discovery the right ROOT fix, or is adding
   `OTR_ZIMAGE_UNET` to `_otr_soak_server_launch.cmd` sufficient? (I read the
   latter as a band-aid -- the default is simply wrong on any nvfp4 box.)
2. Should the SAME discovery apply to CLIP/VAE? (Defaults qwen_3_4b / ae ARE on
   disk, so they did not fail -- lower priority, but symmetric.)
3. The DEEPER systemic bug: the image dispatcher does not enforce the engine's
   `assert_usable` (disk check) before render, so a misconfigured in-stack engine
   fails deep instead of greying early. Fix z_image now + file the dispatcher
   pre-flight as a separate item? Or fix both here?
4. Any test/behavior risk in changing assert_usable from "env required" to
   "installed model discovered" (test_image_engine_c2, test_image_dep_pilot pin
   z_image usability/registration)?
