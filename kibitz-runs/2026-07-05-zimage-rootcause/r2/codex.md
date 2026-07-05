VERDICT: yes-with-fixes. Root direction is viable, but the plan is not build-ready until resolver semantics, tests, and dispatcher scope are made explicit.

MUST-FIX BEFORE BUILD:
1. [Proposed fix A/C] Env override cannot just “basename wins.” Existing launch scripts set `OTR_ZIMAGE_UNET=z_image_turbo_nvfp4.safetensors` as a basename, while current adapter `assert_usable` only accepts `os.path.isfile(ckpt)` and would reject that if called directly. See `scripts/_otr_overnight_420_boot.cmd:35`, `scripts/_otr_run_smoke.cmd:12`, and `nodes/_otr_image_engines/z_image_turbo.py:231-240`. Concrete fix: `_resolve_unet_name()` must validate env values as either absolute files or model-folder names via `folder_paths.get_full_path("diffusion_models", basename)` / `get_filename_list("diffusion_models")`; bad env values must return `None` or raise `EngineUnusable`, not pass through to deep render failure.

2. [Proposed fix A/C] The “folder_paths unimportable -> return env/default name” rule contradicts “assert_usable raises MISSING_MODEL only when nothing installed.” If the helper returns `_DEFAULT_UNET` when it cannot inspect installed models, adapter `assert_usable` will pass in environments where no model was verified. Current tests install a `folder_paths` stub whose `get_filename_list()` returns `[]`, and current z-image test expects missing env to raise. See `tests/conftest.py:72-79` and `tests/test_image_engine_c2.py:66-75`. Concrete fix: return a structured result like `{name, verified, source}` or add a `require_installed` mode so `render_image` can keep a clear loader error while `assert_usable` only passes when env/default/autodiscovery is verified.

3. [Proposed fix B] The edit target is misidentified. Line 127 is inside `_zimage_params()`, not `render_image`; `render_image()` only calls `_zimage_params()` at `nodes/_otr_image_engines/z_image_turbo.py:264-269`. Concrete fix: state the code change as `_zimage_params()["unet_name"] = _resolve_unet_name(...) or _DEFAULT_UNET`, then keep `render_image()` unchanged except through params.

4. [Open question 3 / Root cause 3] Do not claim adapter-level `assert_usable` will grey/fail early unless the dispatcher is also changed. The dispatcher currently calls registry-level `_ireg.assert_usable(engine_id, role)`, and that registry intentionally does no disk IO; `_inprocess_gen_fn()` then calls `eng.render_image()` without `eng.assert_usable()`. See `nodes/otr_image_gen_dispatcher.py:584-597`, `nodes/otr_image_gen_dispatcher.py:773-780`, and `nodes/_otr_shared/engine_registry_base.py:193-226`. Concrete fix: either scope this patch to z-image render-time resolver only, or explicitly add adapter preflight in the dispatcher and update all affected image engines first.

5. [Open question 3] If dispatcher adapter preflight is included, it will likely break other in-stack engines unless their basename/model-folder resolution is fixed too. Lumina, Qwen, and Flux2 adapter `assert_usable` paths currently also use `os.path.isfile()` on env values. See `nodes/_otr_image_engines/lumina_image.py:184-200`, `nodes/_otr_image_engines/qwen_image.py:200-210`, and `nodes/_otr_image_engines/flux2_klein.py:218-230`. Concrete fix: keep dispatcher preflight as a separate follow-up, or normalize all image adapter model resolution in the same change.

SHOULD-FIX:
1. [Proposed fix A] Default-before-autodiscover can choose bf16 even when nvfp4 is installed. The plan says rank `nvfp4 > fp8 > bf16`, but only after `_DEFAULT_UNET` is absent, so a box with both `z_image_turbo_bf16.safetensors` and `z_image_turbo_nvfp4.safetensors` will pick bf16. See `_DEFAULT_UNET` at `nodes/_otr_image_engines/z_image_turbo.py:69` and current params at `nodes/_otr_image_engines/z_image_turbo.py:127`. Concrete fix: either rank all installed `z_image_turbo*.safetensors` candidates when env is unset, or explicitly document bf16-default precedence as intentional.

2. [Proposed fix A] Autodiscovery tie-breaking is underspecified. Multiple `z_image_turbo*.safetensors` files can match the same tier. Concrete fix: sort case-insensitively within each tier and log the full candidate list plus chosen basename.

3. [Proposed fix A/C] Tests need to change with the behavior. Current `test_z_image_adapter_assert_usable_fail_closed` asserts no env means MISSING_MODEL. Concrete fix: add tests for env absolute path, env basename visible in `folder_paths`, env basename missing, default visible, nvfp4 autodiscovery, and empty list -> MISSING_MODEL.

4. [Symptom / root cause] verify: live ComfyUI `folder_paths.get_filename_list("diffusion_models")` actually reports `z_image_turbo_nvfp4.safetensors` on this install. The repo proves tests stub the API, but not the live model listing.

OPTIONAL / NICE-TO-HAVE:
1. [Proposed fix A] Include the resolved model name in the z-image render log at `nodes/_otr_image_engines/z_image_turbo.py:277-281`; it will make future bake-off logs self-diagnosing.

CUT THESE (over-engineering):
1. [Open question 2] Do not add CLIP/VAE autodiscovery in this patch. They did not cause this failure, and `clip_name` / `vae_name` already have defaults at `nodes/_otr_image_engines/z_image_turbo.py:128-129`. Add only targeted preflight errors if live render proves they are missing.

2. [Open question 3] Do not fix dispatcher-wide adapter preflight in the same green chunk unless you also normalize every image engine’s `assert_usable`. The root z-image failure is closed by shared unet resolution in `_zimage_params()`; dispatcher preflight is a broader registry contract change.