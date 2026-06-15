<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The seam strategy is sound, but the plan lacks a telemetry channel to record billed costs in the ledger and misses a critical dependency check for the API nodes.

MUST-FIX BEFORE BUILD:
1. [Section 5 - S1] **Missing billing telemetry channel.** The plan requires writing API costs to `ledger["billing"]`, but `dispatch_images` (in `otr_image_gen_dispatcher.py`) expects `gen_fn` to return only a pixel array or a `.png` path. The adapter has no way to pass the actual billed cost back to the dispatcher.
   *Fix:* Modify `dispatch_images` and `_inprocess_gen_fn` to accept a tuple return type `(result, meta_dict)` from `gen_fn`. Extract `meta_dict.get("billed_cents")` in the dispatcher and append it to the ledger, passing `result` to `_coerce_pixels`.

2. [Section 5 - S1] **Missing package probe in `assert_usable`.** The plan surfaces the `OTR_COMFY_API_KEY` probe via `assert_usable`. However, if the user provides the key but is running a ComfyUI environment without `comfy_api_nodes` installed, `assert_usable` will pass and the engine will hard-crash during `render_image`'s lazy import.
   *Fix:* The `assert_usable` implementation for cloud engines must perform a lazy `import comfy_api_nodes` check, raising `EngineUnusable` if the package is missing.

SHOULD-FIX:
1. [Section 5 - S3] **LTX-2 Descriptor strictness.** To ensure `cloud_ltx2` successfully routes to `background_abstract` (which `role_compat.py` restricts to *only* `{"text_prompt"}`), its `required_inputs` descriptor must be strictly `("text_prompt",)`. If `init_image` is included in its required list, `engine_fits_role` will fail-closed and exclude it from background beats.

OPTIONAL / NICE-TO-HAVE:
- [Section 3] **Spike Async Verification:** When verifying `util.client` in the executor thread, explicitly test it inside a standard `threading.Thread` (no event loop) to perfectly mirror ComfyUI's `PromptExecutor` environment.

CUT THESE (over-engineering):
1. [Section 2] **PyTorch tensor conversion for network images.** The plan specifies using `download_url_to_image_tensor` and converting it via `(t[0]*255).clamp(0,255).byte().cpu().numpy()`. This unnecessarily pulls PyTorch into the execution path for a simple HTTP download, violating the lightweight nature of network engines.
   *Why it is safe to cut:* `dispatch_images` and `_coerce_pixels` natively support returning a `.png` path or a raw numpy array. Use `download_url_to_bytesio` (or standard `requests`), save it to a temporary `.png` (returning the path), or open it with `PIL.Image` and return the numpy array. This avoids the `torch` dependency entirely.