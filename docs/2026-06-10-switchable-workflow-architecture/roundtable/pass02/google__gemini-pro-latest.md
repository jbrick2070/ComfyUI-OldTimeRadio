<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan mandates offline schema patching and in-graph profile stamps, but contradicts the existing API patcher's schema format, triggers the validator's own hard-gate, and strands VRAM data away from the execution context.

MUST-FIX BEFORE BUILD:
1. [How a profile reaches the graph] The plan states `apply()` will run offline via `NODE_CLASS_MAPPINGS` import, but `otr_api.py`'s `patch_widget_by_name` strictly requires the `GET /object_info` JSON schema format (e.g., `{"input": {"required": {"widget": ["TYPE", {...}]}}}`). Python's `INPUT_TYPES()` returns a different shape (tuples instead of lists).
   Fix: Add a schema adapter in the shared `apply` module that converts `NODE_CLASS_MAPPINGS[cls].INPUT_TYPES()` into the exact `/object_info` dictionary shape before passing it to the patcher.

2. [Validator + node 63] The plan states the generator writes `profile_id` + `master_hash` into a "node-63 widget". However, `OTR_WorkflowValidator` only defines 3 widgets. Blindly appending to `widgets_values` will trigger the `widget_vector_drift` hard-gate (which raises and halts execution because `len(wv) != expected`).
   Fix: Explicitly add `profile_id`, `master_hash`, and `generated_by` as `("STRING", {"default": ""})` optional inputs to `OTR_WorkflowValidator.INPUT_TYPES()` so the expected slot count matches the stamped layout.

3. [VRAM safety becomes profile-driven] The plan claims `wrapper_bridge` will resolve the "in-graph profile stamp -> committed profile vram_budget_mb". `wrapper_bridge` is a low-level module that does not receive the workflow JSON or node 63's properties at execution time, and the plan forbids "new data wiring through the graph".
   Fix: Have the "startup assertion" (which reads the stamp at load/first-queue) export the resolved `vram_budget_mb` to `os.environ["OTR_VRAM_CEILING_MB"]` automatically, so `wrapper_bridge` only needs to read the environment variable.

4. [Validator + node 63] The plan says the generator sets node 63's path to the "repo-relative path". But `_otr_workflow_validator.py` resolves non-empty paths via `Path(path)`, which resolves relative to the ComfyUI process CWD, not the repository root. This will cause `FileNotFoundError` depending on how ComfyUI is launched.
   Fix: Modify `_load_workflow` in `_otr_workflow_validator.py` to resolve non-absolute paths against `_REPO_ROOT` (e.g., `p = _REPO_ROOT / path`).

SHOULD-FIX:
1. [Headless = same applier] The plan restricts ad-hoc patching to a whitelist of "target_words, seeds, prompt fields". However, `queue_smoke.py` currently patches `openrouter_slot_a_model` and `comfy_slot_a_model` dynamically based on live schemas.
   Fix: Add the remote-LLM slot pickers (`openrouter_slot_*`, `comfy_slot_*`) to the `apply()` patch whitelist, or handle their default resolution inside `apply()` itself.

OPTIONAL / NICE-TO-HAVE:
- [Determinism] Log a warning if `request_seed` is patched but `seed_mode` is left as `fixed` instead of `request_hash`, as this is a common operator error that breaks determinism.

CUT THESE (over-engineering):
1. [Decision A] "CI asserts `apply(master, 16gb_full) == master` byte-identical."
   Why it is safe to cut: The master JSON contains ComfyUI UI state (node positions, sizes, colors) that `apply()` does not manage. A byte-identical check on the whole file will fail if the UI saves formatting changes. Assert dict-equality of `to_api_prompt(master)` vs `to_api_prompt(apply(master, 16gb_full))` instead.

[ASSUMPTION] I am assuming `OTR_VideoRenderBatch` (node 92) has its engine default widget named exactly "engine" or similar, and that it sits at index 4 (where "humo" is in the JSON excerpt), as its `INPUT_TYPES` was not provided to verify the exact widget name.
[ASSUMPTION] I am assuming `NODE_CLASS_MAPPINGS` can be imported safely without a live ComfyUI server, provided CUDA/torch dependencies are mocked or deferred (which S3 mentions testing).