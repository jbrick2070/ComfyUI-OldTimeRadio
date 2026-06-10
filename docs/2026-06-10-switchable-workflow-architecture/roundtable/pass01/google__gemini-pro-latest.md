<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan misdiagnoses the mechanical root of the drift bug, proposes a build step that will corrupt the workflow contract, and ignores a hardcoded VRAM ceiling that will instantly OOM the proposed 8GB tier. 

MUST-FIX BEFORE BUILD:
1. [Reconciliation] **Drift bug misdiagnosis.** The plan claims generating per-tier JSONs "structurally kills the drift bug." It does not. The grounding explicitly states the drift is between the JSON defaults and the *hard-coded patch lists in headless scripts* (e.g., `queue_smoke.py` patching OpenRouter slots). Generating 3 JSONs multiplies this problem unless the headless scripts change.
   *Fix:* Explicitly mandate that headless scripts (like `queue_smoke.py` and `otr_api.py` consumers) must load the *generated* tier JSON and MUST NOT patch engine/model/toolchain widgets. Headless scripts must be restricted to patching only creative request inputs (prompts, seeds, `act_count`).

2. [The hard parts / 3] **Hardcoded VRAM ceiling will OOM the 8GB tier.** The plan mandates "the 8GB tier must auto-exclude 14B", but grounding shows `VRAM_CEILING_MB = 14500` is hardcoded in `nodes/_otr_video_engines/wrapper_bridge.py:37`. The 8GB profile will instantly crash when the wrapper bridge allows up to 14.5GB of models to co-reside.
   *Fix:* The profile layer must export an `OTR_VRAM_CEILING_MB` environment variable. Modify `wrapper_bridge.py` to read this env var, falling back to 14500 only if unset.

3. [Reconciliation] **Generator build step will corrupt `widgets_values`.** The plan says the BUILD step "loads the master, sets the switches... and exports." If this build script modifies the JSON statically without live ComfyUI schemas, it will violate the `_otr_workflow_validator.py` contract (which enforces `forceInput` omissions and hidden `control_after_generate` companions) and fail at queue time.
   *Fix:* The generator script MUST use `otr_api.patch_widget_by_name` and run against a live ComfyUI instance (using `fetch_schemas()`) during the build process to guarantee the exported JSONs have perfectly aligned `widgets_values` arrays.

4. [Reconciliation] **Missing centralized MPS routing.** The plan acknowledges Mac/MPS is a real tier but leaves the implementation vague ("handled by the sidecar/venv"). Grounding shows ~35 scattered `torch.cuda.is_available()` checks. You cannot toggle Mac support without refactoring these.
   *Fix:* Create a centralized `nodes/_otr_shared/device_routing.py` module to handle `cuda` vs `mps` vs `cpu` resolution, and replace all 35+ hardcoded CUDA checks before shipping the Mac profile.

SHOULD-FIX:
1. [Strawman shape] **`OTR_FORCE_ENGINE_MAP` collision.** The plan dictates switches come from "profile + `OTR_ENABLE_*` flags + Director widgets". It ignores that `OTR_FORCE_ENGINE_MAP` already exists and is actively used by the marathon soak runner.
   *Fix:* Explicitly define the precedence order in the resolver: `OTR_FORCE_ENGINE_MAP` > Director Widgets > `OTR_ENABLE_*` flags > Fallback registry.

2. [The hard parts / 5] **Seed widget naming trap.** The plan mentions "seed-keyed" determinism. Note that `OTR_VideoDirector` explicitly names its seed `request_seed` to avoid ComfyUI's auto-injected `control_after_generate` companion (V-7). 
   *Fix:* Ensure the profile UX/wizard does not attempt to patch a `seed` widget on `OTR_VideoDirector`, but specifically targets `request_seed` as a standard `INT`.

CUT THESE (over-engineering):
1. [Reconciliation] "Parse `HF_HOME` BEFORE downloading".
   *Why to cut:* `huggingface_hub` natively respects the `HF_HOME` environment variable. Writing custom parsing logic for it in the installer is redundant and error-prone. Just ensure the env var is passed to the subprocess.