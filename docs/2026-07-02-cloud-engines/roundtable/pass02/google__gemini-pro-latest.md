<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan breaks ComfyUI widget validation, violates explicit registry invariants, and demands pre-run calculations of runtime-dynamic data.

MUST-FIX BEFORE BUILD:
1. [Section 1 & 2] **Import-gated registration breaks saved workflows.** The plan states cloud adapters will not register if `OTR_ENABLE_COMFY_CLOUD_MEDIA=1` is off. If an engine is unregistered, `engines_for_role()` omits it. Any saved workflow with that engine selected in a COMBO widget will instantly fail to load (ComfyUI validation error: value not in list).
   *Fix:* Always register the cloud adapters so the COMBO values remain valid, but enforce the flag inside `assert_usable()` by raising `EngineUnusable(..., EngineUsabilityReason.GATED_BY_FLAG)` (which the grounding explicitly supports).
2. [Section 4d] **Video registry invariant violation.** The plan adds new video engines (`kling_avatar`, `seedance_2`, etc.) but fails to update the `CAPABILITIES` dictionary. The video `registry.py` grounding explicitly states: "the registry-consistency invariant forbids a CAPABILITIES row without a registered engine" (and vice versa). The profile enable-set will crash.
   *Fix:* Add explicit `CAPABILITIES` rows for every new cloud video engine in `registry.py` with `"vram_class": "cpu"` and `"cpu_ok": True`.
3. [Section 6] **`default_engine_for_role` signature mismatch.** The plan claims the profile applier injects a map "consumed by `default_engine_for_role`". Per the grounding, `default_engine_for_role(role: str)` takes no profile context; it purely reads the static `default_roles` tuple on the registered classes.
   *Fix:* Apply the cloud profile overrides downstream at the widget-population level, or change the function signature to `default_engine_for_role(role, profile)`.
4. [Section 2] **Impossible pre-run cost estimate.** The plan requires a "Pre-run COST ESTIMATE printed per episode (rows x beat counts) before first submit". The LLM writer generates the beats dynamically; the exact beat count is unknown until the writer node finishes executing.
   *Fix:* Move the cost estimate calculation to mid-run (after the LLM writer outputs the parsed script, but before media dispatch).
5. [Section 2] **Mutating environment variables per run.** The plan states "Budget env `OTR_CLOUD_MEDIA_BUDGET_USD`, reset per run". Mutating `os.environ` per run in a server process causes race conditions and bleeds across concurrent sessions.
   *Fix:* Read the env var once as a static ceiling; reset an internal `_run_budget_spent` accumulator per execution (matching the `_run_token_total` pattern in `_otr_comfy_backend.py`).

SHOULD-FIX:
1. [Section 4b] **Invented API provider.** The plan lists `SoniloTextToMusic`. "Sonilo" is not a real generative AI provider; this is a hallucination of "Suno".
   *Fix:* Change to `SunoTextToMusic`.
2. [Section 2] **Episode-scoped cache defeats idempotency.** Keying the billing cache into `otr\episodes\<ep>\` means identical prompts generated in *different* episodes will bypass the cache and re-bill the user.
   *Fix:* Store the billing cache in a global `otr\cache\cloud_media\` directory.
3. [Section 2] **Headless auth hidden inputs.** Hidden inputs (`auth_token_comfy_org`) are populated by the browser's JS app. On a headless server, they will be entirely missing unless the API client explicitly crafts them into the JSON payload.
   *Fix:* Ensure the S0 smoke test explicitly updates the headless API submission script to inject these tokens, rather than expecting them to magically populate.

CUT THESE (over-engineering):
1. [Section 1 & 5] **Surface B (Comfy Cloud Workflow).** Implementing headless submit/poll/download logic inside a ComfyUI node will block the local worker thread indefinitely, stalling the entire local queue. It introduces complex distributed state management that is explicitly "UNPROVEN".
   *Why it is safe to cut:* Surface A (synchronous partner nodes) already handles request lifecycles natively. Cut Surface B entirely for pass01 to guarantee delivery.