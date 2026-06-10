<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core split of semantic patching vs artifact stamping is sound, but the offline generator's schema lifecycle and the graph's topological wiring have critical mechanical flaws that will break the build.

MUST-FIX BEFORE BUILD:
1. [How a profile reaches the graph] Offline schema generation will reject valid profile engines. `patch_widget_by_name` validates values against COMBO choices. For `OTR_VideoDirector`, choices are dynamically built from `_vreg.all_engine_names()`. If the offline generator calls `INPUT_TYPES()` before the engine adapters are imported, the registry is empty, the COMBO choice list is just `["+ Add Custom Model"]`, and patching fails.
   - Fix: The offline generator must explicitly import the OTR package root (triggering adapter self-registration) *before* instantiating the `INPUT_TYPES -> /object_info` shape adapter.
2. [Ordering guarantee] Wiring Node 63 to a *new* input on Node 92 (`OTR_VideoRenderBatch`) fails to protect Node 91 (`OTR_ImageGenDispatcher`), which also uses heavy VRAM and runs *before* Node 92. It also risks widget-vector drift on Node 92.
   - Fix: Wire Node 63's `validation_report` to the *existing* `gate_in` input on Node 87 (`OTR_VideoDirector`). Node 87 already declares `gate_in` with `forceInput=True` (no widget drift), and topologically precedes ALL heavy image and video nodes.
3. [Validator + node 63 fixes] Adding 3 new optional STRING widgets to Node 63's `INPUT_TYPES` will immediately fail the queue-time `widget_vector_drift` contract test on the master JSON, because its saved `widgets_values` array is currently length 3 (`["", true, true]`).
   - Fix: Explicitly state that the master JSON's `widgets_values` for Node 63 must be padded to length 6 (e.g., appending `"", "", ""`) in the exact same commit that updates `INPUT_TYPES`.
4. [Precedence (availability)] The backend registries compute `enabled(P)` to intersect with reality and to log LOUD warnings if `OTR_FORCE_ENGINE_MAP` bypasses the profile. But the backend has no way to know which profile `P` is active during execution, because the profile was applied statelessly at submit-time.
   - Fix: Node 63's `execute` method must also export `os.environ["OTR_ACTIVE_PROFILE"] = profile_id` alongside the VRAM ceiling, so the execution-time registries can load the correct policy.

SHOULD-FIX:
1. [Decision A / Parity gate] `to_api_prompt` does not automatically drop standard STRING widgets. If the Parity Gate compares the API prompts, the snapshot will contain the 4 populated Node 63 widgets, while the applied master will have them empty.
   - Fix: The Parity Gate test must explicitly `del prompt["63"]["inputs"]["workflow_json_path"]` (and the 3 stamp fields) from both dictionaries *before* asserting dict-equality.

OPTIONAL / NICE-TO-HAVE:
- [Headless = same applier] When adding `--profile <id>` to `queue_smoke.py`, ensure the script still prints the resolved profile to stdout so CI logs remain easily readable without digging into the API prompt payload.

CUT THESE (over-engineering):
- (None beyond what the plan already explicitly cuts in "Decision B" and "CUT from this build". The scope is appropriately tight.)