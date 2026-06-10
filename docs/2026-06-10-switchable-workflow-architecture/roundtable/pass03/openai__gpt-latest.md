<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: build-ready as-is? no. Several core safety gates are internally inconsistent or wired too late, and one named implementation target does not exist.

MUST-FIX BEFORE BUILD:
1. [How a profile reaches the graph / Startup assertion] The plan names `OTR_WorkflowValidator.execute`, but the grounded class uses `FUNCTION = "validate"` and implements `WorkflowValidator.validate(...)`; there is no `execute` method. Concrete fix: implement the stamped profile assertion and env export in `WorkflowValidator.validate`, or change the node `FUNCTION` and method consistently.

2. [How a profile reaches the graph / Startup assertion] `OTR_VRAM_CEILING_MB` is exported “if not already set”. In a long-running ComfyUI process, `os.environ` persists across prompts, so a previous 16GB run can leave a 14500/16GB ceiling in place and a later `8gb_lite` snapshot will not downshift. Concrete fix: for stamped snapshots, set the ceiling to the resolved profile value every execution, or reject when an externally-set value conflicts; do not silently keep an older higher value. Also ensure `wrapper_bridge` reads the env dynamically at dispatch time, not via an import-time constant.

3. [How a profile reaches the graph / Ordering guarantee] Wiring node 63 only into node 92 does not make the assertion precede all heavy visual work. In the grounded graph, node 91 `OTR_ImageGenDispatcher` is upstream of node 92 and can execute before node 92’s new gate input; image generation may load heavy models before the profile mismatch aborts. Concrete fix: wire the validator gate into the earliest heavy visual dispatch node(s), at minimum node 91 and node 92, or explicitly accept/document that image generation may run before profile validation.

4. [How a profile reaches the graph / Ordering guarantee] The new node-92 ordering input is underspecified. If it is declared as a normal optional `STRING`, it becomes widget-backed and changes `widgets_values` layout; the existing code already uses `forceInput=True` for socket-only gates such as `OTR_VideoDirector.gate_in`. Concrete fix: declare the new `OTR_VideoRenderBatch` gate input as `("STRING", {"default": "", "forceInput": True, ...})`, update the master links and `last_link_id`, and do not add a widget slot for it.

5. [How a profile reaches the graph / Startup assertion] Existing `validate_anyway=False` currently returns early before any validation. If the profile assertion is added inside the same method without special handling, a generated snapshot can bypass hardware/profile safety by setting `validate_anyway` false. Concrete fix: make stamped profile assertion mandatory whenever `profile_id` is non-empty; `validate_anyway` may only skip the workflow-contract check, not the profile/platform/VRAM/toolchain assertion.

6. [How a profile reaches the graph / Precedence] The selection and availability precedence rules contradict each other. The plan says `OTR_FORCE_ENGINE_MAP` bypasses the profile enable-set, but then says availability is `registry reality INTERSECT OTR_ENABLE_* INTERSECT enabled(P)`, which would exclude any force outside the profile. Concrete fix: define two paths: normal availability = registry reality ∩ env gates ∩ enabled(P); forced availability = registry reality ∩ required models/toolchains/env gates, explicitly not `enabled(P)`, with the planned LOUD warning.

7. [How a profile reaches the graph / Precedence] `OTR_FORCE_ENGINE_MAP` is grounded as existing only in `nodes/_otr_video_engines/render_driver.py` with `role=engine` / `*=engine` grammar. The plan states global engine-selection precedence while also managing audio and image slots. Concrete fix: either scope `OTR_FORCE_ENGINE_MAP` explicitly to video only, or implement equivalent force handling for audio/image/profile-managed slots and add tests for each namespace.

8. [Widget coverage] The proposed coverage test only enumerates `COMBO/STRING` widgets “whose saved value is a registered engine id”. That will miss profile-managed feature booleans and non-engine drift, including the stated acceptance bug class: captions, procgen credits, and LTX radio open. Concrete fix: make coverage enumerate all profile-managed schema keys from the profile mapping, including BOOLEAN/INT/feature widgets and seed-policy widgets, and fail if any profile field has no graph target or any managed widget is patched outside the profile applier.

9. [Widget coverage] The coverage criterion based on “saved value is a registered engine id” misses dropdowns whose current saved value is a sentinel/default but whose choices include registered engines. Concrete fix: classify by schema choices intersecting registry engine ids, not by the current saved value alone; keep explicit exemptions with reasons.

10. [Widget coverage] The initial classification cites “Director per-role dropdowns (nodes 19/88 types)”, but the grounded workflow has `OTR_VideoDirector` at node 87 and `OTR_ImageDirector` at node 88; no node 19 is present in the supplied production JSON. Concrete fix: correct this to node 87/88 or remove raw node ids entirely and rely only on `(node_type, widget_name)` as the plan otherwise requires.

11. [Determinism] The contract says `(profile_id, seed) -> normalized-identical outputs`, but the same section says creative RNGs keep OS entropy by default and the profile example has `cast_seed_env: null`, `style_seed_env: null`. With OS entropy, two runs with the same profile and seed can legitimately diverge in cast/style/ledger fields. Concrete fix: either set deterministic cast/style env values for determinism ship gates, or narrow the determinism contract to exclude OS-entropy-governed creative fields.

12. [Determinism / How a profile reaches the graph] The plan requires ledger records `profile_id + master_hash + snapshot_hash`, but the stamp only has `profile_id`, `master_hash`, and `generated_by`, and no data path is defined from node 63 to the ledger-producing nodes. Concrete fix: add a defined source for `snapshot_hash` and a runtime propagation path into the ledger, or remove `snapshot_hash` from the ledger contract. If adding a fourth stamp field, update parity ignores and widget-vector expectations.

13. [The profile system] `role_overrides` are described as “per-role chains”, but `apply_profile` patches ordinary widgets, and the relevant director widgets hold single engine ids. The validator language also says each override must be “in enabled(P)” as if singular. Concrete fix: define the schema precisely: either role overrides are single selected engine ids for saved widgets, or they are fallback chains with a documented rule for which element is written to the graph and how the rest reaches resolver fallback.

SHOULD-FIX:
1. [How a profile reaches the graph / Schemas] The offline `INPUT_TYPES -> /object_info` adapter must include the same serialization semantics as `scripts/otr_api.py`: forceInput widgets do not consume slots, and `seed`/`noise_seed` companions are synthetic saved slots. The plan mentions tuples-to-lists but not these slot-layout rules. Concrete fix: make the adapter tests compare `_serialized_slot_names` behavior, not just raw schema shape.

2. [Validator + node 63 fixes] The plan says `_load_workflow` should resolve non-empty relative paths against `_REPO_ROOT`; current grounded code uses `Path(path)` directly. This is already identified, but the concrete fix must also update `IS_CHANGED`, which currently uses `Path(workflow_json_path)` directly and would compute mtime against CWD for relative snapshot paths.

3. [Headless = same applier] `queue_smoke.py` currently hard-codes creative patches and remote slot picker patches. The whitelist includes `openrouter_slot_*`/`comfy_slot_*`, but the script dynamically sets them from live schema. Concrete fix: move these through `patch_creative()` or an explicit whitelisted remote-slot helper so the “no direct `patch_widget_by_name` on profile-managed names” regression is enforceable.

4. [Decision A / Identity gate] The identity gate should specify the exact schema source. Today `workflow_to_api_prompt` requires schemas from live `/object_info`; the plan also introduces offline schemas from `NODE_CLASS_MAPPINGS`. Concrete fix: run the identity gate with the same offline adapter intended for CI, plus a separate soak-lane live `/object_info` cross-check, so CI does not depend on a live ComfyUI server.

5. [Decision A / Parity gate] “Ignoring exactly node-63 `workflow_json_path` + three stamp fields” needs a concrete API-prompt path matcher. After conversion, these are entries in node `63.inputs`, not UI widget names. Concrete fix: define the ignore as `prompt["63"]["inputs"][<field>]` by node type lookup, not raw positional widget indexes.

6. [The profile system / Derived enable-set] `model_requirements` in registry metadata should include enough information to distinguish “engine selectable but models missing” from “engine disabled by profile”. The plan says FORCE cannot bypass absent models/toolchains, but does not specify the probe result shape. Concrete fix: define a normalized availability object with reason codes used by validator, wizard, and LOUD queue-start log.

7. [v1 tier set] `cpu_floor` generation is blocked on cold-import, but S0 commits a `cpu_floor` profile before the cold-import result is known. Concrete fix: allow the profile file to exist but mark artifact emission/CI snapshot generation as blocked until S1 passes, or keep `cpu_floor` profile non-shipping until then.

8. [Decision B] The model manifest generated from all of `enabled(P)` can be much larger than needed, because `enabled(P)` is every engine that fits the profile, not necessarily selected/default fallback engines. Concrete fix: generate a default manifest from selected engines/fallback chains, with an optional “download all compatible engines” mode.

9. [How a profile reaches the graph / Startup assertion] The downshift suggestion is underspecified for non-VRAM mismatches such as missing toolchain, platform mismatch, or sidecar absence. Concrete fix: define one reason-to-suggestion table, e.g. `cuda missing -> cpu_floor`, `VRAM < 10GB -> 8gb_lite`, `platform mac -> cpu_floor`.

OPTIONAL / NICE-TO-HAVE:
- Add a small generated “profile application report” artifact listing every patched `(node_id, node_type, widget_name, old, new)` to make parity failures diagnosable.
- Add a read-only note in generated snapshots saying `.gen.json` is generated and naming the source profile.
- Add a CI check that no generated snapshot has `validate_anyway=false`.

CUT THESE (over-engineering):
1. [Decision B] Defer unified optional model download/launcher emission until after S4 tier reality. It is outside the render path and depends on registry metadata that must first stabilize for profile validation.

2. [Determinism] Do not require full per-tier determinism double-run in the same sprint as first snapshot generation. Keep audio byte-identical and API-prompt parity in CI; run full normalized video determinism only as a release gate after the VRAM/profile gates are stable.

3. [The profile system] Avoid adding static `model_requirements` size/download manifest detail for every compatible engine in the first implementation if the immediate goal is switchable workflow profiles. Minimal metadata needed for enable-set validation is enough; detailed downloader manifests can follow in S5.