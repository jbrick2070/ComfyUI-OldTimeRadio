VERDICT: yes-with-fixes -- C3 node/orchestrator/unload order is coherent, but the P3 fresh-process proof as written will bind the wrong loader module, can mis-bind node 62 widgets, and can assert the wrong freeze/cleanup literals.

MUST-FIX BEFORE BUILD:
1. [P3.2 / scripts/otr_canonical_audio_check.py:33-42] Child `load_package()` registers the pack as `ROOT.name.replace("-", "_")` via `spec_from_file_location` and inserts `ROOT.parents[1]` (ComfyUI) on `sys.path`. `import nodes` then hits ComfyUI stock nodes, not OTR. Freeze `run()` lazy-imports `from . import _otr_model_loader` as `<pack>.nodes._otr_model_loader` (`OTR_LedgerFreezeCascade.py:207`, `__init__.py:386`). Poisoning pytest `nodes._otr_model_loader` in that child is a no-op, so acquisition can still run. Fix: `loader = importlib.import_module(package.__name__ + ".nodes._otr_model_loader")` (same pattern for production_ledger, `_otr_freeze_cascade`, `_otr_content_authorship`). Patch `request_slot` / `make_generate_fn` on that object. Do not `sys.path.insert` the OTR repo root in the child.

2. [P3.1 vs audio_check.main:258-260] Child env names only `OTR_OUTPUT_DIR`, `OTR_EXTRA_OUTPUT_ROOTS`, and popping `OTR_OBS_DIR`. `load_package()` execs full `__init__.py` and loads every node. A replacement `env={}` drops `PATH` / `CUDA_VISIBLE_DEVICES`. Fix: inherit the parent environ; `os.environ.update` the same keys as `main()` (`CUDA_VISIBLE_DEVICES=""`, `OTR_TEST_MODE="1"`, output roots); pop `OTR_OBS_DIR`; set UTF-8 / `PYTHONDONTWRITEBYTECODE=1`; do this before `load_package()`.

3. [P3.3 / workflows/otr_canonical.json:1052-1177] Node 62 has five `forceInput` sockets plus two widget inputs; `widgets_values` is `[true, true]`. Binding "widgets by saved order" across all `inputs` assigns those booleans to `script_text`. Fix: zip only `inputs` entries with a `widget` key to `widgets_values` by name (`enable_phase_7_audio_readiness`, `enable_phase_8_video_readiness`). Pass the five synthetic writer values as kwargs by socket name. Link `[115, 1, 4, 62, 4, "STRING"]` is writer `technical_model` output 4 (`otr_canonical.json:494-501, 3050-3056`); follow that by name, do not assume positional sockets equal widget slots.

4. [P3.5 vs _otr_freeze_cascade.py:696-700, 796-799, 1119-1124] "Successful frozen verdict" is not a return literal. Astra's fixture receipt is `frozen_with_warns`; the fixture has no `portrait_path`, so phase 8 warnings promote `frozen_clean` via 1119-1124. Cleanup status is bank-split: `original` -> `retired_no_content_policy`; `scifi_news_pro` -> `not_applicable_content_owned`. Fix: assert `result[4] in {"frozen_clean", "frozen_with_warns"}` (or pin `frozen_with_warns` from the receipt) and assert `meta.same_story_safety_cleanup.status` by bank, not one shared string.

SHOULD-FIX:
1. [P3.5 sequencing] Node `run()` then `run_freeze_cascade` on the same `_CURRENT` ledger is not independent. For `scifi_news_pro`, a second pass re-hashes authorship at 1011-1038. Rebuild or deepcopy the pre-freeze fixture for the poison-callback call. Node path already passes `None`; the callback probe only needs a never-frozen ledger.

2. [C3 comments vs OTR_LedgerFreezeCascade.py:336-413] "Serialization precedes the unload receipt" describes dumps #1 (343) only. Dumps #2 (404) exists so `freeze_unload_ok` reaches the wire. The B1 first-only/second-only cases depend on that split. Say so in the two comments; do not rewrite the blocks.

3. [P3 B1 json patch] Patch `nodes.OTR_LedgerFreezeCascade.json` only. Success-path dumps are exactly 343 then 404. `_no_ledger_error_json` (46) uses the same name. Assert two dumps and that this helper is not entered, or a leaked no-ledger path shifts the counter.

4. [C3 vs tests/test_lfc_phase_7_8_readiness.py:225-280] Those tests still pass a lambda into `run_freeze_cascade`. Signature stays positional-required; `None` from the node is compatible because `_run_inline_safety_cleanup` only `del generate_fn` (676-692) and never calls it. Do not add a default or type-guard that those tests would miss.

5. [P3 owned-file list vs C3 command] `test_lfc_b14_unload_on_exit.py`, `test_freeze_cascade_v2_ports.py`, `test_canonical_replay.py` are run but not edited. b14 still matches `_OTRML.unload_llm_if_local_resident(` after the cascade call (test_lfc_b14:49-61). Keep them as nets; if any still source-pins `request_slot` / `policy_from_meta` in the freeze node, fold that edit into the C3 chunk. verify: no other freeze source-pin beyond `test_llm_runtime_policy.py:339-347` (that one is already owned).

OPTIONAL / NICE-TO-HAVE:
- Pin child receipt fields: pack module name, loader `id(loader)`, node 62 widget names, link 115 endpoints, both cleanup statuses.
- 120 s timeout is per child; parametrize is two processes. Leave it unless first-import of the full pack exceeds it on this box.

CUT THESE:
1. A second workflow JSON or ComfyUI scheduler for this proof. Direct `FUNCTION` invoke on `NODE_CLASS_MAPPINGS["OTR_LedgerFreezeCascade"]` is the right seam; `IS_CHANGED` returns `time.time()` (187-189) and is not in this path.
2. Retaining `request_slot` on freeze to keep shot-lock diagnostics green. Shot-lock still does `request_slot("technical", model_id, policy=policy_from_meta(meta), ...)` (`otr_shot_lock.py:1721-1724`). Under `OTR_TEST_MODE=1` it returns `(None, "")` at 1698-1699 and never admits. That is a different consumer.

Grounded non-defects (do not reopen):
- `run_freeze_cascade(None, led, ...)` matches the live first positional (`_otr_freeze_cascade.py:714-716`). No `generate_fn(` call exists in that module.
- `require_model` stays before `try` (`OTR_LedgerFreezeCascade.py:267-269` vs 308). Blank / `"   "` still raises `MissingModelInputError` (`_otr_model_inputs.py:79-85`) with no unload. Replay / no-ledger still return first (`215-263`). `has_current_ledger()` is true only when `_CURRENT` has at least one line (`production_ledger.py:835-844`); the one-line fixture satisfies that.
- Final `_OTRML.unload_llm_if_local_resident()` in `finally` (376) plus empty-cache no-op (`_otr_model_loader.py:1718-1729`) matches the branch table. `freeze_unload_ok=True` means the call did not raise, not that a model was resident.
- Tooltip-only `INPUT_TYPES` change does not move widgets or links. `test_workflow_json_guardrails.py` TestCascadeB3Surface only requires a `technical_model` key, not the tooltip text.
- C2 before C3 is the right order: the P3 real cascade still runs G8. P0.5 before C3 is required because the child imports `load_package`.
- `stamp_receipt(..., owner_bank=bank)` plus one sayable `c01` line satisfies `require_voice_coverage` (`_otr_cast_voice_coverage.py:128-185`). [ASSUMPTION] `_sayable("The signal is clear.")` is true.

UNTRACED: live shot-lock admission of `unused-model-boundary` outside `OTR_TEST_MODE` (plan already treats that as out of scope). Full-suite failing-node-ID set at implementation HEAD (plan already requires recapture).
