VERDICT: build-ready as-is? yes. The wiring, sequencing, interface contracts, teardown order, and child-process boundaries are tightly specified and grounded in the live repository.

MUST-FIX BEFORE BUILD:
None.

SHOULD-FIX:
1. [§4.P3] Child process sys.path alignment in fresh freeze test: scripts/otr_canonical_audio_check.py:35 inserts ROOT.parents[1] at sys.path[0] inside load_package(). When test_canonical_freeze_without_acquisition_in_fresh_process launches Python with "-", ensure the child script imports scripts/otr_canonical_audio_check.py via absolute path (Path(sys.argv[1]) / "scripts" / "otr_canonical_audio_check.py") and invokes load_package() before attempting any submodule imports, so package-relative resolution (ComfyUI_OldTimeRadio.nodes.*) is consistently bound before fixture setup.

2. [§4.P3] JSON serialization error-injection scoping in test_lfc_b1_cascade_unload_in_finally.py: When wrapping json.dumps to fail selectively on call 1 vs call 2, ensure the counter intercepts only calls originating from nodes/OTR_LedgerFreezeCascade.py (specifically json.dumps(led.data, ...) at lines 343 and 404), so helper functions (e.g. _seed_to_int64 or test assertions) are not inadvertently counted or tripped during fixture evaluation.

OPTIONAL / NICE-TO-HAVE:
- [§1.C3 / §4.P3] In the fresh child-process JSON receipt, include the recorded freeze_verdict and freeze_unload_ok boolean explicitly alongside output SHA hashes to provide immediate readable verification without parsing full ledger payloads.

CUT THESE (over-engineering):
1. [§4.P3] Re-invoking run_freeze_cascade a second time with a poison callback after node.run() has already completed: Safe to cut or make secondary. The primary node invocation with poisoned loader.request_slot and loader.make_generate_fn already exercises the real cascade entry path (_LFC_ORCH.run_freeze_cascade(None, led)) and confirms _run_inline_safety_cleanup never invokes generation.

[ASSUMPTION] None. All assertions, symbol names, and link IDs were verified directly against workflows/otr_canonical.json, nodes/OTR_LedgerFreezeCascade.py, nodes/scene_sequencer.py, nodes/_otr_ledger_freeze.py, and the corresponding test modules.
