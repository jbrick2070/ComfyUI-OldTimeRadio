VERDICT: build-ready as-is? yes.
One line why: The finished diff against e07476eb cleanly implements call-local LMFE grammar lifecycle, safe unbounded-prose guard bypassing with preserved cycle detection, and durable MyStory P1 schema binding with zero test regressions.

MUST-FIX BEFORE BUILD:
None — plan converged.

SHOULD-FIX:
1. [tests/test_constrained_generate.py:L238] In `test_generate_invoked_with_prefix_fn`, `fake_torch.no_grad.return_value.__exit__ = MagicMock()` returns a truthy mock by default, which suppresses any exception raised inside `with torch.no_grad():`. If constraint building or `model.generate` raises an error in partial test environments, the error is swallowed and line 339 fails with an uninformative `UnboundLocalError: local variable 'out' referenced before assignment` rather than the underlying failure. Fix: set `fake_torch.no_grad.return_value.__exit__.return_value = False`.
2. [tests/test_constrained_generate.py:L181] `test_generate_invoked_with_prefix_fn` docstring notes `# Requires lmformatenforcer + torch` but only calls `pytest.importorskip("lmformatenforcer")`. In environments where PyTorch is missing or incompatible with installed `transformers` (<2.4), `transformers.StoppingCriteriaList` is a dummy object that raises on instantiation. Fix: add `pytest.importorskip("torch")` at top of test.

OPTIONAL / NICE-TO-HAVE:
- [nodes/_otr_generation_budget.py:L27] Add a docstring note on `ProviderCapacityMessages._otr_unbounded_json_field` clarifying that it disables open-string token limits while leaving token-id cycle detection active.
- [nodes/_otr_my_story.py:L377] The attempt receipt captures `"raw_completion"`; adding `"halt_reason": getattr(error, "halt_reason", None)` would make degeneracy halts explicitly distinguishable from capacity limits in saved ledger diagnostics without string-parsing error messages.

CUT THESE:
1. [docs/PROD_BUG_LOG.md:PBUG-20260910-05 promotion] Safe to cut any in-tree Bible promotion edits from this diff; preserving the current 6-production/4-test boundary cleanly isolates code fixes from live-media verification receipts.

VERIFY-AT-BUILD checklist:
1. Full test suite baseline comparison: Confirm full run against `tmp/my_story_diagnostics_verified_full.xml` matches the 14135 pass / 53 pre-existing fail / 183 skip / 1 xfail baseline with zero new regressions or quarantined tests.
2. Live P1 local schema constraint: Confirm a live one-act My Story execution with local transformers routes through `_otr_bind_schema(StoryTreatment)` and emits valid, parseable treatment JSON on Attempt 1 without tripping the 20-item LMFE default cap or the 2048 open-string threshold.
3. Live cycle detection during unbounded prose: Confirm that if a model enters a token repetition cycle during treatment or act generation with `_otr_unbounded_json_field=True`, `make_degeneracy_criterion` halts decode with `halt_reason="verbatim_cycle"`.
4. Resident memory and weakref reclamation on ComfyUI model unload: Confirm that unloading the model from ComfyUI drops `_otr_lmfe_constraint_cache` and that no request-specific token enforcer prefix states survive in resident memory [ASSUMPTION: ComfyUI execution lifecycle properly releases cache_entry upon workflow model eviction].
5. PBUG-20260910-05 promotion receipt: Once a live one-act run on the corrected commit verifies P1 treatment binding, update `docs/PROD_BUG_LOG.md` status from OPEN to PROMOTED with the verified run receipt.
