VERDICT: build-ready as-is: yes. The plan and implemented diff have converged; the media_archive collision is properly gated at the source kind, prompt builders and fallback paths are unified with direct attribute access, and test assertions are committed and passing.

MUST-FIX BEFORE BUILD:
None — plan converged.

SHOULD-FIX:
1. [nodes/_otr_source_identity.py] `ADAPTATION_SOURCE_KINDS` is omitted from `__all__`.
   Defect: In [`nodes/_otr_source_identity.py`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_source_identity.py#L38-L42), the module defines `__all__ = ["SOURCE_IDENTITY_VERSION", "SourceIdentity", "identity_from_meta"]` but does not include the newly exported constant `ADAPTATION_SOURCE_KINDS`. While attribute access (`_OTRSID.ADAPTATION_SOURCE_KINDS`) works, public constants exported for cross-module consumption should be declared in `__all__`.
   Fix: Add `"ADAPTATION_SOURCE_KINDS"` to `__all__` in [`nodes/_otr_source_identity.py`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_source_identity.py#L38-L42).

OPTIONAL / NICE-TO-HAVE:
- In [`tests/test_cross_play_frame_leak.py`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_cross_play_frame_leak.py#L222), `test_a_composed_frame_carries_its_own_work_and_no_other` parametrizes three Shakespeare titles ("Twelfth Night", "The Tempest", "Macbeth"). Adding a test case for a public domain title from `config/source_banks/public_domain_story/manifest.sample.json` would provide explicit test coverage parity for the `public_domain` bank.

CUT THESE:
None — the diff is minimal, lean, and appropriately constrained (1 helper function `_work_line`, 1 frozenset constant `ADAPTATION_SOURCE_KINDS`, default-valued fields on `SafeOpenBrief` and `OutlineRequest`, 2 story pack prompt updates, and regression tests). No unnecessary runtime machinery remains.

VERIFY-AT-BUILD:
1. **Live LLM Output on Shakespeare Adaptation (Twelfth Night)**:
   - *Status*: Flagged UNVERIFIABLE via unit tests in R1–R3 because deterministic assertions verify that `WORK: a scene from Twelfth Night` reaches prompt context and that foreign terms are absent from inputs, but cannot prove stochastic LLM adherence without a model execution pass.
   - *Verification Action*: Run a live inference batch on a Twelfth Night scene through the ComfyUI workflow (`workflows/otr_canonical.json` or CLI runner) on GPU, inspect the generated announcer opening in the output ledger receipt, and confirm that the announcer properly contextualizes the work and does not hallucinate a foreign setting (e.g., Verona / Capulets).
2. **Canonical Workflow Integrity (`workflows/otr_canonical.json`)**:
   - *Status*: Verified by diff analysis that no node surface signatures (`INPUT_TYPES`, `RETURN_TYPES`, widget order) were modified.
   - *Verification Action*: Confirm that [`workflows/otr_canonical.json`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/otr_canonical.json) loads cleanly in ComfyUI DevMode without missing widget warnings or slot shift warnings.
