VERDICT: build-ready as-is? yes-with-fixes. The 3 cleanup chunks (C1-C3) are structurally sound and verified against real code, but require explicit test coupling definitions and tooltip/docstring cleanups to prevent stale metadata and phantom test mocks.

MUST-FIX BEFORE BUILD:
1. [Section 1 C3 / Section 4 P3] Formally bind `tests/test_cascade_freeze_unload_visible.py` and `tests/test_lfc_b1_cascade_unload_in_finally.py` to the C3 edit chunk.
   - Defect: Both `tests/test_cascade_freeze_unload_visible.py:59-68` and `tests/test_lfc_b1_cascade_unload_in_finally.py:55-64` explicitly stub `nodes._otr_model_loader.request_slot` and `nodes._otr_model_loader.make_generate_fn`. While Section 4 P3 mentions modifying them, Section 1 C3 lists only `tests/test_llm_runtime_policy.py:339` as mandatory test coupling. If `request_slot` is removed from `nodes/OTR_LedgerFreezeCascade.py:281`, existing test stubs will pass silently on unused mocks without asserting the real invariant (zero acquisition calls).
   - Fix: Add both test files to Section 1 C3's coupling list. Replace the successful mock returns in both test harnesses with poison sentinels (`MagicMock(side_effect=AssertionError("C3 violation: request_slot called during freeze"))`) to verify behavioral elimination of acquisition.

2. [Section 1 C3] Update stale user-facing tooltip in `nodes/OTR_LedgerFreezeCascade.py:144-152`.
   - Defect: In `nodes/OTR_LedgerFreezeCascade.py:147`, `INPUT_TYPES` tooltip for `technical_model` claims: `"the cascade uses it only for bounded same-story safety cleanup on inline banks. Validated via _otr_model_inputs.require_model -- an unwired socket raises MissingModelInputError loud."` Same-story safety cleanup was retired on 2026-08-05 (`nodes/_otr_freeze_cascade.py:676`), and C3 deletes acquisition. Retaining this description publishes false UI documentation on the canonical node canvas.
   - Fix: Update the tooltip string in `nodes/OTR_LedgerFreezeCascade.py:147` as part of C3 to state that `technical_model` validates upstream graph connection from the writer but performs no model loading.

SHOULD-FIX:
1. [Section 1 C2] Align `_check_per_line_invariants` docstring in `nodes/_otr_ledger_freeze.py:268-275`.
   - Defect: `nodes/_otr_ledger_freeze.py:270` docstring states `line_id unique; char_id non-empty for voiced beats; ...`. When duplicate checking (`seen_line_ids:299`, `310-313`) is removed to establish G8 (`_check_g8_line_id_uniqueness:893`) as sole owner, leaving `line_id unique` in the invariant docstring creates contradictory documentation.
   - Fix: Update the docstring at `nodes/_otr_ledger_freeze.py:270` to remove `line_id unique` and explicitly reference G8 for collision diagnostics.

2. [Section 1 C3] Clean up stale model-residency comments in `nodes/OTR_LedgerFreezeCascade.py:336-342`.
   - Defect: Lines 336-339 state: `# Serialize + rebuild WHILE the model is still loaded. Neither touches torch tensors ... so placement order is safe -- the model could already be released here.`
   - Fix: Replace with an accurate comment noting that serialization occurs post-disposition and no model is held in memory.

OPTIONAL / NICE-TO-HAVE:
1. [Section 3 / Section 4 P1] Capture and record sha256 checksums of the deterministic master WAV output from `scripts/otr_canonical_audio_check.py` before and after C1 to provide direct evidence of zero DSP/timing divergence.

CUT THESE (scope / over-engineering):
1. [Section 4 P0-P4] Intermediate full-suite (12,000+ test) runs between isolated candidate commits.
   - Why safe to cut: Candidates C1 (`nodes/scene_sequencer.py`), C2 (`nodes/_otr_ledger_freeze.py`), and C3 (`nodes/OTR_LedgerFreezeCascade.py`) modify disjoint files with separate call graphs. Running focused checks (`tests/test_canonical_audio_check.py`, `tests/test_g8_line_id_uniqueness.py`, `tests/test_cascade_freeze_unload_visible.py`, `tests/test_input_types_signature_parity.py`) per chunk, while running the full regression suite and Bug Bible at P0 intake and P4 final qualification, maintains identical verification rigor while saving substantial cycle time.

ASSUMPTIONS:
- [ASSUMPTION]: External log parsers do not rely on the exact error string `lines[idx] line_id='...' is duplicated` from `_check_per_line_invariants` (which changes to the structured `G8: N duplicate line_id(s) across ledger.lines[]: ...` message).
- [ASSUMPTION]: `nodes._otr_model_loader.unload_llm_if_local_resident()` at `nodes/OTR_LedgerFreezeCascade.py:376` is idempotent and safe when `LLM_CACHE` is empty.
