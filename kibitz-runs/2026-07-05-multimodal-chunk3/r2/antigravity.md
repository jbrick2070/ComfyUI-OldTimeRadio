VERDICT: build-ready as-is? yes-with-fixes. Synthetic test data is missing the new required boolean field `requires_source_contract`, and the parser's dataclass instantiation / key allowlist updates are not fully defined in the plan, which will crash validation and tests.

MUST-FIX BEFORE BUILD:
1. [Section 2] The test helper `_pipe_row` in [test_story_routing_stage2.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_story_routing_stage2.py#L48-L57) is missing the new pipeline field `requires_source_contract`. Because `_parse_pipeline` will require it to be a boolean, loading the synthetic registry in tests will fail with a `RegistryValidationError`.
   Fix: Add `"requires_source_contract": False` to the default dict in `_pipe_row` inside [test_story_routing_stage2.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_story_routing_stage2.py#L48-L57).
2. [Section 1b] `StoryPipeline` dataclass instantiation inside `_parse_pipeline` is not updated to receive the new `requires_source_contract` parameter. This will cause a `TypeError` due to a mismatch between parsed values and dataclass attributes.
   Fix: Update the `StoryPipeline` dataclass definition in [_otr_story_routing.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_routing.py#L76-L82) to add `requires_source_contract: bool` (placed before `notes: "tuple[str, ...]" = ()`), and update the return block of `_parse_pipeline` at [_otr_story_routing.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_routing.py#L273-L280) to pass `requires_source_contract=requires_source_contract`.
3. [Section 1b] In `_otr_story_routing.py`, `_PIPELINE_KEYS` is not updated to include `requires_source_contract`. Consequently, `_check_unknown_keys` will raise `RegistryValidationError` on every registry load because `requires_source_contract` in `pipelines.json` will be treated as an unknown key.
   Fix: Add `"requires_source_contract"` to the `_PIPELINE_KEYS` frozenset in [_otr_story_routing.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_routing.py#L180-L183).

SHOULD-FIX:
1. [Section 1a] The plan specifies typed error classes (`SourcePayloadError` base, etc.) but does not explicitly state that they must all inherit from `SourcePayloadError`. If any error class does not subclass it, catching all source-payload related errors with a single try-except block or performing AST guards will fail.
   Fix: Ensure all typed error classes (`UnknownFetcherError`, `UnknownInterpreterError`, `SourceContractMissingError`, `SourcePayloadContractError`, and `SourceInterpretError`) explicitly inherit from `SourcePayloadError` in [_otr_source_payload.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_source_payload.py).
2. [Section 1b] The import of `_otr_source_payload` in `_otr_story_routing` is not detailed.
   Fix: Use a package-relative lazy-compatible import `from . import _otr_source_payload` at the top level of [_otr_story_routing.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_routing.py) to match the module's established import pattern.

OPTIONAL / NICE-TO-HAVE:
- [Section 1c] In `OTR_LedgerScriptWriter.py`, import `_otr_source_payload` locally inside `_resolve_inputs` and `run` to align with the module's existing convention of lazy/local imports (e.g., [OTR_LedgerScriptWriter.py:2773](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L2773)) and limit unnecessary top-level dependencies.

CUT THESE (over-engineering):
None.

[ASSUMPTION]
None.
