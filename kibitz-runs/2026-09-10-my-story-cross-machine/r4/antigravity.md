VERDICT: build-ready as-is? yes-with-fixes. The plan has converged on core architecture and data contracts, but carries eight residual build-blocking defects, test regressions, and interface ambiguities across error handling, retry state, and test gates.

MUST-FIX BEFORE BUILD:
1. [Section D / Section 5] Unhandled `StructuredCallFailedError` inside author `post_validator`.
   Defect: In [nodes/_otr_structured_call.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py#L215-L220), `StructuredCallFailedError` is intentionally omitted from `_ATTEMPT_ERRORS`. When the reviewer's inner `structured_call(max_attempts=2)` exhausts its schema/parse retries, it raises `StructuredCallFailedError`. If allowed to escape the author's `post_validator`, this exception bypasses author attempt handling and terminates the outer author pass as an unhandled exception rather than recording an uncertain receipt.
   Concrete fix: Explicitly mandate that the author post-validator reviewer wrapper catches `StructuredCallFailedError`, records a durable review receipt with disposition `"uncertain"`, and returns `None` (clearing the author draft). Genuine system failures (`KeyboardInterrupt`, base `CancelledError`, torch OOM) must remain uncaught to propagate with real types.

2. [Section D / lines 253-260] Missing `post_validator` wiring, instructions, and review context for P0 and P3.
   Defect: In [nodes/_otr_my_story.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L417-L449) and [nodes/_otr_my_story.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L600-L623), `_pass_interpret` (P0) and `_pass_frame` (P3) pass no `post_validator` to `_call` and use generic `make_dispatching_repair_factory()`. The plan mandates that P0 and P3 participate in source review and use `_full_artifact_repair`, but omits the repair instruction strings and validator parameters.
   Concrete fix: Update `_pass_interpret` and `_pass_frame` signatures to accept `review_context`. Wire their `post_validator` arguments to evaluate source contradiction against the candidate artifact. Define concrete repair instructions for `_full_artifact_repair`: for P0 (`StoryInterpretation`), `"Preserve the listener's original story concepts and characters while correcting the interpretation."`; for P3 (`StoryFrame`), `"Preserve the announcer lines, attribution, and music cues while correcting the frame structure."`

3. [Section 1 / Section A] Stale LMFE parser state reused on `min_p` compatibility retry.
   Defect: In [nodes/OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L1039-L1060), when `model.generate` raises `TypeError` due to unsupported `min_p`, `_min_p_unsupported[0] = True` is set and `model.generate(**inputs, **gen_kwargs)` is retried. If `prefix_allowed_tokens_fn` is present in `gen_kwargs`, the first failed generate call may have advanced or mutated internal parser state. Reusing `gen_kwargs["prefix_allowed_tokens_fn"]` directly causes state contamination on the retry.
   Concrete fix: In `OTR_LedgerScriptWriter.py` line 1058, if `schema_model` is bound, re-acquire a fresh `prefix_allowed_tokens_fn` closure and assign it to `gen_kwargs["prefix_allowed_tokens_fn"]` before executing the retry call to `model.generate`.

4. [Section 2 / Section B] Unpinned fit inspection attribute name on scheduler slot callables.
   Defect: Section 2 and Section B require scheduler closures to expose CPU-only fit inspection without incrementing generation counters, but leave the attribute name unspecified. Caller code in Section D cannot invoke this capability without an agreed name.
   Concrete fix: Standardize the attribute name on the slot closure as `generate_fn._otr_inspect_fit` (matching established `_otr_bind_schema` convention at [nodes/OTR_LedgerScriptWriter.py:741](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L741)), with signature `(messages: list[dict], **kwargs) -> dict[str, Any]` returning `{"fits": bool, "prompt_tokens": int, "capacity": int, "provenance": str}`.

5. [Section C / Section 6] Partial cleaner replacement schema definition and validation.
   Defect: Section C describes "replacements keyed by approved span IDs", which risks an implementor selecting `dict[str, str]` (where duplicate keys are silently overwritten by Python's `json.loads`). Item 6 correctly specifies a list of pairs, but the schema definition is not formalized.
   Concrete fix: Define the Pydantic schema explicitly as:
   ```python
   class SpanReplacement(BaseModel):
       span_id: str
       replacement: str
   class PartialRepairResult(BaseModel):
       replacements: list[SpanReplacement]
   ```
   Enforce in post-validation that: (a) `{r.span_id for r in replacements} == set(approved_span_ids)`, (b) `len(replacements) == len({r.span_id for r in replacements})` (no duplicate span IDs), and (c) `clean_spoken_text(spliced_text).strip() != ""` after splicing.

6. [Section 8 / Section E] Positional indexing breakage in `_NormalizedPrompt`.
   Defect: Section 8 requires adding `base_prompt_hash` to `_NormalizedPrompt` while preserving existing attribute access and indexing. In [nodes/otr_image_gen_dispatcher.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_image_gen_dispatcher.py#L954-L969), `_NormalizedPrompt` is a 6-field `NamedTuple`. Placing `base_prompt_hash` anywhere except the final slot shifts index 5 (`prompt_hash`) and breaks 6-argument instantiations.
   Concrete fix: Specify that `base_prompt_hash: str = ""` is appended as the 7th field of `_NormalizedPrompt(text, styled, pre_banana_hash, banana_result, banana_receipt, prompt_hash, base_prompt_hash="")`.

7. [Section B / line 148] `MIN_OUTPUT_TOKENS = 1` breaks existing unit test in `test_generation_budget.py`.
   Defect: Lowering `MIN_OUTPUT_TOKENS` from 64 to 1 in [nodes/_otr_generation_budget.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_generation_budget.py#L12) causes `test_context_budget_fails_when_prompt_leaves_no_viable_artifact_room` in [tests/test_generation_budget.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_generation_budget.py#L47-L50) (`context_cap=8192, prompt_tokens=8150`) to pass without raising `GenerationContextOverflowError` because $8192 - 8150 = 42 \ge 1$.
   Concrete fix: Explicitly schedule an update to `tests/test_generation_budget.py`: test failure with `prompt_tokens=8192` (0 available tokens < 1), and preserve a separate test case verifying explicit `min_output_tokens=64` raises on 42 tokens.

8. [Section 1 / Section A] Stale assertions in `test_constrained_generate.py`.
   Defect: [tests/test_constrained_generate.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_constrained_generate.py#L134-L135) explicitly asserts `assert fn.json_schema_parser is not None` and `assert fn.prefix_allowed_tokens_fn is not None`, and line 160 asserts `len(internal["by_schema"]) == 2`. Removing these stateful closure attributes and the `by_schema` cache will cause immediate test failure on the build gate.
   Concrete fix: Explicitly include [tests/test_constrained_generate.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_constrained_generate.py) in the Phase A test updates, removing assertions on `fn.json_schema_parser`, `fn.prefix_allowed_tokens_fn`, and `by_schema` caching, while asserting `fn.schema_model` retention.

SHOULD-FIX:
1. [Section 7 / Section E] Formalize `CleanTransaction.abort` receipt schema.
   Defect: Section 7 introduces `CleanTransaction.abort(reason, evidence=None)` but does not pin its return dictionary schema.
   Concrete fix: Specify the abort receipt structure:
   `{"outcome": "aborted_pre_clean", "authorized_stages": list(CLEAN_WINDOW_STAGES), "reason": reason, "evidence": evidence, "attempted_receipts": attempted, "restored_state_proved": bool}`.
   Ensure `self._terminal_receipt` is stored and returned on subsequent calls to `reconcile()` or `abort()`.

2. [Section A / lines 101-103] String coercion in `_inherit_generation_contract`.
   Defect: In [nodes/_otr_structured_call.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py#L545-L556), `_inherit_generation_contract` ignores `target` when it is a string. If a repair factory returns a string, wrapping it in a plain `list` later strips `ProviderCapacityMessages` and `_otr_unbounded_json_field`.
   Concrete fix: If `isinstance(target, str)`, coerce to `[{"role": "user", "content": target}]`, then apply `copy.copy(source)` and re-assign slice contents so custom container type and attributes are inherited.

3. [Section D / lines 244-247] Candidate path traversal failure handling in review schema.
   Defect: The conflict schema uses `candidate_path: list[str | int]`. If the model emits a path that does not exist in the candidate dict, naive traversal will raise `KeyError` or `IndexError`.
   Concrete fix: Specify that traversal of `candidate_path` uses safe lookup; if the path fails to resolve, treat the conflict as ungrounded and classify the finding as `"uncertain"`.

OPTIONAL / NICE-TO-HAVE:
- Add a timing delta and phase label to the platform-aware MPS telemetry observer on macOS in Section F to differentiate inference duration from memory stabilization delays.
- Add a helper `to_summary_dict()` on `SourceDocument` to produce standardized logging strings without exposing body text.

CUT THESE:
1. [Section E / lines 313-315] Model-based semantic re-verification of deterministic visual fallbacks in `OTR_ImageGenDispatcher`.
   Why safe to cut: Dispatcher is an image generation queue and cache manager, not an LLM evaluation node. Forcing model loads or re-verification during image dispatch introduces VRAM thrashing and latency. Labelling unverified fallback prompts as `source_verified=False` in ledger metadata is completely sufficient.
2. [Section A / lines 90-92] `_otr_lmfe_constraint_cache["by_schema"]` dictionary.
   Why safe to cut: Scanning tokenizer vocabulary (`TokenEnforcerTokenizerData`) accounts for >99% of LMFE initialization cost. Constructing `JsonSchemaParser` per call takes <1ms and guarantees zero state leakage across calls. Cutting the `by_schema` cache simplifies the code and eliminates stale state bugs.

VERIFY-AT-BUILD CHECKLIST:
- [ ] LMFE 0.11.3 array length limit: Verify that constructing `JsonSchemaParser(schema, config=CharacterLevelParserConfig(max_json_array_length=0))` and setting `parser.config.max_json_array_length = 0` allows >20 items in both root and nested arrays without raising or truncating.
- [ ] Mac M4/16GB memory footprint: Confirm that platform-aware telemetry records process RSS and MPS allocations without per-token sync, proving whether memory pressure stems from LMFE or model retention.
- [ ] Row-level cleaner receipts: Verify that `run_ledger_clean` row receipts record exact `f1_finding_spans`, model replacements, and spliced output, demonstrating that 5 model-dirty rows yield exactly 5 scoped edits without touching off-target text.
- [ ] Native HuggingFace context discovery: Verify that `_read_advertised_context` extracts `text_config.max_position_embeddings` for multimodal/text models, and that `(policy.cache_key(), normalized_pin)` invalidates properly on pin changes without mutating `model_config`.
- [ ] Pairwise source-candidate coverage: Verify that partial-window checks produce `disposition="partial"` and that quotes outside delivered spans yield `disposition="uncertain"`.

[ASSUMPTION] Marking: In MUST-FIX item 2, the specific wording of repair instructions for P0 and P3 is inferred based on domain conventions in `_otr_my_story.py` and the authoring invariants established in R2/R3.
