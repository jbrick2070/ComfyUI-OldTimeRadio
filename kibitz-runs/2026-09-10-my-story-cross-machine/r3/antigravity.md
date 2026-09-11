VERDICT: build-ready as-is? no.
Critical sequencing inversions in HF root resolution and CleanTransaction abort/reconcile, unhandled NoneType comparison in decode guard, and prompt-fit prerequisite ordering break runtime execution.

MUST-FIX BEFORE BUILD:
1. [Section E / CleanTransaction] `CleanTransaction.abort` state clobbered by downstream `reconcile()` call.
   - Defect: Section E adds `CleanTransaction.abort(reason, evidence=None)` to restore the pre-clean state upon a grounded conflict. In `nodes/_otr_writer_tail.py:1384-1405`, `_clean_window.reconcile()` is unconditionally called immediately following the clean window. In `nodes/_otr_clean_transaction.py:226-254`, `_reseal()` compares `self._pre_state` with `post`. Because `abort()` already restored the ledger, `post == self._pre_state`, causing `transition is None`. `_reseal()` then treats the transaction as a successful no-op (`{"outcome": "noop"}`), stamping `transition=None` and wiping out the abort degradation receipt.
   - Concrete fix: Introduce an explicit state enum (`OPEN`, `COMMITTED`, `ABORTED`) on `CleanTransaction` in `nodes/_otr_clean_transaction.py`. When `abort(reason, evidence=None)` runs, transition state to `ABORTED`, restore pre-clean state, stamp `meta[DEGRADATION_META_KEY]`, and cache the receipt. In `reconcile()`, check `if self._state == ABORTED: return self._abort_receipt` and return immediately without calling `_reseal()`.

2. [Section B & Section D] Build order dependency: Section D adaptive windowing requires Section B prompt-fit inspection.
   - Defect: Section D mandates adaptive reading and source-window splitting based on measured capacity ("Window sizing uses the real prompt/context... If prompt fits, P0 can organize it directly..."). However, `_SlotScheduler` in `nodes/OTR_LedgerScriptWriter.py:703-745` currently returns generate closures with zero prompt-fit inspection methods. Attempting to build Section D before Section B leaves adaptive windowing with no way to measure token consumption against active chat templates, risking falling back to string character heuristics.
   - Concrete fix: Enforce strict build sequencing: implement Section B's CPU prompt-preparation helper and slot inspection capability (`generate_fn._otr_inspect_prompt_fit`) before building Section D adaptive reading. Ensure inspection runs against the target slot (`technical` for P0/organizer, `creative` for P1-P3) and returns primitive `(prompt_tokens, remaining_tokens, fits)` without moving tensors to GPU or incrementing slot call counters.

3. [Section A & `nodes/_otr_decode_guard.py`] `max_open_string_tokens=None` causes fatal `TypeError` during decoding.
   - Defect: Section A specifies `make_degeneracy_criterion(..., max_open_string_tokens: int|None=2048)` skips open-string tracking when None. In `nodes/_otr_decode_guard.py:298-335`, if `tokenizer` is passed but `max_open_string_tokens=None`, line 298 instantiates `self._tracker = OpenStringTracker()`, and line 330 evaluates `self._tracker.open_tokens >= max_open_string_tokens`. Comparing `int >= None` raises `TypeError`. While caught by the blanket `except Exception: return False` at line 353, it silently disables the entire guard loop, fails open, and wastes CPU cycles running `_text(token_id)` decoding on every single token.
   - Concrete fix: In `nodes/_otr_decode_guard.py:298`, gate tracker creation on `self._tracker = OpenStringTracker() if (tokenizer is not None and max_open_string_tokens is not None) else None`. If `max_open_string_tokens is None`, ensure `_tracker` remains None so open-string token feeding and decoding are entirely bypassed.

4. [Section B & `nodes/_otr_model_loader.py`] Initialization inversion: `validate_model_id` runs before canonical HF hub root resolution.
   - Defect: In `nodes/_otr_model_loader.py:1957`, `normalized = _otr_catalog.validate_model_id(model_id)` is invoked without passing `hub_root`. `validate_model_id` (`nodes/_otr_model_catalog.py:1692`) falls back to scanning ambient `HF_HOME / "hub"`. The canonical HF root is resolved only much later at line 2151 (`hub_root=_Path(_resolved_hf_home) / "hub"`). On systems where ambient environment variables differ from OTR pinned paths, local model discovery scans the wrong directory and fails admission before download/snapshot selection.
   - Concrete fix: In `nodes/_otr_model_loader.py`, move the canonical hub root resolution block above line 1957. Pass `hub_root=hub_root` explicitly to `_otr_catalog.validate_model_id(model_id, hub_root=hub_root)`.

5. [Section A & Section D] `story["attempts"]` data collision between author proposals and reviewer checks.
   - Defect: In `nodes/_otr_my_story.py:1127-1137`, the `finally` block inspects `story["attempts"]` where `pass_id == "treatment"` and parses `raw_output` via `parse_first_json_object` to compute `proposed_acts` and `proposed_characters`. If the treatment source reviewer (invoked from treatment's post-validator) appends its attempts into `story["attempts"]` under `pass_id="treatment"`, the `finally` block attempts to parse the reviewer's JSON schema (`{verdict: ..., conflicts: ...}`), corrupting the proposed counts. Furthermore, if a transport exception occurs, storing partial bytes in `raw_output` causes truncated JSON parsing.
   - Concrete fix: In `nodes/_otr_structured_call.py` and `_call` in `nodes/_otr_my_story.py:373-380`:
     a) Keep `raw_output` strictly as the successfully returned string (or `""` on transport raise), storing transport failure bytes in a dedicated `raw_completion` field.
     b) Reviewer attempt records must never use author pass IDs; store them under `story["source_reviews"]` or use distinct pass IDs (e.g. `treatment_review`).

6. [Section B & `nodes/_otr_model_loader.py`] Native HF cache key omits normalized context pin.
   - Defect: In `nodes/_otr_model_loader.py:2110, 2213`, cache hit checking and publication use `policy_key=_policy.cache_key()`. If an operator or workflow changes `context_cap` / context pin across runs while retaining the same model ID and runtime policy, `_try_cache_hit_locked` matches on `_policy.cache_key()` alone and returns the resident model with a stale context window.
   - Concrete fix: Snapshot `normalized_pin = _normalize_context_pin(raw_pin)` at entry to `request_slot`. In `nodes/_otr_model_loader.py`, construct the native HF cache key as `(policy.cache_key(), normalized_pin)` for both `_try_cache_hit_locked` and the `LLM_CACHE` publication dictionary.

7. [Section C & `nodes/_otr_ledger_clean.py`] Span replacement schema mismatch and line-wide whitespace destruction.
   - Defect: In `nodes/_otr_ledger_clean.py:1420-1426`, `_call_repair` uses `_RepairedLine` returning a single `text: str` with `max_length=2000`, and line 1477 applies `" ".join(str(result.text or "").split())`. This mutates untouched row spans and collapses indentation across the entire dialogue line.
   - Concrete fix: In `nodes/_otr_ledger_clean.py`, replace `_RepairedLine` with a partial repair model: `class PartialRepair(BaseModel): replacements: dict[str, str]`. Validate that `set(replacements.keys()) == set(authorized_span_ids)`. Interleave the original unmodified string slices with `replacements[span_id]`. Remove blanket `" ".join(...split())`, validating only that each individual replacement string is non-empty. Remove `max_length=2000`.

8. [Section E & `nodes/otr_image_gen_dispatcher.py`] `_NormalizedPrompt` NamedTuple positional breakage & cache-hit context blindness.
   - Defect: In `nodes/otr_image_gen_dispatcher.py:954-969`, `_NormalizedPrompt` is a `NamedTuple`. Inserting `base_prompt_hash` into arbitrary positions breaks existing tuple unpacking `(text, styled, pre_banana_hash, bres, receipt, prompt_hash) = ...`. Furthermore, in `nodes/otr_image_gen_dispatcher.py:1539-1572`, `request_cache_key` ignores the source context hash. On a cache hit, `fresh = dict(ref_row or {})` copies the old row wholesale, stamping `provenance: cache_hit` while retaining stale source/context receipts from a previous run.
   - Concrete fix:
     a) In `_NormalizedPrompt`, append `base_prompt_hash: str = ""` as the last field to preserve existing tuple indexing and unpacking compatibility.
     b) Incorporate the hash of the source/context receipt into `request_cache_key`.
     c) In the cache-hit branch (`_process_image_item`), explicitly update `fresh["source_context_receipt"]` with the current incoming receipt from `image_prompts_json["objects"]`.

SHOULD-FIX:
1. [Section D] Cyclic slot thrashing in `structured_call` post-validation review loop.
   - Defect: In `nodes/OTR_LedgerScriptWriter.py:602-646`, when `slot` switches between `creative` and `technical`, `request_slot` unloads the resident model for discrete local configurations. If an author pass (`creative_fn`) runs, its `post_validator` invokes the reviewer on `technical_fn`, forcing an eviction. If the review finds a conflict, `structured_call` triggers a repair on `creative_fn`, forcing another eviction and reload. For 3 attempts, this performs 4+ model reloads.
   - Concrete fix: In author post-validators, ALWAYS execute fast in-memory structural checks (cast uniqueness, act count, line speaker validation) FIRST. Never invoke `technical_fn` if structural checks fail. Wrap reviewer calls in `slot_scheduler.helper_context("my_story_review")` so transition accounting correctly attributes the swap.

2. [Section A & `nodes/_otr_constrained_generate.py`, `nodes/OTR_LedgerScriptWriter.py`] Open-string halt misreported as cycle repetition.
   - Defect: In `nodes/_otr_constrained_generate.py:341-354` and `nodes/OTR_LedgerScriptWriter.py:1106-1135`, when `_guard.hit` triggers on `_guard.reason == "open_string"`, log messages and `GenerationDegeneracyError` state that the model "repeated a %s-token run verbatim", and `OTR_LedgerScriptWriter.py:1135` hardcodes `open_string_tokens=None`.
   - Concrete fix: Branch the halt telemetry in both bound transports: if `_guard.reason == "open_string"`, log "open string exceeded token allowance", pass `open_string_tokens=telemetry.get("open_string_tokens")`, and set the exception reason accurately.

3. [Section A & `nodes/_otr_structured_call.py`] `_inherit_generation_contract` drops markers when repair factory returns string.
   - Defect: In `nodes/_otr_structured_call.py:545-555`, `_inherit_generation_contract` exits early via `if not isinstance(target, (list, tuple)): return target`. When a repair factory returns a string prompt, the original `ProviderCapacityMessages` subtype and its attributes (`_otr_unbounded_json_field`) are dropped.
   - Concrete fix: When `isinstance(target, str)`, coerce target into `[{"role": "user", "content": target}]` before copying `source`'s list subtype and dictionary attributes.

4. [Section D] Unbounded recursion / split stall on unbreakable source blocks.
   - Defect: Section D specifies splitting oversized source windows into smaller disjoint children. If raw input contains an unbroken string (e.g. 20,000 characters without whitespace/newlines) that cannot be split at natural boundaries, naive binary splitting could recurse infinitely or produce sub-token slices that overflow prompt envelopes.
   - Concrete fix: Enforce an explicit minimum window size (e.g. 64 characters) and recursion depth limit (e.g. 8). If a window cannot fit at minimum size, raise `PromptContextOverflowError` rather than continuing to split.

5. [Section D] Candidate path traversal failure mode handling.
   - Defect: In Section D review schema, `conflicts` returns `candidate_path: [str|int, ...]`. If the model emits a hallucinated key or out-of-bounds index (e.g. `["acts", 0, "dialogue"]` instead of `"lines"`), unguarded dictionary/list traversal raises `KeyError` or `IndexError`.
   - Concrete fix: Wrap path traversal in a safe navigator: verify each key exists in dict or index is within list range. If traversal fails or quotes do not match verbatim substrings, classify as ungrounded (`verdict="uncertain"`), record in receipt, and do not trigger author rejection.

6. [Section C & `nodes/_otr_ledger_clean.py`] Context verification omission of raw source in cleaner prompts.
   - Defect: In `nodes/_otr_ledger_clean.py:1439-1454`, `verify_context_landed` verifies that a fixed set of context keys landed in the prompt. If raw source is injected into `_repair_prompt` without updating `verify_context_landed`, the verification will fail and log `THE REPAIR IS PARTLY BLIND`.
   - Concrete fix: Update the context verification schema in `nodes/_otr_ledger_clean.py` to include `raw_source` when raw source context is supplied.

7. [Section B & `nodes/_otr_loader_backends.py`, `nodes/_otr_generation_budget.py`] Hardcoded model context limit and output token floor.
   - Defect: `nodes/_otr_loader_backends.py:79-105` rejects any catalog row with `context_window < 8192` (`HARD_VRAM_CONTEXT_LIMIT`). `nodes/_otr_generation_budget.py:12` hardcodes `MIN_OUTPUT_TOKENS = 64`.
   - Concrete fix: Remove the hard rejection in `check_context_window` for valid models whose context fits the measured prompt. Update `MIN_OUTPUT_TOKENS` default to 1 while preserving explicit caller floor overrides.

OPTIONAL / NICE-TO-HAVE:
1. [Section F] Memory instrumentation overhead: Ensure MPS observer queries `torch.mps.current_allocated_memory()` and driver allocations without issuing `torch.mps.synchronize()`, avoiding GPU pipeline bubbles.
2. [Section A] Pre-compile LMFE regex/schema rules where applicable, ensuring `parser.config.max_json_array_length = 0` is applied directly after `build_transformers_prefix_allowed_tokens_fn`.

CUT THESE (over-engineering):
1. Standing relationship/fact ontology in Section D: Cut any requirement for fixed semantic taxonomies or relational graphs in P0/organizer. The document explicitly specifies model-chosen headings, notes, and raw quote citations; rigid graph ontologies introduce schema brittleness.
2. JSONPath evaluation engine: Cut external JSONPath parsing libraries for `candidate_path`. Plain Python `dict`/`list` iteration with bounds checking is stdlib-only, deterministic, and avoids external dependencies.
3. Secondary visual LLM retry loops in Section E: Cut any additional LLM generation loops for image prompt contradiction repair. Reusing `max_reseed` with contradiction feedback is sufficient; additional loops risk VRAM thrashing between visual models and LLMs.
4. Compulsory summarization for short inputs in Section D: Safe to cut because P0 directly processes inputs that fit in the prompt context, avoiding unnecessary LLM passes.

[ASSUMPTION] marked where inferences were made:
- [ASSUMPTION] `_NormalizedPrompt` is unpacked as a tuple in downstream test suites or consumer modules, making positional field additions breaking.
- [ASSUMPTION] The technical and creative model slots may be configured with distinct local models on a 16GB VRAM system, triggering physical unloads during alternating slot calls.
