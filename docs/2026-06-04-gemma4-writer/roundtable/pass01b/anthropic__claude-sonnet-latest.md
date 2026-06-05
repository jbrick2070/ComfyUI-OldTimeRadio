<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: no. The stated root cause is incomplete (a token-floor override in the backend is the proximate enabler of the 63-line output, not just instruction non-compliance), Approach A's GBNF path as described cannot work through the existing OTR lane, and the primary fix (B) is algorithmically underspecified.

---

## MUST-FIX BEFORE BUILD

**1. [Context / Blocker] Root cause analysis is incomplete — the OpenRouter backend token floor is the proximate cause of 63-line output.**

`_run_inventor` calls `generate_fn(messages, temperature=..., max_new_tokens=_INVENTOR_MAX_TOKENS)` where `_INVENTOR_MAX_TOKENS = 80`. In `OpenRouterBackend.generate()` (grounded in `_otr_openrouter_backend.py`):

```python
floor = _int_env("OPENROUTER_MIN_OUTPUT_TOKENS", DEFAULT_MIN_OUTPUT_TOKENS)  # = 1024
out_tokens = max(int(max_new_tokens or 0), floor)   # max(80, 1024) = 1024
```

The payload to Ollama sends `"max_tokens": 1024`, not 80. At ~5 tokens per short descriptor, 1024 output tokens can fit ~200 descriptors; 63 lines is plausible. The plan calls this an "instruction-compliance gap" and treats 63 lines as if the model ignored a count constraint, but the model also had ~12× its intended token budget. Fix B (take-first-5) recovers from the abort but wastes ~944 tokens of inference per inventor attempt and leaves the underlying token-floor mismatch unaddressed. **Fix:** document this interaction explicitly; consider adding a special-case in the generate_fn factory for inventor calls (or passing `stop=["\n\n"]` to exploit the docstring's "stop on blank line" intent, which is currently unimplemented — `_run_inventor` passes no `stop=` argument at all).

**2. [Candidate approach A / Runtime question] GBNF cannot transit the existing OTR lane.**

The plan states llama.cpp gives "first-class GBNF grammar AND json\_schema, both reachable via its OpenAI-compatible server" and implies the existing `OPENROUTER_BASE_URL` lane can carry GBNF. `OpenRouterBackend.generate()` builds a standard OpenAI-compatible payload with no `grammar` field. llama.cpp's `/v1/chat/completions` endpoint does not accept a `grammar` parameter — GBNF requires llama.cpp's native `/completion` endpoint with `"grammar": "..."`. Any developer who follows the plan's A(a) path under the impression the existing lane suffices will build a non-functional path and not discover it until runtime. Verify: inspect the llama.cpp server `/v1/chat/completions` OpenAPI spec for a `grammar` field. **Fix:** drop GBNF (A(a)) from scope. The json_schema path (A(b)) reaches llama.cpp's schema-constrained decoder through the standard `response_format` key, which `OpenRouterBackend.generate()` already handles (`payload["response_format"] = response_format`). Keep only A(b).

**3. [Recommended starting position step 1] Approach B greedy selection algorithm is unspecified.**

The plan says "take the first 5 that satisfy the distinctness rule." With 63 candidates the pairwise distinctness check is order-dependent and the current `_parse_inventor_output` applies a whole-set check after the count. "First 5 that satisfy the rule" is ambiguous: does each new pick only need to be distinct from already-selected candidates (greedy-add), or must you filter all 63 for grammar-valid lines, then run the pairwise check on any 5-size subset? A test cannot be written against an underspecified algorithm. **Fix:** specify the greedy algorithm precisely — e.g., "iterate `lines` in order; maintain a `selected` list; add candidate `c` iff for every `s` in `selected`, `len(roots(c) & roots(s)) <= _MAX_SHARED_ROOTS`; stop when `len(selected) == 5`; raise `ValueError` if input is exhausted before 5 are selected." The grammar-check loop must also move before the count check (currently grammar check happens after count check in `_parse_inventor_output`, so lines with invalid grammar silently counted toward the 63 before rejection).

---

## SHOULD-FIX

**4. [`pick_style` docstring, `_otr_style_picker.py`] B2 is already landed; the docstring claims B1.**

The docstring reads: "B1 routes both passes through `creative_fn` (no dispatch yet). B2 will flip pass 2 (chooser) to `technical_fn`." The implementation already calls `_run_inventor(creative_fn, ...)` and `_run_chooser(technical_fn, ...)` — B2 is done. Any plan steps that refer to "needing the inventor re-tagged" under approach C are confused by this stale description: the inventor already runs via `creative_fn`; only changing the argument passed to `_run_inventor` (from `creative_fn` to `technical_fn`) is needed for C, not a new routing knob. **Fix:** correct the docstring; note that approach C for the inventor is a 1-line change in `pick_style`.

**5. [Candidate approach C] VRAM math for dual-resident models fails the invariant.**

gemma-4-12b Q4\_K\_M ≈ 7–8 GB resident; mistral-nemo 4-bit NF4 