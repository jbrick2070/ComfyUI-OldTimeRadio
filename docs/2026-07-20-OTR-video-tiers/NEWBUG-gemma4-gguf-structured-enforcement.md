# NEWBUG: gemma-4 fails the structured P0 pass on the GGUF-native lane (2026-07-20)

Static + live finding surfaced while proving the video-tiers `ltx_8gb` route with a
"smallest local LLM" writer. STATIC-grounded against the real files; live-observed
in an 8GB LTX episode leg. NOT a PBUG yet (see "Status"). Upstream of the video tier
(ltx_8gb is proven independently: C0 in-process smoke render + live in the director menu).

## Symptom (live 2026-07-20)
Full 8GB LTX episode via a gemma-4 GGUF writer dies at the FIRST writer node:
```
OTR_LedgerScriptWriter -> [OTR_StructuredCall] 'scifi_codex:P0' failed after
3 attempt(s); last error -> JSONDecodeError
```
Reproduces on gemma-4-E2B (Q4_K_M, 3.2GB), gemma-4-E4B (Q4_K_M, 5GB), AND
gemma-4-12b (Q4_K_M, 6.6GB) -- every size. Failure kind is **JSONDecodeError**
(raw text unparseable at char 0), NOT a pydantic ValidationError -> the output was
never constrained to JSON. Cross-check confirmed this is NOT the truncation class
(BUG-11.50 / the remote ctx=8192 lie) -- a truncation bug scales with context; this
hits every size identically.

## Two structured-output enforcement paths (chosen by lane)
`nodes/_otr_constrained_generate.py::make_constrained_generate_fn`:
- **Transformers lane** (HF safetensors, e.g. Mistral-Nemo NF4): lm-format-enforcer
  `JsonSchemaParser` + `prefix_allowed_tokens_fn` into `model.generate()`
  (:262-269, :309). HARD token-level constraint -> always valid JSON. WORKS.
- **GGUF-native lane** (llama-cpp; every gemma-4 GGUF): :230-238 maps the pydantic
  schema -> OpenRouter-style `response_format`
  (`_otr_openrouter_backend.schema_to_response_format` => `{"type":"json_schema",...}`)
  -> `_otr_gguf_backend._llamacpp_response_format` (:898-911) rewrites it to
  llama-cpp's `{"type":"json_object","schema":schema}` -> passed to
  `llm.create_chat_completion(response_format=...)` (:1219-1223). Enforcement is
  DELEGATED to llama-cpp-python building a GBNF grammar from that schema.

## Root-cause candidates (both live on the gguf path)
1. **Grammar not biting.** JSONDecodeError proves the output is unconstrained free
   text -- almost certainly a ```json markdown fence or a lead-in sentence /
   `<think>`. Whether `create_chat_completion` honors
   `{"type":"json_object","schema":...}` as a real GBNF grammar is version- and
   schema-complexity dependent. The soft text contract (`schema_shape_instruction`,
   _otr_structured_call.py:406-425: "Return exactly one JSON object, no Markdown...")
   is the ONLY thing in force, and gemma-4 ignores it.
2. **No salvage strip for gemma rows.** The leading-`<think>`/preamble strip
   (`_strip_leading_think_envelope`, _otr_gguf_backend.py:640) is wired ONLY for
   `think_policy == "qwen3_no_think"` (:264 on the Qwen row; :1190, :1235-1243) and
   is gated to "when no response_format is in force" (:98-99). The gemma-4-12b row is
   `think_policy="none"` (:241) -> NOTHING is stripped, and there is NO markdown-fence
   (```json) strip anywhere on the gguf path -> a fenced-but-valid JSON object still
   dies in json.loads().

## Grounded confirmations (checked against the real files)
- gemma-4-12b GGUF row `think_policy="none"` (`_otr_gguf_backend.py:241`) -> no strip.
- The `"(baseline, PASS)"` comment on that row (:223) refers to `vram_fit_tier="PASS"`
  (:236) -- a VRAM-tier label, NOT a structured-output pass. No contradiction with the
  live P0 death; gemma-4-12b was never proven on the structured writer.
- **The deciding data is ALREADY logged:** `structured_call` logs the sanitized raw
  head on every failed attempt (`_otr_structured_call.py:1026-1030` via `_raw_head`,
  :756). Probe = re-run the leg + `grep "raw head"` in the server log. NO code change
  needed to disambiguate candidate 1 vs 2.
- **A proven small GGUF writer already exists: Qwen3-8B.** The
  `unsloth/Qwen3-8B-GGUF` row (:243-265) is `think_policy="qwen3_no_think"` (gets the
  strip), context 8192, and was PINNED at the 2026-07-16 live bake-off
  ("3x RESULT SUCCESS + obs asset, both writer slots Qwen"). On disk:
  `Qwen3-8B-Q4_K_M.gguf` 5027784512 bytes (4.68GB) -- SMALLER than gemma-4-12b (6.6GB).
  So "smallest local LLM that works" = Qwen3-8B, not any gemma-4.

## Prior-art / policy
- `project_otr_writer_bakeoff` ("gemma-4 rejected; mistral-nemo stays DEFAULT_LLM; no
  gemma-4 retest without GBNF inventor") + `project_otr_ollama_gbnf_hardening`
  ("/v1 takes no raw GBNF") flag exactly this territory.
- BUT `feedback_no_model_gating_per_slot` (2026-07-14) supersedes "gemma rejected":
  every slot is model-agnostic; a user may pick gemma-4-12b GGUF. So this is a real
  bug to FIX (candidate 2), not a config to wave off.
- Bible kin (fix-pattern analogue, not a match): BUG-11.11/11.18/11.20/11.47 JSON
  resilience (`_strip_json_comments` + tiered `_extract_json`) -- Director path, strips
  JS `//` comments, not ```json fences / `<think>` on the gguf structured-call parse.

## Deciding test (cheap; data already logged)
Re-run one gemma-4 P0 leg, grep the server log for `raw head:`:
- Head is ```json...``` or `<preamble>{...}` -> candidate 2: generalize the strip to
  ALL gguf rows (extend the 11.11 pattern to ```json + `<think>`, ungate from
  qwen3-only + from no-rf). Cheap; makes gemma-4 parse.
- Head is prose / no JSON object at all -> candidate 1: llama-cpp isn't enforcing the
  schema grammar -> needs a real GBNF inventor, or route gemma-4 to the transformers
  lane. Candidate 2 alone won't save it.

## Immediate unblock (independent of the fix)
Set the writer to **Qwen3-8B** (`unsloth/Qwen3-8B-GGUF`, on disk, proven, smaller) --
the video-tiers `ltx_8gb` leg would then run end-to-end. My `otr_8gb_ltx` preset
inherited gemma-4-12b from the `8gb_lite` base; that is the wrong writer for a green leg.

## Status
NEW class (gguf structured-enforcement) -- no PBUG/Bible rule covers it. PBUG-eligible
the moment gemma-4 GGUF is affirmed in-scope as a writer AND a live proof is captured
(the raw-head probe). Suggested PBUG stub id: PBUG-20260720 (fill after the head print).
```
