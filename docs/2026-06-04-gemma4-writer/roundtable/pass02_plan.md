# Final plan: gemma-4-12b as an OPT-IN OTR writer lane (pass 02, converged)

**Date:** 2026-06-04 | Panel: Grok-4.3 + ChatGPT (manual) + Gemini-3.1-pro
(partial). Judge: Claude, grounded against the real code + HF. Converged.

**Verdict: yes-with-fixes.** Add gemma-4-12b as an **opt-in** writer lane; keep
**mistral-nemo the shipping default**. Both writer slots in the canonical workflow
stay `mistralai/Mistral-Nemo-Instruct-2407`.

## Framing (the key correction)

Fitting gemma in 16 GB does NOT fix the bug. The failure was never memory -- it was
gemma returning **63** valid style descriptors instead of exactly **5**. So:

- **A + B fix the exact-count bug** (the blocker).
- **F (Unsloth GGUF + llama-server) fixes deployability** (running it at all on 16 GB).
- **E keeps mistral the default** until gemma clearly writes better.

## Build order (do in this sequence)

**1. Keep mistral-nemo default.** No workflow flip. (`DEFAULT_LLM` already mistral.)

**2. B -- parser safety net, with the contract KEPT exactly 5.**
Do NOT relax `StylePick.candidates` (it stays `Field(min=max=5)`). Change only
`_parse_inventor_output` in `nodes/_otr_style_picker.py`:
```
collect valid snake_case descriptors (DESCRIPTOR_RE)
de-dupe deterministically (respect the distinctness rule)
if >= 5: take the first 5      # survives overgeneration
if  < 5: fail (StyleGenerationFailedError)
return exactly 5 -> StylePick(candidates=5) still satisfies min=max=5
```
This makes gemma's overgeneration survivable WITHOUT weakening the public
contract. Ship with a unit test. (Supersedes pass01, which wrongly relaxed the
Field -- ChatGPT's correction, verified: the parser builds the list the pydantic
model consumes, so truncating upstream keeps the model at 5.)

**3. Telemetry.** Stamp `valid_count`, `distinct_count`, `truncated_count`,
`model_slug` on the style pass (ledger meta) so over/under-generation per model is
visible instead of just a hard abort.

**4. F -- Unsloth GGUF lane via llama-server (opt-in, no Comfy changes).**
Serve `unsloth/gemma-4-12b-it-GGUF` (Apache-2.0, verified on HF) through
`llama-server` as a local OpenAI-compatible server and reach it through OTR's
EXISTING lane (`OPENROUTER_BASE_URL -> http://localhost:<port>/v1`). Start quant:
`UD-Q4_K_XL` (confirm the exact tag in the repo's file list; `Q4_K_M` is what
Ollama already has).
```
llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL --port <p>
```
Rules: do NOT load gemma inside Comfy (transformers path is the rejected BUG-306
route); do NOT add a model_id widget; run gemma ONLY for the writer/style passes,
then explicitly unload/kill the server BEFORE the FLUX/HuMo/LTX video branch
(llama-server supports idle-sleep/unload). One resident LLM at a time -- never
gemma + mistral + video co-resident on 16 GB.

**5. Disable thinking for the structured/exact-count passes.** Unsloth's Gemma 4
docs expose `--chat-template-kwargs '{"enable_thinking":false}'` (PowerShell
escaping caveat). Turn thinking OFF for the style-picker/exact-count tests -- it
directly attacks the overgeneration and makes the `<think>` strip redundant there.

**6. A -- constrain the decode AT the failing pass (only if B+thinking-off
isn't enough).** The inventor is a CREATIVE-slot pass and both passes currently
route through `creative_fn`, so schema/grammar enforcement must be wired to the
inventor call itself, not just the technical path. Prefer **GBNF** (a grammar for
exactly five `DESCRIPTOR_RE` lines) -- it keeps the line-based contract, no
template/parser/pydantic migration. llama-server documents both `grammar` and
`json_schema` over `/v1`. JSON+schema is the heavier alternative (migrate template
+ parser + `StylePick`).

**7. Conformance harness (tiny -- run before declaring the lane "supported").**
```
1. prompt asks for exactly 5 snake_case descriptors -> model must not emit 63
2. duplicate-descriptor case is caught
3. CamelCase / spaces / punctuation rejected
4. GBNF or json_schema path works (or is marked unsupported)
5. gemma server unloads before the FLUX/HuMo/LTX branch
6. audit: grep every OTHER exact-count/shape gate (chooser, cast contract,
   validators) -- gemma may pass style-picker then break the next strict gate
```

**8. Only then test gemma's narrative** vs mistral on the 7-axis rubric.
**9. If gemma does not clearly beat mistral, stop.** Mistral stays.

## Deferred / Cut

- **Defer C** (gemma-creative + mistral-technical slot routing): the technical
  dispatch is staged, not wired, and two resident local models break 14.5 GB.
  Later architecture, not the first build.
- **Cut:** in-Comfy transformers gemma; gemma-as-default; two resident LLMs during
  video; prompt-only "fix" (already failed 3x); the claim that GGUF fixes the bug;
  a full Ollama-vs-llama.cpp-vs-LM-Studio bake-off (the lane takes any OpenAI
  `/v1`; only "supports GBNF or json_schema over /v1" matters).

## Runtime pick (Jeffrey's question, settled)

**llama.cpp `llama-server`** for the structured passes: first-class GBNF +
json_schema over `/v1`, plus idle-sleep/unload (needed for "unload before video"),
fully offline/headless. **LM Studio** = the same llama.cpp engine with a GUI if you
prefer one (json_schema via `/v1`); fine, just less headless. Keep **Ollama** only
if you verify its `/v1` actually enforces the schema. All three reach OTR through
the existing `OPENROUTER_BASE_URL` lane -- no new backend (verify the lane's
OpenRouter headers don't trip a local server; normally ignored).

## Verify-at-build
- Exact Unsloth quant tag (`UD-Q4_K_XL`) in the repo file list.
- llama-server `enable_thinking:false` + GBNF actually yield exactly 5 from gemma.
- The OpenRouter-lane header check against a local llama-server.
- The full list of other exact-count gates (harness step 6).
