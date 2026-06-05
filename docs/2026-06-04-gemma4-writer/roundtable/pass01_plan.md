# Hardened plan: make gemma-4-12b usable as OTR's writer (pass 01)

**Date:** 2026-06-04 | Grounded against the real code. Panel so far: Grok-4.3
(complete), Gemini-3.1-pro (partial, hit token cap). Reasoning panel
(Opus/GPT/Sonnet/DeepSeek) pending (manual ChatGPT route + re-run). Claude judged.

## What the grounding changed vs pass00

Three pass00 ideas are bigger than they looked (verified in `_otr_style_picker.py`):

- **"take-first-5" (B) is not a one-line parser tweak.** `StylePick.candidates`
  is `Field(..., min_length=5, max_length=5)` (line 126) AND there is a
  distinctness validator AND `StyleGenerationFailedError` documents "fewer than 5"
  as a hard failure. Relaxing the count means touching the parser, the pydantic
  Field, and the contract docstring together, with a test.
- **Re-tagging the inventor (C) is staged, not available.** `pass1_slot` defaults
  to "creative" (line 138) and the paired contract currently routes BOTH passes
  through `creative_fn` (technical dispatch is staged at "B2", not wired). Routing
  the inventor to a count-reliable model needs that dispatch finished -- WITHOUT a
  new model_id widget (rule 6).
- **JSON+schema (A-b) changes the inventor's output contract.** Today it is
  line-based snake_case; `_INVENTOR_USER_TEMPLATE`, `_parse_inventor_output`, and
  `StylePick` all assume that. Switching to a JSON array means migrating all three.

## Recommended path (smallest change that actually fixes it)

**1. Primary fix -- constrain the decode, don't trust the prompt (Approach A).**
The root cause is that gemma won't honor "exactly 5" from a prompt. Force it at the
decoder via the runtime's structured-output support, reached through OTR's EXISTING
lane (`OPENROUTER_BASE_URL` already points the OpenRouter backend at any
OpenAI-compatible `/v1`; `schema_to_response_format` already builds json_schema
payloads for technical calls). Two sub-options:
  - **A-GBNF:** a grammar that permits exactly five `DESCRIPTOR_RE` lines. Keeps
    the line-based contract intact -- no parser/pydantic migration. Needs a runtime
    that accepts a GBNF grammar over `/v1`.
  - **A-JSON:** inventor emits a 5-element JSON array under a schema
    (minItems=maxItems=5); migrate template + parser + `StylePick`. More work but
    rides the existing json_schema seam.
  Prefer **A-GBNF** -- it closes the defect with the least contract churn.

**2. Safety net -- leniency (Approach B), scoped correctly.** Independently, make
`_parse_inventor_output` take the first 5 valid+distinct when more are returned,
and relax `StylePick`'s Field + docstring to `>=5`-then-truncate. This hardens
EVERY model, not just gemma, and is the fallback if A's runtime support is weaker
than hoped. Ship with a unit test. (Open: is softening a deliberate strict gate
acceptable? Decide explicitly.)

**3. Defer slot-routing (C)** unless 1+2 fail: it needs the staged technical
dispatch finished and has a VRAM problem (two resident local models break 14.5 GB;
one slot must be a served endpoint, not co-resident).

**4. Audit other exact-count passes.** Before adopting B as "model-agnostic", grep
for the other places OTR demands an exact count/shape (the chooser, cast contract,
validators) so we know gemma's blast radius -- don't leave it as an open question.

**5. Decision gate.** Only pursue any of this if gemma's NARRATIVE is worth it.
It never reached the dialogue stage, so re-bake-off gemma vs mistral on the 7-axis
rubric AFTER it can complete. If it doesn't beat mistral, stop -- mistral stays.

## Runtime: Ollama vs llama.cpp vs LM Studio

All three expose an OpenAI `/v1` endpoint, so all three drop into OTR's lane via
`OPENROUTER_BASE_URL` with **no new backend** (verify the backend's OpenRouter
headers don't trip a local server -- extra headers are normally ignored). The
choice is purely *which enforces the constraint best*:

- **llama.cpp `llama-server` -- recommended for the structured passes.**
  First-class **GBNF grammar** and **json_schema**, both over `/v1`; fully offline,
  headless, scriptable -- matches OTR's "100% local". GBNF gives a HARD "exactly 5"
  guarantee a prompt cannot. Cost: you manage the model + launch flags yourself.
- **LM Studio -- the GUI-friendly equivalent.** OpenAI `/v1`, json_schema
  structured output, llama.cpp-backed under the hood (so the same enforcement).
  Easiest to run; less natural for a fully-headless automated pipeline.
- **Ollama (current) -- keep only if verified.** Easy model management; supports a
  `format` JSON-schema, but GBNF/strict enforcement over the `/v1` path is the
  weak spot -- confirm the installed version actually constrains output before
  relying on it.

**Bottom line:** use **llama.cpp llama-server** for the exact-count passes (hardest
guarantee, most local). Use **LM Studio** if you prefer a GUI -- same engine, same
enforcement, friendlier. Verify current feature support (these move fast;
knowledge here is as of mid-2025).

## CUT (per panel, confirmed)

- A full runtime bake-off in the shipped plan -- the lane already takes any
  OpenAI `/v1`; the only thing that matters is "supports GBNF or json_schema over
  /v1". One sentence, not a project.
- The "re-bake-off narrative" step does NOT belong in THIS plan's build scope; it
  is the decision gate (#5), run only after completion is fixed.

## Verify-at-build / open

- Does OTR's OpenRouter backend send headers a local llama.cpp/LM Studio server
  rejects? (verify)
- Exact list of other exact-count/strict passes gemma would also break. (grep)
- Minimum Ollama version that enforces `format` json_schema over `/v1`, if Ollama
  is kept. (verify)
- Is relaxing the strict 5-count gate (B) acceptable product-wise? (Jeffrey)
