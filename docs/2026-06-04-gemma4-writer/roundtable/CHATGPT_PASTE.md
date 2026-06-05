PASTE EVERYTHING BELOW THE LINE INTO CHATGPT (one message). Then save its
reply as `chatgpt_review.txt` in this folder (or paste it back to me) and I'll
fold it into the synthesis.

================================================================================

ROLE: You are a senior architect + release engineer doing a FINAL, adversarial,
pre-build review of the plan below. Skeptical by default. No praise. Find what
breaks. Be specific and cite the section you mean. A separate judge will verify
every claim against the real source code, so vague criticism is worthless. If a
claim depends on code you were not shown, write "verify: <what>" instead of
asserting it.

OUTPUT (strict, plain text): VERDICT (yes / yes-with-fixes / no, one line why);
MUST-FIX BEFORE BUILD (numbered: [section] + defect + concrete smallest fix);
SHOULD-FIX (same); OPTIONAL; CUT THESE (over-engineering, with why safe to cut).
Mark [ASSUMPTION] where you infer beyond what is shown.

--------------------------------------------------------------------------------
VERIFIED CODE FACTS (already confirmed against the real OTR source):
- `nodes/_otr_style_picker.py`: the style-picker "inventor" (Pass 1) asks for
  exactly `_REQUIRED_CANDIDATE_COUNT = 5` snake_case style descriptors.
  `_parse_inventor_output` raises `ValueError` if the parsed count != 5 (also
  enforces `DESCRIPTOR_RE` = 2-5 lowercase snake_case words, and a distinctness
  rule). 3 attempts, then `StyleGenerationFailedError` -> the whole episode
  fail-closes.
- `StylePick` pydantic model: `candidates: Field(..., min_length=5, max_length=5)`
  -- so relaxing the count is NOT one line; the pydantic Field + the
  StyleGenerationFailedError docstring also assume exactly 5.
- Slot routing: `pass1_slot` default = "creative", `pass2_slot` default =
  "technical". The paired contract currently routes BOTH passes through
  `creative_fn` (dispatch to technical is staged, not wired). The inventor is a
  CREATIVE-slot pass today.
- `nodes/_otr_openrouter_backend.py`: the OpenRouter lane is provider-agnostic --
  `OPENROUTER_BASE_URL` points it at ANY OpenAI-compatible `/v1` endpoint (this is
  how it reached Ollama). It can pass `response_format` json_schema on technical
  calls (`schema_to_response_format`). `resolve_slug` uses a bound slot value
  verbatim, else falls back to `OTR_OPENROUTER_SLOT_A_DEFAULT`.
- Observed failure: gemma-4-12b returned 63 grammar-valid descriptor lines (not 5)
  on all 3 attempts. Output was well-formed (the <think> strip works); it just
  ignored the exact-count instruction. mistral-nemo obeys "exactly 5" reliably.
- Constraints: RTX 5080, 16 GB VRAM, 14.5 GB ceiling, single GPU, Windows,
  Blackwell. Offline-first / 100% local for the shipping product. Writer has two
  model slots (creative_writing_model / technical_model); rule: no new model_id
  widgets; only the writer exposes model-pick widgets.

--------------------------------------------------------------------------------
THE PLAN TO REVIEW:

GOAL: Make gemma-4-12b usable as OTR's writer (at minimum as the CREATIVE slot)
without aborting the exact-count / structured passes -- or decide it is not worth
it. Keep mistral-nemo as the safe default either way.

CANDIDATE APPROACHES (rank/harden these):

A. Constrained decoding (grammar / JSON-schema) on the exact-count passes. Force
   exactly 5 at the decoder, not via the prompt. (a) a GBNF grammar permitting
   exactly 5 descriptor lines; or (b) convert the inventor to a JSON array with a
   schema minItems=maxItems=5. OTR already sends response_format json_schema on
   technical calls, so (b) is closest -- but the inventor is line-based + creative
   today, so (b) means changing its output contract + parser + pydantic model.

B. Post-parse leniency in _parse_inventor_output (model-agnostic): when >5
   grammar-valid distinct descriptors return, take the first 5. Cheap, helps every
   model. BUT it also requires relaxing StylePick.candidates Field(min=max=5) and
   the StyleGenerationFailedError contract -- not a one-line change.

C. Slot routing: gemma on creative narrative, a count-reliable model (mistral) on
   the structured/exact-count passes. The inventor is creative today and both
   passes currently route through creative_fn, so this needs the staged
   per-pass dispatch finished -- without adding a model_id widget. VRAM: two
   resident local models likely break the 14.5 GB ceiling unless one slot is a
   served (remote-style) endpoint.

D. Prompt hardening for thinking models (few-shot the 5-line format, "output ONLY
   5 lines", lower temperature). Weakest alone; cheap to combine.

E. Null option: keep mistral-nemo, shelve gemma-4-12b (it already wins by
   completing). Only worth more effort if gemma's narrative is clearly better
   (unproven -- it never reached the dialogue stage).

RUNTIME QUESTION (important): which local runtime gives reliable constrained
decoding (GBNF or json_schema) via an OpenAI-compatible /v1 endpoint on
Blackwell/Windows, offline, within VRAM, while keeping OTR's existing lane
(OPENROUTER_BASE_URL -> any OpenAI /v1)? Compare Ollama (current; format
json_schema, GBNF-via-/v1 weak?), llama.cpp llama-server (first-class GBNF +
json_schema via /v1), LM Studio (OpenAI /v1, json_schema, llama.cpp-backed, GUI).

RECOMMENDED STARTING POSITION (attack this):
1. Ship B now as a model-agnostic safety net (take-first-5-distinct) + test.
2. Add A via a runtime with real grammar/json_schema; prefer llama.cpp
   llama-server; keep Ollama only if it proves equivalent for json_schema on /v1.
3. Use C as interim default if A slips (gemma creative + mistral technical) after
   confirming VRAM headroom (one slot served, not two resident).
4. Re-bake-off gemma's narrative only after it can complete.

OPEN QUESTIONS: Is B an acceptable softening of a deliberate strict gate, or does
it hide real failure? Where ELSE does OTR demand exact counts that gemma would
break? Is JSON+schema cleaner than GBNF given the existing seam? Does C's VRAM
math work on 16 GB with two local models, or must one slot be served? Is gemma's
likely narrative gain worth ANY of this vs just keeping mistral-nemo?
