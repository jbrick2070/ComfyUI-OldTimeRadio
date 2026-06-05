# Plan to harden: make gemma-4-12b usable as OTR's writer

**Date:** 2026-06-04 | **Project:** ComfyUI-OldTimeRadio (OTR) v2.0-alpha
**Reviewers:** frontier panel (critique only). Claude is the judge/grounder.

## Context

OTR's writer (`OTR_LedgerScriptWriter`) drives an entire radio-drama script
through a chain of LLM passes. Each LLM call is tagged **creative** or
**technical** and routed to one of two model slots (`creative_writing_model`,
`technical_model`). Today's out-of-box default for both is local
`mistralai/Mistral-Nemo-Instruct-2407` (4-bit NF4, ~7.7 GB VRAM).

We bake-off-tested **gemma-4-12b** (GGUF `hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M`)
as a candidate writer, served by **Ollama** on `:11434` and reached through OTR's
existing OpenRouter lane pointed at Ollama's OpenAI-compatible endpoint
(`OPENROUTER_BASE_URL=http://localhost:11434/v1`). A `<think>...</think>` /
harmony-channel strip was added to the lane's response parser and is proven to
work (gemma cleared the early structured passes NewsCuration / NewsCurationDeep).

## The blocker (grounded)

The run **aborts at the style-picker "inventor" pass** (`nodes/_otr_style_picker.py`):

- `_build_inventor_user_prompt` asks the model for exactly `_REQUIRED_CANDIDATE_COUNT`
  (= **5**) style descriptors.
- `_parse_inventor_output` strips list decorations, lowercases, then enforces:
  (1) every line matches `DESCRIPTOR_RE` (2-5 lowercase snake_case words),
  (2) **exactly 5** lines, (3) distinctness (max 1 shared root word per pair).
- gemma-4-12b returned **63** grammar-valid descriptor lines -> count check fails
  on all 3 attempts -> `StyleGenerationFailedError` -> the whole episode fail-closes.

So gemma's output is *well-formed* (the strip works; lines pass the grammar); it
simply **ignores the exact-count instruction** and over-generates. Mistral-nemo
obeys "exactly 5" reliably and completes. This is an instruction-compliance gap,
not a transport/parsing bug, and it likely recurs on every OTR pass that demands
an exact count or a strict shape.

## Invariants the fix MUST NOT break

1. **VRAM ceiling 14.5 GB** (RTX 5080, 16 GB, single GPU, Blackwell, Windows).
2. **Offline-first / 100% local / open-source.** No paid cloud, no API keys for
   the shipping product. (OpenRouter lane is opt-in/default-off; a local
   OpenAI-compatible server is fine.)
3. **Audio is king / safe-for-work** (not touched by this change).
4. **Two-slot model routing (CLAUDE.md rule 6):** every LLM call is creative or
   technical; only the writer exposes model-pick widgets; consumers receive the
   model id via a STRING socket. No new `model_id` widgets.
5. **The OpenRouter lane already accepts any OpenAI-compatible base URL** and can
   pass `response_format` (json_schema) on technical calls
   (`schema_to_response_format` in `nodes/_otr_openrouter_backend.py`).

## Goal

Make gemma-4-12b usable as the writer (at minimum as the **creative** slot)
without aborting the exact-count / structured passes -- or decide it is not worth
it. Keep mistral-nemo as the safe default either way.

## Candidate approaches (to be hardened/ranked by the panel)

**A. Constrained decoding (grammar / JSON-schema) on the exact-count passes.**
Force the model to emit exactly 5 items at the decoder level rather than trusting
the prompt. Two shapes: (a) a GBNF grammar that permits exactly 5 descriptor
lines; (b) convert the inventor to emit a JSON array and enforce a schema with
`minItems=maxItems=5`. OTR's lane already sends `response_format` json_schema on
technical calls, so (b) is closest to the existing seam -- but the inventor today
is a line-based **creative** pass, so this means re-tagging it and/or changing its
output contract. Runtime-dependent (see runtime question).

**B. Post-parse leniency in `_parse_inventor_output` (OTR-side, model-agnostic).**
When more than 5 grammar-valid, sufficiently-distinct descriptors come back, take
the first 5 that satisfy the distinctness rule instead of rejecting. Cheap, helps
*every* model, no runtime dependency. Risk: masks a model that is wildly
non-compliant; changes a writer invariant (needs its own test + review). Could be
the safety net under A.

**C. Slot routing: gemma on narrative, a count-reliable model on structured passes.**
Keep gemma on `creative_writing_model` (where a bigger model may lift prose) and
put mistral-nemo (or a constrained endpoint) on `technical_model` + the exact-count
passes. Tension: the inventor is currently tagged **creative**, so this needs the
inventor (and peers) re-tagged technical, or a finer per-pass routing knob.
VRAM tension: two resident local models may break the 14.5 GB ceiling unless one
is remote/served.

**D. Prompt hardening for thinking models.** Few-shot the exact 5-line format,
add "output ONLY the 5 lines, no preamble, no extra items", lower temperature.
Least reliable on its own; cheap to combine with A/B.

**E. Null option: keep mistral-nemo, shelve gemma-4-12b.** It already wins by
completing. Only worth more effort if gemma's *narrative* is meaningfully better
(unproven -- it never reached the dialogue stage).

## Runtime question (Jeffrey's): Ollama vs llama.cpp vs LM Studio

The fix for A hinges on which local runtime gives **reliable constrained decoding
through an OpenAI-compatible `/v1/chat/completions` endpoint** on Blackwell /
Windows, offline, within the VRAM budget, while preserving the existing OTR lane
(`OPENROUTER_BASE_URL` -> any OpenAI endpoint):

- **Ollama** (current): easy model mgmt; supports `format` JSON-schema structured
  output; GBNF grammar exposure via the OpenAI `/v1` path is the weak spot to verify.
- **llama.cpp (`llama-server`)**: first-class **GBNF grammar** AND `json_schema`,
  both reachable via its OpenAI-compatible server; fully offline/scriptable; the
  most direct control. Cost: manual model/launch management.
- **LM Studio**: OpenAI-compatible server, structured-output (json_schema) support,
  llama.cpp-backed under the hood; GUI-managed. Friendlier; less scriptable/headless.

Panel: rank these for THIS use (strict exact-count constrained decoding via /v1,
Blackwell+Windows, offline, 14.5 GB, must keep working with OTR's lane) and flag
any feature claims that need version-checking (runtimes move fast).

## Recommended starting position (attack this)

1. Ship **B** now as a model-agnostic safety net (take-first-5-distinct) with a
   test -- unblocks gemma AND hardens mistral, no infra change.
2. Add **A via a runtime with real grammar/json_schema**: prefer **llama.cpp
   `llama-server`** for the structured/exact-count passes (GBNF gives a hard
   guarantee the prompt cannot); keep Ollama only if it proves equivalent for
   json_schema via `/v1`. LM Studio if Jeffrey wants a GUI over headless.
3. Use **C** as the interim default if A slips: gemma creative + mistral technical,
   after confirming VRAM headroom (one model served remote-style, not two resident).
4. Re-bake-off gemma's *narrative* only after it can complete, to decide if it
   beats mistral on the 7-axis rubric at all.

## Open questions for the panel

- Is B (take-first-N) an acceptable softening of a deliberate strict gate, or does
  it hide real model failure? Where else does OTR demand exact counts that gemma
  would also break?
- For A, is re-contracting the inventor to JSON+schema cleaner than a GBNF line
  grammar, given OTR already has the json_schema seam?
- Does C's VRAM math work on a 16 GB Blackwell card with two local models, or must
  exactly one slot be a served endpoint?
- Is gemma-4-12b's likely narrative gain worth ANY of this vs just keeping
  mistral-nemo?
