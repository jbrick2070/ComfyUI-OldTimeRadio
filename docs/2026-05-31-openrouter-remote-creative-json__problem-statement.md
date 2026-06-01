# Problem statement — remote frontier LLM won't emit our JSON (OpenRouter creative slot)

**Date:** 2026-05-31 · **Branch:** v2.0-alpha · **For:** round-robin (ChatGPT → Gemini → Claude synthesis)

This is written to be self-contained: a reviewer with no prior session context should be able to reason about it.

---

> ## ⚑ UPDATE 2026-05-31 — ROOT CAUSE CONFIRMED + FIXED (it was a wiring/truncation defect, not prompts)
>
> The team's hypothesis was correct. Direct API capture proved it:
> - Opus returns **valid, on-spec JSON** (the parser already strips ` ```json ` fences fine).
> - The backend clamped the remote `max_tokens` to the writer's **local grammar-era per-call budget (~200)**. That budget only works locally because lm-format-enforcer forces a *compact bare* object. A free-form remote model writes a fuller object + fence and **truncated mid-object** (`finish_reason=length`) → no complete `{...}` → the exact `"no decodable top-level JSON object ... char 0"` failure → fail-closed abort.
> - Reproduced precisely: `max_tokens=50` → truncated + "char 0"; `max_tokens=512` → clean parse.
>
> **Fix shipped:** a remote output-token floor (`DEFAULT_MIN_OUTPUT_TOKENS=1024`, env `OPENROUTER_MIN_OUTPUT_TOKENS`; `max_tokens` is a ceiling so it costs nothing on short replies) + a `finish_reason=length` warning log. After relaunch, the enabled Opus run **passed casting and the full outline** — the remote creative path works with the existing prompts.
>
> **So:** the discussion below about response_format / write-then-extract / prompt robustness is now about *quality and architecture*, **not about getting Opus to work at all** — that's solved. A secondary, non-blocking item remains: Opus is verbose and overruns schema char-caps (e.g. `character_description` 750); the repair ladder recovers but burns extra calls (BUG-263 family — raise caps or prompt for brevity). The write-then-extract recommendation (answer-statement doc) stands as the *quality* upgrade.

---

## TL;DR

We added OpenRouter as an opt-in remote LLM for the OTR writer's two slots (creative + technical). In the first **enabled** end-to-end run — creative slot = `anthropic/claude-opus-4.8` (remote), technical slot = local Mistral-Nemo — the run **aborted at casting**: a creative-slot JSON pass (`llm_write_description`) returned output with **no parseable JSON object** on all 3 attempts, so the fail-closed gate killed the run (`CastValidationLLMError`, ~2.3 min in, fractions of a cent spent).

The remote *plumbing* is proven correct (calls reach Opus, VRAM no-evict holds, cost guard accounts spend). The failure is upstream of plumbing: **the most capable model available would not return our JSON.** The question is whether the right fix is (a) API-level structured-output enforcement, (b) prompt hardening, (c) per-call routing, or some combination — and what that says about whether our prompts were ever actually doing the work.

---

## How OTR gets structured JSON from an LLM today

OTR's writer makes many LLM calls. The JSON-returning ones (cast descriptions, outline macro/phase/beat, news briefs, reviewer/critic verdicts, slot-drama fields) route through one shared helper, `structured_call(prompt, schema, slot_fn, ...)`:

1. Call `slot_fn(messages, *, temperature, max_new_tokens) -> str`.
2. `parse_first_json_object()` extracts the first `{...}` from the string; `pydantic.model_validate` checks it against the call's schema.
3. On parse/validation failure: a **bounded repair ladder** — attempt 1 (base temp) → attempt 2 (lower temp, same prompt) → attempt 3 (typed-repair prompt at temp 0.1). On exhaustion it raises `StructuredCallFailedError` (fail-closed — no bad data is ever written).

There are **two ways** a JSON call gets its `slot_fn` locally:

- **Plain generate fn** (most creative-slot JSON calls, incl. casting): the model is simply *prompted* to return JSON. Nothing mechanically forces it. Local Mistral-Nemo follows this reliably.
- **Grammar-constrained generate fn** (`make_constrained_generate_fn`, used by a few technical passes): lm-format-enforcer binds the Pydantic schema to a token-level grammar so the model **physically cannot emit invalid JSON**, regardless of prompt quality.

So locally, "return JSON" is enforced by *either* the model obeying the prompt *or* a token grammar — and for the calls that use the grammar, **the prompt's JSON instruction was never load-bearing.**

## What the remote (OpenRouter) path does

- Remote calls go through an HTTP backend that POSTs OpenAI-style chat completions to OpenRouter.
- A previous sprint added OpenRouter `response_format` **only** to the remote analog of the grammar path (`make_constrained_generate_fn`). 
- The **plain** remote creative generate fn sets **no `response_format`** — it sends the same prompt the local model gets and hopes the model returns bare JSON.
- OpenRouter/Anthropic support `response_format: {"type":"json_object"}` (forces *valid* JSON) and `{"type":"json_schema", "json_schema": {...}}` (forces *schema-valid* JSON) — the API-level equivalent of the local token grammar. We are not using either on the plain creative path.

## The failure, precisely

Live log (creative slot = claude-opus-4.8):

```
[Selector] slot=creative remote backend for openrouter:slot-a (no local VRAM; resident local model left in place, C2 no-evict)
[OpenRouter] load slot=A handle=openrouter:slot-a slug=anthropic/claude-opus-4.8 ctx=8192 (remote, 0 VRAM)
[OpenRouter] call accounted ~560 tokens
[OTR_StructuredCall] 'llm_write_description:JULIANA CROSS' attempt 1 failed: no decodable top-level JSON object found: line 1 column 1 (char 0)
... attempt 2 (structural retry 0.35) failed: no decodable top-level JSON object ...
... attempt 3 (typed repair 0.1) ...
CastValidationLLMError: Casting failed for 'JULIANA CROSS' after 3 attempts.
```

`"no decodable top-level JSON object ... char 0"` means the parser found **no `{` anywhere** — so Opus likely returned conversational prose / a preamble / an explanation, not even markdown-fenced JSON. Mistral-Nemo (smaller, local) returns bare JSON for the *same* prompt; Opus does not.

## The crux (operator's framing)

> "If the most advanced LLM isn't following our prompts, we have a problem."

Two readings, both plausible, and they imply different fixes:

1. **It's an API-usage gap, not a prompt problem.** Prompt-only JSON is known to be fragile and model-dependent; the industry-standard fix is API-level structured output (`response_format`). Locally we use a token grammar for the hard cases; remotely we should use `response_format`. Under this reading the prompts are fine and we just forgot the remote enforcement knob.

2. **It's a prompt-robustness problem the grammar was hiding.** Because the local path can lean on grammar enforcement (and on Mistral's particular obedience), our prompts may never have had to *clearly* demand "return ONLY a JSON object, no prose." A frontier model with a different default style (more explanatory, more conversational) exposes that the format instruction is weak. Under this reading, forcing `json_object` is a band-aid that masks prompts that should be made model-agnostic.

Likely it's **both**, but the round-robin should decide the weighting and the concrete plan.

**A third framing (operator's):** maybe emitting JSON is the wrong job for the creative model entirely. A frontier model is a *writer*; asking it to be a *serializer* fights its grain. The alternative is **write-then-extract** — let it write the script/character in prose, then parse that into structure with a separate, reliable (grammar-constrained or schema-locked) call. See candidate F; this may be the real answer rather than any JSON-coercion knob.

## Candidate fixes (decision space)

- **A. `response_format: json_object` for remote `structured_call` sites without a schema.** Guarantees *parseable* JSON; does not guarantee *schema-conformance* (Pydantic still validates + the repair ladder still runs). Smallest change; unblocks casting/outline immediately.
- **B. `response_format: json_schema` (strict) where the Pydantic schema is in scope.** Guarantees schema-valid JSON (closest to the local grammar). Stronger, but the schema isn't currently threaded to every call site (the plain `slot_fn` signature carries no schema), so this needs plumbing.
- **C. Prompt hardening for model-agnostic JSON.** Rewrite the JSON-expecting prompts to explicitly demand "output ONLY a single JSON object matching this shape, no prose, no markdown fences," with a schema sketch and possibly a one-shot example — so *any* model obeys without relying on grammar. Addresses reading #2; improves the local path too.
- **D. Per-call-type routing.** Keep remote on the FREE-FORM creative passes (dialogue/prose, where a frontier model buys the most quality) and force the JSON-structured passes (casting/outline/news) onto the local grammar path. The slot architecture is currently per-slot (creative vs technical), not per-call-type, so this needs new routing plumbing.
- **F. Write-then-extract (two calls) — operator's proposal, possibly the cleanest.** Stop asking the *creative* model for JSON at all. Let the remote frontier model **write free-form** (a vivid character sheet / scene / script in prose — its actual strength), then run a **separate, reliable extractor call** that turns that prose into the ledger schema. The extractor is the place to be strict: run it **locally, grammar-constrained** (lm-format-enforcer, JSON physically guaranteed) or via a remote `json_schema` call. This decouples *creativity* from *format compliance* — each model does what it's best at — and preserves fail-closed (the extractor can't emit invalid JSON). **Precedent:** OTR prototyped exactly this as the "Story Room" → transcript-to-structured extraction path (Sprint 10B, since lean-removed), and the building blocks still exist (`make_constrained_generate_fn` grammar extraction + the `structured_call` ladder). Trade-offs: one extra LLM call per structured pass (latency/cost — the extractor can be a cheap/local model), and the extractor must faithfully capture the creative content (a fidelity/QA concern). This reframes the whole feature: remote = the *writer*, local grammar = the *parser*.
- **E. Combination (likely recommended):** F (write-then-extract) for the creative metadata passes + A/B (`response_format`) anywhere we still want the remote model to emit JSON directly + C (prompt hardening) for robustness, with fail-closed preserved throughout.

## Hard constraints (any solution must respect)

- **Fail-closed integrity:** malformed/invalid JSON must never reach the ledger; abort cleanly on exhaustion. (Currently holds.)
- **Local default unchanged / byte-identical:** the offline pipeline (remote off) must not regress; audio is king.
- **No new model picker / widgets;** remote stays opt-in, default-off, env-gated.
- **Cost guard preserved** (per-call + per-run token ceilings; abort before overspend).
- **The plain `slot_fn` signature** is `(messages, *, temperature, max_new_tokens)` and is shared by local + remote, free-form + structured callers — any schema/response_format threading must not break that contract or the local path.

## Questions for the round-robin

0. **(Lead question)** Should the creative (frontier/remote) model emit structured JSON at all, or should we adopt **write-then-extract** — free-form creative generation + a separate reliable extractor (local grammar-constrained, or remote `json_schema`) that serializes prose into the ledger schema? Is this the right architecture for "remote = writer, local = parser," and what's the latency/cost/fidelity trade vs. inline structured output?
1. For reliable **schema-valid** JSON from a frontier remote model on OTR's creative-slot structured calls (where we DO keep inline JSON), what's the correct minimal design — `json_object`, `json_schema`, prompt hardening, per-call routing, or a specific combination?
2. Is relying on prompt-only JSON (no grammar/response_format) an acceptable pattern at all, or should *every* JSON call — local and remote — be mechanically enforced (grammar locally, `response_format` remotely)? I.e., should the prompts ever be load-bearing for format?
3. If we adopt `json_schema` remotely, how should the Pydantic schema reach the call site without breaking the shared `slot_fn` contract or the local path?
4. Does the failure also hint that OTR's prompts are over-fit to local models (system-role handling, terseness, implicit conventions) in ways that will bite other frontier models — and is a model-agnostic prompt pass warranted regardless?
5. Anything about Anthropic-via-OpenRouter specifically (e.g., it prepends reasoning/prose, or wants a particular system-prompt shape) that changes the recommendation?

## What is already proven (so the fix is well-targeted)

- Remote selection, the dispatch wiring, **VRAM no-evict** (remote creative call leaves the resident local technical model in place), the **cost guard**, the **enabled/disabled gate**, and the **metadata stamp** all work live.
- The local pipeline renders complete episodes end-to-end (validated repeatedly this session).
- So this is narrowly: **how to make a remote frontier model return JSON our `structured_call` ladder accepts**, without regressing any of the above.
