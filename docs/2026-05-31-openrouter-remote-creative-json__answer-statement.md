# Answer statement — recommended path: **script-first, parse-second**

**Date:** 2026-05-31 · **Branch:** v2.0-alpha · Companion to `2026-05-31-openrouter-remote-creative-json__problem-statement.md`. This is my opinionated recommendation, written to be compared against other opinions.

---

## The recommendation in one sentence

**Stop asking the creative model to emit JSON. Let any LLM *write the episode as a plain-text script*, then turn that script into the ledger with a deterministic parser plus a few small, grammar-locked extraction calls.** Generation and structuring become separate jobs: the writer writes, the parser parses.

This is your instinct, and I think it's correct — not a workaround. It's how robust story-gen pipelines actually work, and it's the only design that is simultaneously (a) model-agnostic, (b) basic-prompt, (c) fail-closed, and (d) high-quality.

---

## Why "make the LLM output JSON" is the wrong ask

- A frontier model is a **writer**, not a **serializer**. Asking it to emit a JSON object mid-creative-flow fights its grain — Opus returned prose and we fail-closed. Asking the *creative* call to also be *correct JSON* couples two unrelated concerns.
- Locally it only ever "worked" because something else did the enforcing: a **token grammar** (lm-format-enforcer makes invalid JSON physically impossible) on the strict passes, or Mistral-Nemo happening to obey. The **prompt was never load-bearing for format.** That's why a better model broke it.
- JSON-as-the-interface also means every prompt carries schema scaffolding — the opposite of the "basic prompts any LLM can follow" goal.

**Corollary:** the structure should be enforced by *machinery* (a parser / a grammar / a schema), never by *hoping the model formats correctly*. Generation should be free.

---

## The design: two concerns, cleanly split

### Concern 1 — GENERATION (creative, any LLM, basic prompt)

One creative call (or a small handful) asks the model to write the episode **as a script in plain teleplay text** — the single format every capable LLM has seen millions of and produces beautifully with a one-line instruction:

```
Write a short [genre] radio drama inspired by: <news seed>.
Give it a TITLE. List the CAST as "NAME — one-line description".
Then write the script as lines "NAME: dialogue", with scene breaks
and sound/music cues in (parentheses). Keep it ~<N> spoken words.
```

No JSON, no schema, no model-specific tricks. This call is interchangeable across Mistral-Nemo, Opus, GPT-5, Gemini — they all do it well. **This is where remote frontier quality buys the most.**

### Concern 2 — STRUCTURING (deterministic first, grammar-locked LLM only where needed)

Convert that script text into the ledger in layers, strict layer first:

1. **Deterministic parse (no LLM).** OTR *already ships* a multi-format script parser (the v5 `NAME: dialogue` parser + structural-token blacklist). It turns the script into ordered line rows (speaker + text), pulls the TITLE, and isolates `(SFX:)` / `(MUSIC:)` cues. This handles the **bulk** of structuring with zero LLM-JSON risk.
2. **Deterministic cast/voice (no LLM).** OTR already assigns gender/voice/preset from name pools in pure Python (the cast-coherence work). The script only needs to *name* characters + sketch them; Python does the rest. So "casting" stops being a JSON-from-LLM call entirely.
3. **Grammar-locked extraction calls — only for what the parser genuinely can't infer.** Where structure needs judgment (e.g., per-beat dramatic intent, mapping a stray speaker to a known `char_id`, a clean one-line character description, or a normalized scene list), make a **small, separate extraction call** that is **physically constrained to valid JSON**: lm-format-enforcer (local) or `response_format: json_schema` (remote). These calls take the *script text* as input and emit a *tiny, well-defined* object. They can't fail the parse because the grammar forbids it; Pydantic + the existing fail-closed ladder still guard content.
4. **Deterministic assembly + invariants (no LLM).** Python builds the ledger from the parsed lines + cast + extracted bits, then runs the existing FreezeCascade/G1–G8 invariant battery.

### The flow

| Step | Who | Output | How structure is guaranteed |
|---|---|---|---|
| 1. News/premise | technical (local) or fixed | seed | n/a |
| 2. **Write script** | **creative LLM (any/remote)** | plain-text teleplay | n/a — free-form, basic prompt |
| 3. Parse script | deterministic Python | line rows, title, cues | code (no LLM) |
| 4. Cast → voice | deterministic Python | cast roster + presets | code (no LLM) |
| 5. Targeted extracts | small grammar-locked LLM calls | tiny typed objects (beats/char_id/etc.) | **grammar / json_schema — invalid JSON impossible** |
| 6. Assemble + invariants | deterministic Python | the ledger | code + fail-closed gates |

"As many LLM calls as it takes" → in practice: **1 strong creative write + a few cheap grammar-locked extracts.** Every call has a *basic* prompt.

---

## Why this hits every one of your goals

- **"Basic for any LLM"** — the creative call is "write a script"; the extract calls are grammar-locked so the prompt is just "pull X from this text." No model-specific prompting.
- **"Don't need huge prompts"** — generation needs none; extraction needs a sentence + a schema the grammar enforces.
- **"Good ledger story, no matter how many calls"** — quality comes from a frontier model writing freely; correctness comes from deterministic parsing + grammar. Decoupled, so both can be maximized.
- **Works for the best LLMs *and* local** — the writer is swappable; nothing depends on a model emitting JSON.
- **Fail-closed preserved** — structuring is enforced by parser + grammar + the existing `structured_call` ladder; bad data still can't reach the ledger.
- **Reuses what OTR already has** — the v5 script parser, the deterministic cast/voice logic, lm-format-enforcer, the invariant cascade, the no-evict loader, the cost guard. This is *less* new code than threading schemas through every creative call.

---

## How other story-gen workflows do this (validation)

This is the mainstream pattern, not a novelty:

- **Generate-then-extract** is the standard for reliable structured output: produce natural text, then a *separate* pass serializes it — using **constrained decoding / grammars** or **`json_schema` / function-calling** as the enforcement, never prompt-only JSON.
- **Two-model split:** a large creative model writes; a small, fast, cheap model (or a local grammar pass) structures. Pricey creativity, cheap reliability.
- **Screenplay/agentic writers** emit standard script formats (Fountain/teleplay) and parse them — because models are *trained* on scripts and the format is unambiguous to parse.
- The lesson everyone converges on: **make format a property of the machinery, make creativity a property of the model.**

---

## Slot mapping (fits the OpenRouter work already shipped)

- **Creative slot = the script writer.** This is the remote/frontier target. It only ever writes prose/script → the JSON-coercion failure disappears.
- **Technical slot = the extractor/parser-assist.** Keep it local + grammar-locked by default (Mistral-Nemo is fine and free here), or allow a remote `json_schema`-capable model. The extraction calls are exactly the structured_call sites the shipped S4 fail-closed gate already covers.

The OpenRouter feature we built (no-evict, cost guard, enabled gate, stamp, fail-closed) stays — this change is about *what we ask each slot to produce*, not the plumbing.

---

## Go-forward plan (phased)

- **P1 — Script generation + deterministic parse.** Add a "write the full script" creative call with a basic prompt; route its output through the existing script parser to produce line rows. Get a ledger populated from a parsed script with **zero LLM-JSON on the hot path**. Validate locally (byte-identical default preserved as an opt-in path).
- **P2 — Grammar-locked extraction for the few structured bits.** Identify the handful of fields the parser can't infer (beat intent, char_id disambiguation, 1-line descriptions) and back each with an lm-format-enforcer (local) / `json_schema` (remote) extraction call. These are the only JSON calls, and they cannot emit invalid JSON.
- **P3 — Remote enablement.** Point the creative slot at OpenRouter Slot A (Opus). Because the creative call is now free-form, Opus succeeds; the extracts are grammar-locked so they succeed regardless of model. Re-run the enabled smoke.
- **P4 — Quality eval.** A/B the script-first ledger (local vs Opus creative) on a rubric (arc, voice distinctness, payoff) — this is where you find out if Opus is "out of the ballpark." Keep the cost guard + fail-closed throughout.

---

## Trade-offs / risks (be honest)

- **Extra calls = latency + a little cost.** Mitigated: the bulk (parsing, cast/voice, assembly) is deterministic; extracts are small + can be local.
- **Fidelity of extraction.** The ledger is only as faithful as the parse. Mitigated by making the **script the source of truth** (deterministic parse of an unambiguous format) and keeping LLM extraction to small, well-scoped fields.
- **It's a real refactor of the writer's middle.** OTR currently builds structure *as it generates* (outline → compose_line → ledger). Script-first inverts that (generate → parse). Worth it, but not a one-line change — it touches the writer's spine, so it should be its own sprint with the regression suite as the gate.
- **Some structure (act/beat budgeting) is currently planned *before* generation.** Decide whether to keep a light pre-plan (a basic "outline in 5 bullet points" prose call) feeding the script call, or go fully script-first and derive beats by parsing. I lean: **light prose pre-plan → script → parse.**

---

## My bottom line

Adopt **script-first, parse-second**. Make the creative LLM do one thing it's elite at — write a script — with a basic prompt that any model obeys. Make *machinery* (deterministic parser + grammar/schema-locked extraction) responsible for the ledger, never the model's JSON discipline. It directly serves your goal ("good ledger story, any LLM, however many calls"), it's the industry-standard pattern, and it reuses most of what OTR already has. Forcing JSON out of the writer — via `response_format` or prompt nagging — would "work" but it optimizes the wrong thing and keeps the writer doing a job it shouldn't.
