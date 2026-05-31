# Session Handoff — OTR Dual-Route LLM (local-OR-API model router) — 2026-05-31

## Core goal
Start DESIGNING (not yet building) the "dual-route LLM": each of the writer's two LLM slots
— `creative_writing_model` and `technical_model` — can be a LOCAL model OR an API model, with
the SAME workflow wiring and the SAME ledger commit path. This is **CARD 1 of
`workflows/GO_FORWARD_PLAN_v11_model_router_then_cleanup_2026-05-28.md`** ("prove the model
router — architecture risk only"). Operator wants to evaluate **OpenRouter** as the API backend.
This is a research/design session: map OpenRouter onto the existing slot machinery, surface the
decisions, produce a sprint plan. Do NOT mix in any story redesign (that's Card 3, back-pocket).

## Tech stack & constraints
- ComfyUI custom node; Python 3.12 venv `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`;
  branch `v2.0-alpha` only; Windows. git via Desktop Commander cmd; tests via venv pytest (full
  `tests/` walk green = the 5 known pre-existing failures only). CLAUDE.md + ROADMAP + BUG_LOG
  auto-load — don't re-derive them; this handoff is the live add-on.
- **THE decision to get from Jeffrey (everything waits on it):** CLAUDE.md says "100% local, open
  source, offline-first; no cloud services, no API keys, no paid services." OpenRouter is cloud +
  paid + API-key. v11 Card 1 explicitly flags this and only allows it as "API as an *option* with
  local as default/fallback." Operator's stance this session: out-of-the-box behavior must be
  zero-config — so **local stays the zero-config default; API is opt-in.** Keys live in ENV, never
  in the workflow JSON ("no secret keys in JSON").
- Card 1 hard rules: API JSON retries fail LOUD (never silently route back to local mid-call);
  if a slot's API path is unset/unreachable → fall back to local; episode `meta` records
  backend/model/slot; no node may assume Hugging Face / local-only model IDs.

## What's done & decided (this session — the cast sprint, now CLOSED)
- Cast name↔gender↔voice coherence sprint (S0–S9 + S3) **shipped + validated live** on
  `v2.0-alpha`, HEAD `5adda65` (+ tools `7530c52`). Commits: S1 `acc091a` / S0 `a0bd03b` /
  S2 `30c666d` / S4–S8 `7e04dc4` / S3 `31ea332` / S9 `e37df79`. Live proof — episode "Cooling
  Race" cast fully coherent (ANYA→female→female voice, NED→male→male voice, 0 mismatches);
  `meta.writer_llm_unload=unloaded` confirmed S3 active; Bug Bible static checks 23/23 on the pack.
  Full record in BUG_LOG.md sprint marker. **Don't reopen the cast work.**
- Defaults are correct out-of-the-box: coherence repair on in default pool mode (no env var, no
  workflow edit); `OTR_NAME_CROSS_GENDER_RATE=0.0` strict; `OTR_NAME_MODE=llm_slot_fill` + genre
  are opt-in flavor only.
- **Voice age-axis: DROPPED** — operator said "I don't care about ages." S5's `age_band` stays
  inert in pool mode (gated to llm_slot_fill); do NOT make it default. Closed.
- New committed tool: `scripts/otr_tail_logs.py` (live log tailer — venv python, forced UTF-8 on
  read AND stdout, filters fake `%|` progress-bar `[error]` lines, hits `/queue`+`/history`).
  CLAUDE.md is **gitignored** here — edits land on disk (govern the AI) but don't push.

## State of the art (the seam the router plugs into)
- **The two-model selector already exists** and is the ONLY model-id surface (CLAUDE.md PD6 — no
  other node carries a model_id widget; consumers get the id via a STRING socket). Widgets:
  `creative_writing_model` + `technical_model` on `OTR_LedgerScriptWriter`.
- **`_SlotScheduler` is the single injection point** (`nodes/OTR_LedgerScriptWriter.py:430`):
  `for_slot("creative"|"technical")` returns `generate_fn(messages, *, temperature, max_new_tokens,
  stop=None)`; `_account_and_get_entry` calls `nodes/_otr_model_loader.request_slot(slot,
  resolved_id)` to make a model resident. An API backend is injected here — when the resolved slot
  id is an API id, hand back an OpenRouter-backed `generate_fn` with the identical signature, and
  `_SlotScheduler` + the ledger commit path stay untouched.
- Local loader stack to mirror: `_otr_model_loader.py` (`request_slot`/`load_llm`/`unload_llm`,
  4-bit NF4, 1-token warmup), `_otr_loader_backends.py` (`make_generate_fn`,
  `normalize_messages_for_tokenizer` system-role fold), `_otr_model_catalog.py` (dropdown choices,
  license audit, `_snapshot_is_causal_lm`).
- Existing API-call pattern to copy: `scripts/_consult_openai.py` + the Round-Robin keys read via
  `winreg` from User-scope env (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`). Add
  `OPENROUTER_API_KEY` the same way.
- The `generate_fn` contract an API slot must honor: take `messages` (list of {role,content}),
  `temperature`, `max_new_tokens`, optional `stop`; return raw text. The JSON-constrained passes
  (cast validator, critic, GBNF-grammar locally) need an equivalent on a hosted model —
  OpenRouter `response_format`/structured outputs, NOT local grammar.

## Immediate next steps (research/design — NO code yet)
1. Read CARD 1 of `workflows/GO_FORWARD_PLAN_v11_model_router_then_cleanup_2026-05-28.md` in full
   (scope, the 4-combo matrix, pass/fail gates). Skim Card 2 (cleanup) for ordering.
2. **ComfyUI ships an OFFICIAL OpenRouter LLM partner node** (this is the "ad" Jeffrey saw —
   `https://docs.comfy.org/tutorials/partner-nodes/openrouter/llm`): one node, curated models
   (Claude, GPT, Gemini, Grok, DeepSeek, Qwen, Mistral, GLM, Kimi, Perplexity), `temperature` +
   `reasoning_effort`, and OpenRouter routing suffixes `:floor` (cheapest, default-on) / `:nitro`
   (fastest) = built-in cost control. Community nodes also exist (gabe-init/ComfyUI-Openrouter_node,
   EnragedAntelope/ComfyUI-EACloudNodes). **Critical nuance for OTR:** that partner node is a
   STANDALONE node = one output per graph execution, but `OTR_LedgerScriptWriter` makes MANY
   internal LLM calls per episode through `_SlotScheduler`/`request_slot`, so the node almost
   certainly can NOT be wired into the writer's internal slot calls. Likely answer: build an
   INTERNAL OpenRouter-backed `generate_fn` at the slot seam, using the partner node as the
   Comfy-blessed reference + the same `OPENROUTER_API_KEY` env + the `vendor/model` namespace +
   the `:floor`/`:nitro` cost flags. Verify the API shape (`/api/v1/chat/completions`,
   OpenAI-compatible), per-model `response_format`/structured-output support, and a pricing cap.
3. Draft the backend abstraction: a `request_slot`-level branch (or a `make_api_generate_fn`
   sibling of `make_generate_fn`) that returns an OpenRouter-backed `generate_fn`, selected when a
   slot's resolved id is an API id (e.g. `openrouter:vendor/model`). `_SlotScheduler` and the
   ledger path unchanged.
4. Spec: env keys (never in JSON), LOUD-fail retry, local fallback when unset/unreachable, `meta`
   stamps (backend/model/slot), and the JSON-constrained-output story per slot.
5. Take the directive sign-off + model-pick questions to Jeffrey BEFORE any build.
6. Produce a short ADR / sprint plan for Card 1 (mirror the cast `go-forward-sprint-plan` shape)
   with the 4-combo matrix as the exit gate. Stop for operator sign-off.

## Open questions
- **Directive sign-off (Jeffrey owns; gates everything):** confirm OpenRouter is an intentional
  opt-in *option* with local as the zero-config default + fallback.
- Which OpenRouter models for creative vs technical? Cost ceiling / budget guardrail?
- JSON-constrained passes (validators/critic): use OpenRouter `response_format=json_object` /
  structured outputs, or keep those passes local-only and route only free-form creative to the API?
  (Decides which of the 4 matrix combos are even meaningful.)
- Surface API ids through the existing `creative_writing_model`/`technical_model` dropdowns (add to
  the catalog) vs a separate `*_backend` env/socket — PD6 says NO new model_id widget either way.
- **Official partner node vs internal client:** use ComfyUI's official OpenRouter node directly, or
  build an internal OpenRouter `generate_fn` at the `_SlotScheduler` seam? The node is standalone
  (one call per execution) and can't serve the writer's many mid-pipeline passes — so the answer is
  probably "internal client; the partner node is the blessing + reference + key/config + cost-flag
  source." Reference docs: `https://docs.comfy.org/tutorials/partner-nodes/openrouter/llm`.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
