# Session Handoff -- OTR v2.0-alpha -- 2026-06-04 (gemma writer: B/telemetry/A SHIPPED+COMMITTED; NEXT = make gemma a viable FULL writer)

## Core goal
gemma-4-12b as an OPT-IN OTR writer lane. This session executed the prior
handoff's pass02 plan: shipped + COMMITTED the style-picker hardening (B tolerant
parser + telemetry + opt-in A/GBNF), and proved B+telemetry LIVE on real gemma.
**mistral stays DEFAULT_LLM.** gemma now CLEARS the style picker but is NOT yet a
viable FULL writer (halts downstream). Next session: get gemma to finish an
episode (enable_thinking:false + token cap), then decide adoption by quality.

## Tech stack & constraints
(CLAUDE.md / ROADMAP / BUG_LOG auto-load -- not repeated.) NEW this session:
- **Two commits on v2.0-alpha, NOT pushed:** `69de6dc` (docs hygiene: gitignore
  root _otr_* + ROADMAP item) and `39d55c7` (snapshot: gemma hardening + ALL
  accumulated session work, 129 files). Tree CLEAN except `custom_nodes.lnk`
  (junk, intentionally untracked).
- **A is opt-in/default-off:** the inventor GBNF is sent only when
  `OTR_ENABLE_INVENTOR_GBNF` is set AND the backend advertises
  `_otr_supports_grammar`. Default-off so Ollama / real-OpenRouter (no `grammar`
  field) never regress; B is the always-on net.
- **Lane is provider-agnostic** (`OPENROUTER_BASE_URL` -> any OpenAI /v1). gemma
  reached via OLLAMA at localhost:11434/v1 (model
  `hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M`, present). llama.cpp `llama-server`
  NOT installed; LM Studio present (`C:\Users\jeffr\.lmstudio\bin\lms.exe`).
- **Runs >70s MUST be DETACHED** -- the MCP call limit kills inline processes.
  Use `scripts\_otr_bakeoff_detach.py`. ComfyUI is currently RUNNING headless on
  :8000 (bake-off env).
- Baseline: full suite **3726 passed / 12 skipped / 0**; Bug Bible **23 / 0**.

## What's done & decided
- **B (parser net) -- SHIPPED+COMMITTED+PROVEN LIVE.** `_parse_inventor_output`
  skips malformed/dup/near-dup lines, takes first 5 distinct; `StylePick` stays
  strict (min=max=5); mistral byte-identical. Live: gemma's style pass that
  hard-aborted last session now succeeds (`valid=5 distinct=5 truncated=0`).
- **Telemetry -- SHIPPED+COMMITTED+PROVEN LIVE.** `meta.style_pick` carries
  valid_count/distinct_count/truncated_count/model_slug; model_slug resolves the
  `openrouter:slot-a` handle to the gemma slug live.
- **A (GBNF) -- SHIPPED+COMMITTED, opt-in, NOT live-proven.** grammar seam on the
  lane, threaded to the inventor only when `_otr_supports_grammar` +
  `OTR_ENABLE_INVENTOR_GBNF`. Live grammar-constraint proof needs a
  grammar-capable /v1 (llama-server) -- not run this session.
- **Exact-count gate audit -- DONE** (docs/2026-06-04-gemma4-writer/
  exact_count_gate_audit.md). The style picker was the UNIQUE line-based
  hard-fail-on-count gate; every other LLM-output gate is structured_call +
  bounded-repair (tolerant), Python-generated, or non-LLM. Nothing else needs
  B-style treatment.
- **Bake-off -- DONE; gemma NOT viable yet** (docs/2026-06-04-gemma4-writer/
  bakeoff_B_live_result.md). gemma clears the style picker but HALTS at
  `build_news_briefs`: every gemma call hits `finish_reason=length`
  (verbose/thinking burns the token budget before the JSON) -> JSONDecodeError
  char 0 -> hard halt (news_briefs_required=True). NOT a B/A defect.
- **Decided:** mistral stays DEFAULT_LLM. Rejected (do not reopen): in-Comfy
  transformers gemma, relaxing StylePick, sending grammar unconditionally
  (regresses Ollama).

## State of the art (files + surfaces)
- **nodes/_otr_style_picker.py** -- `_InventorParse` dataclass;
  `_parse_inventor_descriptors(raw)` (full distinct set + counts, no early stop)
  + `_parse_inventor_output(raw)` thin wrapper; `StylePick` + 4 telemetry fields
  (defaulted); `_build_inventor_gbnf()` + `_INVENTOR_GBNF` +
  `_inventor_gbnf_enabled()` (reads OTR_ENABLE_INVENTOR_GBNF); `_run_inventor`
  attaches grammar iff marker+enabled (the `# LLM slot: creative` tag sits
  ADJACENT to the call -- rule-6 sweep, SEARCH_WINDOW=8); `pick_style(...,
  model_slug="")`.
- **nodes/_otr_openrouter_backend.py** -- `OpenRouterBackend.generate(...,
  grammar=None)` -> `payload["grammar"]` + require_parameters;
  `make_openrouter_generate_fn(..., grammar=None)` binds + per-call grammar +
  sets `generate_fn._otr_supports_grammar = True`.
- **nodes/OTR_LedgerScriptWriter.py** (~line 2338, D.2 style block) -- resolves
  `_creative_slug` via `_otr_openrouter_backend.resolve_slug` ONLY for
  `openrouter:` handles (defensive try/except), passes model_id + model_slug.
- **Tests:** tests/test_otr_style_picker.py (B/telemetry/A/threading incl.
  flag-off), tests/test_openrouter_backend.py (grammar payload + marker).
- **Scratch tooling (gitignored scripts/_*.py):** `_otr_bakeoff_detach.py`
  (DETACHED driver launch -- REQUIRED for >70s runs), `_otr_live_ledger.py
  [cancel]` (read meta.style_pick from /otr/latest_ledger; cancel before audio),
  `_otr_bakeoff_launch_comfy.py` (headless ComfyUI w/ Ollama lane env; the
  file-handle leak was fixed this session), `_otr_bakeoff_run.py` (driver),
  `_otr_bakeoff_show.py`. Captured: `_otr_bakeoff_gemma4live.json` (FAIL @
  build_news_briefs), `_otr_bakeoff_gemma4live.run.log`, `_otr_comfy_headless.log`.

## Immediate next steps
1. **enable_thinking:false + token cap (THE gemma-viability blocker).** gemma
   halts at build_news_briefs on finish_reason=length. Try: (a) raise the remote
   output budget -- set `OPENROUTER_MIN_OUTPUT_TOKENS` (e.g. 4096) and/or the
   slot max-tokens cap in the ComfyUI launch env (`_otr_bakeoff_launch_comfy.py`
   uses dict(os.environ) as base, so exporting them before launch works); (b)
   disable thinking -- test whether Ollama's /v1 honors a think-off (Ollama
   `think:false` / `chat_template_kwargs`), else this needs llama-server
   `--chat-template-kwargs '{"enable_thinking":false}'`. Re-run:
   `python scripts\_otr_bakeoff_detach.py openrouter:slot-a gemma_thinkoff 320`,
   poll `python scripts\_otr_live_ledger.py`. Goal: STORY_READY (lines>=4 +
   freeze_verdict), or confirm thinking-off is mandatory.
2. **A GBNF live-proof (if a grammar /v1 is available).** Install/serve
   llama.cpp `llama-server` OR verify LM Studio honors the `grammar` field; set
   `OTR_ENABLE_INVENTOR_GBNF=1` + point `OPENROUTER_BASE_URL` at it; confirm
   gemma emits exactly 5 (truncated_count stays 0 even on a draw that would
   otherwise overgenerate).
3. **If gemma reaches STORY_READY:** grade narrative vs mistral on the 7-axis
   rubric (`_otr_bakeoff_show.py _otr_bakeoff_<tag>.json`). If not clearly
   better, STOP -- mistral stays.
4. **Push** v2.0-alpha (commits `69de6dc` + `39d55c7`) -- not pushed this session
   (`cd /d <repo> && git push origin v2.0-alpha` via Desktop Commander cmd).
5. Housekeeping: ComfyUI headless still running on :8000 (kill if opening the
   Desktop app); `custom_nodes.lnk` is an untracked junk shortcut (delete/ignore).

## Open questions
- Does Ollama's OpenAI /v1 expose a thinking-disable (think / chat_template_kwargs)?
  If not, gemma-viability needs llama-server.
- Does LM Studio's /v1 honor the llama.cpp `grammar` field (needed for A's live proof)?
- Is a higher token cap ALONE enough for build_news_briefs, or is
  enable_thinking:false required?
- Product (Jeffrey): is gemma worth the tuning, or stop at mistral as the proven default?

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
