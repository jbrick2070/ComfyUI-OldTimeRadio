# Session Handoff -- OTR v2.0-alpha -- 2026-06-04 (writer bake-off DONE; NEXT = code gemma-4-12b opt-in lane per pass02 plan)

## Core goal
Make **gemma-4-12b** usable as an **opt-in** OTR writer lane without breaking the
exact-count structured passes -- then decide (by quality) whether it's worth it.
**mistral-nemo stays the shipping default either way.** The build plan is already
converged and grounded: `docs/2026-06-04-gemma4-writer/roundtable/pass02_plan.md`.
This session ran the bake-off (gemma aborts as-is), shipped a `<think>` strip, and
roundtabled the fix. Next session = implement pass02 (code).

## Tech stack & constraints
- Branch `v2.0-alpha`. CLAUDE.md + BUG_LOG.md + ROADMAP auto-load -- not repeated here.
- **NOTHING IS COMMITTED. Jeffrey commits.** This session's edits are dirty in the tree.
- VRAM ceiling 14.5 GB (RTX 5080, 16 GB, Blackwell, Windows). Offline-first / 100% local.
- Writer two-slot model (creative_writing_model / technical_model); **no new model_id widgets** (rule 6); consumers get the id via STRING socket.
- The OpenRouter lane is **provider-agnostic**: `OPENROUTER_BASE_URL` points it at ANY OpenAI `/v1` (this is how it reached Ollama). Enabled by `OTR_ENABLE_OPENROUTER=1` + `OPENROUTER_API_KEY` (any non-empty for a local server).
- Run the full suite + Bug Bible after every code change: `"%VENV%" -m pytest -q -p no:cacheprovider` (venv = `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`). Baseline this session: **3705 passed / 12 skipped / 0 failed**; Bug Bible 0-fail.
- **HF_HOME = `C:\ComfyUI-Models\huggingface`** -- a bare headless ComfyUI relaunch MUST set this or local writer models show "[NOT DOWNLOADED]" (the Desktop app sets it; ComfyUI is currently back on Jeffrey's Desktop app).
- DC (Desktop Commander) gotchas this session: cmd mangles `"`/`$_` -> avoid quotes even on space-free paths, and use the subprocess+powershell-list-args pattern for `$_`; `ping`-wait > ~70s trips the MCP transport timeout; the file-tool mount LAGS Windows writes -> read repo files via DC `read_file`. cmd `mkdir` needs NO quotes to create nested dirs.

## What's done & decided
- **Bake-off (docs/2026-06-04-writer-bakeoff.md):** gemma-4-12b (Ollama GGUF via the lane) **aborts at the style-picker "inventor" pass** -- returns **63** descriptors when exactly **5** are required; 3 attempts; fail-closed. mistral-nemo completes ("Serum's Gambit", ~21/35 this draw; best-local baseline 27/35). The `<think>` strip works (gemma cleared NewsCuration/NewsCurationDeep); failure is **instruction-compliance**, not transport.
- **SHIPPED this session (uncommitted):** `_strip_reasoning_tags` in `nodes/_otr_openrouter_backend.py` (called in `_extract_text` before the empty-check) strips `<think>...</think>` (balanced + dangling-close) + harmony `<|channel|>` markers; no-op when absent. +10 tests in `tests/test_openrouter_backend.py`.
- **`DEFAULT_LLM` is already `mistralai/Mistral-Nemo-Instruct-2407`** (`nodes/_otr_model_catalog.py:32`) -- no change; bake-off confirms keep it.
- **Roundtable converged (Grok + ChatGPT + Gemini panel, Claude judge; ~$0.08 of a $5 budget):** the build path is **pass02_plan.md**. Key decisions:
  - **B** = `_parse_inventor_output` takes first-5-distinct on overgeneration but **KEEP `StylePick.candidates Field(min=max=5)`** -- truncate in the parser, do NOT weaken the contract (corrects an earlier "relax to >=5" draft).
  - **F** = serve `unsloth/gemma-4-12b-it-GGUF` (Apache-2.0, verified on HF) via **llama-server** through the existing `/v1` lane. NO in-Comfy transformers, NO new widget; run for writer passes only then UNLOAD before the FLUX/HuMo/LTX branch; one resident LLM at a time.
  - **enable_thinking:false** via llama-server `--chat-template-kwargs '{"enable_thinking":false}'` for structured passes.
  - **A** (GBNF "exactly 5" at the inventor call) only if B + thinking-off is insufficient. Prefer GBNF over JSON+schema (keeps the line-based contract).
  - **GGUF fixes deployability, NOT the count bug.** A/B fix the bug.
- **Rejected (do not reopen):** in-Comfy transformers gemma (BUG-306); gemma-as-default; relaxing `StylePick` to `>=5`; two resident local LLMs during video; prompt-only fix (failed 3x); a full Ollama-vs-llama.cpp-vs-LM-Studio bake-off (lane takes any `/v1`; only GBNF/json_schema-over-/v1 matters). Defer **C** (gemma-creative + mistral-technical slot routing): technical dispatch is staged, not wired, + VRAM trap.

## State of the art (files + exact surfaces)
- **`nodes/_otr_style_picker.py`** -- where B lands. `_REQUIRED_CANDIDATE_COUNT = 5`;
  `DESCRIPTOR_RE` = 2-5 lowercase snake_case words; `_parse_inventor_output(raw)`
  returns `list[str]` and currently RAISES `ValueError` when `len(lines) != 5`
  (the count check ~line 363) -- change THIS to de-dupe + take first 5 (fail if <5).
  `class StylePick` -> `candidates: Field(..., min_length=5, max_length=5)` (line 126)
  -- KEEP. `pass1_slot` default `"creative"` (138), `pass2_slot` `"technical"` (139);
  the paired contract routes BOTH passes through `creative_fn` today (technical
  dispatch staged "B2") -> A must enforce at the inventor (creative) call, and C is
  blocked on finishing this dispatch.
- **`nodes/_otr_openrouter_backend.py`** -- `_strip_reasoning_tags` (shipped);
  `resolve_slug` (bound slot value verbatim else `OTR_OPENROUTER_SLOT_A_DEFAULT`);
  `schema_to_response_format` (json_schema seam, technical calls);
  `_clean_slot_value` treats parenthesized `(...)` choices as unset.
- **Roundtable artifacts:** `docs/2026-06-04-gemma4-writer/roundtable/` ->
  `pass00_plan.md`, `pass01_plan.md`, `pass01_judgment.md`, **`pass02_plan.md` (the build plan)**,
  `CHATGPT_PASTE.md`, `pass01/` (Grok + Gemini reviews + manifest). `pass01b/` recovery stalled (ignore).
- **Bake-off scratch (gitignored `scripts/_*.py`, reusable):** `_otr_bakeoff_run.py`
  (patch writer model + neutralize the 4 slot pickers to their `(...)` placeholder +
  prune to node-7 closure + cancel on `freeze_verdict` before audio + dump ledger),
  `_otr_bakeoff_launch_comfy.py` (headless ComfyUI with Ollama+seeds+HF_HOME env),
  `_otr_bakeoff_wait.py`, `_otr_bakeoff_show.py`, `_otr_bakeoff_procs.py`. Captured
  ledgers: `_otr_bakeoff_gemma4.json` (FAIL), `_otr_bakeoff_mistral.json` (STORY_READY).
- **Ollama** has `unsloth/gemma-4-12b-it-GGUF:Q4_K_M`; pass02 recommends `UD-Q4_K_XL`.
  The OTR canonical workflow `workflows/otr_scifi_16gb_full.json` carries STALE slot
  values (`comfy_slot_b_model='deepseek/deepseek-v4-pro'`) that fail ComfyUI COMBO
  validation when the lane is off -> the driver neutralizes all 4 pickers (pattern to reuse).

## Immediate next steps (code, in order -- from pass02_plan.md)
1. **B:** edit `_parse_inventor_output` (`nodes/_otr_style_picker.py`): collect valid
   `DESCRIPTOR_RE` descriptors, de-dupe deterministically (respect distinctness), if
   `>=5` take first 5, if `<5` raise; return exactly 5 so `StylePick(min=max=5)` holds.
   Add a unit test (over-generation -> 5; <5 -> fail; dup-collapse). Run suite + Bug Bible.
2. **Telemetry:** stamp `valid_count` / `distinct_count` / `truncated_count` / `model_slug`
   on the style pass (ledger meta) so over/under-generation is visible, not just an abort.
3. **F:** `llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL --port <p>` (confirm the
   exact quant tag from the repo files). Point `OPENROUTER_BASE_URL=http://localhost:<p>/v1`;
   verify the lane's OpenRouter headers don't trip llama-server. Reuse the bake-off driver to submit.
4. **enable_thinking:false** for structured passes (llama-server `--chat-template-kwargs`,
   PowerShell-escaping caveat). Re-test the exact-5 style-picker with gemma.
5. **A (GBNF)** at the inventor call ONLY if B + thinking-off still over-generate.
6. **Conformance harness (6 pt):** exact-5 ok / no 63 / dup caught / bad-grammar caught /
   GBNF-or-json_schema works-or-marked-unsupported / **server unloads before video**; AND
   **grep every other exact-count/shape gate** (chooser, cast contract, validators) for blast radius.
7. Only then re-bake-off gemma narrative vs mistral on the 7-axis rubric. **If not clearly better, stop.**

## Open questions
- Exact Unsloth quant tag (`UD-Q4_K_XL`) in the repo file list.
- Does `enable_thinking:false` (+ GBNF if needed) actually yield exactly 5 from gemma?
- Do the OpenRouter-lane headers work against a local llama-server?
- Full list of other exact-count gates gemma would break (harness step 6).
- Product call: is softening the strict 5-count gate (B) acceptable? (Jeffrey)
- Housekeeping: a stalled roundtable `pass01b` python + an idle `ollama serve` may linger from this session (harmless; kill if tidying).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
