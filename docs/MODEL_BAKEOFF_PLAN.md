# Model Bake-off Plan -- 420w + 720w across all valid models (pick the writer)

**Created:** 2026-07-17. **Type:** RENDER-window campaign (GPU days), NOT a coder slot.
**Executes:** GO_FORWARD "Then, in order" item 3 -- the creative-writer model matrix
("the only path from 'best bank on aion' to 'best model'"). **Baseline HEAD:** `5095bc19`.

## Objective
Run a 420-word and a 720-word leg for EVERY valid local model on ONE held-constant
bank, grade the transcripts blind, and pick the best writer model(s). Length is a
RECORDED property, never a pass/fail gate (operator `feedback_never_chase_word_count`):
a leg is GREEN on RESULT SUCCESS + freeze; content-fails (weapons/profanity/schema)
are RECORDED with their reason, never re-rolled to force green.

## Model roster (exact ids -- confirmed live in nodes/_otr_model_catalog.py + _otr_gguf_backend.py, 2026-07-17)
Native HF (transformers):
- `mistralai/Mistral-Nemo-Instruct-2407`  (DEFAULT_LLM, 12B -- the baseline to beat)
- `Qwen/Qwen2.5-14B-Instruct`             (14B -- largest native)
- `google/gemma-4-E4B-it`                 (~4B)
- `google/gemma-4-E2B-it`                 (small; technical-slot class -- may fail creative, RECORD it)
- `google/gemma-2-2b-it`                  (smallest; technical-slot class -- may fail creative, RECORD it)

GGUF (in-process llama-cpp):
- `unsloth/gemma-4-12b-it-GGUF`           (Gemma 4 12B Q8_0 -- baseline PASS, gemma-row registry)
- `unsloth/Qwen3-8B-GGUF`                 (Qwen3-8B reasoning; needs `/no_think` hygiene -- proven 2026-07-16)

Optional cloud (only if the operator wants them; need env/keys, not 100%-local):
`openrouter:slot-a` / `openrouter:slot-b`, `google_api:slot-a` / `google_api:slot-b`.

## What to hold constant
- **Bank (operator call at run start; default `scifi_fable2` -- the flagship #1 per the 720 verdict).**
  Alternative: `scifi_codex_v4` (just shipped). Pick ONE and keep it fixed so the MODEL is the only variable.
- **Slots:** set BOTH `creative_writing_model` AND `technical_model` to the SAME candidate per leg
  (each model does the whole job -- the simplest "which model" comparison). A deeper creative x technical
  matrix is a follow-up, not this pass.
- Fresh source per leg (production mode -- no C7/frozen-source); temperature pinned by the pack.

## Harness -- STORY-ONLY (skip the ~30-min TTS/video tail; story quality is the signal)
`workflows/otr_story_only.json` (validator -> writer -> freeze; built by `scripts/build_story_only.py`).
FIRST: confirm it exists and is current vs the canonical (`git log` on it / rebuild via
`python scripts\build_story_only.py` if the canonical changed since 2026-07-16). Each story-only leg
~12-20 min at 420/720. 7 models x 2 tiers = 14 legs ~= 3-5 GPU hours (more if a model needs retries).

Per-leg command (via a sweep launcher -- see the QUOTING gotcha below):
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 `
  -Profile none -Words <420|720> -Workflow workflows\otr_story_only.json `
  -Set "OTR_LedgerScriptWriter.source_bank=<BANK>","OTR_LedgerScriptWriter.creative_writing_model=<MODEL>","OTR_LedgerScriptWriter.technical_model=<MODEL>"
```
Exit 0 = RESULT SUCCESS; the frozen ledger/transcript is what you grade.

## HARD gotchas (learned 2026-07-17 -- do not relose)
- **`-Set` is a `[string[]]`: pass it as a COMMA-ARRAY** (`-Set "a=..","b=..","c=.."`), NEVER as
  repeated `-Set a -Set b -Set c` -- repeated named params are REJECTED ("ParameterAlreadyBound"),
  whether via `-File` or in-process `&`. Write a sweep launcher `.ps1` (via the file tools) and run it
  with `powershell -File`; keep all quoting inside the script.
- **Reset selectively before EVERY leg** (CLAUDE.md section 4): CIM-kill by CommandLine only the OTR
  server/runner (never a blanket `Stop-Process -Name python` -- that severs the MCP pythons); confirm
  `:8000`/server port free + VRAM at baseline. The wrapper self-resets, but verify.
- **The full-media obs pipeline is NOT run here** (story-only). The base video/obs assets do not apply;
  grade the frozen transcript/ledger.
- **VRAM ceiling 14.5 GB** (host NVML): Qwen2.5-14B and the 12B-Q8 GGUF are the tight ones -- watch the
  BUG-098 tripwire; if a model OOMs, RECORD it as a capacity fail (a real "don't pick" signal), do not
  raise the ceiling.
- **GGUF reasoning models** (`Qwen3-8B`): the lane already handles `/no_think` + dangling-`<think>` strip
  (2026-07-16 fixes). If a GGUF leg fails on thinking artifacts, that's a lane bug -> two-strikes -> kibitz.
- **A model that fails is DATA** (it tells you not to pick it). Root-fix a lane bug (per THE LAW /
  no-fallback / LLM-first), but a genuine model-capability fail is recorded, not forced green.

## Deliverables
1. Receipts CSV per leg (model, tier, bank, RESULT, words, commit, resolved model id, fail-reason).
2. Transcript extraction -> one Fable BLIND-quality pass -> a MODEL SCOREBOARD
   (`docs/2026-07-17-model-bakeoff-scoreboard.md`): named scoring axes, every cell explicitly
   SUCCESS / FAIL / DISQUALIFIED / NOT-RUN (no silent omissions), receipts carrying matrix id + commit +
   resolved model. Reuse `tmp/_extract_for_grading.py` + `tmp/_assemble_matrix.py` from the prior campaign.
3. A recommended pick (best writer model, with the runner-up + the tradeoffs: quality vs VRAM vs speed).
4. One FULL-MEDIA confirmation leg on the WINNER (canonical workflow, not story-only) before adopting it
   as a default -- proves the winner survives the whole pipeline.

## Standing rules
100% local for the build (cloud only if operator opts in); commit any harness/code change to v2.0-alpha
with the full gate (suite 8144 / Bible 17); prod/main GATED; UTF-8 no BOM; SFW. Two failed solo fixes on
one problem -> /kibitz before a third. Update GO_FORWARD + HANDOFF_LOG at wrap-up.
