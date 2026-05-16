# Sprint D -- Period LLM Wire-up

## Status

- Phase: planning
- Current commit: <none>
- Branch: <to-be-cut>
- Cut from: main@0aa6d6e

## Plan (commit chain)

The Sprint D scope as it exists at planning-phase entry. Concrete commit decomposition (sizes, gates, pytest tables, etc.) will be filled in during the round-robin reviews. Below is the scope-level enumeration the planning conversation inherits from the Sprint C close.

| # | Commit | Day est. | What lands | Status |
|---|---|---|---|---|
| 1 | D0a | TBD | branch cut + plan landing | pending |
| 2 | D0b | TBD | license audit precondition: confirm talkie-lm/talkie-1930-13b-it license terms; if MIT-incompatible, gate model into "research lane" with quarterly re-evaluation per ROADMAP standing directive | pending |
| 3 | D1 | TBD | register `talkie-lm/talkie-1930-13b-it` in `nodes/_otr_model_catalog.py` with VRAM bucket sizing (~7-8 GB at int4 quantization per ROADMAP); add to `CURATED_LLM_MODELS` and `CURATED_CONTEXT_OVERRIDES` | pending |
| 4 | D2 | TBD | period-LLM-aware prompt routing in `OTR_LedgerScriptWriter`: when the creative slot is set to a period LLM, route through `_otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT` instead of the modern composer prompt; keep technical slot on the structured-output path unchanged | pending |
| 5 | D3 | TBD | VRAM bench: confirm period LLM + technical slot model + DSP chain all fit under 14.5 GB ceiling on the dual-slot config | pending |
| 6 | D4 | TBD | pytest coverage: model registration, prompt routing, license-audit fallback path, dual-slot VRAM regression guard | pending |
| 7 | D-final | TBD | sprint close (Sprint A handoff still pending; Sprint D feeds the same downstream consumers) | pending |

**Pytest-only acceptance discipline** carries forward from Sprint C: ComfyUI Desktop runtime is out of scope; runtime / quality verification belongs to Sprint A's empirical-verification pass.

**Audio C7 contract:** Sprint D affects writer slot only, not the audio path. The C7 byte-identical canary holds against the prevailing baseline throughout Sprint D. (Reminder: the prevailing baseline at Sprint D open is still the v1.5 fixture pair `tests/fixtures/baseline_v1.5.wav` + `.sha256` because Sprint A has not yet captured the C5g post-MusicGen-wiring canonical b3sum. Sprint D does not need to wait on the Sprint A capture; period-LLM-routed writer slot does not feed MusicGen or audio.)

## Decisions log (cumulative)

| ID | Severity | Decision | Rationale | Lands at |
|---|---|---|---|---|

## Open findings (TRANSIENT -- deleted after each synthesis round)

<paste reviewer text here>

## Acceptance table

| # | Check | Target |
|--:|---|---|

## Code surface citations (verbatim, so reviewers need no repo access)

<populated during planning round-robin>

## Reviewer instructions (paste into Gemini/ChatGPT/etc for round-robins)

> You are reading Sprint D planning -- "Period LLM Wire-up" for a Windows-only, offline, RTX 5080 16 GB ComfyUI custom-node project. Scope: register `talkie-lm/talkie-1930-13b-it` in the model catalog, route the writer's creative slot through `_otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT` when a period LLM is selected, confirm VRAM fit under the 14.5 GB ceiling on the dual-slot config, and gate the whole thing behind a license-audit precondition (if `talkie-lm` license is not MIT-compatible the model goes to "research lane" with quarterly re-evaluation per the ROADMAP standing directive).
>
> Audit for: (1) load-bearing assumptions about model size, VRAM residency, and the dual-slot swap behavior post-S31 B4 (the loader is single-slot at runtime; `request_slot(model_id)` evicts the prior resident model before loading the next, so peak transient VRAM is `max(creative_size, technical_size)`, not their sum -- this property was confirmed at Sprint C's C1 audit); (2) the period-LLM-aware prompt routing -- does it cleanly switch between modern composer prompt and `OTR_PERIOD_SYSTEM_PROMPT` based on the creative-slot model identity, or does it need a more explicit `period_mode` toggle?; (3) license posture handling -- the "research lane" fallback path needs a clear contract for what gets disabled in pytest when the license audit fails; (4) audio C7 preservation -- Sprint D affects writer slot only, but cross-check that the period-LLM-routed composition does not accidentally drift the audio path; (5) test scope -- pytest-only structural pass; runtime quality belongs to Sprint A.
>
> Severity-tag every finding (HIGH / MEDIUM / LOW). List concrete additions, deletions, splits, or kills. Do not be nice. Do not summarize. Do not pad.

## Sprint-close handoff (filled at D-final)

- What shipped:
- What's broken / known issues:
- Post-state contract for next sprint:
- Audio C7 baseline state:
- Forbidden-sweep markers added:
- New tests added (count + categories):

---

## Previous Sprint Handoff (inherited from Sprint C close, 2026-05-15)

**Sprint A inherits the following gates / open work:**

- **Audio C7 baseline reset captures (DEFERRED from C5g per Option 2 path).** Sprint A's first runtime-verification commit must:
  1. Check out parent commit `c86db57` (C5f, the last commit before MusicGen wiring).
  2. Run the audio pipeline under `OTR_REGRESSION_RUNTIME=1` with the FIXED_SEEDS from `tests/test_audio_byte_identical.py`.
  3. Capture the output WAV's b3sum to `tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum` (forensic).
  4. Check out C5g (`600d0de`).
  5. Re-run the same audio pipeline under the same seeds.
  6. Capture the output WAV's b3sum to `tests/fixtures/audio_c7_baseline.wav.b3sum` (NEW canonical).
  7. Run the E-16 absent-brief isolation test: force `meta.story_brief_status="absent"`, render audio, assert byte-identical match to the pre-C5g forensic b3sum from step 3. This proves the audio shift is exclusively the mood-prefix code path.
  8. Commit both fixture files; the three runtime-gated tests in `tests/test_story_brief_musicgen_c5g.py::TestRuntimeOnly` flip from `skip` to `pass` automatically.
- **Empirical visual + audio render quality verification.** All FLUX env + portrait, LTX motion, HuMo lip-sync, and MusicGen audio output quality is unverified at Sprint C close. Sprint A runs each through ComfyUI Desktop with real GPU, captures sample renders, eyeballs / ear-balls quality.
- **Empirical LTX motion fidelity verification** (R-05). The C5e char-counting tests are structural proxy only.
- **Sprint G handoff (parked):**
  - `nodes/story_orchestrator.py` orphan-constant sweep. C2b audit confirmed `SCRIPT_SYSTEM_PROMPT` + `SCAFFOLDING_PREAMBLE` were dead (deleted at C2b). The 3000+ line file likely contains additional orphans from LPL / S31 B3 / S34 extraction sprints. Each candidate gets its own 8-search audit (getattr / glob-import / substring-content / alt-name / git log -S / writer-callsite / `__all__` / tests) before deletion.
  - `tests/test_musicgen_style_palette.py` rename to `tests/test_style_palette.py` (strip the misleading `musicgen` prefix; see archived SPRINT.md §C-final.3).
  - Dead-but-harmless `genre` parameter on `nodes/video_engine.py:_parse_hud_data` and `_write_story_treatment` (always passed `""` post-C3 since the `meta.visual_plan.genre` stamp is gone; Sprint G drops the parameter from the signatures).
  - `_load_canon_for_writer` in `nodes/story_orchestrator.py:2912` -- orphan function discovered during C2b cleanup; no production callers, only its own definition. Sprint G includes in the broad orphan sweep.
  - Comment-only era references at `nodes/story_orchestrator.py` lines 804, 874-877 -- cosmetic cleanup deferred from C2b's scope-tight directive.

**Sprint A audio C7 baseline state:** the canary at `tests/test_audio_byte_identical.py::TestAudioRegressionGate::test_audio_byte_identical_to_baseline` is gated on `OTR_REGRESSION_RUNTIME=1` and skipped throughout Sprint C. The existing fixture pair (`tests/fixtures/baseline_v1.5.wav` + `.sha256`) is a v1.5 baseline that pre-dates Sprint C entirely; Sprint A's first runtime commit re-baselines against the new MusicGen-wired audio path per the steps above.

**Sprint A forbidden-sweep markers added by Sprint C:** `\b1940s\b`, `1980s broadcast`, `1950s Americana`, `golden.age radio`, `\bOmni.Retro\b`, `\bOrson Welles\b`, `\bNorman Corwin\b`, `\bLucille Fletcher\b`, `\b_GENRE_BY_STYLE\b`, `\bmeta\.ltx_style_brief\b`, `\b_LTX_STYLE_BRIEF_PROMPT\b`, `\b_generate_ltx_style_brief\b`. All armed at zero runtime hits.

**New tests added by Sprint C (count + categories):**

- C2a era literals (visual): 6 tests
- C2b era literals (orchestrator): 14 tests (signature + AST + parametrized)
- C3 _GENRE_BY_STYLE deletion: 9 tests
- C3b meta.ltx_style_brief: 6 tests
- C4 VRAM envelope lock-in: 6 tests
- C5a1 reflection pure module: 23 tests (validation + repair + sentinel + AST)
- C5a2 writer wiring: 9 active + 4 runtime-gated skips
- C5b helpers: 21 tests
- C5c FLUX env + bookend: 9 tests
- C5d FLUX portraits: 6 tests
- C5e LTX motion: 8 tests
- C5f HuMo lip-sync: 6 tests
- C5g MusicGen wiring: 11 active + 3 runtime-gated skips

**Total new tests: ~127 active + 7 runtime-gated.** Final pytest count: 2276 passed, 17 skipped (C5g + C5a2 runtime-gated + pre-existing skips), 0 failed.

**Sprint D scope note (Sprint A vs Sprint D ordering):** Sprint A (audio C7 capture + empirical verification) and Sprint D (Period LLM wire-up) are independent: Sprint D touches the writer slot only, not the audio path. Operator may run them in either order or in parallel. If Sprint D opens first, the audio C7 canary holds against the existing v1.5 baseline throughout Sprint D's commits (period-LLM routing does not feed MusicGen).

---

## Standing Project Context

Carried forward from Sprint C close and the canonical project files (`SKILL.md` at repo root, `docs/CLAUDE.md`, `ROADMAP.md`). When this list and the upstream canonical files disagree, the upstream wins -- this section is a working-memory aid, not a fork.

### Hardware envelope

- **GPU:** RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- **VRAM ceiling:** `DEFAULT_VRAM_CEILING_GB = 14.5` (defined in `nodes/_otr_model_catalog.py`; pinned by `tests/test_vram_envelope_c4.py`).
- **Hard context cap:** `HARD_VRAM_CONTEXT_LIMIT = 8192` tokens (overridable via `OTR_HARD_VRAM_CONTEXT_LIMIT` env for hardware with more headroom).
- **Loader architecture:** single-slot. `request_slot(model_id)` calls `unload_llm()` to evict any prior resident model BEFORE loading the next. Peak transient VRAM during a model swap = `max(creative_size, technical_size)`, NOT their sum. Confirmed at Sprint C C1 audit per RR-A1 revised.
- **Default LLM:** `mistralai/Mistral-Nemo-Instruct-2407` (audio C7 byte-identical baseline; do not change without a deliberate baseline-reset sprint).
- **Python (Windows):** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (system `python` is not on PATH).
- **ComfyUI Desktop:** localhost:8000.

### Rules in force (every commit, every session)

- **No curse words.** Anywhere -- code, comments, log strings, commit messages, doc files.
- **No "dummy".** Use "placeholder", "stub", or a descriptive name.
- **No em-dashes in OTR Python source.** Use `--` instead. The 0x97 byte from cp1252-encoded em-dashes crashes UTF-8 decode in `tests/test_b7_forbidden_sweep.py::test_forbidden_sweep_runs_clean`. (Caught + fixed 2026-05-15 during Sprint C C3b.)
- **No-change-logs rule:** existing runtime log strings stay byte-stable. Existing `meta.*` attribute names stay byte-stable. New log lines added by a sprint follow the format conventions of neighboring lines; no surrounding existing line is modified.
- **Commit sizing rule:** if a commit would exceed one safe review-code-wire-pytest-regression-commit loop boundary, split it. Aim for <=0.75 day per commit.
- **Audio C7 baseline:** byte-identical pytest proxy holds at every commit boundary, except at explicit reset events specified in advance (with both pre and post b3sums captured). Sprint C deferred the C5g reset capture to Sprint A; Sprint D does not touch the audio path.
- **Forbidden-pattern sweep:** zero runtime hits at every commit boundary. Tokenize-classified docstring / comment suppression for forensic mentions.
- **Pytest-only acceptance:** no ComfyUI Desktop runtime gates inside the sprint. Runtime quality verification is its own sprint downstream.
- **Git push:** Desktop Commander cmd shell. NEVER PowerShell for git (known hang issue, S30 B1b root cause).
- **One git push attempt max** -- then hand Jeffrey a PowerShell block.
- **Verify after every push:** local HEAD == origin HEAD, no 0-byte files, no BOM, AST parse, all node classes registered in `__init__.py`, workflow JSONs valid and wired to current node surfaces.

### Branch + shipping

- All v2 work on `v2.0-alpha`. Do not touch `main`.
- Only Jeffrey merges to `main` and tags releases.
- The Sprint C branch `sprint-c-story-brief-v2` is pushed to origin and ready for merge or further inspection at Jeffrey's discretion.

### Active per-phase prompt surfaces (Sprint D blast radius)

The writer's prompt surfaces are spread across per-phase modules (the LPL extraction sprint `eec4718` moved them out of `nodes/story_orchestrator.py`). Sprint D's prompt-routing change touches the creative slot's consumer of these:

- `nodes/_otr_outline.py:_SYSTEM_PROMPT` (line 411).
- `nodes/_otr_line_composer.py:_SYSTEM_PROMPT` (line 790), `_POLISH_SYSTEM_PROMPT_CHARACTER` (line 1077), `_POLISH_SYSTEM_PROMPT_ANNOUNCER` (line 1099).
- `nodes/_otr_ledger_reviewer.py:_AUDITOR_SYSTEM_PROMPT` (line 326), `_DOCTOR_SYSTEM_PROMPT` (line 634).
- `nodes/_otr_period_prompts.py:OTR_PERIOD_SYSTEM_PROMPT` (line 37) -- exported via `__all__`. This is the Sprint D routing target.

### v2.1+ watch-list (deferred decisions, parked)

- **`artokun/comfyui-mcp` evaluation OR custom `/mcp-builder` comfyui-runner.** Defer until after v1.9 ships and real iteration friction is measured. Until then: manual ComfyUI Desktop loading is the workflow. Don't build harness infrastructure speculatively.
- **LTX 2.3 LipDub IC-LoRA.** Research addendum in `ROADMAP.md`; deferred indefinitely. Adoption stays a Sprint A acceptance bullet OR later forward feature work at Jeffrey's discretion. The five issues §2 of the ROADMAP addendum names (audio-path non-passthrough, transcription prompt requirement, ingest-socket VRAM peak, single-speaker only, motion-stack incompatibility) must be folded in before any LipDub adoption commit.

### Bug Bible regression baseline

- `pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v` must return **23 passed / 1 skipped / 2 xfailed** at every commit boundary. Baseline held end-to-end through Sprint C.

### Repo-wide sources of truth

- `SKILL.md` at repo root -- workflow rules for any AI assistant.
- `CLAUDE.md` at repo root -- Jeffrey's standing rules.
- `ROADMAP.md` at repo root -- canonical going-forward plan.
- `docs/closed-sprints/` -- archived SPRINT.md per closed sprint, read-only.
- `docs/BUG_LOG.md` -- live defect tracking (per-bug entries; promote to Bible repo when stable).
