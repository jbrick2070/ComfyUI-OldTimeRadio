# Sprint H Go-Forward Plan -- Headless Prompt Tester + Coworker MCP

**Branch:** v2.0-alpha @ 1b3569e
**Date:** 2026-05-17
**Status:** Pre-H2. Build tester, gate every rewrite, then continue.

---

## 0. Operating split

| Tool | Job | Boundary |
|------|-----|----------|
| `artokun/comfyui-mcp` | Claude Code coworker for ComfyUI runtime: `/comfy:debug`, log inspection, workflow validation, node/model discovery, VRAM/system checks | Read/debug only. No mutation, no installs, no downloads until debug loop is proven. Pin version. |
| `scripts/test_all_prompts.py` | Sprint H prompt regression gate: schema pass/fail, retry telemetry, quality counters, JSON report | No ComfyUI server. No graph. No audio/video. No network. No MCP. |

Do not merge these.

---

## 1. File-by-file implementation plan

### 1.1 `scripts/test_prompt_import_isolation.py` (BUILD FIRST)

Subprocess guard. Runs before tester proper. Fails the whole gate if writer-adjacent modules transitively pull `comfy` or `folder_paths`.

- Spawns subprocess with import sentinel: `sys.modules["comfy"] = _Forbidden()` and same for `folder_paths`, where `_Forbidden.__getattr__` raises `ImportIsolationError`.
- Imports each of the 12 prompt entrypoint modules in turn.
- Any access to a forbidden module raises and the subprocess exits nonzero.
- Tester refuses to run if this guard fails.

Why subprocess: static grep misses transitive imports. In-process stubbing pollutes the tester's own namespace.

### 1.2 `scripts/test_prompts_fixtures.py`

One function per prompt returning typed inputs. Realistic by default, minimal on flag.

Shared canonical pieces (build once, reuse):
- `SAMPLE_ARTICLE` -- one ~600-word science/news article with named entities (people, places, orgs, dates) for NER surfacing checks
- `CAST_3` -- 3-character cast with stable names, voices, role tags
- `EPISODE_BUDGET_DEFAULT` -- target 250 words
- `BEAT_CURRENT` + `LAST_3_LINES` -- mid-episode state
- `LEDGER_SMALL` -- realistic but small
- `REFLECTION_INPUT` -- atmosphere, lighting, motion, music terms present so 8-key validator can actually fire

Per-prompt fixture functions return the exact typed object the entrypoint expects. Borrow from `tests/` only for GBNF-shape prompts where shape matters more than scale.

Every fixture function exposes `fixture_hash()` returning a stable sha256 of the inputs.

### 1.3 `scripts/test_all_prompts.py`

Single entry point. Phases:

1. Parse CLI.
2. Run import isolation guard via subprocess. Abort on fail.
3. Load model once. Build `generate_fn` wrapper that resets RNG and clears KV cache before each call.
4. For each prompt in scope (`--only` or all 12):
   - Load fixture (realistic|minimal).
   - For `call_idx` in `range(--calls)`:
     - Reset seed = `seed + call_idx`.
     - Empty cache, poll VRAM. Abort prompt if `--max-vram-gb` exceeded.
     - Call entrypoint with `generate_fn` + fixture.
     - Catch success or `*FailedError`. Bucket the failure.
     - Compute quality counters on raw output.
     - Append per-call record to JSONL if `--jsonl` set.
   - Aggregate prompt summary.
5. Write JSON report with stable key order.
6. Print summary table + the C7 reminder.

Module layout inside the script (keep flat, one file):
- `_isolation` -- launches 1.1, parses result
- `_loader` -- single model load, reset-per-call generate_fn
- `_runners` -- 12 thin per-prompt runners that wire fixture -> entrypoint -> bucket
- `_counters` -- word count, trigram ratio, NER surfacing, refusal detect, hashes
- `_report` -- JSON writer with stable key order, summary printer
- `_provenance` -- git commit, model id/revision, torch, CUDA, Python, seed, gen params

### 1.4 `reports/prompt_runs/`

Output folder. Created on first run. Convention: `<prompt>_<sprint>.json` (e.g. `outline_h2.json`) for gate runs, plus full-run reports as `full_<utc_iso>.json`.

---

## 2. Smallest viable version (ship this first, ~2.5 hr)

Cut surface area. Defer non-blockers.

**Includes:**
- 1.1 import isolation guard
- 1.2 fixtures for: outline, cast_lock, line_composer, polish_character, reflection (5 of 12 -- the ones H2-H4 touch)
- 1.3 with: model load, generate_fn with reset, 5 runners, schema pass/fail, retry_count, raw_output_sha256, prompt_hash, fixture_hash, JSON out
- CLI: `--model`, `--calls`, `--only`, `--out`, `--seed`, `--fixture`, `--max-vram-gb`
- Full 12 failure buckets defined (even if only 5 prompts wired) -- avoids retrofitting
- Provenance recording
- C7 reminder line

**Defers to v2 (after smallest version proves out):**
- 7 remaining prompt runners (style_inventor, style_chooser, cast_validator, polish_announcer, cast_audit, script_doctor, news_interpreter)
- `--verbose`, `--jsonl`, `--fail-fast`
- Quality counters: word_count_ratio, unique_trigram_ratio, named_entity_surfacing_ratio, refusal_detected
- `--dump-outputs`, `--compare`

This lets H2 (a single prompt rewrite) gate immediately. Counters and the other 7 runners land before H3.

---

## 3. Pre-build verification (~80 min, do these before any code)

1. **Model choice.** Confirm E2B vs E4B against 16GB Suitcase rule. If E4B is correct, change default before code.
2. **KV reset tie-breaker.** Run outline x10 on shared load. Diff output 1 vs 10 by trigram overlap. Below 95% confirms KV reset is non-negotiable. (If above 95%, RNG reset alone may suffice -- still do both.)
3. **news_interpreter grammar path.** Pydantic-only or still GBNF? Transformers cannot enforce GBNF. If GBNF: either second backend (+2 hr) or drop news_interpreter from this tester and gate it separately.
4. **Gemma chat template.** Some transformers versions reject bare `system` role. Verify via Context7. Convert to leading `user` turn if needed.
5. **Audio contradiction.** Original brief §2.1 vs §6.3. Resolution: audio is OUT of tester. C7 stays as its own gate (see §6).

---

## 4. Build order (4 hr target, 7 hr ceiling)

1. Run §3 verification (80 min).
2. `test_prompt_import_isolation.py` + run against current `nodes/_otr_*` modules. Fix any leaks before continuing.
3. `test_prompts_fixtures.py` -- canonical pieces + 5 fixture functions.
4. `_loader` -- single load, reset-per-call wrapper. Run outline x10, confirm tie-breaker result.
5. `_runners` -- 5 prompts wired to fixtures + entrypoints.
6. `_counters` minimal version (hashes + retry_count only for v1).
7. `_report` + `_provenance`.
8. CLI surface (v1 subset).
9. Deliberate-fault sanity check: break `_otr_outline._SYSTEM_PROMPT`, confirm outline -> 0/10, other 4 prompts unaffected.
10. Run full pass on current `v2.0-alpha @ 1b3569e` to establish baseline numbers.

---

## 5. Acceptance tests (the gate for H2-H8)

Each prompt rewrite must pass all four before merge:

### 5.1 Tester gate (this script)

```
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\test_all_prompts.py ^
  --only <prompt_name> --calls 10 --fixture realistic ^
  --out reports\prompt_runs\<prompt>_h<N>.json
```

Required: pass rate equal to or higher than the H0/H1 baseline for that prompt. No new failure bucket appears. retry_count mean does not increase by more than 1.

### 5.2 Full tester gate (before final H merge)

```
& ...\python.exe scripts\test_all_prompts.py --calls 10 --fixture realistic ^
  --out reports\prompt_runs\full_h<N>.json
```

Required: overall pass rate within 5 points of baseline. No prompt drops below 70%.

### 5.3 C7 audio byte-identity regression

Existing C7 command. Required: byte-identical default-config audio output, or surfaced diff + explicit new baseline stamp.

### 5.4 Existing unit tests + Bug Bible

Required: 211 unit pass, Bug Bible 23/0 maintained.

### 5.5 Import isolation

`test_prompt_import_isolation.py` exits zero. Any new prompt module added since last run is included.

---

## 6. Regression / test / retest discipline

**Per prompt rewrite (H2 through H8, each prompt):**
1. Write the rewrite.
2. Run 5.5 (import isolation). Fix leaks. Repeat until clean.
3. Run 5.1 (tester gate, single prompt). If fail: read failure bucket, fix prompt, re-run. Do not proceed until clean.
4. Run 5.4 (unit + Bug Bible). If fail: fix.
5. Commit with prompt rewrite + the gate report JSON committed to `reports/prompt_runs/`.

**Before final H merge:**
6. Run 5.2 (full tester gate, all 12 prompts).
7. Run 5.3 (C7 audio).
8. Run 5.4 (unit + Bug Bible) full.
9. Run 5.5 (import isolation).
10. Diff `reports/prompt_runs/full_h<N>.json` against `reports/prompt_runs/full_h0.json`. Investigate any per-prompt drop greater than 10 points or new failure bucket.

**Cadence rule:** Do not write two rewrites in a row without running 5.1 between them. The whole point of building this before H2 is the 30-second feedback loop. Honor it.

---

## 7. Determinism language (commit verbatim to script docstring)

> Same seed and same fixture produce approximately repeatable pass/fail behavior on this hardware (RTX 5080 Laptop, Blackwell sm_120, torch 2.10, CUDA 13). Raw output may drift due to atomic-add nondeterminism, bf16 rounding, and cuDNN nondeterministic algorithms. Schema-equivalence under same seed is the contract. Bit-identity is not.

Recorded provenance per run: git commit, model id, model revision, torch version, CUDA version, Python version, seed, generation params, prompt hash, fixture hash, raw output sha256.

---

## 8. Blockers and bad assumptions

1. **news_interpreter GBNF.** If still GBNF, transformers cannot enforce it. Resolve in §3 verification or this prompt silently degrades the tester's signal. Worst case: drop from tester, gate it separately.
2. **Gemma `system` role.** Recent transformers versions vary on this. If unverified, half your runs may fail for the wrong reason. Verify before §4 step 4.
3. **KV cache growth across 120 calls.** Single-call peak is fine on 14.5 GB. The risk is fragmentation/growth across the run. `_loader` must clear cache + check `memory_stats()` every call, not just at start.
4. **E2B vs E4B mismatch.** The brief says E2B; the 16GB Suitcase rule may require E4B. Validating on the wrong model under-reports VRAM pressure and may hide adherence flaws E4B would expose.
5. **Realistic fixtures will lower pass rates.** The H0/H1 baseline set with realistic fixtures will be lower than what tests/* shows. Do not panic. Establish the baseline first (§4 step 10) before rewriting H2.
6. **NER surfacing counter.** If wired later, do not pull spaCy or HF NER models. Use a simple regex/keyword check against SAMPLE_ARTICLE's known entity list.
7. **MCP version drift.** `npx artokun/comfyui-mcp@latest` will surprise mid-sprint. Pin version in Claude Code config.
8. **Acceptance gate enforcement is manual.** Pre-commit hook or merge checklist; otherwise the gate exists on paper only.
9. **C7 audio is genuinely outside this script.** Keep split. C7 is a binary diff on rendered audio; this tester is text validation.
10. **Solo + chronic foot pain.** Every feature added is permanent maintenance. Resist scope creep into v2 until v1 has gated at least one real H2 rewrite.

---

## 9. Out of scope (do not add)

```
--baseline-commit checkout-and-run
A/B
multi-model benchmarking
cloud model calls
ComfyUI graph execution
audio/video generation
embedding similarity
full 12-prompt fault matrix
MCP dependency inside the tester
```

---

## 10. Riskiest assumption

The 4-hour smallest-version build estimate holds. If news_interpreter forces llama-cpp-python add 2 hr. If Gemma chat template breaks shape add 1 hr. Hard ceiling: 7 hr. Past that, stop and reassess scope before continuing into H2. The tester exists to protect Sprint H velocity, not become Sprint H.
