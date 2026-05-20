# Round-robin -- OTR headless LLM prompt tester design review

**Branch:** v2.0-alpha @ 1b3569e (H1 lean outline rewrite landed 2026-05-17)
**Author:** Jeffrey Brick
**Date:** 2026-05-17
**Reviewers:** ChatGPT (gpt-4.1), Gemini (gemini-2.5-pro)
**Synthesis:** Claude
**Format:** lean -- 250 lines or under in final synthesis
**Constraints:** ASCII only, no em-dashes (use `--`), no "dummy" word (use "placeholder" or "stub"), default-config audio C7 byte-identity is non-negotiable

---

## 1. The question, in one paragraph

Sprint H is rewriting all 12 OTR LLM prompts lean (Option F: normalize for smallest-model-friendly size). H1 just shipped: outline _SYSTEM_PROMPT compressed from ~520 to ~310 tokens on v2.0-alpha at 1b3569e. Before H2 starts, Jeffrey wants a headless validation loop so he can iterate on prompt rewrites without spinning up ComfyUI Desktop for each. The design under review is in §2 below. Stress-test that design and recommend one path forward. The audio C7 byte-identity baseline is non-negotiable; everything else is on the table.

---

## 2. The design under review

### 2.1 Goal

Build ONE standalone Python script that runs every OTR LLM prompt headlessly against the local Gemma-4-E2B-it model, validates outputs against their schemas, and prints pass rate per prompt. No ComfyUI graph, no audio, no video.

### 2.2 Deliverable

```
scripts/test_all_prompts.py
```

Invocation:

```
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\test_all_prompts.py
```

Optional flags:

```
--model      <hf_repo_id>      default: google/gemma-4-E2B-it
--calls      <int>             default: 10 (per prompt)
--only       <name,name>       run only listed prompts
--verbose                      dump first failing output per prompt
--out        <path>            write JSON report to path
```

### 2.3 The 12 prompts to exercise

All live under `nodes/`. Each has an existing pure-Python entrypoint that takes a `generate_fn` callable plus typed inputs.

| # | Prompt | Module | Entrypoint | Schema validator |
|---|--------|--------|------------|------------------|
| 1 | style_inventor | `nodes/_otr_style_picker.py` | `_run_inventor` (via `pick_style`) | regex DESCRIPTOR_RE |
| 2 | style_chooser | `nodes/_otr_style_picker.py` | `_run_chooser` (via `pick_style`) | enum exact-match |
| 3 | outline | `nodes/_otr_outline.py` | `generate_outline` | pydantic `Outline` + `validate_outline_against_budget` |
| 4 | cast_lock | `nodes/_otr_casting.py` | `cast_one_character` | pydantic `CastingResponse` |
| 5 | cast_validator | same as #4, attempt 3 repair | same | same |
| 6 | line_composer | `nodes/_otr_line_composer.py` | `compose_line` | regex `_PREFIX_*` + length cap |
| 7 | polish_character | `nodes/_otr_line_composer.py` | `polish_line(role=character)` | `is_polish_refusal` + length |
| 8 | polish_announcer | `nodes/_otr_line_composer.py` | `polish_line(role=announcer)` | same |
| 9 | reflection | `nodes/_otr_story_brief.py` | `run_story_brief_reflection` | `StoryBriefModel` + 8-key validator |
| 10 | cast_audit | `nodes/_otr_ledger_reviewer.py` | `audit_cast_contract` | pydantic `PreAuditReport` |
| 11 | script_doctor | `nodes/_otr_ledger_reviewer.py` | `run_script_doctor` | pydantic `ScriptDoctorReport` |
| 12 | news_interpreter | `nodes/news_interpreter.py` | `build_news_briefs` | GBNF + pydantic `NewsBriefs` + v1/v2/v3 validators |

Each entrypoint already wraps prompt construction, retry ladder, and schema validation. The tester supplies a `generate_fn` and typed inputs, then catches success-or-`*FailedError`.

### 2.4 generate_fn signature

```python
def generate_fn(messages: list[dict], *, temperature: float,
                max_new_tokens: int, stop: list[str] | None = None) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    Returns raw model output as a string."""
```

The tester builds ONE `generate_fn` backed by a single transformers pipeline loaded once. Re-use across all 12 prompts to avoid 12 model loads. Reference impl: `nodes/_otr_model_loader.py` already has `_build_truncating_generate_fn`.

### 2.5 Fixtures

Tiny `scripts/test_prompts_fixtures.py` with one function per prompt returning realistic typed inputs. Examples: an `OutlineRequest` with a 2-character cast + 200-word target + sample science article + default `EpisodeBudget`; a `CastSlot` for one character; a `LineRequest` with current beat + last 3 lines; a small fake ledger; the sample science article. Borrow from `tests/test_*.py` where possible.

### 2.6 Output

```
[1/12] style_inventor      10/10 PASS  (avg 1.2s/call)
[2/12] style_chooser       10/10 PASS  (avg 0.4s/call)
[3/12] outline              8/10 PASS  (2 schema fails, 1 budget fail)
[4/12] cast_lock           10/10 PASS  (avg 1.8s/call)
[5/12] cast_validator       N/A        (only triggers on cast_lock fail)
[6/12] line_composer        9/10 PASS  (1 length-cap exceed)
[7/12] polish_character    10/10 PASS
[8/12] polish_announcer    10/10 PASS
[9/12] reflection           7/10 PASS  (3 rejection_class: missing_lighting_terms)
[10/12] cast_audit         10/10 PASS
[11/12] script_doctor       9/10 PASS  (1 edit cap exceed)
[12/12] news_interpreter   10/10 PASS

Summary: 102 / 120 PASS (85.0%) on google/gemma-4-E2B-it
Total wall time: 3m 47s
Peak VRAM: 5.2 GB
```

### 2.7 Constraints (hard)

- One model load shared across all 12 prompts.
- No ComfyUI imports. No `comfy.*`, no `folder_paths`. `from nodes._otr_*` is fine.
- VRAM ceiling 14.5 GB. Stop and warn if exceeded.
- Windows paths only (pathlib.Path).
- ASCII only in output and code.
- No silent fallbacks. Unexpected exceptions surface loudly.
- Repeatable seed. Default seed 42; `--seed N` override. Same seed + same fixture = same generation.

### 2.8 Success criteria

1. Runs end-to-end in under 10 minutes for `--calls 10`.
2. Reports pass rate per prompt + summary.
3. Catches a deliberately-broken prompt (sanity check: break `_otr_outline._SYSTEM_PROMPT`, expect outline pass rate -> 0 with others unaffected).
4. JSON report per-prompt: `pass_count`, `fail_count`, `fail_reasons`, `avg_wall_time_s`, `peak_vram_gb`.

### 2.9 Out of scope

- Comparing multiple models (just Gemma-4-E2B-it).
- Side-by-side prompt-A-vs-prompt-B (separate sprint).
- ComfyUI graph testing.
- Audio / video / TTS.
- Network calls of any kind.

---

## 3. Design choices to argue for or against

### 3.1 Single model load shared across 12 prompts

Pros: 11x faster iteration. Cons: shared module state can leak -- cached KV from prior calls, sampling RNG drift even with seed reset, tensor allocator fragmentation that produces slightly different bf16 rounding on call 12 vs call 1. Pass-rate on call 12 might not reflect what an isolated production call would produce.

**Argue:** Is shared loading safe enough for validator-grade pass/fail, or should each prompt get full RNG + cache reset before the call?

### 3.2 Pass rate as the only metric

Pass rate measures shape, not quality. A lean rewrite could pass schema 10/10 while producing visibly worse creative output (flatter dialogue, weaker premise extrapolation).

**Argue:** What additional signal is cheap to add and meaningfully distinguishes "schema-clean but creatively dead" from "schema-clean and richer"? Candidates: (a) output diversity (n-gram overlap across calls), (b) word-count variance vs target, (c) cosine similarity of premise embeddings against a baseline set, (d) named-entity surfacing ratio.

### 3.3 Fixture sourcing

Existing tests/* fixtures are minimal-by-design (1-2 character casts, 50-100 word targets). Real episodes run 3-character casts and 250+ word targets. A lean prompt could pass on minimal fixtures and fail on real shape.

**Argue:** Should the tester ship its OWN realistic fixtures keyed to a canonical sample science article + 3-character cast + 250-word target, borrowing tests/* only for GBNF / structural-validator prompts where shape matters more than scale?

### 3.4 ComfyUI-import isolation

Several writer-adjacent helpers may transitively pull `folder_paths` or `comfy.*` through chains the static check won't catch.

**Argue:** monkeypatch in conftest? sys.modules pre-import shim? `--no-comfy` import guard? Pick one and defend.

### 3.5 Deliberate-fault sanity check

Brief says: test by breaking `_otr_outline._SYSTEM_PROMPT`, expect outline -> 0. The other 11 prompts could silently regress without notice because they were never fault-confirmed.

**Argue:** Should each of the 12 have a small deliberate-fault permutation as a baseline ("fault matrix"), or is per-prompt fault confirmation overkill for a dev-loop tool?

### 3.6 A-vs-B comparison deferred

Brief parks side-by-side prompt-A-vs-prompt-B as "separate sprint". But the tester's primary purpose IS validating Sprint H rewrites against the H0 baseline.

**Argue:** A/B in scope from day 1, or is "run twice with different prompt files and eyeball" enough? If A/B is in: `--baseline-commit <sha>` flag with checkout-and-run capability.

### 3.7 Seed determinism on Blackwell

RTX 5080 Blackwell sm_120 + torch 2.10 + CUDA 13. cuDNN nondeterministic algorithms, atomic-add nondeterminism in attention, bf16 rounding may all introduce drift.

**Argue:** Bit-identical achievable? If not, what tolerance ("first 50 tokens match", "schema-equivalent under N-gram diff")? The "same seed = same gen" claim either needs `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + cuDNN det flags, or it needs to be downgraded to "approximately repeatable".

---

## 4. The sequencing question

**Should `scripts/test_all_prompts.py` be built BEFORE H2 starts (each prompt rewrite ships behind a green pass-rate gate), or AFTER H2-H8 are done (the tester validates the whole rewrite batch)?**

For "before":
- Each rewrite gets a 30-second feedback loop. Catches regression early.
- Matches Sprint H's H4 ("Bug Bible regression after each rewrite, no batching").

For "after":
- Tester effort is its own ~3-4 hour sprint that competes with H2 rewrites.
- OLD prompts already work (211 unit pass, Bug Bible 23/0). There's no pass-rate to BEAT, just to MAINTAIN.

Pick a side. One paragraph.

---

## 5. What we want from the round-robin

1. Rank the 7 design choices in §3 by risk (1 = most likely to derail). Recommend a fix for the top 3.
2. Answer §4. One side, one paragraph.
3. Name anything material the brief MISSES that isn't in §3 (a check, a prompt, a constraint).
4. If you disagree on §4 or a §3 item, name the specific datum that would break the tie (e.g. "measure n-gram overlap on 5 calls of OLD outline and 5 of NEW; if drop > X, A/B is mandatory day 1").
5. Halt with a single recommendation + the riskiest assumption you are making.

---

## 6. Decision criteria

1. **Quality signal.** Tester output must let Jeffrey ship H2-H8 with confidence. Schema-clean is necessary, may not be sufficient.
2. **Sprint H velocity.** Tester effort that doubles Sprint H is not worth it. Lean trumps perfect.
3. **C7 audio non-negotiability.** Default-config audio bytes remain identical post-Sprint-H, or the tester surfaces the diff and stamps the new baseline cleanly.
4. **One-person team.** Jeffrey is solo with chronic foot pain limiting weekly hours. Every additional feature is permanent maintenance.
5. **Local-only.** RTX 5080 Laptop, 16 GB VRAM, 14.5 GB peak. No cloud, no API. Tester must run on the box.

---

## 7. How to share results back (Jeffrey ops only)

1. ChatGPT pass: paste this file into ChatGPT or run `scripts/_consult_openai.py`. Save response to `docs/2026-05-17-headless-tester-rr__01_chatgpt.md`.
2. Gemini pass: paste this file + ChatGPT response, ask Gemini to agree / correct / add. Save to `docs/2026-05-17-headless-tester-rr__02_gemini.md`.
3. Loop §2 if Gemini disagrees materially (`__02b_gemini.md`).
4. Drop both transcripts in `docs/` and tell Claude they're there. Claude reads them, writes `docs/2026-05-17-headless-tester-rr__04_synthesis.md`, and decides.
