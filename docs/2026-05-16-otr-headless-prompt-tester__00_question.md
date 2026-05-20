# New-convo brief -- OTR headless LLM prompt tester

Paste this entire file into a fresh Cowork session. Do not paste it into the existing Sprint H session.

---

## Who you are

You are a Claude agent in Cowork mode helping Jeffrey Brick. The user is autistic, has chronic foot pain limiting walking by 50%, gets bullied a lot. He is direct, no fluff, no emoji, ASCII only, no em-dashes (use `--`), never use the word "dummy" (use `placeholder` or `stub`). Project rules at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\CLAUDE.md`. Read it first.

## Project context

- **Repo:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
- **Branch:** `sprint-d-period-llm @ 5b0d0ba` (baseline). Sprint H lean prompt work is happening in parallel on this branch or a new `sprint-h-lean-prompts` cut from `v2.0-alpha @ aad568c0`.
- **Python:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (system `python` is NOT on PATH)
- **HF_HOME:** `C:\ComfyUI-Models` -- Gemma-4-E2B-it cached at `C:\ComfyUI-Models\hub\models--google--gemma-4-E2B-it\snapshots\<rev>\`
- **GPU:** RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120
- **VRAM ceiling:** 14.5 GB peak. Single GPU. No cloud, no API keys, no paid services.

## The task

Build ONE standalone Python script that runs every OTR LLM prompt headlessly against the local Gemma-4-E2B-it model, validates outputs against their schemas, and prints pass rate per prompt. Goal: iterate on prompt rewrites without spinning up ComfyUI Desktop.

**Deliverable:**

```
scripts/test_all_prompts.py
```

Invocation:

```powershell
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

## The 12 prompts to exercise

All live under `nodes/`. Each has an existing pure-Python entrypoint that takes a `generate_fn` callable plus typed inputs:

| # | Prompt | Module | Entrypoint | Schema validator |
|---|--------|--------|------------|------------------|
| 1 | style_inventor | `nodes/_otr_style_picker.py` | `_run_inventor` (private, but exercise via `pick_style`) | regex DESCRIPTOR_RE |
| 2 | style_chooser | `nodes/_otr_style_picker.py` | `_run_chooser` (private, via `pick_style`) | enum exact-match |
| 3 | outline | `nodes/_otr_outline.py` | `generate_outline` | pydantic `Outline` + `validate_outline_against_budget` |
| 4 | cast_lock | `nodes/_otr_casting.py` | `cast_one_character` (call per slot) | pydantic `CastingResponse` |
| 5 | cast_validator | same as #4, attempt 3 repair | same | same |
| 6 | line_composer | `nodes/_otr_line_composer.py` | `compose_line` | regex `_PREFIX_*` + length cap |
| 7 | polish_character | `nodes/_otr_line_composer.py` | `polish_line(speaker_role="character")` | `is_polish_refusal` check + length |
| 8 | polish_announcer | `nodes/_otr_line_composer.py` | `polish_line(speaker_role="announcer")` | same |
| 9 | reflection | `nodes/_otr_story_brief.py` | `run_story_brief_reflection` | `StoryBriefModel` + 8-key validator |
| 10 | cast_audit | `nodes/_otr_ledger_reviewer.py` | `audit_cast_contract` | pydantic `PreAuditReport` |
| 11 | script_doctor | `nodes/_otr_ledger_reviewer.py` | `run_script_doctor` | pydantic `ScriptDoctorReport` |
| 12 | news_interpreter | `nodes/news_interpreter.py` | `build_news_briefs` | GBNF + pydantic `NewsBriefs` + v1/v2/v3 validators |

Each entrypoint already wraps prompt construction + retry ladder + schema validation. The tester just supplies a `generate_fn` and the typed inputs, then catches the entrypoint's success-or-`*FailedError`.

## How `generate_fn` works

Every OTR LLM entrypoint takes a callable with this signature:

```python
def generate_fn(messages: list[dict], *, temperature: float, max_new_tokens: int, stop: list[str] | None = None) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    Returns raw model output as a string."""
```

The tester builds ONE `generate_fn` backed by a single transformers pipeline loaded once at script start. Re-use it for all 12 prompts to avoid 12 model loads.

Reference implementation: `nodes/_otr_model_loader.py` already has `_build_truncating_generate_fn` -- import and reuse if possible. If not, build a minimal version inline.

## Test inputs (fixtures)

Build a tiny `scripts/test_prompts_fixtures.py` module with one fixture function per prompt that returns realistic typed inputs. Examples:

- **outline**: an `OutlineRequest` with a 2-character cast, 200-word target, the included sample science article string, a default `EpisodeBudget`.
- **cast_lock**: a `CastSlot` for one character with a brief description.
- **line_composer**: a `LineRequest` with a current beat + last 3 lines.
- **reflection**: a small fake ledger dict with 4-6 lines.
- **news_interpreter**: the included sample science article string.

Pull fixtures from existing test files where possible (`tests/test_*.py`). Do NOT recreate fixtures from scratch if `tests/` already has them.

## Output format

```
$ python scripts/test_all_prompts.py --calls 10

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

Summary: 102 / 120 PASS (85.0%) across 12 prompts on google/gemma-4-E2B-it
Total wall time: 3m 47s
Peak VRAM: 5.2 GB

Failing prompts: outline, line_composer, reflection, script_doctor
Run with --verbose to see first failing output per prompt.
```

## Constraints (hard)

- **One model load.** Load Gemma-4-E2B-it once via transformers, share the pipeline across all 12 prompts. 12 reloads = 12x setup time.
- **No ComfyUI imports.** The script must run from a plain venv without ComfyUI's node loader. `from nodes._otr_outline import generate_outline` is fine; importing from `comfy.*` or `folder_paths` is not.
- **VRAM ceiling 14.5 GB.** Stop and warn if exceeded.
- **Windows paths.** Use `pathlib.Path`, no Unix shell tricks.
- **ASCII only in output and code.** No em-dashes -- use `--`.
- **No silent fallbacks.** If a prompt's entrypoint throws an unexpected exception (not `*FailedError`), surface it loudly.
- **Repeatable seed.** Default seed 42; allow `--seed N` override. Same seed + same fixture = same generation for sanity diffs.

## Out of scope

- Comparing multiple models (just Gemma-4-E2B-it for now)
- Side-by-side prompt-A-vs-prompt-B (separate sprint)
- ComfyUI graph testing
- Audio / video / TTS testing
- Network calls of any kind

## Success criteria

1. Script runs end-to-end on Jeffrey's box in under 10 minutes for `--calls 10`.
2. Reports pass rate per prompt + summary.
3. Catches a deliberately broken prompt (test by temporarily breaking `_otr_outline._SYSTEM_PROMPT` -- pass rate should drop to 0 for outline, others unaffected).
4. JSON report includes per-prompt: pass_count, fail_count, fail_reasons (list of strings), avg_wall_time_s, peak_vram_gb.

## When done

Hand back to Jeffrey:
1. Path to `scripts/test_all_prompts.py`
2. Path to `scripts/test_prompts_fixtures.py`
3. One PowerShell block to run it
4. Sample output from a real run

Do NOT commit to git. Jeffrey commits after he confirms it works. Leave the working tree clean of any other changes -- the bug fixer in a parallel session is mid-flight on `nodes/OTR_LedgerScriptWriter.py` and several other writer-adjacent files; do not touch those.

## First moves

1. Read `CLAUDE.md` (project rules)
2. Read `nodes/_otr_model_loader.py` to see existing `generate_fn` factory
3. Read 2-3 fixture-heavy test files: `tests/test_otr_style_picker.py`, `tests/test_otr_casting.py`, `tests/test_news_interpreter.py`
4. Halt with a one-page plan + estimated lines per file before writing code
