# Round-robin synthesis -- OTR headless LLM prompt tester

**Date:** 2026-05-17
**Source:** pasted into Cowork session 2026-05-17 by Jeffrey (provenance: round-robin run external to this session)
**Question file:** `docs/2026-05-17-headless-tester-rr.md`
**Decision status:** locked per below; pending Jeffrey confirmation on §6 verification items

---

## 5. Hard requirements

- **Realistic fixtures default.** Canonical set keyed to one sample science article + 3-char cast + 250-word target. `minimal` flag pulls from `tests/*` only for GBNF-shape prompts.
- **Single model load with state reset.** One pipeline. Before every call: `torch.cuda.empty_cache()` + `torch.manual_seed(seed + call_idx)`.
- **Determinism downgraded.** "Schema-equivalent under same seed" -- not bit-identical. Add `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` best-effort.
- **ComfyUI isolation.** `sys.modules` stub for `folder_paths` and `comfy.*` at script entry. No `comfy.*` or `folder_paths` imports survive.
- **VRAM monitoring.** `torch.cuda.memory_stats()` polling, not nvidia-smi shellout. Watch KV growth across 120 calls, not just single-call peak. Hard ceiling 14.5 GB.
- **Raw output capture.** `--dump-outputs` writes `outputs/<prompt>/<call_idx>.txt`. Eyeball 5 per prompt to catch creatively-dead-but-schema-clean.
- **Diff mode.** `--compare` reports per-prompt deltas between runs. Stable key order in JSON. Without this, regression vs noise is indistinguishable.
- **Sanity check.** Deliberate fault on `_otr_outline._SYSTEM_PROMPT` -> outline pass rate 0, others unaffected.
- **ASCII output, Windows paths, no silent fallbacks.** Unexpected exceptions surface loudly.

## 6. Pre-build verification (~80 min, do first)

1. **Model choice.** E2B vs E4B against the 16GB Suitcase rule. If E4B is correct, switch the default before writing the script.
2. **KV reset tie-breaker.** Run outline x10 on a shared load, diff output 1 vs output 10 by n-gram overlap. Below 95% confirms KV reset is non-negotiable.
3. **news_interpreter grammar path.** Pydantic-only or still GBNF? Transformers pipeline cannot enforce GBNF -- that needs llama-cpp-python. If still GBNF: either add a second backend (+2 hr build) or drop news_interpreter from this tester.
4. **Audio contradiction in original brief.** §2.1 says no audio; §6.3 demands audio byte diff surfacing. A text-only script cannot do both. Decision: §6.3 moves out of this tester. The C7 audio baseline gets its own check.
5. **Gemma chat template.** Some transformers versions reject bare `system` role. Verify via Context7 + Hugging Face MCP against current transformers behavior. Convert to leading `user` turn if needed.

## 7. Build order (4 hours target)

1. `sys.modules` stub + import isolation. Run `test_prompt_import_isolation.py` to confirm no `comfy.*` slips in.
2. Single-load pipeline with reset-per-call wrapper. Implements `generate_fn` signature from the design brief.
3. `scripts/test_prompts_fixtures.py` with one function per prompt returning realistic typed inputs.
4. Per-prompt runner with schema validation catch (success-or-`*FailedError`).
5. VRAM polling via `torch.cuda.memory_stats()`.
6. JSON report writer with stable key order. Enables diff mode.
7. CLI surface from §4 of the brief.
8. Run deliberate-fault sanity check on outline.

## 8. Out of scope

- A/B comparison between prompt versions (separate sprint, after H8)
- Multi-model comparison
- Per-prompt deliberate-fault matrix (overkill for dev loop)
- Using `comfyui-mcp` as the validation harness
- Audio byte diff (moves to its own check)

## 9. Riskiest assumption

The 4-hour build estimate holds. If news_interpreter forces llama-cpp-python add 2 hr. If Gemma chat template breaks shape add 1 hr. Hard ceiling: 7 hr. Past that, stop and reassess scope before continuing into H2.
