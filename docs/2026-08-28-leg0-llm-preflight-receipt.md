# Leg 0 -- local-LLM preflight receipt (2026-08-28 evening)

GO_FORWARD_PLAN 1.2 / Batch R3's precondition, run on a fully reset box
(0 ComfyUI processes, 669 MiB VRAM, port 8000 empty, ~14.7 GiB free).
In-process, no ComfyUI server: for each curated local row, `request_slot` ->
a real ~40-token generate through the writer's OWN seam
(`_build_truncating_generate_fn`, so every backend dispatches exactly as it
does in production) -> `unload_llm`, with `reset_peak_memory_stats()` around
each row.

## RESULT: 7 rows found, 7 PASS, 0 FAIL

| row | backend | peak GiB | load s | gen s |
|---|---|---:|---:|---:|
| `mistralai/Mistral-Nemo-Instruct-2407` | transformers | 7.94 | 23.8 | 1.5 |
| `google/gemma-4-E2B-it` | transformers (mm text-only) | 6.33 | 10.3 | 2.0 |
| `google/gemma-4-E4B-it` | transformers (mm text-only) | 8.73 | 14.4 | 3.3 |
| `google/gemma-4-12b-it` | transformers (mm text-only) | 7.29 | 19.8 | 3.2 |
| `unsloth/gemma-4-12b-it-GGUF` | gguf_native (Q4_K_M) | see note | 8.4 | 0.8 |
| `unsloth/Qwen3-8B-GGUF` | gguf_native (Q4_K_M) | see note | 5.4 | 0.4 |
| `google/gemma-2-2b-it` | transformers | 2.13 | 4.7 | 1.5 |

Every row produced coherent prose (each output head is stored in
`leg0_report.json` beside the harness).

**Corroboration worth noting:** `google/gemma-4-12b-it` peaked at 7.29 GiB,
matching the catalog note's independently-recorded "NF4 measured at 7.15 GiB
allocated / 7.29 GiB peak on the 16 GB RTX 5080" to the digit.

## What this proves, and what it does NOT

* PROVES: every curated local row LOADS, GENERATES coherent text through the
  production seam, and UNLOADS on this box, comfortably inside the 14.5 GiB
  target. No dead rows in the dropdown.
* DOES NOT PROVE the operator's creative/technical PARITY rule. A plain
  generate is the creative job; the technical job is constrained JSON / GBNF.
  That remains Batch R3's four canonical legs.
* The two GGUF rows report ~0.01 GiB torch peak. That is a MEASUREMENT LIMIT,
  not a claim of free VRAM: llama-cpp-python allocates outside torch's
  allocator, so `torch.max_memory_allocated` cannot see it. Their real cost is
  the artifact size (7.12 GB / 5.03 GB on disk).

## Two harness defects found and fixed to get here (both real, not test noise)

1. **The HF cache root.** The user-level `HF_HUB_CACHE` points at
   `C:\ComfyUI-Models\huggingface`, but the models live one level down in
   `...\huggingface\hub`. With the env as-shipped the catalog scanned the
   wrong dir and reported all four HF writer rows ABSENT -- run 1 skipped
   them and "passed" 2 rows. Worth an operator look: a cold box may hit the
   same mismatch.
2. **GGUF load-config threading.** `request_slot` on a gguf_native row
   without a threaded `load_config` hits the gemma-only env fallback and
   REFUSES by design ("would silently load gemma for a non-gemma row") --
   which is the fail-closed behavior working. The harness now builds the
   immutable config via `build_gguf_load_config` exactly as the writer's
   preflight does.

## The deliberate negative probe (recorded from run 1, not re-run)

gemma GGUF at **Q8_0 / n_ctx=4096** (the nv50 BASELINE_POLICY identity):
**FAIL, and correctly** -- `Insufficient VRAM for GGUF n_ctx=4096: free
14.69 GB < needed 14.70 GB`, refusing with NO silent context downgrade. The
margin is 0.01 GB at desktop idle, so Q8_0 is not practically loadable on
this box with anything else resident; **Q4_K_M is the only shippable gemma
GGUF quant here.** Either outcome was informative; this is the one recorded.

Harness: `scratchpad/leg0/otr_llm_preflight_leg0.py` (+ `leg0_report.json`,
`leg0_run1.log`, `leg0.log`). It is a throwaway probe by section-3 rules;
promote it into `scripts/` only if the sweep wants it as a standing gate.
