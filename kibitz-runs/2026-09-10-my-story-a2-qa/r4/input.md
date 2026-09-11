# Finished-code QA: A2 actual native capacity and exact prompt fitting

Read-only actual Windows diff against af9ccb09. Do not edit, load models, run
GPU work or alter remote machines. One CLI reader; root remains sole judge.
Production owners: _otr_model_catalog, _otr_model_loader, _otr_model_runtime,
_otr_loader_backends, _otr_generation_budget, OTR_LedgerScriptWriter,
_otr_constrained_generate, _otr_structured_call. Nine test files also changed.
Full design converged in kibitz-runs/2026-09-10-my-story-cross-machine/r4/.
GO_FORWARD remains the only queue; cleaner/source/visual work follows A2.

Expected contract:
- Actual positive integer decoder config, text_config before wrapper fields.
  Bool/nonpositive/invalid absent. Native capacity wins over historical estimate.
  Explicit positive pin may be below 512; min(native,pin); no invented 8192 pin.
  Unknown capacity retains labelled existing estimate, not a false native claim.
- Native HF captures one normalized pin; (policy.cache_key(),pin) at lookup and
  publication. Same textual integer reuses; change/removal reloads. No env reread
  during load. Direct load_llm(context_cap=...) still works as explicit setting.
- Resolve canonical hub before native validate/discovery, pass same hub to
  discovery/download/load. Remote OpenRouter/Comfy/Google and GGUF route before
  HF work. GGUF n_ctx/reuse unchanged. Preserve admission-before-reuse and epochs.
- Finalize from actual selected AutoConfig and loaded decoder even first download;
  never mutate max_position_embeddings. Sparse later metadata cannot erase known
  snapshot/AutoConfig capacity. HF VRAM remains weights-only, vram_priced_ctx None.
- Remove false project-minimum model-window load gate. Default output room is
  one token; explicit minimum/require_full and reserve-plus-numeric atomic contract
  remain. Google shares the new default; remote estimates/spend reservation and
  OpenRouter/Comfy explicit floors are unchanged. SciFi legacy character-based
  draft-fit heuristic is outside this native-owner correction; no all-bank claim.
- prepare_native_prompt uses existing normalization then exact tokenize=False,
  add_generation_prompt=True/template kwargs, then tokenizer(...,return_tensors='pt').
  All four native paths fit before .to(device); no slicing or duplicated estimate.
- Scheduler bound/unbound closures expose optional _otr_inspect_fit: remote/GGUF
  report unsupported without acquisition; native acquires configured slot, keeps
  real transition records, no generation/helper counts, primitive output only.
  Actual generation reacquires/revalidates. inspect_structured_fit includes the
  schema contract and message conversion from structured_call; source never mutates.
- Existing grammar lifecycle, repetition guards, EOS/output-limit evidence and
  real provider/storage/cancel/OOM failures remain; no story-length gate.

Tests exercise actual loader AST boundaries, exact CPU token sequences shared
by fit and generation, schema overhead, all four routes, ephemeral input release,
counting, explicit atomic refusal before device transfer, first-download metadata,
pin change mid-load/equivalent/removed settings, remote/GGUF no-HF and weights-only
native admission. Focused set had zero new failure IDs against prior full baseline;
known catalog/GGUF fixture failures remain. Latest full regression is running.
Do not hide failures or require the later full canonical story run as code proof.
Canonical has no surface edit in this chunk; qualification will load the real JSON.

Find concrete introduced defects, not a new feature wishlist. Ignore inherited
diff.txt/diff_utf8.txt and unrelated old artifacts. Report uncertain claims as such.
