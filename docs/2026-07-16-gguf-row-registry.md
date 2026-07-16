# GGUF row registry -- FINAL hardened plan (kibitz r1-r4, 2026-07-16)

Panel: Claude anchor (all 4 rounds) + codex@gpt-5.6-sol (all 4 rounds) +
claude(1 leftover, r1). Antigravity produced no review any round. Every folded
claim was grounded against the real files this session; nothing unverified is
stated as fact. First build = **gemma-4-12b + Qwen3-8B ONLY** (14B deferred).

## Contradictions r4 resolved (do not re-open)
- **"Byte-identical" is REDEFINED.** The GGUF branch (OTR_LedgerScriptWriter:705)
  currently DISCARDS the writer's sampling, so gemma runs at llama-cpp defaults
  (top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.0). Threading the writer
  sampling (the operator's goal) WILL change gemma's tokens. So "identity" means
  **registry / path / quant / effective-n_ctx / load behavior identical**; the
  GGUF lane now HONORS the writer sampling widgets it previously ignored -- an
  intentional, announced behavior change for ALL gguf rows. The gemma guard test
  asserts resolution identity, NOT token-identical output.
- **Overrides are an EXPLICIT WHITELIST, not `OTR_GGUF_*` wildcard.** Enumerate
  each name (OTR_GGUF_N_CTX, _N_BATCH, _N_GPU_LAYERS, _VERBOSE, _MAX_NEW_TOKENS;
  NO global _KV_GB_PER_1K -- KV is per-row). Legacy GEMMA4_12B_* apply ONLY while
  building the gemma load-config; Qwen is immune to all of them (incl.
  GEMMA4_12B_MAX_NEW_TOKENS and GEMMA4_12B_GGUF_PATH).
- **Promotion != 3 green renders.** PASS requires: effective n_ctx==8192, CUDA
  exec, NO silent context/quant/path fallback, peak resident <=14.5 GB, no
  sysmem crawl, 3x RESULT SUCCESS + obs_publish OK + both canonical assets. Only
  then pin size/SHA/measured-KV and flip UNKNOWN->PASS.

## Registry (`_otr_gguf_backend.GGUF_ROWS` -- canonical)
`GGUFRow(repo_id, subdir, artifacts:{quant:(filename,size|None,sha|None)},
context_window, kv_gb_per_1k|None, vram_fit_tier, license, license_audit_status,
requires_auth, stop_tokens, think_policy)`. Constants: `GGUF_TOP_K = 40`.
- Validate at construction with explicit `GGUFRegistryError` raises (NOT `assert`
  -- assertions vanish under -O): unique repo_id; regular safe subdir/filename;
  positive context; closed think_policy enum {"none","qwen3_no_think"}; explicit
  license/tier/auth. approx_artifact_gb DERIVED from pinned bytes; an unpinned
  (size=None) artifact projects catalog `approx_safetensors_gb=0.0` = UNKNOWN
  (never a measured estimate) and runtime VRAM uses the validated on-disk
  `stat().st_size` at load.
- Two rows:
  - gemma: `unsloth/gemma-4-12b-it-GGUF`, subdir gemma-4-12b-it, artifacts Q8_0
    (12669646240) / Q6_K(None) / Q4_K_M(None), context_window 4096, kv 0.7,
    vram_fit_tier PASS, license apache_2_0, requires_auth False, think "none".
  - qwen: `unsloth/Qwen3-8B-GGUF` (VERIFIED exists, arch qwen3, license
    apache-2.0, ungated), subdir Qwen3-8B, artifacts Q4_K_M(None,None until
    download), context_window 8192, kv None(measure), vram_fit_tier UNKNOWN,
    license apache_2_0, requires_auth False, stop (), think "qwen3_no_think".

## Catalog projection
- `_gguf_native_virtual_rows` iterates GGUF_ROWS. Guard only the backend IMPORT
  (optional-dep safe); a registry-VALIDATION error must PROPAGATE (remove the
  blanket `except Exception: return ()` swallow at :353) so a malformed row fails
  tests/startup, not silently deletes the lane.
- `_gguf_native_row_on_disk(repo_id)` = "any registered artifact for this row is
  a regular NONZERO file" (dropdown has no quant context, :663). build_dropdown
  calls it per repo_id. Selected-quant readiness lives ONLY in validate_gguf_ready.
- VRAM: gguf_native resident ~= selected (repo_id,quant) on-disk bytes (NO /2 at
  _estimate_resident_gb:1442) + per-row kv_gb_per_1k.

## Effective load-config + cache identity (single object, threaded)
- `_preflight_llm_selection(...)` runs AFTER the bank word-count/refine gates and
  BEFORE `_apply_story_scaffold_env` (:3642-3695) and before any source fetch/
  rerank. It returns: normalized creative+technical ids, ONE validated
  LLMRuntimePolicy, and a row-keyed immutable `load_config` per gguf slot
  (repo_id, resolved path, quant, effective n_ctx, n_batch, n_gpu_layers, kv,
  sampling, seed). Validates quant-in-artifacts (else GGUFNativeConfigError with
  slot/row/available), effective-n_ctx precedence (whitelisted OTR_GGUF_* >
  GEMMA4_12B_*(gemma only) > policy; malformed override RAISES) and
  512<=n_ctx<=row.context_window.
- Thread the load_config through EVERY llm entry point -- not just the main pass:
  `request_slot(..., load_config=)` (:801), backend `load(..., load_config=)`,
  AND the policy-less callers (RSS: _otr_source_payload._fetch_science_rss:265;
  rank/rerank: story_orchestrator:1509,:1614; direct writer:5138). No call may
  rebuild effective config from live env. Include repo_id + resolved path + quant
  + n_ctx + n_batch + n_gpu_layers in the resident-reuse cache identity BEFORE a
  hit (today cache_key:117 / loader:905 compare only raw policy). Serialize the
  load_config into the ledger receipt; downstream (FreezeCascade:357, shot_lock:692)
  reads it, not the env.

## Structured JSON + generate behavior
- Set `_otr_supports_json_object` on the `_SlotScheduler.for_slot()` WRAPPER
  (:595-601, what invoke_structured_slot sees) AND direct factories;
  invoke_structured_slot forces json_object for that marker + `_otr_response_format
  is None` (:603-612), covering openrouter AND gguf. Test through
  `for_slot("technical")` AND the direct factory.
- Sampling: wire :705 `make_gguf_generate_fn(cache_entry, sampling={temperature,
  top_p, min_p, repeat_penalty(=repetition_penalty)})` from the _SlotScheduler
  values (:458-474); generate() forwards non-None; top_k pinned = GGUF_TOP_K(40).
- Seed: `OTR_GGUF_SEED` = REQUIRED uint32 ENV input when a gguf row is selected,
  resolved once in preflight (NOT a widget; a widget named "seed" trips the
  companion-slot rule at _otr_workflow_validator:117-154). Deterministic per-call
  derivation; store algorithm-version + base seed + pinned top_k ONCE + ordered
  call metadata in the receipt.
- Stamp stop_tokens+think_policy into cache_entry at load; merge row+caller stops
  (dedupe); think-strip applies to _extract_text ONLY when effective
  response_format absent (:399,:472,:483): one LEADING complete `<think>...</think>`
  envelope, preserve later literal `<think>` in body. (Drop the empty-after-strip
  assert -- unreachable once structured always carries rf.)
- `validate_gguf_ready(repo_id, quant, load_config)`: row path, wrap
  exists/stat/permission as GGUFNativeConfigError, validate pinned size+SHA when
  present (existence-only when None), memoize hashing by (resolved path, size,
  mtime_ns) with post-replace invalidation test.

## Tests (unit, no GPU)
gemma resolution-identity (path/n_ctx/quant, NOT token output); n_ctx
validate-not-clamp; whitelisted OTR_GGUF_* vs GEMMA4_12B_* precedence + malformed
RAISES; GEMMA4_12B_GGUF_PATH+Qwen -> Qwen file; Qwen-negative (no gemma path/env/
error-string leak into a Qwen load); per-row nonzero on_disk; VRAM no-halve +
per-row KV; quant-mismatch RAISE + mixed-slot; json_object via for_slot wrapper
AND direct factory; every sampling kwarg + per-call seed reach create_chat_completion;
think-strip envelope-only + literal preserved + skipped under rf; stop merge; SHA
mismatch + memo invalidation; cache reload on path/n_ctx change; gemma->qwen->gemma
close order; registry GGUFRegistryError fails loud. Plus FULL Windows regression +
Bug Bible + AST/JSON + canonical round-trip + OTR_WorkflowValidator + widget/link
audit; verify HEAD==origin after each pushed chunk.

## Ordered sequence (GPU-blocked past step 1)
0. VERIFY (done/continue): Qwen repo exists, license apache-2.0, ungated
   (VERIFIED); confirm exact Q4_K_M filename casing + native context + stop/think
   at the download gate.
1. CODE chunk -> unit-green -> PUSH v2.0-alpha. Gemma resolution-identical; Qwen
   row present, UNKNOWN, size/sha/kv=None, on_disk=False.
2. DOWNLOAD: `hf download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir
   C:\ComfyUI-Models\LLM\converted\Qwen3-8B --revision <immutable>`; record CLI
   version, path, byte size, sha256 in the receipt.
3. LIVE 3 legs (GPU free): reset server EACH; per leg set BOTH Node 1 model
   widgets (verify positions models@3,4 / n_ctx@32 / quant@33) to Qwen3-8B / 8192
   / Q4_K_M in otr_canonical.json, validate, restore bytes/hash in finally; RNG
   s1/s2/s3, identical source snapshot + sampling; receipt names both slots +
   load_config key + sampling + source-hash + workflow-hash + ledger + assets.
   Judge format + news-seed fidelity (prose = observational note, NOT a gate).
4. PROMOTION chunk (2nd commit): pin size/sha + measured KV, flip UNKNOWN->PASS
   only if the promotion gates (above) all hold; rerun every gate; push.

## Judgment log (accept / reject / verify-at-build)
- ACCEPTED all confirmed panel MUST/SHOULD across r1-r4 (grounded).
- REJECTED (as contradictions, resolved above): literal gemma token byte-identity
  once sampling threads; `OTR_GGUF_*` wildcard; prose as a promotion gate.
- REJECTED-as-noise: none material; no hallucinations found in codex@5.6 output.
- VERIFY-AT-BUILD: exact preflight line; Node 1 widget positions; loader reuse
  key includes model_id; llama-cpp 0.3.33 create_chat_completion kwargs; Qwen
  Q4_K_M filename/size/sha at download.

## Agent calls: 12 attempted (3 agents x 4 rounds). Produced: codex@gpt-5.6-sol
x4 (all grounded), claude x1 (r1 leftover), antigravity x0.
