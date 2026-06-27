# Roundtable judgment -- pass01 (coding/implementability), CONVERGED

Panel: GPT-5.5 (openai/gpt-5.5-20260423), Gemini-3.1-pro
(google/gemini-3.1-pro-preview-20260219), DeepSeek-v4-pro
(deepseek/deepseek-v4-pro-20260423) + Claude grounded anchor.
Spend: ~$0.0892 (truncated probe pass) + ~$0.2913 (full 12k-token pass) =
**~$0.3805 total**. Stopped at convergence (no R2/R3/R4 grind -- one clean
grounded fix emerged and was code-verified).

## ACCEPTED -- grounded CONFIRMED

1. **VideoVAE topology leak is the real root cause** (GPT MUST-FIX #4 + Gemini
   MUST-FIX #1, independently). CONFIRMED in `eng_ltx_av.py::_build_graph`:
   the single `"videovae"` node (L341) is wired to BOTH `"i2v"` (L370,
   pre-sampler) AND `"decode"` (L404, post-sampler). `wrapper_bridge.run_graph`
   `free_after_use` (L367-372) decrements a remaining-consumer count and frees a
   source only when `remaining[s] <= 0` -> the 1384 MB VideoVAE is pinned in VRAM
   through the ENTIRE sampler/denoise loop. That is the ~1.4 GB of activation
   headroom that tips b004 into the sysmem spill (b003 barely fit, b004 did not).
   **FIX: split into `videovae_enc` (-> i2v) and `videovae_dec` (-> decode)** so
   free_after_use drops the encode-side VAE before the sampler. This REPLACES the
   plan's reserve-VRAM primary -- no invented API, no download, no quality loss,
   self-contained in eng_ltx_av.py.

2. **The "reserve-inference-VRAM global / minimum_memory_required hint" is an
   undocumented, brittle API** (all 3 panelists + Claude anchor). ACCEPTED:
   DROP it as the primary mechanism. The VideoVAE split makes it unnecessary.
   (If a future box still spills after the split, revisit a VERIFIED loader
   budget -- not a guessed global mutation. GGUF UnetLoaderGGUF may ignore it.)

3. **Complementary allocator config B is best-effort, not load-bearing** (GPT
   #8/#9, DeepSeek #4). ACCEPTED with caution: `PYTORCH_CUDA_ALLOC_CONF` must be
   set BEFORE CUDA init; in OTR `prestartup_script.py` its effect on the already-
   running Desktop app is unverified. Add `garbage_collection_threshold:0.8` via
   `setdefault` as defense-in-depth against the cross-beat fragmentation that
   explains the b002/b003/b004 non-monotonicity -- but do NOT claim it creates
   headroom. `expandable_segments` stays operator opt-in (Windows support
   unverified on this build).

4. **Hardening: the 14.5 GB ceiling assert does not guard the sampler peak**
   (GPT MUST-FIX #5). CONFIRMED: `render_clip` calls `reclaim_idle_models` (L467)
   BEFORE `assert_vram_within_ceiling` (L469), so a 15.8 GB sampler peak is hidden
   by post-decode reclaim. SHOULD-FIX (follow-up): sample NVML peak around
   `run_graph`. Noted; not blocking the speed fix.

## REJECTED -- MISREAD

- Gemini SHOULD-FIX #1 "PyTorch 2.10 does not exist (latest is 2.6)": MISREAD --
  `torch/version.py` on the box reads `2.10.0+cu130` (verified on disk). Discard
  the "correct the version" instruction. (Its expandable_segments point survives
  only as the opt-in note above.)

## CUT -- converged across panel + anchor

- **Q2_K unet** (all 3): lossy on a 22B audio-conditioned model + a ~6.5 GB
  download not on disk. Unnecessary once the split reclaims ~1.4 GB natively.
  Keep only as a manual last-resort runbook line.
- **NVIDIA Sysmem Fallback Policy (D)** (GPT, DeepSeek): a driver/Control-Panel
  setting, not an OTR code fix. Operator-doc only.

## Invariants preserved

No fallbacks / fail LOUD (unchanged); no `unload_all_models` (the split only
re-times an existing free); `<=` 14.5 GB ceiling (the split LOWERS the peak);
the internal render graph is built in code (NOT a workflow-JSON node/widget), so
CLAUDE.md s0 JSON-wiring rule does not apply -- the engine is already wired into
otr_scifi_16gb_full.json. UTF-8 no BOM; push to v2.0-alpha.
