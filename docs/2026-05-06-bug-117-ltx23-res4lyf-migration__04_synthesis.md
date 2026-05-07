# Synthesis — 2026-05-06 — LTX 2.3 + RES4LYF migration

## Round summary

- **ChatGPT (gpt-5.5, 127.3s):** Conservative, mirror stock workflow, two
  JSONs for rollback. Hedged on whether `CFGGuider` could substitute for
  `MultimodalGuider` ("may be doing LTX 2.3-specific handling").
- **Gemini (gemini-3.1-pro-preview-customtools, 40.5s):** Corrected ChatGPT.
  `MultimodalGuider` is structurally required for LTX 2.3 DiT (tensor
  shape mismatch with `CFGGuider`, not subtle quality regression). Flagged
  PCIe thrashing risk on 42 GB BF16 streaming through 16 GB VRAM.
- **NVIDIA (llama-3.3-nemotron-super-49b-v1.5, 90.3s):** Sided with Gemini
  on `MultimodalGuider`, agreed with both on env-var engine selector +
  explicit float32 sigmas + Gemma encoder requirement.

## Convergent decisions (all three agree)

| Decision | Final |
|---|---|
| Engine selector | `OTR_LTX_ENGINE` env var (`v0_9` / `v2_3`); loud startup log |
| Sigmas | `torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)` explicit, CPU-side |
| Text encoder | Gemma via `LTXAVTextEncoderLoader` with full 3-widget config; `t5xxl` will not produce coherent output on 2.3 |
| LoRA application | In workflow JSON (not Python) — `LoraLoaderModelOnly` ×2 chained at strengths 0.5 + 0.2 |
| VAE decode | `LTXVTiledVAEDecode` (the 2.3-specific variant), not `VAEDecodeTiled` |
| Per-line GC | `del samples; del decoded; del latent_image; gc.collect(); torch.cuda.empty_cache()` after each iteration |

## Critical correction to Claude's proposed lean

**My lean was WRONG on `MultimodalGuider`.** I proposed substituting plain
`CFGGuider` for video-only simplicity. Gemini + NVIDIA both flagged this
as a structural error: LTX 2.3 is a DiT with packed multimodal
conditioning tensors. `CFGGuider` lacks the structure to hand them to the
sampler — likely an immediate tensor shape mismatch crash, not a subtle
regression.

**Final v2.3 chain mirrors the stock-proven path:**

```
GuiderParameters(...)
MultimodalGuider(model, positive, negative, parameters)
ClownSampler_Beta(...)
SIGMAS = torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)
SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent_image)
LTXVTiledVAEDecode(...)
```

## Disagreement on workflow JSON count

All three external models recommend **two workflow JSONs** (one for v0.9
rollback, one for v2.3) for fast rollback. **OTR's standing rule** (memory:
`feedback_minimum_json_files.md`) is the opposite: minimum JSONs, rely on
git for prior versions.

**Jeffrey's rule wins.** Hard-cut `otr_scifi_16gb_full.json` to v2.3
stack. Document rollback as: `git checkout <pre-cutover-tag>
workflows/otr_scifi_16gb_full.json && export OTR_LTX_ENGINE=v0_9`. Tag
the pre-cutover commit so git rollback is one command.

## Open uncertainties to verify in code (not blocking)

1. Does `LowVRAMCheckpointLoader` (current OTR loader, from
   `comfyui-ltxvideo` pack) handle the 22B BF16 file, or do I need to
   swap node type to `CheckpointLoaderSimple`? Tonight's smoke used
   `CheckpointLoaderSimple` so that path is proven. Try `LowVRAMCheckpointLoader`
   first since it's already in OTR; fall back to `CheckpointLoaderSimple`
   if it can't stream 22B.
2. `ClownSampler_Beta` input shape: `guides=None, options=None` or empty
   dicts `{}`? Read RES4LYF source `INPUT_TYPES` before calling.
3. Does `ClownSampler_Beta` respect ComfyUI memory hooks during step loop?
   Watch Shared GPU Memory in Task Manager during sirens_print regression.
   If it spikes and GPU util drops to <10%, sampler is thrashing — escalate
   to subprocess isolation.

## Implementation plan (final)

### Phase 1: workflow JSON edit (~30 min)

Edit `workflows/otr_scifi_16gb_full.json`:
- Node #54 widget: `ltx-video-2b-v0.9.safetensors` → `ltx-2.3-22b-dev.safetensors`
- Insert new node: `LoraLoaderModelOnly` strength 0.5
- Insert new node: `LoraLoaderModelOnly` strength 0.2
- Re-route link 87 (LowVRAMCheckpointLoader.MODEL) through both LoRA loaders
  to BatchLTXRender.model input
- Node #57: change type from `CLIPLoader` to `LTXAVTextEncoderLoader`,
  widgets to `["gemma_3_12B_it_fp4_mixed.safetensors", "ltx-2.3-22b-dev.safetensors", "default"]`

### Phase 2: batch_ltx_render.py refactor (~90 min)

Refactor `BatchLTXRender.execute()`:
- Add `_engine = os.environ.get("OTR_LTX_ENGINE", "v2_3").lower()`
- Loud log block at the top: engine, sampler family, encoder expectation
- Fail-fast dep check: if `_engine == "v2_3"` and any of
  `[ClownSampler_Beta, MultimodalGuider, GuiderParameters, LTXVTiledVAEDecode]`
  not in `NODE_CLASS_MAPPINGS`, raise loudly
- Extract existing per-line render code into `_render_one_line_v0_9_euler()`
- Add new `_render_one_line_v2_3_res4lyf()`:
  - `LTXVConditioning` (frame_rate=25)
  - `EmptyLTXVLatentVideo` (832x480, ltx_length)
  - `LTXVImgToVideoConditionOnly` (vae, image, latent, strength=0.75, bypass=False)
  - `RandomNoise` (seed)
  - `GuiderParameters(...)` — verify default args from RES4LYF source
  - `MultimodalGuider(model, positive, negative, parameters)`
  - `ClownSampler_Beta(...)` — verify guides/options shape from RES4LYF source
  - `sigmas = torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)`
  - `SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent_image)`
  - `LTXVTiledVAEDecode(samples, vae, ...)`  with same params as `VAEDecodeTiled`
  - Per-line: `del samples; del decoded; del latent_image; del noise; gc.collect(); torch.cuda.empty_cache()`
- `execute()` dispatches based on `_engine`

### Phase 3: regression (~45 min wall, ~30 min wait)

Tag pre-cutover commit: `git tag pre-bug-117-ltx-2.3-cutover`
Run sirens_print episode end-to-end on v2.3
Watch in Task Manager: Dedicated GPU Memory + Shared GPU Memory
Verify per-line peak VRAM stays under 14.5 GB
Spot-check audio C7 byte-identity on final master mix (sanity, not load-bearing)
Visual review: motion smooth, no glitches across 6+ lines

### Phase 4: commit + push

Single commit titled `BUG-LOCAL-117a: LTX 2.3 + RES4LYF cutover`. Update
`BUG_LOG.md` entry from `[FIXED]` to `[FIXED + INTEGRATED]`. Update memory
recipe with the actual chain that shipped.

## Estimated total wall time

- Phase 1: 30 min
- Phase 2: 90 min
- Phase 3: 45 min (30 min wait for render)
- Phase 4: 15 min
- **Total: ~3 hours from now**

## Source transcripts

- `01_chatgpt.md` (gpt-5.5, 127.3s, 18 KB)
- `02_gemini.md` (gemini-3.1-pro-preview-customtools, 40.5s, 4.3 KB)
- `03_nvidia.md` (llama-3.3-nemotron-super-49b-v1.5, 90.3s, 3.7 KB)
