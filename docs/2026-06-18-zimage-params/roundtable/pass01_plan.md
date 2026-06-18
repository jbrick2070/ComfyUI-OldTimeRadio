# Z-Image-Turbo writing/parameter plan -- CONVERGED (pass01, 3-model roundtable grounded)

**Status: converged in one pass.** Panel = GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro (~$0.10).
All three independently confirmed the same defect set; every CONFIRMED item is folded below, the
over-engineering is cut, and the four genuinely-unknowable node specifics are explicit
**VERIFY-AT-BUILD** checks (they need a live `/object_info` from the installed Z-Image nodes, which
are not on the box yet -- a second panel pass cannot resolve them, so the loop stops here).

**Goal:** make `z_image_turbo` a look-matched, in-process peer of `flux_gen1`, reusing the SAME
composed prompts, so the operator can pick it per role and the commercial-clean lane exists.

---

## 1. ARCHITECTURE -- rewrite the stale sidecar stub to in-process (clone `lumina_image.py`)

The current `z_image_turbo.py` gates `assert_usable` on `OTR_ZIMAGE_SIDECAR` and `render_image`
raises `NotImplementedError` -- that is stale (Z-Image is now a ComfyUI-core split-file model).
Replace it with the in-process Lumina pattern:

- Methods: `_zimage_params(request)`, `_node_candidates()`, `_build_zimage_graph(params, wire)`,
  lazy `load()` (resolve classes), `render_image(self, request, prepared=None)` driving
  `wrapper_bridge.run_graph` then `images_to_uint8`, returning `frames[0]`.
- **`render_image` signature MUST be `(self, request, prepared=None)`** (matches flux/lumina; the
  current stub requires `prepared` -> dispatcher mismatch). [CONFIRMED]
- **Wrap the run in `try/finally: _wb.reclaim_idle_models(reason="z_image_turbo post-decode")`** --
  both flux_gen1 and lumina_image do this; the draft omitted it -> model stays resident before
  video. [CONFIRMED]
- Cold-import clean (V-12): only the dep-free registry + role vocab + stdlib at module scope;
  `wrapper_bridge`/torch lazy inside `load`/`render_image`. [CONFIRMED]
- Keep `default_roles=()` (opt-in peer, no model is primary) and `commercial_clean=True`.

### Split-file loaders + env vars (mirror lumina's MODEL_ENV/CLIP_ENV/VAE_ENV) [CONFIRMED]
Comfy-Org split files: `z_image_turbo_bf16.safetensors` (diffusion) + `qwen_3_4b.safetensors`
(Qwen3-4B text encoder) + `ae.safetensors` (VAE).
- `OTR_ZIMAGE_UNET` (default `z_image_turbo_bf16.safetensors`) -- `UNETLoader`
- `OTR_ZIMAGE_CLIP` (default `qwen_3_4b.safetensors`) -- `CLIPLoader`
- `OTR_ZIMAGE_VAE`  (default `ae.safetensors`) -- `VAELoader`
- Pass **basenames** to the loaders (`os.path.basename`, like lumina; Comfy folder_paths resolves).
- **`assert_usable` must fail closed on the DIFFUSION-MODEL file** (`OTR_ZIMAGE_UNET` path), NOT a
  sidecar python; TE+VAE loaders fail LOUD at render -> dispatcher floor. Drop `OTR_ZIMAGE_SIDECAR`
  entirely. [CONFIRMED]

### Graph (clone `lumina_image._build_lumina_graph`) -- with 4 VERIFY-AT-BUILD slots
`UNETLoader -> ModelSamplingAuraFlow(shift) -> KSampler(model)`;
`CLIPLoader -> CLIPTextEncode(pos)/CLIPTextEncode(neg)`; `VAELoader -> VAEDecode`;
`<latent-node> -> KSampler.latent_image`; terminal = `decode`.

**VERIFY-AT-BUILD (capture from a live `/object_info` on the installed Z-Image nodes before wiring --
none are resolvable from the current repo; the Z-Image node/weights are not installed):**
1. **`CLIPLoader` `type` string for Qwen3-4B** -- the draft's `<z-image/qwen>` is a placeholder.
   Read the exact literal (e.g. `qwen`, `qwen3`, `z_image`, ...) from the installed node's
   INPUT_TYPES / the official Z-Image workflow JSON. A wrong type fails the encode. [VERIFY-AT-BUILD]
2. **Latent node = `EmptySD3LatentImage` vs `EmptyLatentImage`** -- depends on Z-Image's VAE channel
   count (16-ch SD3-style vs 4-ch). Wrong node -> crash or noise. Verify the latent channel count;
   use the matching node. [VERIFY-AT-BUILD]
3. **`ModelSamplingAuraFlow` is the correct sigma-shift node for Z-Image** (S3-DiT != guaranteed
   AuraFlow shift). Confirm against the official Z-Image ComfyUI workflow; if it ships a different
   sampling node, use that. [VERIFY-AT-BUILD]
4. **`UNETLoader weight_dtype`** -- `"default"` vs an explicit `"fp8_e4m3fn"`/bf16 (memory + upcast
   behaviour). Confirm on the box. [VERIFY-AT-BUILD]

---

## 2. STARTING KNOBS (env defaults; all `OTR_ZIMAGE_*`, request overrides win)

| knob | default | rationale |
|---|---|---|
| `OTR_ZIMAGE_STEPS` | **8** | distilled design point (8 NFE) |
| `OTR_ZIMAGE_CFG` | **2.0** | distilled; cfg 4+ hurts. 2.0 keeps the NEGATIVE live (a lever Flux@cfg1.0 lacks) |
| `OTR_ZIMAGE_SHIFT` | **3.0** | ModelSamplingAuraFlow; lock after the A/B (candidates 3.0 and 6.0 only) |
| `OTR_ZIMAGE_SAMPLER` | **euler** | matches flux/lumina |
| `OTR_ZIMAGE_SCHEDULER` | **normal** | AuraFlow graph uses `normal` (lumina does); NOT `simple` -- the draft's "match Flux" was wrong here [CONFIRMED fix] |
| `OTR_ZIMAGE_NEGATIVE` | see below | live at cfg 2.0; pushes Z-Image off its clean-digital default toward the Flux filmic grade |
| width/height | request w/h; else env default | **honor request dims EXACTLY; NO snapping, NO upscale in the engine** [CONFIRMED cut of snap/upscale] |

- **No `FluxGuidance` node** (Flux-specific). Z-Image gets adherence from real cfg + the negative.
- **Default negative** (color/material only first; add "blurry/soft" ONLY if Z outputs too soft,
  so it doesn't fight the desired analog grain):
  `oversaturated, glossy, clean digital, plastic skin, waxy skin, sterile studio lighting, cartoon,
  illustration, text, watermark`
- Add a flux/lumina-style log line (w,h,seed,steps,cfg,shift,sampler,scheduler).

---

## 3. PROMPT STRATEGY (the "writing parameter")

- **v1: reuse `compose_still_prompt` output AS-IS** (same subject + the grade tails STYLE_TAIL_DEFAULT
  / IMAGE_GRADE_TAIL / RADIO_BROADCAST_TAIL / NO_TEXT_CLAUSE) so the filmic LOOK carries over. The
  grade tails stay in the POSITIVE; the new live NEGATIVE removes Z-Image's clean sheen.
- **Naturalization is NOT in v1** (cut -- adds a variable before the base graph is proven). Document
  an OPTIONAL future lever `OTR_ZIMAGE_NATURALIZE=1` (Qwen3 is an LLM and may prefer prose over the
  comma-tag string) to try ONLY if the as-is A/B underperforms. Prompt composition stays UPSTREAM
  of the engine (the engine takes a prompt string), so determinism/content-addressing is unchanged.

---

## 4. LOOK-MATCH LEVERS (ranked) + the CUT

The match is, in order: (1) keep the grade tails in the positive; (2) use the now-live negative to
kill clean-digital sheen; (3) shift/cfg tuning. **CUT "a light post grade" from this plan** -- the
engine outputs raw PNG; grading is the post-pipeline's job and would mask whether the
sampling/prompting actually matches Flux. [CONFIRMED cut, all 3 models]

---

## 5. TIER CAVEAT (CONFIRMED -- do not over-claim)

bf16 6B diffusion (~12 GB) + Qwen3-4B TE + VAE comfortably fits the **16 GB 5080** (our box) but is
NOT a true sub-8GB footprint. For the genuine 8GB tier, a later step needs an **FP8/GGUF Z-Image +
a quantized/offloaded Qwen3 TE**. The commercial-clean (Apache-2.0) win holds at every precision;
the "fits 8GB" claim is precision-dependent and deferred to a quant pass.

---

## 6. A/B LADDER (one variable per run; raw-PNG judging; fixed seed set + fixed prompts)

After the graph runs on GPU (VERIFY-AT-BUILD cleared):
1. **Control vs Flux:** same prompt/seed -> z_image @ {steps 8, cfg 2.0, shift 3.0, normal} vs the
   Flux reference still. Judge look-match on raw PNG.
2. **shift:** 3.0 vs 6.0 (lock one).
3. **cfg:** 1.5 vs 2.0 (lock one; higher only if adherence weak).
4. **negative on/off** (confirm it moves the grade toward Flux).
5. **steps:** 8 vs 10 (only if 8 looks under-converged).
6. (only if 1-5 fail the look-match) **naturalize prompt** A/B.
Lock the env defaults from the winners; that is the final engine config.

---

## 7. BUILD CHECKLIST (coder)
- Rewrite `nodes/_otr_image_engines/eng_... z_image_turbo.py` to the in-process pattern above
  (keep the engine `name="z_image_turbo"`, registered, `requires_flag=OTR_ENABLE_ZIMAGE`, dark).
- CPU unit tests: `_zimage_params()` env/override resolution + `_build_zimage_graph()` shape
  (mirror the lumina graph tests); fail-closed `assert_usable` on missing UNET.
- Suite + Bug Bible green; UTF-8 no BOM. Keep OUT of `VALIDATED_ENGINES` until the GPU A/B passes.
- THEN the operator installs the 3 Z-Image weights + node, clears the 4 VERIFY-AT-BUILD checks,
  runs the A/B ladder, and (on a look-match PASS) promotes `z_image_turbo` into `VALIDATED_ENGINES`.
