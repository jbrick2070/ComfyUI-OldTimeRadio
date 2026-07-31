# WAN 8 GB problem statement -- independent cross-check (Claude / Cowork window)

Date: 2026-07-31. Window: Cowork (Claude), independent of the Codex-anchored
`report.md` in this directory. Method: every code claim re-grounded against the
real Windows tree via the device bridge (motion_common.py, eng_wan_ti2v.py,
render_driver.py, wrapper_bridge.py, otr_8gb_wan.json, PROD_BUG_LOG.md); every
external claim re-verified against primary sources by three parallel web
research lanes (ComfyUI internals; Wan empirics/distills; model landscape and
licences). Verdicts were formed from that grounding, then diffed against
report.md. $0 external spend.

Bottom line first: the three claims are each PARTLY right, and each one
overreaches in a way that would misdirect the fix. The recorded failure is
100% the estimator gate and 0% proven to be GGUF or the canvas. The sweep as
designed cannot answer the question. And the single cheapest lever nobody's
document contains is a step-distilled 5B GGUF that drops into the existing
loader unchanged.

---

## 0. Factual errors in the problem statement itself

1. **"The official 8 GB workflow uses fp8 scaled safetensors" -- half false.**
   The official 5B workflow UNET is `wan2.2_ti2v_5B_fp16.safetensors` (10 GB,
   fp16). Only the text encoder (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`,
   6.74 GB) is fp8-scaled. There is no official fp8 5B UNET; the only
   fp8-scaled 5B is Kijai's community file (5.28 GB). (Concurs with report.md;
   independently confirmed against Comfy-Org/Wan_2.2_ComfyUI_Repackaged.)

2. **The docs' 8 GB sentence predates Dynamic VRAM by ~7 months.** The Wan 2.2
   tutorial (and its "should fit well on 8GB vram with the ComfyUI native
   offloading" line) shipped 2025-07-28 (Comfy-Org/docs discussion #291).
   Dynamic VRAM (comfy-aimdo) became default-on in v0.16.0 on 2026-03-05. So
   the official 8 GB claim was made about the LEGACY offload path -- the same
   path our GGUF stack is on. Citing that sentence as evidence that aimdo
   streaming "makes 8 GB work" is citing evidence that predates the mechanism.
   This breaks Claim 2's causal chain at the link that carries all the weight.

3. **"Two independent 8 GB reports" for 14B-beats-5B is actually one**, and it
   is not the mechanism described. lilting.ch (2026-03-06, RTX 4060 8 GB,
   32 GB RAM) is the only head-to-head found by two separate search passes.
   Its "14B" was Phr00t's WAN2.2-14B-Rapid-AllInOne -- a single 22 GB merged
   checkpoint with Lightning/rCM baked in, since deprecated by its own author
   ("I do not maintain this anymore") -- not "Q4 GGUF + Lightning LoRA".
   Timings: 14B 111.41s vs 5B 113.93s; VRAM "4856 MB loaded / 11067 MB
   offloaded". Real, single, and mechanically different from the claim.

4. **"No Lightning/step-distill LoRA exists for the 5B" -- false.** Three
   options exist today:
   - `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` -- Apache-2.0, 3-step
     sparse-distill+DMD full weights (2025-08-04), plus a LoRA form:
     `Kijai/WanVideo_comfy/FastWan/Wan2_2_5B_FastWanFullAttn_lora_rank_128`
     (run cfg=1.0 per Kijai).
   - `quanhaol/Wan2.2-TI2V-5B-Turbo` -- 4-step distill, "eliminates the CFG
     trick", ComfyUI support noted 2026-01-26. Upstream licence text NOT
     found on its GitHub -- diligence required before shipping.
   - `hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF` -- Apache-2.0-tagged GGUF Q2_K-Q8_0
     of the above; "4 steps is enough. CFG 1"; author claims usable on a 4 GB
     GPU; "works fine with most LoRAs made for the regular 5B". Same author
     as the TiledVaeLite GTX-970 decode benchmark.
   The lightx2v catalog being A14B-only is true but irrelevant: lightx2v is
   not the only distiller. This kills the structural argument for a 14B tier
   ("the 5B pays full 20-30 steps") -- and it matters doubly because cfg=1
   halves sampling-stage activations (see section 1). report.md caught
   FastWan; the Turbo GGUF -- the only one that drops into our existing
   `UnetLoaderGGUF` graph with zero topology change -- is new here.

5. **Arithmetic: 33 frames requires 10,577 MB, not 10,647.** (Also flagged in
   report.md.) Conclusion unchanged.

6. **The recorded failure is not an 8 GB artifact.** PBUG-20260723-02 was a
   177-frame request on the 16 GB dev 5080 (cost model afforded 30); the
   verified defect was the dead launch.env ceiling channel, since fixed via
   `video.max_render_frames`. No 8 GB card has ever produced telemetry in this
   repo. (Concurs with report.md.)

7. **"#13953 confirmed as a known limitation" overstates provenance.** It is a
   community-filed feature request ("GGUF dynamic-vram", 2026-05-18), open,
   with no maintainer reply. The MECHANISM is nonetheless true -- verified
   directly in `city96/ComfyUI-GGUF/nodes.py` (class def + `clone()`
   reassigning `__class__`). Cite the source, not the issue.

8. **The research doc conflates two different ComfyUI features.** Issue #11081
   (GGUF slow at ~50% VRAM; umt5 encode 3.5 min -> 27+ min) is about the
   DECEMBER 2025 async-offload / pinned-memory feature, not March 2026
   Dynamic VRAM. Its acute text-encoder regression was patched shortly after.
   Keep three mechanisms separate, because GGUF's status differs per mechanism:
   - classic partial/lowvram loading: GGUF PARTICIPATES (source-confirmed:
     `GGUFModelPatcher.load()` handles `lowvram_model_memory`; issue #375
     shows "loaded partially ... 14663 MB offloaded" for a GGUF model);
   - async-offload prefetch (Dec 2025, default-on NVIDIA, 2 streams): GGUF
     EXCLUDED (maintainer rattus128: "Unfortunately is not implemented for
     GGUF") -- a SPEED cost;
   - aimdo dynamic streaming (Mar 2026): GGUF EXCLUDED (patcher class;
     city96 PR #427 open).

9. **"text_encoder_device() returns the GPU regardless" -- not quite.** It
   still gates on `should_use_fp16()`, and `text_encoder_initial_device()`
   returns the offload device under aimdo. Practical effect on modern NVIDIA
   is GPU compute, but the wording matters if anyone reasons from it.
   (Concurs with report.md/grounding.)

---

## 1. Claim 1 (estimator shape): right conclusion, wrong indictment

The claim says the constants are the wrong SHAPE. The sharper truth is that
there was never a fit to have a shape:

- **(7000, 185) is one equation with two unknowns.** A single datum
  (10,277 MB @ 17 frames) cannot identify an intercept AND a slope; the split
  is invention. It does not even reproduce its own datum (7000 + 185*17 =
  10,145, a 132 MB residual on the only point it has).
- **The datum is from the wrong regime, which no shape can fix.** It was taken
  machine-wide (NVML, ~1.5 GB desktop baseline included) on a 16 GB box under
  ZERO memory pressure -- the regime where ComfyUI offloads nothing. 8 GB
  behavior is pressure-driven offload. No functional form fit to no-pressure
  data predicts pressure behavior. The fix is new data, not new algebra --
  the sweep instinct is correct even though the sweep design is not (sec. 6).
- **A term neither document mentions: CFG.** The frozen recipe pins cfg=5.0,
  so sampling runs cond+uncond -- the activation term of the dominant stage
  is DOUBLED relative to a cfg=1 run. Any refit must pin or model the batch
  factor, and it is exactly why a Turbo-5B at cfg=1 is a memory lever, not
  just a speed lever. Related second-order terms: cost steps per LATENT
  frame ((n-1)/4+1, stepwise every 4 frames), and tiled decode is a mode
  switch, not a scalar.
- **The runtime input has its own double-count, beyond the fit.**
  `free_vram_mb()` is `torch.cuda.mem_get_info` on a warm mid-episode server:
  ComfyUI-cached models from earlier legs (image engine, prior beats) show as
  gone from free, yet ComfyUI would evict them on demand if the render needed
  the room. The `hoisted_vram_mb` correction (eng_wan_ti2v.prepare) proves
  the codebase understands this class -- but it corrects only OTR's own
  hoist, not ComfyUI's reclaimable cache. Post-refit, admission should read
  free + reclaimable (or trigger ComfyUI's free_memory before reading), or
  warm servers will keep over-refusing renders that fit.
- On max(stages) vs pairwise (open question 2): the overlap unit is a BLOCK
  (prefetch depth = NUM_STREAMS, default 2), not a stage, so a pairwise stage
  max overcorrects -- but this graph does not have clean stages anyway (one
  VAE feeds both the pre-sampler latent node and decode; `free_after_use`
  drops RESULTS, deliberately not patchers; BUG_BIBLE 07.22 records a VAE
  staying live through sampling). Concur with report.md: measure the
  continuous lifetime envelope with phase markers; do not model what you can
  measure.

Minor code rot worth fixing while in the file: the FRAME_MOTION_FLOOR comment
(motion_common.py:270-274, "the floor WINS over the budget") describes the
dead clip-fill semantics, contradicted by the S4 raise at :355-362; the
VramPeakProbe header says "TELEMETRY ONLY ... never enforced" while
render_clip's docstring says "assert the mid-render NVML ceiling" (one of the
two is stale); `_DEFAULT_FRAME_COST = (7000, 185)` hands the impossible wan
constants to ANY future engine without a row (dormant landmine); and the
`OTR_VIDEO_COST_OVERHEAD_MB` / `_PER_FRAME_MB` overrides are GLOBAL across
engines and silently swallow malformed values.

## 2. Claim 2 (GGUF opts out): mechanism true, causal story overreached

Confirmed at source: `GGUFModelPatcher(ModelPatcher)`, `clone()` forcing
`__class__` back, #13953 open, `--lowvram` inert under dynamic VRAM. Also
confirmed: no clean core-ComfyUI way to force the text encoder to CPU under
aimdo (`--disable-smart-memory` demonstrably fails for GGUF, issue #14481);
the only targeted control found is third-party (DisTorch2's
`CLIPLoaderGGUFDisTorch2MultiGPU`, which has a device parameter).

But the conclusion drawn -- "we believe it fits BECAUSE it is safetensors, and
our stack is on the legacy 2025 path [therefore 8 GB fails for us]" -- does
not follow:

- The official 8 GB sentence is a July-2025 claim about the LEGACY path
  squeezing a 10 GB fp16 UNET onto 8 GB (sec. 0.2). The legacy path is the
  one path our GGUF verifiably participates in. 2025's 8 GB Wan community ran
  largely ON GGUF via exactly this mechanism.
- What GGUF demonstrably lost in 2026 is the async-offload prefetch path and
  the aimdo upgrade -- both SPEED/residency-efficiency features. "GGUF makes
  8 GB slower than it could be" is defensible; "GGUF is why 8 GB cannot
  render" is not, and the recorded failure needs no GGUF explanation at all:
  the estimator refuses before any loader runs.
- **The aimdo bet has platform edges the docs miss.** comfy-aimdo requires
  Windows 11+ (or Linux), PyTorch 2.8+, CUDA 12.8+, NVIDIA, non-WSL ("WSL may
  never be supported"). A "generic 8 GB consumer" tier includes Windows 10
  boxes that will NEVER run aimdo. So the legacy path must remain a qualified
  configuration REGARDLESS of the fp8-vs-GGUF shootout result -- M1 (GGUF
  legacy) is not the straw man cell, it is the portability floor. Meanwhile
  the floor itself is a wasting asset upstream: PR #14577 (2026-06-23) starts
  nudging users off `--disable-dynamic-vram` with a deprecation timeline, and
  city96 PR #427 (Dynamic GGUF) would moot the whole fork if merged.
  Re-check both before ripping anything.
- The repo's own recorded reason for GGUF -- fp8-scaled umt5 throws
  Float8_e4m3fn on Mac MPS (ComfyUI #9255; eng_wan_ti2v.py:88-91) -- means a
  move to fp8 is a per-platform fork, not a global swap.

Verdict: run the mechanism shootout (report.md Gate 1) exactly because this is
a benchmark question. Nobody anywhere has published GGUF-vs-native under
dynamic VRAM on 8 GB -- confirmed still true by fresh search.

## 3. Claim 3 (canvas): confirmed, one-line fix, two traps

Confirmed at HEAD: `WanTi2vEngine` declares no `render_canvas`; only
`eng_ltx_8gb` (512, 288) does; `render_driver.py:2494` falls through to
`OTR_VIDEO_LANDSCAPE_CANVAS` default 1472x832; the declared-canvas seam
(:2544-2557) wins last; the profile's 832x480 is, per the O1 ruling recorded
at :231-267, a DRIFT GUARD and not authority. `_floor_length` then budgets at
the same 1472x832 (via `_dims`/the request canvas, or its own
`_TI2V_COST_REF_W/H` fallback), so the 3.07x error contaminates both render
and admission consistently.

Trap 1 -- it did not cause, and will not fix, the 8 GB refusal: at 832x480
the gate still demands 9,442 MB free at the floor and 8,306 MB at ONE frame.
Canvas is a 16 GB cost/quality bug and a telemetry-contamination bug.

Trap 2 -- the quality direction. TI2V-5B is a 1280x704@24 model; the model
card offers no 480p mode, ComfyUI ships no 480p preset for it, and the one
8 GB head-to-head that condemned the 5B ("distorted, minimal motion") ran it
at 480x480 -- off-native. Standardizing this tier at 832x480 (or the invalid
768x432 -- 432 fails the /32 declaration gate, as report.md caught) optimizes
VRAM into the model's weakest operating regime and will reproduce the very
evidence being used against the 5B. Frames and pixels are the same latent
currency (VAE 4x32x32): 1280x704 @ 17f is roughly the latent volume of
832x480 @ 38f. The fork the docs never surface: fewer frames at native
resolution vs more frames at 480p. If exact 16:9 is wanted under the /32
gate, the legal ladder is 512x288 / 1024x576 / 1536x864 -- 1024x576 is the
candidate nobody priced. Decide with eyes (operator quality gate per canvas),
not with MB alone.

## 4. Two misses not in any existing document

1. **The rest of the pipeline shares the card, and the clamp cannot see it.**
   `otr_8gb_wan.json` puts the writer LLM on cuda with
   `vram_ceiling_gb: 6.8` (gemma-4-12b Q4_K_M) and routes images to
   `z_image_turbo` -- on the same 8,192 MB device the video stack needs ~7 GB
   of. The single-heavy-engine lease (`_otr_shared.gpu_residency`) serializes
   the VIDEO engines; the LLM lane has its own admission ceiling but is a
   separate mechanism. On a real 8 GB card the writer-phase LLM and the
   render phase cannot overlap AT ALL -- the phase-boundary teardown is
   load-bearing and must be a receipted, measured step in qualification.
   Critically, `--reserve-vram` constrains only ComfyUI's loader planning:
   llama.cpp/transformers allocations ignore it, so a clamped 16 GB canonical
   leg can pass while a real 8 GB box dies in the WRITER phase before video
   ever runs. Full-canonical-leg cells must be judged on whole-leg machine
   peak (writer + image + TTS + render), not the render window.

2. **The negative prompt is the guaranteed cache hit.** `_WAN_DEFAULT_NEGATIVE`
   is a shared constant: one cached CONDITIONING tensor serves every clip of
   every episode, halving encode work immediately -- regardless of how unique
   the shot-derived positives turn out to be (grounding.md is right to demand
   trace data before believing cross-episode positive reuse). Cache the
   negative first; size the rest of the cache from real traces.

## 5. The six "least sure" questions, answered

1. GGUF-vs-fp8 under aimdo: unbenchmarked anywhere (still). DisTorch2 is
   real, GGUF-compatible, single-GPU "virtual VRAM" with system-RAM donor,
   maintainer-claims aimdo coexistence and ~10% over DisTorch V1 (n=1,
   self-reported); it is also the only found path to per-model GGUF CLIP
   device control. New pinned dependency + qualification burden -- Gate 1
   M4 as report.md has it. fp8 umt5 is 6.74 GB vs Q5_K_M 3.86 GiB: under
   aimdo, file size matters less than residency control; on the legacy path
   bigger files mean more offload traffic. Measure, do not argue.
2. max(stages) + reserve vs pairwise: neither -- continuous envelope with
   phase markers (sec. 1; concur report.md). Prefetch overlap is
   block-granular (2 streams default), fragmentation and CFG belong in the
   stage terms, and this graph's stages are not disjoint today anyway.
3. Honest floor for TI2V-5B on 8 GB: no published `max_memory_allocated`
   exists (two independent search passes; confirmed). Closest data: lilting
   4060 480x480@50st "4856 MB loaded" (loader line, not a peak); an HF
   discourse recipe (672x384, 33f, GGUF, TE/VAE offloaded) reported working;
   hum-ma claims Turbo GGUF usable on 4 GB; official Wan-Video repo demands
   24 GB for its OWN offload script -- implementation dependence is total.
   You will be the first to publish a real number; the parameter-analysis
   doc's instinct to publish is right.
4. Keep the 5B? Yes, and try its TURBO first. The anti-5B premise (no
   distill) is false (sec. 0.4); the pro-14B evidence is one deprecated
   merged-checkpoint report; A14B two-expert swap traffic on 8 GB is
   unmeasured; Lightning LoRAs are Apache-2.0 but come as high/low pairs
   with a mid-run expert swap. Order: Turbo-5B GGUF (drop-in, 4 steps,
   cfg=1) -> native fp16/fp8 mechanism shootout -> 14B only if the 5B family
   fails the quality gate at qualified canvases.
5. Landscape: all five rejections verified (Wan 2.5/2.6/2.7 closed -- and
   wan27.org contradicts itself about HF weights, confirming the SEO-farm
   read; HunyuanVideo-1.5 has the EU/UK/KR exclusion + 100M-MAU clause and a
   14 GB stated minimum; LTX-2.3 is 22B/Gemma-3-12B/32 GB prereqs and the
   LTX licence carries the $10M-revenue trigger -- which, note, the shipped
   ltx tiers already accept; Motif-Video-2B publishes 12.53 GB even at
   Q4_K_M; MobileWan is RAI-licensed, BF16-only, no node). Nothing found
   beats Wan 2.2 TI2V-5B for this slot as of today. One under-verified lead
   if the slot ever reopens: Kandinsky 5.0 I2V Lite (MIT, 2B DiT, ComfyUI
   docs exist; the 24 GB pipeline figure is probably dominated by its 7B
   Qwen2.5-VL encoder; zero 8 GB attempts published either way).
6. Embedding cache: right idea, wrong node, and one internal contradiction.
   `WanVideoTextEncodeCached` emits `WANVIDEOTEXTEMBEDS`, consumable only by
   kijai's wrapper sampler -- and this engine runs CORE Wan nodes, having
   excluded the KJ wrapper for recorded reasons (BUG-070 / numpy pin). Its
   cache key hashes only the stripped prompt (no encoder artifact/quant/
   dtype), so it can collide across models. Build the OTR-native
   CONDITIONING cache per report.md's spec (cache-only source node with no
   CLIP input, provenance-keyed, atomic writes, fail-closed); third-party
   packs (e.g. ComfyUI-SaveAndLoadPromptCondition) prove the tensor path.
   The contradiction: "removes 3.861 GiB from the budget entirely" is
   co-resident arithmetic -- under your own Claim-1 shape, and with
   `free_after_use` already freeing the encoder before sampling, a cache
   shrinks the ENCODE stage and the swap/load traffic, not the sampling
   peak. Its real wins are wall clock (the #11081-class encode stalls), RAM,
   and killing encoder-stage load thrash. Do it -- for those reasons, with
   the negative cached first (sec. 4.2).

## 6. The sweep: as designed, it cannot answer the question

Concur with report.md's seven reasons, with independent confirmation of the
two hardest:

- **The T5 axis does not bind.** `CLIPLoaderGGUF` takes no device input --
  this engine's own `session_identity` docstring says so -- and core flags
  cannot place the TE under aimdo (#14481). Cells A/C as written would
  silently measure cell B/D behavior wearing A/C labels, the exact
  fail-open class the LANE-2 tile-geometry fix just killed.
- **The clamp and the gate read different meters.** `--reserve-vram 8`
  constrains ComfyUI's loader budget; OTR's admission reads raw
  `mem_get_info`, which still sees ~14 GB free on the clamped dev box. The
  gate is therefore never exercised at 8 GB-like free, any refit is fit to
  unclamped readings, and torch can physically allocate into the reserved
  region -- so "RESULT SUCCESS under clamp" can certify a config that OOMs
  (or, on Windows drivers with sysmem fallback default-on, silently runs
  10-100x slower) on a real card. Judge cells on measured per-phase
  `max_memory_allocated`/`max_memory_reserved` against an explicit line --
  and set that line at ~6.5-7.0 GB effective, not 8,192: the user's 8 GB
  card also drives their display, and Windows sysmem fallback means the
  failure mode the no-silent-degrade doctrine most needs to catch is a
  silent slowdown no OOM will ever announce. (That last fact is, by the
  way, the strongest argument FOR the predict-do-not-react gate philosophy:
  keep the gate; fix its inputs.)

Deltas to add to report.md's replacement campaign (not a rival plan):

- Gate 1, add cell M5: hum-ma Turbo-5B GGUF, steps=4, cfg=1 -- same loader,
  same graph, tests the largest single activation/wall-clock lever for
  near-zero integration cost. Licence diligence on the quanhaol upstream
  before anything ships. Stamp the different model in the recipe receipt.
- Gate 1, pin the memory backend per cell (`--disable-dynamic-vram` as the
  controlled variable) and record patcher class + aimdo state per receipt
  (Gate 0 already lists this) -- otherwise M-cells on a default-on box
  measure a mixed regime. Note for the tier definition: Win10 users never
  get aimdo, so the legacy result is the portability floor, not a control.
- Gate 3, add one native-canvas cell (1280x704, 17f and 33f, tiled) plus an
  operator quality eyeball per canvas -- the fits-vs-looks fork (sec. 3)
  must be decided by eyes. 1024x576 is the legal exact-16:9 middle rung if
  wanted.
- Gate 4, judge on whole-leg machine peak including writer/image/TTS phases
  and verify the LLM-release interlock explicitly (sec. 4.1); the clamp does
  not constrain non-ComfyUI lanes.
- Admission refit: model per-latent-frame (4n+1 stepwise), pin or model the
  CFG batch factor, and correct warm-server free by ComfyUI's reclaimable
  cache (sec. 1) -- else the first warm-server production leg after the
  refit over-refuses again.

## 7. Order of work (if this window were coding)

1. Declare `render_canvas` on the wan adapter (832x480 behavior-preserving,
   or run the 1024x576 / native-704p question through the quality gate
   first) + the profile drift-guard test. Same change: fix the stale
   FRAME_MOTION_FLOOR / probe docstrings and the `_DEFAULT_FRAME_COST`
   landmine.
2. Build the OTR-native conditioning cache; cache `_WAN_DEFAULT_NEGATIVE`
   day one.
3. Pull the Turbo-5B GGUF next to the base Q5 and wire the prequalification
   cell (steps=4, cfg=1) -- licence check first.
4. Run report.md Gate 0/1 with the M5 cell and per-cell backend pinning;
   then Gates 2-4 with the native-canvas cell and whole-leg peaks.
5. Refit admission from the (phase, latent_frames, canvas, peak) records;
   correct the warm-server free input; keep 17 as the shipped ceiling until
   the data says otherwise.
6. Only then re-open the 14B / model-replacement question -- with the Turbo
   result in hand it may already be closed.

## Primary sources (beyond those in report.md)

- https://github.com/Comfy-Org/docs/discussions/291 (tutorial thread,
  2025-07-28 -- dates the 8 GB sentence)
- https://github.com/Comfy-Org/ComfyUI/discussions/12699 (aimdo default-on
  scope: NVIDIA Win/Linux, "WSL may never be supported")
- https://github.com/Comfy-Org/comfy-aimdo (Win 11+, PyTorch 2.8+, CUDA
  12.8+ requirements)
- https://github.com/comfyanonymous/ComfyUI/issues/11081 (async-offload
  regression, Dec 2025; rattus128: prefetch "not implemented for GGUF";
  TE fix confirmed)
- https://github.com/city96/ComfyUI-GGUF/issues/375 (GGUF partial-load
  telemetry on the legacy path)
- https://github.com/Comfy-Org/ComfyUI/issues/14481 (--disable-smart-memory
  fails for GGUF; no core TE-to-CPU control under aimdo)
- https://github.com/Comfy-Org/ComfyUI/pull/14577 (nudge off
  --disable-dynamic-vram; deprecation trajectory)
- https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers +
  https://huggingface.co/Kijai/WanVideo_comfy/discussions/61 (cfg=1)
- https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo +
  https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF
- https://lilting.ch/en/articles/wan22-comfyui-rtx4060-i2v (the one real
  8 GB head-to-head; 5B at 480x480) +
  https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne (deprecated)
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B (720p/1280x704 support
  statement; Apache-2.0)
- https://github.com/pollockjj/ComfyUI-MultiGPU (DisTorch2; GGUF CLIP
  device parameter)
- https://huggingface.co/tencent/HunyuanVideo-1.5 + its LICENSE (14 GB
  minimum; EU/UK/KR exclusion; 100M MAU clause)
- https://github.com/Wan-Video/Wan2.2/issues/181 (community record of Wan
  2.5+ closed-weights shift)
