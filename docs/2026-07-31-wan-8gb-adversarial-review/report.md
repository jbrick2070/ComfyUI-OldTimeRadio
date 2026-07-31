# Wan 8 GB tier adversarial review

Date: 2026-07-31  
Scope: static repository/runtime audit plus current primary-source research  
Production status: **Wan 8 GB remains draft and unqualified**

## Executive answer

Grok helps with prioritization, but it is not reliable enough to drive the fix
or the product decision.

It is right that the recorded refusal happens in OTR's estimator before Wan can
exercise any offload path; right that the canvas must be authoritative before
calibration; right that stock GGUF does not use AIMDO Dynamic VRAM; and right
that a clean `max(stages)` can miss async transitions.

It does **not** catch the most important evidence correction: the cited July 23
production leg was a 177-frame request on the 16 GB dev RTX 5080, not a
correctly configured 17-frame render on physical 8 GB. It also overstates the
case against legacy GGUF, accepts unsupported community frame/14B claims, calls
an incompatible and weakly keyed embedding cache low-risk, and falsely says no
distilled 5B option exists.

The proposed four-cell CPU/GPU-text-encoder by tiled/non-tiled-VAE sweep would
**not answer the question**. One axis is not currently controllable, it does not
compare the disputed GGUF/native execution mechanisms, and the 16 GB reserve
clamp is not visible to OTR's admission check and is not a physical 8 GB card.

The defensible decision is:

1. Fix request authority and observability before changing the estimator.
2. Compare official native 5B, shipped GGUF 5B, and then FastWan 5B at identical
   requests through the canonical workflow.
3. Measure the continuous lifetime/transition envelope, not an assumed set of
   disjoint stages.
4. Keep the Wan profile draft until a full canonical episode on physical 8 GB
   produces the required artifact and receipts.

No code or workflow was changed in this review, and no GPU render was run. A
static audit cannot create a production bug entry or qualify hardware.

## What the existing evidence actually proves

### The historical artifact is mislabeled in the problem statement

`PBUG-20260723-02` records a profile named `wan_8gb` running on the **16 GB dev
card**. It requested 177 frames; the current model said roughly 30 were
affordable. The verified production defect was that a launch-time 17-frame
ceiling did not reach an already-running server. The ledger-channel fix made the
ceiling authoritative.

That artifact proves a dead configuration channel. It does not prove that a
17-frame request fails on an 8 GB board. The profile label is not hardware
telemetry.

### The refusal arithmetic is real, with one numerical correction

At 832x480, the code's pixel ratio is `0.3260869565` and its per-frame seed is
`60.3261 MiB`. The formula produces:

| Frames | Required free memory |
|---:|---:|
| 1 | 8,306.3 MiB |
| 17 | 9,441.8 MiB |
| 33 | **10,577.4 MiB** |

The problem statement's 10,647 MB value for 33 frames is an arithmetic error.
The code labels these values MB but derives them from binary byte divisions, so
MiB is the accurate unit.

The 7,000 MiB intercept divided by the 0.85 margin is 8,235.3 MiB. The present
guard therefore cannot admit any honest free-memory report at or below 8,192
MiB. That proves the **guard is unusable for the tier**. It says nothing about
whether the underlying graph would succeed after admission.

### The full tier has an earlier writer blocker

At HEAD, the draft profile selects local Gemma-4-12B Q4 on CUDA with a 6.8 GiB
ceiling. The shipped policy test asserts that this exact writer is refused.
Even if selector admission were bypassed, the backend prices its pinned
7,121,860,000-byte weights plus the 2,048-token context at about 8.13 GiB,
before WDDM/display use. The unmodified profile therefore cannot reach video on
physical 8 GB as a full canonical run.

This is a draft-profile qualification blocker, not a new production bug. Gate 0
must select and receipt a viable remote, CPU, or smaller local writer lane
without weakening the writer guard.

### There is no honest frame/resolution floor yet

[ComfyUI says](https://docs.comfy.org/tutorials/video/wan/wan2_2) TI2V-5B
“should fit well” on 8 GB with native offloading. Its linked current template
uses an FP16 5B UNet, scaled-FP8 UMT5, Wan VAE, 1280x704, 121 frames, 20 steps,
and regular VAE decode. That is valuable prior evidence and a concrete example
configuration, but the page publishes no GPU identity, physical-8-GB receipt,
peak metric, host-memory behavior, or repeat protocol.

By contrast, the [official Wan implementation](https://github.com/Wan-Video/Wan2.2#run-wan22)
states at least 24 GB for its 1280x704 TI2V-5B command even with model offload,
dtype conversion, and T5 on CPU. This does not refute ComfyUI—its loader is a
different implementation—but it demonstrates why the implementation and exact
recipe cannot be omitted from an “8 GB” claim.

No primary source found in this review publishes a successful, measured,
physical-8-GB run for the exact OTR request and model stack. Seventeen frames is
currently a configured product floor, not a measured hardware floor.

## Claim-by-claim verdict

### Claim 1: “the estimator has the wrong shape”

**Directionally useful, not proved as stated.**

The current coefficients are poisoned by one absolute, machine-wide NVML peak
at a different canvas on a 16 GB machine. One datum cannot identify both an
intercept and a slope. Absolute NVML used is also mismatched with an admission
input based on current physical free memory; without a matched baseline, desktop
and resident use can be charged once by reducing free and again inside the
fitted intercept.

However, an affine function does not necessarily mean all resources are
co-resident. An empirical affine upper envelope can conservatively bound a
staged pipeline, and the maximum of affine stage curves is piecewise affine.
The real failures are bad evidence, mismatched telemetry semantics, and an
unqualified extrapolation domain.

The proposed replacement is also premature. The current graph:

- uses the same VAE for pre-sampler latent creation and post-sampler decode;
- retains model patchers while `free_after_use` drops Python source results and
  performs only garbage collection plus a soft cache empty;
- may overlap transfers/prefetch with two async streams; and
- hoists the UNet before the render probe, so the probe counts its residency but
  not its load transient.

The internal `run_graph()` path also bypasses ComfyUI's ordinary per-node
finalizer. AIMDO cast buffers and prefetch state are reset only after the outer
OTR batch, so they can span nominal stages or sequential beats. A real lifetime
split needs separate precompute/sample/decode executions plus an explicit
quiescence barrier that resets cast buffers, prefetch queues, and VBAR state.

The quantity to measure is:

`peak = max over time t of all resources live at t, including transition scratch`

Neither `max(clean stages)` nor a fixed `pairwise max` is safe until the graph's
actual lifetimes are proved. Preserve continuous samples and event markers; do
not force every sample into a mutually exclusive stage.

For admission, prefer a recipe-versioned empirical monotone table or upper
piecewise envelope over the qualified canvas/frame domain, with a separately
defined machine reserve. Reject requests outside that domain with a named
error. Do not estimate `W/L/A/workspace/scratch` components from sparse peaks.

### Claim 2: “GGUF opts out of what makes 8 GB work”

**The inheritance fact is true; the causal conclusion is unproved.**

Stock ComfyUI-GGUF forces its UNet and CLIP patchers back to legacy
`ModelPatcher`, so they do not receive AIMDO/VBAR `ModelPatcherDynamic` behavior.
[Dynamic GGUF PR 427](https://github.com/city96/ComfyUI-GGUF/pull/427) remains
open. Dynamic VRAM itself is default only on supported, successfully initialized
configurations—not universally.

Legacy does not mean “no low-VRAM execution.” The base patcher still supports
estimated partial loading, CPU offload, possible host pinning, later partial
reload/unload, and GGUF's per-layer move/dequantization. The precise statement
is: **stock GGUF lacks AIMDO's allocator-pressure adaptation and async
demand-loading path**. Whether that raises or lowers the OTR peak is a benchmark
question.

There is also a historical causal error in Claim 2. ComfyUI published the 5B
8 GB guidance in July 2025, while Dynamic VRAM became the default in March 2026.
AIMDO therefore cannot be the mechanism that originally established that claim;
the native legacy offload path already existed. Dynamic VRAM may improve the
current native path, but that is a new benchmark hypothesis, not the explanation
for the original support statement.

The official 5B workflow is also misdescribed. Its **UNet is FP16**;
`umt5_xxl_fp8_e4m3fn_scaled.safetensors` is the scaled-FP8 component. There is no
official FP8 5B UNet in that workflow. Larger native files may still win through
better residency control, but file size alone cannot predict the result.

DisTorch2/MultiGPU and WanVideoWrapper block swap are genuine alternative GGUF
placement mechanisms only when explicitly installed and wired. They do not
silently improve the current OTR graph and would add a dependency/topology that
must be qualified separately.

The statement that Dynamic VRAM sends the text encoder to GPU “regardless” is
also false. The target depends on dtype/device support, the initial device is
normally the offload device, and GGUF CLIP explicitly starts on CPU before using
its legacy patcher. `--lowvram` is not an isolated T5-placement control under
Dynamic VRAM.

### Claim 3: “the engine never declares its canvas”

**Confirmed at HEAD, with a wording correction.**

Wan has no static `render_canvas`; the render driver therefore falls back to
the shared landscape environment default, 1472x832, unless the process was
booted with another value. A static engine declaration wins last. The profile's
832x480 values are read by and stamped through the director/ledger, so “read by
nothing” is too broad, but they are not authoritative for Wan clip construction.

The behavior-preserving repair candidate is 832x480 because that is what the
draft profile and canonical workflow already declare. The separate 768x432
proposal is invalid under OTR's current rule requiring both axes divisible by
32; 432 is not. Choosing another canvas requires a proved Wan grid contract,
quality decision, profile update, code update, and canonical workflow update in
one change.


This canvas defect is not what triggers today's preflight refusal: the poisoned
7,000 MiB fixed term rejects the tier at either canvas. Canvas authority still
controls actual execution feasibility, output semantics, and every trustworthy
calibration, so it must be corrected before measuring.

## Does Grok's critique help?

| Grok point | Judgment | Why |
|---|---|---|
| Guard fails before render | **Keep** | Correct immediate failure mechanism; does not establish execution feasibility. |
| Canvas first | **Keep** | Required before any calibration. |
| GGUF misses Dynamic VRAM | **Correct wording** | True for AIMDO; false if interpreted as no partial offload/streaming at all. |
| DisTorch2 is an escape hatch | **Experiment only** | Real capability, new dependency, no OTR physical-8-GB proof. |
| Use stage/transition peaks | **Keep** | Continuous overlap must be measured. |
| 480p, 33–49 frames is the community floor | **Reject as evidence** | No controlled primary benchmark or exact stack was supplied. |
| 14B+Lightning beats 5B on 8 GB | **Unproved** | No same-board, same-request peak/time/blind-quality comparison was found. |
| No accelerated 5B exists | **False** | [FastWan TI2V-5B](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers) is Apache-2.0 and supports three-step DMD inference with timesteps `1000,757,522`. |
| Wrapper disk cache is low-risk/drop-in | **False** | Wrong graph type, prompt-only key, direct `.pt` writes, insufficient provenance/validation. |
| Four-cell sweep is close | **Too optimistic** | It is not executable or mechanism-identifying in the current engine. |

Net: Grok improves the list of hypotheses and catches the FastWan opportunity,
but supplies no evidence that permits lowering the guard, promoting 14B, or
shipping Wan as an 8 GB tier.

## Why the proposed four cells do not settle it

### Later supplied critiques

The third critique is the strongest: it adds the pre-AIMDO history, rejects the
affine/co-residency inference, catches the official FP16 UNet, and proposes a
time-indexed live-set model. Its SANA-Video suggestion belongs in research, not
release qualification. The fourth critique contributes the canvas/preflight
causal distinction and the need for a guarded diagnostic bypass, but is wrong
about `--lowvram`, forced GGUF encoder placement, an official FP8 5B UNet, clamp
fidelity, a fixed pairwise/1 GiB reserve, and CogVideoX's license/8 GB evidence.

The independent Claude/Cowork cross-check largely concurs and adds two material
omissions: qualification must include whole-pipeline phase teardown, and the
frozen negative prompt is the first deterministic cache target. It understates
the writer blocker: the current profile is refused before GGUF dispatch and its
backend prices the configured writer above physical 8 GiB. Its Turbo-GGUF lead
is technically real but commercially blocked by conflicting license metadata
and lacks a reproducible 4 GB receipt.

1. **The T5 device axis does not bind.** The Wan recipe exposes no independent
   `t5_device`; `CLIPLoaderGGUF` has no device input. Global boot flags change
   allocator behavior as well, so they confound the intended axis.
2. **It never tests Claim 2.** All four cells retain the same model format and
   patcher. Native Dynamic VRAM versus stock GGUF is the disputed mechanism.
3. **The reserve clamp is not a physical partition.** ComfyUI uses
   `--reserve-vram` as loader headroom. OTR's `torch.cuda.mem_get_info()`
   preflight reads raw physical free memory and does not subtract that reserve.
   A clamped 16 GB run can therefore exercise loader policy while the guard sees
   capacity unavailable on a real 8 GB card. It also does not constrain OTR's
   out-of-band writer/TTS allocators; those are visible only in whole-leg
   telemetry. The shared image/video lease does not cover writer/TTS or prove
   that residual residency was released.
4. **The present guard blocks calibration.** Longer diagnostic rungs are refused
   by the very seed being replaced. Qualification needs an explicit audited
   diagnostic override that is unreachable in production, not a temporary
   global coefficient edit or an ad-hoc graph.
5. **Current telemetry cannot emit stage triples.** It yields one machine-wide
   peak, has no phase markers, and misses the hoisted-UNet load transient. Its
   100 ms polling interval can also miss short workspace/transfer spikes.
6. **One canvas and too few lengths cannot identify the terms.** Decode tiling,
   pixel scaling, frame scaling, transition overlap, and fixed reserve are
   confounded.
7. **A clamp cannot certify customers.** It does not reproduce an 8 GB address
   space, WDDM/display baseline, fragmentation, GPU architecture, PCIe traffic,
   host RAM, or pagefile behavior.

## Replacement qualification campaign

This is deliberately staged so a failed mechanism is eliminated before an
expensive Cartesian sweep.

### Gate 0 — authority and instrumentation

- Declare one legal product canvas and make the engine/profile/director/request
  receipt/canonical workflow agree.
- Add continuous markers for text encode, latent encode, sample, decode,
  transition/prefetch, and explicit unload boundaries.
- Record pre-cell NVML baseline, absolute peak and delta; PyTorch allocated and
  reserved where meaningful; system RAM and pagefile; wall time; actual patcher
  class; AIMDO state; async stream count; loaded/offloaded bytes; server log;
  and canonical asset/OBS proof.
- Record whole-leg writer, Bark, Kokoro, Stable Audio, Z-Image, and Wan phase
  peaks with continuous machine-wide NVML. Require a fail-closed cleanup receipt
  and post-clean baseline at every heavy-phase boundary, including immediately
  before Wan.
- Resolve the profile's writer blocker before any full-leg cell: the current
  local Gemma path requires about 8.13 GiB by its own backend and is refused at
  the configured 6.8 GiB ceiling. Select and receipt a remote, CPU, or smaller
  local lane; do not weaken the writer guard.
- Make qualification cleanup fail closed. Current writer/TTS/image/video cleanup
  paths can log and continue, so a whole-leg maximum alone cannot prove release.
- Add a test-only admission override and frame ceiling that travel through the
  canonical request. Fail closed outside qualification mode.
- Add a separately gated, prequalification-only diagnostic-canvas override so
  alternate-resolution cells cannot silently lose to the static production
  canvas. Stamp both requested and effective canvas in every receipt.
- Fresh-boot every cold cell after the mandated selective reset. Pin ComfyUI,
  plugin, recipe, and model hashes. Replay identical still/audio/prompt/seed.
- Add a purpose-built canonical replay input for fixed prompt, negative prompt,
  init-image hash, seed, canvas, frames, and recipe. The current creative `--set`
  path is not sufficient to prove identical causal inputs.

### Gate 1 — execution mechanism shootout

Before M1, acquire the official native FP16 UNet through a recorded provenance
preflight: source, license, SHA-256, file size, and loader visibility. It is not
installed in the current model inventory, so M1/M2 are blocked until that
receipt exists.

The receipt must preserve resolved model basenames, SHA-256 and sizes; patcher
classes and `is_dynamic`; requested/native/emitted frames; canvas; admission
decision and the counterfactual old-seed decision; free-before and any hoist
correction; cache key/hit/miss; phase peaks; and ComfyUI/AIMDO/GGUF versions.
Current single-mode clip summaries discard several of these fields, so the
receipt path itself is a prerequisite.
Start with the smallest legal production request, fixed canvas, fixed 30-step
OTR recipe, tiled decode on, and at least three cold repeats:

| Cell | UNet / encoder | Runtime purpose |
|---|---|---|
| M1 | Official FP16 native / scaled-FP8 native, Dynamic | ComfyUI's current stated path |
| M2 | Same native artifacts forced onto legacy patching | Isolate Dynamic-versus-legacy on the same files |
| M3 | Q5 GGUF / Q5 GGUF with global Dynamic disabled | Legacy-format/product comparison |
| M4 | Q5 GGUF / Q5 GGUF under the normal Dynamic-enabled server | Actual stock OTR system; GGUF patchers remain legacy while native components/global policy can differ |

M1-versus-M2 is the cleanest global Dynamic-versus-legacy policy contrast on
identical assets; it may change native UNet, encoder, VAE, and runtime behavior
together. M2-versus-M3 remains a format/quantization/loader product contrast,
not a pure quantization A/B. Component crossover cells are warranted only if
attribution remains necessary.

Do not put Turbo-GGUF in the shippable M1-M4 decision. The quantizer card says
Apache-2.0, but the upstream distilled derivative is CC BY-NC-SA. Its exact
`[1000,750,500,250]` plus re-noising sampling contract is also not proved
equivalent to ordinary Comfy `steps=4`. At most, with explicit operator
approval, it is a segregated noncommercial research probe; it cannot qualify
the product or lower the guard. Four steps reduce denoiser work, not a proved
per-step peak, and its Q5 weight footprint is essentially the current Q5's.

Any such research receipt must add repository revision, GGUF/source hashes,
quant, UNet filename, loader commit, sampler, scheduler, shift, steps, CFG, and
license status; the current durable recipe receipt lacks UNet identity.

DisTorch/block placement is a later optional arm, not M1-M4. If selecting a
winner, add same-input blind output quality; peak alone cannot justify a model
change.

### Gate 2 — encoder lifetime and cache

This gate starts only after implementing and proving two first-class seams: a
per-request encoder-placement control, and an OTR-native cache whose provenance,
atomicity, corruption handling, and cache-only no-model-input path have tests.
Global `--lowvram` is not a placement substitute. On the Gate-1 winner, the
smallest useful sequence is:

1. explicit CPU encoder, cache off, tiled decode;
2. explicit GPU/default encoder, cache off, tiled decode;
3. chosen placement, cold cache miss/precompute, tiled decode;
4. fresh-boot cache-only hit, tiled decode; and
5. chosen placement, cache off, untiled decode.

A placement-by-cache Cartesian is wasteful because encoder placement is inert on
a genuine cache-only hit. Precomputed native conditioning must come from a
source with **no model input**.

Measure cold miss, warm hit, and observed hit rate from real OTR prompt traces.
Do not infer placement from `--lowvram`.

Cache the frozen default negative prompt first because it repeats across Wan
clips; continue to key its actual bytes and full encoder provenance. A warm
negative hit removes one encoder forward, not the shared encoder load. Size
positive-prompt caching only from observed trace reuse.

### Gate 3 — decode, frames, and pixels

Alternate-canvas cells require the prequalification-only, receipted diagnostic
canvas override from Gate 0; production authority remains 832x480.

For each surviving mechanism, reuse the 832x480/17/tiled baseline, then add a
compact interaction cross: 512x288 at 17 and 129 frames; 832x480 at 65 and 129
frames; and 832x480 at 129 frames untiled. Run increasing rungs sequentially and
stop/reset after a terminal OOM. These five additional cells expose frame slope,
pixel interaction, and whether tiling matters only at long length without a
full Cartesian sweep. A different ladder is acceptable if the product maximum
changes, but every rung must remain structurally legal and receipted.

Add 1280x704/17/tiled as a model-native quality reference; try 33 frames only
if 17 passes. Compare canvases with a same-input blind quality gate. A reference
cell is not a tier candidate unless it also meets the memory contract.

Use the real recipe for qualification. A cheap low-step screen can find obvious
OOMs but cannot qualify a 30-step OTR recipe. Preserve every continuous
transition peak and report cold/warm host-cache state separately for wall time.

### Gate 4 — stability and transfer

Run three randomized cold repeats of the final 17-frame recipe, then an 8-10
sequential-beat same-server soak. Wan normally emits one capped clip and
ping-pongs it; the soak should exercise that production behavior, not invent a
coverage-planned multi-clip mode. Record per-beat peak, latency, cache state,
host RAM/pagefile, and post-teardown baseline drift. Episode length itself
remains non-gating; the soak tests resource stability.

Dev-card clamp results may be labelled **“16 GB card, 8 GiB
loader-headroom prequalification.”**

Then run one full canonical episode on physical 8 GB and capture writer, Bark,
Kokoro, Stable Audio, Z-Image, and Wan phase peaks. Require a passing fail-closed
cleanup receipt and post-clean NVML baseline at every heavy-phase boundary,
including immediately before Wan; fail qualification if any teardown step
fails. A render-only success does not qualify the product tier.
Clamp results must not be labelled **8 GB qualified**. A physical
8 GB full-canonical run with canonical output proof remains the release gate.

### Acceptance contract

- No OOM, watchdog stall, missing asset, fallback, silent resize, or substitute.
- Predeclared limits for host RAM/pagefile and wall time; avoiding CUDA OOM by
  uncontrolled system-memory thrash is a failure.
- Exact requested canvas, native frame count, steps, scheduler, model hashes,
  patcher, tiling, and placement in the receipt.
- Passing fail-closed phase-boundary cleanup receipts and post-clean NVML
  baselines, including immediately before Wan, plus whole-pipeline phase peaks
  and canonical output proof from physical 8 GB hardware.
- Measured upper envelope plus explicit reserve admits every qualified request
  and rejects every out-of-domain request with a named error.

## Model and cache decisions

### Model candidates

| Candidate | Decision now | Reason |
|---|---|---|
| Official Wan TI2V-5B native | **First experiment** | Matches ComfyUI's stated 8 GB path; still lacks a published physical-8-GB receipt. |
| Current Wan TI2V-5B Q5 GGUF | **Keep as comparator** | Smaller files and legacy offload may work; AIMDO absence is not a verdict. |
| FastWan TI2V-5B three-step | **Second experiment / leading accelerator** | Apache-2.0 and same 5B TI2V family; exact DMD schedule, lower-resolution quality, merge scratch, and Comfy integration must be qualified. |
| Turbo-GGUF TI2V-5B | **Noncommercial research only** | Four-step derivative has a nonstandard schedule/re-noising contract, no reproducible physical-8-GB receipt, and upstream CC BY-NC-SA conflicts with the quantizer card's Apache tag. |
| A14B Q4/Q5 + Lightning | **Separate, lower-priority bakeoff** | Four steps may help, but dual-expert topology, host transfer, and no controlled 8 GB comparison make promotion premature. LightX2V's TI2V-5B item remains a TODO, but FastWan disproves “no 5B accelerator.” |
| LTX-Video 2B | **Strongest existing physical-8-GB evidence, license-qualified** | Its official repo reports RTX 4060 8 GB operation, but v0.9.6+ weights use the custom LTXV 0.X license; entities with at least $10M annual revenue need a separate paid commercial license. |
| SANA-Video 2B | **Research backlog, not an 8 GB candidate yet** | Apache-tagged T2V/I2V with a 832x480x81 example and constant-memory KV design, but no physical-8-GB result, mature OTR/Comfy integration, or completed provenance/legal review. |
| CogVideoX-5B | **Reject for this slot today** | Its model uses the custom CogVideoX license, not Apache-2.0; current Diffusers guidance puts quantized use around 16 GB unless very slow sequential CPU offload is used. |
| Motif 2B / MobileWan / Hunyuan 1.5 / LingBot 1.3B | **Reject for this slot today** | Published peaks/minimums exceed 8 GB, integration is absent, or licensing fails the stated product gate. |

Do not replace 5B with 14B based on anecdotes. Keep 5B as the incumbent
experimental candidate, with FastWan now ahead of 14B in experiment priority.

### Embedding cache

The architecture idea is good: generate conditioning before sampling, release
the encoder, and reuse the tensor. The proposed implementation is not.

`WanVideoTextEncodeCached` returns wrapper-specific `WANVIDEOTEXTEMBEDS`, while
OTR's native sampler consumes `CONDITIONING`. Its key is only a hash of
`prompt.strip()`, omitting encoder artifact/digest, tokenizer/config/template,
precision/quantization, dtype, truncation/max length, extensions, and schema.
It writes `.pt` directly and lacks the atomicity and provenance validation a
shipped cache needs. A cache hit also cannot eliminate a loader if the cache
node has that loader as an upstream input.

A first-class OTR design should:

- precompute positive/negative native conditioning in a distinct phase;
- prioritize the frozen default negative prompt, while recognizing that a hit
  saves one encode forward rather than the shared encoder load;
- expose a cache-only source node with no CLIP/model input;
- key exact prompt bytes and role plus encoder/tokenizer/config/precision/schema
  provenance;
- store CPU tensors in a non-pickle tensor format with a manifest;
- write atomically under a lock and validate shape/dtype/digest on read;
- fail closed on corrupt/stale artifacts; and
- either precompute a miss before the render phase or raise a named missing-cache
  error if the selected production recipe requires precomputed conditioning.

The current wrapper cache should not be wired into OTR.

## Primary sources checked

- [ComfyUI Wan 2.2 guide](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- [ComfyUI official 5B workflow template](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)
- [Wan 2.2 official repository](https://github.com/Wan-Video/Wan2.2)
- [ComfyUI-GGUF source](https://github.com/city96/ComfyUI-GGUF/blob/main/nodes.py)
- [Original July 2025 ComfyUI Wan guide discussion](https://github.com/Comfy-Org/docs/discussions/291)
- [Dynamic GGUF PR 427](https://github.com/city96/ComfyUI-GGUF/pull/427)
- [ComfyUI Dynamic GGUF issue 13953](https://github.com/Comfy-Org/ComfyUI/issues/13953)
- [FastWan TI2V-5B model card](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers)
- [Turbo diffusion upstream license](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/main/LICENSE.md)
- [Turbo reference schedule](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/main/configs/inference/wan22.yaml)
- [Turbo reference inference loop](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/main/pipeline/wan22_fewstep_inference.py)
- [Turbo-GGUF quantizer card](https://huggingface.co/hum-ma/Wan2.2-5B-Turbo-GGUF)
- [LightX2V Wan2.2-Lightning model card](https://huggingface.co/lightx2v/Wan2.2-Lightning)
- [LTX-Video official repository](https://github.com/Lightricks/LTX-Video)
- [LTX-Video license](https://github.com/Lightricks/LTX-Video/blob/main/LICENSE)
- [WanVideoWrapper cache/source code](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/nodes.py)
- [DisTorch2/MultiGPU documentation](https://github.com/pollockjj/ComfyUI-MultiGPU#distorch-how-it-works)
- [Motif-Video-2B model card](https://huggingface.co/Motif-Technologies/Motif-Video-2B)
- [MobileWan model card](https://huggingface.co/Qualcomm-AI-Research/mobilewan)
- [SANA-Video official instructions](https://github.com/NVlabs/Sana/blob/main/asset/docs/sana_video.md)
- [CogVideoX Diffusers memory guide](https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox)
- [CogVideoX-5B model/license](https://huggingface.co/zai-org/CogVideoX-5b)

## Review process receipt

Codex wrote its grounded anchor before panel feedback. Three local read-only
lanes independently audited the repository/runtime, GGUF/Dynamic VRAM, and
models/licensing. The requested live GPT/Gemini/DeepSeek OpenRouter pass was
attempted, but the security boundary rejected transmitting internal repository
text without payload-specific approval. No text was sent and actual spend was
$0.00; the failed manifest is preserved under `roundtable/pass01/`. A separate
Claude/Cowork cross-check was then grounded against the same Windows files. Its
valid whole-pipeline teardown and negative-cache findings are incorporated;
its Turbo recommendation was narrowed after checking the upstream license and
sampling contract.
