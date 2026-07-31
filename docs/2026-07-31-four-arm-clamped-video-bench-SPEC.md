# FOUR-ARM CLAMPED VIDEO BENCH -- tracked spec

**Buildable scope: arms A / B-partial / B. Arm C is CUT. Arm D is BLOCKED.**

**Status:** DRAFT, awaiting operator ratification. NO code, NO workflow edit and
NO GPU run was performed to write it.
**Written:** 2026-07-31, CODER window (bench spec), at HEAD `db0fa304`.
**Hardened by:** kibitz r2 -> r3 -> r4, codex `gpt-5.6-sol` (high), Claude as
anchor panelist and sole judge. Antigravity was quota-held (429) for all three
rounds, so the arc is ONE-SEAT. Round-by-round record:
`docs/2026-07-31-four-arm-video-bench-KIBITZ-JUDGMENT.md`.
**Owns:** the question "which video engine can carry an 8 GB tier".
**Does NOT own:** whether the 8 GB tier is qualified, and -- after r3 -- the
estimator refit. See sections 2 and 9.

## Authority chain

1. `CLAUDE.md` operator directives outrank everything here, including section 3.
2. `docs/2026-07-31-wan-8gb-adversarial-review/report.md` (`aff09bde`, Codex as
   final judge) is the JUDGMENT OF RECORD for the 8 GB question.
3. The three superseded research docs are read only through their superseding
   headers.

Every repository fact was read from the real Windows tree through Desktop
Commander at `db0fa304`, never through the lagging Linux mount.

---

## 1. The question, and the locked scope

**Which video engine can render a shippable clip at a canvas we would actually
ship, at every length an 8 GB tier needs, under an 8 GiB loader clamp, on a
licence we can ship?**

The kickoff commissioned four arms. Grounding them against the tree closed two:

| arm | status | why |
|---|---|---|
| **A** -- Wan 2.2 TI2V-5B Q5_K_M GGUF | **MANDATORY** | the incumbent; the baseline everything is compared to |
| **B-partial** -- GGUF UNet + scaled-FP8 safetensors encoder | **MANDATORY** | promoted from fallback to required: it is the only cell that isolates the encoder bundle (6.3) |
| **B** -- native FP16 UNet + scaled-FP8 encoder | **MANDATORY, gated** | the protected arm; blocked on one weight acquisition (G2) |
| **C** -- FastWan 5B | **CUT** | licence passes; no ComfyUI base graph exists at any priority (6.4) |
| **D** -- LTX 2B | **BLOCKED** | no submit-ready API graph exists (6.5) |

**The mandatory fit matrix is A, B-partial and B at 17 / 49 / 81 frames.** No
`ArmSpec`, mutation, graph or result branch is written for C or D -- a blocked
arm must not add dead branching. Their analysis is retained as the record of why,
and as the starting point for whoever unblocks them.

## 2. BENCH != QUALIFICATION

This bench does not qualify the 8 GB tier, and no sentence in it, in its
results, or in any document quoting it may be worded as if it does.

Qualification must cover EVERY heavy pipeline phase -- writer, Bark, Kokoro,
Stable Audio, Z-Image, then Wan -- with fail-closed cleanup receipts and a
post-clean baseline at every phase boundary, on physical 8 GB hardware,
producing a canonical published episode. This bench does none of that. It smokes
the VIDEO STAGE IN ISOLATION, on a 16 GB card under a loader clamp, because the
configured 8 GB writer is refused at ~8.13 GiB against the profile's declared
6.8 GB ceiling, so a full canonical leg never reaches video at all.

**Mandatory label for every result:**
**"16 GB card, 8 GiB-reserve direct-node prequalification."**
Not "8 GB qualified". Not "stock-loader prequalification" alone -- that drops the
hardware and clamp limitations and can be misquoted.

Four things it cannot do:

- It cannot reproduce an 8 GB address space, a real display/WDDM baseline, real
  fragmentation, real PCIe pressure, or a different GPU architecture.
- It cannot prove teardown. It measures a render, not a pipeline's release.
- It cannot lower the admission guard, edit `FRAME_COST_MODEL`, or promote 14B.
- **It cannot refit the production estimator.** See 9.1 -- withdrawn in r3, and
  the single largest scope change from the first draft.

---

## 3. PRE-BUILD GATES -- nothing is coded until these clear

### G1 (BUILD BLOCKER) -- the canonical-workflow authority conflict

`CLAUDE.md` s0: "EVERY API / headless / soak run MUST LOAD this real JSON --
never a stale copy, a generated `.gen.json`, an ad-hoc graph, or the Linux-mount
snapshot." `docs/PRODUCTION_SPRINT_LESSONS.md:117-119` repeats it: "Always load
`workflows/otr_canonical.json`."

**This bench submits a stock-node API graph, which conflicts with a literal
reading of both.**

It is not a new conflict: `scripts/run_wan_ti2v_bakeoff.py` has loaded the ad-hoc
`scripts/otr_wan_ti2v_bakeoff_gguf.json` since 2026-07-08 and shipped. An
isolated-bench carve-out exists in practice and has never been written down.

**The operator decides before any code is written (O6):** either record a
narrowly scoped exemption naming this runner, this graph, its outputs and its
validation gates -- or redesign the bench to load `otr_canonical.json`. A coder
window must not resolve this quietly.

### G2 -- arm B model provenance

`wan2.2_ti2v_5B_fp16.safetensors` is not on disk. Before arm B renders: immutable
source revision, licence, SHA-256, byte size, and confirmed visibility in the
live `UNETLoader` dropdown.

### G3 -- the helper package must be version-controlled

`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\otr_bakeoff_helper` provides the
measurement nodes this bench depends on, and **it is in no git repository**
(`git -C ... rev-parse` -> "fatal: not a git repository"; `git ls-files` from the
OTR repo -> "outside repository"). A pushed OTR harness cannot depend on
unshipped local code.

**Decision: vendor it into the OTR repo as a tracked bench-only package.** Build
order: vendor -> install -> server restart -> `/object_info` contract check ->
submission.

### G4 -- one owner for port and output root

Two latent bugs in the harness being extended: it hard-codes port 8000
(`run_wan_ti2v_bakeoff.py:79-80,150-160`) while the launcher honours
`OTR_HEADLESS_PORT` (`_otr_soak_server_launch.cmd:76-79`); and it honours
`COMFYUI_OUTPUT` while the launcher hard-codes an output path
(`:60-66`), so the server can write successfully while the harness searches
elsewhere and reports "no mp4 produced". Parse one base URL, derive its port,
propagate to reset/launch/submit/watchdog; choose one absolute output root and
propagate to launcher and validation.

---

## 4. The clamp -- computed, not assumed

Measured on the dev box 2026-07-31:
`NVIDIA GeForce RTX 5080 Laptop GPU, 16303 MiB total, 1651 MiB used at idle`.

We do not own an 8 GB card. An unclamped bench ranks candidates and proves
nothing about fit -- the honest limit already recorded in
`eng_ltx_8gb.py:235-239` against the LTX sweep.

### The channel

`OTR_HEADLESS_RESERVE_VRAM_GB` is consumed in exactly one place,
`scripts/_otr_soak_server_launch.cmd:118-119`, becoming ComfyUI's
`--reserve-vram` at `:138`. It is **boot-only**, so every clamped cell needs its
own server boot.

### The arithmetic

From the installed runtime, `comfy/model_management.py:807-814, 1051-1052`, with
`--reserve-vram N` REPLACING the Windows default (600 MB, +100 MB on 16 GB+
boards) rather than adding to it:

    minimum_inference_memory() = 0.8 GiB + EXTRA_RESERVED_VRAM
    maximum_vram_for_weights() = total * 0.88 - minimum_inference_memory()

**This box, `--reserve-vram 8`** (total 15.92 GiB):

    min_inference = 0.8 + 8.0            = 8.80 GiB
    max_weights   = 15.92*0.88 - 8.80    = 5.21 GiB

**A real 8 GB Windows card, unclamped** (total 8.0 GiB, so EXTRA_RESERVED is
600 MB and the 16 GB+ bump does not apply):

    min_inference = 0.8 + 0.586          = 1.39 GiB
    max_weights   = 8.0*0.88 - 1.39      = 5.65 GiB

Three consequences:

1. **`reserve = total - target` was never the right emulation.** The correct
   target is equal `maximum_vram_for_weights`. Solving
   `15.92*0.88 - (0.8 + R) = 5.65` gives **R = 7.56 GiB**.
2. **`--reserve-vram 8` is conservative by ~0.44 GiB** -- stricter than the card
   it emulates. It is therefore KEPT, as a justified choice rather than an
   unexamined literal. Do not copy the literal 8 to another board: recompute.
3. **The aggregate weights exceed the concurrent weight budget on every arm**
   (A: 8.72 GiB, B-partial: 11.14 GiB, against 5.21 GiB). Note the precise
   claim: each individual component fits under 5.21 GiB, so completion proves
   staging or unloading somewhere -- NOT that every model is partially loaded.
   The bench is a test of staging, which is exactly the point.

Emit `extra_reserved_vram_gib`, `minimum_inference_gib` and
`maximum_vram_for_weights_gib` **per cell, in-process from the running server**.
NVML cannot see a loader reserve.

**Fail closed on NVML.** `total_vram_gb()` currently returns a fictional `16.0`
when NVML raises (`run_wan_ti2v_bakeoff.py:113-123`); under that fallback the
clamp is computed from a fiction. It must raise instead.

### What the clamp does NOT do -- and the one place that helps

`--reserve-vram` is loader accounting; it allocates nothing. OTR's own admission
check reads driver-true free memory (`motion_common.py:293`
`torch.cuda.mem_get_info()`), and nothing in `compute_real_frame_budget`
subtracts a reserve. So:

- **Against us:** the box still has ~14.6 GiB physically free, so an allocation
  escaping the loader's accounting will not OOM the way it would on a real 8 GB
  card. The clamp exercises loader/offload POLICY and nothing stronger.
- **For us:** `compute_real_frame_budget` is called only from
  `eng_wan_ti2v.py:97,698,740`, so an HTTP stock-node graph cannot raise
  `MotionBudgetError` and **the bench needs no diagnostic guard override**. The
  same is true of `eng_ltx_8gb`'s own ceiling refusal (`:1358-1369`). The bench
  therefore validates NEITHER admission channel -- intentional, and the reason it
  cannot clear the four-way LTX ceiling disagreement noted in 6.5.

---

## 5. Base graphs: where each arm comes from

DO NOT REINVENT THE WHEEL is a correctness rule here, not a time saver: if arms
are hand-built, a difference between arms could come from OUR wiring rather than
the engines, and the bench would be measuring us.

### 5.1 The shipped variant workflows cannot carry an arm

The kickoff's priority 1 named `workflows/variants/otr_8gb_wan.json` and
`otr_8gb_ltx.json`. They cannot be base graphs, for a structural reason.

Both are the FULL 23-node / 56-link OTR canonical graph -- node-type multiset
identical to `workflows/otr_canonical.json`, verified by set difference (both
directions empty). **They contain ZERO stock ComfyUI nodes**: no
`UnetLoaderGGUF`, no `CLIPLoader`, no `VAELoader`, no `KSampler`. A whole-file
regex for weight extensions returns `[]` for both -- neither names a video weight
anywhere.

The video engine is ONE STRING: `OTR_VideoRenderBatch` widget 4 (`"wan_ti2v"` /
`"ltx_8gb"`), with canvas and frame budget on `OTR_VideoDirector`. Loaders,
sampler, VAE and tiling are built in Python by the adapter. Their own paired
`.env.json` says they are not even the submission artifact: "never this preset
as the workflow".

There is no loader-node surface to vary, so arms cannot be expressed as variant
JSONs.

### 5.2 The real base graph for arms A and B

`scripts/otr_wan_ti2v_bakeoff_gguf.json` (3609 bytes, 114 lines, modified
2026-07-08) is an API-format ComfyUI graph of stock nodes. Its own header states
it "mirrors `eng_wan_ti2v._build_graph` exactly".

**That claim is true through the decode terminal, and not past it. Two
divergences from production, both stamped in every receipt:**

1. **Encode path.** The JSON continues into `CreateVideo` + `SaveVideo`; the
   engine reads the IMAGE batch out of `_TERMINAL = "vaedecode"` and encodes via
   `wrapper_bridge.encode_frames_to_silent_mp4`.
2. **UNet lifetime -- the important one.** `eng_wan_ti2v.prepare()` (`:428-489`)
   hoists `_SESSION_NODES = ("unet",)`, and `_build_graph` ends with
   `for nid in set(external_results or ()): graph.pop(nid, None)` -- so the
   executed production graph **has no unet node at all**, and runs with
   `free_after_use=True`. The bench graph loads the UNet inside every prompt and
   runs under ComfyUI's own executor caching.

**Correct wording, used throughout: "a structural surrogate for the UNPREPARED
graph through decode."** Every receipt stamps `adapter_hoist=false`,
`free_after_use=false` and the encode path. Production-lifetime claims are
prohibited from this bench's output.

### 5.3 Arms A and B share one base-graph shape

ComfyUI's official 5B template (`Comfy-Org/workflow_templates`,
`templates/video_wan2_2_5B_ti2v.json`, file version `0.4`, frontend `1.27.10`,
node cnr `0.3.45`) has 11 executable nodes plus a `MarkdownNote`. Ours has 11.
**The topology is identical.** The entire difference is three node-class
substitutions:

| role | arm A (ours) | arm B (official) |
|---|---|---|
| UNet loader | `UnetLoaderGGUF` -> `Wan2.2-TI2V-5B-Q5_K_M.gguf` | `UNETLoader` -> `wan2.2_ti2v_5B_fp16.safetensors` |
| text encoder | `CLIPLoaderGGUF` -> `umt5-xxl-encoder-Q5_K_M.gguf` | `CLIPLoader` -> `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAE decode | `VAEDecodeTiled` (256/64/16/8) | `VAEDecode` (untiled) |
| all others | `ModelSamplingSD3`, `CLIPTextEncode` x2, `VAELoader`, `LoadImage`, `Wan22ImageToVideoLatent`, `KSampler`, `CreateVideo`, `SaveVideo` | identical |

Pin the template by **immutable commit SHA and file SHA-256**, not
`refs/heads/main`, and keep a reviewed local copy. A moving ref cannot define a
reproducible `ArmSpec`.

Precision kept throughout, per the report: the official path is an **FP16 UNet
plus a scaled-FP8 text encoder**, not "fp8" wholesale. There is no official FP8
5B UNet in that workflow.

---

## 6. The arms

The live models root is **`C:\ComfyUI-Models`** per
`C:\Users\jeffr\AppData\Roaming\ComfyUI\extra_models_config.yaml`;
`Documents/ComfyUI/models` is an empty skeleton. All byte counts below are from
that tree.

### 6.1 Arm A -- Wan 2.2 TI2V-5B Q5_K_M GGUF (MANDATORY baseline)

| field | value |
|---|---|
| base graph | `scripts/otr_wan_ti2v_bakeoff_gguf.json` |
| UNet | `diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf` -- 3,810,603,360 B (3.549 GiB) |
| encoder | `text_encoders\umt5-xxl-encoder-Q5_K_M.gguf` -- 4,145,878,880 B (3.861 GiB) |
| VAE | `vae\wan2.2_vae.safetensors` -- 1,409,400,960 B (1.313 GiB) |
| weights total | 8.723 GiB |
| licence | Apache-2.0 (`Wan-AI/Wan2.2-TI2V-5B` metadata). UNet pinned in `otr_wan_ti2v_manifest.json`, sha256 `4424633a...42dba`, size matching disk |
| runnable today | **YES** |

**Delta from base:** canvas, frames, seed, measurement hooks, and
`SaveVideo.filename_prefix` per cell.

**Receipt gap to close:** `otr_wan_ti2v_manifest.json` has `unet` and `vae` roles
only -- **no text_encoder** -- though `otr_8gb_wan.launch.md` preflights
`umt5-xxl-encoder`. Arms B and B-partial change the encoder, so encoder source,
SHA-256 and size are recorded directly per cell.

### 6.2 Arm B -- native FP16 UNet + scaled-FP8 encoder (protected)

| field | value |
|---|---|
| base graph | official template (5.3), pinned by commit SHA |
| UNet | `wan2.2_ti2v_5B_fp16.safetensors` -- **NOT ON DISK** |
| acquisition | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`, `split_files/diffusion_models/` -- the same repo our manifest already pins for the Wan VAE |
| encoder | `text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors` -- 6,735,906,897 B (6.273 GiB), **on disk** |
| VAE | `wan2.2_vae.safetensors`, same file as arm A |
| licence | Apache-2.0 |
| runnable today | **NO** -- gated on G2 |

**Deltas from the official base graph, each justified:**

| delta | why |
|---|---|
| 1280x704 -> 832x480 | control parity; the canvas the `wan_8gb` profile declares |
| 121 -> the ladder | control parity |
| seed 898471028164125 -> 42 | control parity; the value the shipped 8 GB presets carry |
| `uni_pc`/20 steps/shift 8 -> `euler`/30/shift 5.0 | **channel isolation.** The arm-B question is whether native packaging changes the memory profile, not whether `uni_pc` is better. `PRODUCTION_SPRINT_LESSONS` s8: "Change one meaningful variable per comparison." |
| `VAEDecode` -> `VAEDecodeTiled` 256/64/16/8 | matches arm A and production |
| prompt / negative -> the bench pair | control parity |
| measurement hooks | mandated; see section 9 |

### 6.3 Arm B-partial -- GGUF UNet + scaled-FP8 encoder (MANDATORY)

Promoted from "documented fallback" to **required**, because A -> B changes the
UNet bundle AND the encoder bundle at once. The only honest decomposition is:

    A  ->  B-partial   isolates the ENCODER bundle (GGUF -> scaled-FP8 safetensors)
    B-partial  ->  B   isolates the UNET bundle    (GGUF -> native FP16)

A safetensors encoder receives `ModelPatcherDynamic` while a GGUF UNet is forced
back to legacy `ModelPatcher`, so this is the cell that actually tests half the
AIMDO question -- and the encoder is the LARGEST single weight in the stack
(6.273 GiB scaled-FP8 vs a 3.549 GiB UNet).

**Runnable today with no code change.** Verified rather than assumed:
`_clip_loader_mode()` (`eng_wan_ti2v.py:511`) reads `OTR_WAN_TI2V_CLIP_LOADER`,
else infers from the filename extension; `_loader_names()` (`:576`) reads
`OTR_WAN_TI2V_CLIP_NAME`; **neither is prequalification-gated**; and the only
filename allow-list, `_WAN22_VAE_ALLOWED` (`:84`), constrains the VAE only, so
the scaled-FP8 encoder is not refused. In the stock-node graph it is two JSON
fields: `class_type` `CLIPLoaderGGUF` -> `CLIPLoader`, `clip_name` -> the
safetensors file, adding the `device: "default"` input the core loader takes.

**Weights total 11.135 GiB.** Prediction recorded in advance, which is what makes
this a test rather than a fishing trip: **if B-partial passes at any length,
staging is demonstrably available to us for a stack well over the weight budget;
if it fails at every length, the scaled-FP8 encoder cannot stage and arm B's
upside shrinks to the UNet alone.**

### 6.4 Arm C -- FastWan 5B: CUT (licence passes, base graph does not exist)

**Licence: PASS.** `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` declares
`apache-2.0`; the upstream `Wan-AI/Wan2.2-TI2V-5B` it is built on is also
`apache-2.0`, checked separately rather than inherited. The derivative chain is
clean -- the material difference from Turbo-GGUF, which is banned for the
opposite reason (CC BY-NC-SA upstream).

**Base graph: FAIL, at every priority.** No shipped variant; no ComfyUI official
template; **no ComfyUI workflow from the model author** -- the card is
Diffusers-only, uses `WanDMDPipeline`, and mentions no ComfyUI support. A search
for a ComfyUI-repackaged FastWan surfaced only stock Wan 2.2 repacks. That leaves
priority 4, hand-build, which the kickoff requires be said plainly. Saying it.

Two compounding problems: the weights are Diffusers layout, not a ComfyUI
single-file artifact, so conversion is itself an unproven delta whose errors are
indistinguishable from memory findings; and the 3-step DMD schedule with fixed
timesteps `1000,757,522` is not ordinary `KSampler steps=3` -- the same class of
problem that disqualified Turbo-GGUF's schedule.

**CUT from this campaign.** It stays on the roster as the leading potentially
shippable accelerator, behind a conversion/loader receipt and a same-input blind
quality check, as a separate research probe. Nothing named `fastwan` exists in
`C:\ComfyUI-Models` today.

### 6.5 Arm D -- LTX: BLOCKED (no submit-ready graph)

The kickoff named LTX-Video 2B v0.9 and said "we already ship the adapter" --
two different models, and only the second half is true.

| | kickoff's literal arm | what we ship |
|---|---|---|
| checkpoint | `ltx-video-2b-v0.9.safetensors` -- 8.727 GiB, on disk | `ltxv-2b-0.9.8-distilled.safetensors` -- 5.905 GiB, on disk |
| adapter | none | `eng_ltx_8gb` |
| recipe | none calibrated | `LTX8_RECIPE_V2` -- 8 steps, cfg 1.0, `t5_device="cpu"`, tiled VAE |
| canvas | none | `render_canvas = (512, 288)` (`:518`) |

**The blocker is that neither has a submit-ready base graph.**
`eng_ltx_8gb._build_graph` (`:1153-1194`) emits abstract aliases
(`"class": "clip"`) resolved by `_node_candidates`, not an API prompt. And
`scripts/otr_ltx_av_q_bakeoff_distilled_native.json` parses with top-level
`last_node_id / last_link_id / nodes / links / groups / config` -- **litegraph UI
format, and the LTX-AV family, not `ltx_8gb`.**

**Unblocking requires** a vendored API-format LTX 0.9.8 graph, pinned by repo
commit and file SHA-256, validated against live `/object_info`, with its own
8-step timing contract. Until then no `ArmSpec` is enabled.

Two further facts for whoever unblocks it:

- **Off-native canvas.** 832x480 is 2.708x the pixels of LTX's shipped 512x288.
  It is structurally legal (both axes /32; 17/49/81/121 all satisfy `min 9,
  quantum 8`), but an arm-D failure at 832x480 would NOT mean the shipped
  `ltx_8gb` tier fails. Arm D needs its own ladder at 512x288 as well.
- **Licence question, gating shipping not measuring.** `Lightricks/LTX-Video` is
  `license: other`; 0.9.6+ including `0.9.8-distilled` falls under
  `LTX-Video-Open-Weights-License-0.X.txt`, which says entities with annual
  revenue >= $10,000,000 "are eligible to obtain a paid commercial use license".
  Two readings exist and I did not resolve it -- see O4. Outputs are ours.

Also recorded but NOT in this bench's path: the LTX frame ceiling disagrees
across four channels -- `49` (render-batch widget), `97`
(`otr_8gb_ltx.env.json`), `0` (director, unpinned), `161` (profile
`video.max_render_frames`). The env file's own `_ceiling_note` flags it. The
stock-node bench bypasses all four; nothing here validates or fixes it.

---

## 7. Controls and the matrix

### 7.1 The fixed cell

Driving stock-node graphs directly puts every control on the surface -- driving
through the adapters buries them (the single-engine smoke pins seed to 7 by
arithmetic, uses a literal prompt, and takes canvas from an env var).

| control | value | why |
|---|---|---|
| declared canvas | **832x480**, every arm | the `wan_8gb` profile's declared canvas; both axes /32, satisfying OTR's grid rule (this is why 768x432 was invalid -- 432 is not) |
| seed | **42** | the value the shipped 8 GB presets carry |
| init image | **`c02_466a19906ccb.png`** | the still the existing bakeoff already stages; keeps results comparable to the 2026-07-08 run |
| positive prompt | `subtle natural motion, cinematic lighting, detailed` | same continuity |
| negative prompt | `low quality, worst quality, blurry, distorted, watermark, text, static` | same continuity. NOTE: this is the bakeoff's negative, not production's `_WAN_DEFAULT_NEGATIVE`. It is the one delta landing INSIDE a measured stage, so its SHA-256 is stamped per cell and may not drift between arms. |
| recipe | euler / simple / 30 steps / cfg 5.0 / shift 5.0 / denoise 1.0 | the real production recipe; a low-step screen cannot qualify a 30-step recipe |
| decode | `VAEDecodeTiled` 256/64/16/8 | production's setting |
| batch_size | 1 | production |
| fps | 24 on `CreateVideo` | the bakeoff graph's value; production encodes at 25. fps does not move the memory profile, so it is recorded, not corrected. |

**Staging:** the headless server reads `LoadImage` from the install-root input
directory, so the still is copied from `C:\Users\jeffr\Documents\ComfyUI\input`
to `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\input` before the run. Reuse
that step verbatim.

**Encoder placement is not intervened on.** `CLIPLoaderGGUF` takes no `device`
input; the core `CLIPLoader` gets `device: "default"`, ComfyUI's own default. No
`--lowvram`, no per-arm device forcing. (Arm D's shipped recipe pins
`t5_device="cpu"`, a real placement intervention -- relevant only when arm D is
unblocked, and it must then be labelled, never averaged against a Wan arm.)

### 7.2 Lengths -- judge on the spread, not the minimum

An arm that only passes at the shortest length is a demo, not a tier. The ladder
is legal on BOTH frame contracts, which is what would make a future cross-family
comparison honest at all:

| length | Wan (`min 17`, `quantum 4`) | LTX 8GB (`min 9`, `quantum 8`) |
|---:|---|---|
| 17 | min | 9 + 8 |
| 49 | 17 + 4x8 | 9 + 8x5 |
| 81 | 17 + 4x16 | 9 + 8x9 |
| 121 | 17 + 4x26 | 9 + 8x14 |

Contracts: `eng_wan_ti2v.py:259-266`, `eng_ltx_8gb.py:534-541`.

- **17** is the configured product floor -- a configured floor, not a measured
  hardware floor. This bench is how it stops being an assumption.
- **49** is what the existing bakeoff graph runs, so arm A at 49 is directly
  comparable to 2026-07-08.
- **81** decides tier versus demo.
- **121** is NOT in the first build (see 7.3). It is the official template's own
  frame count, so a later pass at 121 would answer ComfyUI's published claim
  directly.

### 7.3 The mandatory matrix, and what is cut from the first build

**Mandatory fit matrix: {A, B-partial, B} x {17, 49, 81} = 9 cells.**
Every one is required. A missing, skipped, telemetry-invalid, OOM, or
decode-invalid required cell **blocks PASS for that arm** -- it may never shrink
the denominator.

**Cut from the first build** (none is needed to answer the locked 8 GiB
engine-selection question, and each multiplies runtime or topology):

- the `vram_full` and `vram_6gb` clamp tiers. `CLAMP_TIERS` currently defaults to
  all three (`run_wan_ti2v_bakeoff.py:93-99,529-540`), which would triple the
  matrix and include an unclamped run despite "every cell runs clamped".
  **8 GiB is the sole campaign clamp**; other tiers become opt-in diagnostics
  excluded from the greenlight aggregate.
- the 121-frame rung, the untiled-decode cell, and the official-recipe cell.
- arm C and arm D branches entirely.

### 7.4 Winner and repeats

"Three cold repeats of the winner's full ladder" was undefined and was mis-costed
as three cells in the first draft. It means three repetitions of each required
rung.

**Ranking rule (default, ASSUMPTION -- operator may override):** lowest
worst-case `peak_delta_mib` at 81 frames; ties broken by wall time. **Exactly
three cold measurements per required winner rung**, each after a full reset.
Only the winner repeats.

### 7.5 Cost

9 mandatory cells + 9 winner repeats = 18 boots, each with a 30-step render.
**Every cell needs its own server boot** because the clamp is boot-only. This is
a multi-hour GPU campaign and belongs to a RENDER window, not a coder window.
Rungs ascend within an arm; a terminal OOM stops that arm's ladder and triggers a
reset.

---

## 8. Campaign manifest -- written before cell 1

One manifest, hashed, fixed for the whole campaign. **Any mismatch fails the
cell**, and a campaign whose manifest changed mid-run is void, not averaged.

Pinned identities: ComfyUI revision, ComfyUI-GGUF plugin revision, vendored
helper revision, harness revision, base-graph SHA-256 per arm, model file
SHA-256 + byte size + immutable source revision per role, negative-prompt
SHA-256, positive-prompt SHA-256, init-image SHA-256, recipe, probe topology,
GPU UUID, driver version, and the resolved output root and port.

Per cell, additionally: `host_cache_state` (cold/warm -- the three winner repeats
must be genuinely cold), and the in-process clamp triple from section 4.

`cell_id` includes **arm, clamp, frames, recipe, decode mode and repeat index**,
so ladder rungs and repeats cannot overwrite each other's outputs. The current
runner reuses a `filename_prefix` shape and would collide.

---

## 9. Measurement

### 9.1 What was withdrawn, and why

The first draft claimed the bench would do double duty: pick a winner AND yield
the `(stage, frames, peak_mb)` triples needed to refit the estimator into a
max-over-stages shape. **That claim is withdrawn.**

The reason is structural. The Wan graph's conditioning, image-latent and model
branches are independent until `KSampler`, so a marker after one branch does not
constrain the others -- Bug Bible **BUG-05.05**: "No data dependency between
them. ComfyUI executor picks any order." The obvious remedy is to create the
dependency and force a total order. But forcing the order **changes the memory
schedule under test**: ComfyUI's loader decisions depend on execution order, so a
forced graph has a different peak than the graph we care about. A measurement
that perturbs the thing it measures cannot also be the fit verdict.

Add the lifetime mismatch from 5.2 -- production hoists the UNet out of the graph
entirely and runs `free_after_use=True`, the bench does neither -- and the
conclusion is unavoidable: **production estimator calibration requires
instrumenting the real adapter path (`prepare` + `render_clip`). It is not this
bench's job.** This bench answers "which engine".

### 9.2 Fit metric -- the greenlight authority

    peak_delta_mib = peak_nvml_mib - desktop_baseline_nvml_mib

`desktop_baseline_nvml_mib` is taken **pre-boot, after quiescence**, and is the
sole owner of the delta. `pre_submit_nvml_mib` (post-boot) is ALSO recorded, but
only as server-overhead diagnostics -- it never enters the bar. The first draft
contradicted itself on this; it is now settled.

Continuous NVML + psutil sampling at 0.25 s, preserving the FULL timeline, not
just the scalar high-water mark. Samples must bracket submission through history
completion, carry `time_ns` on one clock, and satisfy a maximum allowed
inter-sample gap.

**Fail closed.** `PeakSampler` currently swallows read failures and records `-1`
(`run_wan_ti2v_bakeoff.py:219-271`) -- and `-1 <= 7168` evaluates TRUE, so a
telemetry failure would silently PASS an arm. That is a correctness bug in the
bar, not a robustness nit. Store initialisation and runtime exceptions, count
samples, join the thread, and **fail the cell unless both NVML and psutil
produced valid samples across the entire render window, with zero sampler
errors**.

### 9.3 Per-stage torch measurement -- the order-safe part ships, the rest is O7

The operator's hard requirement was `torch.cuda.max_memory_allocated` with
`reset_peak_memory_stats` between stages. The panel's finding (9.1) is that a
full four-stage split cannot be had without perturbing the measurement. Rather
than trade the requirement away, it splits cleanly:

**Ships in the first build -- order-safe by construction, no new nodes.** The
already-installed `otr_bakeoff_helper` provides `OTR_BakeoffVramReset` (LATENT
passthrough, `reset_peak_memory_stats`, always-dirty `IS_CHANGED` via uuid4) and
`OTR_BakeoffVramProbe` (IMAGE passthrough, logs `max_memory_allocated` AND
`max_memory_reserved`), both printing with `flush=True` to stdout -- the exact
server-log channel the harness already parses. In the Wan graph the chain
`Wan22ImageToVideoLatent -> [reset] -> KSampler -> VAEDecodeTiled -> [probe] ->
CreateVideo` is a strict data dependency, so **the sample+decode segment is
measured deterministically with no forced ordering and no new topology.** That is
a genuine per-stage torch peak for the dominant segment, available today.

Subject to G3: the helper is vendored into the OTR repo first, its probes stop
swallowing exceptions (today they `return (images,)` after a failure, so a
measurement failure yields a clip with no data and no error), and each probe line
gains `cell_id`, `phase`, `seq` and `time_ns` on the same clock as the NVML
samples.

**Deferred to operator decision O7 -- the text-encode and image-encode
boundaries.** These are the ones that need forced order, three additional typed
probe classes (MODEL and CLIP inputs for `unet_patcher_class` /
`clip_patcher_class`, plus explicit positive and negative CONDITIONING wiring --
one passthrough cannot carry both), and a second run mode. If the operator wants
them, they run as an explicitly DIAGNOSTIC campaign whose stage data may never
drive a production estimator refit or a fit verdict.

Both `max_memory_allocated` and `max_memory_reserved` are recorded, and both stay
DIAGNOSTIC: `max_memory_allocated` is per-process and counts only the torch
caching allocator -- it cannot see the CUDA context, non-torch workspaces, GGUF
dequantisation scratch, or another process. **The fit verdict is 9.2's NVML
delta.**

### 9.4 Receipt schema

Per cell, extending the existing 17-field bakeoff result:

    cell_id, arm, arm_label, repeat_index, base_graph_path, base_graph_sha256,
    base_graph_provenance,
    unet_file, unet_sha256, unet_bytes, unet_loader_class,
    clip_file, clip_sha256, clip_bytes, clip_loader_class,
    vae_file, vae_sha256, vae_bytes,
    canvas_w, canvas_h, frames, seed, steps, cfg, shift, sampler, scheduler,
    denoise, tiled_decode, tile_geometry,
    positive_prompt_sha256, negative_prompt_sha256, init_image_sha256,
    clamp_label, reserve_vram_gb, total_vram_gib,
    extra_reserved_vram_gib, minimum_inference_gib, maximum_vram_for_weights_gib,
    desktop_baseline_nvml_mib, pre_submit_nvml_mib, peak_nvml_mib,
    peak_delta_mib, nvml_sample_count, nvml_max_gap_ms, sampler_errors,
    peak_sysram_mib, sysram_delta_mib, pagefile_commit_mib,
    torch_sample_decode_peak_mb, torch_sample_decode_reserved_mb,
    unet_patcher_class, unet_is_dynamic, clip_patcher_class, clip_is_dynamic,
    adapter_hoist, free_after_use, encode_path,
    wall_s, s_per_it, s_per_it_source, host_cache_state,
    spill_signatures_matched, prompt_executed_in,
    asset_path, asset_bytes, ffprobe_w, ffprobe_h, ffprobe_frames, ffprobe_fps,
    ffprobe_codec, ffprobe_duration_s,
    comfyui_revision, gguf_plugin_revision, helper_revision, harness_revision,
    gpu_uuid, driver_version, status, error

`unet_patcher_class` / `clip_patcher_class` are split deliberately: B-partial is
GGUF UNet + safetensors CLIP by construction, and a singular field cannot
represent it. Without these two pairs, an A-vs-B peak difference is a number
without a mechanism.

---

## 10. Harness -- a second one should not exist

The kickoff named `scripts/queue_smoke.py`; **that file does not exist**. The
real inventory was audited in full.

| script | verdict |
|---|---|
| `_otr_single_engine_smoke.py` | Cannot carry it. `--engine` is a first-class flag -- the only script where it is -- but it loads no workflow, boots no server, reads no clamp, writes no results, takes no peak. Canvas comes from an env var, prompt is a literal, seed is pinned to 7 by arithmetic (`idx*1009+7`, idx always 0). It also discards the peak the engines DO stamp: `render_driver._clip_summary` (`:4399`) returns six keys and `vram_peak_mb` is not one. |
| `otr_wan_smoke.py` | Cannot carry it. Richest knobs and an `nvidia-smi` poller, but the graph is a hardcoded Wan **14B i2v** node set; no engine axis, no workflow, no clamp, no arm loop, no results file. |
| `otr_video_gpu_smoke.py` | Cannot carry it. `--engine` is `choices=list(ENGINES)` and `ENGINES` omits both `ltx_8gb` and `wan_ti2v`. Worse, it drives adapters **in-process, outside ComfyUI's executor thread**, which the repo's own comments say is where `model_management` does not evict the encoders -- its VRAM numbers are not production-comparable by our own rule. |
| `otr_queue_smoke.py` | Cannot carry it. `--workflow` is real and can load the 8 GB variants -- the one requirement it meets -- but it has no engine flag, is structurally barred from gaining one (`otr_api.patch_creative` refuses anything off `CREATIVE_WHITELIST`), and measures no VRAM. |
| `otr_api.py` | Not a runner: no argparse, no `__main__`, no VRAM code. The correct transport dependency, never the bench. |

**`scripts/run_wan_ti2v_bakeoff.py` already IS a clamped multi-arm bench.** It
has `CLAMP_TIERS` + `reserve_for_target()`, `boot_server(reserve_gb)` through the
proven launcher passed as the executable, a selective-CIM `reset_box()`, a
`PeakSampler` context manager, incremental JSON + Markdown results, a
`--dry-validate` preflight, and a `REQUIRED_CLASSES` node-currency check.
**Because it exists and works, a second harness needs no further justification:
it should not exist.**

### The delta

| # | change | why |
|---|---|---|
| D1 | Replace `QUANTS` with a validated `ArmSpec`: label, base graph path + SHA-256, node-class substitutions, model names, **arm-owned step count**, expected API schema, required classes, and stage contract. | The arm axis is the point; today it is a quant filename. |
| D2 | Rewrite `mutate()` to be substitution-driven -- able to change a node's `class_type` AND its inputs, adding new required inputs and **removing stale class-specific ones**. Today it is `_find(p, "UnetLoaderGGUF")` hardwired, so a safetensors UNet returns `None` and raises `TypeError`; there is no CLIP hook at all. | Arms differ by node CLASS, not filename. |
| D3 | Add the length ladder as an outer loop, setting `Wan22ImageToVideoLatent.length` per rung. Today `length: 49` is frozen. | Judge on the spread. |
| D4 | Pin 8 GiB as the sole campaign clamp; other tiers opt-in and excluded from the aggregate. | 7.3. |
| D5 | Vendored helper probes wired on the latent/image edges; `cell_id`/`phase`/`seq`/`time_ns` on every probe line and NVML sample; both fail closed. | 9.3. |
| D6 | Record `desktop_baseline_nvml_mib` (pre-boot) and `pre_submit_nvml_mib` (post-boot); emit the in-process clamp triple; **raise instead of the fictional `16.0` NVML fallback**. | 4, 9.2. |
| D7 | `reset_box()` must ASSERT, not print: residual server count 0 AND NVML settled to baseline before boot. Today it prints the count and never checks `nvidia-smi`. | CLAUDE.md s4 is a hard gate; booting onto residue measures garbage. |
| D8 | Capture `unet_patcher_class` / `unet_is_dynamic` / `clip_patcher_class` / `clip_is_dynamic`. | Without them the mechanism question stays unanswered. |
| D9 | Remove the global `STEPS = 30` from the `wall / STEPS` fallback -- step count becomes arm-owned. | LTX runs 8 steps; the global corrupts s/it. |
| D10 | Replace the bare `offload` spill regex with named signatures; add explicit system-RAM delta and pagefile-commit counters with numeric rejection thresholds. | `offload` is normal loader chatter, so the current hint fires on healthy runs. |
| D11 | Validate the asset from **history**, not a glob: exact newly created path, created within the cell window, under the configured output root, then ffprobe for 832x480, expected frame count, fps, codec/container and nonzero duration. | The current glob can select a stale mp4. |
| D12 | Per-cell heartbeat leg log + configured queue URL so `scripts/otr_render_watchdog.ps1` can actually run; map its exit 2 to a failed cell and a selective reset. | The watchdog expects a heartbeat-bearing leg log and defaults to port 8000; this harness emits no compatible heartbeat today. |
| D13 | Campaign manifest (section 8) written before cell 1; any mismatch fails the cell. | Cross-arm comparison is void if the stack moved. |
| D14 | One base URL parsed and its port derived; one absolute output root; both propagated to launcher, reset, submit, watchdog and validation. | G4. |

Reused verbatim: clamp computation shape, boot through the launcher, submit/poll
via `tests/_run_baseline`, the sampler thread skeleton, incremental result
writing, `--dry-validate`, and the s/it parse.

**Not reused:** the harness renders at `CreateVideo fps 24.0` while production
encodes at 25. Recorded in the receipt, not corrected -- fps does not move the
memory profile and changing it would be a delta with no measurement purpose.

### Executable tests required before any GPU time

Node mappings and display names; `INPUT_TYPES`/`RETURN_TYPES` contracts; both
conditioning branches if O7 is taken; `ArmSpec` validation; exact graph mutation
including stale-input removal on class substitution; per-arm `/object_info`
validation and loader-enum resolution; per-arm step accounting; unique
`cell_id`/repeat paths; telemetry-failure injection proving no PASS is possible;
source-pin mismatch rejection; port and output-root propagation; stale-output
rejection; ffprobe validation; mandatory-cell denominator; repeat aggregation.
Then the focused suite, the full Windows suite, the Bug Bible, AST/JSON checks,
`OTR_WorkflowValidator`, the widget/link audit, and `HEAD == origin/v2.0-alpha`.

---

## 11. Run protocol

Per CLAUDE.md s4, reset before EVERY headless run; never assume a prior run
cleaned up -- the soak harness leaves a server RESIDENT holding ~60% VRAM.

1. **Reset.** Selective CIM kill by CommandLine (never a blanket
   `Stop-Process -Name python` -- that severs the MCP pythons). Kill the
   configured port's owner.
2. **Assert quiescence.** Port listener empty AND NVML settled to the desktop
   baseline. Record `desktop_baseline_nvml_mib`. **Fail the cell if it does not
   settle** -- do not boot onto residue.
3. **Boot** with the computed reserve through `_otr_soak_server_launch.cmd`
   passed as `-FilePath`, with `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` (a
   detached cmd inherits cp1252 and `prestartup_script.py` dies on the first
   emoji). Boot is ~20 s; if it "hangs", read the log -- it has already died.
   Record `pre_submit_nvml_mib` and the in-process clamp triple.
4. **Stage** the init image into the server's input directory.
5. **Preflight** (first cell of each arm): every required node class live in
   `/object_info`, and the model filename present in the loader dropdown. A
   missing enum is a hard campaign-incomplete condition, never a silent
   substitution and never a shrunk denominator.
6. **Submit and sample**, watchdog attached.
7. **Validate the asset from history**, then ffprobe, before recording `ok`. A
   finished render leaves the server resident at ~9-10 GB and 1% utilisation --
   that is NOT a crash, and the file check, not the VRAM reading, distinguishes
   them.
8. **Write the receipt atomically** (temp file plus replace, so an interrupted
   multi-hour run cannot leave a truncated result), then tear down and reset.

**Order:** A -> B-partial -> B, ascending rungs within each arm, then the winner
repeats. Assets land at their canonical path the first time, under
`otr/episodes/_bench_4arm/<arm>/`; nothing is staged in tmp to be moved later.

---

## 12. The greenlight bar

An arm passes only if ALL hold:

1. **`peak_delta_mib` <= 7168 MiB at EVERY required length** (17, 49, 81) at
   832x480.
2. **Every required cell has exactly one valid `ok` receipt.** Missing, skipped,
   telemetry-invalid, OOM, watchdog-failed or decode-invalid cells block PASS
   and never shrink the denominator.
3. **Telemetry valid:** NVML and psutil samples across the whole render window,
   zero sampler errors, no `-1` sentinels, inter-sample gap within tolerance.
4. **No OOM, no watchdog stall, no missing asset**, and no named spill signature
   in the server log. Avoiding CUDA OOM through uncontrolled system-memory thrash
   is a FAILURE: `sysram_delta_mib` and `pagefile_commit_mib` are gating against
   declared numeric thresholds.
5. **The asset validates**: history-owned newly created path, ffprobe-confirmed
   832x480, expected frame count, fps, codec and nonzero duration.
6. **A shippable licence** with a recorded receipt.

### Where 7168 comes from, and the assumption inside it

    peak_delta_mib + display_allowance_mib <= 8192,  display_allowance = 1024

A real 8 GB card has 8192 MiB TOTAL and a desktop/WDDM/compositor baseline
consumes part of it. This box idles at 1651 MiB with a full desktop; a leaner
8 GB machine is typically 400-900 MiB. **1024 MiB is a declared, conservative
assumption, not a measurement** -- operator-tunable (O5), stated so nobody
mistakes it for a fact. If it is wrong, every verdict shifts by the difference.

Note this is a different and complementary control from section 4's loader
budget: section 4 governs what ComfyUI will LOAD, this governs what the board
must HOLD.

### Verdict meanings

- **PASS at 17 only** -> records as **"floor-only"**, never a tier candidate.
- **A PASS** earns permission to proceed to qualification. Nothing else. It does
  not lower the guard, does not refit the estimator, does not qualify the tier.
- **A FAIL** eliminates an arm on evidence, and its measurements are retained --
  a failed arm's data is still valid evidence about staging behaviour.

---

## 13. What this bench must never be used to claim

- It does not qualify the 8 GB tier.
- It does not license lowering `FRAME_COST_MODEL`'s guard.
- It does not promote 14B.
- **It does not refit the production estimator** (9.1).
- It does not prove teardown, cleanup, or whole-pipeline phase behaviour.
- It does not validate either engine's admission channel, nor the four-way LTX
  ceiling disagreement.
- Results are labelled **"16 GB card, 8 GiB-reserve direct-node
  prequalification"** and never "8 GB qualified".

---

## 14. Operator decisions

Each carries a stated default, so a ruling is a yes/no.

| # | decision | default if not ruled |
|---|---|---|
| **O6** | **BUILD BLOCKER -- the canonical-workflow carve-out (G1).** Record a narrowly scoped exemption naming this runner, this graph, its outputs and its validation gates; or redesign the bench through `otr_canonical.json`. | Record the exemption. The existing `run_wan_ti2v_bakeoff.py` has operated this way since 2026-07-08; this writes down what is already practice, scoped to isolated bench harnesses only. |
| **O3** | **Arm B acquisition.** `wan2.2_ti2v_5B_fp16.safetensors` from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` -- the same repo our manifest already pins for the Wan VAE. | Approve. Arm B is the protected arm; without it the campaign answers only the encoder half. |
| **O7** | **The text-encode / image-encode stage split.** Your hard requirement was per-stage torch peaks between every stage. The order-safe sample+decode segment ships in the first build; the remaining two boundaries need forced ordering, which changes the measured schedule. | Ship the order-safe part only. Defer the full split to a separate, explicitly DIAGNOSTIC campaign whose data may never drive a fit verdict or an estimator refit. **This is the one place the panel's advice collides with a requirement you stated, so it is yours to rule, not mine.** |
| **O1** | **Arm D.** Blocked on a vendored, pinned, schema-validated API-format LTX 0.9.8 graph. When unblocked, run the shipped `ltx_8gb` (0.9.8-distilled), not the literal v0.9. | Keep BLOCKED for this campaign. Unblock as its own chunk. |
| **O2** | **Arm C.** CUT from the controlled campaign; licence passes but no ComfyUI base graph exists at any priority. | Keep CUT. Re-open as a separate research probe behind a conversion/loader receipt and a blind quality check. |
| **O4** | **LTX licence reading** -- the >= $10,000,000 clause admits two readings and I did not resolve it. | Gates arm D's SHIPPABILITY only, not its measurement. Settle before any LTX tier ships. |
| **O5** | **The 1024 MiB display allowance** in the 7168 MiB bar. A declared assumption. | Keep 1024 MiB; raise it if the target 8 GB machine runs a heavier desktop. |
| **O8** | **Winner ranking and repeat count** (7.4) -- lowest worst-case `peak_delta_mib` at 81 frames, ties by wall time, three cold measurements per required winner rung. | Adopt as stated. |

**O6 blocks all code. O3 blocks arm B's cells. Everything else can be ruled
after the A / B-partial cells run.**

---

## 15. VERIFY-AT-BUILD checklist

Empirical claims that require build/run confirmation. No r2/r3/r4 artifact marked
any claim UNVERIFIABLE.

- [ ] O6 authority path resolved and recorded **before touching code**.
- [ ] Helper vendored into the OTR repo, installed, server restarted, and every
      required class + loader enum confirmed through live `/object_info`.
- [ ] In-process `extra_reserved_vram_gib`, `minimum_inference_gib` and
      `maximum_vram_for_weights_gib` match the running ComfyUI revision under an
      actual `--reserve-vram 8` boot (expect ~8.80 / ~5.21 GiB on this board).
- [ ] Arm B provenance: immutable revision, SHA-256, size, licence, and live
      `UNETLoader` dropdown visibility.
- [ ] Every cell uses the pinned campaign manifest; any mismatch fails the cell.
- [ ] Inject NVML, psutil, probe and watchdog failures; prove none can produce
      PASS.
- [ ] Timestamps cover the entire render window on one clock, with the
      inter-sample gap within tolerance.
- [ ] Reset clears the configured port and returns NVML to desktop baseline
      before every boot.
- [ ] History returns a newly created exact asset; ffprobe confirms path,
      dimensions, fps, frame count, duration, codec and playability.
- [ ] Focused tests, full Windows suite, Bug Bible regression, AST/JSON checks,
      `OTR_WorkflowValidator`, widget/link audit, and `HEAD == origin/v2.0-alpha`
      before GPU execution.

---

## 16. Sources and provenance

**In-repo, read at `db0fa304` via Desktop Commander on the Windows tree**

- `docs/2026-07-31-wan-8gb-adversarial-review/report.md` -- judgment of record
- `CLAUDE.md` s0; `docs/PRODUCTION_SPRINT_LESSONS.md:117-121` (s7), s8
- `scripts/otr_wan_ti2v_bakeoff_gguf.json` -- arm A/B base graph
- `scripts/run_wan_ti2v_bakeoff.py` -- the harness being parameterized
- `scripts/_otr_soak_server_launch.cmd:60-79,113-138` -- clamp, port, output root
- `scripts/otr_render_watchdog.ps1:19-24,35-97` -- watchdog contract
- `nodes/_otr_video_engines/eng_wan_ti2v.py` -- recipe, contract, loader modes,
  `prepare()` hoist, `_build_graph`
- `nodes/_otr_video_engines/eng_ltx_8gb.py` -- arm D recipe, canvas, alias graph
- `nodes/_otr_video_engines/motion_common.py:190-363` -- probe and estimator
- `nodes/_otr_video_engines/wrapper_bridge.py:250-441` -- `free_after_use`
- `nodes/_vram_log.py:69-105`; `nodes/_otr_shared/gpu_residency.py:218-302`
- `workflows/variants/otr_8gb_wan.json` + `.env.json`, `otr_8gb_ltx.json` + `.env.json`
- `C:\ComfyUI-Models\otr_wan_ti2v_manifest.json`
- `C:\Users\jeffr\AppData\Roaming\ComfyUI\extra_models_config.yaml`
- `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\otr_bakeoff_helper\__init__.py`
  (NOT version-controlled -- see G3)
- `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_management.py:807-814,1051-1052`
- `comfyui-custom-node-survival-guide/BUG_BIBLE.yaml` -- BUG-05.05

**External, fetched 2026-07-31**

- ComfyUI Wan 2.2 guide -- https://docs.comfy.org/tutorials/video/wan/wan2_2
- Official 5B template (v0.4, frontend 1.27.10, cnr 0.3.45) --
  https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json
  (**pin by commit SHA before use as an ArmSpec base**)
- Wan 2.2 TI2V-5B upstream, apache-2.0 -- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- FastWan TI2V-5B, apache-2.0, Diffusers, 3-step DMD `1000,757,522` --
  https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
- Arm B acquisition source -- https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged
- LTX-Video model card, `license: other` -- https://huggingface.co/Lightricks/LTX-Video
- LTX open weights licence 0.X --
  https://huggingface.co/Lightricks/LTX-Video/raw/main/LTX-Video-Open-Weights-License-0.X.txt

**Hardware, measured 2026-07-31**

- `NVIDIA GeForce RTX 5080 Laptop GPU, 16303 MiB total, 1651 MiB used at idle`
