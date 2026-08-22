# Ghost Signal -- code-ready implementation plan

**Date:** 2026-08-22  
**Target lane:** `animatediff15_video`  
**Docs/tooltip label:** `AnimateDiff -- Ghost Signal`  
**Saved/profile value:** bare id `animatediff15_video`; the live director menu
derives `animatediff15_video (16:9)`  
**Source brief:** `docs/2026-08-22-GHOST-SIGNAL-CODEX-BUILD-BRIEF.md`  
**Status:** PLAN ONLY. No dependency install, source edit, workflow edit, render, test,
commit, or push is authorized by this document-writing task.

## 1. Authority and scope receipt

This plan applies the source brief and spec with the operator's later 2026-08-22
instructions taking precedence:

1. **Kibitz Round 1 is omitted.** This means the review campaign starts at R2.
   It does **not** repeal Ghost Signal spec ruling R1: the lane remains prompt-only
   and declares `accepts_still = False`.
2. **No measurement campaign.** Delete the brief/spec's VRAM bench, two-canvas
   bakeoff, SSIM/PSNR round-trip threshold, prompt-token survey, sampler A/B,
   motion-scale A/B, cadence eyeball, quality scoring, and timing campaign from
   the build scope.
3. **AnimateDiff gets its native contract.** Install the canonical Kosinkadink
   `ComfyUI-AnimateDiff-Evolved` repository at commit
   `9257651221002dcba0a12f9cff37e1944e58fb60`, then verify its live
   `/object_info` against the exact R3 node/socket/widget ledger in section 4.3.
   Do not import a fork's workflow, invent node names, or tune a replacement
   graph.
4. **One conservative recipe.** Use the repo-legal `512x288` canvas and enlarge
   it directly to `1920x1080` with clean Lanczos. There is no alternate canvas,
   diffusion refinement, super-resolution, or interpolation arm.
5. **One real publish smoke.** The only GPU acceptance run is a pass/fail
   canonical workflow smoke: it either finishes without OOM and publishes, or
   the lane remains draft/unpromoted. The smoke does not mint performance or
   quality claims.

Consequences of those overrides:

- “Under 4 GB” becomes an **unverified design target**, not a product claim or
  acceptance gate. User-facing wording is “very-low-VRAM-targeted” until a later,
  separately authorized campaign proves more.
- The old `384x216` default is retired without a bakeoff. Height 216 violates the
  live OTR `/32` canvas law in both `declared_render_canvas` and preflight G2;
  `512x288` is exact 16:9 and legal on both axes.
- No `low` or `high` token may be added to the public id. Those tokens carry
  measured cost semantics in this repo. The spec's neutral
  `animatediff15_video` id stands.
- Spec section 8 and all outcomes contingent on its measurements are superseded.
  Static dependency/schema discovery remains mandatory because it prevents
  guessed code; it is not a benchmark.

## 2. Frozen product contract

| Concern | Frozen decision |
|---|---|
| Engine id | `animatediff15_video` |
| Label | Docs/tooltip: `AnimateDiff -- Ghost Signal`; saved/profile id remains `animatediff15_video` |
| Dependency | Canonical `Kosinkadink/ComfyUI-AnimateDiff-Evolved` v1.6.0 at `9257651221002dcba0a12f9cff37e1944e58fb60`; its declared dependency list is empty and it requires ComfyUI >=0.3.68 |
| Model family | `v1-5-pruned-emaonly-fp16.safetensors` plus the real `mm-p_0.5.pth` v2 motion module; the checkpoint's slot-2 embedded VAE is used |
| Internal graph | Eight exact nodes: five Comfy core nodes plus `ADE_StandardStaticContextOptions` and `ADE_AnimateDiffLoaderGen1`; two `CLIPTextEncode` instances make eight total |
| Sampler recipe | One `KSampler`: 20 steps, CFG 8.0, `euler`, `normal`, denoise 1.0; ADE beta schedule `autoselect` |
| Motion influence | Leave `scale_multival` and `effect_multival` unconnected so the pinned ADE implementation owns its normal full-strength behavior; no tuning arm |
| License truth | `commercial_clean = False`: the ADE code is Apache-2.0 and the SD1.5 checkpoint is CreativeML Open RAIL-M, but the selected `mm-p_0.5.pth` host provides no model license grant |
| Family / input | `text_to_video`; `text_prompt` required; no `init_image` |
| Roles | `announcer_visual`, `music_visual`, `character_video` |
| Still ownership | None: explicit `accepts_still = False`, `still_plan = ()`, `subject_ownership = "prompt"` |
| Native canvas | Fixed `512x288`, full-frame 16:9 |
| Delivery | Exact `1920x1080`, 25 fps, clean Lanczos, no crop/pad/bars/unsharp/model upscaler |
| Cadence | Exact 12.5 source fps into 25 delivered fps: generate `U = ceil(T / 2)` fresh source positions for the authoritative delivered target `T`; hold each source frame twice except the once-shown odd tail; never relabel timestamps |
| Coverage | One newly seeded AnimateDiff timeline per OTR beat spanning that beat's complete audio-derived frame budget; non-looped sliding context inside that render; no source batch/path reuse across beats and no OTR clip chaining |
| Context | `ADE_StandardStaticContextOptions`: non-circular, length 16, overlap 4, `pyramid`, `use_on_equal_length=False`, start 0.0, guarantee 1 |
| Continuity | Explicit `CONTINUITY_NONE` |
| Frame contract | Delivered-frame units: `min_frames=1`, `max_frames=0`, `quantum=1`, `native_fps=25`, `allow_tail_trim=True`, explicit continuity |
| Style | Existing visual-style pack is the only medium/material/palette authority; join mode `compose` |
| Story motion | Existing ledger `motion_clause`, read-only during render |
| Prompt budget | Ghost-owned 320-character hard ceiling; normal target 260-280 to reserve one-prop banana-route growth; phrase-aware deterministic trimming |
| Negative | Lane lettering/artifact hygiene plus the selected pack's `effective_negative`; 320-character phrase-aware ceiling; bind to real negative CLIP conditioning |
| References | No IPAdapter, ControlNet, masks, source stills, or reference folders |
| Excluded | Looped context, ping-pong, mirror, loop-fill, RIFE, second KSampler, hires fix, ESRGAN/SeedVR2, per-pack reskinning |

The 16-frame number is a **context window**, not an OTR clip duration. A beat is
one timeline even when it spans several internal context windows.

“Fresh” and “unique” are provenance laws: every source index is generated for
that beat and is never borrowed from another beat. They do not claim that a
diffusion model can be forced to produce pixelwise-different pictures.

## 3. End-to-end ownership

```text
ledger + selected visual-style pack
        |
        +--> ShotLock stamps one durable subject sigil per character/style
        |
        +--> Ghost prompt composer
              pack cue -> subject -> framing -> action/motion
              -> emotion -> one story accent -> shot law
        |
        +--> VideoRequest(text_prompt, negative_prompt, seed, T, 512x288)
        |
        +--> AnimateDiff-Evolved graph from pinned /object_info
              checkpoint -> CLIP conditioning -> motion module/context
              -> sample U fresh per-beat source positions -> decode
        |
        +--> deterministic hold-2 selector -> exact T frames @ 25 fps
        |
        +--> CanonicalClip + cadence/prompt/delivery receipts
        |
        +--> clip manifest -> timeline segment
        |
        +--> clean Lanczos 512x288 -> 1920x1080, no other image processing
        |
        +--> audio mux -> obs_publish -> otr/obs
```

Each arrow has one named owner. Engine-id string tests must not substitute for a
declared capability at any downstream boundary.

## 4. Phase 0 -- dependency and schema lock, no benchmarks

No adapter code starts until this phase is complete.

### 4.1 Install and pin

1. Enter the serialized implementation window only after any operator render is
   finished. Reset exactly as `CLAUDE.md` sections 4-5 require: selectively stop
   only the ComfyUI server/harness processes and verify port 8000 is free.
2. With the server stopped, install the canonical upstream repository at
   `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-AnimateDiff-Evolved`.
   Check out exactly `9257651221002dcba0a12f9cff37e1944e58fb60` from
   `https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git`. Do not
   install a tutorial fork or advance the commit during the build. Its pinned
   `pyproject.toml` declares package version 1.6.0, no additional Python
   dependencies, and ComfyUI >=0.3.68. The installed ComfyUI 0.33.3 satisfies
   that source requirement; do not add speculative packages.
3. Install exactly these two artifacts; there is no fallback list:
   - checkpoint `v1-5-pruned-emaonly-fp16.safetensors`, 2,132,696,762 bytes,
     SHA-256 `e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916`,
     from the [pinned Comfy-Org SD1.5 archive](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/4b97aa05c15972d2628d6f6db8fb2b0a68d9241e/v1-5-pruned-emaonly-fp16.safetensors),
     in the registered `checkpoints` folder;
   - motion module `mm-p_0.5.pth`, 1,817,894,327 bytes, SHA-256
     `d779ab78b3f18e2dc2342a9beecf4af3365fbee63fd83a12c8c94c50dd87b4ed`,
     from [`manshoety/beta_testing_models` commit
     `816cfda7c7193a5705b39f94fc317a4465442654`](https://huggingface.co/manshoety/beta_testing_models/blob/816cfda7c7193a5705b39f94fc317a4465442654/mm-p_0.5.pth), in the registered
     `animatediff_models` folder.
   Hash both local files before boot. `CheckpointLoaderSimple` supplies the
   checkpoint's embedded VAE at output slot 2; do not add or download a
   separate VAE.
4. Boot the now-installed pack once with UTF-8 through
   `scripts/_otr_soak_server_launch.cmd`; only this post-install process may
   supply the Phase-0 `/object_info` capture. A server started before installation
   cannot register ADE's extension classes and is never schema evidence. The
   target ComfyUI source commit is the currently installed
   `4da9e2dbead52fc1e68beae33fe3d7ad63b63241`; record and compare it rather than
   assuming a version label is enough.
5. Do not use an AnimateLCM checkpoint merely to reduce steps: the Ghost
   lettering defense requires live negative conditioning. If the selected
   canonical recipe has inert negative conditioning, stop rather than silently
   move the negative into an unreviewed positive-prompt workaround.

Create `docs/2026-08-22-ghost-signal-dependency-lock.json` containing only:

- upstream URL and exact ADE/ComfyUI commits above;
- exact custom-node directory;
- the two artifact URLs, filenames, byte counts, and SHA-256 values above, plus
  `vae_source="checkpoint_output_2"`;
- registered `folder_paths` categories;
- ADE Apache-2.0 and checkpoint CreativeML Open RAIL-M sources, the absent
  `mm-p_0.5.pth` model license, and resulting `commercial_clean=false`.

Artifact byte sizes may be copied into this identity record if emitted by the
installer, but they are informational and may not drive a VRAM claim.

### 4.2 Verify the locked live node surface

The node choices are no longer Phase-0 placeholders. Current upstream source at
the pinned commit fixes the two custom ids as
`ADE_StandardStaticContextOptions` and `ADE_AnimateDiffLoaderGen1`. A read-only
query of the installed pre-ADE ComfyUI server fixes the five core ids as
`CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, and
`VAEDecode`. Two instances of `CLIPTextEncode` produce the positive and negative
conditionings.

After the post-install clean boot, write the relevant `/object_info` subset to
`docs/2026-08-22-ghost-signal-object-info.json`. This is an equality check
against section 4.3, not a new choice point. Retain each used class's:

- exact class name;
- required and optional input names, types, defaults, minima, and maxima;
- output names/types and output positions;
- combo values for checkpoint, motion module, context schedule, sampler, and
  scheduler inputs;
- the optional `scale_multival` and `effect_multival` sockets and proof that
  omitting them preserves ADE's normal unmodified influence;
- node-pack version/commit association.

The exact semantic map is:

| Semantic stage | Exact live class | Exact inputs consumed | Exact output consumed |
|---|---|---|---|
| SD1.5 checkpoint load | `CheckpointLoaderSimple` | `ckpt_name` | `MODEL[0]`, `CLIP[1]`, `VAE[2]` |
| positive/negative encode | `CLIPTextEncode` x2 | `text`, `clip` | `CONDITIONING[0]` |
| non-looped context | `ADE_StandardStaticContextOptions` | exact six literals in 4.3 | `CONTEXT_OPTS[0]` |
| motion load/apply | `ADE_AnimateDiffLoaderGen1` | base `model`, `model_name`, `beta_schedule`, `context_options` | patched `MODEL[0]` |
| latent creation | `EmptyLatentImage` | `width`, `height`, `batch_size` | `LATENT[0]` |
| sampling | `KSampler` | patched model, both conditionings, latent, exact recipe | `LATENT[0]` |
| decode | `VAEDecode` | sampled `samples`, checkpoint `vae` | `IMAGE[0]` |

The implementation may use only names and inputs present in that committed
capture. `_node_candidates()` contains one name per alias, never alternative
spellings. Any mismatch stops the build and updates this plan; it does not
activate runtime probing or a fallback node.

The pinned ADE classes use ComfyUI's V3 `io.ComfyNode` surface. The installed
ComfyUI base exposes that surface through `FUNCTION=EXECUTE_NORMALIZED`, and the
repo's `wrapper_bridge.normalize_node_output()` already unwraps `NodeOutput`.
Verify those facts statically after install; do not add an ADE-specific executor
or compatibility shim.

Pinned source anchors for the implementer:

- [ADE Gen1 loader schema](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/9257651221002dcba0a12f9cff37e1944e58fb60/animatediff/nodes_gen1.py)
- [ADE context schemas](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/9257651221002dcba0a12f9cff37e1944e58fb60/animatediff/nodes_context.py)
- [ADE extension registration](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/9257651221002dcba0a12f9cff37e1944e58fb60/animatediff/nodes.py)
- [official model placement and folder ids](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/9257651221002dcba0a12f9cff37e1944e58fb60/README.md)

### 4.3 R3 exact node, widget, output, and wire contract

This is the complete internal rendering graph. It has **eight node instances
and ten typed links**. It is constructed as declarative in-process graphs for
`wrapper_bridge.run_graph`; it is not pasted into `otr_canonical.json`.

#### Node ledger

The “GUI widget vector” is the equivalent positional `widgets_values` contract
for audit and fixture construction. The runtime graph passes the same values by
the named API inputs in the previous column. `fixed` is KSampler's GUI-only
control-after-generate value; it is not a FUNCTION argument.

| Id | Exact class / origin | Named inputs and frozen values | Optional sockets | Outputs by slot | GUI widget vector |
|---|---|---|---|---|---|
| `ckpt` | `CheckpointLoaderSimple` / Comfy core | `ckpt_name="v1-5-pruned-emaonly-fp16.safetensors"` | none | `MODEL[0]`, `CLIP[1]`, `VAE[2]` | `["v1-5-pruned-emaonly-fp16.safetensors"]` |
| `positive` | `CLIPTextEncode` / Comfy core | `text=request.text_prompt`; `clip` linked | none | `CONDITIONING[0]` | `[request.text_prompt]` |
| `negative` | `CLIPTextEncode` / Comfy core | `text=request.negative_prompt`; `clip` linked | none | `CONDITIONING[0]` | `[request.negative_prompt]` |
| `context` | `ADE_StandardStaticContextOptions` / ADE custom | `context_length=16`; `context_overlap=4`; `fuse_method="pyramid"`; `use_on_equal_length=False`; `start_percent=0.0`; `guarantee_steps=1` | `prev_context` and `view_opts` deliberately unconnected | `CONTEXT_OPTS[0]` | `[16, 4, "pyramid", false, 0.0, 1]` |
| `ade` | `ADE_AnimateDiffLoaderGen1` / ADE custom | `model` linked; `model_name="mm-p_0.5.pth"`; `beta_schedule="autoselect"`; `context_options` linked | `motion_lora`, `ad_settings`, `ad_keyframes`, `sample_settings`, `scale_multival`, `effect_multival`, and `per_block` deliberately unconnected | patched `MODEL[0]` | `["mm-p_0.5.pth", "autoselect"]` |
| `latent` | `EmptyLatentImage` / Comfy core | `width=512`; `height=288`; `batch_size=source_request` | none | `LATENT[0]` | `[512, 288, source_request]` |
| `sampler` | `KSampler` / Comfy core | `model` linked; `seed=request.seed`; `steps=20`; `cfg=8.0`; `sampler_name="euler"`; `scheduler="normal"`; `positive` linked; `negative` linked; `latent_image` linked; `denoise=1.0` | none | sampled `LATENT[0]` | `[request.seed, "fixed", 20, 8.0, "euler", "normal", 1.0]` |
| `decode` | `VAEDecode` / Comfy core | `samples` linked; `vae` linked | none | `IMAGE[0]` | `[]` |

#### Semantic typed-link ledger

This table names the original producer slots. It is the model-level graph
contract, not the tuple shape passed between the four executor calls.

| # | Source output | Destination input | Type |
|---:|---|---|---|
| 1 | `ckpt.MODEL[0]` | `ade.model` | `MODEL` |
| 2 | `ckpt.CLIP[1]` | `positive.clip` | `CLIP` |
| 3 | `ckpt.CLIP[1]` | `negative.clip` | `CLIP` |
| 4 | `context.CONTEXT_OPTS[0]` | `ade.context_options` | `CONTEXT_OPTIONS` |
| 5 | `ade.MODEL[0]` | `sampler.model` | `MODEL` |
| 6 | `positive.CONDITIONING[0]` | `sampler.positive` | `CONDITIONING` |
| 7 | `negative.CONDITIONING[0]` | `sampler.negative` | `CONDITIONING` |
| 8 | `latent.LATENT[0]` | `sampler.latent_image` | `LATENT` |
| 9 | `sampler.LATENT[0]` | `decode.samples` | `LATENT` |
| 10 | `ckpt.VAE[2]` | `decode.vae` | `VAE` |

#### Executable per-stage alias and `Wire` ledger

`run_graph.external_results` normalizes one value to a one-slot tuple, and
`Wire(src, slot)` indexes that tuple. Therefore every cross-stage alias below
uses slot `0`; no executable graph may address `ckpt[1]` or `ckpt[2]` after the
checkpoint tuple has been split.

| Executor call | Exact external-results aliases | Exact executable links |
|---|---|---|
| Checkpoint | none | no links; execute `ckpt` and receive the sole three-slot tuple `(MODEL, CLIP, VAE)` |
| Encode | `clip=(ckpt_out[1],)` | `Wire("clip", 0) -> positive.clip`; `Wire("clip", 0) -> negative.clip` |
| Sample | `base_model=(ckpt_out[0],)`; `positive_cond=(positive_out[0],)`; `negative_cond=(negative_out[0],)` | `Wire("base_model", 0) -> ade.model`; `Wire("context", 0) -> ade.context_options`; `Wire("ade", 0) -> sampler.model`; `Wire("positive_cond", 0) -> sampler.positive`; `Wire("negative_cond", 0) -> sampler.negative`; `Wire("latent", 0) -> sampler.latent_image` |
| Decode | `sampled_latent=(sampler_out[0],)`; `vae=(ckpt_out[2],)` | `Wire("sampled_latent", 0) -> decode.samples`; `Wire("vae", 0) -> decode.vae` |

That is exactly two encode links, six sample links, and two decode links: the
same ten semantic links above, with no hidden or implicit edge.

#### Execution staging and ownership

Implement the same link ledger in four bounded executor calls so the large
families do not stay artificially pinned by one result dictionary:

1. `load()` resolves exactly the seven class ids above (the repeated text class
   is resolved once) and the two artifact paths; it performs no tensor forward.
2. `prepare()` executes only `ckpt`, requires slot 0 to expose callable
   `detach(unpatch_all=True)`, immediately registers that exact object through
   `on_result`, splits the three outputs into the one-slot owned handles
   `prepared["base_model"]`, `prepared["clip"]`, and `prepared["vae"]`, and
   discards the original three-slot result tuple.
3. Encode stage executes `positive` and `negative` with the exact `clip` alias
   above, extracts one-slot `positive_cond` and `negative_cond` handles, then
   clears the encode result/external dictionaries and removes
   `prepared["clip"]`. Call the existing public
   `wrapper_bridge.reclaim_idle_models("ghost-signal post-encode")` only after
   those strong references are gone and before constructing the sample graph.
   `free_after_use` alone cannot release an external because `run_graph` keeps
   all external ids for the duration of that call.
4. Sample stage executes `context -> ade` and `latent -> sampler`, with the base
   model and two conditionings as external results and `terminal="sampler"`.
   Its `on_result` callback requires and registers the patched ADE `MODEL[0]`
   immediately. After the sampled latent returns, call the adapter helper
   `_release_sampling_patchers_before_decode(prepared, owners)` with ordered,
   identity-deduplicated candidates `ade_model` then `base_model`. For each
   candidate, call `detach(unpatch_all=True)`; only after success remove that
   exact identity from every distinct `prepared["patchers"]`/`self._patchers`
   list in place and clear its Ghost-owned reference. If detach raises or is not
   callable, retain that candidate in the patcher list, raise a named
   `GraphExecutionError`, do not execute `decode`, and let outer `finally`
   teardown retry it and release the lease. After all candidates succeed, clear
   the sample result/external dictionaries, both conditioning handles, context,
   and empty-latent references, then call
   `wrapper_bridge.reclaim_idle_models("ghost-signal post-sample")`.
5. Decode stage executes only `decode`, using sampled latent and VAE as external
   results and `terminal="decode"`; convert the returned IMAGE batch with the
   existing OTR bridge and encode the silent native MP4 with the existing OTR
   ffmpeg helper.

Every stage uses `free_after_use=True` where its terminal shape permits it, but
the explicit owner clearing and reclaim calls above are the cross-stage release
law. Successful sampling leaves neither sampling patcher in the teardown
bucket; `MotionEngineBase.teardown` remains the failure-path backstop and always
releases the lease in `finally`. No stage calls `unload_all_models`, and no
measurement controls this deterministic sequence.

There is deliberately no `ADE_AnimateDiffSamplingSettings`, multival source,
Motion LoRA, keyframe, per-block, view-options, ControlNet, IPAdapter,
`VHS_VideoCombine`, RIFE, second sampler, or VAE-loader node. ADE optional
sockets are omitted from the runtime input dictionaries, not passed invented
`None` widgets. OTR—not a Comfy video-combine node—owns cadence conversion,
silent MP4 encoding, canonicalization, and delivery enlargement.

### 4.4 Phase-0 static stop conditions

Stop before code if the post-install capture differs from any of these locked
structural contracts:

- SD1.5 checkpoint and `mm-p_0.5.pth` can be resolved by `folder_paths` or one
  documented launch-time pin;
- `ADE_StandardStaticContextOptions` exists with the exact section-4.3 surface
  and constructs `STATIC_STANDARD` with no closed-loop input;
- positive and negative conditioning both reach the chosen sampler;
- pinned-source inspection of that exact sampler/CFG path proves unconditional
  conditioning is not skipped at the frozen CFG value (including ComfyUI's
  CFG=1 optimization and any sampler-specific override). If it is inert, stop;
  this is the existing G3.5 semantic gate, not a render comparison;
- `EmptyLatentImage.batch_size` accepts `source_request`, and pinned ADE source
  confirms Standard Static slides length-16 windows with overlap 4 across a
  longer timeline without loop-fill;
- decoded images can be counted before encoding;
- model/patcher ownership can be observed and released without
  `unload_all_models`;
- the dependency/license state can be represented truthfully in the registry.

The committed source already answers the intended chunking law; Phase 0 verifies
that the installed checkout is the same source. Any mismatch stops rather than
becoming a render experiment. Do **not** run VRAM probes, time trials, canvas
comparisons, prompt tokenization, image similarity, motion scoring, or visual
bakeoffs in Phase 0.

## 5. Implementation slice A -- adapter and cadence

Create `nodes/_otr_video_engines/eng_ghost_signal.py` as a cold-import-clean
`MotionEngineBase` subclass. Heavy imports and ComfyUI class resolution stay
inside lifecycle methods.

### 5.1 Declarations

The class declares, and tests pin:

- `name = "animatediff15_video"`
- `family = "text_to_video"`
- `required_inputs = ("text_prompt",)`
- `roles = ("announcer_visual", "music_visual", "character_video")`
- `default_roles = ()`
- `requires_flag = None`
- `render_aspect = "wide"`
- `render_canvas = (512, 288)`
- `target_fps = 25`
- `accepts_still = False`
- `still_plan = ()`
- `subject_ownership = "prompt"`
- `prompt_profile = "ghost_signal_v1"`
- `prompt_budget_chars = 320`
- `style_join = "compose"`
- `delivery_scale_mode = "lanczos_clean_full_frame"`
- the frozen `FrameContract` from section 2.

The adapter declares `commercial_clean = False`, matching the dependency lock.
`CAPABILITIES` has no such key, so do not invent one there. A future licensed
replacement motion module may change that declaration only in a separately
reviewed artifact-lock change.

### 5.2 Lifecycle and failure law

Implement the standard lifecycle with these boundaries:

1. `assert_usable` resolves the two pinned source commits, all seven unique class
   ids in section 4.3, the checkpoint, and `mm-p_0.5.pth` before forward work.
   The VAE requirement is checkpoint output slot 2, not another file. Use established
   resolver names (`_installed`, `_ckpt_path`, and one motion-model resolver) and
   route checkpoint/motion paths through the live `folder_paths` categories so
   `extra_model_paths.yaml` works. Each missing item raises a named
   `EngineUnusable`; there is no download or alternate asset.
2. Override `load()`. `MotionEngineBase.prepare` acquires the AS-3 lease and then
   unconditionally calls `self.load()`; inheriting the base implementation would
   raise `NotImplementedError` on every beat. Ghost's override lazily resolves
   the exact single-name class map and artifact tokens, then marks `_loaded`; it
   does not run a node. Override `unload()` only to clear Ghost-owned class,
   artifact, base-model, VAE, conditioning, latent, and ADE-patcher references
   before restoring base flags. The staged public reclaim seam may detach loaded
   patchers, but neither it nor this adapter clears a global model registry or
   calls `unload_all_models`.
3. `prepare` delegates to `MotionEngineBase.prepare` so lease and load failure
   cleanup remain centralized, then executes only the section-4.3 `ckpt` node.
   Its `on_result` requires and registers callable-detach base `MODEL[0]`, it
   verifies exactly three outputs, and it stores separate one-slot
   base-model/CLIP/VAE handles so the combined result tuple cannot retain a
   released family.
4. `render_clip` validates the request and performs the exact encode, sample,
   and decode executor stages in section 4.3. Every graph uses only the committed
   class/input map. The ADE-stage `on_result` callback registers the patched
   model immediately, before the executor can evict it. A post-`run_graph` scan
   is too late. Use `keep=` only for a handle consumed later in the same stage;
   cross-stage ownership travels through the named one-slot external results.
   The adapter-owned `_release_sampling_patchers_before_decode` method performs
   the exact ordered identity-safe detach/remove law from section 4.3; this is a
   generic Comfy `ModelPatcher` contract, not copied LTX/Wan loader behavior.
5. Any graph/output-count mismatch raises immediately. No padding, alternate
   graph, alternate node name, or still fallback is allowed.
6. `canonicalize` calls `ffprobe_clip_fields` once, feeds those same fields to
   `validate_silent_clip_contract`, and additionally refuses unless
   `width == 512` and `height == 288`. It preserves all Ghost receipts and
   declares no audio only after both checks succeed; no second probe is added.
7. On the success path, both sampling patchers have already been detached and
   identity-removed before decode, so `teardown` sees no sampling owner. On any
   earlier failure—or a pre-decode detach failure—the still-tracked candidates
   remain available to base `teardown`, which retries best-effort and releases
   the lease in `finally`. Never call `unload_all_models`.

The inherited `compute_real_frame_budget` path is unrelated to this lane's
quantum-1 cadence. Ghost never calls it. Add a detonation test that replaces it
with a raising sentinel and proves a complete mocked Ghost render does not touch
it. Do not describe it as a ping-pong extender; the current function only snaps
and conditionally refuses.

### 5.3 Frame and cadence arithmetic

The audio-derived delivered target count `T` is the only duration authority.
Do not recompute from floating-point duration; that disagrees at half-second
banker's-rounding boundaries.

Pure cadence functions implement:

1. `U = (T + 1) // 2`.
2. Request `source_request = max(U, 16)` and discard only the unused source tail
   before cadence conversion. Sixteen is the pinned upstream SD1.5 AnimateDiff
   context/sweet-spot floor used by this fixed recipe, not a measured Ghost
   quality claim. With `use_on_equal_length=False`, an exact 16-frame request
   uses the motion model directly; longer requests activate the static sliding
   context.
3. Require the decoded count to equal `source_request` exactly.
4. Retain the first `U` source frames in order.
5. Build the selector `[0, 0, 1, 1, ...]` and take its first `T` entries.
6. Encode exactly `T` frames at 25 fps. Never relabel a 12.5 fps file as 25 fps.

Each beat keeps the existing `_seed_from_hash(request_hash, shot_id)` authority,
so even two beats with identical composed text receive different deterministic
seeds. The adapter creates a new source batch and output path for every shot;
its cache/session identity includes the shot id, resolved seed, `U`, canvas,
model lock, and prompt hashes. A hit may replay only that same shot's deterministic
artifact. It may never return another beat's batch/path, and no prompt-only cache
key may collapse two beats together.

Follow the existing MiniMax H3 rate-conversion accounting precedent:

- `extension_mode = "none"`
- emitted-scope `native_frame_count = T`
- `model_frame_count = source_request` (every frame actually requested from and
  decoded by the model)
- `cadence_mode = "hold_2"`
- `cadence_source_frame_count = U` (the unique source prefix retained by the
  cadence selector)
- `cadence_delivered_frame_count = T`
- `cadence_tail_trim = 2 * U - T` (only 0 or 1)

This is duration-preserving cadence conversion, not tail extension. Source
frames `0..U-2` appear exactly twice; source frame `U-1` appears twice when `T`
is even and once when `T` is odd. Every output index selects a real decoded
source frame from the same beat. `model_frame_count -
cadence_source_frame_count` truthfully exposes any source frames generated only
to satisfy the live node's structural minimum; `cadence_tail_trim` covers only
the final hold-2 surplus. The separate scopes prevent `native_frame_count` from
being misread as “T distinct diffusion samples.”

## 6. Implementation slice B -- prompt, ledger, and style

Create the pure module
`nodes/_otr_video_engines/ghost_signal_prompt.py`. It performs no I/O, model
load, LLM call, or ledger mutation during rendering.

### 6.1 Durable character sigil

Add a pure `distill_subject_sigil(cast_row, *, episode_seed, char_id, style_id)
-> str` to `ghost_signal_prompt.py`, and add
`subject_sigil: Optional[str] = None` to `ShotRow`. In
`otr_shot_lock.build_execution_plan`, place the sigil-map construction
**after** the nested `engine_for(role)` resolver exists and **before** the
cast-time preflight loop. Filter to beats whose resolved engine is exactly
`animatediff15_video` and whose role is `character_video`; non-Ghost episodes
must not acquire a new seed/style requirement.

For each distinct nonblank `char_id` in that filtered set, read the raw cast row
with `_otr_ledger_consumers.cast_lookup(ledger, char_id)`. Do not call
`_appearance_for_char`: that helper may invoke the optional wardrobe writer and
would turn a deterministic Ghost identity read into a hidden mutation/credit
path. Obtain the episode seed from the existing authority
`ledger["meta"]["episode_seed"]` and the style once from
`get_visual_style(ledger["meta"])`, then compute one map keyed by
`(episode_seed, char_id, style.style_id)`. A non-`None` production ledger
containing a Ghost character beat but no `meta.episode_seed` fails loud by name;
it must never silently collapse to seed 0 or `""`. When `ledger is None`, keep
the map empty; that fixture path already skips cast-time preflight and must
remain valid. A missing cast row becomes `{}` and uses the checked-in neutral
sigil pools; it never triggers wardrobe or another author.

Source priority is the existing cast-row authority:

1. `portrait_prompt`
2. `appearance`
3. `description`
4. `character_description`
5. `name` only as a signal that the cast row is sparse; the name itself is never
   emitted

The distiller is deterministic and credit-free:

1. normalize whitespace and split the selected source into complete
   comma/semicolon/sentence phrases;
2. remove the cast name and discard camera, medium, background, and second-person
   phrases;
3. choose the first phrase matching each checked-in vocabulary bucket:
   silhouette/body shape, asymmetrical landmark, costume color/item, and handheld
   prop;
4. fill any missing bucket from a small checked-in neutral cue pool selected by
   `sha256(episode_seed | char_id | style_id | source_text)`; these are deliberate
   heraldic cues, not a hidden model/LLM fallback;
5. compose the four buckets in fixed order and phrase-trim to the sigil's own
   checked-in ceiling.

Tests pin the vocabularies, pools, hash domain, phrase order, and output ceiling.
Include gender only after checking that an explicit cast-row gender field exists
and is nonblank, then normalize it with the existing roster helper; never call
`normalize_gender` on absence, because absence becomes `"other"`. The output must
not preserve the cast name itself.

Pass the precomputed map into
`_assert_family_inputs_satisfiable_cast_time(engine_id, beat, ledger, policy,
subject_sigils)` and set `shot["subject_sigil"]` on its temporary shot before
`build_request_from_shot`. Stamp the identical value on the later durable shot
dictionary before `_stamp_frame_bounded` and `_stamp_coverage_plan`. This is
required because cast-time preflight runs before those rows are minted; deriving
only inside the durable-row loop would leave the temporary Ghost identity absent
and force the named composer refusal at preflight (the generic request seed is
not acceptable evidence).

### 6.2 Composer contract

Compose in this immutable order, omitting an empty slot without reordering the
others:

1. pack cue;
2. subject identity;
3. framing lock;
4. beat action/motion;
5. emotion;
6. one story accent;
7. affirmative shot law.

Use the existing authorities directly:

- `prefix_style_cue` for the pack-owned positive look;
- `effective_negative` for the pack-owned negative tail;
- `resolve_motion_clause_text` for a valid already-authored beat motion;
- `_ltx_motion_role_key`'s existing role/beat mapping for the relevant pack
  `motion_registers` value;
- `get_open_subject` for the music console/signal subject.

Do not import `render_driver` from the pure prompt module and do not create a
second role-to-register table. The already-owning driver calls
`_ltx_motion_role_key`, performs exact-key lookup on `_vstyle.motion_registers`,
and passes the resolved string into the pure composer as
`pack_motion_fallback`. Thus existing opening-role/env semantics stay in one
place while the composer remains cold and cycle-free.

`resolve_motion_clause_text` legitimately returns `None` when the optional motion
pass is disabled, its stored object is a fallback, or cast-time preflight has only
the temporary shot. That is not permission to emit an empty action slot. The pure
composer therefore resolves motion in this order:

1. use the whole validated motion clause when present;
2. otherwise, for announcer/music, use the selected pack's role register;
3. otherwise, for a character, map the line's existing `beat_intent` through a
   small checked-in action table; an unmapped non-empty intent becomes a bounded
   `moves with <first six normalized words>` phrase;
4. if even `beat_intent` is absent at cast-time preflight, choose one neutral
   kinetic action from a checked-in pool by the same subject-sigil hash domain.

This fallback is deterministic, pack/ledger-derived, and credit-free. It does not
enable the optional motion-clause writer and never consumes dialogue.

Role content is fixed:

| Role | Subject/framing/action law |
|---|---|
| Character | One heraldic actor from the durable sigil; mid-shot or wider; one prop and one action; the whole validated motion clause when present, otherwise one ledger-intent/hash-selected kinetic action; facial stability optional |
| Announcer | One anthropomorphic radio/console host using `announcer_subject_face`; distilled announcer motion register; no lip-sync promise |
| Music | No human; one console/emblem/signal sculpture from `announcer_subject_object` or `get_open_subject`; widest abstract motion; one mood/tempo/palette cue |

Never consume raw dialogue, `episode_title`, the M4 scene wall, a second person,
or proper-noun metadata. Positive wording stays affirmative.

### 6.3 Budget and negative

- Positive hard cap: 320 characters; normal target: 260-280.
- Trim whole phrases in this order: story accent, emotion, dispensable framing
  adjectives, then distilled register phrases.
- Never trim the subject sigil, authored motion clause, mid-shot floor, or shot
  law. If those protected fields alone exceed 320, fail the sigil distiller test;
  do not truncate the law at render time.
- Negative hard cap: 320 characters. Start with the de-duplicated lane hygiene
  head
  (`text, watermark, caption, lettering, subtitles`) followed by
  `effective_negative(selected_pack)`. Retain the complete hygiene head, then add
  de-duplicated pack-negative phrases in authored order while the next whole
  phrase fits; never cut a phrase or claim a tokenizer limit from this character
  budget.
- Bind `VideoRequest.negative_prompt` to the captured negative CLIP encode input.
  The adapter reads that exact field and refuses missing/blank input by name;
  `request.get("negative_prompt") or <engine constant>` is forbidden. No
  engine-side negative constant may replace it.

### 6.4 Render-driver seam and observability

The production policy reaches `derive_creative_directives` first. Its existing
`still_consumer_capabilities(video_policy)` filter sees Ghost's
`accepts_still=False`, removes Ghost character beats from `char_beats`, and
returns before `_resolve_writer_llm`. Preserve that path: a focused test replaces
the writer resolver with a raising sentinel and proves an all-Ghost policy spends
no M4 call while a legacy still-consuming policy remains unchanged.

In `render_driver.build_request_from_shot`, after style/line/role/canvas
resolution, resolve the capability without assuming every id is registered:

```text
prompt_profile = (
    getattr(_vreg.get_engine(_eng_id), "prompt_profile", None)
    if _vreg.is_registered(_eng_id) else None
)
```

Before the existing `if text_prompt:` M4 branch, when the profile equals
`ghost_signal_v1`, resolve the existing motion clause and pack register in the
driver, then call the pure composer with the resolved style, line, role, shot,
`subject_sigil`, `motion_clause`, and `pack_motion_fallback`; assign both
`req["text_prompt"]` and
`req["negative_prompt"]`; publish `_prompt_char_budget = 320`; set
`_prompt_protected_clause` to the **subject identity phrase**, not the trailing
shot law; stamp the prompt metadata; and set one local `_ghost_composed` boolean.
Guard both the existing M4 branch and the later LTX scene branch with
`not _ghost_composed`. This order deliberately outranks any M4 creative wall on
the same shot and also works for the cast-time temporary shot.

`build_request` seeds every request with a generic 1940s-studio prompt, so
required-input presence alone cannot prove this branch ran. For Ghost, ignore
that seed: `_ghost_composed` becomes true only after a non-empty composed
positive, non-empty composed negative, and
`observability.prompt_source="ghost_signal"` are installed. A missing character
sigil, empty composition, or branch miss raises a named `FamilyInputGap` from
`build_request_from_shot` during both cast-time and render-time request building;
the generic seed is never accepted as Ghost input.

Do not:

- add Ghost to the LTX tuple;
- change `_LTX_MOTION_PROMPT_MAX`;
- widen `engine_id.startswith("ltx")` and its face-framing suffix;
- bypass the common banana funnel.

Keep normal composer output at 260-280 characters. Unit tests enumerate every
banana-route prop substitution table entry and prove that the longest legal
one-prop growth still leaves the post-substitution prompt at or below 320. The
common funnel runs once, but its `cap_phrase_safe` branch must stay dormant for
every current Ghost substitution. If a future substitution crosses 320, fail the
table-wide test and rebalance the composer; do not make runtime trimming the
subject/action/shot-law preservation mechanism. After the funnel, assert the
320 cap and restamp the final prompt hash/length.

Reuse `_stamp_prompt_meta` with `prompt_source = "ghost_signal"`, and preserve
through the manifest/report path:

- prompt source/version and prompt SHA/length;
- negative SHA/length;
- selected visual-style id;
- ordered slot-presence list;
- subject-sigil SHA for character beats;
- cadence mode, model/source/delivered counts, tail trim, and delivery scale mode.

Extend the strict `run_episode` trace whitelist with the new prompt version,
negative hash/length, prompt-slot list, subject-sigil hash, cadence fields, and
delivery mode. Prompt fields come from request observability; cadence/delivery
fields come from the returned canonical clip/manifest row, never from request
intent. Otherwise node 92 `/history` would silently drop the receipt.

Concretely, keep the prompt keys in the existing `obs` loop. Immediately after
`render_beat_coverage` returns `clip`, copy the cadence/delivery keys into the
trace row from `clip` with explicit `if key in clip` checks. Do not append those
names to the request-observability loop: they do not live there and would all be
silently absent.

The exact added observability keys are `prompt_version` (constant
`"ghost_signal_v1"`), `negative_sha8`, `negative_chars`, `prompt_slots` (ordered
list of emitted slot names), and `subject_sigil_sha8`; the exact clip/manifest
keys copied into trace are `model_frame_count`, `cadence_mode`,
`cadence_source_frame_count`, `cadence_delivered_frame_count`,
`cadence_tail_trim`, and `delivery_scale_mode`. Existing `prompt_source`,
`prompt_sha8`, `prompt_chars`, and `visual_style` remain in the whitelist.

Add a Ghost-specific prompt audit. The current LTX diversity audit filters on a
different prompt source and must remain untouched.

## 7. Implementation slice C -- honest enlargement

The delivery choice travels as declared data, never as an engine-name check.

### 7.1 Schema and propagation

`CanonicalClip` and `ShotRow` inherit the repository's Pydantic
`extra="forbid"` contract, so every new value must be schema-real. Add these exact
optional fields to `CanonicalClip` in `nodes/_otr_video_engines/schemas.py`:

```text
delivery_scale_mode: Optional[str] = None
cadence_mode: Optional[str] = None
cadence_source_frame_count: Optional[int] = None
cadence_delivered_frame_count: Optional[int] = None
cadence_tail_trim: Optional[int] = None
model_frame_count: Optional[int] = None
```

Add `subject_sigil: Optional[str] = None` to `ShotRow`. Then:

1. stamp all six clip fields from the adapter's actual result;
2. copy all six explicitly in `render_driver.build_clip_manifest` beside
   `native_frame_count`/`extension_mode` rather than hiding them in `qc`, but
   only when the key exists on the clip (`if key in clip`). Do not add six null
   keys to every legacy manifest row;
3. copy all six in `render_driver._clip_summary`, the strict run trace, and
   `otr_video_render_batch`'s lossless `per_clip` receipt; these manual
   projections are independent of the main manifest and must not silently drop
   schema-real data. Use the same present-key-only rule on legacy rows;
4. add only `delivery_scale_mode` and `cadence_mode` to
   `_ROLLUP_IDENTITY_FIELDS`. `model_frame_count`, source/delivered counts, and
   tail trim vary by beat and remain per-clip only;
5. thread only `delivery_scale_mode` through
   `otr_silent_composite.plan_timeline_segments`: add it to `emit`, pass the
   source row's value at both real-clip calls (positioned and sequential), and
   copy the last row's value in the generic closing-loop call. Floor/black rows
   keep it `None`. Ghost's exact coverage makes that closing copy unreachable
   for a successful all-Ghost episode, but the shared projection must remain
   lossless for mixed manifests;
6. add the ordered delivery-mode vector—not cadence counts—to composite
   cache/`IS_CHANGED`, and consume the mode only in the real-clip encoder.

Unknown explicit values fail loud. Field absence retains the existing real-clip
behavior byte-for-byte for all older lanes.

### 7.2 Scale modes

Extend `_scale_filter` to
`_scale_filter(..., *, sharpen=None, mode=None, pad=True, ...)`, retaining its
existing `sharpen=True/False` call semantics. `mode` plus an explicitly
conflicting `sharpen` value fails loud. When `mode is None`, `sharpen=True` maps to legacy real-clip
behavior and `sharpen=False` maps to legacy floor/gap behavior. That preserves
every old caller and emitted filter byte-for-byte while adding one new mode:

| Mode | Required filter behavior |
|---|---|
| legacy real clip | Lanczos + existing unsharp + existing pad + fps, byte-identical |
| legacy floor/gap | Bilinear + existing pad + fps, byte-identical |
| `lanczos_clean_full_frame` | `scale=<w>:<h>:flags=lanczos,fps=<fps>` using the caller's actual `w`, `h`, and `fps`; no aspect-ratio pad/crop stage and no unsharp. Ghost supplies `1920`, `1080`, and `25` |

The new mode replaces only the spatial scale/unsharp/pad portion. `_scale_filter`
must preserve its existing `pre` and `post` strings verbatim, so `_seg_vf` still
emits any `trim,setpts` prefix and its `tpad=stop_mode=clone:stop_duration=3600`
safety suffix. Ghost's exact decoded/encoded count means that tpad is present but
never supplies frames in a successful Ghost beat.

The Ghost source is already exact 16:9, so direct scaling does not distort its
aspect. The adapter's existing `canonicalize` probe must assert exact `512x288`
before it is allowed to stamp the mode. The composite trusts that CanonicalClip
contract and launches no redundant input probe.

The complete existing-call inventory is frozen before refactor:

| Call site | Legacy mapping / new input |
|---|---|
| `normalize_to_silent_canonical` | legacy floor/gap (`sharpen=False`) |
| `_seg_vf` via a normal real clip | legacy real clip (`sharpen=True`) |
| `_encode_segment_from_dir` still background | legacy real clip |
| `_encode_segment_from_dir` video/floor background | legacy floor/gap |
| `_encode_segment_from_dir` RGBA foreground | legacy real clip with `pad=False` |
| `plan_timeline_segments` floor/black branches | legacy floor/gap |
| `plan_timeline_segments` real-clip branch | pass that segment's `delivery_scale_mode`; absent means legacy real clip |

Thread the field through `plan_timeline_segments.emit` -> segment dict ->
`_encode_segment(..., delivery_scale_mode=...)` ->
`_seg_vf(..., mode=...)`. In `_encode_segment`, clean-full-frame forces the ffmpeg
fast path with an explicit first guard, before any source-size/model branch, and
may never call `_run_model_pipeline`, even if a mixed manifest has loaded a model
for legacy rows. Freeze every legacy `_scale_filter` string before the refactor
and assert them byte-for-byte afterward, including `pre`, `post`, labeled, and
`pad=False` forms.

At `assemble_silent_timeline`'s real-clip call, pass
`delivery_scale_mode=seg.get("delivery_scale_mode")` explicitly alongside
`loop`, `sharpen`, and `engine`; otherwise the segment field is dead. The generic
closing-tail `emit` also forwards the last real row's mode. No Ghost acceptance
run may actually exercise that loop: exact beat coverage must leave no Ghost
tail to reuse.

Add one pure `_has_model_eligible_clips(manifest)` predicate. It returns true only
for a manifest row that is a real on-disk video clip (`exists`, non-empty file
path, not `type="directory"`) and whose `delivery_scale_mode is None`, the exact
legacy/model-eligible state. `lanczos_clean_full_frame` is ineligible and every
other explicit value fails validation before this predicate. At
`OTR_SilentComposite.composite`, call it immediately after manifest parsing and
before the current `_assert_upscale_usable`/`_get_upscale_engine` calls. Only
resolve, assert, fingerprint, and load the selected non-off upscaler when
`assemble` is true **and** the predicate is true. Otherwise use the safe existing
`_get_upscale_engine("off")` sentinel without resolving the stale selection.
Thus a Ghost-only manifest ignores a stale ESRGAN choice without requiring its
model/dependency; a mixed manifest loads it once for eligible legacy rows while
Ghost still takes the forced ffmpeg path. The Ghost profile also selects
`upscale_stage.engine = "off"` as the first defense.

`IS_CHANGED` calls the same `_has_model_eligible_clips` helper, includes the
ordered per-row delivery-mode vector, and includes a model fingerprint only when
the helper says the model is active. Changing only a Ghost delivery mode therefore
invalidates the composite without making an inactive stale engine a dependency.
Likewise, exclude `OTR_COMPOSITE_UNSHARP_AMOUNT` from the Ghost-only fingerprint;
it cannot affect any clean-mode row.

## 8. Implementation slice D -- registry, profile, and canonical route

Land all registration/wiring surfaces in the same green implementation change.

1. Add the guarded cold import to
   `nodes/_otr_video_engines/__init__.py` before the final roster audit.
2. Add the complete independent `CAPABILITIES` row in
   `nodes/_otr_video_engines/registry.py` with exactly:
   `required_toolchain=None`, `requires_sidecar=False`,
   `device_backends=["cuda"]`, `requires_vendor=None`, `needs_fp8_te=False`,
   `needs_fp4_te=False`, `practical_without_gpu=False`,
   `sidecar_conditional=False`, and
   `model_requirements=["v1-5-pruned-emaonly-fp16.safetensors",
   "mm-p_0.5.pth"]`. Do not add novel capability keys.
3. Do **not** add an `_PUBLIC_ENGINES` self-alias. The resolver already passes a
   bare internal id through unchanged, matching the existing identity-engine
   precedent. Add only
   `_PUBLIC_LABEL["animatediff15_video"] = "AnimateDiff -- Ghost Signal"` in
   `nodes/_otr_shared/public_engines.py`; the public id intentionally has no
   unmeasured `low`/`high` token. Pin passthrough resolution, menu value, and
   friendly label without changing the alias-table bijection.
4. Add the exact internal-id fallback
   `"animatediff15_video": "text_to_video"` to
   `nodes/_otr_shared/content_oracle.py`. This is registry coherence, not an
   instruction to run the luma/motion oracle during this build.
5. Add `config/profiles/otr_ghost_signal.json` with:
   - the complete validated shape of `config/profiles/16gb_full.json` copied as
     a structural baseline—not a runtime inheritance mechanism—so the checked-in
     file contains `id`, `display_name`, `status`, `platform`, `device_backend`,
     `gpu_vendor`, `toolchains`, `allow_sidecars`, `role_overrides`,
     `slot_overrides`, `features`, `seed_policy`, `llm`, `video`, `image`,
     `audio`, `render`, `preflight`, `launch`, and `upscale_stage`;
   - id `otr_ghost_signal`, a neutral Ghost Signal/AnimateDiff display name, and
     status `draft` until the publish smoke passes;
   - exact `role_overrides` keys `announcer_visual`, `music_visual`, and
     `character_visual` set to the new public id (the profile key is
     `character_visual`; only the engine capability token is `character_video`);
   - `slot_overrides.video_render_engine` set to the same public id;
   - all unrelated voice, music, image, LLM, feature, seed, and sidecar values
     preserved byte-for-byte from that baseline; the retained image selections
     are proven bypassed by `accepts_still=False`;
   - `video.device_policy="cuda"` and
     `video.dtype_policy="no_fp8_no_fp4"`; the files are FP16 and this profile
     must not silently opt them into ComfyUI's optional FP8/FP4 transformations;
   - render `fps=25`, `canvas_w=512`, `canvas_h=288`,
     `composite_res="1920x1080"`, `composite_w=1920`, `composite_h=1080`, and
     the baseline harness-only `frame_budget=25`, `beats=40`; do not add
     `video.max_render_frames` because the beat contract is unbounded and
     audio-derived;
   - `upscale_stage={"engine":"off","device":"cpu"}`;
   - `preflight.required_models` equal to the two exact `model_requirements`
     strings above and baseline `required_keys=[]`;
   - `launch.sage_attention=false`, no `boot_contract`, and empty
     `extra_args=[]` / `env={}`; the pinned graph uses ordinary Comfy sampling
     and declares no sidecar or launch-time patch;
   - no Ghost VRAM ceiling claim or low/high label. Any preserved LLM runtime
     ceiling remains the baseline's unrelated LLM policy and is not evidence
     about this video lane.
6. Use the existing profile applier against the real
   `workflows/otr_canonical.json` in memory; do not hand-type node 87 values and
   do not commit the applied Ghost choices back into the source workflow. Verify
   all three director roles, `OTR_VideoRenderBatch.engine`, native canvas,
   composite dimensions, fps, and node 84's `upscale_engine=off` on the applied
   graph.
7. The implementation makes **no** canonical JSON edit: Ghost is an in-process
   adapter reached through existing widgets, and `apply_profile` is pure. The
   source remains 23 nodes, 57 links, and 140 widget values. If discovery proves
   a new OTR node/socket/widget/link is actually necessary, stop and re-plan;
   this plan's no-topology premise has failed. Any later approved wiring change
   must then land in `workflows/otr_canonical.json` with its code, never only in
   a generated variant.

   The R3 outer-workflow audit is therefore an exact **applied-profile** ledger,
   not a source-JSON delta:

   | Canonical node | Linked inputs that remain unchanged | Applied widget vector, in positional order | Outputs that remain unchanged |
   |---|---|---|---|
   | node 87 `OTR_VideoDirector` | `gate_in <- link 269` | `["animatediff15_video", "animatediff15_video", "animatediff15_video", "z_image_turbo", "z_image_turbo", "z_image_turbo", 25, 512, 288, "request_hash", 42, "{}", "cuda", "no_fp8_no_fp4", 0]` for `announcer_video_model`, `music_video_model`, `character_video_model`, the three image models, `fps`, `canvas_w`, `canvas_h`, `seed_mode`, `request_seed`, `custom_models_json`, `device_policy`, `dtype_policy`, `max_render_frames` | `video_policy_json[0] -> links 251, 270` |
   | node 92 `OTR_VideoRenderBatch` | `patched_ledger_json <- 260`; `master_audio_path <- 264`; `image_done <- 267` | `["episode", 40, 20, 25, "animatediff15_video", "", ""]` for `mode`, `beats`, `oom_index`, `frame_count`, `engine`, `portrait_path`, `audio_path` | `render_report_json[0]` unlinked; `clip_manifest_json[1] -> links 261, 271, 275, 278` |
   | node 84 `OTR_SilentComposite` | `base_video_path <- 246`; `clip_manifest_json <- 261`; optional `gate_in` remains unlinked | `[1920, 1080, 25, "ffmpeg", "", "off", "cpu"]` for `canvas_w`, `canvas_h`, `fps`, `ffmpeg`, `output_path`, `upscale_engine`, `upscale_device` | `silent_video_path[0] -> link 247`; `report[1]` unlinked |

   The live director menu displays the derived
   `animatediff15_video (16:9)` label, while the profile and saved value stay the
   bare internal id by the existing non-aliased-engine rule. The applier audit
   must prove all three bare values resolve back to the one live menu option.
8. Add `animatediff15_video` to
   `scripts/build_video_evidence_manifest.py`'s `admission_unenforced` table with
   this complete sentence: **“Admission is NOT enforced: the operator declined a
   measurement campaign; no qualified cost row exists, so this lane may fail OOM
   and makes no VRAM-fit claim.”** Do not add it to `QUALIFIED_COST_ROWS` or invent
   a cost-model row.
9. Regenerate `docs/ENGINE_MATRIX.md` with `tools/engine_matrix.py`.

Add a separate `gate_g3_7(name, eng)` beside the existing **G3** checks; do not
fold unconditional declarations into `gate_g3_contract`, and do not mint a new
G9 (the live matrix is G1-G7 and the documentation already uses Gate 8 for the
solo smoke). The helper returns no findings unless `family == "text_to_video"`
and `accepts_still is False`; once applicable, it requires:

- explicit `subject_ownership = "prompt"`;
- declared positive prompt profile and budget;
- declared style join (`compose`, `override`, or a named owned pack);
- ledger motion source present;
- negative prompt has a real graph binding.

Invoke it for every registry name in the existing matrix loop. That predicate
keeps older still-owned text-to-video lanes out while ensuring a Ghost class that
forgets `subject_ownership` or `prompt_profile` fails rather than escaping its own
gate.

## 9. Correctness test plan

Tests are deterministic contract checks, not the disallowed measurement
campaign.

### 9.1 New focused tests

Create `tests/test_ghost_signal_lane.py` covering:

- registry/CAPABILITIES, bare-id passthrough/friendly label, and all-role
  eligibility;
- explicit no-still declarations and empty valid still plan, including the real
  image-dispatcher capability lookup proving no still is minted for any Ghost
  role;
- fixed canvas, aspect, frame contract, continuity, prompt/delivery capabilities;
- cold import with torch/transformers/diffusers/AnimateDiff absent;
- the exact seven-name resolver map, eight node instances, named input
  dictionaries, output-slot arities, GUI widget vectors, ten semantic typed
  links, and the four executor graphs' exact one-slot alias/`Wire(..., 0)`
  ledger in section 4.3;
- the ADE optional-socket omission set is exact: no accidental `None` input and
  no sampling-settings/multival/keyframe/per-block/view node appears;
- named refusals for missing pack, checkpoint, motion module, checkpoint VAE
  output slot, or any one live class;
- no runtime download or alternate class/asset;
- decoded-count mismatch refusal;
- concrete `load()` override is called by base `prepare`; base and ADE patchers
  are registered through `run_graph(on_result=...)` before eviction; a normal
  render detaches and identity-removes ADE then base before decode; partial graph
  failure and raising unload still release the lease through base teardown;
- stage-order and ownership proof: checkpoint executes once; positive and
  negative encode before sampling; the encode result/external containers and
  owner CLIP field are cleared before the post-encode reclaim seam; the sample
  graph starts only afterward; exactly one KSampler call executes;
- post-sample release proof: the two distinct patchers detach in ADE/base order,
  a duplicate identity detaches once, successful identities disappear in place
  from both patcher-list aliases, all sample/conditioning owners clear, the
  post-sample reclaim seam runs, and only then may `VAEDecode` execute;
- release-failure proof: a raising or non-callable ADE/base detach keeps that
  exact object tracked, prevents `VAEDecode`, raises the named
  `GraphExecutionError`, and outer `finally` reaches teardown and lease release;
- no `compute_real_frame_budget`, still authoring, chaining, loop, mirror, RIFE,
  IPAdapter, ControlNet, second sampler, or upscaler call.
- `CanonicalClip` accepts each declared cadence/delivery field, rejects unknown
  extras, and `ShotRow.subject_sigil` is optional string data.

Create `tests/test_ghost_signal_prompt.py` covering every role across every
shipped style pack:

- exact slot order and content-gated omission;
- byte-identical input -> byte-identical output;
- 320 hard cap and 260-280 normal target;
- the pinned sigil vocabularies, neutral pools, SHA-256 domain, output ceiling,
  source priority, name/camera/background stripping, explicit-gender handling,
  and durable identity across a character's beats;
- authored motion-clause precedence plus deterministic pack/ledger/hash fallback
  when the optional motion pass is off or the cast-time stub has no clause;
- whole motion-clause and shot-law survival;
- no raw dialogue/title/M4/proper-name/second-person leakage;
- style cue first; lane hygiene first in the negative; pack negative phrases
  preserved in order up to the 320-character whole-phrase ceiling;
- banana-on preservation of subject, framing, action, and shot law for every
  one-prop substitution entry, with the longest substitution still at or below
  320;
- M4 creative text present on a Ghost shot is ignored, the Ghost branch supplies
  positive and negative text, and both the M4 and LTX branches remain skipped;
- the generic 1940s request seed cannot satisfy Ghost: missing sigil or a skipped
  composer raises during cast-time and render-time request construction;
- an all-Ghost policy makes `derive_creative_directives` return before a
  raising writer-LLM resolver, while a still-consuming legacy policy keeps its
  existing authoring path;
- missing/blank `negative_prompt` is refused and no adapter constant is used;
- explicit regression that the LTX face suffix never lands.

Add pure cadence cases for `T = 1, 2, 3, 12, 13, 49, 50` proving:

- `U = ceil(T/2)` from `T`, not from duration;
- `source_request = max(U, 16)` for every case, with only the first `U` decoded
  source frames eligible for the cadence selector;
- exact output length;
- source indices in range and monotonically nondecreasing;
- source indices `0..U-2` appear exactly twice and the last appears twice for
  even `T`, once for odd `T`;
- tail trim is only 0 or 1;
- output duration is exactly `T/25`;
- receipts survive canonicalization and manifest aggregation.
- `source_request > U` records model work and retained cadence source count in
  separate fields without changing `cadence_tail_trim`.
- two beats with identical prompt text but different shot ids receive distinct
  deterministic seeds, source-batch identities, and output paths; no cross-beat
  cache/path reuse occurs.

### 9.2 Existing suites to extend

- `tests/test_motion_clause.py`
- `tests/test_visual_styles_b.py`
- `tests/test_visual_style_negative.py`
- `tests/test_brief_prompt_finishing.py`
- `tests/test_ltx_av_quality_wire.py` for exact third filter and unchanged legacy strings
- `tests/test_clip_fill.py` for manifest-to-segment-to-`assemble_silent_timeline`
  delivery mode, including positioned/sequential/closing emit sites
- `tests/test_video_schemas_additive.py` for strict optional schema admission
- `tests/test_render_engines_recipe_stamp.py` for present-key-only per-clip
  receipts and mode-only engine rollup identity
- `tests/test_upscale_composite_single_base_no_engine_load.py`
- `tests/test_upscale_cache_fingerprint.py`
- `tests/test_capability_profiles.py`
- `tests/test_public_engines.py`
- `tests/test_lane_preflight_matrix.py`
- `tests/test_boot_contracts.py` for the complete-sentence G4
  `admission_unenforced` receipt

Registry-driven gates must also remain green:

- `tests/test_engine_contract_roster.py`
- `tests/test_frame_receipt_conformance.py`
- `tests/test_still_spine_engine_coverage.py`
- `tests/test_still_plan_audit.py`
- `tests/test_multiclip_session_identity_roster.py`
- `tests/test_video_platform_aseam.py`

### 9.3 Canonical workflow gates

Against `workflows/otr_canonical.json`, run:

1. `OTR_WorkflowValidator`;
2. JSON parse/serialize/parse round-trip;
3. link referential-integrity audit;
4. live `INPUT_TYPES` vs positional `widgets_values` count/name audit;
5. profile-apply assertions for all three roles, render engine, fps, canvases,
   dtype, frame ceiling, composite encoder, and upscale-off, using the three
   exact applied widget vectors in section 8;
6. the saved-workflow value resolver tests.

The profile-apply test uses `character_visual` in profile JSON and proves it lands
on the canonical workflow's character-video widget. It also proves the new G3.7
predicate applies to Ghost but not to existing still-owned engines.

The source baseline must remain exactly 23 nodes, 57 links, and 140 serialized
widget slots. A mismatch stops this implementation; it is not permission to
edit topology under the existing plan.

## 10. One allowed GPU acceptance run

After all CPU/unit/canonical gates pass:

1. Reset the box selectively and confirm port 8000 is free.
2. Boot once with the UTF-8 launcher and the Ghost dependency lock.
3. Load the **real** `workflows/otr_canonical.json` and apply
   `otr_ghost_signal`; do not use an ad-hoc graph or generated stale copy.
4. Render one short, fixed-seed episode containing at least one announcer, one
   music, and one character beat.
5. Write all episode assets directly to `otr/episodes/<episode>/` and the final
   published episode directly to `otr/obs/`.

Pass only when:

- the process does not OOM or silently fall back;
- every beat routes to `animatediff15_video`;
- each manifest row has exact target frame count, silence proof,
  `extension_mode="none"`, `cadence_mode="hold_2"`, and
  `delivery_scale_mode="lanczos_clean_full_frame"`;
- every beat has its own shot-derived seed, source batch identity, and clip path;
  `cadence_source_frame_count == ceil(target_frame_count/2)`, no path/batch is
  shared across beats, and no loop, mirror, chained clip, or held-tail extension
  is reported. The only duplicate delivery frames are the declared hold-2 pairs;
- final output is 1920x1080 at 25 fps with no audio before mux;
- `RESULT SUCCESS` and `obs_publish OK` appear;
- the canonical episode assets and final `otr/obs` file exist.

Do not collect or publish peak VRAM, wall time, luma, freeze detection, SSIM,
PSNR, tokenizer counts, prompt comparisons, canvas comparisons, or aesthetic
scores. If the run OOMs, record only the named failure, keep the profile `draft`,
and stop. Do not add a hidden fallback or make an “under 4 GB” claim.

## 11. Green-chunk and git sequence

Future implementation uses three pushed green chunks on `v2.0-alpha`:

1. **Dependency/schema lock:** the two Phase-0 receipt files only. Validate JSON,
   UTF-8/no BOM, then commit and push.
2. **Single wired implementation slice:** adapter, composer, sigil schema/stamp,
   delivery mode, registry/public/profile/preflight/G4 receipt changes,
   in-memory canonical profile-apply coverage, generated engine matrix, and all
   tests. Run focused tests,
   the full regression suite, and the Bug Bible; commit and push only when all
   are green.
3. **Publish receipt/promotion:** run the one smoke. On pass, add the preflight
   receipt, change the profile from `draft` to `shipping`, rerun the full suite +
   Bug Bible + canonical audits, then commit and push. On OOM/failure, add no
   shipping promotion.

After every push verify HEAD equals origin, touched Python parses, no touched
file is zero bytes, and no touched text file has a BOM. No implementation commit
may leave new adapter code registered but unavailable through the real canonical
profile route.

## 12. File-by-file change manifest

| File | Planned change |
|---|---|
| `docs/2026-08-22-ghost-signal-dependency-lock.json` | Pinned upstream/schema/artifact/license identity |
| `docs/2026-08-22-ghost-signal-object-info.json` | Exact live AnimateDiff class/input/output subset |
| `nodes/_otr_video_engines/eng_ghost_signal.py` | Adapter, graph, cadence, receipts, lifecycle |
| `nodes/_otr_video_engines/ghost_signal_prompt.py` | Pure role-aware composer, deterministic sigil distiller, motion fallback, budgets |
| `nodes/_otr_video_engines/schemas.py` | Exact optional `subject_sigil`, delivery, and cadence fields under `extra="forbid"` |
| `nodes/_otr_video_engines/render_driver.py` | Capability-based prompt branch and receipt propagation |
| `nodes/otr_shot_lock.py` | One-time durable subject-sigil stamp and preflight parity |
| `nodes/_otr_video_engines/__init__.py` | Guarded registration import |
| `nodes/_otr_video_engines/registry.py` | Capability/dependency declaration |
| `nodes/_otr_shared/public_engines.py` | Friendly label only; bare internal/public id passes through without a self-alias |
| `nodes/_otr_shared/content_oracle.py` | Bare-script family fallback row |
| `nodes/otr_silent_composite.py` | Explicit clean-Lanczos mode, propagation, model-load gate |
| `nodes/otr_video_render_batch.py` | Present-key per-clip receipts; delivery/cadence modes only in engine identity rollup |
| `config/profiles/otr_ghost_signal.json` | Draft/shipping all-role route, fixed canvases, upscale off |
| `workflows/otr_canonical.json` | Validation/input target only; expected source diff is none (23/57/140 stays fixed) |
| `docs/VIDEO_LANE_PREFLIGHT.md` | Scoped prompt-owned/no-still G3.7 contract |
| `tests/test_lane_preflight_matrix.py` | Scoped G3.7 enforcement plus live-lane coverage |
| `scripts/build_video_evidence_manifest.py` | Truthful G4 admission-unenforced sentence; no cost row |
| `docs/ENGINE_MATRIX.md` | Regenerated registry surface |
| focused/existing tests in section 9 | Contract and regression coverage |

## 13. Definition of done

Ghost Signal is done only when all of the following are true:

- canonical AnimateDiff-Evolved and exact assets/schema are pinned;
- no guessed class/input name exists;
- all three roles select the lane through a checked-in profile and the real
  canonical workflow route;
- still authoring is bypassed for all three roles;
- selected ledger motion and visual-style pack are traceably present in the
  request and wired into positive/negative conditioning;
- every delivered frame traces through the declared hold-2 selector to a decoded
  same-beat AnimateDiff frame, with truthful cadence receipts; each beat owns a
  distinct generated source timeline covering its full audio budget, and no
  cross-beat source/clip reuse or undeclared tail fill occurs;
- full-frame clean Lanczos is capability-driven and no sharpen/upscaler path can
  engage for Ghost rows;
- focused tests, full suite, Bug Bible, engine matrix, preflight matrix, and all
  canonical workflow audits are green;
- G4 states admission is unenforced and the project makes no VRAM-fit claim;
- the single real smoke publishes from the canonical workflow to `otr/obs`;
- the profile is promoted to `shipping` only after that pass;
- no measured 4 GB, quality, or speed claim is made.
