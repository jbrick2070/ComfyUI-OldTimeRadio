# Codex design review — Haunted image-to-video

- **CONFIRMED — Review target:** `docs/SPEC_haunted_image_to_video.md` on branch `v2.0-alpha`, reviewed at commit `07a470083717ca753ad4ced4c06e6e469aa45ac7` on 2026-08-30.
- **CONFIRMED — Scope:** this is a read-only design review. No engine, profile, workflow, or SPEC file was changed.
- **CONFIRMED — Grounding:** the review checked the requested engine, registry, public-engine, oracle, profile, dispatcher, schema, render-driver, still-plan, test, evidence, installer, and production-bug files. Upstream claims use primary AnimateDiff, AnimateDiff-Evolved, Advanced-ControlNet, ComfyUI, and paper sources.
- **UNVERIFIABLE — Supplied live measurements:** the reported ADE commit/node-class counts and weight totals were not re-derived, per instruction. They are treated as measured inputs except where current repository evidence below narrows or contradicts their interpretation.

## VERDICT

**INFERRED — Build it differently:** keep `animatediff15_v3_haunted_video` unchanged, require `init_image` on any new I2V sibling, and do not ship latent-init Route A under the proposed durable engine identity. **CONFIRMED — The SPEC's Route-B blocker is false:** current AnimateDiff-Evolved explicitly delegates SparseCtrl to the separate `ComfyUI-Advanced-ControlNet` pack; current ACN 1.6.0 exposes the non-experimental SparseCtrl loader and RGB preprocessor, so evaluating Route B does not require upgrading the proven ADE 1.6.0 installation. **INFERRED — Qualify ACN 1.6.0 in isolation through the canonical workflow, then build the purpose-built SparseCtrl lane if the fixed-seed probe survives quality and real OOM tests on both cards.** **INFERRED — If Route A is retained after its denoise sweep, it is a separate `latent_i2v` engine with its own recipe and receipt, not an implementation later replaced in place.** **CONFIRMED — No new dropdown/profile row may ship until its exact dependency and model bundle has loaded and rendered on the machine that offers it.**

## MUST-FIX

### 1. Decision: `init_image` is REQUIRED

- **CONFIRMED — Required contract:** the new sibling must declare `family = "image_to_video"`, `required_inputs = ("text_prompt", "init_image")`, and `accepts_still = True`. `schemas.py:55-66` requires `init_image` for every I2V request, while the inherited Ghost renderer independently rejects blank positive and negative prompts at `eng_ghost_signal.py:693-739`.
- **CONFIRMED — H3 is not a drop-in metadata template:** H3 can declare only `("init_image",)` because it has its own prompt fallback and a dedicated first-frame conditioner (`eng_minimax_h3.py:696-699, 1010-1084`). Ghost explicitly has no such text fallback.
- **CONFIRMED — `negative_prompt` cannot simply be added to `required_inputs`:** it is not in the closed request-token vocabulary in `schemas.py:14-49`; its nonblank requirement remains a renderer/composer invariant.
- **CONFIRMED — OPTIONAL is a platform change, not a subclass choice:** current I2V schema validation requires `init_image`, and the renderer's optional-still path is limited to provider-side engines (`schemas.py:171-193`; `render_driver.py:2238-2269`).
- **INFERRED — Cost of REQUIRED:** every selected visual role must mint, load, and hand off a still; image-generation latency, image-model artifacts, upstream image failure, and image-to-video VRAM phases become part of the lane's promise. The existing prompt-only haunted sibling preserves the no-image selling point, so an I2V-to-T2V fallback adds no capability and can hide a broken handoff.

### 2. Decision: reject the Route-A-versus-ADE-upgrade framing

- **CONFIRMED — Route A is mechanically valid generic img2img:** ComfyUI documents VAE-encode plus denoise below 1.0 as image-to-image, and ADE states it works with ordinary/custom KSamplers. Lower denoise preserves more source content; higher denoise permits more change. [ComfyUI img2img](https://docs.comfy.org/tutorials/basic/image-to-image), [ADE README](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/main/README.md)
- **CONFIRMED — Route A is not AnimateDiff v3's official image-conditioning mechanism:** v3's published I2V configuration combines `v3_sd15_mm.ckpt`, `v3_sd15_adapter.ckpt`, and RGB SparseCtrl with the source at frame index `0`. [Official v3 I2V configuration](https://github.com/guoyww/AnimateDiff/blob/main/configs/prompts/3_sparsectrl/3_1_sparsectrl_i2v.yaml)
- **CONFIRMED — Route B needs ACN, not an ADE upgrade:** ADE's own README points to `ComfyUI-Advanced-ControlNet` for SparseCtrl. Current ACN 1.6.0 requires ComfyUI `>=0.3.68` and registers `ACN_SparseCtrlLoaderAdvanced`, `ACN_SparseCtrlIndexMethodNode`, and `ACN_SparseCtrlRGBPreprocessor`. [ADE dependency guidance](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/blob/main/README.md), [ACN package metadata](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/blob/main/pyproject.toml), [ACN SparseCtrl nodes](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/blob/main/adv_control/nodes_sparsectrl.py)
- **CONFIRMED — Matching version numbers are not compatibility proof:** ACN's DinkLink source checks an ADE boundary and warns that this cross-pack API can change. [ACN DinkLink](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/blob/main/adv_control/dinklink.py)
- **CONFIRMED — SparseCtrl is mature in implementation age and continued support:** AnimateDiff v3/SparseCtrl was released in December 2023, and current ACN 1.6.0 still registers the ordinary SparseCtrl loader without the experimental marker; only its merged loader is marked experimental. [AnimateDiff v3 model notes](https://github.com/guoyww/AnimateDiff/blob/main/README.md), [current ACN implementation](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/blob/main/adv_control/nodes_sparsectrl.py)
- **UNVERIFIABLE — Production maturity on this haunted stack:** upstream documentation does not prove ACN 1.6.0 + installed ADE commit `9257651` + this OTR adapter + both GPUs. The repository currently has zero live SparseCtrl classes, so boot, graph, receipt, and memory behavior remain unqualified.
- **INFERRED — Decision:** prefer the correct SparseCtrl mechanism after an isolated ACN qualification; use Route A only as a cheap hypothesis probe. A weaker mechanism is not preferable merely because the SPEC incorrectly assigned the dependency to ADE.

### 3. Decision: one shared ingestion/lifecycle seam, two public engines

- **CONFIRMED — There is no current image seam to inherit:** Ghost request normalization omits `init_image`, node candidates omit an image loader/encoder, and its sampling graph hard-wires `EmptyLatentImage` and module-global denoise (`eng_ghost_signal.py:153-160, 441-462, 693-725, 810-827`).
- **INFERRED — Shared internal seam:** refactor `eng_ghost_signal.py` narrowly so a sibling can own image materialization, resize/crop, VAE preprocessing, initial latent/conditioning construction, denoise/control strength, and lifetime boundaries while the existing T2V path retains identical constants and graph behavior.
- **INFERRED — Separate public identities:** latent-init changes the sampler's initial latent; SparseCtrl adds a control model, sparse index/mask semantics, and a different conditioning path. If both ship, each needs its own engine id, artifact list, recipe id, receipt, cache identity, profile qualification, and evidence row. Silently swapping Route B behind Route A's id would make old renders unreproducible and their receipts ambiguous.
- **INFERRED — Do not force a false abstraction:** image file loading, normalization, canvas policy, source hashing, and staged model cleanup are genuinely shareable; the latent-init and ControlNet graphs are not necessarily the same hook.

### 4. The proposed class cannot merely “inherit everything else”

- **CONFIRMED — Own declarations are required:** the I2V sibling must own a truthful nonempty still plan, subject ownership, image-aware request normalization, init-image identity, denoise/control constants, node candidates/preflight, and a distinct recipe receipt. The parent currently owns prompt-only values at `eng_ghost_signal.py:334-352, 417-462`.
- **CONFIRMED — Image staging is missing:** presence of `VAEEncode`, `RepeatLatentBatch`, `KSampler`, and `ImageScale` does not materialize `asset_refs.init_image` into a ComfyUI IMAGE. Existing local I2V code has explicit staging/materialization machinery (`wan_shared.py:478-604`); Route A needs an equivalent path or a shared helper.
- **CONFIRMED — Cache/provenance identity is currently prompt-only:** `shot_cache_identity` hashes prompt, negative prompt, shot/seed/canvas, and artifacts, but not init-image content or denoise (`eng_ghost_signal.py:441-462`).
- **UNVERIFIABLE — Current runtime cache blast radius:** repository search finds the identity definition and tests but does not establish a current production consumer. Nevertheless, an I2V identity that treats two different stills as the same request is a false contract and must not ship.

### 5. Correct the folded `still_plan` finding

- **CONFIRMED — The folded review's claimed failure mechanism is wrong:** inherited `still_plan = ()` does not make the production dispatcher mint zero stills when the new class declares `accepts_still = True`. The dispatcher gives explicit `accepts_still` priority and then checks `required_inputs` (`otr_image_gen_dispatcher.py:636-653`); the render driver joins the scene still for I2V (`render_driver.py:2238-2270`).
- **CONFIRMED — Empty still plan still lies:** `still_plan_helpers.py:20-22, 271-307` defines `()` as a valid declaration meaning “needs no images.” It would misstate this lane's declared pixel contract and skip plan-authored kind/identity behavior, so the I2V sibling needs a nonempty scene plan even though empty plan is not the zero-mint switch alleged in SPEC lines 219-225.

### 6. Verified edit/output surface under the recommended design

- **CONFIRMED — `nodes/_otr_video_engines/eng_ghost_signal_official.py`:** add the registered sibling(s) and sibling-owned constants/receipts.
- **INFERRED — `nodes/_otr_video_engines/eng_ghost_signal.py`:** add the narrow behavior-preserving image/conditioning hook; otherwise the alternative is a duplicated monolithic render path.
- **CONFIRMED — `nodes/_otr_video_engines/registry.py`:** add a `CAPABILITIES` row. `@register` on the class performs registration; the SPEC's “one registration” description is inaccurate. Capability parity is enforced by `tests/test_capability_profiles.py:390-395`.
- **CONFIRMED — `nodes/_otr_shared/content_oracle.py`:** add the bare-script `_FAMILY_FALLBACK` row; the live registry remains production authority (`content_oracle.py:92-115`).
- **CONFIRMED — `config/profiles/<new-profile>.json`:** select the new video id for all intended visual roles, activate exact image roles, pin launch environment, and list every required model filename.
- **CONFIRMED — `workflows/variants/<profile>.json` and `workflows/variants/<profile>.launch.md`:** `scripts/build_variants.py:131-185, 397-410` emits both for every ordinary committed profile. The SPEC misses these generated artifacts.
- **CONFIRMED — `docs/evidence/video_evidence_manifest.json`:** add an explicit `admission_unenforced` statement until a real measurement qualifies a cost row; `tests/test_lane_preflight_matrix.py:796-808` requires one or the other. No estimated cost may kill a render.
- **CONFIRMED — `docs/ENGINE_MATRIX.md`:** regenerate with `tools/engine_matrix.py`; `tests/test_engine_matrix_doc.py:37-49` enforces parity.
- **CONFIRMED — `tests/fixtures/still_plan_head_parity.json`:** regenerate the exact roster. The folded review names nonexistent `tests/fixtures/still_plan_matrix.json`.
- **CONFIRMED — `tests/test_frame_contract.py`:** update its literal unbounded-engine set if the sibling inherits Ghost's unbounded timing (`tests/test_frame_contract.py:102-115`). This roster is not automatic.
- **CONFIRMED — `scripts/otr_w45_campaign.py` plus profile coverage:** every new local engine joins the W45 roster, which refuses a missing profile unless `PROFILE_EXCEPTIONS` maps it (`scripts/otr_w45_campaign.py:96-102, 140-178`). Add the campaign profile/variants or a deliberate exception.
- **INFERRED — Engine-specific tests:** extend `tests/test_ghost_signal_haunted.py` or add a focused module for missing-init failure, real pixel consumption, latent/control graph shape, image-aware receipt/cache identity, unchanged parent graph, lifecycle, and loud missing-dependency failure. **UNVERIFIABLE — Exact new test filename** is not dictated by existing code.
- **CONFIRMED — `nodes/_otr_shared/public_engines.py` is SHOULD, not a registration blocker:** add a friendly `_PUBLIC_LABEL` for usable menu prose; do not add a self-alias to `_PUBLIC_ENGINES`. The bijection assertion is `_PUBLIC_ENGINES` versus `_INTERNAL_TO_PUBLIC`, not `_PUBLIC_LABEL` (`public_engines.py:24-26, 338-363`).
- **CONFIRMED — `workflows/otr_canonical.json` needs no structural edit** if this remains an internal engine/profile addition using existing selectors, inputs, and links. Every probe and render must still load that exact canonical JSON. A new widget/input would make canonical, `config/profiles/widget_mapping.json`, and all variants mandatory same-change edits.
- **CONFIRMED — `nodes/_otr_video_engines/__init__.py` needs no import edit** if the class remains in `eng_ghost_signal_official.py`, which is already imported. A new module would require one.

### 7. Install promises are currently incomplete

- **CONFIRMED — `z_image_turbo` is not auto-downloading in this adapter:** `z_image_turbo.py:224-251, 481-502` resolves an installed UNET and fails closed when absent; later CLIP/VAE resolution can also fail. `scripts/download_4060_nano_models.ps1:63-68` explicitly says no image engine auto-downloads.
- **CONFIRMED — `scripts/otr_fetch_lane_weights.py` currently fetches only SD1.5, the v3 motion module, and the adapter for Haunted, while its header incorrectly claims image models auto-download (`scripts/otr_fetch_lane_weights.py:1-62, 88-93`).** Activating the inert image role without extending the fetch bundle and its tests violates the dropdown promise.
- **INFERRED — Route A shipping surfaces:** extend `scripts/otr_fetch_lane_weights.py`, `tests/test_lane_weight_fetcher_matches_lanes.py`, and the new profile's `launch.env`/`preflight.required_models` with an exact low-VRAM Z-Image UNET, CLIP, and VAE bundle. **UNVERIFIABLE — Exact bundle:** the existing profile does not select one, and code defaults differ from the measured low-VRAM recipe.
- **INFERRED — Route B adds:** the RGB SparseCtrl checkpoint in the fetcher/profile/preflight, ACN node-class preflight, fetcher tests, README/RunPod installation guidance, and a pinned ACN dependency receipt/lock. If Route B has its own id, it repeats the registry/oracle/profile/variants/evidence/matrix/fixture/test surfaces above.
- **CONFIRMED — The statement “ADE is declared nowhere except README” is stale on current HEAD:** it also appears in `docs/RUNPOD_INSTALL.md:109-112`, the shipping profile display name at `config/profiles/otr_nvidia_8gb_haunted.json:3`, the missing-class hint, and `docs/2026-08-22-ghost-signal-dependency-lock.json`. It remains absent from an automatic installer/machine-readable custom-node dependency mechanism, and root `__init__.py:11` still falsely claims no external node dependencies.
- **UNVERIFIABLE — Correct machine-readable cross-custom-node metadata key:** no authoritative mechanism is present in this repository, so this review does not invent one.

### 8. VRAM framing is inadequate; an open production failure already precedes the proposed stack

- **CONFIRMED — Disk size is not VRAM:** the 1.99 GB checkpoint size proves neither the SparseCtrl runtime peak nor that the complete graph fits 8 GB/14.5 GB.
- **CONFIRMED — The combined path has not completed, but image activation on 8 GB has already failed live:** PBUG-20260829-03 records a canonical 4060 run aborting at Z-Image sampler step 0 while DynamicVRAM streamed the 6.2 GB UNET after the writer (`docs/PROD_BUG_LOG.md:7745-7778`). Its amendment confirms the current shipping haunted profile avoids that hazard only because its T2V engine never activates the image role (`docs/PROD_BUG_LOG.md:8049-8065`).
- **CONFIRMED — This is worse than an ordinary catchable OOM:** the logged DynamicVRAM path called native `abort`, killing the server before OTR could catch, receipt, or degrade. The SPEC's generic “second image consumer changes VRAM shape” language omits this already-proven first-stage failure.
- **INFERRED — First measurement order on the 4060:** use a clean server and the real 512×288/16-source-frame canonical beat; record baseline, post-writer reclaim, Z-Image peak, post-`z_image_turbo post-decode` reclaim, SD1.5+motion+adapter load, Route-A VAE encode or Route-B SparseCtrl preprocess/load, sampling peak, decode peak, host-RAM/sysmem fallback, and wall time. Repeat the video half with an already-produced still to isolate image-model residue from video cost. Let actual OOM/abort decide; do not add an estimate gate.
- **INFERRED — Then measure the 5080:** run the same artifact and confirm absolute peak remains below the practical 14.5 GB ceiling; do not extrapolate from the 4060 or checkpoint bytes.
- **INFERRED — Additional lifecycle risk:** both Route A and ACN's RGB SparseCtrl preprocessor use the SD1.5 VAE before sampling, while Ghost also needs it after sampling for decode. A naive graph can keep/reload that dual-consumer VAE across the largest sampling phase; the repository's Bug Bible 07.16/07.22 warns that this can turn a no-OOM render into sysmem-offload crawl. Measure the phase boundary rather than assuming `free_after_use` proves release.

### 9. Single biggest unnamed failure

- **CONFIRMED — Biggest unnamed failure:** activating `z_image_turbo` on the shipping 8 GB path can native-abort the entire ComfyUI process before AnimateDiff or SparseCtrl starts, as already recorded by PBUG-20260829-03. This is not merely “unmeasured co-residency”; it is a known open lifecycle failure on the target hardware.
- **INFERRED — Consequence:** until the canonical image phase can generate and reclaim the selected exact Z-Image bundle on the 4060, neither Route A nor Route B has a viable 8 GB product path, regardless of the quality of its video conditioning.

## SHOULD-FIX

### 1. Factual corrections to fold into the SPEC

- **CONFIRMED — Count:** the live registry has 12 effective `image_to_video` engines, not six: `cloud_vidu_q2_pro_fast_720p`, `cloud_wan_i2v`, `fastwan_8gb`, `ltx25_foley_plus`, `ltx25_mime`, `ltx25_video`, `ltx_8gb`, `ltx_video`, `mesh_stage`, `minimax_h3_video`, `wan_ti2v`, and `word_razzle`.
- **CONFIRMED — Wording:** “12 declarations across eight files” is also imprecise; inheritance supplies the family to `fastwan_8gb` and two LTX 2.5 siblings. The meaningful verified number is 12 effective registered engines.
- **CONFIRMED — Existing-lane status:** `GhostSignalEngine` and `GhostSignalV3Engine` are unregistered/tombstoned bases; README calls `animatediff15_v3_haunted_video` the one surviving AnimateDiff lane. Saying the golden and clean-v3 “lanes” remain selectable siblings is false, although their classes remain as inheritance references (`eng_ghost_signal.py:285-290`; `eng_ghost_signal_official.py:67-73`; `README.md:359-362`).
- **CONFIRMED — Adapter dial:** Haunted strength is not frozen at 1.0. `OTR_GHOST_HAUNTED_LORA_STRENGTH` overrides the default (`eng_ghost_signal_official.py:109-116, 158-174`). No value has been qualified by eye in the code comment.
- **CONFIRMED — SparseCtrl superiority:** “strictly better” is not verified by local output. It is the purpose-built mechanism; comparative haunted-stack quality remains empirical.
- **CONFIRMED — Official limitations:** AnimateDiff documents small flicker and recommends that image-animation inputs come from the same community SD1.5 model used for animation. [Official AnimateDiff v3 limitations](https://github.com/guoyww/AnimateDiff/blob/main/README.md)
- **INFERRED — Domain risk:** a `z_image_turbo` still fed to the haunted SD1.5 checkpoint violates that upstream same-model recommendation. Cross-model subject/style drift may dominate even when SparseCtrl works correctly.
- **CONFIRMED — The SparseCtrl paper reports out-of-domain failure cases:** its training/data limits make the same-model/domain mismatch a real qualification item, not proof of failure here. [SparseCtrl paper](https://arxiv.org/abs/2311.16933)

### 2. Cheapest falsification experiment

- **CONFIRMED — The folded four-denoise experiment can falsify only Route A:** a bad latent-init sweep does not establish that purpose-built SparseCtrl is not worth building.
- **INFERRED — Cheapest whole-proposal falsifier:** in an isolated dependency environment, install current ACN 1.6.0 without changing ADE, use a private/unregistered probe path through the real `workflows/otr_canonical.json`, and render one fixed-seed, one-beat, 16-source-frame clip at the shipping 512×288 canvas from one representative Z-Image still. Do not add a selectable profile row for the probe.
- **INFERRED — Specific artifact:** inspect the canonical per-beat native MP4 named by the render ledger's `clips[].path`, plus a contact sheet containing the untouched source and frames 0/4/8/12/15. Store render assets directly under `otr/episodes/<probe-episode>/`; **UNVERIFIABLE — exact generated filename** until the ledger exists.
- **INFERRED — Reject the whole proposal if:** after correct phased unload, the 4060 native-aborts/OOMs; or the SparseCtrl clip matches frame 0 but loses subject/layout across later sampled frames, shows alternating/flicker/texture boil, or has no useful macro-motion. Those failures demonstrate that the correct mechanism cannot meet the hardware or visual contract.
- **INFERRED — Route-A-only probe if still desired:** run denoise 0.35/0.50/0.65/0.80 with the same source/prompt/seed/canvas and inspect a four-arm MP4/contact sheet. Reject Route A if low-denoise arms freeze or texture-boil while high-denoise arms lose subject/composition within 2–3 frames, leaving no overlap band.
- **CONFIRMED — Do not use the folded review's standalone probe script:** the repository's hard rule requires every API/headless/soak run to load `workflows/otr_canonical.json`.

### 3. Upstream baseline, not a product promise

- **CONFIRMED — Official example baseline:** v3 motion module + adapter scale 1.0 + RGB SparseCtrl + source index `[0]` + 25 steps + CFG 8.5 at 256×384 is an official example configuration, not a qualified OTR recipe. [Official v3 I2V YAML](https://github.com/guoyww/AnimateDiff/blob/main/configs/prompts/3_sparsectrl/3_1_sparsectrl_i2v.yaml)
- **CONFIRMED — ACN preprocessing contract:** its RGB preprocessor resizes/crops the image to the target latent canvas and VAE-encodes it (`nodes_sparsectrl.py:118-146` upstream). [ACN source](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/blob/main/adv_control/nodes_sparsectrl.py)
- **INFERRED — OTR qualification baseline:** start at one explicit index `0`, one 16-frame source window, the real 512×288 aspect, fixed seed, and no long-context expansion. Preserve Ghost's existing sampler recipe except where SparseCtrl requires a declared difference; record every difference in the recipe receipt.
- **UNVERIFIABLE — Best denoise, SparseCtrl strength, CFG, or adapter strength:** no current local render establishes these values. They must remain probe variables, not undocumented defaults.

### 4. Nonblocking integration hygiene

- **INFERRED — After qualification:** add the new profile to the manual GPU soak/bank sweep rosters (`scripts/otr_gpu_soak_matrix.py`, `scripts/otr_bank_engine_sweep.py`) so it does not disappear from routine hardware coverage.
- **CONFIRMED — Do not add estimate refusal:** use `admission_unenforced` until measurement; the only acceptable resource killer is actual OOM/native abort, while guards remain appropriate for missing pixels, nodes, or weights that would otherwise yield a silent wrong render.

