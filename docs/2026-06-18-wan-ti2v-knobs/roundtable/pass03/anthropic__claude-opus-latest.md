<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-4.8-opus-20260528 -->

VERDICT: yes-with-fixes -- the seven build items are individually sound, but three rest on unverified node schemas / a NVIDIA-only probe, and two are under-specified (frame clamp, VAE name) enough to break the floor or valid configs.

MUST-FIX BEFORE BUILD:
1. [Checklist #3] The "clamp the floor tier so target_frame_count can't push 33+" step has no mechanism. `render_clip` calls `quantize_frames_4n1(... max_frames=_TI2V_MAX_FRAMES)` with `_TI2V_MAX_FRAMES=177`. Just lowering MIN to 17 still lets any upstream `target_frame_count` (e.g. 81) render and blow the 8GB ceiling. Fix: introduce `OTR_WAN_TI2V_MAX_FRAMES` (default 17, i.e. floor==min==max) and pass it as `max_frames`; higher tiers raise the env. State this explicitly.

2. [Acceptance / Checklist #7 / render_clip] The FIT test demands a "measured peak < ceiling at 0.1s probe" on Mac MPS and AMD, but the measurement path is `VramPeakProbe` + `vram_used_mb()` (note the `or 0` guard) + `assert_peak_within_ceiling` + `dynamic_vram_ceiling_mb` -- all of which read like NVML/NVIDIA paths. verify: VramPeakProbe backend on MPS/ROCm. If they return 0/None off NVIDIA, the assert passes trivially and the Mac/AMD acceptance is not actually measured. Fix: either wire a torch.mps / ROCm allocated-memory source into the probe, or restate the acceptance so Mac/AMD validation is "renders + holds the still" (qualitative) and the numeric peak gate is NVIDIA-only.

3. [Checklist #1] "reconcile the clip inputs dict per loader" is required, not optional: the current `clip` inputs are `{"clip_name", "type":"wan", "device":"default"}`. `CLIPLoaderGGUF` very likely does not accept `device` (and may differ on `type`). Passing an unknown input will fail mid-render after weights load. Fix: branch the clip inputs dict on `_clip_loader_mode()` exactly like `unet_inputs` already branches; verify CLIPLoaderGGUF's required inputs from `/object_info` before wiring (you already list this in Verify-at-build -- make it gating).

4. [Checklist #6] Requiring the resolved VAE basename to equal the one approved name contradicts the "keep every knob env-overridable" goal and breaks a legitimately-renamed or GGUF Wan2.2 VAE. The existing guard (empty OR == 2.1) already closes the silent-corruption trap. Fix: instead of exact-equality, reject empty/2.1 and require a "2.2"/"wan2.2" version token in the basename -- catches the real failure (2.1 VAE) without banning valid 2.2 files.

SHOULD-FIX:
1. [Checklist #1] Adding a `CLIPLoaderGGUF` availability check to `assert_usable` conflicts with its own contract ("fail-closed BEFORE any forward; no heavy import") and is asymmetric: `UnetLoaderGGUF` (same ComfyUI-GGUF pack) is only resolved in `load()` via `resolve_graph_classes`, not in `assert_usable`. Either check both GGUF classes there or neither -- otherwise a missing pack still surfaces as a generic `load()` error for the UNET but a friendly one for CLIP. verify: whether node-registry lookup counts as the "heavy import" the docstring forbids.

2. [Checklist #1] Default CLIP basename is changed to the GGUF, but the safetensors fallback path is not given a default name. If an operator sets `OTR_WAN_TI2V_CLIP_LOADER=safetensors` and leaves `OTR_WAN_TI2V_CLIP_NAME` unset, they'll get the `.gguf` default fed to `CLIPLoader` -> mismatch. Mirror the UNET pattern: pick the default clip basename from `_clip_loader_mode()`.

3. [Checklist #2] Wan2.2 VAE is a *video* VAE; `VAEDecodeTiled` for video models exposes `temporal_size`/`temporal_overlap`, not just `tile_size`/`overlap`. The plan only mentions spatial tile/overlap. verify: VAEDecodeTiled required inputs from `/object_info`; if temporal inputs are required, omitting them errors or silently mis-tiles. Also confirm IMAGE is output index 0 (the code reads `results["vaedecode"][0]`).

4. [Checklist #4] The doc's portability claim is "euler / simple", but only the sampler is whitelisted; `OTR_WAN_TI2V_SCHEDULER` (default "simple") is left unconstrained, so an operator can set a non-portable scheduler and defeat the floor's cross-platform guarantee. Add scheduler to the same whitelist/resolver.

5. [Sequencing] Item #5 (shared config resolver providing the canonical default) must land before/with item #4, since #4 says `assert_usable` validation must read "the SAME default" -- implement the resolver first so the default lives in one place and unset-passes is guaranteed.

OPTIONAL / NICE-TO-HAVE:
- The existing `or self.target_fps` (using 25 fps as a frame count) is a latent bug; item #3's switch to `_TI2V_DEFAULT_FRAMES` fixes it -- call that out so the fix isn't reverted.
- A one-line preflight VRAM estimate (already endorsed in CUT) would make the "fail-closed" floor friendlier than discovering the ceiling mid-render.

CUT THESE (over-engineering):
1. [Checklist #6, as written] The exact-name allowlist -- see MUST-FIX #4; the stricter form adds brittleness without closing a new failure mode beyond the version-token check.
2. The plan's own CUT list (LoRA wiring, OOM catch-retry loop, 33-frame expected-OOM baseline) is correctly cut -- no objection; keep them out.

[ASSUMPTION] CLIPLoaderGGUF rejecting `device` and VAEDecodeTiled needing temporal inputs are inferred from typical ComfyUI-GGUF / core node schemas; both are flagged verify because the node input dicts are not in the grounding.
[ASSUMPTION] VramPeakProbe/vram_used_mb/dynamic_vram_ceiling_mb are NVML-backed is inferred from the `or 0` guard and the "mid-render NVML ceiling" wording in render_clip; the implementations of these `_MC` helpers were not shown.