# Fact-Checked Analysis: One Master JSON With User-Controlled Path Switches

Date: 2026-07-06

Scope: improve the earlier cross-platform path-switch analysis around the actual product goal: keep one master ComfyUI workflow JSON if possible, and let the user change visible dropdowns/switches so the same graph can run on different machine classes. This is a documentation-only update. It does not change code, workflow JSON, profile JSON, or tests.

Related doc: `docs/2026-07-06-hardware-portability-analysis.md` covers the delivery/upscale switch. This file covers the broader runtime path switches: LLM, TTS, music, image, video, delivery, and compatibility validation.

## Executive Take

The desired UX is not hidden profile application and not "generate many behind-the-scenes JSONs." The better target is:

- one master workflow graph
- visible user controls inside that graph
- a clear `machine_target` or `compatibility_preset` dropdown if useful
- per-lane override dropdowns for LLM, speech, music, image, video, and delivery
- fail-closed validation when the selected combination cannot run
- no silent engine swaps

A "profile" can still exist as a user-visible preset concept, but it should behave like a dropdown the user controls, not a hidden layer that mutates the graph offstage. The saved JSON remains the truth.

## Corrected Product Model

Do this:

- Keep `workflows/otr_scifi_16gb_full.json` as the master graph.
- Add or consolidate visible compatibility switches in the graph.
- Let one dropdown pick a broad target such as `nvidia_cuda_full`, `nvidia_cuda_lite`, `mac_safe`, `amd_safe`, or `cpu_cloud`.
- Let advanced users override individual lanes with existing or new dropdowns.
- Store all selected values as normal `widgets_values` in the same JSON.
- Validate selected values loudly at queue time.

Do not make this the main UX:

- hidden profile files that silently patch widgets
- automatic host detection that rewrites user choices
- separate generated JSONs as the only way to run on another machine
- fallback behavior that swaps engines without telling the user

Optional export copies are fine for convenience, but the architecture should not require them.

## Current Repo Ground Truth

The live workflow already has many of the manual controls needed for a one-master-JSON model:

- `OTR_LedgerScriptWriter` has LLM model widgets: `creative_writing_model`, `technical_model`, OpenRouter slot widgets, Comfy Credits slot widgets.
- `OTR_BatchCharacterVoices` has a character voice engine widget.
- `OTR_AnnouncerVoice` has an announcer voice engine widget.
- `OTR_StableAudioTheme` has a music engine widget.
- `OTR_VideoDirector` has per-role video and image engine widgets.
- `OTR_ImageDirector` has image granularity/seed controls.
- `OTR_VideoRenderBatch` has a batch video engine widget.
- `OTR_SignalLostVideo`, `OTR_SilentComposite`, `OTR_SceneAwareScopes`, `OTR_CaptionBurn`, and `OTR_MasterAudioMux` already expose some resolution/ffmpeg/caption controls.

The missing pieces are not "more hidden profiles." The missing pieces are:

- one visible compatibility preset/switchboard
- one visible delivery/upscale backend switch
- explicit non-CUDA-safe lane choices
- validation that explains impossible combinations
- MPS/AMD backend semantics if those become real local targets

## Existing Profile Layer: Useful Internally, Wrong As The Main UX

The repo does have `config/profiles/*.json` and `nodes/_otr_workflow_apply.py`, but that layer is narrower than the requested product:

- Only three committed profiles exist: `16gb_full`, `8gb_lite`, and `cpu_floor`.
- There are no committed Mac or AMD profiles.
- `apply_profile()` patches managed engine/feature/seed widgets, but deliberately never stamps `OTR_WorkflowValidator`.
- `config/profiles/widget_mapping.json` explicitly exempts `OTR_LedgerScriptWriter`, so LLM widgets are not profile-managed.
- The live workflow's validator `profile_id` widget is currently empty.
- `device_backend: "mps"` is deliberately rejected by tests today.

So: profile files can be a backing data source for a visible preset dropdown later, but they should not be treated as the user-facing solution.

## Best One-Master Design

Add a visible switchboard concept to the workflow. It could be a new node later, or an expansion of an existing validator/director node, but the product surface should be simple:

### Primary User Control

`machine_target`

Suggested values:

- `nvidia_cuda_full`
- `nvidia_cuda_lite`
- `mac_safe`
- `amd_safe`
- `cpu_cloud`
- `custom`

This is not automatic host mutation. It is the user's declared target.

### Per-Lane Controls

The preset should either expose or feed these visible lanes:

- `llm_path`: local CUDA transformers, local Ollama HTTP, OpenRouter, Comfy Credits
- `character_voice_engine`: `indextts2`, `bark`, `kokoro`, `elevenlabs`
- `announcer_voice_engine`: `kokoro`, `elevenlabs`, `chatterbox`
- `music_engine`: `stable_audio_3`, `musicgen`, `sonilo`
- `image_engine_family`: local GPU stills, cloud stills
- `video_engine_family`: local motion, procgen/still, cloud video
- `delivery_scaler`: none, ffmpeg, NVIDIA RTX VSR, cloud upscale
- `validation_policy`: fail closed, or warn-only for explicitly experimental lanes

The important bit: every lane remains visible and user-owned. A preset may populate recommended defaults, but the final selected values must be inspectable in the workflow.

## Current Device Backend Reality

The profile schema currently allows only:

- `device_backend: "cuda"`
- `device_backend: "cpu"`

It does not model:

- `mps`
- `rocm`
- `directml`
- `amd`
- `linux`

The current availability logic only asks whether an engine is CPU-ok when the profile is `cpu`. It does not distinguish NVIDIA CUDA from AMD ROCm or DirectML. That is why the one-master UI should present conservative machine targets and validate selected engines, rather than pretending every GPU backend is interchangeable.

## LLM Switches

Current facts:

- The live workflow defaults both writer LLM widgets to `mistralai/Mistral-Nemo-Instruct-2407`.
- Mistral-Nemo uses the local in-process transformers loader and should be treated as the CUDA/full path.
- `google/gemma-4-12b-it` is a curated `ollama_local_http` row. It goes through local Ollama over HTTP and avoids the in-process CUDA-biased loader.
- OpenRouter and Comfy Credits are HTTP/cloud lanes when enabled.
- `request_slot()` routes `openrouter_http`, `comfy_credits_http`, and `ollama_local_http` before local model download/load/VRAM behavior.

Current risk:

- `load_llm()` defaults to `device="cuda"` and contains CUDA-specific cleanup, backend flags, device capability checks, memory budgeting, and bitsandbytes quantization assumptions.

Recommended one-master switch behavior:

- `nvidia_cuda_full`: local Mistral-Nemo is allowed.
- `nvidia_cuda_lite`: tiny local CUDA model or HTTP/cloud.
- `mac_safe`: Ollama HTTP or cloud.
- `amd_safe`: Ollama HTTP or cloud unless a specific AMD local path is proven.
- `cpu_cloud`: Ollama HTTP or cloud.
- `custom`: user may pick anything, but validator should fail loudly if CUDA-only local transformers are selected without CUDA.

## Audio Switches

Current capability facts:

- CPU-ok voice engines: `bark`, `kokoro`, `elevenlabs`
- GPU/sidecar voice engines: `indextts2`, `chatterbox`, `dia`
- CPU-ok music engines: `musicgen`, `sonilo`
- GPU-only music engines: `stable_audio_music`, `stable_audio_3`

Repo-grounded behavior:

- `kokoro` serves both character and announcer voice; it selects CUDA, then MPS, then CPU.
- `bark` serves character voice; its loader currently detects CUDA or CPU, not MPS.
- `elevenlabs` serves character and announcer voice as a cloud per-line adapter.
- `sonilo` serves music as a cloud adapter.
- `musicgen` uses `facebook/musicgen-small` and selects CUDA or CPU.
- `stable_audio_3` is the 16 GB music default but is not CPU-ok in the capability table.

Recommended one-master switch behavior:

- `nvidia_cuda_full`: `indextts2` character, `kokoro` announcer, `stable_audio_3` music.
- `nvidia_cuda_lite`: `kokoro` or `bark` voice, `musicgen` or proven `stable_audio_3`.
- `mac_safe`: `kokoro` or `elevenlabs` voice, `sonilo` or `musicgen` music.
- `amd_safe`: same as `mac_safe` until local AMD audio is proven.
- `cpu_cloud`: `kokoro`/`elevenlabs`, `sonilo`/`musicgen`.

## Image Switches

Current capability facts:

- GPU-only local still engines: `flux_gen1`, `flux2_klein`, `lumina_image`, `qwen_image`, `z_image_turbo`
- CPU/cloud still engines: `cloud_flux_pro`, `cloud_nano_banana_2`, `cloud_seedream_2`, `ideo`

Recommended one-master switch behavior:

- `nvidia_cuda_full`: `flux_gen1` or another proven local GPU still engine.
- `nvidia_cuda_lite`: `z_image_turbo` only if proven on the target card; otherwise cloud stills.
- `mac_safe`: cloud stills.
- `amd_safe`: cloud stills unless a specific local AMD still path is proven.
- `cpu_cloud`: cloud stills.
- `custom`: allow any registered image engine, but validate loudly.

Do not document `z_image_turbo` as Mac/AMD-safe until it has real host proof.

## Video Switches

Current CPU/procgen visual engines include:

- `still_motion`
- `still_flat`
- `still_pan`
- `still_word`
- `viz_green`
- `viz_mxc_cpu`
- `viz_mxc_mandala`
- `viz_camera`

Current cloud video engines include:

- `cloud_kling_avatar`
- `cloud_kling_lipsync`
- `cloud_seedance_2`
- `cloud_wan_i2v`
- `word_razzle`

Current local GPU-heavy video engines include:

- `humo`
- `humo_1.7B`
- `humo_1.7B_169`
- `humo_14B_169`
- `ltx_video`
- `ltx_audio_in`
- `wan_i2v`
- `wan_ti2v`
- `mesh_stage`

Recommended one-master switch behavior:

- `nvidia_cuda_full`: current procgen plus HuMo split is acceptable; Wan/LTX only when the asset stack is proven.
- `nvidia_cuda_lite`: procgen/still video.
- `mac_safe`: procgen/still/cloud video.
- `amd_safe`: procgen/still/cloud video until local AMD motion is proven.
- `cpu_cloud`: procgen/still/cloud video.
- `custom`: allow registered engines, but validate loudly.

## Delivery/Upscale Switch

The sibling hardware-portability doc is right that delivery scaling needs a first-class visible switch. For a one-master JSON, this should be a normal workflow control, not a profile-only field.

Suggested control:

- `delivery_scaler`: `none`, `ffmpeg_lanczos`, `ffmpeg_bicubic`, `nvidia_rtx_vsr`, `cloud_upscale`
- `target_resolution`: `source`, `1280x720`, `1920x1080`, `2560x1440`, `3840x2160`
- `delivery_fail_policy`: `fail_closed`, `copy_source_with_warning`

Rules:

- `nvidia_rtx_vsr` must fail closed when `nvvfx`/RTX support is absent.
- Mac/AMD/CPU presets should default to ffmpeg or none.
- Captions should burn after final scaling if the delivery design wants final-resolution text.
- Audio passthrough should stay sacred.

## Validation Behavior

Validation should answer: "Can this visible selection run here?"

Good validation:

- "Machine target is `mac_safe`, but `creative_writing_model` selects a CUDA-only transformers model."
- "Machine target is `cpu_cloud`, but `music_engine` is `stable_audio_3`, which is not CPU-ok."
- "Machine target is `amd_safe`, but `image_engine` is `flux_gen1`; this is not proven for AMD in this repo."
- "Delivery scaler is `nvidia_rtx_vsr`, but RTX VSR support was not detected."

Bad validation:

- silently changing `flux_gen1` to a cloud engine
- silently changing Bark to Kokoro
- treating `cuda` as a generic "some GPU exists" backend
- hiding the actual engine choices outside the workflow UI

## Implementation Direction For Later

No code is changed by this doc, but the proper future implementation is likely one of these:

1. Add a visible `OTR_CompatibilitySwitchboard` node that outputs a policy JSON consumed by downstream nodes.
2. Extend `OTR_WorkflowValidator` into a visible compatibility validator/preset node, but do not let it silently rewrite anything.
3. Keep existing per-node engine widgets, add a single `machine_target` widget, and validate all widgets against that target.

The cleanest user experience is option 1 or 3: one graph, visible switches, no hidden patching. Existing `config/profiles/*.json` can still be reused as preset data if helpful, but the user must see and control the selected preset.

## Bottom Line

The target should be one master JSON with visible switches:

- `machine_target`
- LLM path
- voice engines
- music engine
- image engine family
- video engine family
- delivery scaler
- validation policy

Profiles are not the product unless they are surfaced as a user-controlled dropdown. The saved workflow JSON should remain the source of truth, and changing visible dropdowns/switches should be enough to run the same graph on NVIDIA full, NVIDIA lite, Mac-safe, AMD-safe, or CPU/cloud paths.
