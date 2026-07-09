# Model-Slot Audit - 2026-07-09

Scope: production slots exposed by `workflows/otr_canonical.json` and the
registered engine namespaces. Here, "local" means the engine runs through local
code/assets rather than a provider API; the dropdown label does not need to say
`LOCAL`. Cloud/direct-API engines stay documented in
`docs/CLOUD_ENGINE_COVERAGE.md`; this note is the compact on-box matrix for the
release-order gate in `docs/GO_FORWARD_PLAN.md`.

## Decision

The kept on-box canonical path is:

| Slot | Canonical engine/model | Inputs | Output | Model assets / VRAM class | Evidence |
| --- | --- | --- | --- | --- | --- |
| writer creative LLM | `unsloth/gemma-4-12b-it-GGUF [LOCAL GGUF]` | prompt/messages | story brief + ledger JSON | Gemma 4 12B Q8 GGUF, 16 GB PASS row in catalog | `tests/test_workflow_canonical_baseline.py`, canonical API dry-run tests |
| writer technical LLM | `mistralai/Mistral-Nemo-Instruct-2407` | prompt/messages | repair/validation text | HF gated Mistral Nemo, curated 16 GB PASS | `tests/test_workflow_canonical_baseline.py`, catalog tests |
| character TTS | `indextts2` | dialogue text + reference clip | per-line voice clips, 22050 Hz | sidecar/OOP venv, non-commercial warning | `tests/test_audio_engine_adapters.py`, `tests/test_voice_bank.py` |
| announcer TTS | `kokoro` | announcer text + preset voice | announcer voice clips, 24000 Hz | CPU/in-graph, Apache-2.0 | `tests/test_audio_engine_adapters.py` |
| music | `stable_audio_3` | OTR music prompt | open/close/interstitial music cues | local Stable Audio 3, GPU | `tests/test_audio_engine_adapters.py`, `tests/test_sa3_music_prompt_bug408.py` |
| still image | `z_image_turbo` for announcer/music/character stills | text prompt | PNG stills in episode tree | local Z-Image Turbo, GPU | `tests/test_workflow_live_passes_validator.py`, `tests/test_image_engine_c2.py` |
| video: announcer | `viz_mxc_cpu` | optional audio/prompt | silent MP4 clip | CPU/PIL/numpy/ffmpeg, no model weights | `tests/test_workflow_live_passes_validator.py`, `tests/test_video_viz_rainbow.py` |
| video: music | `viz_mxc_mandala` | optional audio/prompt | silent MP4 clip | CPU/pycairo/ffmpeg, no model weights | `tests/test_workflow_live_passes_validator.py`, `tests/test_video_viz_mandala.py` |
| video: character | `viz_camera` | optional audio/prompt | silent MP4 clip | CPU/PIL/numpy/ffmpeg, no model weights | `tests/test_workflow_live_passes_validator.py`, `tests/test_video_viz_camera.py` |

The saved canonical workflow is the lean 30-word local smoke canvas. Heavier
local video lanes are opt-in profile/manual selections, not saved-workflow
defaults.

## Kept Selectable Local Candidates

| Namespace | Candidates | Status |
| --- | --- | --- |
| audio/TTS/music | `bark`, `kokoro`, `indextts2`, `chatterbox`, `dia`, `musicgen`, `stable_audio_music`, `stable_audio_3` | Registered and selectable. Unsupported role picks fail loud through `EngineUnusable`; deeper model/token checks run on dispatch/load. |
| still image | `flux_gen1`, `flux2_klein`, `lumina_image`, `qwen_image`, `z_image_turbo` | Registered and selectable. Missing weight files fail loud in adapter `assert_usable`; no silent Flux fallback. |
| video | `still_motion`, `viz_green`, `viz_mxc_cpu`, `viz_mxc_mandala`, `viz_camera`, `still_flat`, `still_pan`, `still_word`, `humo`, `humo_1.7B`, `humo_1.7B_169`, `humo_14B_169`, `ltx_video`, `mesh_stage`, `wan_i2v`, `wan_ti2v`, `ltx_audio_in` | Registered and selectable. Runtime failures are hard failures; every registered adapter declares `fallback_engine = None`. |

## Pre-Smoke Readiness Inspection

No live GPU/provider smoke should run until the input/output contract is known.
Inspection of the requested candidates says:

| Candidate | Input contract | Output contract | Readiness call |
| --- | --- | --- | --- |
| `chatterbox` | Per-line text + reference WAV + delivery vector/seed; roles `char_voice`, `announcer_voice`. | Main-venv `AUDIO` dict from worker WAV, 24000 Hz. | Ready for local sidecar preflight and one-line voice smoke if the isolated Chatterbox venv, worker, and CC0 ref bank exist. Missing pieces fail loud. |
| `dia` | Per-line text + reference WAV + optional ref transcript + seed; role `char_voice` only. | Main-venv `AUDIO` dict from worker WAV, 44100 Hz. | Ready for local char-voice sidecar preflight. Not an announcer test yet; no Dia announcer role is registered. |
| `qwen_image` | Text prompt + seed/dims; graph needs Qwen GGUF diffusion, Qwen CLIP, Qwen VAE. | Decoded RGB still frame returned to the dispatcher. | Graph-ready, but live smoke should wait until the CLIP/VAE files are preflighted alongside the GGUF path; current `assert_usable` proves only the diffusion GGUF before render. |
| `wan_ti2v` | Init image + prompt/seed/canvas; graph needs Wan2.2 TI2V-5B UNET, umt5 CLIP, approved `wan2.2_vae.safetensors`. | Silent bt709 MP4 clip dict after ffprobe contract validation. | Best first local Wan test. The adapter already fail-closes on UNET, CLIP, VAE, VAE basename, sampler, and scheduler before forward. |
| `wan_i2v` | Init image + prompt/seed/canvas; graph needs Wan I2V UNET, umt5 CLIP, VAE. | Silent bt709 MP4 clip dict after ffprobe contract validation. | Structurally ready but heavier and not the first Wan smoke. Run after `wan_ti2v` unless there is a specific 14B quality target. |

Requested Comfy Cloud candidates are structurally ready for provider smokes
after auth/credits are confirmed:

| Candidate | Input contract | Output contract | Readiness call |
| --- | --- | --- | --- |
| `cloud_nano_banana_2` | Text prompt + seed; DynamicCombo model dict includes model/resolution/aspect/thinking level. | Canonical PNG path. | Ready for live provider still smoke. Existing tests reject stale selectors. |
| `cloud_seedream_2` | Text prompt + seed; model dict includes model/size preset/max images. | Canonical PNG path. | Ready for live provider still smoke. |
| `cloud_krea_2_turbo` | Text prompt + seed; model dict includes aspect/resolution/creativity. | Canonical PNG path. | Ready for live provider still smoke. |
| `cloud_luma_photon_flash` | Text prompt + seed + aspect/style-weight fields. | Canonical PNG path. | Ready for cheapest Luma still smoke. |
| `cloud_vidu_q2_pro_fast_720p` | Init image + text prompt; fixed `viduq2-pro-fast`, `720p`, no audio input. | Canonical silent MP4 clip dict; provider audio stripped if present. | Ready cheap video smoke candidate. |
| `cloud_vidu_q2_pro_fast_720p_sfx` | Init image + text prompt; same Vidu row with SFX-safe prompt. | Canonical silent MP4 clip dict plus extracted `sfx_stem_path`; fails loud if provider returns no audio. | Ready only if the goal is to test provider SFX extraction; otherwise test the mute-only Vidu row first. |

Recommended live-smoke order after the selective headless reset:

1. Local sidecar voice preflights: `chatterbox` one char line, `chatterbox`
   announcer line, `dia` one char line.
2. Cloud stills: `cloud_luma_photon_flash`, `cloud_krea_2_turbo`,
   `cloud_seedream_2`, then `cloud_nano_banana_2`.
3. Cheap cloud video: `cloud_vidu_q2_pro_fast_720p`.
4. Local heavy visual lanes: `wan_ti2v`, then `qwen_image` after the CLIP/VAE
   preflight is strengthened, then `wan_i2v` if the 14B target is still wanted.

## Retired / Non-Invocable

| Engine | Namespace | Disposition |
| --- | --- | --- |
| `hidream_i1`, `sd35_large` | image | Source files remain, but they are unregistered dark scaffolds and have no capability row. |
| `still_parallax` | video | Source file remains, but the engine is unregistered; the old fallback chain was removed. |
| `triposr`, `triposg_talk`, `hunyuan3d_talk`, `trellis_talk` | video | Source files remain, but dark 3D scaffolds are unregistered until a real forward ships. |

Unknown/retired engine ids must fail as malformed config. Incompatible
registered picks must fail as incompatible profile. There is no silent fallback
or tested-only dropdown gate.

## Smoke Path

Tiny canonical smoke command:

```text
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_canonical_api_run.py --offline-schemas --dry-run --profile none --words 30 --source-bank science_news --dump-prompt scripts\_otr_canonical_api_prompt.json
```

This loads `workflows/otr_canonical.json`, applies no hidden engine patch, and
converts the saved graph through the API prompt builder. A live render smoke
should use the same script without `--dry-run` after the required selective
headless reset.
