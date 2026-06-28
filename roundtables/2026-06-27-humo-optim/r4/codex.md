VERDICT: yes-with-fixes. The bakeoff shape is right, but the mid-graph reclaim node contract is still unsafe/ambiguous enough to measure the wrong thing or detach the sampler model.

MUST-FIX BEFORE BUILD:
1. [r3 §1 / r4 Q1-Q2] `OTR_BakeoffReclaim` cannot call `wrapper_bridge.reclaim_idle_models()` as a blind side effect. That helper detaches every entry in `comfy.model_management.current_loaded_models`, not just encoders: `nodes/_otr_video_engines/wrapper_bridge.py:248-301`; Comfy tracks all loaded models globally at `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\model_management.py:610,849-945`. Concrete fix: make the bakeoff node pass through `model`, `positive`, `negative`, and `latent_image`, and either skip detaching the object identities reachable from the sampler model/VAE or implement an encoder-only reclaim. Add a runtime assert/log that the sampler model object survived or was reloaded intentionally before `KSampler`.

2. [r3 §1 / r4 Q1] The plan still does not define the side-effect node cache contract. Comfy treats nodes without `IS_CHANGED` as unchanged (`execution.py:73-81`) and executes only output-reachable nodes (`execution.py:1151-1156`). Concrete fix: `OTR_BakeoffReclaim` must implement `IS_CHANGED` as always-dirty, emit a unique log/manifest marker per prompt, and route all KSampler inputs through its outputs so `KSampler` cannot run from cached upstream values. KSampler inputs are exactly `model`, `positive`, `negative`, `latent_image`: `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\nodes.py:1557-1577`.

3. [r3 §1] The helper-node install path is ambiguous from the repo cwd. `custom_nodes/otr_bakeoff_helper/__init__.py` in the plan could mean a nested folder under `ComfyUI-OldTimeRadio`, but ComfyUI loads sibling custom-node packages under `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\`; existing sibling packages are listed there, and there is no repo-local `custom_nodes` directory. Concrete fix: spell the absolute path as `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\otr_bakeoff_helper\__init__.py`, and keep it out of the OTR pack `__init__.py`.

4. [r3 §2 / r4 Q3] The sentinel says “LTX-AV+Whisper,” but LTX-AV does not use Whisper in this code path; its graph candidates are GGUF UNET, Gemma text encoder, LTX audio/video VAEs, and `LTXVAudioVAEEncode`: `nodes/_otr_video_engines/eng_ltx_av.py:454-465`. HuMo is the Whisper user: `nodes/_otr_video_engines/eng_humo.py:247-252`. Concrete fix: define the sentinel as “run one real `ltx_audio_in` render in the same server session, assert the LTX-AV classes/model names actually ran, then run HuMo candidate.” Do not claim Whisper residency unless a prior HuMo/Whisper load is deliberately part of that sentinel.

SHOULD-FIX:
1. [r3 §1] Reusing `HuMoEngine._build_graph` needs a precise conversion contract. `_build_graph` emits `wrapper_bridge.run_graph` specs with `"class"` aliases and `Wire` objects (`eng_humo.py:215-283`), while `/prompt` expects `class_type` and list links, as shown in `scripts/render_humo_batch.py:568-665`. Concrete fix: add a dry-validate assertion that every emitted bakeoff node has valid `class_type`, inputs, and an output `SaveImage`.

2. [r3 §4] `SaveImage` is a valid output node (`nodes.py:1620-1646`), but the plan should assert the harness reads exactly those PNG frames and encodes them with `wrapper_bridge.encode_frames_to_silent_mp4` (`wrapper_bridge.py:564-585`). Add `ffprobe` verification: no audio stream, expected frame count, output under `otr/episodes/_bakeoff_humo/`.

OPTIONAL / NICE-TO-HAVE:
Keep the side-by-side clips and blue-cast delta. Face/lip metrics can remain soft-gated, but they should not affect the gate.

CUT THESE:
1. [r3 §3] Cut dlib/mediapipe lip metrics from the first locked build if they add setup time. The plan already has operator clips plus pure PIL/numpy blue-cast delta, and face metrics are explicitly non-gating.
2. [r3 §1] Cut any attempt to round-trip conditioning across two `/prompt` calls. r3 already rejected it; the helper node is the only viable path in this plan.

VERIFY-AT-BUILD checklist:
1. Reclaim node executes once per candidate prompt, before KSampler, with no cache skip.
2. Reclaim does not detach or unpatch the sampler model/LoRA/VAE in a way that changes the intended graph.
3. Manifest asserts actual unet, LoRA present/absent, shift=8, steps, cfg, seed, dims, 4n+1 length, terminal, output path, and engine id.
4. Sentinel first runs a real `ltx_audio_in` render in the same server session, then HuMo candidate, no reboot.
5. Boot uses no `FLOOR`; reset confirms port 8000 empty and VRAM back to desktop baseline.
6. Output MP4 exists in `otr/episodes/_bakeoff_humo/`, is silent, and has the expected frame count.