<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The single-adapter design breaks the fallback chain and role compatibility.

MUST-FIX BEFORE BUILD:
1. [Draft design / Fallback chain (Q8)] Single adapter breaks fallback. `humo` strictly requires `init_image` (`eng_humo.py` line 223). If a single `ltx_av` adapter runs `music_visual` (which may lack `init_image`) and falls back to `humo`, `humo` will raise a `GraphExecutionError` and double-fault.
   Fix: Split `eng_ltx_av.py` into TWO adapters. `LtxAvI2VEngine` (family `audio_driven_face`, roles `announcer/character`, requires `audio_ref`, `init_image`, fallback `humo`). `LtxAvT2VEngine` (family `audio_conditioned_video`, role `music_visual`, requires `text_prompt`, `audio_ref`, fallback `ltx_video`).
2. [Repo grounding / role_compat] `music_visual` fails closed on `audio_ref`. `role_compat.py` defines `ROLE_AVAILABLE_INPUTS` for `MUSIC_VISUAL` as `{"text_prompt", "init_image", "base_clip_ref"}`. An engine requiring `audio_ref` will be silently excluded by `engine_fits_role`.
   Fix: Add `"audio_ref"` to the `frozenset` for `Role.MUSIC_VISUAL.value` in `role_compat.py`.
3. [Registry/schemas touch list] Schema sync assertion failure. `schemas.py` enforces `assert set(FAMILIES) == set(FAMILY_REQUIRED_INPUTS)`. Adding a new family to the docstring/tuple without updating the dict will crash the app at import time.
   Fix: In `schemas.py`, add `"audio_conditioned_video"` to the `FAMILIES` tuple AND add `"audio_conditioned_video": ("text_prompt", "audio_ref")` to the `FAMILY_REQUIRED_INPUTS` dict.
4. [Milestones / M0 PROBE] Missing explicit STOP rule for isolation (Q4).
   Fix: The explicit STOP rule is: If the ComfyUI-native LTX-2.3 audio nodes require pip-installing `torchaudio`, `librosa`, or any audio-processing package into the main ComfyUI cu130 venv, it violates the V-12 cold-import invariant and MUST be forced to `ISOLATION_SIDECAR_REQUIRED`.

SHOULD-FIX:
1. [Fallback chain (Q8)] Mid-chain aspect change. `LtxAvI2VEngine` (landscape 1472x832) falling back to `humo` (portrait 480x832) causes a severe visual pillarboxing shift mid-episode.
   Fix: Route `LtxAvI2VEngine` fallback directly to `still_kenburns` if landscape consistency is preferred, or explicitly document the aspect shift as an acceptable loud degrade.

OPTIONAL / NICE-TO-HAVE:
- Q6 two-stage: Run base-only for V1 to ensure it fits the 14.5GB NVML ceiling and meets the episode time budget.
- Q3 init_image: By splitting into two adapters (Must-Fix 1), `init_image` is naturally hard-required for the I2V roles and omitted for the T2V role, cleanly solving Q3.

CUT THESE (over-engineering):
1. [The zero-new-model comparison lane] Yvann-Nodes audio-feature scheduling. Safe to cut because it introduces a new custom-node dependency (b7 sweep risk) and does not solve the lip-sync requirement (announcer/character), which is the primary goal of the LTX-2.3 A2V integration.

[ASSUMPTION] I am assuming the LTX-2.3 ComfyUI nodes for audio conditioning do not inherently force a joint audio/video VAE decode that cannot be split, and that the audio latent can simply be dropped before `VAEDecode` as proposed.