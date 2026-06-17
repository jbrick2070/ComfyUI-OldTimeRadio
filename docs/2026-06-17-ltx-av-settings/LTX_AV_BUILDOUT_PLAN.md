# LTX-AV path -- FULL BUILD-OUT PLAN (operator GO 2026-06-17)

**Decision (operator):** when the base workflow ships, **`music_visual` + `announcer_visual`
DEFAULT to the new sharp LTX-AV (audio-conditioned) path** -- the audio-reactive radio
b-roll that "wiggles with the music." The standalone smokes proved the recipe; the LANE
is not built yet (the engine still runs the old no-LoRA M0 recipe and `ltx_av` is
dark/non-default). This plan builds it out fully.

## What the smokes PROVED this session (grounded, docs/2026-06-17-ltx-av-settings/)
- The golden sharpness chain runs on the LTX-AV **A2V (audio-concat) graph WITHOUT
  crashing**: distilled **LoRA @0.70** + `euler_cfg_pp` + 8-step **ManualSigmas** +
  **cfg 1.0** + i2v strength 0.75, **NO `ModelSamplingLTXV`** (bypassed to avoid the
  double-shift the panel flagged).
- Sharpness jumps to the distilled-sharp level. Laplacian @832x480: **talk 149 /
  bookend 458 / music 119** vs the no-LoRA AV (**93.6** smooth / 166 grainy) and
  `sharp_c02` (144). All three subjects (face / bookend still / music scene) render
  sharp + audio-conditioned + muxed.
- VRAM ~15.1-15.3 GB at 832x480 on the resident base (over the 14.5 soft cap; a clean
  box / 512x288 is lower). Wall ~125-185 s/clip (the slow part is the CPU Gemma encode).
- **OPEN visual gate:** lip-sync *survival* on the TALK (face) lane -- operator eyeball
  of `p4_av_lora/lora_with_audio`. Music/announcer are audio-REACTIVE (no lip-sync
  needed), so they do NOT block on this.

## Build-out chunks (each: code + JSON-in-lockstep + regression suite + Bug Bible + audio byte-identical)

**C1 -- Sharp recipe in `eng_ltx_av.py`.**
Add a "sharp" graph mode to `_build_graph`: wire `LoraLoaderModelOnly` (distilled @0.70)
on the GGUF unet -> `CFGGuider.model` directly (DROP `ModelSamplingLTXV` in this mode --
the golden has none; ManualSigmas already carries the shift). Sampler chain =
`KSamplerSelect(euler_cfg_pp)` + ManualSigmas (the 9 `LTX_DISTILLED_SIGMAS`, via the
`_SigmasFromValues` injector pattern from `eng_ltx_video`) + `CFGGuider(cfg=1.0)`. Keep
the audio path (LTXVAudioVAEEncode -> LTXVConcatAVLatent -> sample -> LTXVSeparateAVLatent)
and i2v strength 0.75. Add the distilled LoRA to `_weight_paths()` + `assert_usable`
(license-clean: Apache GGUF + LTX-2 Community). Keep the old M0 recipe selectable
(`OTR_LTX_AV_SHARP` default ON for the shipped lane; OFF restores M0). GOLDEN
`eng_ltx_video.py` UNTOUCHED.

**C2 -- Make `ltx_av` the music/announcer default.**
Today `eng_ltx_video` owns `default_roles=(announcer_visual, music_visual)`. Switch the
defaults to the **`ltx_av_music`** lane (family `audio_conditioned_video` -- no face,
audio-reactive scene; the bookend/music smokes) for BOTH roles. Un-dark `ltx_av`
(default the enable for these roles; keep `OTR_ENABLE_LTX_AV` honored). Keep `ltx_video`
SELECTABLE as the non-audio alternative. **Update `otr_scifi_16gb_full.json` in the SAME
change** (OTR_VideoDirector announcer/music dropdowns -> ltx_av_music) + re-validate
(`OTR_WorkflowValidator` + JSON round-trip + link/widget audit).

**C3 -- Canvas / VRAM (DECISION NEEDED).**
`render_driver` clamps `ltx_av` to **512x288** (`OTR_LTX_AV_RENDER_CANVAS`) today. The
sharp smokes ran 832x480 (~15.2 GB, over the soft cap). DECIDE the ship canvas:
- **512x288** (VRAM-safe, ~13-14 GB; composite upscales to delivery anyway) -- RECOMMENDED
  default, verify the LoRA recipe peak on a CLEAN box first.
- **832x480** (sharper; ~15.2 GB, over the 14.5 soft cap but no OOM on 16 GB) -- opt-in.
Bonus: the slow CPU Gemma encode -- consider `device=default` (GPU) if the canvas leaves
headroom (the av-lane currently forces CPU).

**C4 -- no-fallbacks alignment.**
Under `547671d` (no fallbacks) `ltx_av` fails LOUD; the `fallback_engine` attrs are
no-ops. Confirm every music/announcer beat carries `audio_ref` (the lane requires it) so
the default never fails closed for lack of audio; if a beat has no audio, that's a LOUD
authoring error, not a silent degrade.

**C5 -- M4 GPU smoke + the lip-sync gate (talk only).**
music/announcer (audio_conditioned, no face) ship on the audio-reactive proof already in
hand. The **talk/character lane** still needs the operator lip-sync-survival eyeball
before `ltx_av_talk` becomes a CHARACTER default -- that stays opt-in until then.

**C6 -- full-episode soak.**
Render a real episode with ltx_av music+announcer defaults; gate = suite 4452/0 + Bug
Bible 16/7/3 + `test_audio_byte_identical` green + per-beat NVML <= ceiling at the chosen
canvas + workflow JSON git-clean & re-validated. Operator eyeballs 2-3 finals.

## Open decisions for the operator
1. **Ship canvas** for the AV default: 512x288 (safe, recommended) vs 832x480 (sharper, over soft cap).
2. Confirm the **audio_conditioned_video (no-face) lane** for announcer+music (the "radio reacts to audio" look), with `ltx_video` kept selectable.
3. Talk/character lane stays opt-in pending the lip-sync eyeball -- OK?

## Sequencing
C1 -> C2 -> C3 (decision) -> C4 -> C6 ship music/announcer; C5 (talk lip-sync) runs parallel/after.
One coder window in the code at a time; commit+push per green chunk on `v2.0-alpha`;
prod/main stays operator-gated.
