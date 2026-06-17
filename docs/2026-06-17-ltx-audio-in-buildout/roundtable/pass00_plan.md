# LTX (Audio In) -- full build-out SPRINT PLAN (for roundtable hardening before coding)

**Goal:** ship a testable **full LTX Audio-In workflow**: `music_visual` + `announcer_visual`
DEFAULT to the sharp LTX audio-conditioned lane (`ltx_av_music`), engine + workflow JSON
wired in lockstep, regression-green, committed + pushed to `v2.0-alpha`.

**HARD CONSTRAINT -- NO FALLBACKS (operator; `547671d`):** the render path makes a SINGLE
attempt and RAISES on failure -- NO engine swap, NO still floor, NO degrade. Therefore
(a) the `SYNTH_FALLBACKS` entries for ltx_av (`talk->humo`, `music->ltx_video`) are DEAD
-- remove/neutralize them, never rely on them; (b) making ltx_av the music/announcer
DEFAULT means a failed ltx_av beat fails the EPISODE LOUD -- so announcer + music beats
MUST always carry `audio_ref` (they are audio beats, so they do) and `assert_usable` must
pass up front (weights + nodes present); (c) `OTR_LTX_AV_SHARP` OFF is a CONFIG MODE chosen
at graph-build (the M0 recipe), NOT a runtime fallback. Nothing in this plan adds an
in-render degrade.

**Proven this session (GPU smokes, docs/2026-06-17-ltx-av-settings/):** the sharp chain
runs on the A2V (audio-concat) graph WITHOUT crashing and renders sharp on talk/bookend/
music: distilled **LoRA @0.70** + `euler_cfg_pp` + 8-step **ManualSigmas** + **cfg 1.0**
+ i2v strength 0.75, **NO `ModelSamplingLTXV`** (bypass; the golden has none, ManualSigmas
carries the shift). Canvas A/B: **512x288 and 832x480 peak the SAME ~15.2-15.4 GB**
(model/LoRA-bound, not canvas) -> ship **832x480** (sharper, no VRAM cost). Both ~0.7 GB
over the 14.5 soft cap, no OOM on 16 GB. Per-beat ~89 s warm (CPU Gemma encode dominates).

## Reference: the SHARP recipe already lives in `eng_ltx_video.py`
`eng_ltx_video` (the non-audio LTX) already implements the exact pattern to mirror:
`_sampler_mode()` (distilled/ksampler), `_SigmasFromValues` injector for ManualSigmas,
`_distilled_lora_file()`, LoRA wired `unet -> LoraLoaderModelOnly -> CFGGuider.model`
(NO ModelSamplingLTXV), `euler_cfg_pp`, cfg 1.0, LoRA in `_weight_paths`/`assert_usable`.
**The audio-in build reuses this pattern, ADDING the audio path** (LoadAudio ->
LTXVAudioVAEEncode -> LTXVConcatAVLatent -> ... -> LTXVSeparateAVLatent).

## Grounded wiring touchpoints
- `eng_ltx_av.py`: `_build_graph` is a dual-branch A2V graph (talk=i2v LTXVImgToVideo;
  music=EmptyLTXVLatentVideo) + ModelSamplingLTXV + LTXVScheduler + euler + cfg 3.0,
  NO LoRA. `requires_flag="OTR_ENABLE_LTX_AV"` default OFF (dark). `default_roles=()`.
- `registry.py`: CAPABILITIES rows for ltx_av_talk/ltx_av_music exist (heavy/14000).
  `default_engine_for_role` reads each engine's `default_roles`.
- `render_driver.py`: `ENGINE_FAMILY[ltx_av_music]=audio_conditioned_video`;
  `SYNTH_FALLBACKS[ltx_av_music]=ltx_video` (no-op under no-fallbacks); L893 clamps both
  ltx_av lanes to `OTR_LTX_AV_RENDER_CANVAS` (default **512x288**); L974 ltx_av_music
  joins the ltx_video scene-prompt branch.
- `eng_ltx_video` currently owns `default_roles=(announcer_visual, music_visual)` -- must
  be UN-claimed there so the default for those roles becomes ltx_av_music.
- Workflow JSON `otr_scifi_16gb_full.json`: `OTR_VideoDirector` per-role engine dropdowns
  (announcer/music currently -> ltx_video) -- change to ltx_av_music IN THE SAME CHANGE.

## RENAME decision (NEEDS roundtable judgment)
`ltx_av` appears **314x across 40 files** (engine, registry, render_driver, av_dims,
role_compat, tests, the 2 mini JSONs, docs). **Recommendation: do NOT rename the internal
engine IDs** (`ltx_av_talk`/`ltx_av_music`) -- huge blast radius + breaks saved workflows/
tests. Instead a **label rename**: present the lane as **"LTX (Audio In)"** in the
OTR_VideoDirector dropdown display + docs + a `display_name`/family-description, IDs stable.

## Sprint chunks (each: code + JSON-in-lockstep + regression suite + Bug Bible + audio byte-identical + commit/push)
- **S1 -- label rename.** "LTX (Audio In)" user-facing label (dropdown display + docs);
  keep IDs. Lowest risk; first so the naming is settled.
- **S2 -- sharp recipe in `eng_ltx_av`.** Add a "sharp" mode mirroring eng_ltx_video:
  LoRA @0.70 wired unet->lora->CFGGuider.model (DROP ModelSamplingLTXV in sharp mode),
  KSamplerSelect `euler_cfg_pp` + ManualSigmas (`_SigmasFromValues`) + cfg 1.0, i2v
  strength 0.75. Add the distilled LoRA to `_weight_paths`+`assert_usable` (license-clean).
  Env `OTR_LTX_AV_SHARP` default ON; OFF restores the M0 recipe. Both talk + music branches.
- **S3 -- defaults.** `ltx_av_music.default_roles=(music_visual, announcer_visual)`;
  UN-claim them from eng_ltx_video; un-dark ltx_av for the default path (requires_flag
  default ON, mirroring eng_ltx_video's opt-OUT pattern); render_driver canvas default
  512x288 -> **832x480** for ltx_av (per the A/B). Keep ltx_video selectable.
- **S4 -- JSON wiring.** OTR_VideoDirector announcer/music dropdowns -> ltx_av_music in
  `otr_scifi_16gb_full.json`; re-validate (OTR_WorkflowValidator + JSON round-trip +
  link/widget audit). Hard rule: code that is not wired into this JSON is DEAD.
- **S5 -- end-to-end.** A live episode render with the new defaults: suite 4452/0 + Bug
  Bible 16/7/3 + test_audio_byte_identical green + per-beat NVML reported + JSON git-clean.
  Operator eyeballs the music/announcer beats ("radio wiggles with the music").
- **S6 -- commit/push** per green chunk on `v2.0-alpha`; verify HEAD==origin, no BOM, AST.

## Questions for the panel (wiring-bug hunt + convergence)
1. **Rename:** confirm label-only (keep IDs) vs a full ID rename -- any cleaner middle
   (e.g. an alias map) given 314 refs? Risks of the label approach (dropdown value vs id)?
2. **Sharp-mode wiring in the dual-branch A2V graph:** dropping `ModelSamplingLTXV` while
   keeping the audio concat -- any dependency the talk (i2v) vs music (empty-latent)
   branch has on ModelSamplingLTXV that breaks when removed? Correct insertion point for
   the LoRA so the audio path is unaffected?
3. **Default un-claim:** moving `(announcer_visual, music_visual)` default from
   eng_ltx_video to ltx_av_music -- does `default_engine_for_role` tolerate the handoff,
   and does the saved JSON dropdown OVERRIDE the registry default (so the JSON change is
   the load-bearing one)? Any place that hard-assumes ltx_video for those roles?
4. **Un-dark safety:** flipping `OTR_ENABLE_LTX_AV` default ON -- does anything (tests,
   soak gates, dep-pilot) assume it's OFF? no-fallbacks means a missing audio_ref now
   fails LOUD -- do announcer + music beats ALWAYS carry audio_ref?
5. **Canvas:** ship 832x480 (peaks ~15.3 GB, over the 14.5 soft cap, no OOM) -- acceptable,
   or must it stay <=14.5 (then the LoRA VRAM is the lever, not the canvas)?
6. Anything that breaks `test_audio_byte_identical`, the V-12 cold-import rule, or the
   workflow-source-of-truth invariant.

Deliver: a ranked, grounded, build-ready sprint with the wiring bugs caught BEFORE coding.
