# FOLEY BED -- operator rulings, 2026-08-26

The LTX 2.5 foley bed: the video model's OWN generated audio, mixed UNDER the
TTS and music.

## TERMINOLOGY -- these are two different things and must never be conflated

* **SFX bed** -- separately GENERATED sound effects from a dedicated SFX model.
  **RIPPED 2026-08-06** (`rip-sfx`). Dead, and staying dead. The retired
  `clip_manifest_json` connector on `OTR_MasterAudioMux` is what fed it.
* **FOLEY bed** -- the audio LTX 2.5 already computes as part of the video
  render and currently DISCARDS at `LTXVSeparateAVLatent`
  (`eng_ltx25.py`; `LTXVAudioVAEDecode` is never wired). Never built. Has
  nothing to do with the rip.

Operator, 2026-08-26: *"sfx bed is different than foley bed, i won't get the
two confused."* Rebuilding the ripped SFX path under the name "foley" would be
resurrecting exactly what was deliberately killed.

## RULING 1 -- FIXED MIX, NO DYNAMIC DUCKING

Operator: *"ducking fixed i would say .20 foley .80 voice."*

**Foley 0.20 / voice 0.80. A fixed ratio, not a sidechain.**

Consequences, stated so the build cannot drift:

* NO sidechain compression, NO per-beat loudness analysis, NO envelope
  following. A static gain is deterministic and reproducible -- the same
  inputs give the same master every time, which a dynamic ducker would not.
* This is NOT the retired SFX bed's `OTR_SFX_BED_GAIN` default of 0.45. The
  foley bed sits lower because it plays under dialogue continuously rather
  than in gaps. Do not inherit that constant or its name.
* The TTS/music master is not attenuated by the foley's presence -- voice
  holds 0.80 whether or not a foley stem exists for that beat, so a beat
  without foley does not get louder.

### Ledger-driven ducking was considered and DEFERRED, not rejected

Raised 2026-08-26 in the same conversation: *"maybe there's a more intelligent
way, auto ducking sounds cool."* The option costed out was NOT envelope
sidechain -- it was **ledger-driven** ducking, which is available here and is
not available to a normal mixer: every line's `line_id`, `start_s` and
duration are already frozen in the ledger BEFORE video renders, so the bed can
duck from the SCRIPT rather than by detecting the voice. Foley 0.20 under a
line, ~0.50-0.60 in the gaps, ~150-250 ms ramps -- ambience swelling between
lines the way radio drama actually does, while staying fully deterministic
(same ledger, same mix, no take-to-take variance, no pumping).

Operator's call: *"let's start simple 80/20."* So the FIXED ratio ships first
and proves the foley path end to end; the ledger-driven envelope is a later
build on top of a working bed, not a prerequisite. **0.20 remains the
speech-duck floor in that later design**, so this ruling is not superseded by
it -- the gaps simply get to open above it.

## WHAT THE LAB ALREADY PROVED (evidence, not design)

`vram-recipe-lab/LTX_2_5_ON_16GB.md`: *"Foley and score are the strong suit.
The audio the model generates for a scene it is also rendering -- footsteps,
room tone, a theremin cue -- is where the joint model earns its keep over
bolting a separate audio pass on afterwards."*

Working recipes exist: `recipes/ltx_2_5_golden_i2v_foley.json` and
`ltx_2_5_golden_t2v_action_foley.json` (93.0 s render, 15.51 GiB). Both wire
`LTXVAudioVAEDecode` against `ltx-2.5-audio-vae-bf16.safetensors` with
`audio_cfg = 1.0`.

**But no design exists for keeping it.** Searched both repos on 2026-08-26:
EVERY mux path in OTR and in the lab discards the model audio and muxes TTS
instead -- `ENVELOPE_LADDERS.md:84,125`, `mux_source_delivery.py`,
`HUMO_BAKEOFF.md:148`, `H3_LAB_CANDIDATE_HANDOFF:147`, and
`OTR_MasterAudioMux` itself. The foley is proven good and then thrown away,
every time. Keeping it is new ground.

## RULING 2 -- NORMALIZE THE WHOLE THING AT THE END (already true; do not rebuild it)

Operator: *"of course normalize the whole thing when done."*

**This already exists and needs no new code.** `scene_sequencer._master_loudness()`
measures integrated LUFS and applies a SINGLE LINEAR GAIN to `-14.0` LUFS
(`_MASTER_TARGET_LUFS`), peak-safe at `-1.0` dBFS.

Why it is safe for the foley bed, and why that is not luck: a single linear
gain moves the finished mix as a whole and therefore **cannot disturb the
0.20/0.80 ratio**. A compressor or limiter at this stage would have
re-balanced foley against voice unpredictably. The ratio is set upstream and
survives delivery levelling unchanged.

**THE LINE THE BUILD MUST NOT CROSS**, in that function's own words: *"THIS
STAGE SETS THE DELIVERY LEVEL. It does NOT balance the mix -- that already
happened per clip, upstream... The two do different jobs and must not be
confused."* So: the foley bed is mixed at 0.20/0.80 UPSTREAM, and
`_master_loudness` stays purely a delivery stage. **The foley stem gets no
normalization pass of its own** -- normalizing it separately would fight the
ratio rather than serve it.

The `-14` target is measured, not chosen: 8 real masters across two months
delivered a mean `-9.87 LUFS (std 0.41)`, ~4 dB hot, so every episode was
attenuated at playback while the limiting used to buy that loudness stayed in
the audio. Do not master hot.

## STILL OPEN -- for the design arc

1. **Where it mixes.** `OTR_MasterAudioMux` (after video, master already
   frozen -- the ordering never bites) versus earlier at SceneSequencer.
2. **Timing mismatch.** Foley is generated per-clip at the clip's own length;
   TTS timing is frozen before video renders. What happens when they disagree
   -- trim, loop, pad, or refuse?
3. **Beats with no foley.** Still lanes (`still_flat`, `still_pan`, ...)
   generate no model audio. Silence under those beats, or bed continuity
   carried from the neighbouring clip?
4. **Engine shape.** `ltx25_foley_plus` as its OWN internal engine (lesson L5:
   two public ids on one internal id collapses `_INTERNAL_TO_PUBLIC` and trips
   the bijection assert AT IMPORT) versus a capability flag on `ltx25_video`.

## WHAT IS NOT BLOCKED

The 2026-08-19 standing ruling says *"Chunk B (the foley bed) remains BLOCKED
on execution order."* That blocker is real for **mime** -- where the clip's
audio REPLACES the beat audio and therefore must exist before the master
freezes. It does NOT bite the foley bed: a bed mixed UNDER an already-frozen
master happens at `OTR_MasterAudioMux`, which runs AFTER video. The
`clip_manifest_json` connector is still physically wired on the canonical
graph and still hashed by `IS_CHANGED` -- only the compiler behind it was
deleted -- so no workflow JSON surgery is needed to reach the mux.
