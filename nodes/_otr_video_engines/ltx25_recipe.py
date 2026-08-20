"""LTX 2.5 Distilled -- the LOCKED recipe constants, and why each one is locked.

THIS MODULE IS DATA, NOT BEHAVIOUR. It holds the parameters the lab measured
and froze, with the reason beside each one, so the adapter reads as wiring and
the numbers have a single home. Nothing here imports torch or touches the GPU.

PROVENANCE. `~/.gemini/antigravity/brain/588e.../ltx_2_5_final_qa_and_output_
review.md` (2026-08-19), the lab's "Final QA & Output Review", with its
predecessor `otr_ltx_2_5_integration_handoff.md`. Reviewed against
`docs/VIDEO_LANE_PREFLIGHT.md` in `kibitz-runs/2026-08-19-ltx25-integration/`.

**THE LAB'S NUMBERS ARE LAB NUMBERS.** `CLAUDE.md` section 0A is explicit that a
bench result "may never be worded as qualification" for OTR and must be
re-proved through the canonical workflow. So the VRAM figure below is recorded
as a lab observation and is NOT an envelope key; the G4 admission row and the
`low`/`high` token in the public id both wait on OUR OWN solo smoke (G8).

**AND THE RECIPE IS NOT ON THE TABLE** (operator, standing rule, restated
2026-08-19: *"no chasing vram recipes please... we are running on the Q3, that's
the safe one"*). These values are implemented exactly as measured. Do not tune
them to make a number nicer, do not try Q5, do not raise the canvas, do not
chase frames. If a value here turns out to be wrong, that is a finding to
report, not a knob to turn.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Weights. Resolved through `folder_paths` by the adapter (G1) -- never a bare
# os.path.exists on a hardcoded default, which is the defect G1.1 exists for.
# ---------------------------------------------------------------------------

#: The DiT. Q3 is LOCKED and it is the SAFE one: the lab measured Q5_K_M
#: breaching the clamp at 832x480 (15.58 GiB). The Q5 file is on this box under
#: `C:\ComfyUI-Models\quarantine\` -- quarantined, not a candidate.
LTX25_DIT_GGUF = "LTX-2.5-Distilled-Q3_K_M.gguf"

#: The text encoder. Gemma-4 12B with the LTX 2.5 projection, Q5 quantised.
LTX25_TEXT_ENCODER_GGUF = "gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf"

#: Native BF16 VAEs. The AUDIO vae is named here even though the first lane
#: never decodes with it -- the later foley lanes will, and the lab measured
#: that decode as cheap (<50 MB VRAM, <1 s). Naming it now costs nothing and
#: means the foley work does not have to rediscover the filename.
LTX25_VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX25_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"

#: THE AUDIO VAE IS REQUIRED EVEN BY THE SILENT LANE, and this surprises people.
#: ``LTXVEmptyLatentAudio`` (golden JSON node 12) takes ``audio_vae`` to MINT the
#: audio latent, and ``LTXVConcatAVLatent`` (node 30) needs that latent to build
#: the joint AV tensor the sampler consumes. So a silent lane still loads the
#: audio VAE and still computes the audio side through all 8 steps -- it only
#: skips ``LTXVAudioVAEDecode`` (node 34) at the end. This is exactly what
#: ``eng_ltx_av.py`` already does, and it means "we discard the audio" never
#: meant "we avoid paying for it".
LTX25_AUDIO_VAE_REQUIRED_EVEN_WHEN_SILENT = True

# ---------------------------------------------------------------------------
# Canvas and length
# ---------------------------------------------------------------------------

#: 832x480. BOTH axes must be cleanly divisible by 32 -- 832/32 = 26,
#: 480/32 = 15. The lab rejected 768x432 because 432/32 = 13.5 corrupts the
#: tensor and fails PyTorch VAE decoding, and rejected 1024x576 on OOM. This is
#: also exactly the canvas `VIDEO_LANE_PREFLIGHT` G2.4 exists to police, and it
#: matches the canvas the rest of the OTR video fleet already renders.
LTX25_CANVAS_W = 832
LTX25_CANVAS_H = 480

#: 97 frames at 25 fps = 3.88 s, the standard OTR shot length. The temporal
#: contract is `(97 - 1) % 8 == 0`, which the model's temporal downsampling
#: requires. `CLAUDE.md` separately forbids raising the 97 trained-length cap,
#: so this satisfies both constraints at once rather than by coincidence.
LTX25_FRAMES = 97
LTX25_FPS = 25

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

#: Ancestral sampling keeps motion alive across only 8 distilled steps; the lab
#: found non-ancestral samplers freeze the latent at this step count.
LTX25_SAMPLER = "euler_ancestral_cfg_pp"
LTX25_STEPS = 8

#: ALL THREE CFGs ARE 1.0 AND THAT IS A VRAM CONTRACT, NOT A TASTE SETTING.
#: CFG 1.0 evaluates batch size 1. Any value above 1.0 forces batch size 2
#: (positive + negative evaluated together) and the lab measured that pushing
#: past 16 GiB -- an instant OOM against the 14.5 GiB clamp. So "turn the CFG up
#: a little" is not a small change here; it doubles the batch.
LTX25_CFG_VIDEO = 1.0
LTX25_CFG_AUDIO = 1.0
LTX25_CFG_MODALITY = 1.0

#: No negative prompt. Not a style choice: negative conditioning is INERT at
#: CFG 1.0, so carrying one buys nothing and costs memory. This is also why
#: "just add a negative to suppress X" is not available on this lane -- the
#: only steering channel is the positive prompt.
LTX25_NEGATIVE_PROMPT = ""

# ---------------------------------------------------------------------------
# Decode -- the tiled VAE knobs are RECIPE VALUES, not house defaults
# ---------------------------------------------------------------------------

#: ``VAEDecodeTiled`` (golden JSON node 33), verbatim. These live HERE rather
#: than as literals in the adapter for one specific reason: the sibling
#: ``eng_ltx_av`` decodes through an ENV-DRIVEN helper whose default is
#: **4096 / 8** -- whole-clip, no temporal tiling -- and its comment praises
#: that default for having no inter-tile seam. Copying that helper into this
#: lane, which is the natural thing to do when modelling one adapter on
#: another, would silently replace a measured recipe value with a different
#: one on a lane that has 0.02 GiB of headroom. Whole-clip decode of 97 frames
#: is exactly the kind of allocation that spends headroom this lane does not
#: have.
#:
#: So: 33 frames per temporal tile with a 4-frame overlap, 512-pixel spatial
#: tiles with 64 overlap, as measured. NOT env-overridable, because there is no
#: number here an environment is entitled to move (the recipe is locked), and a
#: knob that reaches nothing is worse than no knob.
LTX25_DECODE_TILE_SIZE = 512
LTX25_DECODE_OVERLAP = 64
LTX25_DECODE_TEMPORAL_SIZE = 33
LTX25_DECODE_TEMPORAL_OVERLAP = 4

# ---------------------------------------------------------------------------
# Conditioning -- the first-frame anchor
# ---------------------------------------------------------------------------

#: I2V first-frame anchor, via ``LTXVImgToVideoInplace`` at **strength 1.0**.
#:
#: CORRECTED AGAINST THE GOLDEN JSON. The lab's PROSE describes this as
#: ``SetLatentNoiseMask`` with frame 0 at 0.0 and frames 1-96 at 1.0. The
#: executable recipe
#: (`vram-recipe-lab/recipes/ltx_2_5_golden_i2v_foley.json`, node 16) actually
#: uses ``LTXVImgToVideoInplace`` with ``strength: 1.0, bypass: false``. The
#: JSON is authoritative -- it is the file that ran. Same effect (frame 0 is
#: pinned to the still), different node.
#:
#: THE NODE IS ALREADY KNOWN HERE, WHICH IS THE DE-RISK: ``eng_ltx_av.py:822``
#: and ``eng_ltx_video.py:1233`` both wire ``LTXVImgToVideoInplace`` already.
#:
#: BUT AT A DIFFERENT STRENGTH, AND THAT IS THE ONE THING TO WATCH. The audio
#: lane deliberately uses **0.7, a SOFT anchor** (`eng_ltx_av.py:972`: "a SOFT
#: anchor so the audio can..."), and this file's own note at `:423` records why
#: -- "strength 1.0 hard-pins the still". So 1.0 here is a HARDER anchor than
#: the sibling lane uses. It holds identity firmly at frame 0 and leaves frames
#: 1..96 free, which is a different trade from a soft anchor applied across the
#: clip (`eng_ltx_video.py`'s "i2v-anchor doctrine").
#:
#: **CORRECTED 2026-08-19: 1.0 IS THE NODE'S UNTOUCHED DEFAULT, NOT A
#: MEASUREMENT.** This note previously said "adopted as the lab measured it",
#: and the lab has since confirmed it measured nothing of the kind --
#: ``LTXVImgToVideoInplace`` ships ``strength`` defaulting to 1.0 and the
#: recipe never touched the widget. So the difference from the sibling lane's
#: 0.7 is NOT a deliberate opposing choice by two teams; it is one deliberate
#: choice (0.7, ours, with a written reason) and one default nobody set. Read
#: it that way before treating 1.0 as evidence of anything.
#:
#: IT IS STILL WHAT SHIPS, and that is not laziness. A hard pin at frame 0 is
#: the correct default for OTR's actual problem -- "a character's face changing
#: between beats" is a live CORRECTNESS defect in `CLAUDE.md`, and the harder
#: anchor is the one that fights it. Changing it would be a recipe change, and
#: recipes are not on the table. If it ever wants revisiting, the honest
#: framing is "0.7 vs 1.0 has never been A/B'd on this model", not "the lab
#: chose 1.0".
LTX25_I2V_ANCHOR_STRENGTH = 1.0
LTX25_I2V_ANCHOR_NODE = "LTXVImgToVideoInplace"

#: The scheduler's ``latent`` port. CORRECTED AGAINST THE GOLDEN JSON: the prose
#: says "connected to the EmptyLTXVLatentVideo output", but node 7 in the
#: executable recipe takes ``latent: ["16", 0]`` -- the ImgToVideoInplace
#: OUTPUT, which is itself fed by EmptyLTXVLatentVideo (node 11). On the I2V
#: path those are not the same tensor, and wiring it to node 11 directly would
#: hand the scheduler a latent with no still baked in.
LTX25_SCHEDULER_LATENT_SOURCE = "LTXVImgToVideoInplace"

# ---------------------------------------------------------------------------
# Explicitly CLOSED options -- recorded so nobody re-opens them by accident
# ---------------------------------------------------------------------------

#: 161-frame multishot: REJECTED. Spikes to 18-20 GiB at this canvas. The
#: replacement for multi-shot continuity is the first-frame anchor above.
LTX25_MULTISHOT_ALLOWED = False

#: In-graph 2x latent upscaling: BANNED from this graph. It forces the video
#: VAE to decode 1664x960x97 and hard-OOMs. The upscaler weight IS on this box
#: (`latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0`) and may
#: be used as a SEPARATE offline pass -- never inside this lane's graph.
LTX25_INGRAPH_UPSCALE_ALLOWED = False

#: The scheduler's `latent` port MUST be connected to the empty/init latent.
#: Left dangling it silently defaults to a 4096-token curve and ruins the motion
#: shift maths -- a wrong-but-running failure, which is the worst kind. Pinned
#: by test rather than trusted to review.
LTX25_SCHEDULER_LATENT_MUST_BE_CONNECTED = True

# ---------------------------------------------------------------------------
# Lab observation -- NOT an OTR qualification
# ---------------------------------------------------------------------------

#: What the LAB measured for the locked recipe. Recorded for traceability and
#: deliberately NOT used as an admission number: per `CLAUDE.md` 0A a bench
#: result may never be worded as qualification, and this figure sits 0.02 GiB
#: under the 14.5 clamp, which is far too tight to inherit on trust. OUR figure
#: comes from the G8 solo smoke on OUR boot lane, and it is that figure -- not
#: this one -- that fills the G4 envelope key.
#:
#: **THE PUBLIC TOKEN NO LONGER WAITS ON THIS (2026-08-19).** It used to, and
#: the note said so. The operator then ruled the 4060 out entirely, which
#: settles the naming by deleting the question rather than answering it: the
#: lane is 5080-only, so `high` is what the token means and
#: `ltx25_high_video` is registered. The envelope key still waits on the smoke.
LTX25_LAB_OBSERVED_PEAK_GIB = 14.48
LTX25_LAB_CLAMP_GIB = 14.5

#: **WHERE THE 14.48 GiB ACTUALLY GOES, and why staging cannot shrink it**
#: (lab correction, 2026-08-19 -- it overturned the driver's own framing).
#:
#: The peak decomposes as **9.80 GiB DiT weights + 3.20 GiB activations +
#: 1.48 GiB allocator context**, with the text encoder and both VAEs at
#: **ZERO**. It was measured with Gemma ALREADY evicted -- ComfyUI spills the
#: encoder to system RAM before sampling on its own.
#:
#: SO THE OBVIOUS INFERENCE IS WRONG. "Load the encoder, free it, then load the
#: transformer" does not buy headroom here, because at the moment of the peak
#: the encoder was never resident in the first place. The adapter still runs
#: `free_after_use` and still calls the canonical residue-freer, and both are
#: worth keeping -- but as HYGIENE against residue from the writer LLM and the
#: TTS stages EARLIER IN THE SAME PROCESS, not as the thing that makes this
#: lane fit. It does not make it fit; the DiT and its activations do that on
#: their own, with 0.02 GiB to spare.
#:
#: THE CONSEQUENCE FOR ANYONE READING A FAILING SMOKE: if this lane OOMs, the
#: cause is upstream residue or allocator fragmentation, NOT a staging bug in
#: the adapter, and the fix is not to add another free() call. Report it.
LTX25_PEAK_DECOMPOSITION_GIB = {
    "dit_weights": 9.80,
    "activations": 3.20,
    "allocator_context": 1.48,
    "text_encoder": 0.0,
    "vaes": 0.0,
}

#: Staging is hygiene, not headroom. Pinned as a constant so the claim has one
#: home and a test can hold the adapter's comments to it.
LTX25_STAGING_REDUCES_PEAK = False

# ---------------------------------------------------------------------------
# OUR OWN MEASUREMENT -- the G8 solo smoke, 2026-08-19. THIS is the number that
# describes this box; everything above describes the lab's.
# ---------------------------------------------------------------------------

#: WHAT THE G8 SOLO SMOKE ACTUALLY MEASURED, and it does NOT match the lab.
#:
#: Two renders on the Sage-free LTX boot, one on the lab's own 832x480 still and
#: one on a real OTR 1472x832 scene still. Both succeeded (97 frames, 832x480,
#: silence proved by ffprobe on the emitted file) and both reported the SAME
#: peak, which is what makes this a measurement rather than a reading:
#:
#:   * adapter ``VramPeakProbe`` @0.1 s -- **16152 MiB** on both runs
#:   * an INDEPENDENT ``nvidia-smi`` sampler @0.5 s -- **15848 MiB**
#:   * card total on this box -- **16303 MiB**
#:
#: The two instruments differ by 304 MiB, which is what a 5x coarser sampler
#: missing the top of a spike looks like; they corroborate rather than conflict.
#: Take 16152 as the peak and 15848 as its independent floor.
#:
#: **THAT IS 1324 MiB ABOVE THE LAB'S 14.48 GiB AND 1304 MiB OVER THE 14.5 GiB
#: CLAMP -- roughly 99% of the card.** It is exactly the failure mode this
#: family already has on record: in-pipeline peaks read HIGHER than isolation
#: probes, which is how the sibling LTX lane breached the same ceiling at
#: 14,716 MB after passing its own isolation bench. The 0.02 GiB of lab margin
#: was, as this module's own note predicted, far too tight to inherit on trust.
#:
#: **THIS IS A FINDING, NOT A KNOB** (operator, standing: a wrong value is
#: reported, never tuned). Nothing here is adjusted in response to it: Q3,
#: 832x480, 97 frames and the 8-step ladder are unchanged. Do not "fix" this by
#: lowering the canvas -- and note that the exact-16:9 rung the 2026-07-26 arc
#: judgment would prefer, 1024x576, is 1.48x these pixels, so this measurement
#: is evidence FOR the lab's OOM rejection of it rather than against.
LTX25_OTR_MEASURED_PEAK_MIB = 16152
LTX25_OTR_MEASURED_PEAK_FLOOR_MIB = 15848
LTX25_OTR_CARD_TOTAL_MIB = 16303
