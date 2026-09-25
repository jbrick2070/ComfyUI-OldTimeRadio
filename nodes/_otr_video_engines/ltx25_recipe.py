"""LTX 2.5 Distilled -- the LOCKED recipe constants, and why each one is locked.

THIS MODULE IS DATA, NOT BEHAVIOUR. It holds the parameters the lab measured
and froze, with the reason beside each one, so the adapter reads as wiring and
the numbers have a single home. Nothing here imports torch or touches the GPU.

PROVENANCE. `~/.gemini/antigravity/brain/588e.../ltx_2_5_final_qa_and_output_
review.md` (2026-08-19), the lab's "Final QA & Output Review", with its
predecessor `otr_ltx_2_5_integration_handoff.md`. Reviewed against
`apple/VIDEO_LANE_PREFLIGHT.md` during the 2026-08-19 LTX 2.5 integration review.

**THE LAB'S NUMBERS ARE LAB NUMBERS.** `CLAUDE.md` section 0A is explicit that a
bench result "may never be worded as qualification" for OTR and must be
re-proved through the canonical workflow. So the VRAM figure below is recorded
as a lab observation and is NOT an envelope key, and the G4 envelope admission
still waits on OUR OWN solo smoke (G8).

The `low`/`high` token in the public id is NO LONGER waiting -- that naming is
settled and the `high` lanes are registered and shipping (corrected 2026-08-28;
this paragraph used to bundle the naming decision with the open envelope
question, so a reader could not tell which half was still undecided).

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

#: The DiT and the text encoder are per-tier and live in eng_ltx25
#: (``LTX25_NATIVE_*``): the recipe below is the same on every tier.

#: Native BF16 VAEs. The AUDIO vae is DECODED TODAY by the Foley and MIME
#: lanes (a second decode pass in eng_ltx25; measured cheap in the lab at
#: <50 MB VRAM and <1 s, which is why it was affordable). It was named here
#: before those lanes existed, and this comment described the decode as
#: future work until 2026-08-28.
LTX25_VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX25_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"

#: The official LTX 2.5 latent spatial upscaler used by the selected HQ
#: two-stage graph. The lab's executable recipe and Comfy's downloadable I2V
#: workflow use this exact filename.
LTX25_UPSCALER_MODEL = (
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

#: THE AUDIO VAE IS REQUIRED EVEN BY THE SILENT LANE, and this surprises people.
#: ``LTXVEmptyLatentAudio`` (golden JSON node 12) takes ``audio_vae`` to MINT the
#: audio latent, and ``LTXVConcatAVLatent`` (node 30) needs that latent to build
#: the joint AV tensor the sampler consumes. So a silent lane still loads the
#: audio VAE and still computes the audio side through all 8 steps -- it only
#: skips ``LTXVAudioVAEDecode`` (node 34) at the end. This is exactly what
#: the retired ``eng_ltx_av`` lane already did, and it means "we discard the
#: audio" never meant "we avoid paying for it".
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

#: Stage one stays at the model's locked 832x480 canvas. The accepted HQ path
#: doubles the LATENT and decodes exactly 1664x960; it does not ask the first
#: sampler to run above its native envelope.
LTX25_RENDER_CANVAS_W = LTX25_CANVAS_W * 2
LTX25_RENDER_CANVAS_H = LTX25_CANVAS_H * 2

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
#: Raising any of them is measured by the lab to push past 16 GiB -- an instant
#: OOM against the 14.5 GiB clamp. So "turn the CFG up a little" is not a small
#: change here. Leave them.
#:
#: **THE MECHANISM ORIGINALLY GIVEN FOR THAT WAS WRONG -- CORRECTED 2026-08-19.**
#: This note used to say "CFG 1.0 evaluates batch size 1; any value above 1.0
#: forces batch size 2". That is the ordinary ComfyUI behaviour
#: (``comfy/samplers.py``: ``sampling_function`` sets ``uncond_ = None`` when
#: ``cond_scale`` is close to 1.0) -- but **it does not apply to this recipe**,
#: because the locked sampler is CFG++:
#: ``sample_euler_ancestral_cfg_pp`` explicitly passes
#: ``disable_cfg1_optimization=True`` (``comfy/k_diffusion/sampling.py:1284``)
#: and then CONSUMES ``uncond_denoised`` in its own derivative (``:1297``).
#: So the unconditional branch is evaluated at CFG 1.0 on this lane, every step.
#:
#: THE NUMBER IS UNAFFECTED, THE REASONING IS NOT. The lab measured 14.48 GiB
#: running THIS sampler, so whatever the true batching, the measurement already
#: includes it. What was wrong was the explanation -- and it mattered, because
#: it made an empty negative prompt look free (see ``LTX25_NEGATIVE_PROMPT``).
LTX25_CFG_VIDEO = 1.0
LTX25_CFG_AUDIO = 1.0
LTX25_CFG_MODALITY = 1.0

#: The negative prompt TEXT is empty. That is the recipe and it is locked.
#:
#: **BUT THE NEGATIVE CONDITIONING IS NOT INERT, AND THIS NOTE USED TO SAY IT
#: WAS -- CORRECTED 2026-08-19.** The old wording ("negative conditioning is
#: INERT at CFG 1.0, so carrying one buys nothing and costs memory") is the
#: ordinary ComfyUI rule, and it is FALSE for this recipe: the locked sampler
#: ``euler_ancestral_cfg_pp`` forces ``disable_cfg1_optimization=True``
#: (``comfy/k_diffusion/sampling.py:1284``) and uses ``uncond_denoised`` in its
#: step derivative (``:1297``). The unconditional branch really is computed,
#: every step, and it really does steer the result.
#:
#: WHY THE ERROR WAS EXPENSIVE. Believing the negative was inert made an
#: obvious-looking optimisation available -- feed the POSITIVE conditioning
#: into both guider slots and skip a whole 12B encode. It would have silently
#: changed every render on this lane. It was proposed during the 2026-08-19
#: OOM panel, survived one reviewer, and was killed by another that checked
#: which sampler was actually selected. Do not re-propose it.
#:
#: WHAT REMAINS TRUE: the empty STRING is the locked recipe value, and "just
#: add a negative to suppress X" is still unavailable -- not because the
#: channel is dead, but because the text is a locked recipe value.
LTX25_NEGATIVE_PROMPT = ""

# ---------------------------------------------------------------------------
# Decode -- the tiled VAE knobs are RECIPE VALUES, not house defaults
# ---------------------------------------------------------------------------

#: ``VAEDecodeTiled`` (golden JSON node 33), verbatim. These live HERE rather
#: than as literals in the adapter for one specific reason: the retired
#: sibling ``eng_ltx_av`` decoded through an ENV-DRIVEN helper whose default was
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

#: Terminal decode geometry from the executable two-stage recipe. There is one
#: decode: after the refinement sampler, at the doubled canvas.
LTX25_STAGE2_DECODE_TILE_SIZE = 512
LTX25_STAGE2_DECODE_OVERLAP = 64
LTX25_STAGE2_DECODE_TEMPORAL_SIZE = 64
LTX25_STAGE2_DECODE_TEMPORAL_OVERLAP = 16

#: What the terminal decode actually needs on the card, MEASURED (2026-09-22).
#:
#: An isolated probe -- only the video VAE, a synthetic stage-2 latent of the
#: real shape (1,128,13,30,52), the tiling above, and an inert ballast tensor
#: as the only variable -- decoded in **30.0 s at a peak of 8,080 MB** on a
#: free card, and had NOT finished at 412 s with 11 GB occupied. So this is not
#: a safety margin invented around a guess; it is the number the decode was
#: observed to want, rounded up by the width of one tile's working set.
#:
#: It exists to answer ONE question at run time: is there room for the decode,
#: or is the sampler's DiT still sitting where the decode needs to be? Below
#: this figure the same work takes more than twelve times longer -- a
#: 100%-utilisation, low-memory-controller, low-wattage signature that reads
#: like thrashing and is actually a card waiting on transfers over PCIe.
#:
#: THE STALL SIGNATURE, MEASURED ON TWO CARDS, and the reason a wattage
#: threshold cannot be the detector:
#:
#:     4060 (90 W cap)   working 60-79 W    stalled 32.6 and 33.4 W  (~36% cap)
#:     5080 laptop       working unmeasured stalled ~60 W  (operator's eye)
#:
#: A threshold tuned on the 4060 never fires on the 5080; one tuned on the 5080
#: fires constantly on the 4060. What transfers is the RATIO to that card's own
#: working baseline -- roughly a third of `power.max_limit`, or about half of
#: what the same card draws doing real work in the same run. The sampler stage
#: earlier in the same leg is a free per-run reference: it is unambiguously
#: doing work, so its mean draw is that card's healthy figure on that day, at
#: that clock, in that thermal envelope.
#:
#: BUT POWER IS THE HUMAN'S INSTRUMENT, NOT A WATCHDOG'S. The portable half of
#: the pair is `utilization.memory`: 0% memory-controller while
#: `utilization.gpu` reads 100%, against 55-65% in healthy phases. That ratio
#: has no TDP dependence at all, so it needs no per-card calibration. Power is
#: what lets a person spot a stall in one glance; an idle memory controller
#: under pinned utilisation is what code can trust.
#:
#: AND `torch.cuda.memory_reserved() > mem_get_info()[1]` CANNOT BE THE ONLY
#: CONDITION. It is definitive where it applies, but a 151-sample 4060 stall on
#: 2026-09-23 never tripped it -- reserved stayed under physical throughout,
#: with global sysmem fallback confirmed off and the CUDA context created after
#: the change. A detector built on that check alone would have missed the one
#: stall anybody has actually traced.
#:
#: The 4060 owns this watchdog: it is the portability surface and the only card
#: here that reproduces the stall on demand.
#:
#: COMFYUI IS NOT THE ONE STREAMING IT -- corrected 2026-09-23. This paragraph
#: said "ComfyUI streams the decode over PCIe", and that is wrong. The LTX
#: diffusion-VAE branch in `comfy/sd.py` sets ``disable_offload = True``, which
#: is handed straight to ``force_full_load`` on the ``load_models_gpu`` call,
#: so ComfyUI loads this VAE COMPLETELY and never streams it. A live 4060 log
#: agrees: "loaded completely; 1403.92 MB loaded, full load: True".
#:
#: What actually spills is the WINDOWS WDDM CUDA SYSMEM FALLBACK: the driver
#: backs an over-large allocation with pageable host memory instead of failing,
#: so `cudaMalloc` succeeds, PyTorch never raises, and the card sits at 100%
#: utilisation moving bytes over PCIe forever. That is why the stall never
#: OOMs, and it is why this class of hang is WINDOWS-ONLY -- Linux has no such
#: fallback. `nvidia-smi` cannot see it; Task Manager's "Shared GPU memory"
#: can, and from outside the process ComfyUI's own `/system_stats`
#: `torch_vram_total` exceeding the physical `vram_total` is the tell.
#:
#: The owner matters because it changes the fix. It is not a ComfyUI setting to
#: tune: it is either the driver profile ("CUDA - Sysmem Fallback Policy" ->
#: Prefer No Sysmem Fallback, an operator step with no env var or API), or
#: geometry that fits.
#:
#: AND THE POLICY IS READ WHEN THE CUDA CONTEXT IS CREATED. A server process
#: that predates the change does not pick it up, so a leg measured through an
#: already-running backend is a confound. Restart before measuring.
#:
#: Verified on the 4060 by the peer box, which read the install this repo's
#: 5080 checkout could not locate.
#:
#: 8300 IS A 5080-DERIVED NUMBER. The 8,080 MB peak above was measured on a
#: free 16 GB card. An 8 GB 4060 offers ~7,399 MB with nothing else resident,
#: so it is ~680 MB short of that peak by arithmetic -- which is why the 5080's
#: fix (evict, make room) does not transfer to it. A measured 4060 peak is
#: pending and replaces this figure when it lands.
LTX25_STAGE2_DECODE_NEEDS_MB = 8300

#: The exact three-step refinement schedule. Production resolves core's V3
#: ``ManualSigmas`` (registered before duplicate custom nodes), whose direct
#: Python input is ``sigmas``. The value is byte-identical to the lab recipe.
LTX25_REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
LTX25_TWO_STAGE_RECIPE_ID = "ltx_2_5_two_stage"

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
#: THE NODE IS ALREADY KNOWN HERE, WHICH IS THE DE-RISK: the retired
#: ``eng_ltx_av`` and ``eng_ltx_video`` lanes both wired ``LTXVImgToVideoInplace``
#: already.
#:
#: BUT AT A DIFFERENT STRENGTH, AND THAT IS THE ONE THING TO WATCH. The retired
#: audio lane deliberately used **0.7, a SOFT anchor** ("a SOFT
#: anchor so the audio can..."), and this file's own note at `:423` records why
#: -- "strength 1.0 hard-pins the still". So 1.0 here is a HARDER anchor than
#: that sibling lane used. It holds identity firmly at frame 0 and leaves frames
#: 1..96 free, which is a different trade from a soft anchor applied across the
#: clip (the retired ``eng_ltx_video``'s "i2v-anchor doctrine").
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

#: In-graph 2x latent upscaling: SELECTED for the shipping HQ path. The former
#: ban said decoding 1664x960x97 hard-OOMed; the lab subsequently ran exactly
#: that graph and produced the accepted HQ video. Keep it in one graph so a
#: canonical OTR render cannot publish while silently skipping refinement.
LTX25_INGRAPH_UPSCALE_ALLOWED = True

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
