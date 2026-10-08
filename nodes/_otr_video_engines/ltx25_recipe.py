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
re-proved through the canonical workflow. So the VRAM figure the lab measured
is a lab observation and is NOT an envelope key, and the G4 envelope admission
still waits on OUR OWN solo smoke (G8).

The lab's figure is a 14.48 GiB peak under a 14.5 GiB clamp: 9.80 GiB of DiT
weights + 3.20 GiB of activations + 1.48 GiB of allocator context, with the text
encoder and both VAEs at ZERO (Gemma was already evicted -- ComfyUI spills the
encoder to system RAM before sampling on its own). So staging cannot shrink it:
`free_after_use` and the residue-freer are hygiene against the writer LLM and
the TTS stages earlier in the same process, not what makes this lane fit, and a
smoke that OOMs is an upstream-residue or allocator-fragmentation finding to
report, not a missing free() call.

The `low`/`high` token in the public id is settled: the naming is decided and
the `high` lanes are registered and shipping.

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
#: future work until 2026-08-28. It is required even by the SILENT lane:
#: ``LTXVEmptyLatentAudio`` takes it to MINT the audio latent the joint AV
#: sampler consumes, so a silent lane still loads it and only skips
#: ``LTXVAudioVAEDecode`` (see ``eng_ltx25._weight_paths``).
LTX25_VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX25_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"

#: The official LTX 2.5 latent spatial upscaler used by the selected HQ
#: two-stage graph. The lab's executable recipe and Comfy's downloadable I2V
#: workflow use this exact filename.
LTX25_UPSCALER_MODEL = (
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

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
#: so this satisfies both constraints at once rather than by coincidence. The
#: lab's 161-frame multishot was REJECTED (it spikes to 18-20 GiB at this
#: canvas); the replacement for multi-shot continuity is the first-frame anchor
#: below.
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
#: **THE ORDINARY COMFYUI RULE DOES NOT APPLY TO THIS RECIPE.** That rule
#: ("CFG 1.0 evaluates batch size 1; any value above 1.0 forces batch size 2")
#: is how ``comfy/samplers.py`` behaves (``sampling_function`` sets
#: ``uncond_ = None`` when ``cond_scale`` is close to 1.0) -- but the locked
#: sampler is CFG++: ``sample_euler_ancestral_cfg_pp`` explicitly passes
#: ``disable_cfg1_optimization=True`` (``comfy/k_diffusion/sampling.py:1284``)
#: and then CONSUMES ``uncond_denoised`` in its own derivative (``:1297``).
#: So the unconditional branch is evaluated at CFG 1.0 on this lane, every
#: step.
#:
#: The lab measured 14.48 GiB running THIS sampler, so whatever the true
#: batching, the measurement already includes it. The mechanism still matters:
#: believing the ordinary rule makes an empty negative prompt look free (see
#: ``LTX25_NEGATIVE_PROMPT``).
LTX25_CFG_VIDEO = 1.0
LTX25_CFG_AUDIO = 1.0
LTX25_CFG_MODALITY = 1.0

#: The negative prompt TEXT is empty. That is the recipe and it is locked.
#:
#: **THE NEGATIVE CONDITIONING IS NOT INERT.** The ordinary ComfyUI rule
#: ("negative conditioning is INERT at CFG 1.0, so carrying one buys nothing
#: and costs memory") is FALSE for this recipe: the locked sampler
#: ``euler_ancestral_cfg_pp`` forces ``disable_cfg1_optimization=True``
#: (``comfy/k_diffusion/sampling.py:1284``) and uses ``uncond_denoised`` in its
#: step derivative (``:1297``). The unconditional branch really is computed,
#: every step, and it really does steer the result.
#:
#: So the obvious-looking optimisation -- feed the POSITIVE conditioning into
#: both guider slots and skip a whole 12B encode -- would silently change every
#: render on this lane. Do not propose it.
#:
#: WHAT REMAINS TRUE: the empty STRING is the locked recipe value, and "just
#: add a negative to suppress X" is still unavailable -- not because the
#: channel is dead, but because the text is a locked recipe value.
LTX25_NEGATIVE_PROMPT = ""

# ---------------------------------------------------------------------------
# Decode -- the tiled VAE knobs are RECIPE VALUES, not house defaults
# ---------------------------------------------------------------------------

#: ``VAEDecodeTiled`` as the executable two-stage recipe has it, verbatim. There
#: is ONE decode: after the refinement sampler, at the doubled canvas. (The
#: lab's one-stage golden tiled its decode 33 frames with a 4-frame overlap;
#: that decode is not in the shipping graph.)
#:
#: These live HERE rather than as literals in the adapter for one specific
#: reason: an ENV-DRIVEN decode helper whose default is **4096 / 8** --
#: whole-clip, no temporal tiling, praised for having no inter-tile seam --
#: is the natural thing to copy when modelling one adapter on another, and
#: would silently replace a measured recipe value with a different one on a
#: lane that has 0.02 GiB of headroom. Whole-clip decode of 97 frames is
#: exactly the kind of allocation that spends headroom this lane does not have.
#:
#: So: 64 frames per temporal tile with a 16-frame overlap, 512-pixel spatial
#: tiles with 64 overlap, as measured. NOT env-overridable, because there is no
#: number here an environment is entitled to move (the recipe is locked), and a
#: knob that reaches nothing is worse than no knob.
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
#: COMFYUI IS NOT THE ONE STREAMING THE DECODE OVER PCIe. The LTX
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
#: The lab's PROSE describes this as ``SetLatentNoiseMask`` with frame 0 at 0.0
#: and frames 1-96 at 1.0. The executable recipe
#: (`vram-recipe-lab/recipes/ltx_2_5_golden_i2v_foley.json`, node 16) actually
#: uses ``LTXVImgToVideoInplace`` with ``strength: 1.0, bypass: false``. The
#: JSON is authoritative -- it is the file that ran. Same effect (frame 0 is
#: pinned to the still), different node.
#:
#: 1.0 is a HARD anchor: it holds identity firmly at frame 0 and leaves frames
#: 1..96 free, which is a different trade from a SOFT anchor (e.g. 0.7)
#: applied across the clip ("strength 1.0 hard-pins the still").
#:
#: **1.0 IS THE NODE'S UNTOUCHED DEFAULT, NOT A MEASUREMENT.**
#: ``LTXVImgToVideoInplace`` ships ``strength`` defaulting to 1.0 and the
#: recipe never touched the widget, so 1.0 is not evidence of anything.
#:
#: IT IS STILL WHAT SHIPS, and that is not laziness. A hard pin at frame 0 is
#: the correct default for OTR's actual problem -- "a character's face changing
#: between beats" is a live CORRECTNESS defect in `CLAUDE.md`, and the harder
#: anchor is the one that fights it. Changing it would be a recipe change, and
#: recipes are not on the table. If it ever wants revisiting, the honest
#: framing is "0.7 vs 1.0 has never been A/B'd on this model", not "the lab
#: chose 1.0".
LTX25_I2V_ANCHOR_STRENGTH = 1.0
