# ARM C (FastWan 5B) -- build spec, and the ONE question that blocks it

**Status:** DRAFT for kibitz. No code, no graph, no GPU run performed to write it.
**Written:** 2026-07-31 at HEAD `4872b1f6`, branch `v2.0-alpha`.
**Owns:** standing arm C up in `scripts/run_video_arm_bakeoff.py` so it produces a
render and a VRAM number under the same clamp as arms A / B-partial / D.
**Does NOT own:** shipping FastWan into an engine, the estimator refit, or the
8 GB tier's qualification.

---

## 1. WHY ARM C WAS CUT, AND WHAT JUST CHANGED

`run_video_arm_bakeoff.py:303-307` cuts arm C with this reason:

> Its licence PASSES (apache-2.0 at both levels) but no ComfyUI base graph exists
> at any priority -- the weights are Diffusers layout and the 3-step DMD schedule
> (timesteps 1000,757,522) is not ordinary `KSampler steps=3`.

**Half of that is now false.** A GGUF repack exists:
`Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF`, apache-2.0, holding
`FastWan2.2-TI2V-5B-q6_k.gguf` (4,210,247,200 bytes) and
`FastWan2.2-TI2V-5B-q8_0.gguf` (5,412,844,128 bytes). GGUF is exactly the format
`UnetLoaderGGUF` already loads for arm A, and
`C:/ComfyUI-Models/diffusion_models` is already on the `diffusion_models` key in
`scripts/_otr_headless_model_paths.yaml`. So the weights-layout half of the cut
reason is gone.

**The schedule half is not.** That is what this spec is for.

Provenance caveat, stated once and carried into the receipt: the repack has ~358
downloads and a 31-byte README with no conversion notes, against 366K downloads
on the official `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`. Per the arm D
precedent, an unresolved provenance question **gates SHIPPING, not measuring** --
so we measure it and record source repo, file size and our own SHA-256.

## 2. THE GRAPH DELTA -- genuinely one ArmSpec row plus one graph file

Arm C's graph is `scripts/bench_graphs/arm_a_wan_ti2v_gguf.json` with exactly two
changes:

- node `1` `UnetLoaderGGUF.unet_name`: `Wan2.2-TI2V-5B-Q5_K_M.gguf` ->
  `FastWan2.2-TI2V-5B-q6_k.gguf`
- node `9` -- **the open question.** Arm A is
  `KSampler{seed 42, steps 30, cfg 5.0, euler, simple, denoise 1.0}`.

Everything else is identical and must stay identical, because the whole value of
this arm is that it differs from arm A in the model and the schedule and nothing
else: same encoder (`umt5-xxl-encoder-Q5_K_M.gguf`), same `wan2.2_vae.safetensors`,
same `Wan22ImageToVideoLatent`, same still `c02_466a19906ccb.png`, same
`VAEDecodeTiled 256/64/16/8`, same probes, same seed, same canvas, same 17/49/81
ladder.

## 3. THE BLOCKING QUESTION

FastWan is DMD-distilled. Its published inference contract is **3 steps at
timesteps 1000, 757, 522**, i.e. normalised sigmas `[1.000, 0.757, 0.522]` then 0.

`KSampler{steps: 3, scheduler: simple}` on a flow-matching model produces a
LINEAR ramp `[1.000, 0.667, 0.333]` -> 0, and `ModelSamplingSD3(shift)` warps it
by `s' = shift*s / (1 + (shift-1)*s)`.

**A single shift cannot reproduce the DMD schedule.** Solving the first interior
step for the target:

    shift * 0.667 / (1 + (shift-1)*0.667) = 0.757   ->   shift = 1.556

and that same shift puts the second interior step at

    1.556 * 0.333 / (1 + 0.556*0.333) = 0.437       (target: 0.522)

So the shift knob is one degree of freedom against two constraints. **Please
check this arithmetic -- if it is wrong, the whole spec collapses into "just set
shift and go", which would be the best possible outcome.**

**Q1 -- ANSWERED BY GROUNDING, now a design to ATTACK rather than an open
question.** Core ComfyUI ships `ManualSigmas` in
`ComfyUI/comfy_extras/nodes_custom_sampler.py`. No custom node is needed:

    node_id  = "ManualSigmas"          category "model/sampling/sigmas"
    inputs   = String "sigmas", default "1, 0.5", multiline=False
    outputs  = SIGMAS
    execute  = re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas) -> [float] -> FloatTensor

so the literal string `"1.0, 0.757, 0.522, 0.0"` parses to exactly the DMD
schedule. `SamplerCustom` (same file) takes
`model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas,
latent_image` and returns `(output, denoised_output)`, so its slot 0 is a drop-in
for the `KSampler` LATENT that node `10` already consumes.

**Proposed replacement for node `9`, three nodes for one:**

    "9a": KSamplerSelect { sampler_name: "euler" }                 -> SAMPLER
    "9b": ManualSigmas   { sigmas: "1.0, 0.757, 0.522, 0.0" }      -> SIGMAS
    "9" : SamplerCustom  { model: ["2",0], add_noise: true,
                           noise_seed: 42, cfg: 1.0,
                           positive: ["4",0], negative: ["5",0],
                           sampler: ["9a",0], sigmas: ["9b",0],
                           latent_image: ["13",0] }                -> LATENT

`SamplerCustom` has no `denoise` input, which is correct here: the sigma list IS
the schedule.

**In-tree precedent, so this is not a new pattern for this repo:**
`nodes/_otr_video_engines/eng_ltx_av.py:299-312` already ships a `manual_sigmas`
recipe switch that injects a fixed `LTX_DISTILLED_SIGMAS` ladder for its
distilled recipes and falls back to `LTXVScheduler` otherwise. Arm C is the same
shape applied to Wan.

**Q1a -- THE THING THIS SPEC ORIGINALLY MISSED, and the likeliest real defect.**
Arm A wires `ModelSamplingSD3 { shift: 5.0 }` at node `2`, and node `2` is what
feeds `model`. If arm C keeps `shift: 5.0` while ALSO supplying absolute sigmas,
the shift still governs the sigma -> timestep mapping the model sees, so the
explicit `[1.000, 0.757, 0.522]` would land on timesteps that are NOT
1000/757/522 and the DMD contract is silently broken -- a wrong render with no
error, which is the exact failure mode this bench forbids. **Confirm or refute:
arm C needs `ModelSamplingSD3 { shift: 1.0 }`, or node `2` bypassed entirely with
`UnetLoaderGGUF` feeding `SamplerCustom.model` directly.** Say which, and say how
to verify it from the server log rather than by eyeball.

**Q1b.** `ManualSigmas` is declared `is_experimental=True`. Does that carry any
risk for a pinned bench graph -- rename, signature change, removal? Is
`KJNodes CustomSigmas` / `FloatToSigmas` the more stable choice, given KJNodes is
installed?

**Q2. If no exact path exists, what is the most honest approximation**, and how
would we know it is wrong? A distilled model given the wrong schedule usually
does not error -- it renders something plausible-but-degraded, which would hand
this bench a VRAM number from a bad render. That is the failure mode this project
forbids. Propose the CHEAPEST discriminator: what would a wrong-schedule 17-frame
clip look like next to arm A's, and can it be caught without a human eyeball?

**Q3. Does the DMD contract also pin cfg?** Distilled models are normally
cfg 1.0 with the negative prompt inert. Arm A runs cfg 5.0 with a real negative.
If arm C must run cfg 1.0, the negative-prompt node is dead weight in the graph
and the two arms differ in one more axis than intended -- say so explicitly, and
say whether the negative CLIPTextEncode should stay wired (for graph symmetry) or
be cut (for honesty).

## 4. THE HEADROOM PROBLEM -- likely fatal, and worth knowing early

Measured 2026-07-31 (`docs/GO_FORWARD_PLAN.md`, "MEASURED -- the 8 GiB-clamped
video bench"): arm A at `Wan2.2-TI2V-5B-Q5_K_M.gguf` (3.55 GB on disk) peaks at
a delta of 6563-6568 MiB against a **7168 MiB** bar -- about **600 MiB of
headroom**. The same bench established that frames and pixels are both nearly
free at this scale, so the cost is essentially resident weights.

`FastWan2.2-TI2V-5B-q6_k.gguf` is **4.21 GB, i.e. ~660 MB heavier than the file
arm A fits with.** Naively that consumes all the headroom and then some.

**Q4. Is arm C dead on arrival at q6_k?** The repack ships only q6_k and q8_0 --
there is no Q5_K_M, which is the quant the incumbent needs to fit. If the
prediction is that q6_k misses the bar, is the right move to (a) measure it
anyway and record the miss, (b) quantise Q5_K_M ourselves from the official
Diffusers weights, or (c) something else? Note that (b) is a real conversion
project, not a row in a table, and would need its own receipt.

## 5. CONSTRAINTS THAT DO NOT BEND

- **No engine module is touched.** Arm C is an isolated bench graph under the
  CLAUDE.md s0A carve-out (O6), pinned by SHA-256 in the campaign manifest, same
  as arms A / B-partial / D. `wan_ti2v` stays sacred.
- **No fallbacks, no silent degrade.** If the schedule cannot be expressed, the
  arm must refuse to run, not run approximately and report a number.
- **The bench grades on `peak_delta_mib`** against each cell's own desktop
  baseline, bar 7168 MiB = 8192 - 1024 display allowance. Arm C changes nothing
  about grading.
- **A blocked arm must not add dead branching** (spec 1, four-arm bench). If arm
  C stays blocked after this review, it gets no `ArmSpec` and no graph file --
  only an updated cut reason.
- Licence: apache-2.0 at both levels, already verified. Do not re-litigate it.

## 6. WHAT I WANT BACK

Ranked, concrete, and grounded in what is actually on this disk:

1. A verdict on the arithmetic in section 3. Right or wrong.
2. The exact node graph for Q1 -- class_types and wiring, or a clear "not
   expressible without <named custom node>".
3. A yes/no on whether to build arm C now, given Q4.
4. Anything in this spec that is wrong.

If the honest answer is "leave arm C cut and put FastWan in the research brief
instead", say that -- `docs/2026-07-31-PROBLEM-STATEMENT-under-8gb-still-to-video.md`
question B is already written to receive it.
