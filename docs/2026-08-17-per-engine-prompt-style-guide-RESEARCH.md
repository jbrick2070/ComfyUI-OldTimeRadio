# Per-engine prompt style guides -- the research prompts

Purpose: collect a SHORT, deployment-specific prompting directive for each
local engine, store it beside that engine, and let the prompt writer act on it.

## The schema (decided 2026-08-17)

Two fields per engine. The split is the point.

| field | cap | who reads it |
|---|---|---|
| `prompt_style_directive` | **240 chars, hard** | the local writer model -- the ONLY part that ever reaches an LLM or a prompt |
| `prompt_style_notes` | uncapped | humans only. Never injected, never sent to a model. |

**Why 240.** It mirrors `_LTX_MOTION_PROMPT_MAX` already in `render_driver.py`,
and it is roughly 4-6 concrete rules. Our own measured evidence says that is the
ceiling where a small local model complies reliably: the kinetic motion clause
(bounded instruction, 130-char validated output, deterministic fallback) ran
`generated: 6, invalid: 0` on a live leg, while F2 -- a many-rule judgment task
on the same class of model -- swung 3/6 then 1/6 on IDENTICAL fixtures and
ships disabled. Bounded and validated works; open-ended judgment does not.

**Rules for the directive text**
1. Imperative, concrete, checkable. "Lead with camera movement" not "be cinematic."
2. No adjectives about quality ("good", "beautiful") -- unactionable.
3. Never contradict a visual style pack. Style is the pack's job, not the engine's.
4. It describes HOW to phrase, never WHAT the scene contains.
5. If a rule cannot be stated in under ~50 chars it belongs in the notes.

**Adoption gate.** A directive is a hypothesis until measured. Adopt only after a
before/after on `scripts/otr_talking_radio_probe_eval.py` at the same seed -- the
P4 probe measured articulation collapsing 4.15 -> 1.18 from a prompt-register
change on this very lane. Store it, measure it, then enable it.

**Offline rule.** Author the string ONCE and store it. Never fetch at runtime --
renders stay reproducible and offline-first.

---

## The reusable research prompt

Paste this, substituting the block for the engine you want. Ask for exactly the
two fields.

> I run a local ComfyUI pipeline on a 16 GB RTX 5080 (Windows, torch 2.10,
> CUDA 13, sm_120, SageAttention + SDPA; NO Flash-Attention 2). Everything is
> offline and local -- no cloud APIs.
>
> I need a prompting style guide for ONE specific model as I actually run it,
> not generic advice. My exact configuration is:
>
> <<ENGINE BLOCK -- paste from below>>
>
> Give me exactly two things:
>
> 1. `prompt_style_directive` -- **240 characters MAXIMUM**, imperative and
>    concrete, the 4-6 highest-value rules for phrasing a prompt for THIS model
>    at THIS configuration. Cover positive-prompt phrasing and, only if the
>    negative is live at my cfg, negative-prompt phrasing. Every rule must be
>    checkable by reading a prompt. No quality adjectives. It must describe HOW
>    to phrase, never WHAT the scene contains. Count the characters and stay
>    under 240.
> 2. `prompt_style_notes` -- unlimited length: the full reasoning, the failure
>    modes, anything model-specific I should know (token/attention behaviour,
>    whether it ignores negatives, phrasing that reliably breaks it, ordering
>    effects, how it handles motion or camera language, length sensitivity).
>
> Be explicit about anything UNIQUE to this model versus its family. If a common
> piece of advice does NOT apply at my configuration, say so and say why. If you
> are unsure whether something holds for the local weights rather than the
> hosted version, mark it uncertain rather than guessing.

---

## Engine blocks -- the real shipped configuration

Facts below are read from the engines in this repo. A negative prompt is INERT
at cfg 1.0 (no classifier-free guidance branch), which is why several blocks
tell the model not to bother with negative advice.

### z_image_turbo -- the stills default
> Z-Image-Turbo, a distilled 6B S3-DiT (Alibaba Tongyi), run in-process in
> ComfyUI as split files: UNETLoader + CLIPLoader (Qwen3-4B text encoder) +
> VAELoader (Flux `ae`), through ModelSamplingAuraFlow (shift 3.0) -> KSampler
> -> VAEDecode. 8 steps, cfg 2.0, euler / normal, 1024x1024, fp8/nvfp4 weights.
> The NEGATIVE IS LIVE at cfg 2.0. Text encoder is Qwen3-4B, not CLIP or T5.

### flux_gen1 -- stills
> FLUX.1-dev fp8 in ComfyUI, 20 steps, **cfg 1.0** with a FluxGuidance embedding
> of 3.5. Because cfg is 1.0 the negative prompt is INERT -- do not give me
> negative-prompt advice, tell me what replaces it. Guidance-distilled.

### lumina_image -- stills
> Lumina-Image 2.0 in ComfyUI, split-file, 30 steps, cfg 4.0, shift 6.0.
> The NEGATIVE IS LIVE at cfg 4.0.

### ltx_video / ltx_8gb -- video, the daily driver
> LTX-Video distilled, run locally in ComfyUI on 16 GB. Default sampler mode is
> "distilled" at **cfg 1.0** (negative INERT; an optional non-default ksampler
> mode runs cfg 3.0). Image-to-video: a rendered still is the conditioning
> anchor and carries the LOOK, so the prompt's job is MOTION, not describing the
> set. Prompt budget is ~188 characters for a normal beat and ~240 for the
> motion clause. 832x480 @ 25fps typical, frames in 8k+1 steps up to 193.

### ltx_av -- video with audio
> LTX-2.3 AV, local GGUF (Q3_K_M) via an env-overridden UNET, in ComfyUI.
> Favoured recipes (sharp_lora, distilled_native, ia2v_canonical) all run
> **cfg 1.0**, so the negative is INERT. The ia2v path is a two-stage lip-sync
> graph where the prompt must support mouth motion synced to supplied audio.

### wan_i2v -- video
> Wan 2.2 image-to-video, local in ComfyUI, **cfg 3.5**. The NEGATIVE IS LIVE.

### wan_ti2v -- video
> Wan 2.2 text+image-to-video, local in ComfyUI, **cfg 5.0**. The NEGATIVE IS
> LIVE and cfg 5.0 is high, so negative phrasing matters more than usual.

### fastwan_8gb -- video, low VRAM
> FastWan on 8 GB, one forward pass per step with NO unconditional branch
> (**cfg 1.0**), so the negative is INERT -- tell me what replaces it.

### humo -- talking portraits
> HuMo 14B (and a 1.7B tier) for audio-driven talking portraits, local in
> ComfyUI. The 14B and 1.7B portrait tiers run **cfg 1.0** (negative INERT); the
> 1.7B landscape tier runs cfg 2.5 (negative LIVE). Lip-sync fidelity is the
> priority and long prompts are suspected of drowning the speech tokens.

### minimax_h3 -- video
> MiniMax H3 video. It takes NO negative prompt at all -- positive-only advice.

---

## Where the answers go

`prompt_style_directive` and `prompt_style_notes` live as constants beside the
engine that owns them, the same way `_HYGIENE_NEGATIVE` now lives in
`z_image_turbo` and the style negative lives in the pack: one authority, no
drift, nothing to keep in sync.

**Not yet wired.** Storing them is safe and free. ACTING on them is a separate,
measured change, gated on the probe above -- and it must never override a visual
style pack, which owns style; the directive owns phrasing only.
