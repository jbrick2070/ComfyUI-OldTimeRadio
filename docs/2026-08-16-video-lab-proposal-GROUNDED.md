# The video-lab video-engine proposal, GROUNDED against the shipped tree

Source: `vram-recipe-lab/docs/2026-08-14-PROPOSAL-otr-video-engine-updates.md`
(QA-corrected, empirical, RTX 5080 16 GB + physical RTX 4060 8 GB).
Grounded here 2026-08-16 against the live registry and the running server.

**Headline: most of it is ALREADY SHIPPED, at the same recipes.** The proposal
reads as five engine additions; four are existing lanes under different public
ids. Only ONE item is genuinely missing.

## P2 "8 GB additions" -- ALREADY IN PRODUCTION, do not re-add

| Proposal id | Proposal figure | Shipped public id | Shipped figure |
|---|---|---|---|
| `h3_lowvram` | 7.21 GiB @ 864x480 | `h3_low_video` | 7.3 GiB @ 864x480 |
| `h3_audioin_lowvram` | 7.00 GiB @ 864x480 | `h3_low_audio_in` | 6.9-7.2 GiB @ 864x480 |
| `ltx_audio_hq` | 7.36 GiB @ 1024x576 | `ltx23_low_audio_in` | **7.36 GiB warm @ 1024x576x193** |

The `ltx_audio_hq` row is the clearest case: the proposal's "upgrade LTX
low-VRAM to 1024x576 HQ GGUF" describes the resolution the lane ALREADY
ships, and the VRAM figure matches to the digit. Weights are on disk
(`minimax_h3_fl2va_*`, `minimax_h3_ref2va_*`).

The public ids differ from the internal engine ids by design --
`nodes/_otr_shared/public_engines.py` maps `h3_low_video` ->
`minimax_h3_video` and `h3_low_audio_in` -> `minimax_h3_audio_in`, with an
import-time bijection assert. Reading `registry.py` alone makes the lanes look
absent; they are not.

## P3 HuMo "Clamp-13 diet" -- ALREADY DIETED, and the delta is not worth a
## recipe change

Proposal: cage HuMo 1.7B from 15.23 GiB to 12.84 GiB. Shipped: the
`humo*_high_audio_in_wide` lanes already run **13.06 GiB warm at 832x480x97
on the humo_diet boot**. The remaining delta is 0.22 GiB.

**Standing operator directive (CLAUDE.md): "The recipes are not on the table
... No VRAM, speed or cap finding justifies a recipe change; measurement runs
the SHIPPED recipe unchanged."** A 0.22 GiB saving is exactly the kind of
finding that rule exists to refuse. NOT RECOMMENDED without an explicit
operator reversal.

## THE ONE REAL GAP: `ltx_distilled` has no lane

* Weights ARE downloaded:
  `C:\ComfyUI-Models\diffusion_models\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`
* The distilled LoRA is already referenced in `registry.py` model_requirements.
* But there is **no `ltx_distilled` public engine id** -- confirmed by grep of
  `public_engines.py` and by the live `/object_info` dropdown.

That is the 13.8-second sprint/draft lane, ~29.5x faster than WAN, measured at
13.11 GiB on a 16 GB card. It is the only item in the proposal that is real,
missing, and unblocked by any ruling. **If one thing gets built, build this.**

## P1 prompt-engine upgrade -- HALF BUILDABLE, HALF BLOCKED BY A LIVE RULING

* **BUILDABLE: the verbatim-dialogue / viseme half.** Injecting the actual
  spoken line into the prompt so a multimodal encoder can predict visemes is
  ADDITIVE. It rewrites nothing and rejects nothing.
* **BLOCKED: the "strip damping words + inject kinetic verbs" half.** The
  proposal routes it into `nodes/otr_meta_brief_image_prompt.py`, whose
  docstring at :1706-1707 states: *"No Python vocabulary or overlap classifier
  can reject, rewrite, or block the prompt."* Stripping damping words IS a
  Python vocabulary rewrite of the prompt. That ruling survived the 2026-08-05
  item-8 four-round campaign, where BOTH a Codex proposal and an agy proposal
  to do this were overruled at r4, and "update the stale comment" was
  explicitly rejected as a resolution.

  The proposal's own QA appendix caught that its integration points were
  invented and corrected them to the real functions -- but did not notice that
  the corrected target is the one file where the rewrite is forbidden.

  **Two legitimate routes, both needing an operator call:** narrow the ruling
  for this case, or apply the kinetic language on the VIDEO-prompt path
  (`_otr_line_composer._build_user_prompt`) which the ruling does not cover.
  The second needs no reversal and is the cheaper path if it satisfies the
  motion goal.

## Recommended order

1. `ltx_distilled` lane (real gap, weights present, no ruling in the way).
2. P1's verbatim-dialogue half (additive, improves lip sync on the audio-in
   lanes we already ship).
3. Operator ruling on P1's kinetic half -- narrow, or reroute to the video
   prompt.
4. NOTHING on P2. It is done.
5. HuMo Clamp-13 only if the operator explicitly reverses the recipe rule for
   0.22 GiB.

## Deferred by the operator (2026-08-16)

Full-workflow + TTS + image-model + upscaler testing on the physical 4060 8 GB
happens on the main repo AFTER the low-VRAM work is settled. Worth knowing
when scheduling: per the table above, the low-VRAM lanes are already shipped,
so that gate is closer than it looks.
