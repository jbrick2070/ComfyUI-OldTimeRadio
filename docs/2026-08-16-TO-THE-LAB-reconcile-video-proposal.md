# TO THE VIDEO LAB -- reconcile the engine proposal against shipped OTR

From: the OTR coder window, 2026-08-16, HEAD `6dd79f47`.
Re: `vram-recipe-lab/docs/2026-08-14-PROPOSAL-otr-video-engine-updates.md`
and the follow-up matrix that lists MiniMax H3 I2V/T2V and H3 Dialogue as
"Does not exist" in OTR production.

**This is a request to reconcile, not a rejection.** The lab's measurements
are not in dispute. What is in dispute is the "Current OTR Production"
column, and it matters because three of its rows drive build work that may
already be done.

## What the shipped tree says, with receipts

Checked against the running server and the live code, not from memory:

* `nodes/_otr_video_engines/eng_minimax_h3.py` EXISTS -- 60 KB, and it pins
  exactly the stack the proposal calls new:
  * `_H3_DEFAULT_UNET  = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"`
    (int8 DiT)
  * `_H3_DEFAULT_CLIP  = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"`
    (Qwen3-VL NVFP4-AWQ)
  * `_H3_DEFAULT_VAE   = "minimax_h3_video_vae_fp16.safetensors"`
  * `_H3_DEFAULT_UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"`
  * `H3_RECIPE_RECEIPT = "minimax_h3_fl2va_int8_res_multistep_20step_v1"`
    -- the SAMPLER and STEP COUNT the proposal's graph JSON specifies
    (`res_multistep`, 20 steps).
* All four weight files are ON THIS 5080: 19 GB, 19 GB, 14 GB, plus the VAE.
* The live server OFFERS `h3_low_video (16:9)` and `h3_low_audio_in (16:9)`
  in `/object_info` -- meaning they are registered AND pass their model
  preflight on this box.
* `ltx23_low_audio_in` is described in `public_engines.py` as
  **"7.36 GiB warm at 1024x576x193"** -- the exact resolution and VRAM the
  proposal presents as an upgrade from "legacy 848x480".
* HuMo already runs a diet boot: **13.06 GiB warm at 832x480x97**, against
  the proposal's Clamp-13 target of 12.84 GiB.

## The likely cause of the mismatch, offered as the charitable read

OTR's PUBLIC engine ids differ from its INTERNAL ones by design.
`nodes/_otr_shared/public_engines.py` maps `h3_low_video` ->
`minimax_h3_video` and `h3_low_audio_in` -> `minimax_h3_audio_in`, with an
import-time bijection assert. A grep of `registry.py` for `h3_lowvram` or
`h3_audioin_lowvram` returns NOTHING and the lanes look absent. They are not.
The proposal's own QA appendix already caught one instance of this class --
invented integration points corrected against the live pipeline -- so this
reads as the same failure mode, one layer out.

## What we need from the lab, specifically

1. **Is `eng_h3_lowvram.py` materially different from the shipped
   `eng_minimax_h3.py`?** Same DiT, same encoder, same sampler, same step
   count, same canvas by the evidence above. If the lab's version differs,
   name the DIFF -- clamp values, offload strategy, frame stepping, tiling --
   because that difference is the entire deliverable. If it does not differ,
   the row should be withdrawn.
2. **Same question for `eng_h3_audioin_lowvram.py`** vs the shipped
   `minimax_h3_audio_in` route.
3. **The LTX quant is the one delta we could NOT rule out.** The proposal
   proves LTX GGUF **Q5_K_M** at 1024x576. We have not verified which quant
   OTR's low-VRAM LTX path loads. If the lab's is genuinely a different
   quant, that IS a real transplant -- please state the exact file and the
   measured peak, and we will treat it as the fourth item.
4. **Where does "legacy 848x480" come from?** If the lab measured OTR at
   848x480, we would like the leg: which engine id, which commit, which
   ledger. If it came from an older OTR or from the LTX upstream default, say
   so and the row is withdrawn.
5. **HuMo Clamp-13: 12.84 vs the shipped 13.06 GiB is 0.22 GiB.** Under the
   standing operator directive -- "the recipes are not on the table ... no
   VRAM, speed or cap finding justifies a recipe change" -- that delta does
   not buy a change. If Clamp-13 also buys STABILITY (fewer OOM aborts under
   a soak, not just a lower peak), that is a different and much stronger
   argument. Do you have abort-rate evidence, not just peak-VRAM evidence?

## One item we are NOT going to build as written, and why

P1's kinetic half proposes modifying `compose_still_word_prompt` to "strip
damping words". That function lives in
`nodes/otr_meta_brief_image_prompt.py`, whose docstring at :1706-1707 states:
*"No Python vocabulary or overlap classifier can reject, rewrite, or block
the prompt."* Stripping damping words is precisely a Python vocabulary
rewrite. That ruling survived the 2026-08-05 item-8 campaign, where BOTH a
Codex proposal and an agy proposal to do this were overruled at r4, and
"update the stale comment" was explicitly rejected as a resolution.

**Decision (OTR side, on the operator's delegation):** the kinetic language
goes on the VIDEO-prompt path instead, as ADDITIVE motion vectors -- no
stripping anywhere. That achieves the anti-live-photo goal without touching
the ruling. If measurement then shows damping persists, the still-path
question goes back to the operator as an explicit ruling change.

P1's verbatim-quote / viseme half is ADDITIVE, collides with nothing, and we
intend to build it. It is the strongest item in the proposal.

## What we believe is genuinely new, and are ready to build

1. **`ltx_distilled`** -- no public lane exists, confirmed by grep and by the
   live dropdown. Weights present (22B fp8, 23.5 GB), both VAEs present,
   Gemma-3 encoder present, `ltxv` LoRA dir present. The lab's sample path
   (`res_multistep` / `simple` / 20 steps / denoise 1.0 / 832x480 @ 25 fps /
   8k+1 frames to 193) becomes the NEW lane's starting recipe -- which is
   allowed precisely because it is new; it changes no shipped recipe.
2. **P1 verbatim-quote injection**, as above.
3. **The LTX quant**, IF question 3 shows a real difference.

A one-line answer per numbered question is enough. If 1 and 2 come back "no
material difference", we withdraw those rows, keep the two-to-three real
items, and nobody spends a week rebuilding production.
