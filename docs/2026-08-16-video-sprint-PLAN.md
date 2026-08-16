# Video engines sprint -- PLAN (item 1.2, reconciled)

**Status:** DRIVER ANCHOR for a full `kibitz-plugin:kibitz` arc. Grounded
against the real Windows tree at HEAD `5662cd16`; every asset figure below was
measured off disk this session, not quoted from the proposal.

**Authority:** `docs/2026-08-16-video-lab-proposal-GROUNDED.md` plus the lab's
answers to `docs/2026-08-16-TO-THE-LAB-reconcile-video-proposal.md`. The lab
WITHDREW three rows on receipt of evidence -- both H3 engines (they had missed
the `public_engines.py` alias mapping), the 848x480 legacy-resolution claim
(LTX upstream defaults, not OTR), and HuMo Clamp-13 (peak-VRAM-only evidence,
no abort-rate data, refused by the recipe rule). **Three items survive.**

## 1. BUILD: the `ltx_distilled` sprint/draft lane

The one real gap. There is no `ltx_distilled` public engine id; the weights
have been on disk the whole time.

**Verified on disk this session** (`C:\ComfyUI-Models\`):

| asset | path | size |
|---|---|---|
| distilled transformer fp8 | `diffusion_models\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` | 23.49 GB |
| distilled LoRA | `loras\ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.08 GB |
| distilled GGUF ladder | `unet\distilled-1.1\` Q4_K_M 13.22 / Q3_K_M 9.90 / Q2_K 7.39 | on disk |
| dev Q3 (what the SHIPPED LTX-AV lane loads) | `unet\ltx-2.3-22b-dev-Q3_K_M.gguf` | 10.03 GB |

**Starting spec, operator-folded 2026-08-16 -- the lab-proven recipe IS the new
lane's spec:** `res_multistep` sampler, `simple` scheduler, 20 steps, denoise
1.0, 832x480 @ 25 fps, frames `8k+1` up to 193 (7.72 s), first/last-frame
chaining. Measured 13.11 GiB on the 16 GB card. Rides the LTX sage-free boot
token.

**No sampler change to any SHIPPED lane is proposed, and none would be accepted
without the operator reversing the recipe rule.** This is a new lane; its recipe
is its own.

**The wiring is a bijection, and that is the sharp edge.**
`nodes/_otr_shared/public_engines.py` maps public id -> internal id with an
**import-time bijection assert** (`_PUBLIC_ENGINES`, `:35-141`). Two public ids
on one internal id trips it at import -- i.e. the whole node pack fails to load,
not a test. So the lane needs: one new internal engine + adapter, one
`_PUBLIC_ENGINES` row, one description row (`:214-230`), and nothing that
aliases an existing internal id.

**Naming needs a decision, not a guess.** The convention is
`<family><ver>_<measured-vram-bucket>_<capability>`, and the bucket is a
MEASURED token, never a quality claim -- lane 8's comment refuses to let `low`
mean "runs on an 8 GB card", and lane 9 assigns `high` at 13.3 GiB net. At
13.11 GiB the distilled lane sits in the same bucket as `ltx23_high_video`,
which already owns the obvious name. Candidates: `ltx23_high_distilled`,
`ltx23d_high_video`. **Panel question 1.**

## 2. BUILD (additive): P1's verbatim-dialogue half

Inject the actual spoken line into the video prompt on the audio-in lanes so a
multimodal encoder can predict visemes. **Rewrites nothing, rejects nothing,
blocks nothing** -- so the no-rewrite ruling is untouched by construction.

## 3. DECIDED (driver, on the operator's delegation): P1's kinetic half goes on
## the VIDEO-prompt path

Additive motion language in `_otr_line_composer` / the motion clause. **No
damping-word stripping anywhere.**

**This is the whole reason the item is buildable.** The proposal routed the
kinetic half into `nodes/otr_meta_brief_image_prompt.py`, whose live design
contract reads *"No Python vocabulary or overlap classifier can reject,
rewrite, or block the prompt."* Stripping damping words IS a Python vocabulary
rewrite of the prompt. That ruling survived the 2026-08-05 item-8 four-round
campaign, where a Codex proposal and an agy proposal to do exactly this were
BOTH overruled at r4, and *"update the stale comment"* was explicitly rejected
as a resolution -- editing away a deliberate decision to permit a new classifier
is the quiet reversal the directives exist to prevent.

**Revisit condition, named in advance:** if measured damping persists after
video-path injection, the still-path stripping question goes BACK TO THE
OPERATOR as a ruling change. It is never slipped in.

## 4. The Q5_K_M quant -- a SIBLING VARIANT, never a swap

Operator 2026-08-16: *"there are no defaults, we have multiple video lanes...
I'm going to ship w/ multiple JSONs."*

The shipped LTX-AV lane loads `ltx-2.3-22b-dev-Q3_K_M.gguf` through the
`OTR_LTX_AV_UNET` env override. So **testing Q5 is zero-code** (set the env, run
a leg, measure); **productizing it is a sibling variant JSON / profile**, never
a swap inside the shipped lane. No recipe-rule collision by construction.

**Source VERIFIED by direct HF listing, not by a search result:**
`unsloth/LTX-2.3-GGUF / ltx-2.3-22b-dev-Q5_K_M.gguf`, **16.07 GB** (a UD-Q5_K_M
sibling exists at 18.3 GB). `QuantStack/LTX-2.3-GGUF` has NO Q5 file; `city96`
is v0.9.1, the wrong architecture. **It is NOT on disk** -- confirmed this
session -- so this item carries a 16 GB download.

**16.07 GB means dev-Q5 is a 16 GB-card weight with offload, not an 8 GB one.**
The lab's "8 GB baseline upgrade" framing needs its 4060 peak-VRAM receipt
before anyone repeats it.

## 5. What is NOT built

Nothing on P2 -- it is already shipped at the same recipes, and re-adding those
lanes would trip the bijection assert. HuMo Clamp-13 (0.22 GiB, refused by the
recipe rule absent an explicit operator reversal). Any sampler/step/resolution
change to a shipped lane. Any damping-word stripping. The multi-GPU
learned-upscale stage, which is CLOSED.

## 6. Gates

Full suite (baseline **10529/110/1**), Bug Bible (**20/26/3 at 284**),
`build_variants.py --check` (**50/0**), AST parse on touched `.py`, a JSON
round-trip + link/widget audit if `workflows/otr_canonical.json` is touched,
Sonnet QA on the finished diff BEFORE the push.

**Acceptance legs need the LTX boot lane, so they PAUSE THE SOAK** (kill
`otr_gpu_soak_matrix` + any in-flight `otr_canonical_api_run` selectively by
CommandLine, never a blanket python kill -- MCP lives there -- then resume).

## 7. Questions for the panel

1. What should the distilled lane be called, given `ltx23_high_video` already
   owns the natural name and the VRAM token is measured rather than chosen?
2. Does a new engine adapter need a capability-profile row
   (`config/audio_engine_profiles.yaml` has the audio analogue; what is the
   video equivalent, and does the variants gate at 50/0 move when a lane is
   added)?
3. Is the verbatim-dialogue injection safe on every audio-in lane, or does any
   of them cap prompt length in a way that would silently truncate the scene
   description to make room for the line?
4. Is `OTR_LTX_AV_UNET` read once at import or per render? If it is import-time,
   "set the env and run a leg" is not zero-code -- it needs a server restart,
   which changes how the Q5 measurement is scheduled against the soak.
