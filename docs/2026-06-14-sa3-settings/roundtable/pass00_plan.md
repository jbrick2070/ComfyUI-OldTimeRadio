# SA3 Settings Bake-In — Problem Statement for Roundtable (2026-06-14)

**Goal:** recommend the BEST fixed default values for the Stable Audio 3 (SA3) **small music**
model so OTR's sci-fi old-time-radio music cues sound good with NO operator tweaking. We will BAKE the
converged values in as the code defaults. Give concrete numbers + a one-line rationale each.

## Use case (be specific)
- Show: a sci-fi **old-time-radio drama** ("Signal Lost"). Music is INSTRUMENTAL underscore only
  (no vocals). Vintage/cinematic, eerie, period-flavored (often 1950s atomic-age sci-fi).
- Three cues per episode, fixed lengths: **opening 12s, closing 8s, interstitial 4s.**
- Engine: **Stable Audio 3 small music** (`stable_audio_3_small_music.safetensors`, Comfy-Org), run via
  ComfyUI's NATIVE graph: `CheckpointLoaderSimple -> CLIPTextEncode(pos/neg) ->
  ConditioningStableAudio(seconds_start, seconds_total) -> EmptyLatentAudio(dur) -> KSampler ->
  VAEDecodeAudio`. Box: RTX 5080 16GB, 100% local.
- We already added (BUG-408): a genre/instrument prompt anchor, a real negative prompt, and a per-cue
  `seconds_start` within a structural `seconds_total` CONTEXT while the LATENT stays exactly the cue
  length (so clip length + seed determinism are unchanged).

## Current defaults (to confirm or change)
- `OTR_SA3_STEPS = 100`
- `OTR_SA3_CFG = 6.0`
- `OTR_SA3_SAMPLER = dpmpp_3m_sde_gpu`, `OTR_SA3_SCHEDULER = exponential`
- `OTR_SA3_CONTEXT_S = 30.0` (the `seconds_total` structural context; per-cue `seconds_start` =
  intro→0, outro→`context-dur`, else→`(context-dur)/2`)
- negative prompt: `vocals, singing, speech, spoken words, lyrics, voiceover, crowd noise, harsh
  clipping, digital distortion, muddy mix, out of tune, low quality`
- determinism: a single seeded KSampler pass per cue (seed is the carrier).

## Questions (give EXACT values to bake in)
1. **CFG:** is 6.0 right for SA3 small instrumental cues, or should it be ~7 (SA3's common default)?
   Trade-off: higher = stronger prompt adherence but risk of harshness/over-saturation. Recommend a value.
2. **Steps:** is 100 worth it for the SMALL model, or do 50-60 dpmpp-3m-sde steps look/sound identical
   (faster, same quality)? Recommend a value that maximizes quality without pointless cost.
3. **seconds_total context:** SA Open is trained up to ~47s. For SHORT cues (4-12s) rendered as a SLICE
   of a `seconds_total`-conditioned piece, is 30s the right structural context, or larger (40-47s) /
   smaller? Note a 4s interstitial is a very small slice of 30-47s — does too-large a context make short
   cues sparse/incoherent? Recommend a value (one value, or per-cue if strongly justified).
3b. **seconds_start placement:** is intro→head / outro→tail / interstitial→middle the right mapping,
    given `seconds_total`? Any refinement?
4. **Sampler/scheduler:** keep `dpmpp_3m_sde_gpu` + `exponential`, or is another combo
    (e.g. `dpmpp_2m` / `karras`) better for Stable Audio? Recommend.
5. **Negative prompt:** is the list above good, or add/remove terms for clean instrumental sci-fi
   (avoid killing eerie/tape texture — do NOT blanket-ban "dissonant")? Give the final string.
6. Anything else that would most improve perceived musicality of short SA3 cues (e.g. a specific genre
   phrasing, BPM hint, or a denoise value)?

## Constraints
- These become FIXED code defaults (still env-overridable, but the operator should never NEED to set
  them). 100% local, no new deps, ≤14.5GB. Determinism: single seeded pass (no best-of-N).
- Keep the latent length = cue duration (do NOT propose render-longer-then-trim).
- Out of scope: changing the engine, the master mux, or the prompt-composition contract.

## Deliverable
A single recommended set: `STEPS`, `CFG`, `SAMPLER`, `SCHEDULER`, `CONTEXT_S`, the seconds_start
mapping, and the final negative prompt — each with a one-line rationale. Claude is the judge and will
bake the converged values in.
