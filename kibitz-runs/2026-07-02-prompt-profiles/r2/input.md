# Cloud row prompt/param profiles -- draft for kibitz (2026-07-02)

Goal: per-row prompt templates + parameter/seed policy so every Comfy
credit buys maximum quality. Grounded in (a) the PINNED input schemas
(partner_nodes.yaml, 14 rows), (b) the pricing stamp (PRICING.md), and
(c) canonical model behavior. These profiles feed the adapters at
build time; the writer's finish_visual_prompt chain stays the CONTENT
source -- profiles are the WRAPPER (style tail + params), consistent
with the existing prompt-profile pattern (modern/otr_1940s_v1 in the
LLM catalog; portrait_mint_3d/tin_toy_v1 in the formats plan).

## Global policies (all rows)

- SEED: every request carries an explicit seed from the C7
  reproducibility pattern (OTR_CAST_SEED/OTR_STYLE_SEED-style env
  overrides; per-episode RNG otherwise) WHERE the pinned schema says
  seed_supported -- true for all image + video rows; ElevenLabs TTS
  pins `seed: INT` as well (determinism best-effort per vendor docs).
  Seeds land in RequestCacheKey (cache correctness) and the ledger.
- NEGATIVE/AVOID terms ride the profile, not the writer: modern
  objects, anachronisms ("smartphone, computer, LED, plastic"), text
  gibberish (except Ideogram, where text is the point).
- PERIOD TAIL (shared, appended to every stills/video prompt):
  "1946 radio-theater era, warm tungsten key light, subtle film grain,
  period-correct wardrobe and props" -- the existing get_era_tail
  machinery is the source of truth; profiles reference it, never fork
  it.
- RESOLUTION: request the ROLE CANVAS exactly (e.g. 1472x832
  landscape); never upscale-by-prompt. Aspect params set explicitly
  where the schema exposes them.

## Voice -- cloud_elevenlabs_tts / _flash (+ voice_selector AUX)

Pinned inputs: text, voice (ELEVENLABS_VOICE via AUX selector), model
(DYNAMICCOMBO), stability FLOAT, language_code, apply_text_
normalization COMBO, output_format COMBO, seed INT.
- model: flash row -> `eleven_flash_v2_5`-class (latency-cheap, SAME
  price); tts row -> `eleven_multilingual_v2`/v3-class (richest
  prosody). Price is FLAT -- pick by QUALITY need only.
- stability: 0.45 announcer (consistent broadcast cadence w/ life);
  0.30-0.38 characters (more expressive variation); NEVER >0.6 (flat,
  robotic) or <0.2 (drift/artifacts). Per-line delivery vectors may
  nudge +/-0.08 within [0.2, 0.6].
- output_format: highest PCM/sample-rate offered (44.1kHz) -- the
  canonicalizer owns loudness; never mp3 low-tier (transcode loss).
- apply_text_normalization: OFF ("auto" only if numbers misread) --
  the script writer already normalizes; double-normalization mangles
  1940s idiom ("$2.50" etc. are pre-spelled by the writer).
- language_code: "en"; period diction comes from the SCRIPT, not the
  voice layer.
- text hygiene: per-line only; strip stage directions (existing
  scrubs); ElevenLabs reads punctuation as prosody -- keep the
  writer's em-dashes/ellipses, drop bracketed cues.

## Stills

### cloud_recraft (RecraftTextToImageNode) -- LOW $0.04
- style: realistic_image family; the v3 API responds strongly to
  photographic-style keywords -- lead with medium ("press photograph,
  silver gelatin print") then subject, then era tail.
- Use for: scene b-roll stills, set dressing, wide establishing.
- Avoid: faces needing cross-shot identity (that is nano_banana_2's
  job) and any in-image TEXT (Ideogram's job).

### cloud_ideogram_v4 (IdeogramV4) -- $0.043-0.13 by rendering_speed
- THE text-renderer: posters, marquee cards, newspaper front pages,
  case-file labels, evidence-board clue notes (F1 dressing layer).
- rendering_speed policy: TURBO for iteration/drafts + b-roll copy;
  DEFAULT for episode deliverables; QUALITY only when text is the
  hero shot (title cards). Adapter default: DEFAULT; format engines
  may request TURBO for dressing bulk.
- Prompt shape: put the EXACT text in quotes first ('a yellowed
  newspaper, headline "MARTIANS OVER MERCER COUNTY"'), then medium
  ("1946 letterpress, aged newsprint"), then era tail. Keep quoted
  strings <= 8 words per surface (legibility cliff).

### cloud_seedream_2 (ByteDanceSeedreamNodeV2) -- LOW $0.035
- Cheapest stylization; strong graphic/poster looks and bold
  compositions. Use for: stylized interstitial art, music-beat
  poster frames, experiments. Weakest claim to photoreal faces --
  keep it off character portraits.

### cloud_nano_banana_2 (GeminiNanoBanana2V2) -- MID ~$0.08
- The CHARACTER CONSISTENCY row: reference-image editing preserves
  identity -- feed the canonical portrait as reference for every
  derived character still (new pose/angle/costume state).
- Prompt: imperative edit language ("same person, now seated at the
  microphone, three-quarter view"), never re-describe the face (the
  reference owns identity; re-description fights it).
- selected_output pinned to the IMAGE slot (returns [IMAGE, STRING,
  IMAGE] -- index 0).

### cloud_flux_pro (Flux2ProImageNode) -- LOCAL-MATCH (price: verify)
- Same model family as local flux_gen1 -> REUSE the existing local
  FLUX prompt conventions verbatim (finish_visual_prompt output works
  unchanged). This row exists for continuity: cloud renders that sit
  next to local renders without a style seam.
- Use for: portrait minting when running cloud-only (portrait_mint_3d
  + tin_toy_v1 profiles run here or on recraft/nano per formats plan).

## Video (profiles recorded now; S3 builds them)

### cloud_kling_avatar / cloud_kling_lipsync
- kling_avatar pinned inputs include image + sound_file (avatar
  GENERATES a talking face); kling_lipsync takes video+audio+
  voice_language (syncs the GIVEN face -- THE row for formats).
- voice_language: "en"; resolution: 720p ALWAYS for lipsync crops
  (the crop is small; 1080p doubles cost for invisible gain --
  26.59 vs 35.45 cr/sec).
- Clip discipline: shortest duration covering the line + 250ms pad;
  never batch multiple lines into one clip (cache granularity +
  per-line captions).
### cloud_seedance_2 (ByteDance2ReferenceNode)
- Reference image = canonical portrait/still; audio-ref = the beat
  slice; prompt describes MOTION ONLY ("slow dolly-in, she leans
  toward the microphone") -- subject identity comes from the
  reference, never the prompt.
### cloud_wan_i2v (mute opt-down)
- init_image = minted still; prompt = motion verbs + atmosphere;
  720p-class output; used only via explicit opt-down or auto-mute
  fallback (LOUD).

## Music

### cloud_sonilo_music (0.5275 cr/sec -- charge scales with DURATION)
- Request EXACT theme lengths (open/close/interstitial specs), never
  round up "for safety" -- 30s over-request = 16 wasted credits.
- Prompt: era + instrumentation + mood + tempo ("1946 radio drama
  opening theme, brass and strings, mysterious, 96 BPM") -- the
  existing Meta-brief music protocol supplies content; profile adds
  era instrumentation guardrails ("no synthesizers, no drum kit").
### cloud_stability_audio (42.2 cr flat/run)
- Flat price -> prefer for LONGER beds (>80s breaks even vs Sonilo);
  Sonilo for short stings.

## Open questions for the kibitz pass

1. Are the stability ranges for ElevenLabs right for 1940s announcer
   cadence, or should announcer sit higher (0.5-0.55)?
2. Ideogram quoted-text length cliff -- 8 words per surface: too
   conservative? too loose?
3. Is 720p-always right for kling_lipsync when the crop gets pasted
   back into a 4K board (F1) -- or should paste-target size drive
   resolution selection per shot?
4. Seedream negative-prompt support (schema shows prompt only --
   verify whether avoid-terms must inline).
5. Anything canonical about Flux2 Pro prompting that DIVERGES from
   flux.1-dev conventions the local lane uses?
