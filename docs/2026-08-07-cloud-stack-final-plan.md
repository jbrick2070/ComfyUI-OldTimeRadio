# Cloud Stack Final Plan -- 2026-08-07

**Decision record for the two all-cloud production profiles:
`otr_cloud_low` and `otr_cloud_hq` (config/profiles/).**
Research basis: 24-agent verified pricing sweep, all figures fetched live
2026-08-07 from vendor pages, adversarially re-verified. Every price below
carries its source. Where two research passes disagreed, the disagreement
and its resolution are recorded.

## The two profiles

Both profiles orchestrate from a CPU-baseline host (any box, `--cpu` launch)
and reuse ONLY engines already registered in this repo. Zero new engine code
was required -- this plan is a wiring decision, not a build.

| Slot | otr_cloud_low | otr_cloud_hq | Engine file |
|---|---|---|---|
| Script | OpenRouter slot-a/b | same | `_otr_model_catalog` lane |
| Voice | `google_tts` (Gemini 2.5 Flash TTS) | same (announcer may pin Pro via `OTR_GOOGLE_TTS_MODEL_ID`) | `nodes/_otr_audio_engines/eng_google_tts.py` |
| Music | `google_lyria` (lyria-3) | same | `nodes/_otr_audio_engines/eng_google_lyria.py` |
| Images | `google_image` (Gemini image) | same | `nodes/_otr_image_engines/eng_google_image.py` |
| Video | `cloud_wan_i2v` (Comfy partner Wan, mute) | `google_veo_video` (Veo 3.1) | `nodes/_otr_video_engines/eng_cloud_video.py` / `eng_google_veo_video.py` |
| Upscale | none (local 5080 if wanted) | none (local) | n/a |

## Per-episode cost (list, before rerolls)

Episode quantities: ~30k in / 6k out LLM tokens, 9,000 TTS characters,
one 30-90 s music bed, 10 stills, 8 video clips x 6 s = 48 s.

| Leg | LOW | HQ | Source (fetched 2026-08-07) |
|---|---:|---:|---|
| Script (flash-class via OpenRouter) | ~$0.01 | ~$0.01 | openrouter.ai (zero inference markup; 5.5% credit-purchase fee) |
| Voice Gemini 2.5 Flash TTS | $0.13 | $0.13 ($0.25 if Pro announcer) | cloud.google.com/text-to-speech/pricing ($0.50/1M text in, $10/1M audio out, 25 audio tokens/s) |
| Music Lyria 3 Pro | $0.08 | $0.08 | ai.google.dev/gemini-api/docs/pricing (updated 2026-08-05; $0.04 Clip 30s / $0.08 Pro to 184s, per song) |
| Images Gemini Flash Image batch x10 | $0.20 | $0.20 | ai.google.dev/gemini-api/docs/pricing ($0.0195/image batch, $0.039 standard) |
| Video | ~$0.40-0.80 (Comfy credits, ~11 cr/5s anchor) | $2.40 (48s x $0.05/s Veo 3.1 Lite 720p) | comfy.org/pricing (anchor, model/settings undisclosed) / ai.google.dev pricing |
| **Total list** | **~$0.85-1.25** | **~$2.85** | |
| **Budget (x ~1.85 accept-rate multiplier)** | **~$1.60-2.30** | **~$5.00** | engineering estimate, NOT vendor data -- measure it |

At 100 episodes/month: LOW ~$85-125 list, HQ ~$285 list.

**The acceptance-rate multiplier is the biggest number in this plan and it is
UNMEASURED.** Every vendor price assumes 8 clips generated = 8 clips used.
At a realistic 50% first-pass video accept the real bill is ~1.85x list.
Settle it empirically: generate ~20 real clips against real beats, count
keepers. That number moves more money than any provider choice.

## Why these picks (and what was rejected)

- **Voice = Gemini TTS, not Chirp 3 HD, not Azure, not ElevenLabs.**
  Chirp is $30/1M with a real recurring 1M-chars/month free allowance --
  but it has NO style/emotion prompt of any kind (Google's own FAQ: fix
  delivery by rewriting punctuation) and cannot do the 1940s announcer.
  Gemini TTS is prompt-steerable and Google's own docs use literally
  "Speak like a 1940s radio news announcer" as the canonical good prompt.
  Azure Neural has express-as styles at $15/1M BUT Microsoft's Code of
  Conduct v4.0 requires synthetic-voice disclosure and prohibits "graphic
  violence and gore" -- a direct collision with the unfiltered
  Macbeth/King Lear lanes (operator no-guardrails directive 2026-08-03).
  ElevenLabs multilingual_v2 is $0.90/episode -- 7x Gemini TTS for the
  same 9,000 chars; the swap saves ~$77-90/month at 100 eps.
- **Chirp 3 HD stays available as a future flat-read tier** (30 voices,
  14F/16M, x4 English locales = 120 IDs; free under 1M chars/month) for
  continuity copy and station IDs. It is a reader, not an actor.
- **Music = Lyria 3.** $0.08/song flat, self-serve
  (`POST generativelanguage.googleapis.com/v1beta/interactions`, API key
  only), 184 s Pro ceiling covers any bed in one call. Suno: no official
  API (partner-gated Typeform). Udio: downloads disabled platform-wide
  post-UMG-settlement -- the file cannot leave the site. Both unusable.
- **Video LOW = `cloud_wan_i2v`** (already-shipped Comfy partner lane,
  wan2.7-i2v, 720P, watermark:false, 2-15 s). Cheapest arm with readable
  terms that needs zero new code. Alibaba Wan 2.6 direct (~$0.0125/s,
  terms read: output is customer content, no training without consent)
  is the cheaper future option IF a DashScope engine is ever built --
  recorded here, not scheduled.
- **Video HQ = Veo 3.1 Lite.** $0.05/s at 720p WITH native audio --
  there is NO cheaper silent tier (verified three times: every Veo row
  is "video with audio price (default)"). Readable Google terms, same
  bill as the rest of the stack.
- **Rejected: fal.ai-hosted anything** (LTX $0.02/s, Seedance $0.0216/s
  -- cheapest on paper) because fal's Terms of Service could not be read
  (Vercel checkpoint / HTTP 429 on repeated attempts) and the operator's
  rule is: unreadable terms = unrecommendable. **Rejected: Replicate**
  (ToS s5.2 licenses Replicate to train on customer data).
  **Rejected: cloud upscale** ($13-498/episode for cosmetic gain on
  synthetic 720p; the 5080 does it locally for $0; if 4K matters,
  upscale the ~63 s of real clips + stills, never the 600 s master).
- **Script stays on OpenRouter.** Zero markup on inference; script cost
  is 0.5% of the bill at flash-class prices ($0.0039-0.02/episode).
  Model choice is a reliability question, not a cost question.

## Commercial terms summary (read 2026-08-07)

- Google paid tier: output not claimed by Google, paid prompts NOT used
  for training ("Used to improve our products -- Paid Tier: No"), no
  attribution required. SynthID watermark on Veo/Lyria (invisible; does
  not block YouTube monetization). If load-bearing ownership/indemnity is
  ever needed, build on Vertex (SST s20(a) ownership + s20(i) indemnity),
  not the AI Studio key surface.
- OpenRouter: passes ownership through to the upstream provider; 5.5%
  ($0.80 min) fee on credit purchases; credits expire after one year --
  do not pre-buy a large balance.
- YouTube: AI disclosure required for realistic synthetic media (does not
  restrict monetization); "inauthentic content" policy CAN demonetize
  templated mass-produced output -- distinct scripts and real editorial
  variation matter at 100 eps/month.

## Known fragilities

1. **~90% of the HQ bill sits on Google preview-priced SKUs** (Lyria 3,
   Gemini TTS models, Veo 3.1, Gemini Flash Image) under a 30-day
   price-change clause. Mitigation: the local pipeline remains the
   fallback for every modality; LOW's video leg rides Comfy credits.
2. **Gemini TTS has documented voice drift** ("output may not always
   strictly match the selected speaker") and NO version pinning.
   Mitigation is the audio cache (ratify gate in both profiles).
3. **Imagen/Veo safety filters cannot be loosened below block_only_high**
   without a Google account team (sales gate). Gemini TEXT filters
   default off, so the script lane is fine. The macbeth_probe ratify
   gate exists because a refused generation mid-episode is a broken
   render, and no vendor except Replicate documents whether a blocked
   generation is billed.
4. **Google Gemini API billing tiers:** Tier 1 caps at $250/month; a
   100-episode HQ month (~$285+) needs Tier 2, which is reached
   AUTOMATICALLY after $100 cumulative spend + 3 days -- no sales call.
5. Chirp free-tier permanence is not contractually promised; the TTS
   allowance is absent from Google's Free Tier framework page.

## Test-and-build order (recorded in GO_FORWARD_PLAN queue)

1. Ratify OpenRouter slugs for slot-a/b (existing ratify gate).
2. Build the content-addressed audio cache (blocks production voice).
3. Macbeth probe: one violent beat through Gemini TTS + each video arm.
4. 20-clip accept-rate measurement on each video arm.
5. First full `otr_cloud_low` episode end-to-end via the canonical
   workflow; then `otr_cloud_hq`.
Per the standing 2026-08-04 directive, the coding chunks in this order
get the full `kibitz-plugin:kibitz` four-round arc when they start.
