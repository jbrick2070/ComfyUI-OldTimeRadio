# Comfy partner-node pricing stamp -- v2026-07-02

Source: https://docs.comfy.org/tutorials/partner-nodes/pricing
(fetched 2026-07-02). Conversion: 211 credits = 1 USD. This file is
the versioned pricing source pass04 verify #3 requires; re-stamp at S0
pin time and on any roster change.

## Voice (ElevenLabs -- FLAT across model tiers)

| Row / node | Credits | ~USD |
|---|---|---|
| cloud_elevenlabs_tts / _flash (TextToSpeech) | 50.64 / 1K chars | $0.24 / 1K chars |
| TextToDialogue (experiment flag) | 50.64 / 1K chars | $0.24 / 1K chars |
| InstantVoiceClone (deferred lane) | 31.65 / run | $0.15 / voice |
| SpeechToSpeech | 50.64 / min | $0.24 / min |
| TextToSoundEffects (unused: sfx role deleted) | 29.54 / min | $0.14 / min |
| VoiceSelector (aux) | free | -- |

IMPORTANT: price does NOT vary by model tier -- the flash/premium row
split is QUALITY-only. Per-episode reality: ~790-word script ~= 4.6K
chars ~= 233 cr ~= **$1.10 for ALL dialogue**.

## Music

| Row | Credits | ~USD per theme |
|---|---|---|
| cloud_sonilo_music | 0.5275 / sec | 60s ~= 31.65 cr ~= **$0.15** |
| cloud_stability_audio (stable-audio-2.5) | 42.2 / run | **$0.20** |

## Stills (per image)

| Row | Credits | ~USD | Tier |
|---|---|---|---|
| cloud_seedream_2 (5.0-lite / 4.5) | 7.39 / 8.44 | $0.035-0.04 | LOW (stylization) |
| cloud_recraft (v3/v4 std) | 8.44 | $0.04 | LOW |
| cloud_ideogram_v4 TURBO | 9.05 | $0.043 | LOW + text rendering |
| cloud_ideogram_v4 DEFAULT / QUALITY | 18.1 / 27.16 | $0.086 / $0.13 | MID / HIGH |
| cloud_nano_banana_2 (token-billed, ~1.3K img tokens) | ~16 | ~$0.08 | MID (char consistency) |
| cloud_flux_pro (Flux2 Pro) | NOT ON PRICING PAGE | verify at pin | LOCAL-MATCH (flux family) |
| Recraft v4_pro (not curated) | 52.75 | $0.25 | PREMIUM reference |

Ideogram's rendering_speed param spans LOW->HIGH inside ONE row.
Typical episode stills budget: 3-6 images ~= **$0.12-0.50**.

## Video (context -- the REAL spend)

| Node family | Credits/sec | 6s clip ~USD |
|---|---|---|
| Kling generation 720p/1080p (no audio) | 17.72 / 23.63 | $0.50 / 0.67 |
| Kling video-to-video / EDIT (lipsync class) 720p/1080p | 26.59 / 35.45 | $0.76 / 1.01 |
| Kling turbo/omni w/ audio | 23.63-29.54 | $0.67-0.84 |

PER-LINE LIPSYNC IS THE DOMINANT COST: ~$0.25-1.00 per line clip.
This is why the formats plan lipsyncs CLOSE-UP LINES ONLY and the
estimate report prints per-line Kling rows.

## 3D (Prop Shot / Tin-Toy references)

| Node | Credits | ~USD |
|---|---|---|
| Tripo Image/Multiview->Model v1.4 | 63.3 / run | $0.30 |
| Meshy Multi-Image->Model (no tex / tex) | 42.2 / 126.6 | $0.20 / 0.60 |
| Meshy Rig / Animate (future-lane) | 42.2 / 25.32 | $0.20 / 0.12 |
| Tencent 3.0 Geometry | 63.3 | $0.30 |

## Episode envelope (all-cloud, stills-heavy, few lipsync lines)

voice $1.10 + music $0.20 + stills $0.50 + 5 lipsync close-ups
(~$0.75 ea) $3.75 ~= **$5.55/episode**; mute-I2V b-roll adds
~$0.50-0.70 per 6s Wan/Kling-std clip. Budget env suggestion:
OTR_CLOUD_MEDIA_BUDGET_USD=10 for a standard episode, 25 for
video-heavy.
