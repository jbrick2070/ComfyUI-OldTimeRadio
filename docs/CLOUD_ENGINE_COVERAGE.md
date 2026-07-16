# CLOUD ENGINE COVERAGE LEDGER

> **STATUS 2026-07-15 (baseline): PARKED -- awaiting operator cloud keys.**
> Coverage rows remain the source of truth, but the harness named below
> (`scripts/_otr_cloud_audio_babysit.py`) is GONE at HEAD -- re-derive from
> `scripts/otr_api.py` direct-submit -- and node 83 wiring changed @ 6899d940 (music-bus
> links 280-283). Re-verify recipes before resuming the sweep. Predates the
> `otr_cloud_lanes` variant rename.

The methodical test matrix: every cloud engine x every role POSITION it can serve x
its 30-word full-episode status. "PASS" = rendered end-to-end with a real `otr/obs`
final. Keep this updated as legs run (source of truth for the cloud sweep).

Legend: PASS (obs final) | FAIL (root-cause noted) | TESTING | UNTESTED |
N/A (engine not valid for that position by design).

Harness: `scripts/_otr_cloud_audio_babysit.py` (direct-submit; managed engines via
`apply_profile`, the CastLock `voice_bank`/node-83 music engine via
`patch_widget_by_name`). Key loaded per-command from User env. One fresh server per leg.

RULE learned 2026-07-04: face/avatar VIDEO engines (HuMo-family: kling_avatar,
seedance, etc.) must be scoped to `character_video` ONLY. Forcing them on the
`announcer_visual`/`music_visual` BOOKENDS trips the RADIO-IS-HOST guard -> redirect
to `ltx_audio_in` -> radio-face-still-missing RenderError (this is what killed cv2 kling
before it made any cloud call). So bookend positions are N/A for face engines.

---

## CLOUD AUDIO

| engine | position (role) | status | evidence |
|--------|-----------------|--------|----------|
| elevenlabs | char_voice | **PASS** | live 2026-07-04: c02->el_laura (FGY2..), c03->el_george (JBFq..), engine=elevenlabs, no 401/422. Fix @fae7081f |
| elevenlabs | announcer_voice | UNTESTED | announcer defaults to kokoro; pool rows carry the role (George/Adam/Sarah/Brian) -- force to test |
| sonilo | music (open/close/inter) | **PASS** | live 2026-07-04: 3 cues cleared the 422 at floor=30, obs signal_lost_breath_of_the_cosmos_...final.mp4. Fix @8f146394 |

## CLOUD IMAGE  (positions: announcer_image, music_image, character_image)

| engine | announcer_image | music_image | character_image | evidence |
|--------|-----------------|-------------|-----------------|----------|
| ideo (ideogram_v4) | **PASS** | **PASS** | **PASS** | cv1 2026-07-04: all image roles=ideo, stills minted, obs signal_lost_the_vibecode_loop_...final.mp4 |
| recraft | -- | -- | **PASS** | live 2026-07-04: minted c02+c03 character portraits |
| flux_pro | **PASS** | -- | -- | live 2026-07-04: minted the announcer portrait |
| nano_banana_2 | -- | **FIXED** (retest) | -- | model must be the DYNAMICCOMBO_V3 DICT {model: slug} not a bare slug (Gemini node reads model["model"]); fixed @606dc7f1 |
| seedream_2 | UNTESTED | UNTESTED | UNTESTED | adapter shipped; ByteDance node model:str (NOT a dict -- different from Gemini); no live run |
| krea_2_turbo | UNTESTED | UNTESTED | UNTESTED | adapter shipped as `cloud_krea_2_turbo`; Krea2ImageNode text-only Medium Turbo, ~3.17cr/run |
| luma_photon_flash | UNTESTED | UNTESTED | UNTESTED | adapter shipped as `cloud_luma_photon_flash`; Luma Photon Flash text-only, ~0.57cr/run |
| ideogram (ideo_word words-specialist) | UNTESTED | UNTESTED | UNTESTED | plain `ideo` PASS; words-variant not yet run |

## CLOUD VIDEO  (positions: announcer_visual, music_visual, character_video)

| engine | announcer_visual | music_visual | character_video | evidence |
|--------|------------------|--------------|-----------------|----------|
| word_razzle (pixverse i2v) | **PASS** | **PASS** | **PASS** | cv1 2026-07-04: *=word_razzle all roles, obs final |
| cloud_kling_avatar | N/A (radio-host) | N/A (radio-host) | **TESTING** | scoped character_video only (prompt 0ff48b6d, 2026-07-04) |
| seedance_2 | N/A (radio-host) | N/A (radio-host) | UNTESTED | GO_FORWARD: honest DARK ROW (raises loud) until the V3-expansion pin |
| cloud_kling_lipsync | N/A | N/A | UNTESTED | needs a base_clip_ref (none in a fresh episode) -> fails loud by design |

---

## 100% ALL-CLOUD PATH -- recipe + the ONE remaining blocker (2026-07-04)

GOAL: one episode where EVERY generative engine is cloud. Status of the pieces:
- cloud IMAGE (ideo PROVEN; cloud_nano_banana_2 config applies clean) -- headless OK.
- cloud VIDEO (word_razzle PROVEN on all roles) -- headless OK.
- cloud VOICE (elevenlabs PROVEN) -- headless OK.
- cloud MUSIC (sonilo PROVEN) -- headless OK.
- cloud LLM (Comfy Credits) -- **PROVEN from the DESKTOP app 2026-07-04** (slot A
  google/gemini-3.5-flash wrote the style-picker, auth injected, credit-billed OK).
  GOTCHA: keep the TECHNICAL slot (comfy_slot_b_model) OFF reasoning models --
  `perplexity/sonar-reasoning-pro` hit an upstream 502 (returns no choices / times out);
  use a fast non-reasoning model (google/gemini-3.5-flash or deepseek/deepseek-v4-flash).
  Still **BLOCKED headless** (see below). The writer's Comfy Credits lane
  auths via HIDDEN node inputs (auth_token_comfy_org) that only the logged-in DESKTOP
  UI injects; a headless /prompt gets `ComfyCreditsConfigError: No Comfy credential`.
  The media engines work headless because invoke_partner_node pulls auth from
  OTR_COMFY_API_KEY -- the writer's Comfy Credits path does NOT do that env fallback.
  FIX OPTIONS: (a) run the all-cloud episode from the Desktop app (logged in); or
  (b) code: give the writer's Comfy Credits path an OTR_COMFY_API_KEY env fallback like
  the media lane (then it runs headless).

EXACT all-cloud config (harness leg `all_cloud_llm`, or set these in the Desktop UI):
- writer node 1: creative_writing_model=`comfy:slot-a`, technical_model=`comfy:slot-b`,
  comfy_slot_a_model=`google/gemini-3.5-flash`, comfy_slot_b_model=`google/gemini-3.5-flash`;
  env `OTR_ENABLE_COMFY_CREDITS=1`. (Comfy models: anthropic/claude-opus-4.7,
  openai/gpt-5.5-pro, google/gemini-3.5-flash, x-ai/grok-4.20, deepseek/deepseek-v4-pro...)
- image role_overrides: `announcer_image`/`music_image`/`other_beats_image` = `cloud_nano_banana_2`
  (NOTE naming drift: the profile role key is `other_beats_image`, NOT `character_image` --
  the rename only touched the VIDEO widget `character_image_model`. Cleanup item.)
- video: `OTR_FORCE_ENGINE_MAP=*=word_razzle` (+ `OTR_LTX_RADIO_FACE=0`).
- voice: slot_overrides char_voice_engine=`elevenlabs` + node 80 voice_bank=`elevenlabs_cloud`.
- music: node 83 engine=`sonilo`.

RULE (kling lesson): face/avatar VIDEO engines (kling_avatar, seedance) must be scoped to
`character_video` ONLY (`OTR_FORCE_ENGINE_MAP=character_video=<engine>`); on bookends they
trip the RADIO-IS-HOST redirect -> ltx_audio_in -> radio-face-still-missing RenderError.

## OPEN / NEXT
- CLOUD IMAGE sweep: recraft, flux_pro, nano_banana_2, seedream_2, krea_2_turbo, luma_photon_flash, ideogram-words (force image roles; video=still_word fast, audio local).
- CLOUD VIDEO: confirm kling_avatar on character_video; seedance_2 needs the V3-expansion pin before it can be tested.
- CLOUD AUDIO: elevenlabs announcer_voice position (force announcer to elevenlabs).
- Then MIX passes (coherent same-model combos) + the 800w all-visualizer credits run.
