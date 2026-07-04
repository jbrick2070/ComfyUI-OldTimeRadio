# CLOUD ENGINE COVERAGE LEDGER

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
| recraft | UNTESTED | UNTESTED | UNTESTED | adapter shipped (B1 @b5ef58bc); no live run |
| flux_pro | UNTESTED | UNTESTED | UNTESTED | adapter shipped (B1); no live run |
| nano_banana_2 | UNTESTED | UNTESTED | UNTESTED | adapter shipped (B1); no live run |
| seedream_2 | UNTESTED | UNTESTED | UNTESTED | adapter shipped (B1); no live run |
| ideogram (ideo_word words-specialist) | UNTESTED | UNTESTED | UNTESTED | plain `ideo` PASS; words-variant not yet run |

## CLOUD VIDEO  (positions: announcer_visual, music_visual, character_video)

| engine | announcer_visual | music_visual | character_video | evidence |
|--------|------------------|--------------|-----------------|----------|
| word_razzle (pixverse i2v) | **PASS** | **PASS** | **PASS** | cv1 2026-07-04: *=word_razzle all roles, obs final |
| cloud_kling_avatar | N/A (radio-host) | N/A (radio-host) | **TESTING** | scoped character_video only (prompt 0ff48b6d, 2026-07-04) |
| seedance_2 | N/A (radio-host) | N/A (radio-host) | UNTESTED | GO_FORWARD: honest DARK ROW (raises loud) until the V3-expansion pin |
| cloud_kling_lipsync | N/A | N/A | UNTESTED | needs a base_clip_ref (none in a fresh episode) -> fails loud by design |

---

## OPEN / NEXT
- CLOUD IMAGE sweep: recraft, flux_pro, nano_banana_2, seedream_2, ideogram-words (force image roles; video=still_word fast, audio local).
- CLOUD VIDEO: confirm kling_avatar on character_video; seedance_2 needs the V3-expansion pin before it can be tested.
- CLOUD AUDIO: elevenlabs announcer_voice position (force announcer to elevenlabs).
- Then MIX passes (coherent same-model combos) + the 800w all-visualizer credits run.
