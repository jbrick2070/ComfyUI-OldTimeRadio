# B0 -- bad-bark-clip fixture + spine verification (no code)

Source episode (from the 2026-06-21 nightly anthology soak, `OTR_SOAK_CHAR_VOICE=bark` forced):
`output/otr/episodes/signal_lost_the_pencil_stays_down_20260621_053019/`

- Title: "The Pencil Stays Down" / style `orbital_lifeboat_countdown`
- Writer: openrouter slot-a/b (deepseek-v4-pro + gpt-5.5), creativity = maximum chaos, 384w target / 437 actual
- Cast voices (from the ledger):
  - c01 ANNOUNCER -> `tts_model="kokoro"`, `voice_preset="bm_fable"` (NOT bark)
  - c02 CHRIS SHAW, c03 SKIP SPENDER, c04 KELLY SMITHERS -> `tts_model="bark"`, `voice_preset=null`
  - (The bark `v2/*` preset is resolved at dispatch from the seeded picker; the persisted
    cast row stores `null`. The first-line `[clears throat]` guard keys on the first
    occurrence of each resolved preset -> fires on b002, the first character line.)
- master.wav: `audio/pending_20260621_051232_master.wav` (48 kHz mono, 225.03 s)

## The ~0:24 clip = b002 (the FIRST bark character line)
| line | start_s | dur_s | role | preset(cast) |
|------|---------|-------|------|--------------|
| b001 | 9.50 | 9.67 | announcer | bm_fable (kokoro) |
| **b002** | **19.17** | **14.70** | **character (bark)** | null -> dispatch-resolved |
| b003 | 33.87 | 13.20 | character (bark) | null |

0:24 falls inside b002 (19.17-33.87 s). b002 is the FIRST bark character clip ->
`is_first_line=True` -> `_generate_single_line` prepends `[clears throat]`.

b002 RAW   : "All stations, start the count; Swift is falling faster than we can prove, and if that decay curve leaves even a crack open, we treat it like a rescue."
b002 CLEANED (`_clean_text_for_bark`): identical to RAW (no tokens present)
b002 FINAL (is_first -> anchor injected): "[clears throat] All stations, start the count; ..."

## Token audit across ALL lines (cleaned bark text)
- NO line carries `[music]` / `[whistles]` / `[sneezes]` / `[gasps]`.
- The ONLY artifact-position non-speech token is the **auto-injected first-line `[clears throat]`**
  on b002 (pass01 point 3). (b006 / b013 are `music_inter` rows carrying "Musical interlude
  bridging ..." text -- a SEPARATE issue fixed by Story-Quality R2 S1; they are non-voiced and
  never reach the bark path.)

### B0 conclusion (honest)
For THIS episode the squeal is NOT a `[music]`/`[whistles]` token render -- it is the self-inflicted
**first-line `[clears throat]` anchor** at the start of the first bark clip (b002 ~ 0:19-0:24).
- B1's `inject_first_line_anchor=False` (disable throat-clear for dialogue) is the lever that fixes
  THIS clip.
- B1's whitelist shrink (drop `[music]`/`[whistles]`/`[sneezes]`/`[gasps]` under `speech_only`) is the
  preventive lever for episodes whose dialogue DOES carry those tokens.
Both ship together in B1, so the build is unchanged. The operator's "confirm a
`[music]`/`[whistles]`/`[clears throat]` token" is satisfied: `[clears throat]` is confirmed at the
artifact position.

## High-band probe note (informs the QA metric)
A >4 kHz-RMS / total-RMS scan over the whole master is NOISY (p50=0.004 but p99=0.977; the near-1.0
windows are scattered across music interludes + sibilant/quiet segments, NOT pinned to 0:24). This
CONFIRMS the plan's decision to scope the QA gate to the **first+last ~150 ms of each per-line bark
clip**, not the master mix. The metric must run on a controlled bark re-render (B1 before/after),
not the mixed master.

## SPINE VERIFICATION -- `test_audio_byte_identical` is NOT a bark path
- The byte-identical regression (`test_audio_byte_identical_to_baseline`) runs the as-saved
  `workflows/otr_scifi_16gb_full.json` via live ComfyUI (env-gated; it was the 1 SKIP in the
  `-k audio_byte_identical` run, the 9 PASS are config/seed tests).
- Workflow node "3a. Character Voices (v2)" (`OTR_BatchCharacterVoices`) `widgets_values =
  ["indextts2", "mono_safe"]` -> char voice = **indextts2**.
- `eng_indextts2.default_roles = ("char_voice",)` (shipped default); `eng_bark.default_roles = ()`
  (demoted 2026-06-04). So the baseline master is produced by indextts2, not bark.
- => B1's bark-output change CANNOT alter the byte-identical hash. Spine stays frozen.
