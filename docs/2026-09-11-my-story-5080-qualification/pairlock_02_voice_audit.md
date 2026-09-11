# Pairlock 02: cast metadata, actual voices and credits

Read-only production audit of code `bd8141487f2bbc093451cd7573cb3b338874b8e4`.
No production code, running process or episode artifact was changed. No models,
tests, audio audition or image inspection were run for this audit. The server
log records publication and `Prompt executed in 00:13:54`; successful execution
does not establish that the voice or visual casting matched the source.

## Evidence identity

- Episode: `signal_lost_the_la_table_20260911_022847`.
- Published ledger: `C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes/signal_lost_the_la_table_20260911_022847/audio/signal_lost_the_la_table_20260911_022847_ledger.json`.
- Final observed ledger SHA256: `2d5a896bafc5753d67c4f8edca390cee88cabd3a962a2e37532d2c635a4f918d`;
  409764 bytes; last write `2026-09-11T09:34:54Z`.
- Earlier running snapshot SHA256:
  `cc8031617cc12d6961b9afb61c91cfc1a08ef076b2b6b7ce86917d3b2fab629d`;
  409380 bytes; last write `2026-09-11T09:33:59Z`.
- Request: `pairlock_02_request.json`; P0 raw-output comparison:
  `pairlock_02_p0_audit.json` in this directory.
- Actual voice observations: `tmp/my_story_5080_eos_server.log:591-598`.
  Publication is recorded at line 1238, completion at line 1241.

## What the saved data and registry prove

Both final dramatic cast rows have `gender: null`, legacy `tts_model: bark`,
and no `voice_ref_id`, `voice_engine` or `presentation_gender` field. Jeffrey's
legacy preset is `v2/en_speaker_4`; Mother's is `v2/en_speaker_6`.
Jeffrey's description starts "An adult man" and Mother's starts "A loving,
spirited woman". These descriptions were authored metadata, not audio evidence.

The final ledger records `meta.char_voice_engine: indextts2`,
`meta.announcer_voice_engine: kokoro`, `meta.cast_voice_policy: auto_registry`
and `meta.voice_bank_id: default`. Its announcer row has an actual Kokoro stamp
(`voice_ref_id: bm_george`, `voice_engine: kokoro`). The character rows do not.

The actual per-line log reports `engine=indextts2` on all eight dramatic lines:

| Character | Lines | Actual logged reference | Registry gender |
| --- | --- | --- | --- |
| c02 Jeffrey | shot_001_b1, b3, b5, b7 | vz_pd_librivox_mark_f_smith_elder.wav | male |
| c03 Mother | shot_001_b2, b4, b6, b8 | vz_stuart_bell.wav | male |

These gender labels come from the explicit entries in
`config/voice_reference_bank.json:1376` and `:39`, respectively. Both entries
declare `engine: indextts2` and role `char_voice`. No gender was inferred from
the filenames, and no claim is made about how either recording sounds.

## Metadata ownership: no gender compiler drop

The original P0 interpretation explicitly supplied Jeffrey's
`stated_gender: male`; Mother's was already empty. Its assumptions explicitly
said Mother's gender was unspecified. The raw source correction omitted both
`stated_gender` fields. Pydantic defaults then left both empty in the accepted
interpretation. That omission/default problem is separately documented in
`pairlock_02_p0_audit.json`; it explains the loss of Jeffrey's value, but does
not explain Mother's original blank.

The accepted P1 treatment explicitly has empty gender for both characters,
despite the descriptions quoted above. `_assign_voices`
(`nodes/_otr_my_story.py:659-701`) copies `member.gender` into its slot and cast
row. `Ledger.set_cast` (`nodes/production_ledger.py:1335`) normalizes blank to
null. The compiler did not discard a nonempty P1 gender.

The existing P1 fidelity comparison (`_otr_my_story.py:1025-1034`) only compares
nonempty P0 genders after acceptance. Its final discrepancy list is empty here.
It cannot discover Mother's source relationship or recover omitted P0 data.

## Distinct delivered-voice provenance and credits defect

`CastLock._auto_registry` (`nodes/cast_lock.py:1174-1183`) skips a blank-gender
character, preserving the historical Bark fields without selecting or stamping
a registry reference. The shared render resolver
(`nodes/_otr_voice_node_common.py:176-209`) then selects a real reference through
`gender_agnostic_fallback_ref` and returns its path without stamping the cast.
The shared voice node verifies its selected engine against
`meta.char_voice_engine` at lines 1057-1074 and resolves the clone reference at
line 1336. This is IndexTTS rendering, not a silent fallback to Bark.

`nodes/otr_credits_roll.py:384-394` reads each row's actual reference/engine
first, then falls back to its legacy preset plus literal `bark`. Consequently,
the final saved rows produce Bark/preset credit text despite the logged
IndexTTS references. This is a direct data/consumer mismatch; this audit did
not inspect the rendered credits pixels.

The smallest existing owner for correcting this occurrence is CastLock's
registry caster. Its unservable-gender branch already uses the same shared
fallback and calls `_stamp_row` (`cast_lock.py:1218-1245`). Extend that existing
selection/stamp route to genuinely blank genders, preserving their blank
story gender and recording the chosen reference's separate presentation
gender. `_stamp_row` tracks which rows were actually cast; `_stamp` at
1624-1657 writes the reference, engine, fallback reason and presentation gender.
`lock` at 470-486 already durably persists the cast and emits the matching wire
ledger. No TTS-time frozen-ledger mutation or new persistence owner is needed.
Preserve replay, `preserve_ledger`, the existing Google-specific behavior and
unavailable-bank behavior. An absent source gender must not become a new
content rejection gate.

## Bounded author remedy without another model pass

The existing P0/P1 prompts in `nodes/story_packs/my_story/my_story.json` say
"stated gender" and prohibit inference from names, but do not explain how to
read relationships and descriptions in context. P1 also asks for a casting
description while telling the model to leave an unstated gender blank. This
allows the observed combination of an explicitly authored woman description
and blank casting metadata.

Clarify those existing prompts and the corresponding Pydantic field
descriptions: resolve gender from source descriptions, relationships and
pronouns in context; honor an explicit identity over a conventional role;
never infer it from a proper name alone; leave it empty when genuinely
unspecified. P1 should keep its casting description and structured gender
consistent. This uses the same model calls and source context, not a new
semantic checker, name lookup, regex classifier or universal nonempty rule.

Once P0 preserves its declared values, `_make_treatment_validator` at line 469
can reuse `StoryInterpretation.gender_by_name()` at line 181 with the existing
normalized exact name join. A blank/conflicting P1 value for a known P0 gender
can receive a precise repair instruction through the existing author ladder.
`_call` already passes the same validator into the source-correction ladder at
393-397. This consistency check alone cannot fix Mother: the existing author
must first resolve her source relationship/description.

Do not reuse `_otr_character_roster.infer_gender` for these narrative
descriptions. Its Folger-specific relation-word scan (`:217-248`) would read
Jeffrey's mention of his "mother" as female and Mother's mention of her
"son" as male. That helper assumes a cast-list description of the subject,
which these multi-person narrative descriptions do not satisfy.

## Existing regression owners to extend

- `tests/test_my_story_runner.py`: the current stated-gender test at line 222
  starts with aligned fixtures. Add known P0 gender with blank/opposite P1,
  repaired output, normalized exact name joins, source-correction parity and
  genuinely unspecified gender remaining allowed. Prompt/schema tests should
  cover the model's context contract without pretending fixtures prove live
  gender understanding.
- `tests/test_cast_lock.py:291`: `test_auto_registry_skips_genderless_character`
  currently pins the unstamped blank-gender branch and must change with the fix.
- `tests/test_cast_lock_voice_ref_completeness.py`: extend existing shared-draw
  and cast completeness coverage to blank and null genders, verifying the
  story gender remains unchanged and the stamped reference is the one the
  render resolver opens. Cover multiple characters and existing usage tracking.
- Existing CastLock durable-stamp and credits-spec fixtures can establish
  saved cast, wire cast, render resolver and credits text equality without
  running TTS, a model or a media render.

These are proposed follow-ups, not completed fixes or test results. This
receipt does not claim voice suitability, scene fidelity or release readiness.
