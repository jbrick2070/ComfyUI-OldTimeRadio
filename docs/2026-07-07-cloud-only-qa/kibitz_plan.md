# Cloud-Only Partner Route QA

## Scope

Audit the current diff in `ComfyUI-OldTimeRadio` for cloud-only media routes:

- `OTR_CastLock` now has explicit `char_voice_engine` and
  `announcer_voice_engine` widgets. Profiles patch those plus `voice_bank`.
- `cloud_all` profile should route voices to ElevenLabs, music to Sonilo,
  stills to Nano Banana 2, and video to `cloud_wan_i2v_audio`.
- `cloud_wan_i2v_audio` is a new adapter sharing the installed Partner node
  key `cloud_wan_i2v` and only adding the declared optional `audio` input.
- Image and video request handoffs append the safety clause:
  no blood, no guns, no knives, no smoking.
- Character portrait prompts add a cast-gender anchor when the cast says
  male/female and the writer prompt forgot it.

## Files To Inspect

- `nodes/cast_lock.py`
- `nodes/_otr_video_engines/eng_cloud_video.py`
- `nodes/_otr_video_engines/registry.py`
- `nodes/_otr_image_engines/eng_cloud_image.py`
- `nodes/otr_image_gen_dispatcher.py`
- `nodes/_otr_video_engines/render_driver.py`
- `nodes/_otr_story_brief_helpers.py`
- `nodes/otr_meta_brief_image_prompt.py`
- `config/profiles/*.json`
- `config/profiles/widget_mapping.json`
- `workflows/otr_scifi_16gb_full.json`
- focused tests touched in `tests/test_cloud_*`, `tests/test_workflow_apply.py`,
  `tests/test_still_spine_helpers.py`, and `tests/test_image_platform_c1.py`

## Must-Fix Checks

- No profile or dispatcher path should silently load local Flux/HuMo when the
  all-cloud profile is selected.
- CastLock durable cast metadata and the TTS nodes must agree for ElevenLabs;
  no Kokoro/IndexTTS2 durable stamp when the cloud voice profile is selected.
- The new Wan audio adapter must not feed undeclared Partner-node inputs and
  must fail loud when `audio_ref` or `init_image` is missing.
- Existing mute `cloud_wan_i2v` must continue omitting `audio`.
- No stale Comfy template-only audio-to-video route should be represented as a
  Partner API node.
- Safety clauses should reach actual render requests, including env negative
  prompt overrides.
- Workflow widget edits must be append-only for node 80 and `16gb_full` apply
  must remain an API-prompt no-op.

## Current Focused Test Result

`293 collected`: `288 passed, 3 skipped, 2 xfailed`.
