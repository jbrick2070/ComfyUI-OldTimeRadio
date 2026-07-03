VERDICT: yes-with-fixes — core direction is converged, but the plan still has a few build-blocking ambiguities around fallback removal ordering, image-slot wiring, and cloud audio registration.

MUST-FIX BEFORE BUILD:
1. [Sprint A / Internal build order] A2 is ordered before A1, but current `render_driver.make_fallback_of()` still maps any registered non-floor engine with `fallback_engine=None` to `UNIVERSAL_FLOOR` (`nodes/_otr_video_engines/render_driver.py:153`, `:165`, `:170`; `UNIVERSAL_FLOOR` at `:56`). That violates the plan’s own “No window where fallback_engine=None adapters coexist with live chain resolution.” Concrete fix: make A1+A2 one atomic green chunk, or reorder to rip `render_driver`/soak fallback resolution before setting fleet adapters to `None`.

2. [Sprint B7] `character_image_model` is requested, but current image policy has only three image slots and routes characters through `other_beats_image_model` (`nodes/otr_video_director.py:338-341`, `nodes/otr_image_director.py:59-64`, `nodes/otr_image_director.py:332-333`, `nodes/otr_image_gen_dispatcher.py:152-162`). Concrete fix: either cut `character_image_model` from B7, or explicitly add it end-to-end: `OTR_VideoDirector.INPUT_TYPES`, policy serialization, `OTR_ImageDirector.IMAGE_SLOT_ROLES`, validation/picks/granularity/3D-lock maps, dispatcher role-to-slot map, tests, and `workflows/otr_scifi_16gb_full.json` appended positional widget.

3. [Sprint D3] Audio cloud TTS registration omits the audio capability table and package import surface. Current package import is the self-registration surface (`nodes/_otr_audio_engines/__init__.py:1-4`, `:23-31`), and `tests/test_capability_profiles.py:211-217` requires `areg.CAPABILITIES == areg._REGISTRY`. Concrete fix: add the ElevenLabs adapter import to `nodes/_otr_audio_engines/__init__.py` and add matching `CAPABILITIES` rows in `nodes/_otr_audio_engines/registry.py` in the same D3 change.

4. [Sprint B5 / D2] “Parametrize over partner_nodes.yaml ROWS (14)” conflicts with the auxiliary ElevenLabs selector row: `cloud_elevenlabs_voice_selector` is `api_node: false` with `hidden: {}` (`nodes/_otr_shared/partner_nodes.yaml:74-82`), and `invoke_partner_node` rejects rows with no hidden auth input (`nodes/_otr_shared/cloud_media_invoke.py:321-331`). Concrete fix: make the conformance test distinguish billed adapter rows from auxiliary helper rows; D’s “remove ElevenLabs xfails” should apply to `cloud_elevenlabs_flash`/`cloud_elevenlabs_tts`, not the selector helper, unless a separate local-helper conformance path is specified.

5. [Sprint D1 / Open verify-at-build] `canonicalize_audio` still depends on an unresolved loudness source. The code explicitly has `LOUDNESS_REFERENCE_SOURCE = "UNRESOLVED..."` (`nodes/_otr_shared/cloud_media_canonical.py:29-31`, `:42-44`), while D1 only says “existing local reference.” Concrete fix: name the exact source module/constant/procedure before build, and add a verify step proving cloud TTS clips match that local reference.

SHOULD-FIX:
1. [Sprint A1] `retry_taxonomy` is said to keep only non-fallback failure classification, but the current API and docstring are fallback-action-centric (`nodes/_otr_shared/retry_taxonomy.py:7-16`, `:103-111`). Specify whether `RetryDecision.escalate_to_fallback`, `build_fallback_decision`, restamp helpers, and fallback ledger records are deleted, renamed, or kept only for historical manifest parsing.

2. [B4/B5] The plan says `partner_nodes.yaml` without a path; the live loader uses `nodes/_otr_shared/partner_nodes.yaml` (`nodes/_otr_shared/cloud_media_invoke.py:58`, `:136`). Use the full path in the plan to avoid a builder creating a second config file.

OPTIONAL / NICE-TO-HAVE:
- Add a small table mapping each Sprint B/D cloud engine row to adapter name, provider row key, capability row, dropdown slot, and expected xfail state.

CUT THESE:
1. [B5] Cut the “Optional inverse check” from the locked build plan. The row-param conformance plus registry capability invariants already cover the critical failure mode; keep the inverse check as a later cleanup if desired.

VERIFY-AT-BUILD checklist:
- [A5] `content_oracle.check_manifest` does not require fallback trails.
- [B6] Portrait-mint gate executes before ShotLock/video engine selection.
- [A1] `fallback.py` rip preserves retry-taxonomy non-fallback classification only.
- [D2] `ELEVENLABS_VOICE` custom object marshals through the billed TTS invoke path.
- [Sprint C] `seedance_2` V3 dynamic inputs are resolved to pinned kwargs, never `COMFY_DYNAMICCOMBO_V3`.
- [E1] No serialized `VideoRequest` cache exists on disk before making `audio_motion_profile` required.
- [D1] Loudness reference source is resolved and tested against the existing local lane.
- [D3] ElevenLabs adapter is imported, registered, in `_LEGACY_FIRST_ENGINES`, in audio `CAPABILITIES`, in profile YAML, and surfaced in the workflow widget options.