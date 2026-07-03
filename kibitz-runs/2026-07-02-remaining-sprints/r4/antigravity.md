VERDICT: yes-with-fixes. The plan has converged on its core directives (NO-FALLBACKS, cloud lanes), but several build-blocking ambiguities in fallback order, image model routing, and audio registration still need fixes to prevent build/test failures.

MUST-FIX BEFORE BUILD:
1. [Sprint A / Internal build order] Fallback-rip vs adapter None co-existence: A2 is ordered before A1 in Sprint A, but the current `render_driver.make_fallback_of()` (in `nodes/_otr_video_engines/render_driver.py#L153-172`) maps registered non-floor engines with `fallback_engine=None` to `UNIVERSAL_FLOOR` (`nodes/_otr_video_engines/render_driver.py#L170`). Setting adapters' `fallback_engine` to `None` in A2 will thus not prevent them from falling back to `still_motion` until A1 is completed.
   - Concrete fix: Combine A1 and A2 into a single atomic step, or reorder Sprint A to rip the fallback logic in `render_driver.py` before or simultaneously with modifying the adapters.
2. [Sprint B7] Image director dropdown mismatch: The plan lists dropdowns as `(announcer/music/other/character image models)` in B7. However, there is no `character_image_model` dropdown in the codebase (`nodes/otr_image_director.py#L58-65`). Character stills are instead routed through `other_beats_image_model`. An implementor might attempt to add a new `character_image_model` field, introducing unnecessary schema churn.
   - Concrete fix: Clarify in B7 that no new `character_image_model` input is to be added, and that character stills must continue to use `other_beats_image_model`.
3. [Sprint D3] Missing audio capabilities and import: Cloud TTS registration specifies profile rows in `config/audio_engine_profiles.yaml` but omits adding the ElevenLabs adapter import in `nodes/_otr_audio_engines/__init__.py` and the capability mapping in `nodes/_otr_audio_engines/registry.py#L185-210`. Since `tests/test_capability_profiles.py#L216` asserts `set(areg.CAPABILITIES) == set(areg._REGISTRY)`, registering ElevenLabs without adding capability rows will break the test suite.
   - Concrete fix: Add the ElevenLabs adapter import to `nodes/_otr_audio_engines/__init__.py` and add matching `CAPABILITIES` rows in `nodes/_otr_audio_engines/registry.py` in the same D3 change.
4. [Sprint B5 / D2] Conformance test vs Selector row: B5 specifies parametrizing over all 14 `partner_nodes.yaml` rows, but `cloud_elevenlabs_voice_selector` is an auxiliary `api_node: false` helper row with no adapter module (it is resolved locally by the TTS adapter in D2). Parametrizing all 14 rows will fail because the selector row lacks a corresponding registered engine/adapter.
   - Concrete fix: Explicitly specify that the conformance test should distinguish billed/adapter-driven rows (e.g. by filtering on `api_node: true`) from helper rows, and that the Sprint D requirement to remove ElevenLabs xfails only applies to `cloud_elevenlabs_tts` and `cloud_elevenlabs_flash`.

SHOULD-FIX:
1. [Sprint A1] Retry taxonomy cleanup: `nodes/_otr_shared/retry_taxonomy.py` contains fallback-specific fields and functions (`RetryDecision.escalate_to_fallback` #L136, `build_fallback_decision` #L251, `restamp_shot_row` #L282, `append_runtime_fallback_decision` #L300, and `format_swap_log` #L323) which become dead code under the NO-FALLBACKS directive.
   - Concrete fix: Specify that these obsolete fallback helper functions are to be deleted or marked deprecated, while keeping ledger schemas intact per A5.
2. [Sprint A3] API JSON fallback setting: The plan specifies updating the fallback settings in `workflows/otr_scifi_16gb_full.json` but omits `otr_scifi_16gb_full_api.json`, which still carries `allow_auto_fallback: true` in the live repo.
   - Concrete fix: Explicitly state that both `workflows/otr_scifi_16gb_full.json` and `otr_scifi_16gb_full_api.json` must be updated to change `allow_auto_fallback` to `False`.
3. [Sprint D1] Loudness reference source resolution: The plan says loudness should be matched to the "existing local reference", but `nodes/_otr_shared/cloud_media_canonical.py#L40` currently holds a draft value `LOUDNESS_REFERENCE_SOURCE = "UNRESOLVED: locate existing local-lane loudness reference (S2)"`.
   - Concrete fix: Specify that `LOUDNESS_REFERENCE_SOURCE` must be resolved to target `-16.0` dBFS active RMS using the constants and algorithms from `nodes/scene_sequencer.py#L400-500` (specifically `_loudness_normalize_clip`).
4. [Sprint B4 / B5] partner_nodes.yaml path consistency:
   - Concrete fix: Specify the full path `nodes/_otr_shared/partner_nodes.yaml` in the plan to avoid ambiguity.

OPTIONAL / NICE-TO-HAVE:
- Include a checklist mapping each cloud engine row to its corresponding adapter name, capability row, and expected xfail state to guide implementation.

CUT THESE:
1. [Sprint B5] "Optional inverse check: every registered cloud engine has a yaml row."
   - Rationale: Safe to cut as it is redundant with B2 capability declarations and the row-param conformance test; cutting it simplifies the test harness.

VERIFY-AT-BUILD checklist:
- [A5] Verify that `content_oracle.check_manifest` (`nodes/_otr_shared/content_oracle.py#L172`) does not require fallback trails in its parsing logic.
- [B6] Verify in `nodes/otr_shot_lock.py` or the orchestrator's beat loop that the portrait-mint gate runs before video engine selection, and that if selection occurs first, it raises an assertion/error.
- [A1] Verify that `nodes/_otr_shared/fallback.py` is successfully deleted and `retry_taxonomy.py` preserves its non-fallback classification role only.
- [D2] Verify via unit tests that custom object type `ELEVENLABS_VOICE` is successfully resolved in the adapter and marshaled/passed through `invoke_partner_node` without error.
- [Sprint C] Verify that the dynamic inputs for `cloud_seedance_2` are mapped in the V3-expansion pin and no longer raise a `RuntimeError`.
- [E1] Verify that no serialized `VideoRequest` cache files exist on the build machine before making `AudioMotionProfile` required in `nodes/_otr_video_engines/schemas.py`.
- [D1] Verify that `canonicalize_audio` uses the `-16.0` dBFS target RMS and matches local reference normalization behavior.

[ASSUMPTION] We assume that OTRVideoDirector and OTRImageDirector are class names corresponding to the node modules mentioned in the logs and workflow JSON.
[ASSUMPTION] We assume that no hidden serialized VideoRequest cache files exist on other user directories or sub-folders.
