# Kibitz r3 (wiring/sequencing) — Claude anchor of the CLEAN BUILD_PLAN

VERDICT: yes-with-verify. The fold integrated all seven Fable fixes inline
(C1 CAPABILITIES parity + adapters; C2 three tuples; C3 adapter-metadata voice
id + provider_voice_id threading; C5 adapter-pinned announcer; C6 admission gate;
C7 per-line estimator; S6 dropdown-values; S7 profile-less). Pressure-test for
NEW wiring/sequencing defects the fold may have introduced:

CHECK:
1. Sprint-order integrity: does any S0..S7 step still remove a contract an
   earlier step depends on? (S0 explicitly defers xfail removal to S1/S5 now —
   confirm nothing else regressed.)
2. Does adding `elevenlabs` to `_LEGACY_FIRST_ENGINES` BEFORE the profile/
   CAPABILITIES rows exist (ordering within S1) transiently break
   build_engine_combo or the capability set-equality test?
3. announcer: is `announcer_voice_ref("elevenlabs")` called at a point where the
   bank is loaded and the S2 manifest is present (begin_episode timing)?
4. Does the S6 dropdown-VALUE selection actually route to the cloud adapter, or
   does some resolver still need the profile is_default flip?
5. Any positional-widget hazard when S6 sets the engine widget VALUE in the JSON.

CONFIRMED anchors (re-verified r2): _VALID_RUNTIMES/extra=forbid
(_otr_engine_profiles.py:35/67/95); _resolve_ref_to_disk bark fallback
(_otr_voice_node_common.py:472-552); _DEFAULT_ANNOUNCER_ENGINE=kokoro
(cast_lock.py:41); canonicalize_audio stub (cloud_media_canonical.py:127);
16gb_full.json slot_overrides; test_capability_profiles.py:213 set-equality.
