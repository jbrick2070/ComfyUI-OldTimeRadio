# r3 JUDGMENT -- remaining-sprints plan (wiring round)

Judge: Claude (Cowork), 2026-07-02/03 night. Panel: codex + antigravity + claude CLI + anchor.
VERDICT: plan advances to r4 with the wiring deltas below. Panel was strong this round;
one misread discarded, one cut rejected.

## GROUNDED SURVIVORS

Sprint A:
- [codex MF5, CONFIRMED] A4 triage list was incomplete: test_video_humo.py:53/:57/:216-232
  (asserts fallback_engine VALUES + resolve_fallback_chain convergence) and
  test_video_mesh_stage.py:68 break on fallback_engine=None. Added to A4.
- [JUDGE, grep-proven] The plan never names `nodes/_otr_shared/fallback.py`
  (resolve_fallback_chain -- the AS-2 resolver). Its consumers: render_driver,
  scripts/otr_video_soak.py, scripts/otr_video_gpu_smoke.py,
  tests/test_video_fallback_chain_additive.py, tests/test_video_retry_taxonomy.py,
  tests/test_video_survival_guide_vectors.py, test_video_humo.py. E1 scope now
  includes retiring/rewriting this module + consumers (module DELETED; retry
  taxonomy keeps its non-fallback classification role -- verify split at build).
- [claude MF3, ACCEPTED] Sprint A INTERNAL order pinned: A4 (triage) -> A3b
  (Policy default False) -> A2 (adapter fallback_engine=None) -> A1 (rip
  machinery + fallback.py) -> A3a/c/d/e (widget flip + runtime-ignore + JSON
  audit + test pins). No window where None-adapters coexist with live chain
  resolution.
- [codex SF2, CONFIRMED] Stale fallback comments/logs at render_driver
  :1035-1037/:1287-1289/:1880-1882 updated in A1.

Sprint B:
- [claude MF1, CONFIRMED] B3 must REQUIRE: canonicalize_image writes a real PNG
  to disk (transcode JPEG/WEBP provider output), adapter returns str(asset.path),
  file exists and >= the dispatcher's minimum PNG bytes before _coerce_pixels reads.
- [claude MF2, CONFIRMED] prepare(None, None, None) must not crash (dispatcher
  calls it with three Nones; auth/budget live in the invoke bridge, not prepare).
- [codex MF3, CONFIRMED] invoke_partner_node defaults estimated_usd=0.0 and
  reserves exactly that -- every cloud image/TTS invocation MUST pass a nonzero
  per-row estimate (config/env, like eng_cloud_video). Budget machine is inert
  otherwise.
- [codex SF3 + AG MF4, CONVERGED] V3/COMBO single source of truth: resolved
  model IDs live in checked-in config/adapter constants (env-overridable);
  passing the literal "COMFY_DYNAMICCOMBO_V3" placeholder is a build error;
  conformance test asserts the resolved kwargs.
- [codex SF1 over AG cut, JUDGE CALL] B5 conformance stays over ALL yaml rows
  with an EXPLICIT xfail list; Sprint D's definition-of-done includes removing
  the ElevenLabs xfails. AG's "cut the unimplemented rows" REJECTED -- silent
  permanent masking is the exact failure the test exists to prevent.
- [claude SF3, VERIFY-AT-BUILD] B6 gate call order: confirm the gate point in
  the beat loop precedes engine selection (ShotLock) -- if selection happens
  first, the gate fires too late and becomes a swap (directive breach).

Sprint C:
- [claude OPT2] Note stamp consumers: ShotLock cloud-row stamps are
  observability-only for now ("consumed by nothing yet").

Sprint D:
- [codex MF1, CONFIRMED] Voice dispatch expects generate_voice() to return an
  AUDIO dict {"waveform": tensor, "sample_rate": int} (packed by
  pack_audio_batch -- _otr_voice_node_common.py:559/:565/:581); a CanonicalAsset
  or path return BREAKS the seam. D contract: adapter calls canonicalize_audio,
  then LOADS the canonical WAV back into the AUDIO dict; duration metadata
  stamped separately.
- [codex MF4 + claude SF1 + AG MF3, CONVERGED] voice_selector is NOT routed
  through invoke_partner_node (row has no hidden auth input; _inject_hidden_inputs
  raises AUTH for such rows -- cloud_media_invoke.py:331-336). The TTS adapter
  resolves the selector class, executes it in-process, UNWRAPS the returned
  tuple ([0]), and passes the ELEVENLABS_VOICE object as a pre-resolved input
  kwarg to the billed TTS call; the bridge's input marshaling must accept
  non-file custom types.
- [claude SF2, ACCEPTED] _LEGACY_FIRST_ENGINES: APPEND to the END of char_voice
  + announcer_voice tuples -- index 0 is the shipped default and must stay
  indextts2/kokoro.

Sprint E:
- [codex MF2 + AG MF1/2, CONVERGED, directive-safe rewording] The profile needs
  a CARRIER both request builders can read: compute once in run_real_episode
  (where master_audio_path exists), stamp into ledger["video"]["audio_motion_profile"],
  and BOTH build_request (:225) and build_request_from_shot (:972) emit it.
  AG's "fall back gracefully to a zeroed profile" is REWORDED to stay inside the
  directive: soak/test paths construct an EXPLICIT test-fixture profile (a
  visible, named constant in the fixture) -- production paths fail LOUD on a
  missing profile; nothing is silently defaulted at render time.
- [claude SF4, ACCEPTED] The VideoRequest field is REQUIRED (no None default);
  every fixture constructing VideoRequest updates in the SAME change; no
  serialized-request cache survives (verify none exists at build).

Cross-cutting:
- [codex SF4 + AG SF1, JUDGE CALL] Retry policy: v1 is LOUD STOP / NO RETRY,
  documented as intentional (directive-consistent: a transient failure is a
  failure). A bounded transport-only retry is a possible LATER opt-in, not in
  these sprints. AG's 3-attempt backoff REJECTED for v1.

## DISCARDED
- [AG SF2, MISREAD] "add ELEVENLABS_API_KEY env fallback": auth is comfy-org-
  mediated (OTR_COMFY_API_KEY / logged-in hidden inputs, shipped @ cc349c1d);
  an api_key_env auth kind already exists in the broker. No direct provider key.
- [AG cut of B5] rejected above.
- [claude cuts 1/2] were no-ops (agreed nothing to cut).

## CARRIED VERIFY-AT-BUILD
- B6 gate-vs-selection call order in the beat loop.
- fallback.py vs retry_taxonomy split (what survives the module rip).
- content_oracle trail requirements (from r2); ELEVENLABS custom-type marshal;
  seedance V3 pin dynamic inputs.
