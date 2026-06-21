<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Candidate fix 1 is underspecified, the existing per-beat audio block in build_request_from_shot still permits FamilyInputGap for audio_conditioned_video, and open questions Q1/Q2/Q4 remain unclosed.

MUST-FIX BEFORE BUILD:
1. [build_request_from_shot: per-beat audio fallback, lines after "if not audio and master_audio_path"] When neither line nor shot supplies start_s/dur_s the slice is skipped and audio_ref stays None for family audio_conditioned_video; the ltx_av_music special case only reads shot.get("start_s") but does not synthesize a window, so _assert_family_inputs_satisfiable (render_driver.py:1278) still raises. Concrete fix: add an else that computes a non-zero window from ledger line order (or aborts the request with a clear error) before the "if (start_s is not None..." test; do not emit the "HuMo will degrade" warning for ltx_av_music.
2. [schemas.py: FAMILY_REQUIRED_INPUTS] audio_conditioned_video lists ("text_prompt", "audio_ref") with no optional path; this is the direct cause of the hard FamilyInputGap on music beats that reach the gate. Either make audio_ref optional for this family (with documented degradation) or guarantee the builder always supplies it; the current "no-fallbacks" rule makes the latter mandatory.
3. [Candidate fixes option 1] The proposal to "compute the beat's window from its POSITION ... or even partition of the master" is not implemented and cannot be reviewed; without an authoritative source for cumulative durations (ledger['lines'] order? shot order? master length?) any slice may be wrong for an audio-reactive engine. Supply the exact algorithm + unit test before build.

SHOULD-FIX:
1. [build_request_from_shot: _beat_id_for_shot + line_index] Inter-music beats b006/b013 lack start_s/dur_s while b000 has them; the code path that stamps synthetic openers is not visible here (verify: ShotLock / OPENING_MUSIC_SUFFIX), so the render-side guard in fix 1 is a band-aid. Stamp timings at ledger construction time instead.
2. [run_real_episode + build_request_from_shot] Q4 is unaddressed: ltx_av_talk on announcer beats uses the same audio_conditioned_video? No, it uses audio_driven_face, but any future audio_ref-requiring engine will hit the identical gap; add a family-level check in the audio slice block rather than an engine_id whitelist.
3. [_assert_family_inputs_satisfiable] The FamilyInputGap message cites only the first missing token; for audio_conditioned_video it should also log the beat_id and whether a master slice was attempted, so the nightly soak log is actionable.

OPTIONAL / NICE-TO-HAVE:
- Add a unit test that feeds build_request_from_shot a music shot with zero timings and asserts audio_ref is still populated (or the request is rejected early).
- Emit the per-beat warning at WARNING only for audio_driven_face; downgrade to INFO for audio_conditioned_video once the synthesis path exists.

CUT THESE (over-engineering):
- Candidate fix 2 (stamp timing upstream in ShotLock): safe to cut because the render driver already owns the master slice and the no-fallbacks rule; touching ShotLock adds a cross-module dependency that is not required to close the crash.
- Candidate fix 3 (capability-aware routing): safe to cut; it would require changing the router and violates the operator directive that ltx_av_music must work on music beats.

[ASSUMPTION] Cumulative beat durations or master length are always present in the ledger passed to build_request_from_shot; the provided excerpts do not show this field.