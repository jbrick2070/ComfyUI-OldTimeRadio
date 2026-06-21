# ltx_av_music crashes on a music beat with no audio (FamilyInputGap)

## Symptom
In the nightly soak, the legs whose MUSIC bookend = `ltx_av_music` intermittently HARD-FAIL the
render (status=error) even though the story wrote fine (18 lines). Operator: "one ltx/humo/3d
bookend should not error -- that's a simple render; if it's crashing, something else is happening."

## Root cause (grounded in the server log + render_driver.py)
`ltx_av_music` is family **`audio_conditioned_video`**, whose `FAMILY_REQUIRED_INPUTS` include
**`audio_ref`** (it is an AUDIO-driven engine -- it renders music conditioned on an audio input).
For certain MUSIC beats the request carries NO `audio_ref`, so the pre-render gate
`_assert_family_inputs_satisfiable` (`render_driver.py:1255-1281`) raises **`FamilyInputGap`** ->
`render FAILED (no fallback) ... FailureKind.DEPENDENCY_MISSING` (fallbacks are disabled by the
2026-06-16 no-fallbacks directive, so it hard-crashes the episode).

Why no `audio_ref`: `build_request_from_shot` slices the per-beat audio from the frozen master mix
when the line has no per-line `*_wav_path`. For the audio-reactive lanes (`ltx_av_music` /
`visualizer`) it already falls back to the SHOT row's `start_s`/`dur_s` when the LINE lacks them --
BUT when the SHOT ALSO lacks `start_s`/`dur_s`, the slice is skipped and `audio_ref` stays empty.
The log shows **28 beats with "has no start_s/dur_s on line"** this run; HuMo degrades LOUD and the
visualizer falls back to idle-scopes-from-silence, but `ltx_av_music` (which HARD-requires
`audio_ref`) raises FamilyInputGap. Net: **only `ltx_av_music` crashed (4x)**; humo/ltx_video/wan/
mesh_stage/visualizer did not.

So it is NOT the LTX render itself failing -- it is a per-beat-AUDIO gap: a music beat that reached
an audio-conditioned engine with no master-audio slice.

## Evidence
- `[OTR video] render FAILED (no fallback) shot shot_b006 engine ltx_av_music: FamilyInputGap:
  candidate 'ltx_av_music' (family audio_conditioned_video) requires input(s) ['audio_ref'...]`
- `FailureKind.DEPENDENCY_MISSING` at `_assert_family_inputs_satisfiable` (`render_driver.py:1278`).
- `[OTR.render_driver] per-beat audio: beat b006 has no start_s/dur_s on line` (x28 across beats).
- FamilyInputGap engine tally: `ltx_av_music 4x` (no other engine).

## Invariants
Audio SPINE is frozen (byte-identical master, mux-LAST) -- the fix only READS/slices the master, never
mutates it. No-fallbacks directive holds (do NOT add a silent fallback engine). Capability-gated;
deterministic; the workflow JSON + ledger schema untouched if possible.

## Candidate fixes (to converge on)
1. **Always slice the master for audio-conditioned lanes (the source fix).** In
   `build_request_from_shot`, when an `audio_conditioned_video` (or any audio_ref-requiring) engine
   is selected and neither the line NOR the shot has `start_s`/`dur_s`, compute the beat's window
   from its POSITION in the episode (cumulative beat durations, or an even partition of the master),
   or as a last resort slice the WHOLE master. Every beat occupies a real span of the master mix, so
   `audio_ref` can always be provided -> the gate passes, the engine renders. (Keeps no-fallbacks;
   just supplies the input the engine legitimately needs.)
2. **Guarantee music-beat timing upstream.** Ensure every music/announcer beat gets `start_s`/
   `dur_s` stamped (the synthetic opener b000 already does; inter-music beats b006/b013 do not).
   Deeper (touches ShotLock / timing), higher risk.
3. **Capability-aware routing (last resort).** If a beat genuinely has no audio, do not ROUTE it to
   an audio_conditioned engine -- but per no-fallbacks this is a router change, and the operator
   wants ltx_av_music to WORK on music beats, so the real fix is supplying the audio (option 1).

## Open questions for the panel
- Q1: Is option 1 (always-slice from the beat's timeline position) correct, or does ltx_av_music need
  the EXACT beat audio (vs the whole master) to render a sensible music clip?
- Q2: Where is the beat's timeline position authoritatively available in `build_request_from_shot`
  (cumulative shot durations? the master length / beat count? the ledger timing block)?
- Q3: Why do b006/b013 (inter-music beats) lack `start_s`/`dur_s` when b000 has them -- is option 2
  (stamp timing upstream) the more correct root fix, or is option 1 the robust render-side guard?
- Q4: Does any other audio_conditioned engine (ltx_av_talk on announcer beats) have the same gap, and
  does the fix cover it?
