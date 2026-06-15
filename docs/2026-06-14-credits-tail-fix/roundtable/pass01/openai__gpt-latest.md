<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The safe direction is A, but the spec is not build-ready because it contradicts the shown mux/composite code, does not define how `hud_tail` is conveyed to the mux/blend, and would let either real drift or §4D/OBS publishing re-clamp the credits unless additional gates are specified.

MUST-FIX BEFORE BUILD:
1. [Grounded facts #3 / Candidate A] The document says `otr_master_audio_mux.py` “ASSERTS composite_duration == master_audio,” but the grounded code does not: `mux_master_audio()` currently permits `a_dur > v_dur` and only rejects `v_dur > a_dur + tol`. Fix the spec and implementation target to modify the actual gate: “reject video longer than audio unless an explicitly authorized silent credits tail is present,” not “relax equality.”

2. [Candidate A / Q2] `a <= v <= a + hud_tail` is not a safe replacement gate as written because it would reject the currently supported `a_dur > v_dur` case documented in `mux_master_audio()` comments. Concrete fix: preserve the existing no-tail behavior, and add a separate authorized-tail branch:
   - default `expected_silent_tail_frames = 0`: keep current gate, or require `v_frames <= audio_budget_frames + tol_frames`.
   - when `expected_silent_tail_frames > 0`: require `abs((v_frames - audio_budget_frames) - expected_silent_tail_frames) <= 1`.
   Do not use a broad “video may be longer by up to N seconds” gate; that would hide real drift.

3. [Candidate A / Q2 / Q3] The mux cannot currently know `hud_tail`. Its inputs are only `silent_video_path`, `master_audio_path`, `audio_done`, `fps`, `ffmpeg`, `output_path`. The plan says “assert `v - a == hud_tail`,” but no wiring/metadata source exists in the shown mux API. Concrete fix: add an authoritative `expected_silent_tail_frames` input to `OTR_MasterAudioMux` or a durable sidecar/manifest field generated upstream, then update `workflows/otr_scifi_16gb_full.json` to wire it. Default must be `0` so old workflows remain fail-closed.

4. [Q3 / otr_silent_composite.py] The current assemble path explicitly caps `target_total` to the longest sibling `*_master.wav` duration minus one frame:
   `base_total = max(0, int(round(master_dur * fps)) - 1)`.
   That is the clamp cutting the credits in the shown code. Concrete fix for the intentional credits-tail path: when `floor_frames > audio_budget_frames`, set `target_total = floor_frames` or `audio_budget_frames + expected_tail_frames`, not `round(master_dur * fps) - 1`. Also update the misleading gate/report strings saying the assembled count is the “audio-derived budget.”

5. [Q3] Do not derive the tail from the floor MP4’s embedded audio stream. The grounded `otr_silent_composite.py` comment says the procgen base may carry silence padded to the video length, so `_probe_audio_duration(base_video_path)` can equal the too-long floor, not the master. Concrete fix: derive `audio_budget_frames` from the real `master_audio_path`/ledger/audio manifest, or pass it into the composite explicitly. Only fall back to base audio/container in legacy non-tail mode, not in the intentional HUD-tail mode.

6. [Q2 / Q3] The spec mixes seconds and frames in a way that will produce off-by-one failures with the shown durations: video `45.68s` at 25 fps is 1142 frames; audio `45.706s` rounds differently. Concrete fix: define all gates in frames:
   - `v_frames = decoded video frame count`
   - `audio_budget_frames = round(master_audio_duration * fps)` or, better, the same ledger/master frame budget used by the render driver
   - `expected_silent_tail_frames = floor_frames - audio_budget_frames`
   - final expected video frames = `audio_budget_frames + expected_silent_tail_frames`
   Allow only a declared `±1` frame tolerance where unavoidable.

7. [Q4] Padding only “the scopes input” is underspecified. The §4D blend uses `shortest=1` according to the document, so every input to `[composite][floor][scopes]` must be length-consistent after the fix. Concrete fix: define the blend target as the same `final_video_frames` used by the composite/mux and tpad/trim all three inputs to exactly that frame count before `shortest=1`, or remove `shortest=1` only if an equivalent explicit frame gate is added. Add a post-blend frame-count assertion before mux.

8. [Hard constraints / grounding `OTRMasterAudioMux._publish_to_obs`] The hard constraint says `-c:a copy` / byte-identical audio, but the actual operator-facing OBS publish path re-encodes audio to AAC-320k. The archival final may remain PCM-identical by decoded SHA, but the watched OBS copy will not be byte-identical. Concrete fix: split the requirement:
   - archival final: `-c:a copy`, decoded PCM SHA identical to master
   - OBS viewing copy: AAC allowed, but must preserve full video duration and not use `-shortest`
   Add a duration check on the OBS copy too.

9. [Q5] The background-loop requirement is unresolved and affects where the fix belongs. Current composite tail-fill uses the floor’s own credits region. If another loop is added in composite while also tail-filling the floor HUD, the result can double-overlay credits or replace the existing HUD background. Concrete fix: for the minimal BUG-410 fix, use the floor’s existing HUD credits region as the source of truth. Defer any separate “loop background under credits” change unless verified in `_TelemetryHUDRenderer`/floor render behavior.

10. [Hard constraints / workflow-source-of-truth] The plan says JSON wiring changes go in `workflows/otr_scifi_16gb_full.json`, but does not list the required new wires. Concrete fix: explicitly update the workflow to carry the same authoritative tail/frame-budget value through:
   - floor/composite target frame count
   - §4D blend target frame count
   - mux `expected_silent_tail_frames`
   Then revalidate the workflow.

SHOULD-FIX:
1. [Grounded facts #6 / Hard constraints] “FULL audio bytes” is ambiguous against the shown mux code. `audio_pcm_sha()` compares decoded PCM, not raw WAV/container bytes. Concrete fix: state whether the protected artifact is the master WAV file bytes, decoded PCM samples, or the muxed MP4 audio payload. Do not claim raw final-MP4 audio bytes are identical unless a raw packet-level test exists. [ASSUMPTION] Existing `test_audio_byte_identical` may be checking the master WAV/golden, not the muxed MP4 payload.

2. [Q2] Muxing shorter audio onto longer video without `-shortest` should work, but the plan needs a local probe gate because the OBS copy re-encodes audio. Concrete fix: after both archival mux and OBS publish, ffprobe:
   - video duration/frame count equals expected final tail length
   - audio duration is unchanged except AAC encoder padding in OBS
   - container duration is at least video duration
   Fail if the credits tail is missing.

3. [Q3] The composite currently has an optional `gate_in` but does not use it except as a Comfy ordering input. If the composite starts deriving target length from the master audio, ordering becomes mandatory. Concrete fix: wire the audio-done gate into composite and document that composite must run after the master audio path/ledger exists.

4. [Q3] The current `assemble_silent_timeline()` error message says `assembled %d frames != audio-derived budget %d frames`. Once the intended output is audio plus silent credits tail, this diagnostic is false. Concrete fix: rename the checked value to `target_total_frames` / `video_budget_frames`.

5. [Q3] Need a fail-loud case for missing/short floor HUD. Concrete fix: if `expected_silent_tail_frames > 0` but `floor_frames <= audio_budget_frames` or the composite cannot source floor frames through the tail, return an explicit BUG-410 error instead of silently emitting a no-credits final.

6. [Q4] Specify trim-vs-pad behavior. If scopes/floor/composite are longer than target, trim; if shorter, pad with clone/black as appropriate. Do not rely on container durations alone.

7. [Tests] [ASSUMPTION] Existing mux tests may assert that `v_dur > a_dur` fails. Concrete fix: update/add tests:
   - no `expected_silent_tail_frames`: video longer than audio still fails
   - exact expected tail: passes and audio PCM SHA matches
   - wrong tail by more than tolerance: fails
   - §4D blend output retains full tail
   - OBS copy retains full video duration

OPTIONAL / NICE-TO-HAVE:
- Add the computed values to reports: `audio_budget_frames`, `floor_frames`, `expected_silent_tail_frames`, `composite_frames`, `blend_frames`, `final_frames`.
- Add a one-line BUG-410 marker in mux/composite reports when intentional silent post-roll is active.
- Add a probe script/fixture using the known failing shape: ~45.7s audio, ~65.7s floor, 25 fps.

CUT THESE (over-engineering):
1. [Candidate B] Do not pad the frozen master audio for BUG-410. It violates the byte-identical audio constraint and requires a golden rebaseline.
2. [Candidate C] Do not extend the closing theme for BUG-410. It couples this wiring fix to BUG-408/audio generation.
3. [Candidate D] Do not speed/compress the credits. It avoids the actual mux/composite clamp and risks unreadable credits.
4. [Q5] Do not build a new background-loop compositor for this bug. The existing floor HUD renderer already owns the credits visual; the minimal fix is to stop cutting its tail.