# PROBLEM: the transplanted ia2v_canonical recipe articulates in ISOLATION but the lips DO NOT TALK in production episodes

Operator verdict 2026-07-02 (sound-on eyeball of the proof episode): "the lips
don't talk." The SAME recipe, run in isolation, visibly articulates. Find the
delta that kills articulation in the production path. We have a WORKING
reference render and a FAILING production render on the same box, same
weights, same day -- this is a controlled A/B; the answer is in the diff.

## The WORKING reference (isolation smoke -- lips articulate)

- Harness: `scripts/_otr_canonical_ia2v_smoke.py` (flattens the canonical
  comfy.org IA2V workflow to a raw API prompt; graph spec =
  `docs/2026-07-02-canonical-ia2v/ia2v_flat_api_prompt.json` -- this exact
  prompt RENDERED AND ARTICULATED, operator-confirmed with sound).
- Output: `output\otr\episodes\canonical_ia2v_probe\ia2v_smoke_00001_.mp4`
  (177s render). Frames: `docs/2026-07-02-canonical-ia2v/c_t*.png` -- wide
  open -> closed -> wide open across 0.5/1.7/3/4.3s.
- Key parameters: base pass **640x360**x121f @ **24fps** (SolidMask 1280x720,
  EmptyLTXVLatentVideo 640x360), refine at **1280x720** via LTXVLatentUpsampler
  x2; 5.0s audio (TrimAudioDuration 0-5); prompt EXPLICITLY narrates talking
  ("A vintage radio with a huge rubbery cartoon mouth is talking to the
  viewer, its big grille-cloth lips opening and closing naturally in sync
  with the speech..."); negative "pc game, console game, video game, cartoon,
  childish, ugly"; audio = 10s wav sliced from a real episode (announcer
  speech over a music bed) -- the SAME kind of audio production uses.

## The FAILING production path (lips frozen / no talk)

- Engine: `nodes/_otr_video_engines/eng_ltx_av.py` `RECIPE_IA2V` /
  `_build_graph_ia2v` (transplanted node-for-node @ f03d2184; topology-locked
  by `tests/test_ltx_av_ia2v_canonical.py` -- the WIRING matches canonical,
  verified; so the delta is likely PARAMETRIC, not topological).
- Proof episode: `signal_lost_jwsts_gaze_20260702_074554` (obs + repo copy
  `docs/2026-07-02-canonical-ia2v/proof_jwsts_gaze_ia2v.mp4`), histogram
  {ltx_audio_in: 6}, server log `docs/2026-07-02-canonical-ia2v/
  proof_server.log` (PLAN lines show recipe=ia2v_canonical,
  canvas=512x288, frames up to 241).
- Frames `proof2_t11.png` / `proof2_t14.png`: SOME aperture difference
  between stills 3s apart, but the operator (watching with sound) says the
  lips do not talk -- no syllable-level articulation.

## KNOWN DELTAS between working and failing (rank these, find the killer)

1. **Base-pass resolution**: production render canvas is
   `OTR_LTX_AV_RENDER_CANVAS` default **512x288** (render_driver.py ~1346) ->
   ia2v base pass = **256x144** (halved) vs canonical **640x360**. In latent
   space (/32) that is ~8x4.5 cells -- is the mouth region simply too small
   for the motion pass to articulate? (The old single-pass recipe rendered
   at the FULL 512x288; the two-stage halves it further. The canonical's
   base is 6.25x more pixels than ours.)
2. **Clip length / fps**: production bookends run **241 frames @ 25fps**
   (9.6s) vs canonical 121 @ 24fps (5s). Does LTX-2.3's audio coupling
   degrade past ~121-161 frames? LTXVConditioning frame_rate=25 vs 24?
3. **Text prompt**: production text prompts describe the SCENE (brief-driven
   radio subject, e.g. get_open_subject / ltx_scene_open composer in
   `nodes/_otr_story_brief_helpers.py` + render_driver M4 path); they do NOT
   say the subject IS TALKING. The canonical prompt narrates the act of
   speaking. (Known open thread from the transplant.)
4. **SolidMask dims**: engine passes width/height = the RENDER canvas
   (512x288); canonical passed the REFINE target (1280x720). Does the
   audio-latent noise mask scale matter?
5. **Audio slice content**: per-beat slices from the master (speech over
   music bed) -- same class as the working reference audio, but bookend b000
   is MUSIC-ONLY (no speech). b001/b005 have speech. Character beats have
   dialogue. NONE talk per the operator.
6. **Negative prompt**: engine default `OTR_LTX_AV_NEGATIVE`
   (`_LTX_DEFAULT_NEGATIVE` in eng_ltx_av.py) vs canonical's short negative.
   Could our negative suppress mouth motion tokens?
7. **Still content**: production stills are brief-driven radio faces (mouth
   present, e.g. `stills/still_*_radio_face_169_*.png` in the episode dirs);
   the working reference used `batch_face1_mouth_still.png` (the lanterns
   mint). Similar but not identical stills.
8. **Frame count 8n+1 vs the audio window**: engine computes length from
   target_frame_count (`_AVD.next_8n1`), canonical used fps*seconds+1.

## What the kibitz must deliver

- The RANKED root cause (with repo-grounded evidence) for why isolation
  articulates and production does not.
- The MINIMAL A/B probe plan to CONFIRM the top suspect(s) empirically on
  the box (one variable at a time, using the EXISTING isolation harness --
  e.g. re-run the flat canonical prompt at 256x144 base / 241 frames / 25fps
  / production prompt text, and see which change kills articulation).
- The production fix (engine/env/driver change) once confirmed, with the
  exact files/lines.
- Constraints: 16GB VRAM ceiling 14.5GB; NO FALLBACKS; audio byte-identical
  (mux-LAST untouched); the workflow JSON only changes if a node/widget
  changes; suite + Bug Bible green per chunk.
