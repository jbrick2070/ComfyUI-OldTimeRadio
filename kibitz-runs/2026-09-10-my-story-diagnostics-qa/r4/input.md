# Independent finished-diff QA: reported diagnostic corrections only

Inspect actual current git diff against59a40131. Production changes are four
files: nodes/_otr_video_engines/render_driver.py, nodes/otr_shot_lock.py,
nodes/otr_master_audio_mux.py, nodes/otr_image_gen_dispatcher.py. Five test
files extend existing coverage. No edits; report actionable defects or clean.

Expected: optional keyword phase=render on build_request_from_shot, solely
ShotLock passes cast_preflight. Only missing-still diagnostic level/text changes;
request bytes/hash, placeholder and family/postimage checks unchanged. ShotLock's
typed deferred gap catch is INFO; untyped errors propagate. Default/unknown phase
still warns missing image. Real renderer uses default phase.

Successful active-episode rename reconciliation is INFO in the three owners.
Remove only the successful stills warnings.append; preserve earlier warnings,
foreign freeze/identity/path refusals and exceptions. Do not claim audio byte
identity is checked by path helper; downstream mux owns PCM proof.

Focused cases pass except pre-existing test_canonical_workflow_wires_clip_manifest
_to_master_audio_mux (assert291==289), verified in tmp/my_story_boundary_full.xml.
Full regression/Bible still running. No new media leg. The4060report explicitly
identifies these as warning-only follow-ups; no duplicate PBUG invented. Existing
WIRE-W2 typed-deferral and BUG-12.66 active rename identity contracts stay intact.

One CLI QA, not a design arc. The separate cross-machine R3 architecture campaign
is out of scope. Ignore inherited diff.txt/diff_utf8.txt and review artifacts.
