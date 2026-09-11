# Finished-diff review: My Story music parent references

This is one mechanical root correction, not a new design arc. Review the current
uncommitted diff, read-only, on the real Windows repository. Production changes
are in nodes/_otr_my_story.py; tests are test_my_story_runner.py and
test_music_cue_duration_reaches_the_beat.py. Also review the current diff in
C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide for the
existing Bible11.48/12.58 verification and independent coverage catalog.

Live baseline0327850a produced pending_20260910_175015 with two gap warnings:
shot_000_music and shot_002_music referenced nonexistent beats. The same shape
exists in published SciFi ledgers. My Story's music timeline rows have no
authored parent; the correction sets beat_id to null, retaining every line_id,
shot_id, role, cue anchor and row position. Legacy supplemental music already
uses null. Do not invent beat records, remove sentinels, change downstream
line-based render identity, rewrite story prose, or add a rejection gate.

Grounded consumers: production_ledger.set_lines preserves null;
stable_audio_theme keys cue/placement/anchor_line_id; SceneSequencer dispatches
music by role/line_id; ShotLock detects sentinels by role/line/timing and emits
downstream beat identity from line_id. The regression runs real freeze for
1/3/6 acts and breaks disabled, checks no gap warnings, durable frozen_clean,
unchanged rows/cues, plus mirror/interstitial timing. Normal tail title/style
metadata is supplied at the component test boundary.

Also: the previously missing LLM-slot annotation is restored and the existing
full-artifact repair test now compares the entire failed response, not only a
trailing marker. Bible verification is extended under existing rules; no retired
catalog is reactivated. No canonical schema/widget/wiring change is made.

Root anchor: CONFIRMED narrow parent-identity correction and unchanged consumer
ownership. Expected tests and live requalification are required before closing.
The active5080 and RunPod baseline runs loaded0327850a; do not mislabel their
warn verdicts as post-fix proof. The separate repeated spoken credit originated
in shared cleaner output and is not fixed by this change.

Return only grounded must-fix findings or a clean review. Do not edit files.
