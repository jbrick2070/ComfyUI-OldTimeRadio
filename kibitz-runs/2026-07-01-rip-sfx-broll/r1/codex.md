VERDICT: no. The plan has the right cleanup target, but it conflates dead `sfx` speaker-role removal with still-live `sfx_cue` ambient-writing behavior, leaves fallback semantics undecided, and has a phase order that can break imports.

MUST-FIX BEFORE BUILD:
1. [Scope decision / Q1 / P3] Defect: `sfx_cue` is not only procedural-SFX audio; it is also a dialogue-writing ambient cue. `nodes/_otr_outline.py:120` defines it on every Beat, `nodes/_otr_line_composer.py:835` and `nodes/_otr_line_composer.py:1412` feed it into “SOUND IN THE ROOM”. Removing “sfx cues” as part of P3 would change voiced-line writing, not just delete dead SFX rows. Concrete fix: split `speaker_role == "sfx"` / `[SFX: ...]` / renderer plumbing from the `sfx_cue` ambient-context field. Keep `sfx_cue` for dialogue, or rename/migrate it in a separate campaign with explicit behavior tests.

2. [Phased build / P1-P2] Defect: P1 removes `Role.SCENE_BROLL`, but P2 removes the `sfx` speaker-role later. Current `nodes/otr_shot_lock.py:70` maps `"sfx": Role.SCENE_BROLL.value`; deleting the enum first leaves a live reference to a removed member. Concrete fix: either combine video-role removal and `sfx` speaker-role removal into one atomic phase, or move the `SPEAKER_TO_VIDEO_ROLE["sfx"]` deletion into P1 before deleting `Role.SCENE_BROLL`.

3. [Why / Scope decision] Defect: “writer never emits `sfx` lines” is not a stable architectural premise. The current writer contract explicitly allows `sfx`: `nodes/_otr_outline.py:69` includes it in `SpeakerRole`, `nodes/_otr_outline.py:551` tells the model `speaker_role` may be `sfx`, and `nodes/OTR_LedgerScriptWriter.py:147` treats it as non-voiced output. [ASSUMPTION] Real render ledgers may show zero `sfx`, but the code still exposes the capability. Concrete fix: reframe the plan as “remove an allowed-but-unused capability,” and make the outline schema/prompt/validators the first-class contract removal target.

4. [Q2 / Invariants] Defect: “default-to-character” contradicts “No silent fallback.” Current fallback is silent: `nodes/otr_shot_lock.py:81` returns `SPEAKER_TO_VIDEO_ROLE.get(role, _DEFAULT_VIDEO_ROLE)`, and `_DEFAULT_VIDEO_ROLE` is `background_abstract` at `nodes/otr_shot_lock.py:72`. Concrete fix: choose fail-loud. Add an explicit error for unknown `speaker_role` at the writer/freeze boundary and remove the normal-path fallback from shot/still routing.

5. [P3 / Structural touch points] Defect: the procedural-SFX audio cleanup omits SceneSequencer and workflow wiring. `nodes/scene_sequencer.py:625` still exposes `sfx_audio_clips`, `nodes/scene_sequencer.py:648` exposes `sfx_offset_ms`, and `nodes/scene_sequencer.py:840` overlays SFX clips. The real workflow node 3 still has `sfx_audio_clips` and `sfx_offset_ms` in `workflows/otr_scifi_16gb_full.json:1`. Concrete fix: if Q1 is “remove audio subsystem,” include SceneSequencer input/widget/overlay removal and the node 3 workflow JSON update in the same phase.

SHOULD-FIX:
1. [WORKFLOW-JSON change] The workflow plan only discusses node 87. If SceneSequencer SFX support is removed, node 3 also needs a positional-widget audit; its `sfx_offset_ms` widget is currently terminal, but `sfx_audio_clips` is an input in the middle of the input list. Concrete fix: expand the JSON validation checklist to include node 3 input/link integrity, not just node 87 widget truncation.

2. [Invariants] “Audio byte-identical for episodes that never had sfx” is underspecified and may be false if outline prompts/schema change and episodes are regenerated. Concrete fix: define it as ledger-to-audio determinism for fixed existing no-SFX ledgers, not fresh LLM generation after prompt/schema changes.

3. [P4 sweep] A raw grep for `"sfx"` is too broad because `sfx_cue` may remain intentionally. Concrete fix: use two sweeps: forbidden `speaker_role == "sfx"` / `[SFX:]` / renderer fields, and allowed ambient cue references if retained.

OPTIONAL / NICE-TO-HAVE:
- Add a migration/compat note for old saved ledgers containing `speaker_role: "sfx"`: reject loudly, strip, or quarantine them with a clear error path.

CUT THESE (scope / over-engineering):
1. [Q2] Cut `default-to-character`. It is a new silent fallback and directly conflicts with the stated invariant.
2. [P3] Cut global `sfx_cue` deletion from this campaign unless the goal is expanded beyond dead role/render removal. It is load-bearing for dialogue atmosphere.
3. [Ask for panel 1] Cut “confirm background_abstract is dead” as an open decision after grounding: current valid speaker roles all map elsewhere, and the only normal fallback is `_DEFAULT_VIDEO_ROLE`; solve that fallback instead of re-litigating the role.