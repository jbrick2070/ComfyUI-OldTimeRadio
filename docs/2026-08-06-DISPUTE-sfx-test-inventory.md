# ADJUDICATION -- two reviewers disagree about the SFX rip's test inventory

You are the tiebreaker. Two independent reviews of
`docs/2026-08-06-BUILD-SPEC-rip-sfx.md` reached opposite conclusions. Decide who
is right, with file:line evidence. Do not restate the spec.

Context: the operator has ruled "rip out SFX 100%". `OTR_MasterAudioMux` is the
TERMINAL publish node (node 85, order 22, the only writer of `otr/obs/`).

## THE DISPUTE

**Reviewer A (a fan-out audit)** concluded:

> `tests/test_google_video_sfx_workflow.py` is misleadingly named and contains
> ZERO SFX assertions. All five of its tests must SURVIVE COMPLETELY UNTOUCHED.
> One of them, `test_canonical_workflow_wires_clip_manifest_to_master_audio_mux`,
> IS the workflow-topology re-verification the spec demands -- running it green
> after the rip is the proof that no topology moved.

**Reviewer B** concluded:

> The test analysis is FALSE. `tests/test_google_video_sfx_workflow.py:72`
> explicitly asserts the SFX connector exists, and `:86` asserts link 278 and the
> eight-input node topology. The file does not contain "ZERO SFX assertions" and
> all five tests CANNOT survive untouched.

Reviewer B also demands, as a MUST-FIX, that the rip go further than the spec
plans:

> Remove the vestigial `clip_manifest_json` contract instead of preserving it.
> Once bed compilation is deleted the input and its manifest-only cache key have
> zero semantic effect. Delete the input and the mux argument, remove link 278
> and node 85 input slot 4 from `workflows/otr_canonical.json`, remove 278 from
> node 92's output links, and validate the resulting seven-input node. Keeping a
> wired, hashed no-op contradicts both "100%" and the root-cause rule.

The spec currently plans the OPPOSITE: keep the input wired, accept that it
becomes vestigial, change only its tooltip, and declare NO workflow topology
change.

## QUESTIONS -- answer each with evidence, not preference

1. Read `tests/test_google_video_sfx_workflow.py` yourself. Exactly which of its
   five tests would still pass if `clip_manifest_json` were KEPT (the spec's
   plan)? Which would fail if it were DELETED (Reviewer B's plan)? Give the
   line numbers. This is the factual core -- settle it.

2. THE REAL QUESTION BEHIND THE DISPUTE: should `clip_manifest_json` and link
   278 be deleted, or kept as a vestigial wired input? Weigh:
   - `CLAUDE.md` section 0 makes `workflows/otr_canonical.json` the source of
     truth and requires any node/wiring/widget change to be made IN that file in
     the SAME change as the code, then re-validated (`OTR_WorkflowValidator`,
     JSON round-trip, link/widget audit).
   - `widgets_values` is POSITIONAL; the repo has a documented drift bug
     (BUG-LOCAL-097) from mid-list edits.
   - This is the TERMINAL publish node. If it breaks, there is no episode.
   - Against that: a wired, hashed input with zero semantic effect is a lie in
     the graph, and `IS_CHANGED` still hashes it.
   State which risk you would take and why. A clear recommendation, not both
   sides.

3. If the input IS deleted, enumerate EVERYTHING that must change in the same
   commit for the canonical workflow to stay valid -- node 85's `inputs` array,
   node 92's output `links` list, `last_link_id`, the `links` array, any
   `widgets_values` consequence, and every test asserting the current topology.
   Miss nothing; this is the part that breaks the terminal node.

4. INDEPENDENT CHECK, unrelated to the dispute. Reviewer B claims the spec's
   deletion inventory omits live SFX surfaces:
   `nodes/_otr_story_brief_helpers.py` (`SFX_AUDIO_SAFETY_CLAUSE`,
   `append_sfx_audio_safety_clause`) and `nodes/_otr_shared/cloud_media_canonical.py`
   (`SFX_LOUDNESS_REFERENCE_SOURCE`, `_sfx_loudnorm_params`,
   `_normalize_sfx_stem_audio`). For EACH symbol: who calls it, does any
   SURVIVING engine reach it, and does deleting it break a surviving module at
   import or at runtime? Note `eng_cloud_video.py:46` imports the helper at
   module scope and `:886` calls it inside `_conditioned_prompt` gated on
   `self.wants_provider_sfx`.

5. Also check `scripts/soak_operator.py`, `scripts/build_silent_test_episode.py`
   and `scripts/audit_otr_full_run.py` for SFX behaviour the spec never
   classified, and say for each whether it is live, dead, or a tombstone that
   must survive.

OUTPUT: numbered answers with file:line. End with one line naming which reviewer
was correct on question 1, and your recommendation on question 2.
