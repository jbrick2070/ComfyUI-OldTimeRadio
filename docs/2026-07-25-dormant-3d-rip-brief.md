# DORMANT 3D TALKERS -- rip or leave? (operator-directed consult)

**Decision owner:** operator. **Seats:** codex `gpt-5.6-sol` (high) + Claude
(anchor + judge). Baseline HEAD `0bc863f4` on `v2.0-alpha`.

**Operator's stated instinct:** *"ripping it out if it isn't being used is the
right thing to do."* The question is whether the code supports that, and if so
what the correct rip ORDER is.

## What is dormant

`triposg_talk`, `hunyuan3d_talk`, `trellis_talk` (in
`nodes/_otr_video_engines/eng_character_3d.py`, 441 lines) and `triposr` (in
`eng_triposr.py`, 159 lines). All four were UNREGISTERED on 2026-06-29 under
the "registry IS the menu" rule (C3), with a re-entry note in
`_otr_video_engines/__init__.py`: re-add the import + `@register` + a
CAPABILITIES row in the SAME change when a real forward ships. Their
`render_clip` raises `NotImplementedError`.

## Grounded inventory at HEAD (verified, not assumed)

- **31 engines registered; none of the four dormant ids.** Roster audit is
  clean: `{'missing': (), 'unexpected': ()}`. 31 CAPABILITIES rows, none for a
  dormant id. So they are cleanly OUT, not half-in.
- **`requires_mesh_portrait` has ZERO live declarers.** Driving the live
  registry over all 31 engines: no registered engine declares it. Every
  consumer of that capability is therefore currently dead weight at runtime.
- **`requires_mesh_fodder` is a DIFFERENT attribute and IS live** -- declared
  by `mesh_stage`, which is registered. The mesh-FODDER lane is independent of
  the 3D-TALKER lane; they share vocabulary, not code.
- **NO PRODUCTION FILE IMPORTS EITHER DORMANT MODULE.** The only hard
  dependencies are three TEST files: `tests/test_video_character_3d.py`,
  `tests/test_video_triposr.py`, and `tests/test_still_aspect_and_labels.py:70`
  (which imports the three classes directly precisely BECAUSE they are no
  longer in the registry).
- Production files that MENTION the ids/attributes without importing:
  `render_driver.py` (16), `otr_image_director.py` (13), `schemas.py` (8),
  `resolver.py` (2), `registry.py` (2), `otr_silent_composite.py` (2),
  `portrait_ledger.py` (1), `role_compat.py` (1), `directory_clip.py` (1),
  `otr_image_gen_dispatcher.py` (1), plus `scripts/otr_video_soak.py` (6),
  `scripts/otr_video_dep_pilot.py` (4) and the `_otr_b_spikes` probes.

## Why this is being asked NOW

A defect was found and fixed today in exactly this dormant surface:
`otr_image_director.three_d_locked_slots` resolved the PICKED video engine
rather than the EFFECTIVE one, so a force map that routed a role ONTO a
mesh-portrait engine would have left the fail-closed "no per-beat mesh rebuild"
lock disarmed. It is the same picked-vs-effective class as the decapitation bug
fixed this morning in `aspects`. It sat wrong because nothing exercises it.
**Dormant code that guards a live invariant rots silently and is only found by
audit.** That is the real cost being weighed, not the 600 lines.

## The question for the panel

Recommend ONE: **RIP**, **KEEP AS-IS**, or **KEEP + FENCE** (leave the code but
add a guard/test that makes the dormancy explicit and self-checking).

Answer these specifically, with file:line:

1. **Blast radius of a full rip.** If `eng_character_3d.py` and
   `eng_triposr.py` are deleted, what else MUST change in the same commit?
   Enumerate: the three test files, the `requires_mesh_portrait` field in
   `schemas.py:273,288-290`, the whole `three_d_locked_slots` /
   `enforce_3d_granularity_lock` path in `otr_image_director.py`, the
   `render_driver.py` mentions, `resolver.py`, `portrait_ledger.py`,
   `role_compat.py`, `directory_clip.py`, `otr_silent_composite.py`, and the
   soak/pilot scripts. For each, say whether it is a DELETE, an EDIT, or NO
   CHANGE, and whether removing it loses a live behaviour.
2. **Is the 3D-lock path genuinely orphaned, or does it also guard something
   live?** `three_d_locked_slots` raises fail-closed via `_is_3d_engine` when an
   engine's capability cannot be read (`otr_image_director.py:98-120`). If the
   capability disappears, does any LIVE lane lose a real protection --
   specifically the mesh_stage / mesh-fodder lane, which uses the DIFFERENT
   attribute `requires_mesh_fodder`?
3. **What does a rip COST that keeping does not?** The re-entry note says these
   return when a real forward ships. Is there evidence in the tree (ROADMAP,
   docs, plan files) that a 3D forward is actually planned, or is this a
   scaffold from an abandoned subproject? Cite what you find.
4. **If RIP: the correct ORDER**, as independently green pushed chunks, so the
   tree is never half-ripped. Name the chunks and what proves each one.
5. **If KEEP: the minimum fence** that would have caught today's defect
   automatically -- e.g. a test asserting no live engine declares
   `requires_mesh_portrait` so the dead branch is provably dead, or an
   assertion that fails loudly the day someone re-registers a 3D talker without
   re-auditing its consumers.

## Constraints the answer must respect

- "Registry IS the menu" (C0/C3): a registered engine needs a CAPABILITIES row,
  and an unregistered engine must not be selectable.
- `workflows/otr_canonical.json` must stay byte-identical unless a node,
  widget, input or link genuinely changes.
- Every chunk: focused tests + full Windows suite + Bug Bible + commit + push.
- Do NOT recommend deleting a test that is the only proof of a live behaviour.

**Answer format:** a recommendation line (RIP / KEEP AS-IS / KEEP + FENCE),
then the numbered answers with file:line, then the ordered chunk plan if RIP.
State plainly where you are uncertain rather than guessing.
