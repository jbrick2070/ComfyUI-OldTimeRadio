# pass08 (finishing) judgment -- CONVERGENCE DECLARED

Panel verdicts: 3x "NO" + Claude panelist "CONVERGED". Judge grounding
splits the NOs: the majority are misreads of locked/grounded facts; the
remainder are SPEC-TIGHTENING folds that change wording, not design.
Per the convergence rule (no new MATERIAL must-fix surviving grounding),
the campaign is **CONVERGED** after folding the items below into the
final plan.

## FOLDED (spec clarifications, no design change)

- F1 (GPT-1): M0's deliverable now explicitly includes the GRAPH SPEC
  artifact in M0_RESULTS.md -- exact topology, node-class candidates,
  widget/input names, terminal node, loader artifact names, talk-vs
  -music field diffs (the eng_ltx_video _node_candidates precedent).
  M2/CW-LTXAV-3 consumes it; a coder never guesses the graph.
- F2 (GPT-2): third config env OTR_LTX_AV_VAE (+ default lookup under
  the shared models root; download script documents the pull). Weight
  gate's three artifacts each have a path rule.
- F3 (GPT-3 partial): av_dims contract completed: W%32==0, H%32==0,
  frames%8==1, frames >= 9 (the grounded _LTX_MIN_FRAMES), frames <=
  LTX_AV_MAX_FRAMES enforced separately; nearest-valid hint names BOTH
  directions (floor/ceil multiples); raise message format pinned for
  tests. No aspect buckets (none grounded upstream).
- F4 (GPT-4): the quality gate is the OPERATOR A/B verdict by design:
  side-by-side vs the named existing 2B proof clip (the
  predicting_the_winner LTX open), labels LIPSYNC/STYLIZED/INERT +
  keep/no-keep, recorded in the sheet. Subjectivity is the project's
  standing look-QA mechanism, not a gap.
- F5 (Gemini-2): assert_usable catches the av_dims error and re-raises
  EngineUnusable with an EXISTING reason code + the dims message (no
  raw ValueError escapes into lock/validate paths).
- F6 (Gemini-3): pad-tail metrics travel STRUCTURED, not via log
  parsing: canonicalize stamps pad_tail_frames/padded_s onto the clip
  (existing extras-capable field VERIFY-AT-BUILD; else add an optional
  int/float pair to CanonicalClip -- our schema, additive). run_episode
  aggregates from clip fields; log lines remain the human surface.
- F7 (Gemini-4 + DeepSeek-1): character_description is NOT in the M4
  creative dict (grounded: creative = expression/motion/camera/
  text_prompt/source/prompt_hash) -- the talk fallback subject resolves
  from the ledger cast/images section by char_id (exact path
  VERIFY-AT-BUILD; default subject when absent). The announcer portrait
  alias spec is one concrete sentence in the ticket: on
  announcer_visual + empty char_id + engine ltx_av_talk, populate
  asset_refs[init_image] from the shipped announcer portrait recorded
  in ledger["images"] (object id from M0/grounding); missing ->
  classified pre-render fail -> chain.
- F8 (DeepSeek SF1): tickets carry "re-base the :387/:418/:490 line
  refs on the current file before editing".
- F9 (Claude): golden capture precedes driver edits inside CW-LTXAV-2;
  M1 may run parallel to M0 (only CW-LTXAV-3 is M0-gated); ship note:
  INERT-everywhere still keeps CW-1/2 scaffolding (dark, zero runtime
  cost) -- operator may revert instead.

## REJECTED (misreads -- grounded against the repo)

- Gemini-1 "assert_usable runs on a CPU planner box; NVML/node gates
  always fail": assert_usable executes in the RENDER process
  (_render_one :490, the GPU box's ComfyUI process); reading
  NODE_CLASS_MAPPINGS lazily inside a method is not a V-12 module
  -scope import (wrapper_bridge.resolve_graph_classes precedent). The
  gates stay in assert_usable.
- GPT-5 ShotLock-asserts worry: pass04 grounded ShotLock NEVER calls
  assert_usable (the registry docstring is the stale artifact and its
  correction is already in the touch list). No behavior edit exists to
  add.
- DeepSeek-2 "M1 needs the pad-vs-crop decision": the plan already
  fixes v1 = COVER+center-crop (pass02); the outpaint cell is an M0
  EXPERIMENT, not an open decision. Restated in the ticket.
- Gemini-5 (truncated SYNTH_FALLBACKS doubt): belt-and-braces rationale
  stands (pass04).

## CONVERGENCE STATEMENT

Eight grounded passes; pass08 produced zero surviving design changes.
The plan is BUILD-READY: docs/2026-06-10-ltx-av-lane/
LTX_AV_SPRINT_PLAN.md (final, includes the folds, the ticket cut
CW-LTXAV-1..4 + the M0 operator ticket/checklist).
