# OTR Lean-Down Removal Plan -- go-forward execution (2026-05-29)

Source of truth for the next dead-code pass. Synthesizes the round-2 audit
(`DEAD_CODE_AUDIT_2026-05-29.md`) with the operator's final feedback. Execution
plan, not a discussion. Branch `v2.0-alpha` only.

**Core correction:** OTR is a multi-LLM / model-agnostic workflow. The cleanup
target is dead code, not architectural flexibility. Do not let this pass quietly
turn the pipeline into a Mistral-only appliance.

---

## The dead-code test (the only test that counts)

A symbol is dead **only if it is dead across all model routes** -- not just unused
by today's Mistral default. Delete only if unused by ALL of:

1. current Mistral / local path
2. an alternate local-model path
3. the API model path
4. the Gemini / Gemma-family path
5. workflow JSON surfaces
6. tests that protect real fallback behavior

If a symbol supports provider switching, fallback, or model routing, it is **not
dead code** even if the current default never calls it. "Unused by the Mistral
path" is the wrong test.

## Hard rule -- one deletion per commit + five gates

**One deletion per commit.** The risk is not leaving dead code in. The risk is
deleting five "obvious" things at once and losing the ability to tell which one
broke the graph.

After EVERY commit, run all five gates in order; stop on the first red:

1. Zero-ref proof: `rg "<symbol>" --type py | rg -v "^tests/"` -> expect empty.
2. Workflow load test -- `workflows/otr_scifi_16gb_full.json` loads and is wired
   to current node surfaces (class names, input names, widget defaults, sockets).
3. Bug Bible regression.
4. `test_core`.
5. `test_audio_byte_identical` (byte-identity gate -- audio stays baseline).

Git per CLAUDE.md: commit msg via `.git\COMMIT_EDITMSG` + `-F`, cmd shell only,
never PowerShell. Re-wire the workflow JSON in the same commit as any node-surface
change.

## VRAM budget

Target safe operating budget: **under 16 GB VRAM** (the card) with practical
headroom for fragmentation, loaders, text encoders, and ComfyUI overhead. The
14.5 GB figure is the observed peak runtime guardrail, not a hardware limit -- do
not treat it as a fake hardware ceiling.

## Pre-flight

- Confirm `local HEAD == origin HEAD == cbee72c`, repo clean.
- Handle docs separately from code: commit `DEAD_CODE_AUDIT_2026-05-29.md` +
  this plan in their own commit, or leave them untracked. **Do not mix docs with
  any code deletion in the same commit.**

---

## Execution order (one deletion per commit, full gate run after each)

1. **Docs** -- commit the audit + this plan on their own, or leave untracked. No
   code changes in this step.

2. **Stage-7 shadow-critic dead branch.** Delete
   `nodes/_otr_freeze_cascade.py:790-843` (the
   `if "stage1_shadow_attempts" in meta:` block, ~54 LOC) AND, same commit, the
   pin `tests/test_stage7_shadow_critic_wiring.py::test_block_gated_on_stage1_shadow_attempts`.
   Safe because the gate key is never written by production code (only the read at
   :790, the comment at :768, and that test). Freeze cascade is audio-path-adjacent
   -> run the **audio byte-identity gate immediately after this commit**.

3. **`_otr_lfc_context` module.** Delete `nodes/_otr_lfc_context.py` AND
   `tests/test_lfc_context_helpers.py` (29 cases) TOGETHER, one commit. Prod
   import was removed in a prior sprint; test-only now.

4. **`OTR_BatchProceduralSFX`** -- delete `nodes/batch_procedural_sfx.py`, drop
   the `NODE_CLASS_MAPPINGS` entry in `__init__.py`, fix the voice-node list in
   `test_workflow_json_guardrails.py`.

5. **`OTR_VideoConcat`** -- delete `nodes/otr_video_concat.py`, drop registration,
   adjust the test guardrail.

6. **`OTR_CheckpointLoaderGated`** -- delete `visual/checkpoint_loader_gated.py`,
   drop registration, adjust the test guardrail, clear the docstring mention in
   `visual/flux_prompt_extractor.py`.

7. **`OTR_SaveCopy`** -- delete `nodes/otr_save_copy.py`, drop registration,
   adjust the test guardrail. **Delete it unless it was used this week for active
   export testing.** Git history is the archive; the production branch should not
   carry a manual tee/QA helper that is not part of the current workflow.

8. **`OTR_WorkflowValidator` (node 63)** -- remove the disconnected node-63
   instance from `workflows/otr_scifi_16gb_full.json` ONLY. It has no inputs and
   no linked output, so it never validates the live graph. **Keep the Python node
   class** as a manual diagnostic. Re-run the workflow load test after.

9. **`wall_clock.py`** -- delete `visual/wall_clock.py` (`WallClockEstimate` /
   `estimate()`, ~230 LOC) + its tests. Not wired into `v2.0-alpha`; do not keep
   "maybe later" code on the production branch. Restore from git history or park
   in an experimental branch if needed later.

10. **`character_regression.py`** -- delete `visual/character_regression.py`
    (~150 LOC, SSIM portrait gate) + its tests. Same reasoning as step 9.

11. **STOP.** Do not chase DRY cleanup. Tier-4 and the LoRA stack are separate
    follow-ups (below), not part of this pass.

---

## Do NOT touch -- explicit no-touch list

Architectural / multi-model plumbing -- removing any of this narrows OTR to a
single-model appliance:

- model-selection plumbing
- local / API routing
- Gemini / Gemma-family routing
- fallback-provider logic
- deferred loaders
- `OTR_UnloadAll`
- FLUX gate
- LTX gate
- HuMo tier loader
- intentional mirror helpers (`_resolve_radio_still_path` BUG-LOCAL-121,
  `_load_ledger[_with_path]` BUG-LOCAL-076)
- `_word_count` (touches writer/cascade validator pass/fail; regex vs split can
  flip a gate)
- LoRA stack (workflow nodes 60 + 61)

Mistral-Nemo-Instruct-2407 is only the **current** tested default -- not a reason
to delete any of the above. Verified-live / intentionally retained, also keep:
`nodes/project_state.py` (live via `story_orchestrator.py:53`), the 5x
`OTR_Visual*` sidecar nodes, `OTR_VRAMGuardian` + `OTR_VRAMContextTest`,
`SlotScheduler.for_polish()`, and the live flags `use_exchange` +
`enable_production_stage3_validators`.

---

## Separate follow-ups (NOT this pass)

### LoRA stack tuning (workflow nodes 60 + 61)
Same LTX distilled LoRA loaded back-to-back at `0.5` then `0.2`. Suspicious, but
it runs -- **not dead code, excluded from the lean-down.** Separate task only.
Compare four variants with a short deterministic render and SSIM / pixel
comparison (do not eyeball):

- A. current `0.5 + 0.2`
- B. single `0.7`
- C. single `0.5`
- D. single `0.2`

Keep whichever matches visual quality without breaking regression. Do not touch
it during the lean-down pass.

### Tier-4 DRY -- non-mirror duplicates only (later dedicated sprint)
These run; they are copies, not dead code:

- `_resolve_input_still` (`ltx_motion.py`, `wan21_loop.py`,
  `florence2_sdxl_comp.py`) -- highest value, but verify those backends are live
  first.
- `_voiced_beats` (`_otr_beat_validators.py`, `_otr_editor_constraints.py`) --
  trivial one-line filter dup.
- default-model-path resolvers (`flux_anchor.py`, `pulid_portrait.py`).

Excluded from Tier-4 entirely: `_word_count` and the two intentional mirrors.

---

## Parallelization & speed

Serial by design -- do NOT parallelize the commits:

- The delete -> commit -> gate loop is one-at-a-time on purpose. The audio
  byte-identity and Bug Bible gates run BETWEEN deletions so a red localizes to
  exactly one change. Collapsing commits forfeits that and is the single biggest
  risk in this whole pass.
- Step 2 (shadow-critic) is audio-path-adjacent -> isolated commit, audio gate
  alone, never batched.
- Steps 4-7 (Tier-2 nodes) all edit the SAME two shared files (`__init__.py`
  `NODE_CLASS_MAPPINGS` + `test_workflow_json_guardrails.py`). The node files are
  disjoint, but the shared registration + guardrail edits collide -> commit these
  serially (or in separate worktrees they will conflict on `__init__.py`).

Parallelizes well -- where to actually spend the speedup:

- **Phase A prep (read-only fan-out, do up front).** One subagent per target
  cluster, no write contention. Each returns a go/no-go with: the zero-ref proof
  run across ALL six model routes (not just Mistral), exact line ranges, the
  `__init__.py` registration line, the guardrail-test location, and any docstring
  mentions. This front-loads every fact the serial commit train needs, so each
  commit is then a fast mechanical edit. Suggested split: (1) shadow-critic +
  `_otr_lfc_context`, (2) the four Tier-2 nodes, (3) `OTR_WorkflowValidator`
  node-63 + workflow JSON, (4) `wall_clock` + `character_regression`, (5)
  multi-route verification of the no-touch plumbing (confirm nothing on the delete
  list is reachable from the API / Gemini / fallback paths).
- **Within each commit, run the three gate suites concurrently** -- Bug Bible,
  `test_core`, and `test_audio_byte_identical` are independent pytest runs. This is
  the biggest per-step win.
- **Steps 9 + 10 (`wall_clock`, `character_regression`)** are visual placeholders,
  not registered nodes -- verify they do not touch `__init__.py` / the guardrail
  test, and if so they are independent of each other and of steps 4-7. They can run
  as a parallel worktree pair while the 4-7 train proceeds, then replay as commits.
- **Step 1 (docs) and step 8 (workflow JSON only)** touch no Python -> they collide
  with nothing and can be slotted whenever.

Recommended fast path:

1. Fan out Phase-A prep subagents now (parallel, read-only) -> per-target proofs.
2. Serial commit train for steps 2-7 (apply prepped edit, rg proof, 3 gates in
   parallel; audio gate alone right after step 2).
3. Optionally run steps 9/10 in a parallel worktree alongside the train, then
   replay them as commits.
4. Step 8 JSON edit + workflow load test. Stop.

Net: prep collapses from serial reading into one parallel batch and each gate cycle
shortens with concurrent suites, but the commit count stays ~8 -- the serialization
is the safety feature, not overhead to optimize away.

## Final rule

A thing is dead **only if it is dead across all model routes**, not just dead on
the current Mistral default path. That is the key correction governing this whole
pass.
