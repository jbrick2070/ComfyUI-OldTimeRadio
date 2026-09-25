## 2026-09-25 -- HEAD e7a4806a (main) -- CODE + DOCS (row 0b asset_cleanup built; stale pointers; one local copy)

Did (5080 window, now the only window; the server on :8000 was not touched):
  3e01b1c6: row 0b built as written. `asset_cleanup` on OTR_LedgerScriptWriter,
  three full-text labels, default off, the trailing widget (canonical node 1
  widgets_values[35], descriptor last in inputs[], no dst_slot moved; 24
  variants regenerated, --check 24/0). The writer stamps the first word beside
  delivery_intent, and inside the replay branch with led.save() before the
  wire is built; `asset_cleanup` is run-volatile on a replay. New pure module
  `nodes/_otr_asset_cleanup.py` (plan + execute, stdlib only), called ONLY
  from `OTRMasterAudioMux._asset_cleanup`, the last step of mux() after the
  preview frame. Linked folders inside the episode are a REFUSAL (the row's
  test list and the standing ruling; the row's prose said unlink-as-entry --
  refusal is the stricter reading and is what shipped). Paths compare by
  realpath so a junctioned output tree still binds. Docs: RUN.md "Saving disk
  space", the standing ruling, the janitor header.
  c292539c: two stale pointers -- `__init__.py` named `otr_4060_floor` as live
  and said workflows/variants/ was deleted; the rulings named
  docs/GO_FORWARD_ARCHIVE.md as a live file (it is `a0ff6c8c~1`).
  e7a4806a: README widget row + "Where things land" paragraph; INSTALL.md's
  "one thing the pack deletes" is now two.
  Local cleanup (operator): one OTR copy on disk. Removed the stray baseline
  worktree and the 2026-09-24 agent worktree (its WIP commit 40aef8ab stays on
  local branch worktree-agent-aaf3612b6248a9d9e; main already supersedes all
  of it except a client.py rename and three EXPECTED_RED G2 entries, which
  are a matrix-row call for him). Six retired projects moved out of
  custom_nodes and deleted by the operator; their six GitHub repos deleted by
  him too (confirmed gone).
Suite: full run on the tree, 16 reds; the 16 re-run at fe17f426 in a temp
  worktree outside custom_nodes: 12 inherited. The 4 new ones were mine and
  are fixed in 3e01b1c6: the one-act writer count pin, and a ledger-singleton
  leak from the new replay tests into test_audio_cache_wiring (alphabetically
  next) -- the fixture now restores `production_ledger._CURRENT`. New file 40
  passed, incl. a real junction and two real Windows locks. build_variants
  --check 24/0. Touched .py AST-clean, no BOM, LF. Bug Bible not run.
Models: QA on 3e01b1c6 -- Sonnet subagent (REFUTE, executed the planner and
  executor on temp trees: real junction refused and target untouched, locked
  file skipped and named with the done receipt still written, replay of a
  `full` source with the widget off leaves no key on wire or disk): HOLD, no
  must-fix. Cursor lane cursor-grok-4.6-high in ask mode via kibitz: critiqued
  the brief, not the code; its three code claims checked -- linked-dir
  policy (shipped as refusal, above), `peek_ledger` DOES exist
  (production_ledger.py:808, so row 0b's review record was wrong about
  ChatGPT), and the stem-prefix identity (held: a replay's silent video sits
  in its own folder, so the inside-the-folder check refuses). Antigravity Gemini 3.8 Flash (High) via
  kibitz (r4): ran its OWN junction and locked-file probes independently
  of Sonnet's, got the same results (refused/target untouched; skipped,
  named, receipt still written), and read the mux ordering, the replay
  drop test, and build_variants --check the same way. HOLD, no must-fix;
  one nice-to-have (a debug log on an empty-dir full run) not taken.
  Codex out of credits until 2026-09-29 14:59 PDT. Reviewers left HEAD and
  the tree untouched (checked).
Next risk (not a defect): a replay keeps its source's delivery_token, so the
  token alone cannot tell a replay from its source; the stem check and the
  inside-the-folder check are what do. A fresh token per replay import would
  close it.
Next: 2.3.4 carries asset_cleanup, when he says (plan, registry section).
  4060: re-cloned main at c292539c, confirmed node 1 is 36 wide, has the GO for
  Part B once his ComfyUI Desktop reinstall is done. The plan's "Live box"
  note describes the 1-act chain that was due to end at 08:00.

## 2026-09-25 -- HEAD eaf07017 (main) -- CODE + PLAN (orphan cleanup d245cc27, asset-cleanup design row 0b)

Did (5080 window; the 1-act chain on port 8000 was not touched):
  d245cc27: the orphan/stale-reference cleanup. `_validate_scene_envelope`
  wired at `_build_envelope`'s one call site (standing ruling #3 satisfied);
  the VRAM sentinel chain ripped whole (`vram_sentinel`, `force_vram_offload`,
  `_CLEANUP_CALLBACKS`, `register_vram_cleanup` and the one registration in
  story_orchestrator) on the operator's word that the VRAM measures say
  little (ruling #4 records it); the Bark squeal metric,
  `slot_matrix.eligible_engines_for_role`, `SCIFI_ORCHESTRA` and
  `MAX_PROVENANCE_NOTE_CHARS` gone with their test mentions; about twenty
  stale references repointed (docs/ -> apple/ and the sibling
  vram-recipe-lab/docs/, config/profiles/ -> the matrix row). Two review
  recommendations were wrong and were not applied: the `otr_soak` reset
  marker (its launcher exists) and `animatediff15_v2_video` (a RETIRED_ENGINE_IDS
  tombstone). Only story_orchestrator's own two hunks were staged; the other
  window's unfinished patch at ~774 and ~901 stays in the working tree.
  eaf07017: GO_FORWARD_PLAN row 0b, the asset-cleanup design (`asset_cleanup`
  off / partial / full on the writer; the delete is the last step inside the
  mux after publish, ledger stamp and canvas preview; identity guard from
  BUG-LOCAL-014; partial is a delete-list plus a receipt in the surviving
  ledger). Measured: 90.5 GB in 135 media-bearing episode dirs, partial keeps
  0.3%. A ChatGPT design-review prompt was handed to the operator.
  Output-tree facts for the next window: otr/ top level carries more than
  episodes/ + obs/ (audio, replay_bundles, _gemini_judge, _probe_clips,
  _quarantine_model_translated_20260920, a smoke still, the chain scripts and
  logs); the ComfyUI-Installs output/otr tree is 8.3 GB and last written
  2026-06-13; 2,291 episode dirs hold only ledger and text (his hand cleanup).
Suite: scoped set 1062 passed / 3 skipped before d245cc27; build_variants
  --check 24 / 0; touched .py AST-clean, no BOM, no CRLF. Bug Bible not run.
Next: fold whatever survives of ChatGPT's lettered answers into row 0b, then
  build 0b. 5080: TEST_WAVE Part A once the chain ends (08:00 local). The
  10-file red-test patch stays uncommitted and separate.
Models: QA on d245cc27, all three no must-fix -- Sonnet subagent (executed
  the envelope guard for act_count 1..7, grep for every ripped name), Cursor
  lane cursor-grok-4.6-high in ask mode via kibitz (no defect in the commit;
  it mostly critiqued the brief's scoping), Antigravity Gemini 3.8 Flash
  (High) via kibitz. Codex is out of credits until 2026-09-29 14:59 PDT.
  Reviewers left HEAD and the tree untouched (checked).

## 2026-09-25 -- HEAD d167f0ab +branch claude/practical-hamilton-ftvd2j -- CODE + PLAN (AnimateDiff auto-download, required_models, test wave)

Did (cloud window, no hardware touched; the 5080 chain was not reached):
  AnimateDiff weights now download at queue time. `_SOURCES` in
  `nodes/_otr_visual_assets.py` gained v3_sd15_mm.ckpt and
  v3_sd15_adapter.ckpt (guoyww/animatediff, Apache-2.0),
  animatediff_lightning_8step_comfyui.safetensors (ByteDance, OpenRAIL-M) and
  vae-ft-mse-840000-ema-pruned.safetensors (stabilityai, MIT) -- all ungated
  on the Hub API. Each lane names its own files through the new
  `GhostSignalEngine._weight_tokens()`; the three lanes joined `_COVERED`, so
  the dropdown matrix reads them "auto". Measured before the change: with an
  empty server the runner gate refused otr_8gb_animatediff and
  otr_16gb_animatediff over the motion module and adapter, and every other
  shipping row passed. After: no shipping row is refused
  (`test_no_shipping_row_is_refused_on_an_empty_server`).
  `preflight.required_models` now accepts weight FILENAMES only (schema in
  `capability_profiles.py`). Removed the writer repo ids from both
  AnimateDiff rows (the 8 GB row named gemma-4-E2B-it; its writer is
  Qwen3.5-4B) and replaced otr_8gb_video's logical ids with
  ltxv-2b-0.9.8-distilled.safetensors and t5xxl_fp16.safetensors. The
  runner's report-only branch for ids is gone. Variant JSONs unchanged; three
  launch recipes regenerated.
  Plan cleaned: dead links fixed, the Google row cut to its fork, the
  resolved word-counter row removed (WORD_RE is already Unicode), Shakespeare
  moved to Parked with a git pointer. Row 0 and 0a specs restored as
  apple/ROUTE_DELETION_PLAN.md and apple/HF_HOME_WINDOWS_PIN.md and
  re-grounded. The wave is apple/TEST_WAVE.md.
  Registry, read 2026-09-25: 2.3.3 Active, 2.3.2 Active; 2.3.0, 2.3.1, 2.1.5,
  2.1.6 Flagged.
Suite (Linux sandbox, not the Windows venv): scoped set 1,631 passed; two
  red, both already red on d167f0ab before this change --
  test_lane_preflight_matrix.py::test_g2_canvas_truth and
  ::test_g4_admission_honesty. Full suite result is on the PR. Bug Bible not
  run.
Next: 5080 -- TEST_WAVE Part A once the chain ends. 4060 -- Part B on a head
  that carries this branch. Then Part C. Code rows 0 and 0a stay open.
Models: Sonnet QA on the pushed diff (see the PR).

## 2026-09-25 -- HEAD 9b9e0766 +handoff (main) -- RENDER (16 GB 1-act rotation, registry 2.3.3)

Did: Stills descriptions now say z_image_turbo. Lumina is not a default in any
  profile JSON. Mac stays sd15. Cloud and Google rows were not touched.
  Commits 248bf651 (descriptions) and 9b9e0766 (pyproject 2.3.2 -> 2.3.3).
  Registry publish Action 36103232501 succeeded. GET
  /nodes/comfyui-old-time-radio/versions shows version 2.3.3 status
  NodeVersionStatusPending (created 2026-09-25T06:31:46Z). Zip contents were
  not checked. Active waits on Comfy-Org's scan.
  Live 1-acts on this 5080, canonical graph, act_count forced to 1. Published
  to C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\:
  covenant_ink_20260924_232810__arch__stmo__zimg__koko__pubd__g412__sa3_final.mp4
  (otr_16gb_still),
  notched_key_20260924_234023__scif__l25v__zimg__koko__orig__g412__sa3_final.mp4
  (otr_16gb_video),
  black_fog_20260925_011016__shst__n16f__zimg__koko__sspr__g412__sa3_final.mp4
  (otr_16gb_foley). All three filenames say zimg.
  Mime prompt ea876b5c-3646-4b37-8603-a92fe79a4657 was still running at
  handoff (server log on beat shot_001_b15, ltx25_native_mime_16gb).
  AnimateDiff (otr_16gb_animatediff) is next. Chain script
  C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_chain.ps1 repeats stills,
  video, foley, mime, animatediff until 2026-09-25 08:00 local or the first
  failure. Server log: C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_server.log.
  Do not kill python and do not free port 8000.
  Full pytest suite and Bug Bible were not run. A render occupied the box.
  Dirty and uncommitted, do not git add with other work: a partial red-test
  patch (worktree commit 40aef8ab, branch
  worktree-agent-aaf3612b6248a9d9e). The EXPECTED_RED ledger edit and the
  google client quota_state rename were rejected and are not in the tree.
  The rest is unstaged: apple/evidence/video_evidence_manifest.json,
  nodes/story_orchestrator.py (LLM-slot comments only), and the test files
  named in git status. apple/MACHINES.md shows modified from CRLF only.
  Operator asked for a Stable Audio 3 house/techno check of prompt, inputs,
  temperature, and seed. Not done. Rendering was the priority. The
  music-is-done ruling still covers cue-wording chases.
Current step: mime 1-act still on the GPU. AnimateDiff has not started.
Next: leave the chain alone until 08:00 or a RESULT FAIL. Then take the
  next GO_FORWARD row. Do not reset the box while :8000 is serving this chain.
Models: no panel. Description strings plus the version bump. No Composer or
  Sonnet QA on that diff.
Commits: 248bf651, 9b9e0766. The sha above is the second-to-last on the
  branch; the last is this handoff commit.

## 2026-09-25 -- No quant pack, no Wan, no harnesses; LTX 2.5 downloads itself

Branch `claude/practical-hamilton-ftvd2j` (draft PR #5). Operator directives:
no quant-format mentions outside the Bug Bible, no harnesses, no rigs, no saved
fixtures, delete all history (every dated `docs/2026-*` folder, `kibitz-runs/`,
this log's older entries and `GO_FORWARD_ARCHIVE.md` -- all recoverable from git
history), and "less friction for the end user, auto download things to work".

What the pack is now: the canonical graph plus 24 variants generated from
`config/workflow_matrix.json` rows -- the only workflow definitions. The LTX 2.5
lanes are native safetensors on stock loaders and fetch their own weights at
queue time from ungated mirrors (about 25 GB for the 16 GB lanes). Removed:
`wan_ti2v`, `fastwan_8gb`, `ltx_video`, `ltx_audio_in`, the quant LTX 2.5 lanes,
the DMD sampler node (24 nodes now), 98 harness scripts, 72 rigs,
`tests/fixtures/`. Needed reference docs moved to `apple/`.

Verification: full suite on the Linux cloud box (no GPU, no Windows paths) --
zero new failures against the 8ffb09d baseline (289 baseline reds, 225 now).
`build_variants --check`: 24 variants, 0 failures. Sonnet refute-QA ran on the
pushed diffs; findings folded in. NOT yet proven on hardware: the LTX 2.5
queue-time download and the 16 GB silent lane on its new weights. Bug Bible not
run from here -- its repo is not reachable in the cloud session.
