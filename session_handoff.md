# Session Handoff -- OTR Lean-Down -- 2026-05-29

## Core goal
Execute the OTR pipeline lean-down: remove dead/dormant story machinery (multiturn,
Story Room, shadow/fan-out, polish surface, an unused writer output, a disabled graph
node) so the shipped path is just writer (use_exchange) -> freeze cascade -> audio ->
video. Run it one sprint at a time (REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT), then a
Phase-2 headless ComfyUI API test loop until a full episode renders clean. Do it all
autonomously via Desktop Commander + Windows MCP -- no input needed from Jeffrey.

## THE SPEC LIVES IN ONE FILE
`docs/LEAN_DOWN_AUDIT_2026-05-29.md` (committed, HEAD 4020923) is the complete, verified
go-forward plan: preconditions, keep-list, deletion inventory, the 12-step execution order,
the widget-removal method, and the Phase-2 headless plan. READ IT FIRST. This handoff only
adds the live session state + operational gotchas that file does not capture. Do not
re-derive the plan; follow it.

## Tech stack & constraints
Windows, RTX 5080 16GB (14.5GB ceiling), ComfyUI Desktop @ localhost:8000, branch
`v2.0-alpha` only (never `main`). venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
CLAUDE.md rules are in force (auto-loaded): UTF-8 no BOM; never the word "dummy" (use
placeholder/stub); audio byte-identity is sacred; wire every node change into the workflow
JSON; every LLM call tagged creative/technical. Git only via Desktop Commander cmd shell.

## What's done & decided (this session)
- Sprints/steps 1-4 of the plan are DONE. Step 4 (Node 42 cut) shipped: commit `7b67503`.
  Graph now 30 nodes / 68 links, last_link_id 230. `tests/test_core.py` 59 passed.
- Node 42 was `mode:0` (active) but its `sage_attention` widget == `'disabled'` (title:
  "...DISABLED, BUG-LOCAL-070") -> a true MODEL passthrough. Cut via the link-203 bridge
  (delete master link 69, retarget link 203 -> node 23, node 23 model input 69->203, node 71
  output list untouched). LESSON: verify a node's EFFECT (widget state), not just mode/presence.
- Verified and locked as KEEP (do NOT delete -- these are the landmines):
  `_otr_stage1_plan` (live outline; easily confused with shadow-only `_otr_stage1_call`);
  `make_polish_generate_fn` (shared conservative-sampling factory called by the live freeze
  cascade + writer base + line composer, test-pinned -- NOT polish-only despite the name);
  `OTR_VRAMGuardian` / `OTR_VRAMContextTest` (manual 16GB VRAM tools, test_core-pinned).
- Widget order in writer `widgets_values` verified via ast: [15]=enable_polish_pass,
  [17]=enable_stage1_shadow_pass, [18]=use_multiturn_dialogue, [19]=use_exchange(KEEP),
  [22]=use_stage1_fanout. They are INTERLEAVED with keepers -> remove via value-asserted
  name-keyed regen (method in the doc), never index-popping.
- Registry `_NODE_MODULES` in `__init__.py` is explicit + `importlib.import_module` per entry
  -> a deleted module's entry MUST go in the same commit or startup throws ImportError.
- All 10 workflow JSONs have zero refs to the to-be-deleted node types -> tombstoning safe.

## State of the art
- HEAD `v2.0-alpha` == origin == `4020923`. Working tree: pre-existing untracked plan docs +
  ROADMAP.md / session_handoff.md modifications only; no in-flight half-edits.
- The only graph touched so far is `workflows/otr_scifi_16gb_full.json` (Node 42 removed).
- No writer code has been edited yet -- sprints 5-10 are untouched.

## Operational gotchas (cost real time this session -- do these the proven way)
- COMMIT MESSAGES: write the message to `.git\COMMIT_EDITMSG` with the FILE TOOL (DC
  write_file), then `git add <path> && git commit -F .git\COMMIT_EDITMSG`. Do NOT use
  `echo msg> .git\COMMIT_EDITMSG` inside a chained cmd -- the redirect-in-chain silently
  no-ops through this shell layer (commit skipped, exit 0). Confirmed twice.
- DC start_process: pass `shell:"cmd.exe"` (default is powershell, which fails `cmd`/`&&`).
  Output capture on chained cmd is flaky -- after a commit/push, verify with a separate
  `git rev-parse HEAD` + `git rev-parse origin/<branch>` rather than trusting captured stdout.
- Inspect JSON/code by writing a probe `.py` via DC write_file and running it with the venv
  python; inline `python -c "..."` gets quote-mangled by the shell layer. Delete probes after.
- JSON surgery pattern that worked: an assert-preconditions -> mutate -> run the 7-check
  link-table validation -> write (`json.dumps(indent=2, ensure_ascii=False)`, no trailing
  newline to match the canonical file) script. Then review `git --no-pager diff` before commit.
- Bug Bible repo is ABSENT in this Cowork sandbox (`...\comfyui-custom-node-survival-guide`
  missing) -- run `tests/test_core.py` + audio here; run the Bug Bible on the full machine.
- The bash/VM mount serves CORRUPTED file copies -- always operate on the real Windows path
  through Desktop Commander, never the sandbox mount.

## Immediate next steps
1. Re-read `docs/LEAN_DOWN_AUDIT_2026-05-29.md` (the spec). Confirm HEAD == 4020923.
2. Execute step 5 -- Multiturn: delete `_otr_wave0_multiturn.py`, `_otr_stage2_call.py`,
   `_otr_stage2_prompt.py` + their tests; remove the writer dispatch block + the
   `use_multiturn_dialogue` widget via value-asserted name-keyed regen; remove its kwarg +
   resolved-dict key. Run `test_core.py` (+ Bug Bible on the full machine); cache-safe ComfyUI
   restart; commit + push.
3. Continue steps 6 (Story Room), 7 (Shadow+fan-out), 8 (loader cleanup), 9 (Polish -- LAST,
   per-symbol, keep make_polish_generate_fn), 10 (prune creative_writing_model output). Step 11
   is KEEP (VRAM guardians); step 12 is scan-gated cruft.
4. Then Phase 2 -- headless API testing per the doc: launch ComfyUI under Desktop Commander so
   its log is AI-readable, export the workflow to API format, POST /prompt, poll /history +
   tail the process log, iterate to a clean episode (success gate in the doc).

## Open questions
- None blocking. Two verify-at-execution items, both flagged in the doc: in step 6 the two
  shared constraint-editor tests must be EDITED (not deleted) and `test_dialogue_slot_id`
  confirmed non-live; in step 9 each polish symbol (needs_polish / polish_line /
  is_polish_refusal / _POLISH_*) must be proven enable_polish_pass-exclusive before removal.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when
you're ready to start." Desktop Commander + Windows MCP can perform every step (code edits,
git, tests, headless ComfyUI launch + log tail) without Jeffrey present.
