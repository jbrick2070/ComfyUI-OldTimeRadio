# Session Handoff -- OTR Lean-Down COMPLETE, Phase 2 (live render) next -- 2026-05-29

## Core goal
The pipeline lean-down (remove dormant story machinery so the shipped path is
writer(use_exchange) -> freeze cascade -> audio -> video) is **DONE and pushed**:
steps 5-12 of `docs/LEAN_DOWN_AUDIT_2026-05-29.md` are complete. The ONLY remaining
work is **Phase 2: drive a full episode through ComfyUI's HTTP API headless and
iterate to a clean render** (the live end-to-end validation the static gates can't
do). Do it autonomously via Desktop Commander + Windows MCP -- no input needed.

## State of the art (verify first)
- HEAD `v2.0-alpha` == origin == **`00c0880`**. Confirm before starting.
- Six lean-down commits shipped: `608eb88` (5 multiturn), `6c0943a` (6 Story Room),
  `b0db85b` (7 shadow+fanout), `5e238ce` (9 polish), `46bfa77` (10 output prune),
  `00c0880` (12 bisect cruft). Step 8 = verified no-op; step 11 = VRAM guardians KEPT.
- ~19K lines of dormant machinery removed. Writer surface is now **19 widgets**
  (was 23) and **5 outputs** (script_text, script_json, news_used, estimated_minutes,
  technical_model -- the creative_writing_model output was removed, zero consumers).
- `workflows/otr_scifi_16gb_full.json`: 30 nodes, 68 links, last_link_id 230; writer
  node id=1 widgets_values len 19; technical_model output now slot 4, link 115 =
  `[115,1,4,62,4,"STRING"]`. It is the ONLY workflow with the writer node.
- Working tree: only pre-existing `ROADMAP.md` + `docs/s28_diff_tmp.txt` mods +
  untracked planning docs (not from the lean-down). No in-flight code edits.

## Verification toolkit (ALL GREEN at HEAD -- run these every Phase-2 code change)
Run from repo root with the venv python + BACKSLASH paths (forward-slash/quoted
paths get mangled by the shell layer -- retry with backslash + cmd.exe if "not found"):
- `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\tests\bug_bible_regression.py -q`  -> 23 passed (the Bug Bible repo IS present on this machine via Desktop Commander; the handoff's old "absent in sandbox" note was only about the bash/VM mount).
- `...python.exe -m pytest tests\test_core.py -q`  -> 59 passed
- `...python.exe -m pytest tests\test_audio_byte_identical.py -q`  -> 9 passed / 1 skipped (Prime Directive 1: audio byte-identity held at EVERY lean-down step).
- Full suite `pytest tests/ -q` -> **only 2 failures, both PRE-EXISTING + unrelated to
  the lean-down** (confirmed identical at the pre-work commit 608eb88):
  `test_bark_freeze_halt_bypass.py::TestWorkflowJsonNode11::test_node_11_widgets_values_includes_bypass_default`
  and `test_llm_slot_sweep.py::test_every_llm_call_site_has_slot_tag`. Treat these as
  the known baseline; a 3rd failure is a real Phase-2 regression. (Worth a separate fix.)

## Decisions made this session (don't reopen)
- Step 5: `_otr_wave0_multiturn.py` could NOT be deleted wholesale -- the KEPT Stage-3
  validators path uses `build_inloop_stage1_plan` + `line_request_to_stage1_beat`.
  Those (+ `_coerce_beat_id`, `_build_synthetic_led_data`) were migrated into the kept
  `nodes/_otr_legacy_to_stage1_adapter.py`; the writer imports them as `_OTRL2S1`.
- Step 9: full polish removal, BUT `OTR_LedgerScriptWriter` slot scheduler's
  `for_polish()` was RETAINED as a creative-slot conservative-sampling primitive (it
  wraps the KEPT `_otr_model_loader.make_polish_generate_fn`; the slot-routing tests
  pin it). It has no production caller now -- intentional, not a bug.
- The freeze cascade's `_otr_freeze_cascade.py:790  if "stage1_shadow_attempts" in meta:`
  Stage-7 shadow-critic block is now DORMANT (the writer never stamps that key post
  step 7). Left in place -- audio-path-adjacent, harmless, out of scope. Candidate for
  a future cleanup, NOT a Phase-2 blocker.
- Step 12 kept (not proven zero-ref): OTR_Visual* sidecar, OTR_CheckpointLoaderGated,
  OTR_VideoConcat, OTR_BatchProceduralSFX, OTR_ProjectStateLoader, OTR_SaveCopy,
  `_otr_lfc_context` (test-covered). Don't tombstone these without a fresh zero-ref proof.
- Downstream audit done: no consumer reads a removed meta-key (only the dormant cascade
  gate above) and zero consumers of the removed creative_writing_model output.

## Immediate next steps (Phase 2 -- per docs/LEAN_DOWN_AUDIT_2026-05-29.md "Phase 2")
1. Confirm HEAD == 00c0880. Launch ComfyUI headless under Desktop Commander so its
   stdout/stderr is AI-readable:
   `cd /d C:\Users\jeffr\Documents\ComfyUI && .venv\Scripts\python.exe main.py --port 8000`
   (DC start_process, shell:"cmd.exe"). Keep the PID; read_process_output tails
   PARSE_FATAL / tracebacks / VRAM spikes / FFmpeg. Wait for full node load + the
   "OK - All N nodes loaded" line (folder_paths-dependent nodes load fine inside ComfyUI;
   they only fail in the standalone import smoke).
2. Export an **API-format** copy of `otr_scifi_16gb_full.json` (the UI nodes/links JSON
   is NOT what /prompt accepts -- it wants the api-prompt dict keyed by node id with
   class_type + inputs). Use ComfyUI "Save (API Format)" via Windows MCP/Chrome, or
   convert programmatically; keep it beside the UI json.
3. POST `http://127.0.0.1:8000/prompt` with `{"prompt": <api_graph>, "client_id": <uuid>}`;
   capture prompt_id. A 400 here is a node/input contract error -- the body names the node
   (likely surfaces any leftover lean-down wiring gap).
4. Poll `http://127.0.0.1:8000/history/<prompt_id>` until done AND tail the launched
   process log. (Optional: ws://127.0.0.1:8000/ws?clientId=<uuid> for execution_error events.)
5. On error: read the traceback from the process log, fix autonomously (code / JSON /
   widget), re-run the relevant pytest + the 3 gates above, re-queue. Iterate.
6. **Success gate:** /history shows completed; audio output file exists + non-empty; no
   PARSE_FATAL; dialogue line count > 0; VRAM peak <= 14.5 GB; FFmpeg returned 0; audio
   byte-identity vs the pre-lean baseline still holds.
7. Record the proven API JSON + prompt_id + runtime in `BUG_LOG.md`.

## Operational gotchas (proven this session)
- Git only via Desktop Commander **cmd** shell (never PowerShell). Commit messages:
  write to `.git\COMMIT_EDITMSG` with the file tool, then `git commit -F .git\COMMIT_EDITMSG`
  (do NOT use `echo msg> file` in a chained cmd -- the redirect silently no-ops).
- For code/JSON surgery: write an assertion-guarded probe `.py` (atomic: write only if all
  asserts + ast.parse pass), run with the venv python, delete after. Inline `python -c "..."`
  gets quote-mangled.
- Operate on the REAL Windows path via Desktop Commander, never the bash/VM mount (it
  serves stale/corrupt copies). The venv python is the full Windows interpreter.
- Cache-safe ComfyUI restart before loading a mutated JSON (clear `__pycache__`, hard-refresh
  browser) -- the frontend can force a stale node definition over the new JSON.

## Open questions
- None blocking Phase 2. Two pre-existing full-suite failures (node-11 bypass + llm_slot_sweep)
  are unrelated baseline reds. The dormant freeze-cascade shadow gate + the uncalled
  `for_polish()` primitive are intentional retentions, optional future cleanups.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps (Phase 2 live
render). Acknowledge when you're ready to start." Desktop Commander + Windows MCP can do
every step (launch ComfyUI, tail its log, POST /prompt, fix + re-run, git) without Jeffrey.
