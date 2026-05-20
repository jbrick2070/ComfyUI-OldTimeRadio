# Sprint H -- forward plan
Branch: v2.0-alpha @ dc0774b (+ uncommitted)
Constraints: ASCII only, `--` not em-dash, "placeholder"/"stub" not dummy, audio C7 byte-identity locked, no new branches

---

## 1. Decisions

1. **Supervisor architecture: two-process design.** Supervisor spawns `scripts/worker_iter.py` per iteration via `subprocess.Popen`. Worker owns one iter: starts ComfyUI as its direct child (no .bat wrapper, no PowerShell middleman -- a wrapped launch breaks `taskkill /T` tree-walking), queues one run, writes a per-iter result file, exits. Supervisor only kills the worker's PID tree.

2. **Python-sweep routine (two scopes, both logged).**
   - **Pre-launch sweep (outer .bat, blanket).** `scripts/sweep_and_launch.bat` runs before the supervisor exists. Kills every `python.exe` and `pythonw.exe` on the box, ComfyUI or not. Then launches the supervisor. Safe because no python the loop cares about is alive yet. Logs every killed PID with command line and timestamp to `logs/sweep_prelaunch.log`.
   - **Between-iter sweep (supervisor, filtered).** Supervisor calls `scripts/sweep_python_excluding.bat <SUPERVISOR_PID>`. Kills every python on the box EXCEPT the supervisor's own PID. Logs every killed PID with command line and timestamp to `logs/sweep_betweeniter.log`. No silent kills. PID passed via argv, never hardcoded.
   - `kill_all_python.bat` is removed from the supervisor codepath entirely. Remains in repo as a manual tool, never invoked by the loop.
   - Operator note: between-iter sweep WILL kill unrelated python tools (editors, jupyter, test runners) running on the box. Acceptable for a closed overnight machine. Do not run the loop while doing other python work.

3. **OOM cleanup: three layers, structured as `try/except` not blind `finally`.**
   - **Layer 1 -- outer wrapper in `request_slot()` (already applied at line 741+).** `try` build, `except` call `unload_llm()`, re-raise. In the working tree, uncommitted.
   - **Layer 2 -- inside `load_llm()` itself.** Currently at `nodes/_otr_model_loader.py:549-552` the exception handler re-raises as `ModelLoaderError` without touching the local `model` reference. Patch: `try` build model + tokenizer, on success return cache_entry; on `except Exception` move `model` to CPU if bound, `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, then re-raise as `ModelLoaderError`. **Not `finally`** -- a bare `finally` would unload on the success path too.
   - **Layer 3 -- success-path unload.** Explicit `unload_llm()` call placed AFTER `OTR_LedgerFreezeCascade` emits the frozen `script_json` and BEFORE any FLUX/HuMo/LTX node loads. Two options: (i) graph-level via the existing `OTR_UnloadAll` node (id 24 in `_full.json`) -- verify its `dependencies` input wires from the writer/freeze block and any FLUX node depends on `OTR_UnloadAll`'s output. (ii) node-side -- add `unload_llm()` at the end of `OTR_LedgerFreezeCascade.execute`. Prefer (i) if wiring is already correct; fall back to (ii) if any doubt. Verify no downstream node lazily re-invokes style/critic/revision after this point.

4. **C7 audio byte-identity: protect via separate bug-hunt workflow file.** Writer drives TTS, TTS drives audio bytes. Swapping the writer default changes C7. Resolution: do not touch `workflows/otr_scifi_16gb_full.json`. Create `workflows/otr_scifi_16gb_bughunt.json` as a clone with Gemma defaults in node 1. C7 default-config resolution stays pointed at `_full.json`; bug-hunt loop targets `_bughunt.json`. After cloning, run `normalize_workflow_widgets.py` against BOTH files to prevent the clone from being born with stale widgets.

5. **C7 verification path.** Use the existing `tests/test_audio_byte_identical.py` (which does not require the full video pipeline) for C7 verification while the bug-hunt loop runs `_bughunt.json`. Full-pipeline C7 verification against `_full.json` deferred until the Mistral loader path is resolved.

6. **Gemma model registry key: `google/gemma-4-E4B-it`** (confirmed against `/object_info` dropdown and on-disk cache at `C:\ComfyUI-Models\huggingface\hub\models--google--gemma-4-E4B-it`). No `[NOT DOWNLOADED]` suffix when downloaded. Use this exact string in node 1 of `_bughunt.json` for both `creative_writing_model` and `technical_model`.

7. **Commit strategy: four narrow commits, push all this session.**
   - **A**: workflow normalization + node 12/55 widget fixes (verified).
   - **B1**: failure-path cleanup -- `request_slot()` wrapper + `load_llm()` except-cleanup. Defense-in-depth; safe to ship.
   - **B2**: success-path `unload_llm()` after Ledger Freeze (graph-level or node-level per §1.3 option (i) vs (ii)).
   - **B3**: new `workflows/otr_scifi_16gb_bughunt.json` with Gemma defaults (additive, no impact on `_full.json`).

8. **Target shift: one controlled diagnostic run first, not three cleans.** Then 3 consecutive cleans.

9. **Kill ComfyUI between every iter during bug-hunt.** Leak isolation beats startup speed.

10. **Retry policy: one OOM retry max, not three.**

11. **No auto-bisect on target_words.**

12. **Wire `OTR_WorkflowValidator` into worker preflight.** Validator runs as a ComfyUI node and needs live `/object_info` schemas, so it runs at worker step 6 (after `/system_stats` returns 200), not at offline pre-flight. Failure short-circuits before `/prompt` POST.

---

## 2. Per-iter recipe (supervisor + worker)

Supervisor (long-lived, never touches GPU directly):
1. For iter in range(N):
2.   Sweep: invoke `scripts/sweep_python_excluding.bat <SUPERVISOR_PID>`. Logs every killed PID + cmdline.
3.   Choose a free TCP port (probe loopback starting at 8000, first unbound port wins). Pass to worker via `--port`. Fixed `8000 + iter` is acceptable for the very first smoke only, not for repeat runs.
4.   Spawn worker via `subprocess.Popen(["python", "scripts/worker_iter.py", str(iter), "--port", str(chosen_port)])` detached. Capture worker PID.
5.   Wait on worker with 20-min timeout. On timeout: `taskkill /F /PID <worker_pid> /T`, then targeted ffmpeg sweep (see step 9 of worker).
6.   Read `logs/worker_iter_<n>.json`.
       - If file missing or unparseable: classify `worker_crash`, write that row to master log.
       - Else: append worker's class to master log.
       Supervisor alone appends to `logs/overnight_results.jsonl`. Worker never writes to the master log (avoids race).
7.   Stop conditions check (section 7). Else sleep 10s, continue.

Worker (one iter, then exits):
1. **Always write a result file on exit, even on crash.** First action of the worker: install an exit hook that writes `logs/worker_iter_<n>.json` with a partial/error row if the worker exits before normal completion.
2. Pre-flight port check: confirm chosen port is free via `netstat -ano | findstr :<port>` (no output = free). If occupied, write JSONL class=`port_occupied`, exit.
3. Launch ComfyUI as direct child process (no .bat wrapper) with Blackwell flags, explicit HF_HOME, `--port <chosen_port>`. Capture ComfyUI PID for teardown. Use the inline command from `scripts/start_comfy_h0_baseline.bat` (set TORCH_SDPA_BACKEND=math + set HF_HOME=C:\\ComfyUI-Models\\huggingface, then the venv python invocation against `C:\\Users\\jeffr\\AppData\\Local\\Programs\\ComfyUI\\resources\\ComfyUI\\main.py --port <port> --highvram --force-fp16 --cuda-malloc --user-directory C:\\Users\\jeffr\\Documents\\ComfyUI`). Redirect stdout+stderr to `logs/comfy_session_iter_<n>.log`.
4. Poll `http://127.0.0.1:<port>/system_stats` until HTTP 200, 60s timeout. Verify the responding process owns the port: `Get-NetTCPConnection -LocalPort <port> | Select-Object OwningProcess` (PowerShell) or `netstat -ano | findstr :<port>` and confirm OwningProcess matches the ComfyUI child PID. Catches stale orphan binding the port. If timeout or PID mismatch, class=`comfyui_startup`.
5. Live validator pass: `OTR_WorkflowValidator` against the target workflow via `/prompt`-style submit (validator-only subgraph, no execution). On error: class=`graph_widget`, exit.
6. Pre-flight VRAM check via `nvidia-smi`. Target idle: under 2 GB.
   - If 2-4 GB: log warning, continue.
   - If > 4 GB: attempt filtered-kill of known ComfyUI process tree, recheck after 5s.
     - If killed something AND VRAM now under 4 GB: continue. Log class=`orphan_detected` (for the JSONL only) but proceed with the iter.
     - If still > 4 GB after kill: class=`vram_contaminated`, exit. Do not run workflow against contaminated card.
7. POST `workflows/otr_scifi_16gb_bughunt.json` to `/prompt`. Capture prompt_id.
8. Poll `/history/<prompt_id>` every 5s, 15-min timeout.
9. On result: tail last 200 lines of ComfyUI log. Classify outcome. Write `logs/worker_iter_<n>.json` (overwrites the exit-hook stub from step 1).
10. Teardown (unconditional, success or fail):
    - `taskkill /F /PID <comfyui_pid> /T` (catches ffmpeg children of the worker tree first)
    - If phantom ffmpeg still detected via task list scan, escalate: `taskkill /F /IM ffmpeg.exe` AND log `ffmpeg_global_kill` event to `logs/teardown.log` so we know we hit the unrelated-ffmpeg edge case.
    - exit.

Failure classes (fixed dictionary, no overlap):
- `graph_widget` -- validator caught widget drift or wiring break
- `missing_model` -- model not on disk, path resolution, HF_HOME, wrong registry key
- `llm_oom` -- writer-side OOM during script generation
- `video_oom` -- FLUX/HuMo/LTX OOM after writer succeeds
- `ffmpeg_composite` -- video assembly or audio mux failure
- `comfyui_startup` -- worker could not reach `/system_stats` or port-PID mismatch
- `port_occupied` -- launch port already bound at pre-flight
- `orphan_detected` -- process found and killed, VRAM recovered, iter proceeded
- `vram_contaminated` -- VRAM still high after filtered-kill + recheck, iter aborted
- `worker_crash` -- worker exited without writing a parseable result file
- `timeout` -- run exceeded 15 min
- `unknown` -- everything else (requires manual triage)

---

## 3. Execution order (this session)

1. **Commit A.** `git add workflows/otr_scifi_16gb_full.json scripts/normalize_workflow_widgets.py`, commit (subject `H bug-hunt: normalize workflow widgets, fix nodes 12/55 stripped-mode drift`), push.

2. **Commit B1.** Apply layer-2 cleanup inside `load_llm()` at `nodes/_otr_model_loader.py:549-552`. Wrap the existing `except Exception` block: move model to CPU if bound, `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, then re-raise as `ModelLoaderError`. Keep the layer-1 wrapper in `request_slot()` at line 741+ (already in working tree). Commit both layers together. Push.

3. **Commit B2.** Success-path `unload_llm()` after `OTR_LedgerFreezeCascade`. Prefer graph-level via `OTR_UnloadAll` node id 24 in `_full.json` -- verify wiring places it between the writer/freeze block and FLUX/HuMo/LTX. If wiring is unclear, fall back to node-level by adding `unload_llm()` at the end of `OTR_LedgerFreezeCascade.execute` in `nodes/OTR_LedgerFreezeCascade.py`. Commit, push.

4. **Commit B3.** Clone `workflows/otr_scifi_16gb_full.json` -> `workflows/otr_scifi_16gb_bughunt.json`. Edit node 1 widget values 4 and 5 to `google/gemma-4-E4B-it` for `creative_writing_model` and `technical_model`. Run `scripts/normalize_workflow_widgets.py --workflow workflows/otr_scifi_16gb_bughunt.json`. Commit (additive, no impact on `_full.json`). Push.

5. **Build sweep scripts with logging.**
   - `scripts/sweep_and_launch.bat` -- outer, blanket pre-launch python sweep with PID+cmdline log to `logs/sweep_prelaunch.log`, then launches supervisor as its last action.
   - `scripts/sweep_python_excluding.bat <SUPERVISOR_PID>` -- between-iter filtered sweep with PID+cmdline log to `logs/sweep_betweeniter.log`. Skips the PID passed as argv[1].

6. **Build `scripts/worker_iter.py`. Refactor `scripts/overnight_bug_hunt.py`** to spawn-and-wait. Supervisor picks free port per iter. No blanket python kills inside the supervisor itself. Worker launches ComfyUI as a direct subprocess child, never via .bat or PowerShell wrapper.

7. **Process-tree validation before any overnight.** Launch supervisor, let one worker spawn, manually `taskkill /F /PID <worker_pid> /T` from another terminal. Confirm ComfyUI and ffmpeg both die. Confirm supervisor survives. If tree breaks, the launch chain has a wrapper somewhere -- fix before proceeding.

8. **Attended smoke test, one iter, daylight.** Watch worker fire. Confirm:
   - Pre-launch sweep wipes stray python before supervisor starts.
   - Between-iter sweep does NOT kill the supervisor.
   - Worker survives independently of supervisor.
   - Supervisor survives worker death.
   - Worker writes `logs/worker_iter_<n>.json` even on forced kill.
   - Gemma loads under 4 GB.
   - LedgerFreezeCascade emits frozen JSON.
   - `unload_llm()` fires on success path before FLUX loads.
   - C7 byte-identity verified via `tests/test_audio_byte_identical.py` (NOT by running `_full.json` end-to-end).
   - Teardown leaves no orphan `python.exe` or `ffmpeg.exe`.

9. If smoke is clean: 3-iter attended. If 3-iter clean: overnight 12-iter via `scripts/sweep_and_launch.bat`.

---

## 4. Open followups (do not block this session)

- **Per-head smoke matrix.** After Gemma baseline is green, run one iter each at default config swapping only the writer head in `_bughunt.json`: Gemma, Qwen2.5-7B-4bit, Mistral-Nemo-4bit, plus any other registered head. Proves which heads fit the 16 GB suitcase under the full pipeline.
- **LTX 22B is the next VRAM cliff.** Expect `video_oom` to dominate iter 4+. Plan: confirm `LowVRAMCheckpointLoader` in use, add staged unload between FLUX and HuMo, between HuMo and LTX.
- **Full-pipeline C7 verification.** Once Mistral loader is resolved, re-establish full-pipeline C7 byte-identity baseline against `_full.json`. `tests/test_audio_byte_identical.py` covers the gap in the meantime.

---

## 5. Decision criteria reaffirmed

- Audio C7 byte-identity preserved by routing bug-hunt loop through `_bughunt.json`. `_full.json` (the C7 default-config target) is not modified.
- Solo + chronic foot pain: smallest viable change set that produces one green iter. Two-process supervisor + sweep-with-exclusion + always-write-a-result-file is the architectural investment; everything else is config and cleanup.
- Local-only. RTX 5080 Laptop, 16 GB VRAM, peak ceiling 14.5 GB.
- v2.0-alpha only. No sprint branches.
- Multi-LLM is the architecture, not a feature on the chopping block. All registered heads remain selectable in both workflow files.

---

## 6. Riskiest assumptions

1. **`unload_llm()` may not actually release to the OS.** PyTorch can hold internal references via cuDNN handles, accelerate hooks, or BNB quant state that survive `del model` + `empty_cache()`. If smoke test shows idle VRAM > 2 GB after a clean iter, this is the culprit. Mitigation: kill ComfyUI child between iters (already in the recipe).

2. **Process tree assumption.** `taskkill /F /PID <worker_pid> /T` only works if ComfyUI is a direct child of the worker. Any .bat or PowerShell wrapper in the launch chain can break the tree. Step 3.7 validates this explicitly before any overnight.

3. **Between-iter filtered sweep is destructive to unrelated python.** Acceptable for a closed overnight box, dangerous if shared with active dev work. Sweep logs to `logs/sweep_betweeniter.log` make this auditable but do not prevent it.

---

## 7. Stop conditions

- 3 consecutive cleans: success, move to next sprint.
- 2 consecutive same-class fails: halt, manual triage.
- 1 timeout: advance to next iter. **2 consecutive timeouts: halt** (likely systemic hang).
- 2 consecutive `worker_crash`: halt (supervisor cannot trust the harness).
- C7 audio bytes diverge from `tests/test_audio_byte_identical.py` baseline: halt, rollback.
