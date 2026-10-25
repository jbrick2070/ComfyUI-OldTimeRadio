# Hand-off, 2026-10-24: CreatorOS concepts -> OTR

A work order and a memory for the Claude that continues this on the desktop. The cloud
session that wrote it did Sprint 1 and then lost its link to this machine again and again;
the chat that held its context is gone, so what it knew is written here. `CLAUDE.md` at the
repo root wins over this file wherever they differ. Delete this file and the `handoff/`
folder next to it in the commit that finishes the last item.

Read order: `CLAUDE.md`, this file, `PLAN.md` (same folder).

## 1. Where things stand (2026-10-24, about 18:40 PT)

| Item | State |
|---|---|
| Plan | `planning/creatoros-concepts/PLAN.md`, pushed. Scope rule 8: nothing may wait for an operator mid-run |
| S1 rows 1.1 live view, 1.2 failure hints, 1.3 phone ping | PUSHED: 65f1fa29, test fix 8e5b37b7 |
| S1 follow-up: QA fixes + `--preview-file` | CODED and green on the targeted tests, UNCOMMITTED in the working tree. Items A to G in section 2 finish it |
| S2 environment fingerprint | Anchor review half done (section 5). Next: kibitz arc, then code |
| S3 Studio proposals | Goes to the owner of `planning/studio-ui/DESIGN_ANCHOR.md` (section 6) |
| S4 broadcast feed, D1 = A | Full kibitz arc first (section 6) |

Checkout: branch `claude/sharp-heisenberg-no3hzt`, HEAD 65f1fa29 == origin/main when this was
written. Ship with `git push origin HEAD:main`, named files only, never `git add .`.
Untracked and NOT mine: `apple/source_chunks_claude_constitution.md`. The ComfyUI server is up
on port 8000 (leave it resident).

Backups of this work: the working tree on this machine is the live copy. A snapshot commit of
the uncommitted S1 follow-up is on origin as branch `wip/creatoros-s1-followup`, taken right
after this file was committed (a safety net only: never merge it; finish items A to G in the
tree and commit the named files). A copy of this file is also a Project doc,
`claude/creatoros_handoff_2026-10-24.md`.

## Operator directives that govern all of it

- Idea source: thePloggy/CreatorOS-Studio (Apache-2.0). Open source, safe to take ideas; no
  code copied. If code is ever copied verbatim it keeps its Apache-2.0 header and a change
  note.
- No Ollama, no llama.cpp, no models. No cloud services, no API keys, no paid services:
  100% local, offline-first. Notify stays OFF unless `OTR_NOTIFY_URL` is set.
- Broadcast feed (S4): D1 = A, air by default, pull from air.
- Nothing may require an operator to approve anything mid-run (PLAN scope rule 8).
- "This is proven code so please code it, it should be easy" and "Keep going": implement the
  plan autonomously, do not stop between chunks to ask.
- Replies to Jeffrey: brief (1-2 sentences), no fluff, minimal emoji; paste-ready content in
  one fenced block at the very end. Banned vocabulary is in CLAUDE.md (the naming rule near
  the top): use "placeholder", "stub" or a descriptive name, and keep the banned word out of
  code, comments, logs, commit messages, docs and replies. Safe for work. UTF-8, no BOM.
- Project rules, restated so they are not missed: you run everything, never hand Jeffrey a
  command; fix bugs at the root cause, no shims; any node, wiring or widget change goes into
  `workflows/otr_canonical.json` in the SAME change; renders write straight to
  `otr\episodes\<ep>\` and finals to `otr\obs\`, never tmp, and an asset is confirmed with
  Test-Path before success is declared; regression suite + Bug Bible after every code change;
  commit and push every green chunk, then QA the PUSHED diff; state the 5080/4060 blast
  radius in every commit message; chain commands with `;`; never `python -c "..."` with
  nested quotes (write a temp .py, run it, delete it); before a headless run reset the box
  with a SELECTIVE CIM-CommandLine kill, never a blanket python kill (it severs the MCP
  pythons), and confirm port 8000 is free and the GPU idle first.
- Git habits from Jeffrey's notes: commit with an explicit pathspec (another window may have
  staged work); use `cmd` for commit messages (`git commit -F file`); afterwards
  `git show --stat HEAD` must list exactly the expected files.

## 2. Finish the S1 follow-up

### What the working tree already holds (uncommitted)

- `scripts/otr_api.py`: SERVER-GONE marker in `submit_prompt` and `poll_history`
  (`_hint_suffix`); the hint on the error branch comes from `repr((error, node_errors))`;
  fallback hint in `describe_execution_error`; `LiveState.adopt_prompt_id` (ComfyUI
  broadcasts `execution_interrupted` to every client, so state taken from another client's
  prompt is dropped and ours is kept); preview support (`preview_by_node`, `last_preview`,
  `preview_serial`, `_image_from_preview_frame`, `_JPEG_START`, `_PNG_START`); the `LiveView`
  preview writer (`_write_preview` holds `_preview_io_lock`, the work is in
  `_write_preview_now`; at most once per `_PREVIEW_WRITE_EVERY_S = 1.0`, atomic replace,
  newest frame flushed when the leg ends); `submit_prompt(api_prompt, client_id=None,
  preview_method=None)` (argument beats env `OTR_PREVIEW_METHOD`, validated against
  `_PREVIEW_METHODS`, sent as `extra_data.preview_method`); the LIVE line ends
  `preview_nodes=<node>:<count>`.
- `scripts/otr_canonical_api_run.py`: `preview_plan(...)`; `--preview-file PATH`; the new
  kwargs reach `submit_prompt` / `LiveView` only when the flag is given (existing test stubs
  have the old signatures); `compose_result_ping` pulls the hint with
  `re.search(r"-- hint:[ \t]*([^\r\n]*)", text)`; `main()` pings on an exception (before the
  prompt was queued, or while watching) and re-raises; `finally: live.stop()`.
- `scripts/otr_known_failures.py`: `server_gone` matches only `SERVER-GONE`;
  `model_family_mismatch` anchored on torch's own wording.
- `scripts/otr_notify.py`: REPLACED by the hardened module (237 lines, sha256 starts
  65668cc70466). A redirect is never followed (urllib turns a 301/302/303 POST into a bodiless
  GET that the server answers 200, so the ping would be reported as sent and never
  delivered). The whole POST has a 15 s deadline on a thread, because urllib's timeout is per
  socket operation. Error text is scrubbed of the URL, its path segments (the topic) and the
  credentials. A set-but-unusable value prints `notify: ON BUT UNUSABLE (...), no ping will
  be sent`.
- `scripts/otr_render_watchdog.ps1`: the ping with the same credential and trim handling.
- Tests: `tests/test_otr_live_view.py` (+17), `tests/test_otr_notify.py`,
  `tests/test_otr_known_failures.py`.

Last targeted run, 18:3x: `tests\test_otr_notify.py tests\test_otr_known_failures.py
tests\test_otr_live_view.py tests\test_canonical_headless_api.py`, 160 passed. The last FULL
suite (18:06) ran before the notify module was replaced: green apart from one environment
failure (section 3). Re-run everything after the items below.

### Items still to do, in this order

**A. `tests/test_otr_notify.py`** (the new module is in, so these tests are missing):
1. `import time` at the top. Replace its `NtfyStub` with the one in
   `handoff/notify_tests_scratch.py.txt` (adds status, delay, location, records the method,
   answers GET as well as POST).
2. Copy the four tests under "new: scrubbing, redirects, the deadline" from that file (scrub,
   redirect parametrized 301/302/303/307/308, deadline, worker-ends-without-an-answer). They
   were run green, and 5 of 5 hand-made mutants of the module were caught by them.
3. Runner tests: extend `_StubLive` with a class attribute `last` (set in `__init__`),
   a `stopped` flag set by `stop()`, and `prompt_id`. Assert
   `_StubLive.last.prompt_id == "prompt-live"` and `.stopped` in the success test, and
   `.stopped` in the crash and refusal tests (today they would pass even if the view were
   never stopped or never adopted the prompt id).
4. Ctrl-C test: `poll_history` raising `KeyboardInterrupt` propagates, the stub receives
   nothing (an operator's Ctrl-C is not a failure to report), and stop is called.
5. `_run_watchdog` helper: add `"-QueueUrl", "http://127.0.0.1:1/queue"` so the watchdog
   never probes a real ComfyUI queue on :8000 during tests.
6. Windows-only watchdog tests (read lines ~520-end first for the existing ones): a redirect
   is not followed (the redirect target sees no request), and an unusable URL prints the loud
   state. These need item B.

**B. `scripts/otr_render_watchdog.ps1`** (ASCII only; PowerShell 5.1):
1. `Send-Ping`: add `-MaximumRedirection 0` to the `Invoke-RestMethod` call (5.1 then throws
   on a 3xx instead of following it).
2. Add the URL's path segments (the topic) to `$secrets`, as `otr_notify._parse` does.
3. In the catch, when `$why -match 'redirect|\b30[1-8]\b'`, append: put the final URL
   (usually https://) in OTR_NOTIFY_URL.
4. `Get-NotifyState`: require scheme http/https and a host; otherwise return
   `notify: ON BUT UNUSABLE (OTR_NOTIFY_URL is not an http:// or https:// URL with a host), no
   ping will be sent` (same wording as `otr_notify._UNUSABLE_STATE`).

**C. `scripts/otr_known_failures.py` and `tests/test_otr_known_failures.py`:**
1. `out_of_memory`: delete the `r"|not enough memory"` line. It also matches torch's CPU
   allocator message (`DefaultCPUAllocator: not enough memory`), which is system RAM, and
   the hint then blames the GPU.
2. New row `system_memory` right after `out_of_memory`: matches
   `DefaultCPUAllocator: not enough memory` and the Windows page-file error (`WinError 1455`,
   "The paging file is too small for this operation to complete"); the hint says the
   computer ran out of system memory (RAM or the Windows page file), not GPU memory: close
   other programs or enlarge the page file; `refs=()`.
3. The `value_not_in_list` and `model_family_mismatch` hints name `scripts/otr_asset_index.py`:
   make both `scripts/otr_asset_index.py --stdout`. Its only flag is `--stdout`; with no flag
   it WRITES `apple/MODEL_ASSET_INDEX.md`, so today's hint makes a reader overwrite a tracked
   file.
4. Update the module docstring's list of failure classes.
5. Tests: a `FIXTURES["system_memory"]` entry with the CPUAllocator and WinError 1455
   strings (`test_every_row_has_a_fixture_and_every_fixture_has_a_row` demands it); CPU
   memory exhaustion is not called a GPU failure; every hint that names
   `otr_asset_index.py` includes `--stdout`; a wiring test through
   `describe_execution_error`.

**D. `scripts/otr_api.py` and `tests/test_otr_live_view.py`:**
1. `_as_int`: `except (TypeError, ValueError):` must also catch `OverflowError`
   (`int(float("inf"))` raises it).
2. Tests: `_as_int` of inf, -inf and nan gives None (do NOT use `10**400`; that is a valid
   int); a parametrized test that `execution_success`, `execution_error` and
   `execution_interrupted` each finish the view; a test that `view.wrap(...)` survives a
   raising `snapshot()` (read `wrap`'s except branch, about lines 1457-1490, first, and write
   the exact assertion from it).

**E. `scripts/otr_canonical_api_run.py`:** a comment at the `except Exception` in `main()`
saying Ctrl-C (KeyboardInterrupt, not an Exception) sends no ping on purpose.

**F. `tests/test_canonical_headless_api.py`:** in `test_canonical_runner_emits_poll_heartbeats`
add `monkeypatch.setattr(canonical, "episode_of_prompt", lambda _pid: None)` right after
`monkeypatch.setattr(canonical, "LiveView", _NoSocketLiveView)`, so the test cannot reach
for a real ledger lookup.

**G. Mutation spot-check.** For each new branch (redirect refusal, deadline, secrets scrub,
hint extraction, `adopt_prompt_id`, `_as_int`, atomic preview replace, `system_memory`) break
it, confirm a test fails, restore it. The refutation pass found surviving mutants in four
areas; items A to F are the fixes for them.

Design decisions already made (keep them): Ctrl-C sends no ping; a bad `OTR_NOTIFY_URL` is
reported loudly at start-up and never gates the run; preview kwargs are conditional; the
watchdog's gap before its first heartbeat is pre-existing and out of scope (report it).

## 3. Verify

Targeted (about 30 s):

    cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    set PYTHONUTF8=1
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -o addopts= -q -p no:cacheprovider tests\test_otr_known_failures.py tests\test_otr_live_view.py tests\test_otr_notify.py tests\test_poll_history_names_a_dead_server.py tests\test_canonical_headless_api.py tests\test_comfy_credential_rip.py

Full suite (minutes; run it in the background and poll): the same python with
`-m pytest -o addopts= -q -p no:cacheprovider tests`, output to a log.

Bug Bible, run from the survival-guide repo against the pack and compare with the clean
baseline worktree (`C:\Users\jeffr\Documents\ComfyUI\_otr_baseline_wt`, 65f1fa29):

    cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
    python -m pytest tests\bug_bible_regression.py -q -p no:cacheprovider --pack-dir <pack_dir> --tb=line

Last logs: `tmp\_bible_base.log` (clean), `tmp\_bible_mine.log`, `tmp\_bible_fix.log`. Your run
must add no failure that `_bible_base.log` does not show, except BUG-09.02 and BUG-12.183
while untracked scratch sits under the pack's `tmp\`; delete the scratch (section 4) and run
the Bible once more before pushing.

Facts about the test rig:
- `tests/conftest.py` has a known-failure guard: it prints `NEW failures (REGRESSION)` for any
  failing test not in `EXPECTED_FAILED_NODEIDS`, and it HIDES the traceback. To see why a
  test failed, run its body from a small script (example: `tmp\_voice_probe.py`).
- `conftest` pops `OTR_NOTIFY_URL` and `OTR_PREVIEW_METHOD`; use `-o addopts=` when
  collecting.
- Known environment failure, not caused by this work:
  `tests/test_output_tree_containment.py::test_a_remote_voice_reference_is_refused_before_it_is_statted`
  fails whenever the mapped drive `U:` (`\\10.55.0.2\4060-TRANSFER`, the 4060 box's share) is
  Disconnected; reproduced on the clean worktree. It passes with `U:` connected. Report it,
  do not add it to the expected-failures list.

## 4. Commit, push, QA, clean up

1. Commit message: start from `tmp\_commit_msg_3.txt` and add the newer fixes (no redirect
   following, 15 s deadline, topic path segments scrubbed, CPU-RAM row, `--stdout` hints,
   overflow, loud unusable state in both notify and the watchdog, test isolation and
   strength), the exact suite numbers, and the blast radius: `scripts/` and `tests/` only,
   display and diagnostics only; nothing under `nodes/`, no workflow JSON, no profile, so
   the 5080 and 4060 render paths are provably untouched. End with your session's
   attribution lines.
2. Named files only: `scripts/otr_api.py`, `scripts/otr_canonical_api_run.py`,
   `scripts/otr_known_failures.py`, `scripts/otr_notify.py`,
   `scripts/otr_render_watchdog.ps1`, `tests/test_otr_known_failures.py`,
   `tests/test_otr_live_view.py`, `tests/test_otr_notify.py`,
   `tests/test_canonical_headless_api.py`. Check `git show --stat HEAD`.
3. `git push origin HEAD:main`, ONE attempt; if it fails, hand Jeffrey a PowerShell block.
   Afterwards: HEAD equals origin/main, `python tmp\_verify_files.py <the files>` (zero-byte,
   BOM, CRLF, AST).
4. QA of the PUSHED diff: one Sonnet subagent briefed to REFUTE (not to confirm), plus one
   local CLI lane from a different family (agy in the throwaway worktree: `tmp\_run_agy_qa.ps1`
   with a brief like `_otr_baseline_wt\tmp\_qa_brief.md`, both adapted to the new commit).
   A QA finding becomes the next commit. Two HOLDs: stop reviewing and move on.
5. Live proof of `--preview-file`: profile `otr_16gb_still` (MOSS-Image). Measured earlier:
   425 preview frames, the last 74,135 bytes, with `latent2rgb`. NOT measured on the LTX
   video walk (`QuietInnerBars` may silence previews): run a one-act canonical leg
   (`--act-count 1`) with `--preview-file` and read `LIVE ... preview_nodes=`. Reset the box
   first (selective kill, port 8000 free, GPU idle), Test-Path the output.
6. PLAN.md edits (after the code commit), commit and push:
   - Delete rows 1.2 and 1.3 (their receipts are in the commit messages); reduce 1.1 to the
     residual: measure previews on the LTX walk, and record the measurement above.
   - "22" labels becomes 23 (`config/node_phase_labels.json`).
   - Replace "30-word leg" with "a one-act canonical leg (`--act-count 1`)" (at-a-glance
     note, 1.1 Done-when, 2.1 Done-when).
   - Intro ("Plan only: nothing is built") and the S1 at-a-glance row: say what shipped.
   - S3 cross-refs: 3.2 points at `config/node_phase_labels.json`, 3.5 at
     `scripts/otr_notify.py`.
   - S2 2.1 `meta.env` text: per section 5.
7. Clean up LAST (the Bible fails with scratch present):
   - Kill the ntfy stub: the two pythons whose CommandLine contains `_ntfy_stub` (selective
     CIM kill; never a blanket python kill).
   - Delete everything in `tmp\` whose name starts with an underscore (all mine; the list is
     `Get-ChildItem tmp -Filter "_*"`, keep `__pycache__` if you like), and this
     `handoff\` folder.
   - `git worktree remove --force C:\Users\jeffr\Documents\ComfyUI\_otr_baseline_wt` then
     `git worktree prune`.
   - NOT mine, leave alone: `after_fix.py`, `after_leg17.py`, `rerun_after_gauntlet.py`,
     `launch_soak.py`, `overnight_randomizers.before_stall_watch.py`, `rps_live/`,
     `apple/source_chunks_claude_constitution.md`, `.claude/worktrees/`.
   - Leave the ComfyUI server resident.

## 5. Sprint 2: environment fingerprint (anchor review half done)

Read in the real files at 65f1fa29; the plan text (PLAN.md 2.1) differs from the code in
three places.

1. **There is no `meta` at ledger creation.** `Ledger.__init__` (`nodes/production_ledger.py`
   ~1175) has a top-level `"commit": _git_head_short()` (:1178; helper at :219, cached per
   process, `git rev-parse --short HEAD`, 5 s). `meta` appears later: `Ledger.save()` (:1918)
   stamps `meta.paths` and `meta.schema_version` on every save, and `_merge_with_disk`
   (~:2030) merges `meta` per key (disk fills keys missing or empty in memory; memory wins
   otherwise). Audio nodes stamp `meta.git_commit` and `meta.phase_ms` through
   `_otr_ledger.set_meta` / `record_phase_ms` / `save_ledger_safe`
   (`nodes/audio_enhance.py:507`, `nodes/scene_sequencer.py:1300` and `:2439`;
   `lookup_git_commit` at `nodes/_otr_ledger.py:912`, cached per process). So `meta.env` has
   precedent next to `meta.git_commit`. The arc must pick WHERE: in `Ledger.save()` beside
   `meta.paths` (stamp when absent), in `new_ledger()`, or node-side like `git_commit`.
2. **Replay.** `_REPLAY_RUN_VOLATILE_META` (:584) lists meta keys popped on replay import
   (the loop at :842); `import_replay_bundle` replaces `led.data` with the bundle's data, so
   `env` must be popped AND re-stamped by the replay's own run, and the tuple's comment
   requires every member to be something the replay re-stamps. Trap to test: the disk merge
   can pull a stale `meta.env` back when memory lacks it.
3. **Schema.** `CURRENT_SCHEMA_VERSION = "l4-2026-08-07"` (`nodes/_otr_ledger.py`) is pinned
   by tests. An optional additive meta key needs no bump; document it in the lineage
   comment (`_otr_ledger.py` ~55-140).
4. **Env access rule.** `nodes/_otr_shared/env.py` is the ONLY module under `nodes/` allowed
   to spell `os.environ` (registry YARA scan); import it as `otr_env`. The new
   `nodes/_otr_shared/env_fingerprint.py` reads knobs through it.

Not read yet: `otr_state_dir` (`nodes/_otr_paths.py:744`); `scripts/otr_verify_replay.py`;
`scripts/otr_freeze_replay_bundle.py` (own `_git_commit()` at :58, `source_commit` at :87);
tests that pin ledger shape (grep `tests` for `"meta"` and `CURRENT_SCHEMA_VERSION`); how
the runner finds the episode ledger for the known-good snapshot (`episode_of_prompt`);
privacy: the fingerprint travels in ledgers and replay bundles, so no paths, user names or
machine names; GPU name, driver, OS build and pack commits only.

Process: `nodes/` is 5080-owned, so `ListAgents` and `SendMessage` the peer window first.
Full kibitz arc before any code: your anchor review, then r1 to r4 with the S2 section of
PLAN.md plus this section as `input.md`. Pass `--driver claude` (or `--only codex --only
antigravity`) so no third Claude seat runs; set `KIBITZ_AGY_MODEL="Gemini 3.5 Flash (High)"`;
check `antigravity.log` per round (a bad model name silently makes the arc codex-only) and the
run folder after r1 for any `claude*` file. Ground every agent claim in the code; you judge.
Then code, tests, suite + Bible, push, QA as in section 4. Done-when is in PLAN.md 2.1/2.2.

## 6. Sprints 3 and 4

- **S3** (PLAN.md S3, rows 3.1 to 3.5): proposals, not code. Hand them to the window that owns
  `planning/studio-ui/DESIGN_ANCHOR.md` (the Studio, `contest/radio-play-studio/`, belongs to
  another owner); do not edit the Studio.
- **S4** (PLAN.md S4): full kibitz arc before code; D1 = A decided. The publish boundary and
  `nodes/` are 5080-owned (message the peer first). The feed goes under
  `otr/episodes/_shared/broadcast/`; `otr/obs/` must stay byte-identical.

## 7. Tell Jeffrey at the end

- The Studio bug seen at `contest/radio-play-studio/radio_play_studio/runner.py:805`
  (BUG-05.06 territory; other owner, not touched).
- Bible stub drift: `sanctioned_gap_reason` at `nodes/_otr_video_engines/render_driver.py:6827`.
- Untracked scratch under `tmp\` makes Bible BUG-09.02 and BUG-12.183 fail.
- The ping was proven against a local stub only; a real phone needs his own ntfy topic in
  `OTR_NOTIFY_URL`.
- The watchdog is not armed until the first heartbeat (pre-existing): a runner that dies
  before any heartbeat is covered only by the in-runner ping for exceptions after the prompt
  is built.
- Minor wording of the `value_not_in_list` hint.
- The earlier test hole that exposed `HF_TOKEN` (fixed in 8e5b37b7).
- The `U:` drive environment failure (section 3).

## 8. Gotchas from the remote session (a local session may not hit them)

- PowerShell one-liners containing `$_` were mangled by the shell layer: use a temp `.py`.
- `echo EXIT=%ERRORLEVEL%>> file` in cmd prints an empty value (a digit right before `>>`
  is a handle): put a space before `>>`.
- Desktop Commander `start_search` output mangled content: read files directly.
- The Grep tool of the cloud container cannot see Windows paths.
