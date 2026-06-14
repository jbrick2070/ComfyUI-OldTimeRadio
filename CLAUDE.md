# OTR -- Project Operating Rules (Claude in Cowork)

Hard rules + the REAL Cowork operating model for this repo + hard-won gotchas. Operator directives
here win over any handoff / doc / memory that disagrees.

Fix bugs in pure prbt manner not a shim and dont woiat for me to fix do a fix witout asking em its ok if it works ifi n doubt /routdtabel chat gpt and gemi and ore deek desk for convergcne

## 0. WORKFLOW SOURCE OF TRUTH (hard)

`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json`
IS my workflow.

- ANY json / node / wiring / widget change MUST be made IN that file, in the SAME change as the code.
  Code that is not wired into this JSON is DEAD -- "your updates are for naught" (the 2026-06-13 §4D
  miss: a node + a new blend input shipped + tested but UNWIRED -> ran dormant in production).
- EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot.
- Schema: litegraph. Top level = `nodes[]` + `links[]` + `last_node_id` + `last_link_id`; a link is
  `[link_id, src_node, src_slot, dst_node, dst_slot, type]`; one output fans out via its `links` list.
  `widgets_values` is POSITIONAL -- only ever APPEND a new optional widget at the END (inserting mid-list
  shifts every saved value -> silent drift, BUG-LOCAL-097). A widget converted to an input keeps its
  value slot AND gains an input with `"widget": {"name": ...}`.
- After editing it, re-validate: `OTR_WorkflowValidator` + a JSON round-trip + a link/widget audit
  (widget-count vs live INPUT_TYPES, every wired input-name in INPUT_TYPES, link referential integrity).

## 1. HOW COWORK ACTUALLY WORKS HERE (read this first)

- **Two separate filesystems.** The file tools (Read / Write / Edit) operate on the REAL Windows files --
  that is your primary editor. **Desktop Commander** (`mcp__Desktop_Commander__*`) runs PowerShell on the
  same real Windows box -- use it for git, the venv python, tests, and process control. The
  `mcp__workspace__bash` Linux sandbox is a DIFFERENT machine: its mount LAGS the file-tool writes (shows
  stale/truncated copies -> phantom "corruption") and has NO torch. Use bash only for quick greps of
  UNCHANGED files; never trust it for current state and never run the suite there.
- **The loop:** edit with the file tools -> verify/test with the Windows venv via Desktop Commander ->
  commit AND push via DC git -> verify HEAD == origin.
- **Test runner:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (torch 2.10). Run with
  `$env:PYTHONUTF8=1`; `pytest -q -p no:cacheprovider`. The conftest sets `OTR_TEST_MODE`/`CUDA_VISIBLE_DEVICES=''`.
  Bug Bible lives in a SEPARATE repo: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`
  -- `cd` to its root and use the RELATIVE path `tests\bug_bible_regression.py` (an absolute forward-slash
  path fails to collect).
- **PowerShell reality (DC runs powershell.exe):** use `;` to chain, NOT `&&`. Do NOT use
  `python -c "..."` with nested quotes -- PowerShell mangles them; instead WRITE A TEMP `.py` file, run
  it, then delete it. `2>&1` makes stderr render as scary red text -- that is NOT a failure; check the
  exit code / output. Pipe noisy output through `Select-Object -Last N`.
- **The ~60s MCP ceiling:** any single DC command that blocks longer (a `Start-Sleep` > ~45s, a big render
  loop, the full ~4200-test suite, a slow boot wait) TIMES OUT and orphans the process. Background it to a
  log and poll the log, or shrink the job (a test subset, fewer frames). DC itself is NOT flaky -- the
  command was too long.
- **Subagents (Agent tool)** are excellent for read-only fan-out audits, but TELL them to use Desktop
  Commander + the Windows venv/path -- left to default they read the lagging Linux mount and report
  phantom truncation/corruption (happened 2026-06-13; the third agent that used the Windows path was right).
- **Stale git `index.lock`:** if `git add`/`commit` fails with "index.lock: File exists" AND
  `Get-Process git` is empty, the lock is STALE (a real git op finishes in seconds) -- remove
  `.git\index.lock` and retry. Do NOT remove it while a git process is actually running.
- **One coder window in the code at a time** (serialize via `docs/GO_FORWARD_PLAN.md`). Two windows
  editing the same file -- especially the workflow JSON -- is how it gets corrupted.
- Use **AskUserQuestion** for genuine operator decisions; use the **task list** for any multi-step work.

## 2. AUTONOMY / PRIME DIRECTIVE

- NEVER ask me to run scripts, commands, or anything. YOU run it: Desktop Commander first; if DC can't,
  Windows MCP; then the filesystem tools. Never hand me a bat/cmd/PowerShell block and say "run this."
- You can drive the 5080 GPU yourself -- spin up the headless ComfyUI API (port 8000) and run it; don't
  ask me. (Reset the box first -- section 4.)
- If a senior pair-programmer would just do it, just do it. Stuck choosing between options? Roundtable
  2-3 panels (GPT + Gemini + one other) for opinions BEFORE asking me -- you are the judge.

## 3. CODING DISCIPLINE

- Keep coding until all sprints are done unless you genuinely need me.
- Run the regression suite + the Bug Bible after EVERY code change (don't wait to be asked). Commit AND
  push per green chunk (section 6).
- Prefer editing the file you're already in; don't spray new throwaway files (and delete any temp probe
  scripts before committing). Keep handoff files current.
- Names: never "dummy" -- use "placeholder", "stub", or a descriptive name ("dummy" makes me feel bad).
  SFW always. UTF-8, no BOM. Clean logs, meaningful names -- the reader matters.

## 4. RESET BEFORE EVERY HEADLESS RUN (hard)

The soak/quick-smoke harness boots ONE server, runs all legs against it, and does NOT tear it down -- it
sits RESIDENT holding ~60% VRAM. Never assume a prior run cleaned up. Before launching:

- Kill SELECTIVELY by CommandLine (CIM) -- NOT a blanket `Stop-Process -Name python,pythonw`. A blanket
  python kill ALSO kills the Claude MCP extension pythons (Desktop Commander / computer-use) and severs
  your own tools mid-run. Use `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and kill the
  ones whose CommandLine matches the ComfyUI server + the soak/sweep harness; plus the port kill via
  `Get-NetTCPConnection -LocalPort 8000`. (`.Path` is BLANK for half-booted servers, so filter on
  CommandLine, not Path.)
- Confirm `Get-NetTCPConnection -LocalPort 8000 -State Listen` is EMPTY and
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` dropped to the desktop baseline (~1.5 GB)
  before booting fresh.

## 5. HEADLESS BOOT + MONITORING GOTCHAS (2026-06-12 -- do not relose)

- **A render that FINISHED leaves the server RESIDENT (~9-10 GB, 1% util) -- that is NOT a crash.** Before
  declaring a run dead, read the server log: `Prompt executed in HH:MM:SS` + `obs_publish OK` = it
  COMPLETED. The idle resident VRAM is the no-teardown behavior. (Misread twice 2026-06-12.)
- **Use the watchdog for long renders** (`scripts/otr_render_watchdog.ps1 -LegLog <leg.log>`): declares the
  run DEAD on a 5-min heartbeat stall OR a down :8000/queue endpoint (exit 2), exits 0 with the verdict
  when the leg finishes. It REPORTS only; reset per section 4.
- **Headless boot needs UTF-8.** A detached cmd inherits the Windows cp1252 codec, so OTR
  `prestartup_script.py` crashes the instant it prints an emoji (UnicodeEncodeError on U+2705/U+2713) ->
  boot dies ~13s, exit 1 ("SERVER DID NOT COME UP"). The launcher (`scripts/_otr_soak_server_launch.cmd`)
  sets `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`; any new boot path MUST too.
- **Boot is ~20s, NOT 7-8 min.** If a boot "hangs", read the log -- it has already died.
- **Launch the server via the .cmd as `-FilePath`, never `cmd.exe /c "<cmd>" "<log>"`.** The `/c`
  two-quoted-token rule eats the outer quotes -> mangled path -> ZERO log output. Use
  `Start-Process -FilePath $LAUNCHCMD -ArgumentList "`"$LOG`""`.

## 6. GIT POLICY (operator directive 2026-06-10 -- never lose work)

- ONE branch: `v2.0-alpha`. COMMIT AND PUSH TOGETHER: every green commit gets pushed to origin
  immediately, same session, no exceptions. Local-only commits are the failure mode we guard against.
- The operator eyeball gates TAGS and PROMOTIONS (`v2.0-alpha-stable`, prod, main, v2 release) -- NEVER
  pushes. Pushing to `v2.0-alpha` is always safe, expected, and required.
- This SUPERSEDES any "do not push until the eyeball passes" line written before 2026-06-10 evening.
- A stable branch only exists if the operator explicitly declares one.
- After every push verify: HEAD == origin, no 0-byte files, no BOM, AST parse on touched .py files.
