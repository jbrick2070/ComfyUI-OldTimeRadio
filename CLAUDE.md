## HEADLESS BOOT + MONITORING GOTCHAS (2026-06-12 -- do not relose)

- **DIRECTIVE -- before EVERY new headless run, aggressively reset first.** The
  soak/quick-smoke harness boots ONE server, runs all legs against it, and does
  NOT tear it down at the end -- it sits RESIDENT holding ~60% VRAM. Never assume a
  prior run cleaned up. Kill ALL python BEFORE launching and verify the GPU is idle:
  `Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue`; confirm
  `Get-NetTCPConnection -LocalPort 8000 -State Listen` is empty; confirm
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` dropped to the desktop
  baseline (~1.5GB) before booting the fresh server.

- **Headless ComfyUI boot needs UTF-8.** A detached cmd inherits the Windows
  cp1252 console codec, so OTR `prestartup_script.py` crashes the instant it
  prints an emoji (UnicodeEncodeError on U+2705/U+2713) -> boot dies ~13s, exit 1,
  the "SERVER DID NOT COME UP" failure. The launcher (`scripts/_otr_soak_server_launch.cmd`)
  now sets `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`. Any new boot path MUST too.
  (ComfyUI Desktop used to set this for us; the v2 install move dropped it.)
- **Boot is ~20s, NOT 7-8 min.** Old handoffs feared a slow boot; it was never
  slow, it was crashing. If a boot "hangs", read the log -- it has already died.
- **Launch the server via the .cmd as `-FilePath`, never `cmd.exe /c "<cmd>" "<log>"`.**
  The `/c` two-quoted-token rule eats the outer quotes -> mangled path -> launcher
  never runs -> ZERO log output. Use `Start-Process -FilePath $LAUNCHCMD -ArgumentList "`"$LOG`""`.
- **Kill servers by CommandLine (CIM), not `Get-Process .Path`.** `.Path` is BLANK
  for processes whose path can't be read, so a `Where Path -like` filter silently
  skips a half-booted server. Use `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`
  + a CommandLine match, plus the port kill (`Get-NetTCPConnection -LocalPort 8000`).
- **Desktop Commander is NOT flaky -- long `Start-Sleep` in a DC command blocks past
  the ~60s MCP request ceiling and times out.** For quick FILE polls use the
  built-in Read (workspace-folder files) or Windows MCP FileSystem (any absolute
  path, e.g. the server log outside the mount); keep DC for shell/git/process/boot.
  Never poll with a `Start-Sleep` longer than ~45s in one DC call.

you cn run 5080 gpu test comfy on yrou own just spin up ahealdess api thing i think it runs inn 8000 dont adk me you do it and ask roubndatbel if youc ant

PRIME DIRECTIVE: never ask the operator to run scripts, commands, or anything. Use Desktop Commander to run everything. If Desktop Commander can't do it, use Windows MCP. Never hand the operator a bat/cmd/PowerShell block and say "run this" -- YOU run it.

when it doubt yuuse dekstop commander to do anyting don ask me tio push scripts or cmd or pwoertsheell you di  t

i desktop commander doesnt do it use windows mcpo

be sure handoff fiels and try not to creatre new fiels try to kleep to te file ou are using t omake tinsg esier

dont use word dummy use placehodler as duymmy makels me feel bad

when coding try to keep coding into all sprints are done unles yo absoyult need me you can roubndtabel for a ocupel opions gpt and gemni and otehr if needed before aksing mne but only 2-3 panels

## GIT POLICY (operator directive 2026-06-10 -- never lose work)

- ONE branch: v2.0-alpha. COMMIT AND PUSH TOGETHER: every green commit gets
  pushed to origin immediately, same session, no exceptions. Local-only
  commits are the failure mode we guard against (fear of losing work).
- The operator eyeball gates TAGS and PROMOTIONS (v2.0-alpha-stable, prod,
  main, v2 release) -- NEVER pushes. Pushing to v2.0-alpha is always safe,
  expected, and required.
- This SUPERSEDES any "do not push until the eyeball passes" line in any
  handoff, doc, or memory written before 2026-06-10 evening.
- A stable branch only exists if the operator explicitly declares one.
- After every push verify: HEAD == origin, no 0-byte files, no BOM, AST
  parse on touched .py files.
