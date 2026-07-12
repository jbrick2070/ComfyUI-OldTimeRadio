# OTR -- Project Operating Rules (Claude in Cowork)
**Operator directives -- these win over any handoff, doc, or memory that disagrees.**
- **Fix bugs properly, at the root cause -- never a shim or band-aid.** Don't wait for me to
  fix anything: make the fix yourself, without asking. If it works, it's fine.
- **When genuinely torn between approaches, run the roundtable LIVE for convergence
  (ChatGPT + Gemini + DeepSeek).** Skip the dry-run / cost estimate -- just run it, pronto.
  You are the judge. Escalate to me only if the panel still leaves it unresolved.
- This file is hard rules + the real Cowork operating model for this repo + hard-won gotchas.
## 0. WORKFLOW SOURCE OF TRUTH (hard)
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_canonical.json`
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
- **Knowledge gate before implementation or diagnosis:** read `docs\PRODUCTION_SPRINT_LESSONS.md`, then the
  relevant entries in `docs\PROD_BUG_LOG.md`, and the matching portable rules in
  `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\BUG_BIBLE.yaml`. The lessons are
  mandatory project context, the production log is the staging record, and the Bible is the reusable
  cross-project contract. A newly fixed, repeatable production failure must be recorded in the log and
  promoted with a Bible entry plus executable coverage whenever its verify condition is automatable.
- **Admission rule:** only a bug verified by a live production artifact, headless run, smoke, soak, or
  published episode may enter `PROD_BUG_LOG.md` or be promoted to the Bug Bible. A review observation,
  static-audit finding, or invented test fixture may verify a known production bug, but never creates a
  new PBUG or Bible rule on its own.
- **PowerShell reality (DC runs powershell.exe):** use `;` to chain, NOT `&&`. Do NOT use
  `python -c "..."` with nested quotes -- PowerShell mangles them; instead WRITE A TEMP `.py` file, run
  it, then delete it. `2>&1` makes stderr render as scary red text -- that is NOT a failure; check the
  exit code / output. Pipe noisy output through `Select-Object -Last N`.
- **QUOTING / `$`-INTERPOLATION RULE (hard -- this bites EVERY session that ignores it):** any command
  that would need NESTED quotes, backtick-escaped quotes (`` `" ``), a `$` variable inside a quoted
  argument, a here-string fed to another program, or `cmd.exe /c "<...>"` -- DO NOT attempt it inline
  and DO NOT iterate on escaping ("one more tweak" never works; PowerShell mangles `$var` and eats
  quote layers). IMMEDIATELY write a temp LAUNCHER SCRIPT instead -- a `.ps1` (or `.py`/`.cmd`) written
  via the FILE TOOLS (Write, not echo/Set-Content with its own quoting problem), with all variables and
  quoting inside the script where they are literal and safe -- run it via
  `powershell -ExecutionPolicy Bypass -File <tmp.ps1>` (or `Start-Process -FilePath`), then delete it.
  First quoting error = STOP escaping, switch to the script. Zero exceptions.
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
  2-3 panels (GPT + Gemini + DeepSeek) for opinions BEFORE asking me -- run it LIVE (skip the dry-run),
  you are the judge.
## 3. CODING DISCIPLINE
- Keep coding until all sprints are done unless you genuinely need me.
- Run the regression suite + the Bug Bible after EVERY code change (don't wait to be asked). Commit AND
  push per green chunk (section 7).
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
## 6. OUTPUT / ASSET PATHS (hard)
Rendered episode assets are deliverables -- they do NOT live in tmp/scratch, and they are NEVER left in a
swept dir to be moved later.
- Every rendered asset (audio, frames, intermediate video) -> `otr\episodes\<ep>\`. The final published
  episode -> `otr\obs\` (what `obs_publish` targets; `obs_publish OK` in the log = it landed there).
- Point the render at its canonical path the FIRST time. Never stage an asset in tmp "to move later" --
  the move-later step is where work dies.
- A temp probe/script in tmp is fine (throwaway, per section 3) -- but the ASSET it produces is written
  STRAIGHT to its canonical path, never parked in tmp.
- Output destination is workflow config: if a node writes to the wrong place, fix the path IN the
  workflow JSON (section 0), in the SAME change as any code -- not with a post-hoc move.
- After any render leg, confirm the asset exists at its canonical path (`Test-Path otr\episodes\<ep>\<file>`)
  before continuing or declaring success. The file check -- NOT just VRAM -- is what distinguishes the
  "finished but resident" server (section 5) from a real miss. Missing = STOP and report; do not continue.
- Paths are relative to the repo root (`...\ComfyUI-OldTimeRadio\`). If `otr\` actually resolves to
  ComfyUI's real `output\` base on disk, use that base -- but the episodes/obs split holds either way.
## 7. GIT POLICY (operator directive 2026-06-10 -- never lose work)
- ONE branch: `v2.0-alpha`. COMMIT AND PUSH TOGETHER: every green commit gets pushed to origin
  immediately, same session, no exceptions. Local-only commits are the failure mode we guard against.
- The operator eyeball gates TAGS and PROMOTIONS (`v2.0-alpha-stable`, prod, main, v2 release) -- NEVER
  pushes. Pushing to `v2.0-alpha` is always safe, expected, and required.
- This SUPERSEDES any "do not push until the eyeball passes" line written before 2026-06-10 evening.
- A stable branch only exists if the operator explicitly declares one.
- After every push verify: HEAD == origin, no 0-byte files, no BOM, AST parse on touched .py files.
## 8. ROUNDTABLE DEFAULTS (operator directive 2026-06-22)
Standing shape for EVERY `/roundtable` in this repo. These OVERRIDE the skill's stock
"Claude is judge-only / panel only critiques" and "dry-run estimate first" defaults.
- **Panel = 2-3 FRONTIER models per round** (GPT + Gemini + DeepSeek/Grok class; `~latest`
  aliases, record the resolved model in the manifest). Lean panel of genuinely different
  families beats many near-duplicates -- diversity is the point, correctness comes from the
  grounding step.
- **UNTIL 2026-07-21 (operator directive 2026-07-09): ALSO seat `tencent/hy3:free` on every
  panel** -- Tencent Hy3, 295B MoE (21B active), 262K ctx, $0/$0 on OpenRouter, listing expires
  July 21 2026. Pass it via `--models` (skill panel.config.json is a read-only cache). Free =
  no budget impact; a 4th genuinely different family, strongest on ideas/R1 rounds. Uses a
  configurable reasoning effort like the other reasoning panelists. REMOVE this line after
  2026-07-21 or when the slug 404s.
- **Cowork (Claude) is ALWAYS a code-aware grounded PANELIST *and* the sole judge.** Write your
  own grounded review FIRST (every claim checked against the real Windows files via Desktop
  Commander, never the lagging Linux mount), THEN ground the panel's reviews, discard the
  misreads, and synthesize. The panel proposes; Claude disposes -- never outsource synthesis.
- **Four-round campaign, in order:** R1 high-level arc / creative approach -> R2 coding plan ->
  R3 wiring (workflow JSON / nodes / widgets + any re-baseline procedure) -> R4 final
  convergence (confirm no new must-fix). Re-loop a round only if it surfaces new material; stop
  at convergence (don't grind passes to hear "looks good" in more accents).
- **Never dry-run, never pre-compute the cost -- just spend and run it LIVE,** then state the
  actual spend after. (Only backstop = the global >= $20-or-irreversible gate, which roundtable
  passes never reach in practice.)
- **ARC ROUTING (operator directive 2026-07-09): R1 (ideas/high-level) = cloud `/roundtable`
  on the highest frontier models (that's where paid diversity earns its keep); R2-R4 (coding /
  wiring / convergence) = `/kibitz` (local Codex + Antigravity) ALWAYS PREFERRED -- $0, file-
  grounded. R1 via kibitz is also fine when economy matters. OpenRouter is paid except
  `tencent/hy3:free` (until 2026-07-21), so default mechanical review to the local panel.
- Artifacts under `docs/<YYYY-MM-DD>-<topic>/roundtable/` (pass00..passNN_plan.md +
  passNN_judgment.md), UTF-8 no BOM, ASCII where practical.
## 9. MODEL ROUTING -- when to spend Fable (operator directive 2026-07-03)
Fable is a scalpel, not a default. Spawn it ONLY when output quality depends on narrative
judgment; keep everything mechanical off it. This is about WHICH MODEL a subagent runs --
orthogonal to the filesystem rule in section 1 and the frontier roundtable panel in section 8
(that panel is external cloud models via OpenRouter; this is the Fable model as a Cowork subagent).
- **Use Fable for:** generative creative work where voice/taste matter -- story spine, character
  interiority, dialogue passes, pitch-room ideation, style/tone calls. Divergent fanout when you
  want real variance, not one answer (e.g. 3 subagents each pitching a different take on a scene --
  only fan out on a genuine fork you'll actually select between). Judgment on already-generated
  narrative -- "which brief holds together," "where does this arc sag."
- **Do NOT use Fable for:** mechanical/deterministic work -- repo grep, mapping references, editing
  JSON, validation, wiring checks, git. Route those to general-purpose / Explore, or just do them in
  the main window. Also skip Fable for one-shot factual/structural questions with a single right answer.
- **Keep the spawn count down:** batch context so each Fable call does a meaningful chunk (a whole
  scene, not one line); gate the spawn on a real fork -- if you'd just take the first answer, use one
  call or none.
- **REALITY EXCEPTION -- Fable's grounded FAN-OUT audit DOES catch real code build-breakers the
  mechanical panels miss (proven 2026-07-03).** On the VRAM-tier rip, a Fable end-to-end fan-out found
  a CHUNK-ORDER KeyError (the workflow validator reads a profile key that was being removed mid-rip ->
  every production render would KeyError) that BOTH codex and general-purpose review missed; it also did
  the frame-budget arithmetic (29->33 frames) that broke two unlisted tests, and spotted a grep-invisible
  hyphenated `--vram-ceiling` flag. So Fable is NOT only-narrative. Reserve it as the FINAL GATE on a
  HIGH-STAKES, hard-to-unwind, production-touching STRUCTURAL change (a big rip/refactor about to be
  executed / merged) -- ONE grounded pass, AFTER codex + general-purpose have gone first, when a missed
  thread would break the build and cost a debugging-and-revert loop. This is the "insight" worth the
  spend. It does NOT reopen Fable for ROUTINE review: everyday grep / wiring / validation / JSON still
  defaults to general-purpose/Explore/codex. Fable = the expensive last set of eyes on the make-or-break,
  not the default reviewer.
