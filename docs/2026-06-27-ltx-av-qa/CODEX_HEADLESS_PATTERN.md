# Robust headless Codex (+ AntiGravity reality) -- for /roundtable-local

Proven on this box 2026-06-27 (codex-cli 0.142.3): file-output contract works; quick test wrote
`codex_final.md` = `CODEX_FILE_OK`, exit 0, JSONL log present. No more stdout scraping.

## The proven Codex pattern (use this everywhere)
Pipe the prompt via stdin (`-`); Codex writes its FINAL answer to a file via `-o`; success =
exit code 0 AND that file exists. Read the answer from the file.

    Get-Content prompt_codex.md -Raw |
      codex exec -C <repo> --sandbox read-only --json --color never -o codex_final.md - *> codex_log.jsonl

- `--sandbox read-only` for REVIEWS (cannot edit -- correct posture for a panelist).
- `--full-auto` (workspace-write) only when Codex is delegated to EDIT.
- `--dangerously-bypass-approvals-and-sandbox` ONLY inside an isolated git worktree/clone, never
  the real home dir.
- Never scrape terminal text for DONE. Never set a short subprocess timeout (Codex batches; can
  take minutes). Don't run two full-permission Codex jobs in the same repo at once -- use worktrees.

Reusable runner (in this repo): `scripts/run_codex_agent.ps1`
  - `pwsh scripts/run_codex_agent.ps1 -RunId r4 -Mode review`   (read-only)
  - `pwsh scripts/run_codex_agent.ps1 -RunId wire1 -Mode edit`  (full-auto)
  It reads `roundtables/runs/<RunId>/prompt_codex.md` and writes
  `codex_final.md` + `codex_log.jsonl` + `codex_exitcode.txt`.

## The exact change for the roundtable-local skill (agent_roundtable.py)
The skill currently does `CODEX_CMD = ["codex", "exec"]` and captures stdout, with
`ok = returncode==0 and stdout.strip()`. Replace the Codex leg with the file contract:

    out_md = run_dir / "codex_final.md"
    cmd = ["codex", "exec", "-C", str(repo), "--sandbox", "read-only",
           "--json", "--color", "never", "-o", str(out_md), "-"]
    with open(run_dir / "codex_log.jsonl", "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, input=prompt, text=True,
                              stdout=log, stderr=subprocess.STDOUT, timeout=None)
    ok = proc.returncode == 0 and out_md.exists()
    review_text = out_md.read_text(encoding="utf-8") if ok else ""

(Codex must be on PATH, or use the full exe at
`C:\Users\jeffr\AppData\Local\OpenAI\Codex\bin\<hash>\codex.exe`.)

## AntiGravity (agy) headless -- SOLVED via the FILE-HANDOFF protocol (proven 2026-06-27)
`agy` is installed + signed in (jbrick2070@gmail.com, Google AI Pro). Its `--print`/`-p` SWALLOWS
stdout when redirected on Windows (needs a real TTY) -- capturing stdout fails (exit 0, empty),
tested every way (`-p`, `--new-project`, sessions killed, repo cwd, patient wait). The FIX: agy is
an AGENT, so instruct it IN THE PROMPT to WRITE its review to a file with its own write tool, then
read that file -- stdout is never used. PROVEN: `agy --dangerously-skip-permissions -p "...write
AGY_FILE_OK to <file> then stop..."` -> the file contained `AGY_FILE_OK`, exit 0.

Runner: `scripts/run_agy_agent.ps1` (reads prompt_agy.md, appends the strict write directive, runs
agy with skip-permissions, reads agy_final.md; success = exit 0 AND file exists; `-Worktree` for
hard isolation).

Of the three proposed bypasses: **#3 file-handoff WORKS**; #1 `| clip` does NOT (still a stdout
redirect -> empty); #2 log-scraping is an untested, fragile fallback.

SAFETY: file-handoff needs `--dangerously-skip-permissions` (UNSANDBOXED agent). Mitigate with the
review-only prompt directive + the git-committed repo (any stray edit shows in `git status`, is
revertible) + optional `-Worktree` (agy runs in a throwaway checkout, can't touch your real tree).
Codex's `--sandbox read-only` is a HARDER guarantee, so Codex stays the primary lane; agy is the
real second voice.

So /roundtable-local now has a true 2-agent HEADLESS local panel on this box: Codex
(run_codex_agent.ps1) + AntiGravity (run_agy_agent.ps1). Claude still writes its own grounded
anchor and is the sole judge.

## How to actually update the saved skill
Claude (Cowork) CANNOT persist edits to the saved `roundtable-local` skill from a chat session --
the skill files it can see are a READ-ONLY cache. To make the change stick, edit the skill in
**Settings > Capabilities** (or your skills source) and paste the Python change above. The repo
runner `scripts/run_codex_agent.ps1` captures the same improvement so it's usable immediately
regardless.
