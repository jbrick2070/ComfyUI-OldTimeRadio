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

## AntiGravity (agy) reality on this box -- do NOT rely on it headless
`agy` is installed + signed in (jbrick2070@gmail.com, Google AI Pro) and the INTERACTIVE CLI
works. But headless `agy -p` with a captured/redirected stdout returns EMPTY (exits 0, writes
nothing to the pipe) -- tested with `-p`, `--new-project -p`, sessions killed, repo cwd, patient
wait. Its `--print` answer goes to the TUI/session store, not a pipe (a TTY requirement). So:
- roundtable-local's automated panel = **Codex only** on this machine.
- AntiGravity contributes the second voice by being run in its INTERACTIVE window (or the IDE) and
  the result pasted in -- exactly how Gemini's review reached the 2026-06-27 convergence.
- Full agy automation later would need a ConPTY/winpty bridge or an agy build with a file-output
  flag.

## How to actually update the saved skill
Claude (Cowork) CANNOT persist edits to the saved `roundtable-local` skill from a chat session --
the skill files it can see are a READ-ONLY cache. To make the change stick, edit the skill in
**Settings > Capabilities** (or your skills source) and paste the Python change above. The repo
runner `scripts/run_codex_agent.ps1` captures the same improvement so it's usable immediately
regardless.
