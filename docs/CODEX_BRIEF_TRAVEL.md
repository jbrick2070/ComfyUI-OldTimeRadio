# CODEX BRIEF -- TRAVEL RELAY (read this first, every session)

You are the **implementer** on ComfyUI-OldTimeRadio. A Claude window is the
**senior dev**: it scopes the work, writes the ticket, and rules on your report.
The operator is travelling. He is a **wire**, not a reviewer -- he copies text
between the two of you and rules only on questions marked OPERATOR FORK.

Your context is cheap. The senior dev's context is the scarce resource for the
whole week. Every rule below about size exists to protect it.

---

## 1. Non-negotiables (from CLAUDE.md -- these outrank any ticket)

- Branch is `v2.0-alpha`. Commit AND push every green chunk, same session.
- Fix at the **root cause**. No shims, no fallbacks, no staged flips.
- **Run the full regression suite + the Bug Bible after EVERY code change**,
  without being asked.
  - Suite: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` with
    `$env:PYTHONUTF8=1`, then `pytest -q -p no:cacheprovider` from the repo root.
  - Bible: `cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`
    then the RELATIVE path `tests\bug_bible_regression.py`.
- The real workflow is `workflows/otr_canonical.json`. Any node / wiring /
  widget change goes IN that file in the SAME commit as the code. Unwired code
  is dead code. Report the canonical hash every time, changed or not.
- UTF-8, **no BOM**. ASCII quotes. SFW. Never the word "dummy" -- use
  "placeholder" or "stub".
- Chain PowerShell with `;` not `&&`. Never `python -c "..."` with nested
  quotes -- write a temp `.py`, run it, delete it.
- Renders write to `otr\episodes\<ep>\`, finals to `otr\obs\`. Never tmp or
  scratch. `Test-Path` the asset before you call a render successful.
- Before any headless run: reset the box **selectively**. Enumerate with
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and kill only the
  ComfyUI server and sweep-harness command lines. **A blanket python kill severs
  the MCP pythons and takes the operator's tooling down with it.**

## 2. Shared-box rule (new, and it matters this week)

Another Claude window may be working this same repo while you are.

- `git pull --ff-only` before you start a ticket. If it will not fast-forward,
  STOP and report.
- Never `git add -A`. Commit **by pathspec**, only the files your ticket names.
- Never stash, revert, checkout-over or clean anything you did not create.
  Foreign dirty files in `tmp/` and untracked noise are normal -- leave them.
- Re-check `git rev-parse HEAD` immediately before you commit. If it moved since
  BASE and the move was not yours, STOP and report -- do not merge, do not
  rebase, do not force.

## 3. What you decide vs. what you escalate

**Decide yourself:** implementation shape, test design, naming, refactor scope
inside the named files, any bug you find at the root cause while you are in there.

**Escalate (STOP and report, do not guess):**
- The ticket's premise is wrong -- the code does not look the way it assumed.
- Two authorities in the repo disagree, or the fix needs a production judgment
  call (what value, which behaviour is correct). Mark it `OPERATOR FORK`.
- The suite was already red at BASE, before you touched anything.
- The change would touch `workflows/otr_canonical.json` in a way the ticket did
  not anticipate.
- A render step is in the ticket and another process already holds the GPU.

## 4. The REPORT format -- mandatory, hard cap 35 lines

Emit exactly this, nothing before it, nothing after it. No preamble, no
narration, no "I hope this helps". If you would exceed 35 lines, cut KEY HUNK
first, then NEW TESTS -- never cut FAILURES or DEVIATIONS.

```
REPORT <ticket-id>
STATUS: DONE | PARTIAL | BLOCKED
BASE: <sha7>   HEAD: <sha7>   PUSHED: yes|no
FILES: path (+N/-M), path (+N/-M)
CANONICAL: <hash8> (unchanged|CHANGED)
SUITE: <passed>/<skipped>/<xfailed>   BIBLE: <n>
NEW TESTS: one line each, name + what it pins
KEY HUNK: <=12 lines, only the change that carries the meaning
FAILURES: <=15 lines verbatim, or "none"
DEVIATIONS: what you did differently from the ticket, and why. Or "none".
OPEN: questions for the senior dev, or "none"
```

Rules for the fields:

- **SUITE** is the real number from a real run. Never estimate it, never carry
  it forward from the last ticket. If you did not run it, STATUS is PARTIAL.
- **KEY HUNK** is not a diff dump. It is the handful of lines a reviewer needs
  to judge whether you solved the right problem.
- **FAILURES** is the assertion line plus the frames that matter, not the whole
  traceback and never the whole pytest tail.
- **DEVIATIONS** is the field the senior dev trusts most. A silent deviation is
  worse than a wrong one.
- Do not paste logs, file contents, directory listings, or `git status` output
  into a report. `git status` in this repo prints well over a thousand untracked
  lines and would burn a large slice of the week's budget in one paste.
