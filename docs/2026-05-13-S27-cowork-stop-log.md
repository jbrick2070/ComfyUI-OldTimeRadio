# S27 Cleanbreak Tail — STOPPED at Phase 0 preflight

## STOPPED — 2026-05-13 (run start)

- **Trigger:** Phase 0 preflight failure — three independent blockers detected before any Item 1/2 work could begin. Directive Phase 0 says `git status --short` non-empty aborts the sprint; it is massively non-empty (~600 files staged) AND the index is corrupt AND HEAD points at a non-existent branch.
- **Current state:**
  - Branch HEAD pointer: `refs/heads/s26-cleanbr` (typo / truncated — this ref does NOT exist on disk)
  - Real branches present and healthy:
    - `refs/heads/s26-cleanbreak` → `5bf9d3ad8e46e64b54eb244fa06e907a8fb4247c` (matches directive's `5bf9d3a` cut point — intact)
    - `refs/heads/v2.0-alpha` → `98489da90f4289f7ada27fe934faa38becb44449`
    - `refs/heads/main` → `0aa6d6e1ba8c39c8ec616c6a56dbd779ac6b3304`
    - `refs/heads/s25-musicgen-parity` present
  - `.git/index`: DIRC magic + version 2 present, file size 41153 bytes, but `git fsck` reports `error: bad index file sha1 signature; fatal: index file corrupt`. Content is parseable; only the trailing checksum is wrong. No object corruption — pack/loose objects are fine, refs are fine, reflog is fine.
  - Branch `s27-cleanbreak-tail` NOT cut. No edits made. No commits made. Repo is exactly as Cowork found it.
- **What needs Jeffrey:** Re-point HEAD to the real ref and rebuild the index from HEAD — two commands — then re-launch S27 on the Windows venv where the regression gates can actually run. See "Recovery" below.

---

## What Cowork already tried

Static-only inspection, no writes to git state:

| Probe | Result |
|---|---|
| `git status --short` | ~600 staged-add entries + ~40 AM (added-then-modified) — non-empty by a wide margin. Directive §3 says "Any output here aborts the sprint." |
| `git branch --show-current` | `s26-cleanbr` (truncated; not a real ref) |
| `cat .git/HEAD` | `ref: refs/heads/s26-cleanbr` |
| `cat .git/refs/heads/s26-cleanbreak` | `5bf9d3ad8e46e64b54eb244fa06e907a8fb4247c` (intact, matches directive) |
| `git fsck --no-progress` | `error: bad index file sha1 signature; fatal: index file corrupt`; `notice: HEAD points to an unborn branch (s26-cleanbr)` |
| `head -c 8 .git/index \| xxd` | `4449 5243 0000 0002` — magic = `DIRC`, version = 2 (file body looks intact, only trailing checksum is the issue) |
| `cat .git/logs/HEAD \| head` | Reflog intact, showing legitimate commit history through May 2026 |
| `cat .git/packed-refs` | Healthy; tags v1.0–v1.5+ intact, `refs/heads/main` packed at `b719490af516d2f17ed281eb1a5e44110b3adb95` |

No autonomous repair attempted because:
1. The wrong HEAD name (`s26-cleanbr` vs `s26-cleanbreak`) is a diagnostic clue. Some tool wrote that — silently rewriting HEAD would lose forensic evidence about whatever process did it.
2. CLAUDE.md global rule: "Run regression tests after every code change without being asked." This Linux sandbox cannot reach `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` or the Bug Bible at `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\tests\bug_bible_regression.py`. Even if I made code edits, I could not satisfy the regression gate. A change with no regression is not a change per project rules.
3. Directive §11 stop condition "Design judgment required" covers this. Cowork's autonomy authorization in §0 is for the cleanbreak deletes — not for unilateral repair of an unexpected git state on the user's working repo.

---

## Recovery (for Jeffrey, on the Windows machine)

Run from `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio` in **cmd shell** (per project rule "Never use PowerShell for git"):

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

REM 1. Verify the diagnosis still matches what Cowork saw
type .git\HEAD
REM Expect: ref: refs/heads/s26-cleanbr   (the truncated, broken pointer)

git fsck --no-progress
REM Expect: bad index file sha1 signature; HEAD points to unborn branch s26-cleanbr

REM 2. Re-point HEAD at the real ref
git symbolic-ref HEAD refs/heads/s26-cleanbreak

REM 3. Rebuild the index from HEAD's tree (drops the corrupt-checksum index)
del .git\index
git read-tree HEAD
git checkout-index -a -f

REM 4. Verify clean
git status --short
REM Expect: empty (or only files you legitimately have uncommitted)

git log -1 --format=%H%n%s
REM Expect: 5bf9d3a... and the S26 close subject

git fsck --no-progress
REM Expect: clean (no index errors)

REM 5. If clean, the S27 directive's Phase 0 can proceed normally:
git checkout -b s27-cleanbreak-tail
```

**Before running the recovery — consider what wrote the truncated HEAD.** Most likely culprits:
- A path-length-truncated filesystem operation (Windows NTFS edge case with long paths under the `local-agent-mode-sessions` tree)
- A previous Cowork / Claude session that wrote `s26-cleanbr` to HEAD after path truncation
- A tool that read HEAD's ref name through a buffer too small for `s26-cleanbreak\n`

If any process is currently holding a write handle on `.git/HEAD` or `.git/index`, finish that first.

---

## After recovery: re-launch S27

The directive is sound. The work queue is unchanged. Re-paste the directive into a fresh Cowork session **on the Windows venv** (where pytest + Bug Bible regression can actually run) and Phase 0 should produce the durable baseline cleanly.

The static-only portion of S27 (the `git grep` audits, the deletions, the workflow JSON scrub, the `tools/validate_workflow_links.py` checks) is Cowork-runnable. The regression gate at Phase 5 needs to be on Jeffrey's machine regardless — that is unchanged from the directive's design.

---

## Why Cowork did NOT just fix it

Senior pair-programmer test: a senior who walked up to a teammate's machine, saw `git fsck` reporting index corruption AND a typo'd HEAD, would not silently rewrite HEAD and rebuild the index without asking — they'd ask "what were you doing when this happened?" because the symptom suggests a tool bug or a concurrent-process bug worth understanding. The 60 seconds of human input averts a class of recurring failure.

This is also the directive's own posture: §11 stop condition "Design judgment required → STOP."
