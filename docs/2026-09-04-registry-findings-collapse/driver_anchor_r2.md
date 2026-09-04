# Driver anchor, r2 -- registry findings collapse (Claude / Fable 5.1, sole judge)

Written BEFORE the r2 fan-out, against the r1 `final.md` and the real tree.
Round focus: implementability of the coding plan.

VERDICT: yes-with-fixes. The three owners are small and stdlib-only, the
migration is spelling-only by rule, and the guards have a proven shape. What
is NOT yet pinned down is exactly what the implementor types: the owner
signatures, the exception contract of the process owner, the AST guard's
precise rule set, and which of the two r2 questions is answered.

MUST-FIX BEFORE BUILD (driver's own):
1. [A] Signature, stated: `get(name: str, default: str | None = None) ->
   str | None` returning `os.environ.get(name, default)` UNCHANGED; `pin(name,
   value: str)`; `setdefault(name, value: str) -> str`; `unpin(name) -> str |
   None` (the `pop`). No `bool`/`int`/`Path` parsing, no name registry. A
   call site that reads `os.environ.get("X") or "d"` migrates to
   `env.get("X") or "d"` -- the `or` stays at the site.
2. [A] The guard is an AST walk, not a grep: flag `Attribute` chains rooted at
   the `os` module (`os.environ`, `os.getenv`, `os.putenv`, `os.unsetenv`) and
   any `Subscript`/`Call` on `environ` (`environ[...] =`, `.get`, `.pop`,
   `.setdefault`), plus `from os import environ` / `getenv` imports. Comments,
   docstrings and string literals do not count (the current grep proxy
   over-counts them). Scan `nodes/**` plus the two root files by explicit
   path; the allowlist is `nodes/_otr_shared/env.py` and
   `prestartup_script.py`, each with its reason string.
3. [B] The process owner must NOT change the exception contract: callers catch
   `subprocess.CalledProcessError`, `subprocess.TimeoutExpired` and `OSError`
   (`FileNotFoundError` when a binary is missing). `proc.run` returns the
   `CompletedProcess`, re-raises as-is, and never wraps. `proc.popen` returns
   the real `Popen` so `with proc.popen(...) as p:` keeps working at the
   sidecar sites. `check_output` is its own function (production_ledger.py:222
   uses it) -- three call sites in the owner, three findings, by the per-site
   rule.
4. [B] The guard for B is an AST rule on CALLS whose callee resolves to
   `subprocess.run|Popen|check_output|check_call|call` or `os.system|popen`
   outside the owner; `subprocess.PIPE`, `subprocess.CompletedProcess`,
   `subprocess.CalledProcessError` in an `except` are not findings.
5. [Build] The ratchet question, answered: the guards ship FIRST, as ratchets
   -- each asserts `offenders <= N` with N written in the test and lowered in
   every batch commit -- and flip to the allowlist form in the final batch.
   Reason: both boxes push to `v2.0-alpha`; a guard that ships last protects
   nothing for the days the migration takes, and a new `os.environ` read
   added by the other box mid-migration would land silently. A ratchet turns
   that into a red test the same hour.
6. [D] The Windows liveness path without psutil returns True ("do not steal")
   and must LOG ONCE at WARNING that liveness is unknown, so a stale lock is
   diagnosable from the server log instead of looking like a live occupant.
   The POSIX branch keeps `os.kill(pid, 0)` with `ProcessLookupError` -> False
   and `PermissionError` -> True (alive, not ours).

SHOULD-FIX:
1. [B] The executable-allowlist question: the driver's lean stays yes, but as
   a BASENAME allowlist checked on `argv[0]` after `os.path.basename` and
   lower-casing (`ffmpeg`, `ffprobe`, `python`, `pythonw`, and whatever the
   enumeration below adds), raising a NAMED error on anything else. The
   enumeration of today's `argv[0]` literals is a build-time receipt; the
   panel should list what it finds so the allowlist is measured, not guessed.
2. [C] The urllib helper's signature: `open_json(url, *, timeout, headers)`
   and `download(url, dest, *, timeout)`; the two sites' timeouts and headers
   are preserved verbatim (spelling-only applies here too).
3. [A-0] The default-drift list is a script under `scripts/` (unshipped), not
   a test: it prints `name -> {default: [sites]}` for every name read at more
   than one site, and the receipt is its output pasted into the plan.

CHECKED-CLEAN by the driver: `_otr_shared/ffmpeg.py` and `ffprobe.py` are the
only files that may keep `which()` (the ffmpeg guard keys on it); `env.py`
importing nothing from the pack means no cycle with either; `psutil` is in
ComfyUI core `requirements.txt`; `tests/conftest.py` pops at import and the
owner's per-call read preserves that.

UNVERIFIABLE until built: the residual count (the scanner is private); whether
`$media_list_assign_indirect` matches the `cmd = [` shape or the literal;
whether `python_network_operations` dedupes per file or per site.
