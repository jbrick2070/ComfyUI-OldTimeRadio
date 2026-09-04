# Driver anchor, r4 -- registry findings collapse (Claude / Fable 5.1, sole judge)

Written BEFORE the r4 fan-out, against the r3 `final.md` and the real tree.
Round focus: convergence -- residual defects only; no new design.

VERDICT: yes-with-fixes, converging. Three rounds have not moved the
architecture since r1 (two stdlib owners, named-set ratchets, spelling-only
migration, an obs leg as acceptance); what moved is the wiring detail, and
r3 closed the last open seams (aliases, the sweep's names and import forms,
per-batch seam patches, batch scopes that match file sets, orphans first).
What remains is a short list the build must not forget.

MUST-FIX BEFORE BUILD (driver's own, residual):
1. [Phase 0] Every rip is re-verified at build time with `git grep -n -w -F`
   on the CURRENT tree, not the audit's line numbers (the audit is hours
   old and `video_engine.py` has moved since). A symbol that has grown a
   caller is wired, and the plan row says so.
2. [A] `snapshot()` returns `dict(os.environ)` -- a plain dict, not
   `os.environ.copy()` (which is also a dict, but say the type so no one
   returns a live `_Environ` view by mistake). `route_freeze.py:72` becomes
   `src = otr_env.snapshot() if env is None else env`, behaviour-equal
   because the function only calls `.get`.
3. [B] The allowlist dict is keyed by NORMALIZED basename (lower, `.exe`
   stripped) and the interpreter rule is a separate predicate
   (`startswith("python")`), so the dict never needs a `python3.12` key. The
   unit test enumerates the receipt's basenames PLUS the Linux shapes
   (`/usr/bin/ffmpeg`, `/workspace/.venv/bin/python3.12`).
4. [Ratchet] The shared helper lives at `tests/fixtures/ratchet.py`
   (`tests/` is a package; `.comfyignore` strips it from the zip) and takes
   `(pending: set[str], predicate, roots)`; both guard tests call it. The
   sets are relative POSIX paths from the repo root so the 4060's pull and
   the pod compare equal.
5. [Sequence] Commit order is fixed and each commit runs the FULL suite:
   Phase 0 (rips) -> A-0 receipt (no code) -> ratchet commit (owners,
   guards, network allowlist, terminal-frame sweep) -> (a) `_otr_shared`
   -> (b) audio, image, upscale, video, google_api -> (c) `_otr_*.py` ->
   (d) the rest of `nodes/` + root `__init__.py` -> obs leg. About eleven
   commits, about eighty minutes of suite time.
6. [Acceptance] The obs leg is run from a launcher whose `--profile`,
   roots and workflow path are READ BACK from the leg log before the
   receipt is written (two derived launchers lied today).

SHOULD-FIX:
1. [A] `pin(name, value)` also rejects a non-`str` value (an `int` default
   passed by mistake) with the same `TypeError`; `os.environ` would raise a
   less useful one.
2. [B] The named error class is `ExecutableNotAllowed(RuntimeError)` and its
   message carries the normalized basename AND the raw `argv[0]`.
3. [Docs] README's knob table is untouched by this arc (names do not move);
   the PROD_BUG_LOG gets no entry unless a live leg shows a defect.

CHECKED-CLEAN by the driver: no workflow JSON, widget, node signature or
`pyproject.toml` change anywhere; `.comfyignore` unaffected; `scripts/`
untouched except the Phase 0 rips inside it (unshipped files, safe).

UNVERIFIABLE until built: the residual count on the next published scan;
whether the scanner counts the owner's re-exported names.
