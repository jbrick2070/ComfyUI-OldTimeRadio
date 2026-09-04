# Collapse the registry scan by the one-owner rule -- coding plan, hardened by r2

**Provenance:** r1 hardened by Antigravity + a Sonnet seat + Cursor (manual);
r2 by Antigravity (CLI, Gemini 3.8 Flash High) + a Sonnet 5 seat + Codex by
the operator's manual paste (the CLI lane was quota-held both rounds; Cursor
did not review r2). Every claim below was grounded against the real Windows
files by the driver; the argv receipt
(`docs/2026-09-04-registry-findings-collapse/argv0_receipt.txt`) measured 35
execution sites, matching the scanner's 35 findings.

**Measured, not assumed** (alpha.17's real payload, 158 findings, all
`info`): the env rule fires ONCE PER FILE (103); the subprocess rule PER SITE
(35 across 20 files); six of the twelve "url command" hits are the words
`ffprobe -count_frames` in error strings; the network rule fires PER FILE
(`_otr_google_api/client.py` carries two `urlopen` sites and ONE finding --
answered in r2); the process, file and bytecode rules are one finding each
at one site. The
gate is ZERO findings or a manual admin approval; nothing here reaches zero
(ffmpeg is a subprocess and that is the render path). This plan changes what
the human reviewer reads: about eight lines instead of 158, and the
`credential-access` tag on one file instead of eleven.

## The invariant

*A machine fact has ONE owner, and a test proves the copies agree.* Same
shape as `nodes/_otr_shared/ffmpeg.py` + `tests/test_ffmpeg_single_resolution.py`,
and the allowlist-with-reasons shape of `tests/test_output_root_single_owner.py`.

## The collapses (final scope after r2)

**A. One env owner -- 103 -> 1, plus 1 by decision.** `nodes/_otr_shared/env.py`,
stdlib only, importing nothing from the pack, reading the live `os.environ`
on EVERY call (no cache: `conftest` pops names at import, hundreds of tests
`monkeypatch.setenv`, the launchers pin at boot). Surface, stated exactly:

* `get(name: str, default: str | None = None) -> str | None` -- returns
  `os.environ.get(name, default)` unchanged. A site that reads
  `os.environ.get("X") or "d"` becomes `env.get("X") or "d"`; the `or`, the
  cast and the default stay at the site. Spelling only.
* `pin(name: str, value: str) -> None` -- `os.environ[name] = value`; a
  `None` value raises `TypeError` naming the knob (never a silent unpin).
* `setdefault(name: str, value: str) -> str`.
* `unpin(name: str) -> str | None` -- `os.environ.pop(name, None)`; never
  `KeyError`.
* `snapshot() -> dict[str, str]` -- a COPY of `os.environ`, for the two
  mapping consumers (`eng_mesh_stage.py:809` hands the environment to
  `build_blender_env`; `route_freeze.py:72` reads it with `.get`). Read-only
  consumers; a copy is correct.

No typed getters, no catalog, no credential names inside it. The twelve
mutation sites migrate to the write verbs; the writer's style-grammar
restore keeps its own branch (`None` -> `unpin`, else `pin`). The two tool
owners (`ffmpeg.py:71`, `ffprobe.py:172`) migrate in the first batch.
`prestartup_script.py` KEEPS its five inline writes by decision and stays
one finding.

**A's guard** is an AST walk over `nodes/**/*.py` plus the two root files by
EXPLICIT PATH (`__init__.py`, `prestartup_script.py`; the gitignored
`scratch_check_server.py` is excluded by construction). It resolves
`import os as X` aliases (`__init__.py:510` aliases `os` as `_otr_ro`) and
`from os import environ / getenv`, and flags: `environ` as an attribute or
bare name in a `Subscript` (read or write: `eng_humo.py:423`,
`eng_mesh_stage.py:508`), `.get` / `.pop` / `.setdefault` calls on it,
`getenv`, `putenv`, `unsetenv`. Comments, docstrings and string literals do
not count. Allowlist: `env.py` and `prestartup_script.py`, each with a
reason string.

**A-0. The default-drift list, BEFORE the rename.** A script under
`scripts/` (unshipped) prints `name -> {default: [sites]}` for every knob
read at more than one site with a different default or cast. Its output is
pasted into the plan as a receipt; anything on it is fixed in its own commit
with a test, never inside a rename commit. Follow-on, not this arc: three
named `_bool_env` functions plus dozens of inline `== "1"` checks.

**B. One process owner -- 35 -> 2.** `nodes/_otr_shared/proc.py`, stdlib
only, owning exactly two execution sites: `run(argv, **kwargs)` and
`popen(argv, **kwargs)`. Both forward EVERY keyword to `subprocess` (sites
pass `stdout=DEVNULL`, `stderr=<open file>`, `stdin=PIPE`, `text`,
`encoding`, `bufsize`, `cwd`, `env`, `timeout`, `check`); both refuse
`shell=True` and a `str` argv with a named error; `run` returns the real
`CompletedProcess`, `popen` the real `Popen` (so `with proc.popen(...) as p`
keeps working at the sidecar sites); neither wraps exceptions --
`CalledProcessError`, `TimeoutExpired`, `OSError` propagate as-is. The owner
re-exports `PIPE`, `DEVNULL`, `STDOUT`, `CompletedProcess`,
`CalledProcessError`, `TimeoutExpired` so a migrated module can drop
`import subprocess` entirely. `check_output` is NOT in the owner: its one
site (`production_ledger.py:222`) becomes
`proc.run(argv, stdout=PIPE, stderr=DEVNULL, timeout=5, check=True).stdout.decode("utf-8", errors="ignore")`
-- bytes mode kept, so the decode policy does not move.

**B's executable allowlist -- the r2 question, answered YES.** `proc.py`
checks `argv[0]` by basename, lower-cased, `.exe` stripped, against a named
set with a reason each: `ffmpeg`, `ffprobe`, `python*` (the three sidecar
venv interpreters: chatterbox, dia, indextts2 -- `startswith("python")`
because interpreters are `python3`, `python3.12`, `pythonw`), `git`
(`_otr_ledger.py:884`, `production_ledger.py:222`), `nvidia-smi`
(`_otr_sys_specs.py:110`), `blender` (`eng_mesh_stage.py:810`). No `py`
launcher: the sidecars resolve `_venv_python()`, a venv `python.exe`.
Anything else raises a NAMED error BEFORE spawning. That is the measured
set; a new executable is a one-line, reviewed addition. It turns the owner
from a pass-through into a boundary a human reviewer can read. Sonnet's r2
objection -- a per-box absolute path for blender or an interpreter -- is
exactly what basename matching normalizes, and its residual risk is kept as
a test: every argv[0] basename in the receipt passes the check, and an
unlisted one raises the named error.

**B's guard** is an AST rule on CALLS whose callee resolves to
`subprocess.run | Popen | check_output | check_call | call` or
`os.system | popen` outside the owner. `subprocess.PIPE`,
`CompletedProcess` annotations and `except CalledProcessError` are not
findings. Two test-side obligations travel with the FIRST batch:
`tests/test_terminal_frame.py:356-372` (the clip-encoder spawner sweep)
learns `from ._otr_shared import proc` / `from ._otr_shared.proc import run,
popen` and classifies `proc.run` / `proc.popen` as spawn calls -- otherwise
its `:495` assertion on `scope_draw.py::encode_silent_mp4` goes red the
moment scope_draw migrates; and every test that patches
`<module>.subprocess.run/Popen` (twelve files: ten via `monkeypatch.setattr`,
two via `mock.patch.object` and a dotted-path `mock.patch`) migrates to
patching the `subprocess` MODULE's attribute, which the owner looks up at
call time, in the batch that migrates the module under test.

**C. Network -- CUT.** The five FILES (six sites: `requests` in two
backends, `urllib` at `client.py:187` and `:235` and `cloud_media_invoke.py:571`,
the SSRF-hardened socket in the RSS fetcher) stay as NAMED exceptions with
reasons in a network guard of the `test_output_root_single_owner.py` shape,
keyed by file because that is how the rule counts. A urllib helper would buy at most one finding and possibly none (the
scanner may key on `import urllib.request`), for a new module and an unknown
regex; the 2026-09-02 review request already names and justifies all five.
The urllib helper from r1 is withdrawn.

**D. The three singletons -> 0, one of them carefully.** `gpu_residency.py`
(`:73-74` tries `psutil.pid_exists` first; `psutil` is line 20 of ComfyUI
CORE's `requirements.txt` in the booted install,
`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\requirements.txt:20`, and
7.2.2 imports in the venv -- it is not in OTR's own requirements and does not
need to be). The exact shape: replace the whole
`if os.name == "nt":` block at `:77-90` with `if os.name == "nt": <log once
at WARNING: liveness unknown without psutil>; return True` ("do not steal"),
which makes the currently UNCONDITIONAL `os.kill(pid, 0)` at `:91-97`
POSIX-only, where `ProcessLookupError` -> False and `PermissionError` ->
True. The ctypes `OpenProcess` block goes; the `os.kill`-terminates-on-Windows
trap is closed by the platform branch, not by hope. Test: Windows + psutil
import failure -> True, `os.kill` never called, one WARNING.
`eng_spandrel_esrgan.py:259`: chunked `open(path, "rb")` sha256 (also removes
a whole-file RAM spike). `eng_ltx25.py:287`: `import sys`.

**E. The six strings.** `wan_shared.py:202-232`: "the frame-count probe"; no
test asserts the old wording (checked). One comment beside them says why:
error text names the probe, not the CLI syntax, because the registry scanner
matches command syntax inside strings.

## Test-side obligations, collected (each ships in the batch it names)

* `tests/test_master_mux_terminal_knob.py:94` -- its unguarded-read predicate
  matches `"environ" in ast.dump(fn)`; it must also match `env.get(...)` or
  it passes vacuously after the mux migrates.
* `tests/test_terminal_frame.py` -- the spawner sweep (above), first batch.
* The twelve subprocess-patching test files -- per batch (above).
* `tests/test_ffmpeg_single_resolution.py` -- `env.py` must never spell
  `OTR_FFMPEG` or call `which()`; run it, do not assume it.

## Build shape (r3 hardens the wiring and order)

1. **A-0's drift list** (read-only, a receipt).
2. **Owners + guards as RATCHETS, first commit.** `env.py`, `proc.py`, each
   guard shipping with an explicit `_PENDING_MIGRATION` set of files
   asserted in BOTH directions: no offender outside the set, every file
   inside it still offends. A count can be gamed by offender swapping; a
   set cannot, and a converted file left in the set fails the test. Each
   batch shrinks the set; the last batch empties it and the allowlist
   remains.
3. **Batches, one commit each, full suite green per commit:** (a)
   `_otr_shared` + the three singletons + the six strings + the two tool
   owners; (b) the engine subpackages, one commit each: audio, image,
   upscale, video, google_api; (c) the internal libraries `nodes/_otr_*.py`;
   (d) the top-level nodes `nodes/otr_*.py` + `OTR_*.py` and the two roots.
4. **Local proxy for acceptance:** the two guards green with empty pending
   sets; `grep -rl 'os\.environ\|getenv' nodes/ __init__.py
   prestartup_script.py` lists exactly the owner and the prestartup file.
   The next PUBLISHED version's scan is the receipt and the only number the
   review request quotes.

## Revised floor, conditional on the next scan

env 1 + 1 (prestartup, by decision) + proc 2 + network 5 named exceptions +
singletons 0 + strings 0 = **about 9 findings**, all `info`, from 158. Not
Active -- that is zero or the manual review -- but one screen, and the
`credential-access` tag on one file instead of eleven.

## Not this arc, said so it stays out

Active without the manual review; any env name, default, precedence or cast;
command construction; `argv()` on the ffmpeg owner (deferred: unknown
value); `scripts/`; the knob catalog; typed getters (follow-on, fed by A-0);
a urllib helper; `pyproject.toml` (a publish is the operator's eyeball).

## Risks r3 should keep trying to break

Import order at boot (`__init__.py` pins `OTR_OUTPUT_DIR` at line 51 -- can it
import `nodes/_otr_shared/env.py` that early, before the package's own
imports?); the twelve patched tests across two boxes pushing concurrently
(the 4060 owns some of those test files' subjects); the ratchet set drifting
between the two boxes' commits; the allowlist refusing a legitimate binary
the receipt did not see because a branch never ran on this box (the pod's
`ffmpeg` path, a Linux `python3.12` -- both pass the basename rule, but r3
should look for a fourth interpreter or tool behind a branch); the last
batch's proof is a canonical leg that PUBLISHES to `otr/obs/`, not a green
suite.
