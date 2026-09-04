# Driver anchor, r1 -- registry findings collapse (Claude / Opus 5, sole judge)

Written BEFORE the fan-out, from the real scan payload and the real tree.

VERDICT: the plan is buildable and its arithmetic is grounded, but it makes
three claims the panel should break before code: that the env rule is per-file
(CONFIRMED: 103 findings across 103 distinct files, one each, in the payload),
that a spelling-only migration of 101 files is semantics-neutral (PLAUSIBLE --
true only if the owner reads `os.environ` live on every call and preserves
every call site's default and cast exactly), and that a typed knob catalog is
worth the cost (UNVERIFIED -- a design choice with a real downside: every read
site's meaning gets touched once).

MUST-FIX BEFORE BUILD (driver's own):
1. [A] The owner must be `os.environ`-live on every call and cache nothing.
   `tests/conftest.py` pops `OTR_OBS_DIR`, `OTR_VOICE_REFERENCE_BANK` and the
   Google keys at import; hundreds of tests `monkeypatch.setenv`; the launchers
   pin at boot. One cached read breaks all of it silently.
2. [A] Preserve every call site's DEFAULT and CAST exactly during migration --
   grep shows at least four hand-rolled bool parsers ("1"/"true"/"yes"/"on")
   and int parses with `or` fallbacks; a typed getter that normalizes them
   changes behaviour on the 4060 without anyone noticing. Rule: the migration
   commit for a file changes only the SPELLING of the read; typing is a
   separate, later commit with its own test.
3. [A] The guard must flag `os.getenv`, `os.putenv`, `os.unsetenv`, `environ.pop`
   and `os.environ.setdefault`, not only `os.environ.get`, or the count lies.
4. [B] Do not merge `proc.py` into the ffmpeg owner. Resolution and execution
   are different facts; the ffmpeg owner stays stdlib+ffprobe, `proc.py` stays
   stdlib-only, and an `argv()` helper on the ffmpeg owner is the only link.
5. [Build] The guard is red from owner creation until the last batch -- that is
   a red intermediate state on `v2.0-alpha` for days if batches are separate
   pushes. Either the guard ships LAST (with the final batch) or it ships with
   an explicit, shrinking offender count asserted per batch (a ratchet with a
   number, updated each commit). The panel should pick one.

SHOULD-FIX:
1. [D] `psutil` -- confirm it is in ComfyUI core's `requirements.txt`, not just
   present in this venv, before `gpu_residency` depends on it.
2. [E] The six `wan_shared` strings are log/exception text; rewording them is
   free but must keep the frame-count fact in the message.
3. [C] The RSS fetcher's `socket.socket` is a deliberate bounded connect; if it
   stays a named exception, the guard for network needs an allowlist WITH the
   reason -- the pattern `test_output_root_single_owner.py` already uses.

CHECKED-CLEAN by the driver: the scan reads two root files (`__init__.py`,
`prestartup_script.py`) and nothing under `scripts/`, `tests/`, `docs/`; the
`credential-access` tag is rule-driven (it lands on `OTR_TEST_MODE`); psutil
7.2.2 imports in the venv; `grep -rl 'os.environ\|getenv'` returns 104 files,
matching the scanner's 103 + `prestartup_script.py`.

UNVERIFIABLE until built: the exact residual count (the scanner is private; the
grep proxies are the only local measure; the next published version's scan is
the receipt).
