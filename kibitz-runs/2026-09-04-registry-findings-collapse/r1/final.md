# Collapse the registry scan by the one-owner rule -- plan, hardened by r1

**Measured, not assumed** (alpha.17's real payload, 158 findings, all `info`;
today's tree is 104 env-reading files because the ffmpeg owner arrived after
the scan): the env rule fires ONCE PER FILE (103 findings, 103 files, verified
against a 23-site file that still produced one); the subprocess rule fires PER
SITE (35 findings, 20 files, four in one file); six of the twelve "url command"
hits are the words `ffprobe -count_frames` inside error strings; the network,
process, file and bytecode rules are one finding each at five, one, one and
one sites. The gate is ZERO findings or a manual admin approval; nothing here
reaches zero (ffmpeg is a subprocess and that is the render path). This plan
changes what the human reviewer reads: about nine lines instead of 158, and
the `credential-access` tag on one file instead of eleven.

## The invariant

*A machine fact has ONE owner, and a test proves the copies agree.* Same shape
as `nodes/_otr_shared/ffmpeg.py` + `tests/test_ffmpeg_single_resolution.py`.

## The collapses (final scope after r1)

**A. One env owner -- 103 -> 1, plus 1 by decision.**
`nodes/_otr_shared/env.py`, stdlib only, importing nothing from the pack, and
reading the live `os.environ` on EVERY call (no cache: `conftest` pops names at
import, hundreds of tests `monkeypatch.setenv`, the launchers pin at boot).
Surface: `get(name, default=None) -> str | None` exactly like
`os.environ.get`; `pin(name, value)`; `setdefault(name, value)`;
`unpin(name)` (the `pop`). No typed getters, no catalog, no credential names
inside it -- it is a spelling, not a schema. The migration is SPELLING-ONLY:
a site's default and cast do not move. The twelve mutation sites migrate to
the three write verbs (`OTR_OUTPUT_DIR` pin in the root `__init__`, the root
`__init__`'s `HF_HUB_DISABLE_TELEMETRY` setdefault, the style-grammar flag's
set/pop/restore trio, `OTR_ACTIVE_PROFILE`, `OTR_SNAPSHOT_HASH`, `HF_HOME` and
`HF_HUB_CACHE` in `_otr_hf_env`, `HF_TOKEN` and the `HUGGING_FACE_HUB_TOKEN`
setdefault in `hf_token`). The two tool owners (`ffmpeg.py:71`,
`ffprobe.py:172`) migrate in the first batch; `env.py` is a leaf, so no cycle.
`prestartup_script.py` KEEPS its five inline writes by decision (it must run
before transformers and stay decoupled from node loading) and stays one
finding. Guard: an AST test that outside the owner (and the prestartup file,
named with its reason) no file under `nodes/` or at the root contains
`os.environ`, `os.getenv`, `os.putenv`, `os.unsetenv`, `environ.pop`,
`environ.setdefault` or `environ[...] =`.

**A-0. The default-drift list, BEFORE the rename.** A grep that lists every
knob read at more than one site with a different default or cast. That list
is a design-review input with its own receipts -- the ffmpeg precedent earned
its keep by exactly this drift (nine of ten sites ignored the pin), and a
mechanical rename must not paper over the same class. The rename touches
spelling only; anything on the list is fixed in its own commit, with a test.
Follow-on, logged, not this arc: three named `_bool_env` functions plus dozens
of inline `== "1"` checks -- typed getters are the honest fix once the rename
has shipped and stayed green.

**B. One process owner -- 35 -> 3.** `nodes/_otr_shared/proc.py`, stdlib
only, owning `subprocess.run`, `subprocess.Popen` and `subprocess.check_output`
(`production_ledger.py:222` uses the third) -- one call site each, so three
findings by the per-site rule. It executes an argv list a caller built;
contracts pass through (check, capture, timeout, stdin, cwd, env). It does
NOT build commands. `popen()` returns the standard `Popen` (the three
sidecar-venv launches keep their pipes; no second owner for streams). **THE
r2 QUESTION:** does `proc.py` carry an executable allowlist -- ffmpeg,
ffprobe, git, the sidecar interpreters, blender -- so a human reviewer reads
a boundary rather than an evasion shim? The driver's lean is yes: it is the
one-owner rule applied to what may run, and it strengthens the review request.
Guard: an AST rule on CALLS whose callee is `subprocess.run` / `Popen` /
`check_output` / `check_call` / `call` (and `os.system`, `os.popen`) outside
the owner -- NOT "any `subprocess.` attribute": callers legitimately pass
`subprocess.PIPE` and annotate with `subprocess.CompletedProcess` (Cursor
r1). The env owner, for its part, must never spell the constant `OTR_FFMPEG`
or call `which()`: the ffmpeg guard treats either as a second resolver
(Cursor r1); a generic `get(name)` cannot trip it. The ffmpeg owner may gain
`argv(*args) -> [resolved_binary, *args]`; whether that clears the six
`cmd = [` findings is UNKNOWN (the pattern may match the assignment shape) --
the next scan says, the plan does not promise.

**C. Network -- 5 -> 4, and it is not worth more.** A minimal helper for the
two `urllib.request.urlopen` sites only (`_otr_google_api/client.py:187`,
`cloud_media_invoke.py:571`). The two `requests` backends stay as NAMED
exceptions; the RSS fetcher's socket is SSRF hardening (resolve, reject
private ranges, connect to the validated address, TLS-wrap pinned to it --
DNS-rebinding-safe) and stays as a named exception with that reason; the
helper must never become an alternate path for feed URLs. Guard: an
allowlist WITH reasons, the `test_output_root_single_owner.py` shape.

**D. The three singletons -> 0, one of them carefully.** `gpu_residency.py`
already tries `psutil.pid_exists` first (`:73-74`; `psutil` is a ComfyUI core
requirement, `requirements.txt:20`); the ctypes `OpenProcess` block (`:77-90`)
is the Windows FAIL-SAFE behind it, and the path behind THAT is
`os.kill(pid, 0)` -- which on Windows is not a probe: CPython's Windows
`os.kill` calls `TerminateProcess` for any signal other than the two console
events, so with psutil absent and the ctypes block simply deleted, "is the
lease holder alive?" would KILL the lease holder (Cursor r1 caught the
safety change; the driver read the fallback and found it worse than stated).
The honest removal: on Windows without psutil, return True -- uncertain, do
not steal a possibly-live lock, the block's own documented policy -- and keep
`os.kill(pid, 0)` for POSIX only, where it is a real probe. `OpenProcess` goes,
safety does not. Its test: Windows + psutil absent (monkeypatched import
failure) -> True without touching `os.kill`.
`eng_spandrel_esrgan.py:259`: chunked `open(path, "rb")` sha256 -- which also
removes a whole-file RAM spike on multi-hundred-MB checkpoints.
`eng_ltx25.py:287`: `import sys`.

**E. The six strings.** `wan_shared.py:202-232`: say "the frame-count probe";
no test asserts the old wording (checked).

## Not this arc, said so it stays out

Active without the manual review; any env name, default, precedence or cast;
command construction; `scripts/` (unscanned, unshipped); the knob catalog;
typed getters (follow-on, fed by A-0's list); unifying `requests` and the
socket; `pyproject.toml` (a publish is the operator's eyeball).

## Build shape (r2/r3 harden this)

1. A-0's drift list first (read-only, a receipt).
2. Owners (`env.py`, `proc.py`, the urllib helper), each stdlib-only, each
   with its guard test. **THE OTHER r2 QUESTION:** the guard ships LAST with
   the final batch, or ships now as a RATCHET asserting a shrinking offender
   count per batch (a number, updated each commit, red only if it grows).
3. Migrate BY SUBSYSTEM, one commit each, full suite green per commit:
   `_otr_shared` (including the two tool owners) -> audio engines ->
   video and upscale engines -> top-level `nodes/*.py` and the roots.
   Singletons and the six strings ride in the first commit.
4. Local proxy for acceptance: `grep -rl 'os\.environ\|getenv' nodes/
   __init__.py prestartup_script.py` lists exactly the owner and the prestartup
   file; `grep -rl 'subprocess\.' nodes/` exactly `proc.py`. The next
   PUBLISHED version's scan is the receipt, and the only number the review
   request quotes.

## Risks the panel should keep trying to break

Import cycles (owners import nothing from the pack); the per-call read; the
red-guard interval across batches; the `credential-access` tag concentrating
on `env.py` (a reviewer will ask why `HF_TOKEN` is read at all -- gated model
downloads, opt-in -- the answer belongs in the request); the unknown regex
behind `$media_list_assign_indirect`; whether `python_network_operations`
dedupes per file or per site (five findings, one per file today, consistent
with either).
