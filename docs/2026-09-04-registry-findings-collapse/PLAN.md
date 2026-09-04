# Collapse the registry scan: 158 findings to about five, by the one-owner rule

**Status:** plan for a kibitz arc (r1). Nothing built. Written 2026-09-04 from
alpha.17's REAL scan (`api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true`,
158 findings, all `severity: info`), classified by rule, pattern and file -- not
from the summary numbers in the review-request draft.

## What the scan actually counts (measured, not assumed)

| rule | findings | files | how it counts |
|---|---:|---:|---|
| `python_environment_manipulation` | 103 | 103 | **ONCE PER FILE** -- 103 findings, 103 files, exactly one each; patterns `$env_read1/2/3` (97 reads), `$env_mod1/4` (6 writes), `$proc_env1` (1) |
| `python_command_injection_risk` | 35 | 20 | per SITE -- `$subprocess_run_direct` (26), `$subprocess_popen_direct` (9); one file carries up to 6 |
| `python_url_command_execution` | 12 | 3 | 6 x `$media_list_assign_indirect` on `cmd = [` in `eng_google_lyria.py:128` and `foley_stems.py:499`; 6 x `$media_cmd_string_presence` on the WORDS `ffprobe -count_frames` inside error strings in `wan_shared.py:202-232` |
| `python_network_operations` | 5 | 5 | `requests.post/get` (2), `urllib.request.urlopen` (2), `socket.socket` (1, the RSS fetcher) |
| `windows_process_manipulation` | 1 | 1 | `kernel32.OpenProcess(SYNCHRONIZE, ...)` in `gpu_residency.py:82` (a liveness check on a lease-holding PID) |
| `python_sensitive_file_access` | 1 | 1 | `Path(model_path).read_bytes()` for a sha256 in `eng_spandrel_esrgan.py:259` |
| `python_bytecode_manipulation` | 1 | 1 | `__import__("sys").modules.get(...)` in `eng_ltx25.py:287` |

Admin tags ride on the findings: `system-modification` 103 (every env file),
`any-code-execute` 36, `any-folder-access` 33, `credential-access` 11 (env reads
whose NAME the scanner considers credential-ish -- `HF_TOKEN` twice, but also
`OTR_WRITER_HEARTBEAT_EVERY` and `OTR_TEST_MODE`, so it is the rule that tags,
not the value), `any-network-requests` 5, `obfuscated-code` 1.

Two files outside `nodes/` are scanned: `__init__.py` (the `OTR_OUTPUT_DIR`
pin) and `prestartup_script.py` (`HF_HOME`). `scripts/`, `tests/` and `docs/`
are not in the zip or not scanned.

**The gate, restated so nobody plans around a threshold that does not exist:**
a version goes Active on a ZERO-finding scan or a manual admin approval (peers
confirmed the pattern on other packs: `comfyui-amdmonitor` Flagged on 5 info,
`easyuse-anima` on 3). Nothing below reaches zero -- ffmpeg is a subprocess and
that is the render path -- so the manual review request remains the only path
to Active. What this plan changes is what the human reviewer reads: five lines
instead of a hundred and fifty-eight, and no `credential-access` tags.

## The invariant, applied once more

*A machine fact has ONE owner, and a test proves the copies agree.* The
2026-09-04 chunk did it for ffmpeg resolution, the output root, session
residency and the models root, with AST guards that make the claim a test.
Every rule above is the same shape: a decision spread across a hundred files
that has exactly one honest home.

## The five collapses, in yield order

**A. One env reader -- 103 -> 1 (plus 1 for `prestartup_script.py`, which runs
before the package exists and cannot import it).** `nodes/_otr_shared/env.py`
(stdlib only, imports nothing from the pack) is the only module that touches
`os.environ`. Every `os.environ.get("OTR_X", default)` / `os.getenv` in the
other 102 files becomes `otr_env.get("OTR_X", default)`; the six writes
(`__init__`'s output pin, the style-grammar flag, `OTR_ACTIVE_PROFILE`,
`HF_HOME`, `HF_TOKEN`) become `otr_env.pin("OTR_X", value)`, which is the same
`os.environ[...] =` inside the owner and nothing else. The env NAMES do not
change, so the 4060's launcher, the pod scripts, every profile `launch.env` and
every README knob keep working unchanged -- this is a spelling change, not a
semantics change. Guard: an AST test that no file under `nodes/` (and neither
root file, except the owner) contains `os.environ`, `os.getenv`, `os.putenv`
or `os.unsetenv`. Design questions for the panel: (1) typed getters
(`get_bool`, `get_int`, `get_path`) vs one `get` plus casts at the call site --
the pack has at least four hand-rolled bool parsers today; (2) a declared
KNOB CATALOG (name, type, default, one-line meaning) that the README's knob
table is generated from, vs free names -- the catalog makes the guard stronger
(an undeclared name is a finding) and the docs true by construction, at the
price of touching every read site's meaning once; (3) the test seams:
`conftest` pops `OTR_OBS_DIR`/`OTR_VOICE_REFERENCE_BANK` and tests
`monkeypatch.setenv` freely -- the owner must read the live `os.environ` on
every call (no caching), or every such test silently breaks.

**B. One process runner -- 35 -> about 2, and the 6 argv-list findings with
it.** `nodes/_otr_shared/proc.py` owns `subprocess.run` and `subprocess.Popen`
(the two patterns the rule matches); every caller passes an argv list and gets
the completed process back, contracts unchanged (check, capture, timeout,
stdin, cwd, env pass through). It does NOT construct commands -- that stays at
the call site -- so it is a thin seam, not a framework. If the ffmpeg owner
gains an `argv(*args)` helper that returns `[resolved_binary, *args]`, the two
`cmd = ["ffmpeg", ...]`-shaped assignments in Lyria and foley_stems stop
matching `$media_list_assign_indirect` for free. Design questions: (4) one
`run()` with `**kwargs` vs a small typed surface; (5) whether the four
sidecar-venv launches (TTS engines with their own interpreters, `Popen` with
long-lived pipes) fit the same seam or keep a second owner for streams.

**C. One HTTP client -- 5 -> 1.** `requests` in two backends, `urllib` in two,
a raw `socket` in the RSS fetcher. One `http.py` owner with `get_json` /
`post_json` / `download` and the fetcher's connect-with-timeout helper. Design
question: (6) the RSS fetcher's socket use is deliberate (a bounded connect
before the real fetch); does it belong in the HTTP owner or stay as a named
exception with its own reason?

**D. The three singletons -> 0.** `gpu_residency.py:82` `OpenProcess` becomes
`psutil.pid_exists(pid)` (`psutil` 7.2.2 is in the venv via ComfyUI core;
verify it is a core requirement, not a coincidence, before depending on it);
`eng_spandrel_esrgan.py:259` hashes through a chunked `open(path, "rb")`
instead of `Path(...).read_bytes()`; `eng_ltx25.py:287` spells `import sys`.
Each is a five-line change with an obvious test.

**E. The six error strings.** `wan_shared.py:202-232` says "ffprobe
-count_frames" in six messages; say "the frame-count probe" and the rule stops
matching. Cosmetic, honest, zero risk.

**Projected result:** env 1 (+1 prestartup) + proc 2 + http 1 + 0 + 0 = **about
5 findings**, every one `info`, every tag gone except `any-code-execute` on the
two owner files. From 158.

## What this is NOT

* Not a path to Active without the manual review. Say so in the request.
* Not a semantics change: no env name, default, precedence or knob meaning
  moves. The 4060's numbers do not move (measure: its launcher and profiles
  set the same names; the owner reads them the same way).
* Not a rewrite of command construction (ffmpeg argv stays where it is built).
* Not `scripts/` (unscanned, unshipped) -- they may adopt the owners later.

## Build shape (for r2/r3 to harden)

1. Owners first (`env.py`, `proc.py`, `http.py`), each stdlib-only, each with
   its guard test written RED against the current tree.
2. Migrate in batches of ~20 files, alphabetical, one commit each, full suite
   green per batch; the guard's offender count is the progress bar.
3. Singletons and the six strings in the first batch (cheap, visible).
4. Bump nothing in `pyproject.toml` until the operator says: a version bump is
   a registry publish and a publish is his eyeball.
5. Acceptance: a local proxy for the scanner -- `grep -rl 'os.environ\|getenv'
   nodes/ __init__.py prestartup_script.py` must list exactly the owner and
   `prestartup_script.py`; `grep -rl 'subprocess\.' nodes/` exactly `proc.py`
   (and whatever B question 5 decides); then the next published version's real
   scan is the receipt.

## Risks the panel should try to break

* Import cycles: the owners must import nothing from the pack (the ffmpeg owner
  imports `ffprobe`; `env.py` must not import the ffmpeg owner).
* The per-call read: any owner-side caching breaks `monkeypatch.setenv` in
  hundreds of tests and the launcher's pins.
* `prestartup_script.py` cannot import the package; its one write stays.
* Batching: a half-migrated tree is green (both spellings read the same
  `os.environ`) -- true, but the guard is red until the last batch; the guard's
  count is the honest progress bar, not a failing CI.
* The `credential-access` tag follows the RULE, not the value: with one owner
  the tag lands on the owner file once. A reviewer may still ask why
  `HF_TOKEN` is read at all; the answer (gated model downloads, opt-in) belongs
  in the review request.
