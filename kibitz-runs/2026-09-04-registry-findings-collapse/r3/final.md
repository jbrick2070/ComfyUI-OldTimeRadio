# Collapse the registry scan by the one-owner rule -- wiring and sequencing, hardened by r3

**Provenance:** r1 by Antigravity (3.8 Flash) + a Sonnet seat + Cursor
(manual); r2 by Antigravity (3.8 Flash) + a Sonnet seat + Codex (manual
paste); r3 by a Sonnet seat and Antigravity on Gemini 3.1 Pro (High), each
grounded by an adversarial verifier per claim and by the driver. The r3
Antigravity 3.8 Flash lane read the r1 document by launcher error and is
kept as a receipt, its tree-level observations folded where true; Codex had
no seat in r3 (standard tier out of credits until 2026-09-07; Spark
overflowed its context). Every claim below was grounded against the real
Windows files.

**Measured, not assumed** (alpha.17's real payload, 158 findings, all
`info`): the env rule fires ONCE PER FILE (103); the subprocess rule PER SITE
(35 across 20 files -- the argv receipt in
`docs/2026-09-04-registry-findings-collapse/argv0_receipt.txt` resolves all
35); six of the twelve "url command" hits are the words `ffprobe
-count_frames` in error strings; the network rule fires PER FILE
(`_otr_google_api/client.py` carries two `urlopen` sites and one finding);
the process, file and bytecode rules are one finding each at one site. The
gate is ZERO findings or a manual admin approval; nothing here reaches zero
(ffmpeg is a subprocess and that is the render path). This plan changes what
the human reviewer reads: about nine lines instead of 158, and the
`credential-access` tag on one file instead of eleven.

## The invariant

*A machine fact has ONE owner, and a test proves the copies agree.* Same
shape as `nodes/_otr_shared/ffmpeg.py` + `tests/test_ffmpeg_single_resolution.py`;
the guards are named-allowlist ratchets of the `tests/test_output_root_single_owner.py`
shape, which already asserts BOTH directions (`:65-79` every allowlisted file
still needs its exception; `:82-89` no offender outside it).

## The collapses (final scope after r3)

**A. One env owner -- 103 -> 1, plus 1 by decision.** `nodes/_otr_shared/env.py`,
stdlib only, importing nothing from the pack, reading the live `os.environ`
on EVERY call (no cache). Surface, stated exactly:

* `get(name: str, default: str | None = None) -> str | None` -- returns
  `os.environ.get(name, default)` unchanged; a site's `or`, cast and default
  stay at the site. Spelling only.
* `pin(name: str, value: str) -> None` -- `os.environ[name] = value`; a
  `None` value raises `TypeError` naming the knob.
* `setdefault(name: str, value: str) -> str`.
* `unpin(name: str) -> str | None` -- `os.environ.pop(name, None)`.
* `snapshot() -> dict[str, str]` -- a COPY, for the two read-only mapping
  consumers (`eng_mesh_stage.py:809`, which copies again at `:215`;
  `route_freeze.py:72`, `.get` only).

The twelve mutation sites migrate to the write verbs; the writer's
style-grammar restore keeps its branch. The two tool owners migrate in the
first batch and keep the `OTR_FFMPEG` constant and `which()` inside
themselves. `prestartup_script.py` KEEPS its inline writes by decision (it
runs with no package context) and stays one finding; from its flat context,
`_otr_kokoro_voice_prefetch.py` can still say `from _otr_shared import env`
because `nodes/` is on `sys.path` there, and both test files that import
that module (flat at `test_kokoro_voice_prefetch.py:36`, packaged at
`test_kokoro_backends.py:32`) cover its five reads, so a broken migrated
import fails the suite at collection.

**A's guard** is an AST walk over `nodes/**/*.py` plus the two root files by
explicit path (`__init__.py`, `prestartup_script.py`), resolving `import os
as X` aliases (`__init__.py:510` aliases `os` as `_otr_ro`) and `from os
import environ / getenv`, flagging `environ` when it RESOLVES TO `os` -- an
attribute on an `os` alias, or a name imported from `os` -- in ANY context
(Load or Store: `eng_humo.py:423` and `eng_mesh_stage.py:508` read by
subscript), plus `.get` / `.pop` / `.setdefault` on it, `getenv`, `putenv`,
`unsetenv`. A bare local named `environ` is NOT a finding:
`motion_common.py:69-70` binds `environ = os.environ if env is None else env`
to a caller-supplied mapping, and a name-only rule would flag it falsely.
Comments, docstrings and string literals do not count. Allowlist: `env.py`
and `prestartup_script.py`, each with a reason.

**A-0. The default-drift list is a RECEIPT, not a fix.** A read-only script
under `scripts/` prints `name -> {default: [sites]}` for every knob read at
more than one site with a different default or cast; its output is pasted
into the plan. NOTHING on it is changed in this arc -- the r2 wording
"fixed in its own commit" contradicted the "not this arc" list and is
withdrawn. Every drift item is the follow-on typed-getters arc's input.

**B. One process owner -- 35 -> 2.** `nodes/_otr_shared/proc.py`, stdlib
only, exactly two execution sites: `run(argv, **kwargs)` and
`popen(argv, **kwargs)`. Both forward EVERY keyword to `subprocess`; both
refuse `shell=True` and a `str` argv with a named error; `run` returns the
real `CompletedProcess`, `popen` the real `Popen`; neither wraps exceptions.
The owner re-exports `PIPE`, `DEVNULL`, `STDOUT`, `CompletedProcess`,
`CalledProcessError`, `TimeoutExpired` as IDENTITY aliases (tests construct
`subprocess.CompletedProcess` and raise `subprocess.TimeoutExpired`; an
alias that is the same object keeps `except` clauses true). `check_output`
is not in the owner: `production_ledger.py:222-226` becomes
`otr_proc.run(argv, stdout=PIPE, stderr=DEVNULL, timeout=5,
check=True).stdout.decode("utf-8", errors="ignore").strip()` -- bytes mode
kept, and the trailing `.strip()` kept (the r2 wording dropped it; a
newline in `_GIT_HEAD_CACHE` would corrupt every name built from it --
Gemini 3.1 Pro, r3). `run` refuses an EMPTY argv with the named error, so
the allowlist's `argv[0]` never raises `IndexError` first.

**Owner import spelling, everywhere.** The owners are imported by the
three-form recipe `ffmpeg.py:28-34` already uses (`from . import` /
`from _otr_shared import` / bare), and ALWAYS under the aliases `otr_env`
and `otr_proc`: `env` is a parameter name in `route_freeze.py:64`
(`routing_env_snapshot(env=None)`) and `motion_common.py:69`, and `proc` is
a local in eleven files (`video_engine.py:1082 proc = subprocess.Popen(`),
so a bare `env` / `proc` module name would be shadowed and raise
`UnboundLocalError` at the first migrated site. Module form only
(`otr_proc.run(...)`), never `from ...proc import run`, so a test can patch
`<module>.otr_proc.run` the way it patches `<module>.subprocess.run` today.

**B's executable allowlist -- YES, and spelled so the ffmpeg guard does not
bite.** `proc.py` checks `argv[0]` by basename, lower-cased, `.exe`
stripped, against a NAMED set with a reason each: `ffmpeg`, `ffprobe`,
`python*` (the three sidecar venv interpreters; `startswith("python")`),
`git`, `nvidia-smi`, `blender`. Anything else raises a named error BEFORE
spawning. The set is declared as a DICT literal keyed by basename with the
reason as the value (`ast.Dict`) -- the allowlist-with-reasons shape -- and
never as a list or tuple, and never as `frozenset([...])` either:
`tests/test_ffmpeg_single_resolution.py:48-51` flags any `ast.List` /
`ast.Tuple` whose first element is bare `"ffmpeg"` outside the two tool
owners, and a list inside a `frozenset(...)` call is still a list literal.
Versioned basenames (`ffmpeg-7.1`, `blender-4.2`) are not in the receipt and
are not admitted; if one ever appears, the named error says so at the
spawn, which is the point. A unit test feeds every argv[0] basename in the
receipt (Windows and Linux-shaped absolute paths) through the check and
asserts an unlisted one raises.

**B's guard** is an AST rule on CALLS whose callee resolves to
`subprocess.run | Popen | check_output | check_call | call` or `os.system |
popen` outside the owner; `PIPE`, `CompletedProcess` annotations and
`except CalledProcessError` are not findings.

**C. Network -- CUT.** The five FILES (six sites) stay as NAMED exceptions
with reasons in a network guard keyed by file.

**D. The three singletons -> 0, split by subpackage.** `gpu_residency.py`
(batch a): `psutil.pid_exists` first (`psutil` is line 20 of ComfyUI core's
`requirements.txt` in the booted install; 7.2.2 in the venv); replace the
whole `if os.name == "nt":` block at `:77-90` with a WARNING emitted ONCE
PER PROCESS through a module-level flag ("liveness unknown without psutil")
and `return True` -- `_pid_alive` is polled every 0.25 s by `acquire()`'s
loop at `:178-196`, so a plain log line would flood four times a second for
up to 120 s (Gemini 3.1 Pro, r3) -- which makes the unconditional
`os.kill(pid, 0)` at `:91-97` POSIX-only. Test: Windows + psutil import
failure (`monkeypatch.setitem(sys.modules, "psutil", None)`) with the flag
reset first (`monkeypatch.setattr(gr, "_WARNED_PSUTIL", False)`, or the
assertion is order-dependent) -> True, `os.kill` never called, exactly one
WARNING across many calls. The trade this makes, named: today the ctypes
block returns False for a dead pid on a psutil-less Windows box, so a stale
lease is reclaimed (`tests/test_video_platform_aseam.py:794-801` relies on
that through psutil); after D a psutil-less Windows box never reclaims and
times out loudly instead. Accepted, because psutil is a ComfyUI core
requirement and a box without it is already broken in other ways.
`eng_spandrel_esrgan.py:259` (batch b-upscale): chunked sha256.
`eng_ltx25.py:287` (batch b-video): `import sys`.

**E. The six strings** (batch b-video): `wan_shared.py:202-232` say "the
frame-count probe", with one comment saying why; no test pins the old
wording (the three tests that mention `-count_frames` assert the PROBE ARGS,
which E does not touch).

## Sequencing (the r3 result)

1. **A-0's drift receipt** (read-only).
2. **Owners + guards + the sweep, ONE commit (the ratchet commit), touching
   no engine file.** `env.py`, `proc.py`, both guards as named-set ratchets
   sharing one `tests/fixtures/` helper (asserts in both directions; each
   guard owns its own set and predicate), the network guard with its five
   named files (no pending set; they stay), AND `tests/test_terminal_frame.py`
   taught the owner in the same commit: `_SPAWN_CALLS` at `:239` is
   `("Popen", "run", "call", "check_call", "check_output")` -- capitalized --
   so the lowercase `popen` is added, and the sweep's alias set (`:358-365`
   today admits only `import subprocess` forms) learns every import that
   binds `otr_proc`: `from . import proc as otr_proc` inside `_otr_shared`
   (ast: `module=None, level=1`), `from .._otr_shared import proc as
   otr_proc` in engines (`level=2`), `from ._otr_shared import proc as
   otr_proc` in top-level nodes (`module="_otr_shared", level=1` -- the ast
   never carries the leading dot in `module`), and the flat
   `from _otr_shared import proc as otr_proc` (`level=0`). The test's `:495`
   assertion on `scope_draw.py::encode_silent_mp4` is the receipt.
   **Test seams migrate PER BATCH, in the module's own commit:** each
   `monkeypatch.setattr(M.subprocess, "run", spy)` becomes
   `monkeypatch.setattr(M.otr_proc, "run", spy)` (the two `mock.patch`
   forms likewise), which mirrors today's seam exactly and reaches whichever
   `proc` module object `M` imported (packaged or flat) -- ten in-scope
   files; `test_soak_title_provenance.py` and `test_w45_campaign_bank_pinning.py`
   patch `scripts/` modules that do not migrate and are left alone. A test
   that keeps patching `M.subprocess` after `M` drops the import fails with
   `AttributeError` in the same commit, which is the self-enforcing check.
   The owners hold no state beyond constants, because packaged and flat
   `proc` are two module objects.
3. **Batches, one commit each, full suite green per commit, the set edited
   in the same commit:** (a) `_otr_shared` including the two tool owners and
   `gpu_residency.py`; (b) engine subpackages, one commit each -- audio,
   image, upscale (with the spandrel sha256), video (with the `eng_ltx25.py`
   import and the six strings), google_api; (c) the internal libraries
   `nodes/_otr_*.py`; (d) EVERY remaining top-level `nodes/*.py` -- the
   `otr_*.py` and `OTR_*.py` nodes AND the eleven unprefixed modules
   (`video_engine.py`, `production_ledger.py`, `scene_sequencer.py`,
   `cast_lock.py`, `announcer_voice.py`, `audio_enhance.py`,
   `batch_character_voices.py`, `news_interpreter.py` and the rest; a
   prefix glob would leave them in no batch, and the guards rglob all of
   `nodes/` regardless)
   -- and the root `__init__.py`, whose `env` import goes ABOVE line 51 and
   OUTSIDE the swallowing try/excepts at `:77-83` and `:97-110` -- a
   stdlib-only import cannot fail, and if it ever did the boot must say so
   rather than skip the `OTR_OUTPUT_DIR` pin with a debug line.
4. **Every migrated module uses the three-form aliased import** (section
   "Owner import spelling"), because tests import many modules flat and the
   bare names collide. `tests/test_master_mux_terminal_knob.py:90-96`
   extends its predicate in the same commit as the mux: it resolves the
   owner's bound name from the mux's OWN imports (an `ImportFrom` of
   `_otr_shared` at level 0 or 1 whose names include `env`, honouring
   `asname`) rather than hard-coding `otr_env`, and KEEPS the existing
   `os.environ` / `getenv` branch, so neither a re-alias nor a regression
   to the old spelling can dodge it.
   `__init__.py:560` (`_otr_ro.environ.get`) and `:437` migrate in (d) like
   any other alias site; the walker resolves every `import os as X`
   generically (six such sites, including `_otr_freeze_cascade.py:911`).
5. **Peers:** `ListAgents` before each batch; message any live window the
   batch's file list. `nodes/` and `tests/` are this box's surface; until
   the pending sets are empty, the 4060 adds no new `os.environ` or
   `subprocess` site under `nodes/` (a new one fails the ratchet on its own
   next pull, loudly).
6. **Acceptance, in this order:** both guards green with empty pending
   sets (the AST guards ARE the proxy; the `grep -rl 'os\.environ\|getenv'`
   line is dropped -- it misses the six `import os as X` aliases); then a
   canonical leg (`workflows/otr_canonical.json`, one act) that PUBLISHES
   to `otr/obs/` with the boot log showing no "Kokoro voice prefetch
   unavailable" line; then, and only then, the operator bumps
   `pyproject.toml` and the next published scan is the registry receipt.

## Phase 0, before any owner: orphans are ripped completely or wired back
(operator directive 2026-09-04, mid-arc: "if orphaned and not deleted we
should rip out 100%; if orphaned and we need them, wire them back in")

The 2026-09-04 dead-code audit (GO_FORWARD_PLAN 1.4a rows E and F,
driver-verified by `git grep -w`) is the list, and it is executed FIRST so
the migration touches fewer sites:
* **Rip (row E, twelve symbols with one repo-wide reference -- their own
  definition):** `MAX_HEADLINE_CLEAN_CHARS` (`_otr_scifi_p0_contract.py`),
  `PLATE_DIRNAME` (`eng_ghost_signal_stillin_lab.py`), `RADIO_HOST_FACE_NEG`
  (`otr_meta_brief_image_prompt.py`), `CRT_RED` / `CRT_WHITE` /
  `CRT_MAGENTA` (`video_engine.py`), `HARNESS_VERSION` / `MACHINE_CEILING_MB`
  / `NVML_FLOOR_MB` (`scripts/_otr_b_spikes/_b_harness.py`),
  `ffprobe_timebase` (`scripts/audit_otr_full_run.py`), `corpus_ledgers`
  (`scripts/otr_clean_stage_lab.py`), `_negative_phrases`
  (`scripts/otr_style_traceroute.py`, whose logic is duplicated inline in
  the function below it -- the helper goes, the inline copy stays).
* **Rip (row F, vestigial):** the unused public `REPAIR_TEMPERATURE` export
  in `_otr_structured_call.py` (every consumer uses the underscore name;
  the underscore name becomes the only one); the re-import of
  `_stamp_durable` at `otr_shot_lock.py:3396`; the unreachable
  `elif engine.startswith("humo")` in `scripts/otr_asset_index.py`; the
  never-read `_profiles` parameter of `load_classes` in
  `scripts/otr_machine_matrix.py`; `_otr_paths.episodes_for_obs_dir()`
  (zero callers, "kept for back-compat" with nothing).
* **OPEN, and it is the operator's call -- `_otr_shared/content_oracle.py`
  has ZERO production importers.** The driver first recorded it as alive on
  two "importers" that are COMMENT mentions of `content_oracle.MOTION_FAMILIES`
  (`eng_minimax_h3.py:1094`, `render_driver.py:1957`), not imports; the r4
  Sonnet seat caught the misread and `git grep -nE "^[^#]*\b(import|from)\b.*content_oracle"`
  confirms it: the only importers are eight test sites. It is a shipped
  module (under `nodes/`, so it goes in the zip; `tests/` does not) whose
  verdict reaches no gate. It is NOT in rows E or F and Phase 0 does not
  touch it. Three defensible answers, and the choice is not the driver's:
  (a) RIP it -- the C5 soak it was built for never called it, and
  `scripts/otr_w45_campaign.py:217` already carries its own `MOTION_FAMILIES`
  plus a wired held-frame invariant; (b) WIRE it -- its luma floor and
  freeze check are exactly the black-frame and frozen-clip defects the pack
  keeps chasing, at the cost of an ffmpeg pass per clip on the render path;
  (c) MOVE it to `tests/` or `scripts/`, where its only consumers live, so
  it stops shipping. The duplicated `MOTION_FAMILIES` is a one-owner defect
  either way and is logged for the follow-on.
* **Receipt per rip:** `git grep -n -w -F <name>` returns nothing after the
  commit; AST parse of every touched file; the full suite green. Anything
  on the list that turns out to have a live caller at build time is WIRED,
  not deleted, and the plan row says which.

## Revised floor, conditional on the next scan

env 1 + 1 (prestartup) + proc 2 + network 5 (files) + singletons 0 +
strings 0 = **about 9 findings**, all `info`, from 158. Not Active; one
screen, and the `credential-access` tag on one file.

## Not this arc

Active without the manual review; any env name, default, precedence or
cast (A-0's list included); command construction; `argv()` on the ffmpeg
owner; a urllib helper; `scripts/`; the knob catalog; typed getters; a
second `git rev-parse` owner (`_otr_ledger.py:884` and
`production_ledger.py:222` both look it up -- a one-owner follow-on, noted,
not taken); `pyproject.toml`.

## Risks r4 should keep trying to break

The frozenset allowlist refusing a legitimate binary a branch never ran here
(the receipt is this box's); whether the scanner counts `proc.py`'s two
sites as two findings or also matches the re-exported names; the twelve
rewritten tests on the 4060's pull; the ratchet sets across two boxes'
commits (edited in the same commit, failing loudly on a stale pull -- the
intended behaviour, but a surprise to the box that pulls).
