# r2 judgment -- registry-findings-collapse (driver: Claude / Fable 5.1)

Roster, as it happened: Antigravity CLI `Gemini 3.8 Flash (High)` returned a
real review (16 file reads in its log, every claim with a file and line);
Codex CLI QUOTA-HELD again (retry window 11:03 PDT; `codex.md` is the hold
notice, discarded) -- its seat is the operator's manual paste of
`docs/2026-09-04-registry-findings-collapse/MANUAL_PROMPT_r2.md`; a Sonnet 5
subagent fills a third seat. Driver anchor written before the fan-out
(`docs/2026-09-04-registry-findings-collapse/driver_anchor_r2.md`); the argv
receipt for the allowlist question was measured before the fan-out too
(`argv0_receipt.txt`: 35 execution sites, matching the scanner's 35).

## Antigravity (r2), grounded

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | the allowlist omits `nvidia-smi` (`_otr_sys_specs.py:110`) | CONFIRMED (the driver's own argv receipt lists it) | allowlist by basename, `.exe` stripped: ffmpeg, ffprobe, python* (sidecar interpreters), git, nvidia-smi, blender |
| MF2 | at least 8 test files monkeypatch `module.subprocess.run/Popen`; a module that drops `import subprocess` breaks them | CONFIRMED and WIDER: twelve files. Ten patch `<module>.subprocess.<run/Popen>` via monkeypatch at 25 sites (credits_roll_spec, ffprobe_boundary x5, nvenc_probe_cache x7, soak_title_provenance x4, upscale_held_frame_reuse, video_scope_draw_encoder x2, video_wrapper_bridge, w45_campaign_bank_pinning x2, wave1_boot, wire_w4b_segment_audio_slice); two more use `mock.patch.object(M.subprocess, "run")` (post_upscale_procgen_blend) and a DOTTED-PATH `mock.patch("nodes._otr_video_engines.render_driver.subprocess.run")` (video_render_driver_perbeat_audio), which resolve `<module>.subprocess` by attribute and raise `AttributeError` the moment the module drops the import | rule for every batch: the owner calls `subprocess.run` / `subprocess.Popen` through the MODULE attribute at call time (never `from subprocess import run`), so a test that patches the `subprocess` module's attribute is seen by every caller; each batch migrates its tests to `monkeypatch.setattr(subprocess, "run", spy)` on the real module. Modules may keep `import subprocess` for `PIPE` / `DEVNULL` / the exception classes -- the guard flags CALLS -- but `proc.py` re-exports those names so a module can also drop the import cleanly |
| MF3 | `tests/test_terminal_frame.py:356-372` keys its spawner sweep on `import subprocess` aliases and `from subprocess import run/Popen`; `:495` asserts `scope_draw.py::encode_silent_mp4` is seen | CONFIRMED (read both ranges) | first batch (`_otr_shared`, which holds scope_draw and wrapper_bridge): teach the sweep `from ._otr_shared import proc` / `from ._otr_shared.proc import run, popen` and classify `proc.run` / `proc.popen` as spawn calls, in the same commit |
| MF4 | the plan's "check, capture, timeout, stdin, cwd, env" is not the real surface: sites pass `stdout=DEVNULL`, `stderr=<file object>` (`video_engine.py:1082-1087`, `encode_sink.py:183`), `text`, `encoding`, `bufsize` (`eng_chatterbox.py:108-111`); "capture" is not a `subprocess` kwarg | CONFIRMED | `proc.run(argv, **kwargs)` and `proc.popen(argv, **kwargs)` forward everything; refuse `shell=True` and a `str` argv with a named error; return the real `CompletedProcess` / `Popen` |
| MF5 | `eng_mesh_stage.py:809` passes `os.environ` as a mapping to `build_blender_env`; `route_freeze.py:72` reads `os.environ` as a mapping | CONFIRMED (both lines read) | owner gains `snapshot() -> dict[str, str]` (a copy of `os.environ`); both sites take the copy (read-only consumers) |
| MF6 | `unpin` must be `pop(name, None)`; `OTR_LedgerScriptWriter.py:1754-1757` restores a baseline that can be `None` | CONFIRMED (the site already branches: `None` -> `pop(name, None)`, else assign) | `unpin(name) -> str \| None` is `os.environ.pop(name, None)`; `pin(name, value)` REJECTS `None` with a `TypeError` naming the knob rather than silently unpinning -- a silent unpin hides a caller bug; the writer's existing branch migrates as written |
| MF7 | a numeric ratchet allows offender swapping (fix one file, add a read in another, count unchanged) | CONFIRMED as design | the ratchet is an explicit `_PENDING_MIGRATION` set in the guard test, asserted in BOTH directions (no offender outside the set; every file inside it still offends); each batch shrinks the set; supersedes the driver anchor's "count N" |
| SF1 | drop `check_output` from the owner by migrating `production_ledger.py:222` to a `proc.run(..., check=True)` call | CONFIRMED equivalent, with one correction: the site runs in BYTES and decodes with `errors="ignore"` (`:226`), so the migration is `proc.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, check=True).stdout.decode("utf-8", errors="ignore")` -- no `text=True`, which would change the decode policy; `check=True` raises `CalledProcessError` exactly as `check_output` does | B yields 35 -> 2 |
| SF2 | `tests/test_master_mux_terminal_knob.py:94` matches `"environ" in ast.dump(fn)`; after migration it passes vacuously | CONFIRMED (read `:92-96`) | the predicate also matches `<env-owner>.get(...)` in the batch that migrates the mux |
| SF3 | aliases and subscripts: `__init__.py` aliases `os` as `_otr_ro`; `eng_humo.py:423` and `eng_mesh_stage.py:508` read `os.environ[...]`; `scratch_check_server.py` is a gitignored root file | CONFIRMED (alias at `__init__.py:510`, not 560; the subscript reads as cited; `.gitignore:315`) | the guard resolves `import os as X` aliases and `from os import environ/getenv`, flags `Subscript` on `environ`, and scans ONLY the two root files by explicit path (the anchor already said so; the scratch file is excluded by construction) |
| SF4 | the batch list omits `_otr_image_engines/` and `_otr_google_api/`; "top-level nodes + roots" is one oversized batch | CONFIRMED (six subpackages on disk) | batches: (a) `_otr_shared` + singletons + the six strings; (b) audio, image, upscale, video, google_api engines -- one commit each; (c) `nodes/_otr_*.py` internal libraries; (d) `nodes/otr_*.py` + `OTR_*.py` + the two roots |
| SF5 | interpreter basenames vary (`python3`, `python3.12`) | ACCEPTED | basename lower-cased, `.exe` stripped, `startswith("python")` |
| OPT1 | the `.pth` sha256 change may not clear the file rule (private regex) | verify-at-build | already listed as UNVERIFIABLE |
| OPT2 | defer `argv()` on the ffmpeg owner | ACCEPTED | not this arc |
| CUT1 | the urllib helper: at most one finding, possibly zero if the scanner keys on `import urllib.request` / `Request`; the five sites are already documented in the 2026-09-02 review request | ACCEPTED (the request doc names them) | item C is CUT; the five network sites become NAMED exceptions with reasons in the network guard; this reverses r1's "minimal helper" |

## Driver anchor, disposition

MF1 (owner signatures) STANDS, with `snapshot()` added (MF5). MF2 (AST
walk, aliases, the two roots by path) STANDS, sharpened by SF3. MF3 (the
exception contract unchanged, real `Popen` returned) STANDS. MF4 (calls, not
attributes) STANDS. MF5 (ratchet first) STANDS, but the ratchet is a SET,
not a count (Antigravity MF7 is the better mechanism). MF6 (Windows
liveness logs once) STANDS. SHOULD-FIX 1 (basename allowlist) STANDS with
the measured set; SHOULD-FIX 2 (the urllib helper's signature) is MOOT --
C is cut.

## Sonnet r2 seat (subagent), grounded

Re-derived every count from the payload (sum 158, per-rule exact). Reviewed
the PBUG-06 diff as landed (`22baa861`): no redirect widening, no cycle, no
reader changes behaviour.

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | guards must ship NOW as named-allowlist ratchets; the repo's two live ratchets (`test_output_root_single_owner.py`, `test_node_temp_hygiene.py`) are named sets, not counters | CONFIRMED (`test_node_temp_hygiene.py:53 _ALLOWLIST`, "A NEW offender fails here") | same as Antigravity MF7: the ratchet is a set, shipped in the owners' commit |
| MF2 | NO runtime executable allowlist: it is "a schema" where A promised "a spelling", and blender resolves to a per-box absolute path (`eng_mesh_stage.py:539-546`) so basename matching is fragile | REJECTED, with the reason: A's rule is about not moving any KNOB's meaning; B's allowlist is about what may EXECUTE -- a boundary, not a schema of env. And a per-box absolute path is exactly what basename matching normalizes (`C:\...\blender.exe` -> `blender`; the sidecar `py = self._venv_python()` -> `python`). Two lanes and the driver say yes | YES stands. Sonnet's residual risk is kept: the refusal is a NAMED error, the list is a one-line reviewed addition, and a unit test feeds every argv[0] basename from the receipt through the check |
| SF3 | "psutil is in ComfyUI core `requirements.txt:20`" is a bad citation: OTR's line 20 is about beautifulsoup4 and `psutil` is not in OTR's file | MISREAD, and the driver briefly agreed before measuring: Sonnet opened OTR's own `requirements.txt`; the citation was to ComfyUI CORE's file, and the booted install's `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\requirements.txt:20` reads exactly `psutil` (grep receipt in the r3 anchor). The driver's first grounding attempt looked under a path that does not exist and wrongly withdrew the line number | citation kept and made unambiguous: "ComfyUI core `requirements.txt:20` (the booted install), venv 7.2.2" |
| SF4 | D under-specified: replace the whole `if os.name == "nt":` block (`:77-90`) with warn-once + `return True`, which makes the currently UNCONDITIONAL `os.kill(pid, 0)` at `:91-97` POSIX-only | CONFIRMED (read `:70-98`) | that exact shape is written into D |
| OPT5 | `unpin` = `pop(name, None)` | already in | |
| PBUG-06 | the `CURRENT_SCHEMA_VERSION` doc string (`_otr_ledger.py:56-64`) now sits under `SHOW_PREFIX` | CONFIRMED (the constant was inserted between the version constant and its own doc string) | moved below the doc string in a follow-up commit; zero functional effect |

## Codex r2 (operator's manual paste), grounded

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | preserve the exception contract and the real `Popen` (`otr_post_upscale_procgen_blend.py:1002 except subprocess.CalledProcessError`; `eng_chatterbox.py:108`) | CONFIRMED (both lines read) | already in (anchor MF3/MF4); the guard's non-finding set is spelled out |
| MF2 | `client.py` has TWO `urlopen` sites (`:187`, `:235`), so the urllib count is three sites, not two | CONFIRMED by grep -- and it answers an open question: the payload carries ONE network finding for `client.py` (line 187) despite two sites, so `python_network_operations` fires PER FILE | C is cut anyway; the network guard's named exceptions are five FILES (six sites), each with its reason |
| MF3 | ratchet as a monotonic COUNT | SUPERSEDED (Codex marked it [ASSUMPTION]; two lanes and the repo's precedents say a named set) | set |
| MF4 | allowlist YES on `basename.casefold().removesuffix(".exe")`, eight names including `py` and `pythonw` | ACCEPTED minus `py`: no site runs the Windows `py` launcher (the sidecars resolve `_venv_python()`); `startswith("python")` covers `python`, `pythonw`, `python3.12` | the set in the final is the MEASURED set |
| SF1 | signatures; `__init__.py:51 setdefault`, `:107 os.environ["OTR_OUTPUT_DIR"] =` | CONFIRMED (both lines read) | already in |
| SF2 | warn once on the Windows no-psutil path | already in (anchor MF6) | |
| SF3 | A-0 as a script under `scripts/` | already in | |
| OPT | a one-line contract comment beside the reworded `wan_shared` strings | ACCEPTED | |
| CUT | typed parsers; requests/socket unification | already out | |
| VERIFY 2 | "all 35 sites pass through `proc.py` and run with their binaries" | turned into two receipts: the unit test above (every receipt basename passes the allowlist) and a canonical leg that PUBLISHES to `otr/obs/` after the last batch -- the operator's success signal, not a log line |
| PBUG-06 | clean on all three questions | agrees with Sonnet | |

## Roster, stated exactly

r2: Antigravity (CLI, Gemini 3.8 Flash High), Sonnet 5 (subagent seat),
Codex (operator's manual paste of `MANUAL_PROMPT_r2.md`; the CLI lane was
quota-held). Cursor did not review r2. Three seats, none of them the driver.
