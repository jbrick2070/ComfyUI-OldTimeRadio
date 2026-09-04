# r3 judgment -- registry-findings-collapse (driver: Claude / Fable 5.1)

## Roster, as it actually happened

* **Sonnet 5 seat** (subagent, read the r2 final): a real review, grounded
  below by a five-agent workflow (one extractor, one adversarial verifier per
  claim) and by the driver.
* **Antigravity, Gemini 3.8 Flash (High)**: REVIEWED THE WRONG DOCUMENT. The
  first r3 launcher was derived from the r2 one by `sed`, and its `--doc`
  line kept `r1/final.md`; the review quotes the urllib helper, the
  `subprocess\.` grep proxy and the r1 guard wording, all of which r2 had
  already cut or fixed. Kept on disk as `antigravity_READ_R1_FINAL_flash38.md`;
  its code-level claims are still grounded (a wrong input does not make a
  true observation about the tree false) and folded where they survive.
* **Antigravity, Gemini 3.1 Pro (High)**: re-run with the r2 final named
  explicitly (`agy_model_selected.txt` = `Gemini 3.1 Pro (High)`, `input.md`
  header = "hardened by r2"); the operator pointed at this model, and the
  standing rule counts 3.1 Pro and 3.8 Flash as different reviewers.
* **Codex**: no seat this round in any form. The standard tier is out of
  credits until 2026-09-07 08:34; the Spark model, forced through a patched
  selector (all four kibitz copies, clone commit `648fd06`), overflowed its
  context window mid-review and then hit Spark's own limit (resets 14:47
  today). Kept as `codex_spark_context_overflow.md`.
* **Cursor**: did not review r3.

## Sonnet r3 seat, grounded (workflow verdicts + driver)

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | batch (a) "`_otr_shared` + the three singletons + the six strings" reaches into batch (b)'s subpackages: `eng_spandrel_esrgan.py` is upscale, `eng_ltx25.py` and `wan_shared.py` are video | CONFIRMED (paths on disk); the verifier added that the video batch MUST edit those two files anyway -- `eng_ltx25.py:346,610-622` and `wan_shared.py:430` carry env reads -- so the overlap is real, not nominal | D is SPLIT, not moved: `gpu_residency.py` stays in (a) by its own rule; the spandrel sha256 rides in (b)-upscale; the `eng_ltx25.py` `import sys` and the six `wan_shared.py` strings ride in (b)-video. Each commit's file set matches its stated scope |
| MF2 | the Risks section cites `__init__.py:51` for the `OTR_OUTPUT_DIR` pin; line 51 is the telemetry setdefault, the pin is guarded at `:97` and written at `:107`, and `:80-81` already imports a `_otr_shared` submodule before it | CONFIRMED, with a caveat the verifier found: BOTH the `:80` precedent and the `:97-110` pin block sit inside try/excepts that swallow to `log.debug` | risk closed as answered-yes with the right lines; and the migrated root file imports `env` OUTSIDE any swallowing try -- it is stdlib-only and cannot fail, and if it ever did, boot must say so, not skip the pin silently |
| SF3 | `_otr_kokoro_voice_prefetch.py` has five env reads and is loaded only through `prestartup_script.py`'s flat `sys.path` import under a swallow-all; a broken migrated import would be silent at boot | PARTLY: the reads (`:138,170,174,232,236`) and the swallow-all (`prestartup_script.py:88-98`) are real; but `tests/test_kokoro_voice_prefetch.py:36` imports the module FLAT and `tests/test_kokoro_backends.py:32` PACKAGED, covering all five sites, so a broken import fails the suite at collection | the "add a direct unit test" half is redundant and dropped; the boot-log receipt stays: on the acceptance leg, the "Kokoro voice prefetch unavailable" line must be ABSENT. Noted for the wiring: from prestartup's flat context `from _otr_shared import env` resolves because `nodes/` is on `sys.path` |
| SF4 | the bidirectional `_PENDING_MIGRATION` ratchet has no precedent; write one shared helper | MISREAD on the precedent: `tests/test_output_root_single_owner.py:65-79` asserts every allowlisted file STILL needs its exception and `:82-89` asserts no offender outside it -- both directions, already in the repo | the shared helper is still cheap and taken (one `tests/fixtures/` function both guards call); cross-box drift of the SET is not solved by a helper: it is solved by editing the set in the same commit as the batch and by the peer message before each batch -- a stale set fails loudly on the other box's next pull, which is the point |
| OPT5 | `build_blender_env(os.environ)` must be read-only for `snapshot()` to be safe | CONFIRMED read-only (`eng_mesh_stage.py:215` copies first: `env = dict(base_env)`) | `snapshot()` is correct there |
| V-a-b | `check_output` -> `proc.run` faithful; `route_freeze` works on a dict | agrees with r2 (bytes mode kept; `.get` only) | |

## Antigravity 3.8 Flash (read the r1 final), grounded by a 12-agent workflow

Its document-level findings are against text r2 had already removed; its
tree-level observations were verified anyway.

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | test seams break per batch; patch `proc.run` | PARTLY: the seams are real (and already an obligation in the plan); the proposed target is WRONG -- packaged and flat `proc` are two module objects, and a module binding `run` at import is missed | tests patch the stdlib `subprocess` module (as the plan says); the count is ten in-scope files, two target `scripts/` modules that do not migrate; owners hold no state |
| MF2 | the `subprocess\.` grep proxy contradicts the call-only guard | MISREAD: the r2/r3 document has no such proxy; that sentence was r1's | moot; a call-only grep as an extra proxy would agree file-for-file today (22/22) |
| MF3 | a list-shaped allowlist in `proc.py` trips `test_ffmpeg_single_resolution.py:48-51` | CONFIRMED, with the verifier's sharpening: `frozenset([...])` is STILL a list literal | the allowlist is a dict literal keyed by basename with the reason as value |
| MF4 | A-0 "fixed in its own commit" contradicts "not this arc" | PARTLY (a wording contradiction in the r2 final) | A-0 is a receipt only; every drift item is the follow-on's input |
| MF5 | the guard misses subscript reads; flag `environ` as any name in any context | PARTLY: the subscript reads are real and the r2 final already covers them; the proposed name-rule is a false positive at `motion_common.py:69-70` (a local `environ`) | the guard flags `environ` only when it resolves to `os` |
| MF6 | `unpin` = `pop(name, None)`; the writer's baseline can be `None` | already in the plan | |
| SF1 | drop `check_output` | already in the plan; the verifier adds that `capture_output=True` cannot combine with `stderr=DEVNULL` (ValueError) | the `stdout=PIPE, stderr=DEVNULL` form stands |
| SF2 | partition batch 4 | MISREAD (r2 already partitioned) -- but its globs exposed a real gap: fourteen unprefixed top-level modules were in no batch | batch (d) is "every remaining top-level `nodes/*.py`" |
| SF3 | a directory-scoped, one-direction ratchet | MISREAD: strictly weaker than the two-direction set the plan has | set stands |
| SF4 | a regex allowlist for versioned binaries | PARTLY: the regex DROPS `nvidia-smi` and `pythonw`, both live | measured set stands; versioned names are not admitted until seen |
| SF5 | import `env` above line 51 with the two-form import | CONFIRMED | in the plan |
| CUT1 | cut the urllib helper | already cut in r2 | |

## Antigravity, Gemini 3.1 Pro (High), read the r2 final -- grounded by a 16-agent workflow (shared with the paste below)

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | a per-call WARNING in `_pid_alive` floods: `acquire()` polls it every 0.25 s (`:178-196`) | PARTLY: the poll is real and the r2 text said "once" without a mechanism | module flag (`_WARNED_PSUTIL`, the `cloud_media_invoke.py:277` shape); the test resets the flag and stubs psutil; D's trade on a psutil-less Windows box named in the final |
| MF2 | the `proc.run` rewrite of the git lookup drops `.strip()` | CONFIRMED (live: the stdout is `c5648864\n`); the cited path was wrong (`nodes/production_ledger.py:222-227`) and the consequence is the ledger `commit` field, not filenames; `_otr_ledger.py:893` already strips | `.strip()` restored in the final |
| MF3 | the ast never carries the leading dot: `module="_otr_shared", level=1` | CONFIRMED fact; the proposed predicate (level 1 only) is too narrow for the three-form import | the sweep resolves level 0/1/2, bare, and `asname="otr_proc"` |
| SF1 | empty argv raises `IndexError` before the named error | PARTLY: real; use a named error, never `assert` (stripped under `-O`) | in the final |
| OPT | `__init__.py:560` `_otr_ro.environ.get` must migrate in (d) | CONFIRMED | in the final |
| CUT | the boot-order risk | already closed | |

## Operator's manual paste (lane not stated), grounded by the same workflow

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | batch (d)'s globs miss unprefixed offenders (`video_engine.py:1082`, `production_ledger.py:222`, `scene_sequencer.py:368`, `cast_lock.py:499`) | CONFIRMED: eleven unprefixed top-level modules | (d) is "every remaining path under `nodes/`" |
| MF2 | `_SPAWN_CALLS` (`:239`) is capitalized, so `proc.popen` is invisible; the sweep admits only `subprocess` aliases (`:358-365`) | CONFIRMED | `popen` added; the sweep learns every import binding `otr_proc` at level 0/1/2 and bare, in the ratchet commit |
| MF3 | do not patch stdlib `subprocess.run`; patch the owner; identity re-exports; two of the twelve target `scripts/` | PARTLY: the "leak" premise is a misread (today's `cr.subprocess` IS the stdlib module, so today's patches are already process-wide); the owner-patch disposition is taken anyway because `M.otr_proc.run` mirrors today's seam, reaches whichever module object `M` imported, and lands per batch beside the module change; identity aliases and the two `scripts/` exclusions confirmed | per-batch `M.otr_proc.run` patches |
| MF4 | one file, one batch; (a) = `_otr_shared/**` only | CONFIRMED (already the r3 split) | |
| MF5 | three-form import recipe (`ffmpeg.py:28-34`); `env` and `proc` collide with parameters and locals (`route_freeze.py:64`, `motion_common.py:69`, eleven files with a local `proc`) | CONFIRMED | aliases `otr_env` / `otr_proc`, module form only |
| MF6 | a list/tuple allowlist trips the ffmpeg guard | CONFIRMED (same as 3.8 Flash MF3) | dict literal |
| SF7 | the grep proxy misses `import os as X` (six sites incl. `audio_enhance.py:453`, `otr_video_render_batch.py:595`) | CONFIRMED | grep dropped; the guards are the proxy; alias resolution generic |
| SF8 | line 51 is the first write; import `env` above it | CONFIRMED; tests slice the root file by markers, not line numbers, so the insert is safe | in the final |
| SF9 | teach the sweep in the ratchet commit; ship the network allowlist there; the 4060 adds no new sites | PARTLY (r2 had the sweep in batch (a); r3 moves it to the ratchet commit) | in the final |
| SF10 | the mux knob test's predicate goes vacuous; bind it to the real import | CONFIRMED; refined: resolve the alias from the mux's own imports, keep the `os.environ` branch | in the final |
| CUT 1-4 | stdlib patching; soak/w45; env from prestartup; urllib | all already the final's position | |

## Roster, stated exactly

r3: Sonnet 5 (subagent), Antigravity Gemini 3.1 Pro (High) on the r2 final,
the operator's manual paste (lane not stated), and Antigravity Gemini 3.8
Flash (High) on the WRONG document (r1 final; tree-level claims folded).
Codex: no seat. Cursor: none. Every claim grounded by a per-claim adversarial
verifier (three workflows, 33 agents) and by the driver.
