# Session Handoff -- OTR Audio + Voice-Casting Overhaul -- 2026-06-02

## Core goal
Build the model-agnostic, per-role audio engine registry + voice-casting subsystem for the
OldTimeRadio ComfyUI pipeline on `v2.0-alpha`. Character voice / announcer / music each selectable
per role; deterministic voice bank + caster; a post-freeze cast-lock node; a frozen
`ResolvedVoiceRequest` identity/cache contract. Legacy path stays a permanent **byte-identical**
fallback. **R0a is complete; Wave 0 is in progress (2 of 8 pieces done).**

## Canonical spec (read first, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` is the single source of truth -- invariants
I-1..I-11, ComfyUI first-run invariants C-1..C-7, the wave/sprint build order, the `ResolvedVoiceRequest v1`
field list, the per-sprint test names, re-baseline triggers, and the verify-at-build list. `CLAUDE.md`
(prime directives, git flow, testing) and `ROADMAP.md` / `BUG_LOG.md` auto-load -- not repeated here.

## HOW TO RUN TESTS + GIT -- everything is OUTSIDE the sandbox (read this before doing anything)
The Linux sandbox (`mcp__workspace__bash`) has **no torch** and does **not** mount the venv or the Bug
Bible repo. Use it ONLY for read-only repo exploration (grep/ls on the mount). **All tests and all git
run on the Windows host via Desktop Commander (DC).**

- **venv python (full path required; system `python` is not on PATH):**
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`
- **Full regression -- the gate (~32s). Run after EVERY change. Via DC `cmd` shell:**
  `cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
  Current baseline: **3525 passed, 12 skipped, 0 failed.** To save context: let it finish, then read ONLY
  the tail (`read_process_output` with `offset: -3`) for the `==== N passed ====` line -- do not re-read the
  whole dump.
- **conftest KNOWN-FAIL-GUARD:** ANY new failing test is flagged a regression against a 0-fail baseline.
  Fix it; do not register it.
- **Targeted run** (fast feedback while iterating): `... -m pytest tests/test_X.py tests/test_Y.py -q -p no:cacheprovider`.
- **Bug Bible repo is NOT locatable** at the CLAUDE.md path
  (`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\`). `Documents\ComfyUI` reads as
  OneDrive-virtualized (`dir` of it returns empty). `tests/bug_bible_regression.py` could not be found at
  Documents / OneDrive\Documents / home. **Gate on the in-repo `tests/` suite instead.** Ask Jeffrey to
  confirm the bible repo location before relying on it (and before any Three-File-Contract bible push).
- **Commit flow (DC `cmd` only -- never PowerShell for git):** write `.git\COMMIT_EDITMSG` with the file
  tool, then `git add <explicit paths>` (**NEVER `-A`** -- ~20 untracked planning docs + `custom_nodes.lnk`
  must stay unstaged), then `git commit -F .git\COMMIT_EDITMSG`. Verify with `git log -1` + `git status
  --porcelain --untracked-files=no` (should be empty). The `CRLF will be replaced by LF` warning on
  `*.json` / `config/*` commits is the repo's existing eol policy -- harmless.
- **Workflow-JSON edits** (`workflows/*.json` are CRLF + litegraph schema): do a surgical, CRLF-preserving
  byte-replace via a one-shot py script run through DC that asserts `json.loads` parses before writing. Do
  NOT `json.load`/`json.dump` the whole file (it reflows every node). Delete the temp script after.
- File creation/edits: the Claude file tools (Read/Write/Edit) write the real Windows FS that DC + git see.
  (Sandbox bash git misses file-tool edits -- that's why git goes through DC.)

## What's done & decided this session (committed on v2.0-alpha)
- **R0a (a-e) COMPLETE** -- `3ebf9f4` (a-c) + `892c280` (d-e):
  - `_seed_to_int64(*parts)` is **type-tagged** (`i:`/`s:`/`b:`/`y:`/`j:`) so `1`, `"1"`, `True` never
    collide; returns a non-negative 63-bit int (torch/random take it directly; numpy callers mask `& 0xFFFFFFFF`).
  - `ResolvedVoiceRequest` frozen dataclass: IN_KEY/IGNORED partition, `cache_key = sha256(canonical_json(IN_KEY))`.
  - `deterministic_inference(seed)` CM is SCOPED (process default stays non-strict, C-2/C-3); launchers export
    the determinism env BEFORE python (C-1); TF32 default OFF at `_otr_model_loader.py:243-244`.
  - **Node 62 (`OTR_LedgerFreezeCascade`): `episode_seed`(INT)+`v2_ledger_json`(STRING) APPENDED at out
    indices 5,6 -- never inserted.** out[1] `script_json` (fans to 13 consumers) stays byte-identical;
    `episode_seed` is derived read-only from the frozen ledger (NEVER stamped back). Wired into
    `workflows/otr_scifi_16gb_full.json` (5->7 output sockets, new slots empty links).
  - **Legacy seeding** in Bark/Kokoro/MusicGen/AudioGen: `seed_all_rngs(_seed_to_int64("<engine>_legacy_v1",
    script_json))` placed right after `load_ledger(script_json)` (no parse). Makes render-twice reproducible
    (I-2); the operator GPU baseline (step f) locks the bytes in.
  - `config/legacy_invocation_manifest.json` is GENERATED from each node's INPUT_TYPES via
    `nodes/_otr_legacy_manifest.build_manifest()`; the drift test re-derives + compares (a `legacy_manifest_sha`
    re-baseline trigger guard).
- **Wave 0 pieces 1-2 COMPLETE** -- `d0decf1`:
  - `build_resolved_request(...)` + integer-tick `quantize_params(...)` (floats -> `round(v*1000)`, bool->0/1,
    non-numeric->stable 31-bit tick) in `nodes/_otr_resolved_request.py`. Builder derives `prepared_text_sha256`,
    quantized params, and `stable_line_seed` (`stable_line_seed_v1` reduction per G1).
  - `prepare_text` + `PREPARE_TEXT_VERSION` in `nodes/_otr_script_prep.py` (layers on shipped
    `clean_spoken_text`: strips `*`/`♪`/delivery-tags, ellipsis->one `...` pause, keeps `. , ?`).
- Don't reopen the locked design decisions in the EXECUTION-PLAN ("Decided this session" / "Rejected" /
  engine defaults IndexTTS2 + Stable Audio 3 / commercial warn-not-block).

## State of the art (current files; read the existing surface before building the next piece)
- `nodes/_otr_resolved_request.py`, `nodes/_otr_determinism.py`, `nodes/_otr_script_prep.py`,
  `nodes/_otr_legacy_manifest.py` + `config/legacy_invocation_manifest.json` -- as above.
- Engine adapters ALREADY EXIST under `nodes/_otr_audio_engines/` (`eng_bark.py`, `eng_chatterbox.py`,
  `eng_indextts2.py`, `eng_kokoro.py`, `eng_musicgen.py`, `eng_stable_audio.py`) + an `__init__.py`, shipped
  in Sprint A/C. **Read `nodes/_otr_audio_engines/__init__.py` + `tests/test_audio_engine_registry.py` +
  `tests/test_audio_engine_adapters.py` before building Wave 0 pieces 3/4** -- a registry/base may partly
  exist; build with/beside it, don't duplicate.
- `nodes/OTR_LedgerFreezeCascade.py` now returns a 7-tuple; `_episode_seed_from_ledger` helper added.

## Immediate next steps (resume at Wave 0 piece 3)
Per piece: read the existing surface -> build -> wire JSON if it's a node -> full `tests/` via DC -> commit
explicit paths. Plan SSOT table has file->imports->tests.
3. `nodes/_otr_audio_engines/registry.py` -- `assert_usable(engine, role)` + 6-class enum
   (`gated_by_flag|missing_model|missing_hf_token|incompatible_profile|noncommercial_blocked|malformed_config`),
   FAIL CLOSED.
4. `nodes/_otr_audio_engines/base.py` -- adapter base + `pack_audio_batch` + `supports_external_generator`.
5. `nodes/_otr_audio_cache.py` -- cache PROTOCOL only (interface; impl is Wave 1f).
6. `nodes/_otr_class_registry.py` -- mapping keys + display names + categories for the NEW nodes.
7. `config/*_schema.json` -- bank-entry + cache-sidecar schemas.
8. `config/audio_engine_profiles.yaml` + `nodes/_otr_engine_profiles.py` -- 0d resolver, lazy pydantic load
   (cached, exception-wrapped), INPUT_TYPES legacy-first, NO module-scope IO (C-5).
Then Wave 1 (8 files: `batch_character_voices`, `announcer_voice`, `stable_audio_theme`, `_otr_voice_bank`+caster,
`_otr_delivery_profiles`+`_otr_release_gate`, `_otr_audio_cache` impl, `batch_humo_render` S0.1), Wave 2a
(`cast_lock.py` + writer refactor), Wave 2b (opt-in `workflows/otr_scifi_16gb_audio_v2_optin.json` + the two
explicit link tables, drop node 15).

## Open questions / deferred
- Bug Bible repo path -- flag to Jeffrey (see test section above).
- **Deferred operator/GPU (need Jeffrey + the RTX 5080; batch to the very end):** R0a (f) render-twice
  bit-identity baseline, R0b box-fresh smoke, Wave 3 F/G1/I. Directive in force: **no live/GPU testing until
  ALL headless sprints are built; the pytest regression stays in every commit gate** ("regress" in the loop is
  the automated suite, not the GPU render).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when you're ready
to start."
