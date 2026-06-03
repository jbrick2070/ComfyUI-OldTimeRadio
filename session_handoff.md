# Session Handoff -- OTR Audio + Voice-Casting Overhaul -- 2026-06-02 (Wave 0 complete)

## Core goal
Build the model-agnostic, per-role audio engine registry + voice-casting subsystem for the
OldTimeRadio ComfyUI pipeline on `v2.0-alpha`. Character voice / announcer / music each selectable
per role; deterministic voice bank + caster; a post-freeze cast-lock node; a frozen
`ResolvedVoiceRequest` identity/cache contract. Legacy path stays a permanent **byte-identical**
fallback. **R0a complete; Wave 0 COMPLETE (8/8); resume at Wave 1 piece 1a.**

## Canonical spec (read first, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` is the single source of truth -- invariants
I-1..I-11, ComfyUI first-run invariants C-1..C-7, wave/sprint build order, `ResolvedVoiceRequest v1`
fields, per-sprint test names, re-baseline triggers, verify-at-build list. `CLAUDE.md` + `ROADMAP.md` /
`BUG_LOG.md` auto-load.

## HOW TO RUN TESTS + GIT -- everything is OUTSIDE the sandbox (read before doing anything)
The Linux sandbox (`mcp__workspace__bash`) has **no torch** and does NOT mount the venv or the Bug
Bible repo. Use it ONLY for read-only repo exploration (grep/ls on the mount). **All tests + git run
on the Windows host via Desktop Commander (DC) `cmd` shell.**

- **venv python (full path; system `python` is not on PATH):**
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`
- **Full regression -- the gate. Run after EVERY change. DC `cmd`:**
  `cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
  **Current baseline: 3569 passed, 12 skipped, 0 failed (~34s).** Context-saver: redirect to a temp
  file and read only the tail, e.g. append `> _otr_pytest_tail.txt 2>&1` then DC `read_file` offset
  `-3`; **`del _otr_pytest_tail.txt` before committing** (it is untracked -- never stage it).
- **conftest KNOWN-FAIL-GUARD:** `EXPECTED_FAILED_NODEIDS` is empty; ANY new failing test hard-exits
  `SystemExit(2)`. Updating an existing test to a new contract is fine **as long as it passes green**.
- **Targeted run** (fast iterate): `... -m pytest tests/test_X.py tests/test_Y.py -q -p no:cacheprovider`.
- **`python -c "..."` is UNUSABLE through DC cmd** -- inner quotes / `&`-chaining get mangled
  ("unterminated string literal"). For any Python probe, **write a tiny `.py` file and run
  `python file.py`**, then `del` it. (Reinforces CLAUDE.md's no-heredoc rule.)
- **Commit flow (DC `cmd` only -- never PowerShell for git):** write `.git\COMMIT_EDITMSG` with the
  file tool, then `git add <explicit paths>` (**NEVER `-A`** -- ~20 untracked planning docs +
  `custom_nodes.lnk` must stay unstaged), then `git commit -F .git\COMMIT_EDITMSG`. Verify:
  `git log -1 --format=%H%n%s` + `git status --porcelain --untracked-files=no` (empty). `CRLF will be
  replaced by LF` on `config/*` / `*.json` is the repo's existing eol policy -- harmless.
- **Bug Bible repo still NOT locatable** at the CLAUDE.md path -- gate on the in-repo `tests/` suite;
  ask Jeffrey before any Three-File-Contract bible push.
- **Workflow-JSON edits** (`workflows/*.json` CRLF + litegraph): surgical CRLF-preserving byte-replace
  via a one-shot py script that asserts `json.loads` parses before writing; never `json.load/dump` the
  whole file. (Not needed yet -- first workflow edit is Wave 2b.)
- File creation/edits: Claude file tools (Read/Write/Edit) write the real Windows FS DC + git see.
- **Env confirmed this session:** `pydantic 2.12.5` (v2 API) + `PyYAML 6.0.3` are installed in the venv.

## Shipped this session -- Wave 0 pieces 3-8 (committed + PUSHED on v2.0-alpha)
Wave 0 CODE ends at `34bbbd2`; current branch HEAD = `073e4ca` (this handoff doc) sitting on top.
Both pushed (`git rev-parse HEAD` == `origin/v2.0-alpha`; verified 41 files, no BOM, no 0-byte). The
next session should `git rev-parse HEAD` first to confirm nothing landed after `073e4ca`. Commits:
- `dd0f86a` **#3 fail-closed registry** -- `nodes/_otr_audio_engines/registry.py`: `assert_usable` now
  FAILS CLOSED (C-6, **never silent-swap**). New `EngineUsabilityReason` enum (the 6 codes
  `gated_by_flag|missing_model|missing_hf_token|incompatible_profile|noncommercial_blocked|malformed_config`)
  + `EngineUnusable(engine, role, reason, detail)`. `assert_usable` raises GATED_BY_FLAG (opt-in flag
  off), MALFORMED_CONFIG (unknown engine), INCOMPATIBLE_PROFILE (role mismatch); returns the validated
  name on success. **The byte-identical safety property now comes from the shipped workflow defaulting
  its engine widget to the legacy engine until promotion (I), NOT from a dispatch substitution.** The
  two old silent-swap tests were rewritten to the fail-closed contract. Downstream (profile resolver,
  release gate) reuse the same enum.
- `5235055` **#4 base** -- `nodes/_otr_audio_engines/base.py`: `AudioEngineAdapter` (OPTIONAL base; the
  registry duck-types, legacy adapters don't inherit) + `pack_audio_batch(items, *, sample_rate, mono)`
  (the existing Bark `[B,1,T]` / empty `[1,1,0]` contract, mono-safe + right-pad, built on
  `canonical_audio`/`mono_safe`/`empty_audio_batch`) + `engine_supports_external_generator` (default
  False until the F pilot verifies an external `torch.Generator` -- G1).
- `a5349cc` **#5 cache protocol** -- `nodes/_otr_audio_cache.py` (interface only; impl is 1f):
  `cache_key_for(request)` == `ResolvedVoiceRequest.cache_key` (the single I-6 keying rule),
  `AudioCacheRecord` frozen sidecar (mirrors the JSON schema), `record_from_request(...)`,
  `AudioCache` runtime_checkable Protocol (key_for/has/get/put/iter_records).
- `863fb13` **#6 class registry** -- `nodes/_otr_class_registry.py`: `NewNodeSpec` rows for
  `OTR_BatchCharacterVoices` / `OTR_AnnouncerVoice` / `OTR_StableAudioTheme` / `OTR_CastLock`, all
  CATEGORY `OldTimeRadio/v2/audio`, **bare class names** (BatchBarkGenerator-style),
  `new_node_modules_table()` -> the `{key:(module,class,display)}` shape the top `__init__._NODE_MODULES`
  consumes. **NOT yet wired into the top `__init__`** -- wire per node as each module lands so box-fresh
  load stays clean (no phantom "Skipped", banner stays N/N).
- `9efc292` **#7 schemas** -- `config/voice_bank_entry_schema.json` (E.1 fields) +
  `config/audio_cache_sidecar_schema.json` (mirrors `AudioCacheRecord`, drift-guarded by test).
- `34bbbd2` **#8 profiles + resolver** -- `config/audio_engine_profiles.yaml` (7 profiles) +
  `nodes/_otr_engine_profiles.py`: `EngineProfileResolver v1`, lazy pydantic-v2 load cached by content
  sha (`source_sha256` = re-baseline trigger), `resolve_casting_plan(role, engine, voice_bank)`
  fail-closed ladder (reuses the registry enum; ref engines reject `bark_legacy` via
  `allowed_voice_banks`), `legacy_first_engines(role)` HARDCODED for INPUT_TYPES (C-5 never-empty),
  `assert_token_for_profile`/`assert_model_available` (generate-path only, never INPUT_TYPES).
  - **profile_ids:** char_bark_v1, char_chatterbox_v1, char_indextts2_v1, announcer_kokoro_v1,
    announcer_chatterbox_v1, music_musicgen_v1, music_stable_audio_v1.
  - **voice-bank ids:** `bark_legacy` (bark presets), `kokoro_builtin` (kokoro), `default` (chatterbox +
    indextts2 reference clips). Music profiles carry no bank. `music_stable_audio_v1.requires_hf_token=true`.

## Decisions locked this session (do not re-litigate)
- assert_usable fail-closed (above) supersedes the Sprint A/C silent-swap. This was the one real
  contract change; everything else is additive.
- New nodes' class names are bare; CATEGORY is `OldTimeRadio/v2/audio`; mapping keys carry the `OTR_`
  prefix. Source of truth = `_otr_class_registry.py`.
- Engine matrix license/`commercial_clean` values were NOT churned (kept as the shipped adapters set):
  bark False, kokoro True, musicgen False, chatterbox True, indextts2 False, stable_audio_music True.

## Deferred -- needs Jeffrey + the RTX 5080 (batch to the very end)
- **I-11 `_resample_audio` `.cuda()` at `nodes/scene_sequencer.py:126` is NOT changed.** It is on the
  LEGACY byte path; flipping it to CPU before the R0a-f baseline is captured could silently shift the
  not-yet-captured baseline. Do the CPU flip **deliberately, then capture the baseline on the CPU
  path.** All NEW v2-node post-engine DSP must be authored on `.cpu()` tensors per I-11.
- R0a (f) render-twice legacy bit-identity baseline; R0b box-fresh live smoke; Wave 3 F/G1/I.
- Directive in force: **no live/GPU testing until ALL headless sprints are built; the pytest
  regression stays in every commit gate.**

## Immediate next steps -- Wave 1 (resume at piece 1a)
Per piece: read the existing surface -> build -> wire JSON if a node -> full `tests/` via DC -> commit
explicit paths. Plan D / D5 / E / S0.1 / G0 have the field lists + test names.
- **1a `nodes/batch_character_voices.py`, 1b `nodes/announcer_voice.py`, 1c `nodes/stable_audio_theme.py`.**
  Gate sockets + dispatch; **lazy-import engine libs INSIDE `generate()`**; INPUT_TYPES legacy-first via
  `legacy_first_engines(role)` + exception-safe + never-empty (C-5); forceInput keys (`script_json`,
  `ledger_json`, `gate_in`) carry NO `widget`; output adds `done` (STRING); `gate_in->done`; teardown
  in `finally` BEFORE `done` (I-7); **batch path = raw delegation** to the legacy node (I-3, byte-
  identical), **per-line path** = `build_resolved_request` -> `prepare_text` -> `generate_voice/clip` ->
  `pack_audio_batch` (C-4). Wire `new_node_modules_table()` into the top `__init__._NODE_MODULES` as
  these land; each class `CATEGORY == expected_category(key)`.
- 1d `_otr_voice_bank.py` validator + `assign_voice_for_slot` caster + `config/voice_reference_bank.json`
  (codes to `voice_bank_entry_schema.json`; E.2 scoring gender100/timbre40/role20/age10 -> stable sort
  -> one seeded `rng.choice` from `stable_cast_seed`).
- 1e `_otr_delivery_profiles.py` (neutral-only) + `_otr_release_gate.py` `assert_release_clean`
  (codes to `audio_cache_sidecar_schema.json`; fail-closed on missing `commercial_clean`, reuse enum).
- 1f `_otr_audio_cache.py` IMPL against the piece-5 protocol (read/write keyed on `cache_key_for` +
  slim migration: `detect_ledger_schema_version`, reject old ledger missing `voice_ref_id`,
  `request_schema_version != target -> re-render`).
- 1g S0.1 HuMo 16k-mono ANALYSIS-clone contract -- **existing-file edit to `nodes/batch_humo_render.py`**
  (own worktree if parallel); do NOT pin master mono.
Then **Wave 2a** (`nodes/cast_lock.py` `OTR_CastLock` + move cast/stamp OUT of the writer -- the writer
is heavily tested, refactor carefully) and **Wave 2b** (build `workflows/otr_scifi_16gb_audio_v2_optin.json`
via the headless litegraph builder + the TWO explicit link tables + drop node 15 + link-integrity tests;
R0b live smoke is operator).

## Surfaces already read this session (save a re-read)
- `nodes/_otr_audio_engines/` adapters: `interface` is `"batch"` (bark/kokoro/musicgen -> legacy default,
  `make_batch_node()`), `"per_line"` (chatterbox/indextts2 -> `generate_voice(text, ref_clip_path,
  delivery_vector, seed)`), `"clip"` (stable_audio_music -> `generate_clip(prompt, duration_s, seed)`).
- `nodes/_otr_resolved_request.py`: `build_resolved_request(...)`, `quantize_params`, `_seed_to_int64`,
  `empty_audio_batch`, `assert_audio_batch_contract`. `nodes/_otr_script_prep.py`: `prepare_text` +
  `PREPARE_TEXT_VERSION`. `nodes/_otr_audio_utils.py`: `canonical_audio`/`mono_safe`/`audio_sha16`.
- `nodes/kokoro_announcer.py` = the announcer-node analog: INPUT_TYPES(required `script_json` STRING,
  optional widgets), `RETURN_TYPES=("AUDIO","STRING","STRING")`, `FUNCTION="render"`, reads the ledger via
  `_otr_ledger_consumers.load_ledger(script_json)` + `iter_lines(led, roles={"announcer"})`, seeds RNGs
  from the frozen `script_json` (`seed_all_rngs(_seed_to_int64("<engine>_legacy_v1", script_json))`),
  builds `[B,1,T]`, tears down (`gc.collect()` + `empty_cache()`).
- `nodes/batch_bark_generator.py`: empty batch `{waveform: zeros(1,1,2400), sr:24000}`; pads to longest
  in batch. `nodes/scene_sequencer.py` ~L518-538 unbinds `waveform[b]` per line (dim 0 = lines). `.cuda()`
  resample at L126 (the I-11 deferral above).
- Registration: top-level `__init__.py` `_NODE_MODULES = {"OTR_<Name>": (module, class, display)}` loop
  (isolated try/except per node; banner counts loaded==total). Display names lead with a space; node-file
  local `NODE_DISPLAY_NAME_MAPPINGS` is NOT what the package exports (the `_NODE_MODULES` value is).
  `tests/test_naming_conventions.py` only constrains `_otr_*_lib.py` files + package display names (no
  `[EMOJI]/[TODO]/[PLACEHOLDER]/[FIXME]`). STILL TO READ for 1a-c: a gate node
  (`visual/flux_branch_gate.py`) for the `gate_in` STRING forceInput -> passthrough pattern, and
  `tests/test_workflow_audio_widget_vectors.py` for the widget-vector contract.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and continue the audio overhaul at Wave 1 piece 1a. Acknowledge when ready."
