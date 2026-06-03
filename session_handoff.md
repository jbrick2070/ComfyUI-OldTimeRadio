# Session Handoff -- OTR Audio + Voice-Casting Overhaul -- 2026-06-02 (Wave 1 + Wave 2a-node COMPLETE)

## Core goal
Model-agnostic, per-role audio engine registry + voice-casting subsystem for the
OldTimeRadio ComfyUI pipeline on `v2.0-alpha`. Character voice / announcer / music
each selectable per role; deterministic voice bank + caster; a post-freeze
cast-lock node; a frozen `ResolvedVoiceRequest` identity/cache contract. Legacy
path stays a permanent **byte-identical** fallback.

**STATUS: Wave 0 done. Wave 1 (1a-1g) done. Wave 2a CastLock NODE done. All
headless code is shipped + pushed behind a green gate. The remainder
(writer cast/stamp removal, Wave 2b opt-in workflow JSON, and all GPU gates) is
operator/GPU-gated -- see "Remaining (operator / GPU)".**

## Canonical spec (read first, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` -- single source of truth
(invariants I-1..I-11, ComfyUI C-1..C-7, wave order, ResolvedVoiceRequest fields,
per-sprint tests, re-baseline triggers). `CLAUDE.md` + `ROADMAP.md` + `BUG_LOG.md`
auto-load.

## Git state
- Branch `v2.0-alpha`. **HEAD = `4ea3fc2` == `origin/v2.0-alpha`** (pushed; verify
  with `git rev-parse HEAD` vs `origin/v2.0-alpha` first).
- Wave 0 code ended at `34bbbd2`. This session added, in order:
  - `6b15e7c` 1a OTR_BatchCharacterVoices
  - `dd11a2f` 1b OTR_AnnouncerVoice + shared voice-node base (`_otr_voice_node_common.py`)
  - `8f77b0a` 1c OTR_StableAudioTheme
  - `56ce95b` 1d voice reference bank + caster (`_otr_voice_bank.py` + `config/voice_reference_bank.json`)
  - `83eeb52` 1e delivery profiles + release gate (`_otr_delivery_profiles.py` + `_otr_release_gate.py`)
  - `1d45e78` 1f FileAudioCache impl + slim migration (added to `_otr_audio_cache.py`)
  - `622bbff` 1g HuMo 16k-mono ANALYSIS clone (edit to `batch_humo_render.py`)
  - `4ea3fc2` 2a OTR_CastLock (`nodes/cast_lock.py`)
- **Full `tests/` baseline now: 3679 passed, 12 skipped, 0 failed (~36s).**
  Wave-0 baseline was 3569; this session added 110 passing tests, zero regressions.

## HOW TO RUN TESTS + GIT -- everything is OUTSIDE the sandbox
The Linux sandbox (`mcp__workspace__bash`) has **no torch** and **cannot parse the
CRLF litegraph workflow JSON** (stale-mount; `json.load` raises "Unterminated
string"). Use the sandbox ONLY for read-only grep/ls on `.py`/`.md`. **All tests,
git, and any workflow-JSON parsing run on the Windows host via Desktop Commander
(DC) `cmd`.**
- venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (system
  `python` not on PATH).
- Full regression (the gate, run after EVERY change). DC `cmd`, redirect + read the
  file (interactive capture hits a ~60s MCP timeout; the redirect always completes):
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider > _otr_pytest_tail.txt 2>&1`
  then read `_otr_pytest_tail.txt` with the file tool. **`del` every `_otr_*_tail.txt`
  before committing** (untracked; never stage).
- conftest KNOWN-FAIL-GUARD: `EXPECTED_FAILED_NODEIDS` is empty; ANY new failing test
  hard-exits `SystemExit(2)` (fires on subset runs too).
- **`python -c "..."` is UNUSABLE through DC cmd** (quote mangling). Write a tiny `.py`
  file, run `python file.py`, `del` it.
- Commit flow (DC `cmd` only, never PowerShell for git): write `.git\COMMIT_EDITMSG`
  with the file tool, `git add <explicit paths>` (**NEVER `-A`** -- ~20 untracked
  planning docs + `custom_nodes.lnk` must stay unstaged), `git commit -F
  .git\COMMIT_EDITMSG`, verify `git log -1 --format=%H%n%s` + `git status --porcelain
  --untracked-files=no` empty.
- **ASCII-only `.py` source, no em-dash (`--` not the long dash)** -- a test in every new
  suite enforces it; CLAUDE.md cp1252 subprocess-decode rule.

## What shipped this session (do NOT redo)

### The dispatch contract (1a/1b shared base `nodes/_otr_voice_node_common.py`)
`OTRVoiceNodeBase` + `voice_input_types(role, fallback)` + helpers
(`frozen_batch_widgets`, `coerce_int_seed`, `build_engine_combo`). Every voice node:
- INPUT_TYPES is C-5 safe (no IO, never-empty, exception-wrapped engine combo from
  `legacy_first_engines(role)`); forceInput sockets `script_json`/`ledger_json`/
  `gate_in` carry no widget; serialized widgets are exactly `engine` + `stereo_policy`;
  no `seed`/`model_id` widget.
- Dispatch: `assert_usable(engine, role)` FAIL CLOSED (6-class); `batch` interface ->
  **raw verbatim delegation** of the EXACT `script_json` string + the frozen
  `config/legacy_invocation_manifest.json` widget tuple (byte-identical, I-3); `per_line`
  -> `build_resolved_request` -> `prepare_text` -> adapter -> `pack_audio_batch` (C-4).
- Teardown (`unload`+`gc`+`empty_cache`) in `finally` BEFORE the `done` sentinel (I-7).
  `done` = `"<role>:done:engine=...:clips=N"`; zero workload still emits `done` + `[1,1,0]`.

### Nodes (auto-registered via `__init__` table merge + file-existence guard)
- **`batch_character_voices.py` / `BatchCharacterVoices` / OTR_BatchCharacterVoices**
  -- role `char_voice`; engines bark(legacy default, batch)/chatterbox/indextts2.
- **`announcer_voice.py` / `AnnouncerVoice` / OTR_AnnouncerVoice** -- role
  `announcer_voice`; kokoro(legacy default, batch -> OTR_KokoroAnnouncer)/chatterbox;
  routes only `speaker_role=announcer`.
- **`stable_audio_theme.py` / `StableAudioTheme` / OTR_StableAudioTheme** -- role
  `music`; musicgen(legacy default, batch -> OTR_MusicGenTheme, 3 cue AUDIO outputs)/
  stable_audio_music(clip). Outputs: opening/closing/interstitial_theme_audio +
  render_log + done. Self-contained (3 outputs, clip path) -- NOT on the voice base.
- **`cast_lock.py` / `CastLock` / OTR_CastLock** -- single v2 ledger authority after
  FreezeCascade. In `script_json`(forceInput) + widgets voice_bank /
  cast_voice_policy(preserve_ledger|auto_registry) / delivery_profile / allow_voice_reuse
  + gate_in(forceInput). Out ledger_json / cast_lock_revision(INT) / cast_report / done.
  preserve_ledger (default) re-casts nothing (byte-safe); auto_registry runs the caster,
  stamps voice_ref_id/voice_engine/commercial_clean, announcer takes the per-engine pin.
  No voice_engine_mode/deterministic_inference/model_id widget (E.4).

`__init__.py` merges `new_node_modules_table()` ONLY for keys whose module file exists
(so OTR_CastLock appeared once `cast_lock.py` landed). All 4 v2 keys now live. The merge
is by table, NOT literal `"OTR_..."` keys (the class-registry collision test enforces this).

### Libraries
- **`_otr_voice_bank.py`** -- `load_voice_bank` (validates each entry vs
  `config/voice_bank_entry_schema.json`, dependency-free, rejects dup voice_ref_id, caches
  by content sha); `get_all_registered_voices`; `assign_voice_for_slot` (own seeded RNG,
  I-4; gender100/timbre40/role20/age10; ladder g+t+r+age -> drop age -> drop role ->
  gender-only -> raise unless allow_voice_reuse re-walks permitting reuse, gender floor
  holds); `announcer_voice_ref(engine)` pin; `CASTING_POLICY_VERSION="1"`.
- **`config/voice_reference_bank.json`** -- 8 starter entries (chatterbox + indextts2 char
  refs, kokoro `bm_george` + chatterbox `cc_announcer_male` announcer pins). `ref_sha256`
  is `"pending"` until F pins real clip hashes; `ref_path` is the intended on-disk location.
- **`_otr_delivery_profiles.py`** -- `neutral`-only identity profile;
  DELIVERY_PROFILE_VERSION / DELIVERY_PROJECTION_VERSION; get/apply/available.
- **`_otr_release_gate.py`** -- `assert_release_clean(items, *, strict_commercial,
  require_allowed_for_release)` three-state scan (true=clean / false=warn / missing|null=
  FAIL CLOSED), reuses EngineUsabilityReason (MALFORMED_CONFIG / NONCOMMERCIAL_BLOCKED);
  `ReleaseReport`; `mangle_release_filename` (hashes a gated stem).
- **`_otr_audio_cache.py`** (impl added) -- `FileAudioCache` (sidecar `<key>.json` + buffer
  `<key>.npy`, key = `cache_key_for` = ResolvedVoiceRequest.cache_key, all IO in methods);
  slim migration `needs_rerender` (schema drift -> miss) + read-only
  `assert_registry_ledger_has_voice_ref_id` (cast-locked ledger missing voice_ref_id ->
  CacheMigrationError; legacy/no-cast_lock ledger left alone) + `detect_ledger_schema_version`.
- **`batch_humo_render.py`** (edit) -- `_humo_analysis_audio` = CLONED 16k-mono copy for
  `AudioEncoderEncode` (Whisper) ONLY; `.clone()` first so the shared master `episode_audio`
  is never aliased/mutated (may stay stereo, I-5); CPU-only DSP (I-11). Wired at the Phase-B
  encode call only. `is_never_humo_role` skip was already present (line ~2160).

## Decisions locked this session (do not re-litigate)
- Extracted a shared voice-node base (1a refactored to a thin subclass; public surface
  unchanged so its tests stayed green). The theme node is self-contained (3 outputs).
- Voice nodes take BOTH `script_json` (raw FreezeCascade, for byte-identical bark/kokoro/
  musicgen batch delegation) AND `ledger_json` (CastLock, for the per-line/cast-aware path).
  CastLock rewriting the ledger is therefore byte-safe: the legacy batch path uses the raw
  string, the new per-line path uses CastLock's ledger.
- CastLock auto_registry resolves the character engine from the bank via the engine profiles
  (`voice_bank="default"` -> chatterbox; bark_legacy/kokoro_builtin -> preset engines -> chars
  preserved). Announcer engine defaults to kokoro.

## Remaining (operator / GPU) -- NOT done; gated by design

### Wave 2a tail -- writer cast/stamp removal [SERIAL, operator GPU]
Move the cast/stamp OUT of `OTR_LedgerScriptWriter`, keep only the cheap char_id-subset
validator. **Deferred deliberately:** its only real verification is the R0a render-twice
legacy bit-identity diff vs the (not-yet-captured) baseline -- removing it blind risks
silently shifting the legacy audio. INDEPENDENT of Wave 2b (CastLock stamps regardless;
the writer can keep stamping for the legacy workflow). Do this WITH the R0a baseline in hand.

### Wave 2b -- opt-in workflow JSON + link tests [SERIAL]
**There is NO headless litegraph builder in the repo yet** (only patch/audit scripts:
`scripts/normalize_workflow_widgets.py`, `_audit_workflow_json.py`, `_inspect_workflow.py`,
`_build_videoplan_test_workflow.py`). Wave 2b = build that builder + emit
`workflows/otr_scifi_16gb_audio_v2_optin.json` + link tests, then **R0b live smoke is the
operator gate** (it must load in ComfyUI Desktop -- litegraph socket/widget fidelity + the
`d06560a` graph/widget guards are exactly what R0b catches; do not ship this without a load).

**Exact migration spec (derived from `otr_scifi_16gb_full.json`, 29 nodes, last_node_id=79,
last_link_id=230):**
- Node **62 OTR_LedgerFreezeCascade has 7 outputs** (R0a-c ports already present):
  out[0]=script_text, out[1]=script_json, out[2]=news_used, out[3]=estimated_minutes,
  out[4]=freeze_verdict, **out[5]=episode_seed (INT, unwired)**, **out[6]=v2_ledger_json
  (STRING, unwired)**.
- 62.out[1] currently fans to 13 consumers: links L2->3.in0, L12->11.in0(bark), L16->12.in1,
  L19->13.in0(kokoro), L21->14.in0(musicgen), L24->15.in0(SFX), L47->20.in1, L113->20.in0,
  L114->59.in3, L79->51.in5(HuMo), L82->52.in2, L90->55.in3, L202->71.in0.
- **Partition for the opt-in copy:** raw out[1] STAYS for 3,12,20,52,55,59,71. The 3 NEW
  audio nodes REPLACE legacy 11/13/14 (rebuild instances, not type-swap): their `script_json`
  <- 62.out[1] (raw, for delegation), their `ledger_json` <- CastLock.ledger_json. **HuMo
  51.in[5] <- CastLock.ledger_json** (was raw out[1], L79). **CastLock.script_json <-
  62.out[6] v2_ledger_json** (the dedicated port).
- **Theme slot map BY NAME** (14 MusicGenTheme -> StableAudioTheme): 14.out[0](opening) ->
  7.in[1] (L22) becomes opening_theme_audio -> 7.in[1]; 14.out[1](closing) -> 7.in[2] (L23)
  AND 12.in[3] (L105) becomes closing_theme_audio -> 7.in[2] + 12.in[3]; interstitial unwired.
- **Drop node 15 (OTR_BatchAudioGenGenerator)** + L24 + L25 (15.out0 -> 3.in3). SceneSequencer
  in[3] (sfx) becomes None -- E.5 says make it optional/None-safe (pure prepend).
- **done->gate chain:** CharacterVoices.done -> AnnouncerVoice.gate_in -> StableAudioTheme.gate_in;
  plus the post-unload `gate_signal` into the first video loader (I-7).
- STILL TO PROBE for the builder: outputs of 11(bark)/13(kokoro) -> their consumers, and the
  full input socket names of 3 SceneSequencer / 7 EpisodeAssembler. Parse on the HOST (venv
  python), never the sandbox.
- Tests: per-consumer landing, per-theme-slot landing, `no_orphan_links_after_drop_node15`,
  widget-vector-exact for the 4 new instances, no-legacy-node-instances-on-active-path,
  CastLock ledger source. Run `validate_workflow_contract` (`nodes/_workflow_validation.py`)
  on the new file as a headless gate, but treat R0b as the real acceptance.

### R0a-f / R0b / Wave 3 [operator, RTX 5080]
- R0a (f): render-twice legacy bit-identity baseline (audio only); the I-11 `_resample_audio`
  `.cuda()` at `scene_sequencer.py:126` CPU flip should be done deliberately THEN baselined.
- R0b: box-fresh live smoke of the opt-in workflow (stub engines, no weights).
- Wave 3: F dep-isolation pilots (IndexTTS2 + Stable Audio 3 are the promotion gate) -> G1 live
  inference + capture -> I promotion (flip shipped defaults, retire OTR_ENABLE_* flags).

## Surfaces already read this session (save a re-read)
Registry/base/profiles/resolved_request/script_prep/audio_utils/ledger_consumers/manifest/
all 6 engine adapters/kokoro_announcer/batch_bark_generator/musicgen_theme/top `__init__`/
conftest/class_registry/`_otr_speaker_role` -- all current. The bark/kokoro/musicgen legacy
nodes are the batch-delegation targets; their manifest widget vectors are in
`config/legacy_invocation_manifest.json` (script_json dropped; bark=[temp,bypass],
kokoro=["",random,0.95], musicgen=["",model_id,3.0,False]).

---
## Resume instructions
Open a fresh window with ComfyUI Desktop reachable, attach this file, and say:
"Read this handoff and continue at Wave 2b (build the litegraph builder + the opt-in workflow
JSON per the migration spec), then run R0b. Acknowledge when ready."
For the writer cast/stamp removal, do it only with the R0a baseline captured first.
