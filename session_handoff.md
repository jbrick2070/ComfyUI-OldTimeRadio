# Session Handoff -- OTR v2.0-alpha audio/voice CLEAN-BREAK -- 2026-06-03 (post-1a)

## Core goal
Finish `docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` (the SSOT) under
the CLEAN-BREAK directive: a model-agnostic per-role engine registry (character
voice / announcer / music) as the SOLE audio path, wired into the ONE workflow
`workflows/otr_scifi_16gb_full.json`. Work the clean-break sprints to completion,
removing each legacy item IN LOCKSTEP with building its replacement: build ->
wire -> full suite green -> delete the legacy + all refs in the SAME change ->
green again -> guard test that fails if it reappears. Full directive is in the
EXECUTION-PLAN header.

## Tech stack & hard rules (cause rework if forgotten)
Python 3.12 + torch, Windows, RTX 5080 16 GB. Branch `v2.0-alpha`, never `main`.
- Tests + git on the WINDOWS HOST via **Desktop Commander cmd** (NOT PowerShell
  for git; NOT the GitHub connector). venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Full regression after EVERY code change: `python -m pytest -q -p no:cacheprovider`
  (~35 s; must stay fully green). Baseline: **3727 passed, 12 skipped, 0 failed**.
  Bug Bible regression also green (23 passed, 1 skipped, 2 xfailed).
- Commit via `.git\COMMIT_EDITMSG` + `git commit -F` (cmd); `git add` explicit
  paths, never `-A`. ASCII-only `.py`, no BOM, never the word "dummy".
- After every push verify: local HEAD == origin HEAD, no 0-byte files, no BOM,
  AST parses, node classes registered.
- DC `interact_with_process` reliably TIMES OUT on long pytest even though pytest
  finishes -- run pytest with `> .log 2>&1` redirect, then read the log tail (the
  log completes regardless of the MCP timeout). Delete the .log before committing.
- **Audio is king** (Prime Directive #1). `baseline_v2` (render-twice from the
  NEW engines, operator GPU) is THE reference -- no v1.7 byte-identity.

## SHIPPED this session -- sprint 1a COMPLETE (HEAD `673a2bc` == origin/v2.0-alpha)
Commit `673a2bc` "Audio v2 1a: flip bark to per_line engine, retire the batch
node" (20 files, +460/-1316). Verified: HEAD==origin, AST+BOM+non-empty OK, bark
registered per_line, batch module gone. Suite green 3727/12/0; Bug Bible green.

What landed (Gate A + temp decision were confirmed by the operator):
- **eng_bark.py** is now self-contained `interface="per_line"`: sources inference
  from `_otr_bark_lib` (load/generate/unload), `voice_ref_field="voice_preset"`,
  `text_temp=0.7` (== char_bark_v1 profile SSOT, pinned by a drift guard).
  Gate 3 (v2/* preset contract) fails closed with `EngineUnusable(MALFORMED_CONFIG)`
  before any model load. No `make_batch_node`.
- **_otr_voice_node_common._render_per_line**: voice_ref routing via
  `voice_ref_field` (appendix A) -- bark routes `cast.voice_preset` into the ref
  slot; cloning engines keep `voice_ref_path` (non-breaking).
- **cast_lock.py**: re-homed the freeze-halt gate (BUG-276/300) AND the
  `freeze_unload_ok` defensive VRAM unload (E9) from the legacy audio nodes into
  `OTR_CastLock._enforce_freeze_gate` (runs first in the chain -> one gate covers
  CharacterVoices+Announcer+Theme). Per-node `bypass_freeze_halt` widget ->
  env `OTR_BYPASS_FREEZE_HALT`. **This shared re-home already covers 1b + 1c**, so
  deleting kokoro/musicgen does NOT need to re-home their freeze-halt copies.
- **Deleted** `nodes/batch_bark_generator.py` + removed `OTR_BatchBarkGenerator`
  from `__init__.py`, `_otr_legacy_manifest.LEGACY_AUDIO_NODES`, and
  `config/legacy_invocation_manifest.json` (bark entry only).
- **Tests**: new `tests/test_bark_legacy_node_retired.py` guard (file/import/
  registration/manifest absent; bark per_line, no make_batch_node). Converted
  test_batch_character_voices (per_line dispatch), test_bark_cast_contract (Gate 3
  in eng_bark), test_bark_freeze_halt_bypass (CastLock functional + env),
  test_audio_engine_adapters (bark per_line, kokoro/musicgen still batch).
  Repointed test_core / test_no_orchestrator_legacy_symbols /
  test_freeze_unload_ok_consumed (now pins `nodes/cast_lock.py`). Dropped bark from
  test_legacy_audio_seeding + test_audio_byte_identical + the manifest "present"
  test. Deleted `tests/test_bark_ledger.py` (per-line timing writeback redundant --
  scene_sequencer.py:768-903 writes authoritative timings).

## OUTSTANDING for 1a (operator-only, NOT a blocker for 1b)
- **Gate B (operator GPU):** capture render-twice `baseline_v2` for bark on the
  RTX 5080 and wire it in as the byte reference. The bark byte-tests are currently
  headless CONTRACT pins (shape/SR/fail-closed/preset-routing), not byte pins.

## State of the art
Engine adapters in `nodes/_otr_audio_engines/`: `bark`(now per_line, self-
contained), `chatterbox`/`indextts2`/`stable_audio`(per_line G1 bodies, flag-
gated default-off), `kokoro`/`musicgen`(STILL `interface="batch"` -- delegate).
`LEGACY_AUDIO_NODES` = {OTR_KokoroAnnouncer, OTR_MusicGenTheme,
OTR_BatchAudioGenGenerator} (3, bark removed). Shared dispatch
`_otr_voice_node_common`: `_delegate_batch` (~245) for batch, `_render_per_line`
(~280) for per_line. CastLock owns the freeze-halt + unload for the whole chain.

## Remaining clean-break sprints (each LOCKSTEP; build->green->delete->green->guard)
- **1b (kokoro)** -- NEXT. eng_kokoro self-contained announcer body; delete
  `kokoro_announcer.py` + refs (drop its freeze-halt copy -- CastLock covers it;
  drop its redundant ledger writeback -- scene_sequencer covers timings). Convert
  kokoro tests, add guard. SEE OPEN DESIGN DECISIONS BELOW -- 1b needs a call
  before cutting code (no brief exists yet, unlike 1a).
- **1c (musicgen)** -- eng_musicgen self-contained clip body; delete
  `musicgen_theme.py` + `batch_audiogen_generator.py`, then REMOVE the batch
  dispatch entirely (`_otr_voice_node_common._delegate_batch` + `frozen_batch_widgets`
  + `stable_audio_theme.py` batch branch) -- last batch user, I-3 retired. Then
  `_otr_legacy_manifest` + `legacy_invocation_manifest.json` go empty/removed.
- **2** -- remove the writer bark voice_preset stamp in `_otr_casting`
  (`python_assign_voice_preset` + `_assert_voice_preset_invariant` + uniqueness
  guard); OTR_CastLock owns casting (bank `voice_ref_id`); bark draws from the bank.
- **3** -- remove R0a legacy seeding + `config/legacy_invocation_manifest.json`;
  `baseline_v2` replaces the render-twice-LEGACY tests (ties to Gate B operator GPU).
- **4** -- promotion: flip full.json engine-widget defaults to the new engines per
  role; retire `OTR_ENABLE_*` gating. GATED on Wave-3 F GPU pilots (operator).
- **5** -- F probes for indextts2 + stable_audio (isolated venvs); operator GPU.

## 1b OPEN DESIGN DECISIONS (resolve before cutting -- analogous to 1a's Gate A)
`kokoro_announcer.py` per-line body is NOT a thin extract. Decisions:
- **D1 announcer voice sourcing.** Legacy `_pick_announcer_voice(episode_seed,
  voice_override)` picks ONE voice per episode (seeded) from `ANNOUNCER_VOICE_POOL`
  = {bm_george, bm_fable, bf_emma, bf_lily}. For per_line eng_kokoro:
  (a) CastLock `auto_registry` assigns via `announcer_voice_ref("kokoro")` from the
  voice bank, cast carries `voice_ref_id`, eng_kokoro.voice_ref_field reads it
  (clean-break-aligned -- CastLock is the cast authority); OR
  (b) preserve the per-episode seeded pick inside eng_kokoro (reads `episode_seed`
  from ledger meta). NOTE preserve_ledger (default) cast may carry NO kokoro voice,
  so (a) needs auto_registry + a kokoro bank entry to exist
  (check `config/voice_reference_bank.json` -- it currently notes bark is not
  represented; confirm kokoro announcer refs are present, else (b) or add them).
- **D2 C-7 fetch-during-execute.** Legacy `_ensure_voice_file` does
  `hf_hub_download` DURING render -- violates C-7 ("NEVER fetch/network during
  execute"). Clean-break options: move to a named-fetch preflight (C-7 correct) OR
  preserve (pragmatic; ~12 MB once). Recommend the preflight to honor C-7/offline.
- **D3 inference relocation.** bark used a 1a-PREP that relocated inference to
  `_otr_bark_lib`. kokoro inference (KPipeline, ~30 lines) is small enough to INLINE
  in `eng_kokoro.generate_voice`, or relocate to `_otr_kokoro_lib.py` for symmetry.
- kokoro tests to convert/retire: `test_kokoro_ledger.py` (retire -- redundant
  writeback), `test_announcer_voice.py` + `test_announcer_passes.py` (batch ->
  per_line; they stub a name-mirror `KokoroAnnouncer` for the manifest lookup --
  same pattern test_batch_character_voices used for bark), drop kokoro from
  `test_legacy_audio_seeding` (CASES -> 2) + `test_legacy_invocation_manifest`
  (set -> 2) + `legacy_invocation_manifest.json` kokoro entry + `__init__.py`
  registration. Add `tests/test_kokoro_legacy_node_retired.py` guard. eng_kokoro
  needs `sample_rate=24000` (KOKORO_SAMPLE_RATE) + a `voice_ref_field`.

## Immediate next steps
1. Read this handoff + the EXECUTION-PLAN + (for 1a context) the 1a decision brief.
2. Verify HEAD == origin/v2.0-alpha (`673a2bc`) + suite green 3727/12/0.
3. Resolve 1b D1/D2 (operator call), then build eng_kokoro per_line + wire +
   green + delete kokoro_announcer.py + refs + convert tests + guard + green +
   commit + push. Then 1c -> 2 -> 3 -> 4 (F-gated) -> 5 (F operator GPU).

## Resume instructions
Open a fresh window with the project mounted, attach this file, and say:
"Read this handoff + docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md.
Verify HEAD == origin/v2.0-alpha (673a2bc) and the suite is green (3727/12/0).
Continue the EXECUTION-PLAN under the CLEAN-BREAK directive, removing legacy in
lockstep, until all sprints are done -- starting sprint 1b (kokoro): resolve the
D1/D2 design decisions, then the flip+delete. Acknowledge when ready."
