# Session Handoff -- OTR v2.0-alpha audio/voice CLEAN-BREAK -- 2026-06-03 (post-1a+1b)

## Core goal
Finish `docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` (the SSOT) under
the CLEAN-BREAK directive: a model-agnostic per-role engine registry (character
voice / announcer / music) as the SOLE audio path, wired into the ONE workflow
`workflows/otr_scifi_16gb_full.json`. Remove each legacy item IN LOCKSTEP with
building its replacement: build -> full suite green -> delete the legacy + all
refs in the SAME change -> green again -> guard test that fails if it reappears.

## Tech stack & hard rules (cause rework if forgotten)
Python 3.12 + torch, Windows, RTX 5080 16 GB. Branch `v2.0-alpha`, never `main`.
- Tests + git on the WINDOWS HOST via **Desktop Commander cmd** (NOT PowerShell
  for git; NOT the GitHub connector). venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Full regression after EVERY code change: `python -m pytest -q -p no:cacheprovider`
  (~35 s; must stay fully green). Baseline now: **3731 passed, 12 skipped, 0 failed**.
  Bug Bible regression also green.
- Commit via `.git\COMMIT_EDITMSG` + `git commit -F` (cmd); `git add` explicit
  paths, never `-A`. ASCII-only `.py`, no BOM, never the word "dummy".
- After every push verify: local HEAD == origin HEAD, no 0-byte, no BOM, AST
  parses, node classes registered. (A throwaway `_otr_verify_*.py` is handy.)
- **DC `interact_with_process` TIMES OUT on long pytest even though pytest
  finishes** -- run pytest with `> .log 2>&1` redirect, read the log tail (the
  log completes regardless of the MCP timeout). `del` the .log before committing.
- **Audio is king.** `baseline_v2` (render-twice from the NEW engines, operator
  GPU) is THE reference -- no v1.7 byte-identity.
- Operator mandate (2026-06-03): continue 1c -> 2 -> 3 WITHOUT pausing to ask;
  only pause for genuinely operator/GPU-gated steps (Gate B baseline_v2 capture)
  or another real change to what listeners hear.

## SHIPPED this session
- **1a (bark) -- commit `673a2bc`.** eng_bark self-contained `per_line` (sources
  `_otr_bark_lib`; voice_ref_field="voice_preset"; text_temp=0.7 pinned). Gate 3
  preset contract fails closed in generate_voice. Deleted batch_bark_generator.py
  + refs. Freeze-halt + freeze_unload_ok re-homed to `OTR_CastLock._enforce_freeze_gate`
  (covers the whole chain). Guard: tests/test_bark_legacy_node_retired.py.
- **1b (kokoro) -- commit `89e90a7`.** eng_kokoro self-contained `per_line`.
  begin_episode picks ONE announcer voice per episode, seeded from episode_seed
  over the curated pool (legacy parity -- no change to listeners); bank takes over
  at promotion via cast voice_ref_id. prepare_text identity (legacy parity).
  C-7: begin_episode verifies the voice .pt on local disk, NAMED MISSING_MODEL +
  offline fetch cmd if absent, never networks in execute. Added an additive,
  engine-agnostic **begin_episode(meta) hook** to `_otr_voice_node_common`
  per-line dispatch (runs once before the loop; other engines ignore it).
  announcer_kokoro_v1 profile speed 1.0 -> 0.95 (matches shipped cadence; pinned
  by a drift guard). Deleted kokoro_announcer.py + refs. Guard:
  tests/test_kokoro_legacy_node_retired.py.

**HEAD now `89e90a7` == origin/v2.0-alpha.** (This handoff commit sits on top.)

## OUTSTANDING (operator-only; NOT blockers for 1c)
- **Gate B (operator GPU):** capture render-twice `baseline_v2` for bark AND
  kokoro on the RTX 5080 and wire them as the byte references. Until then the
  bark/kokoro byte-tests are headless CONTRACT pins (shape/SR/fail-closed/preset/
  seeded-pick), not byte pins. per_line is intentionally NOT byte-identical to the
  old batch paths.

## State of the art
`nodes/_otr_audio_engines/`: `bark`(per_line, self-contained), `kokoro`(per_line,
self-contained), `chatterbox`/`indextts2`/`stable_audio`(per_line/clip G1 bodies,
flag-gated default-off), `musicgen`(STILL `interface="batch"` -- delegates to
musicgen_theme.py). `LEGACY_AUDIO_NODES` = {OTR_MusicGenTheme,
OTR_BatchAudioGenGenerator} (2 left). The shared per-line dispatch
`_otr_voice_node_common` has: build_engine_combo, coerce_int_seed,
frozen_batch_widgets + _manifest_path (batch only -- DEAD after 1c), _delegate_batch
(batch only -- DEAD after 1c), begin_episode hook, _render_per_line. CastLock owns
the freeze-halt + freeze_unload_ok for the whole audio chain. The theme node
`stable_audio_theme.py` (OTR_StableAudioTheme, self-contained, 3 AUDIO cue outputs)
has BOTH a `_delegate_batch` (musicgen) and a `_render_clips` (stable_audio_music,
clip interface: `generate_clip(prompt, duration_s, seed)`).

## NEXT = 1c (musicgen + audiogen + retire batch dispatch) -- LARGEST sprint
No new operator decision (musicgen per_line preserves music-gen behavior; SFX is
already out of v2). Steps (one atomic lockstep change -- cannot sub-split cleanly
because the manifest + batch dispatch serve musicgen until it is flipped):
1. **eng_musicgen batch -> clip.** Give it a self-contained `interface="clip"` +
   `generate_clip(self, prompt, duration_s, seed)` body. Relocate the MusicGen
   inference out of `musicgen_theme.py` (model load + generate) -- mirror
   `eng_stable_audio.py`'s clip body for structure (lazy import, fail-closed
   "not installed" RuntimeError headless, peak-normalize as the legacy did).
   The theme node's `_render_clips` ALREADY calls `generate_clip(prompt,
   duration_s, engine_seed)` per cue, so once musicgen is `clip` it routes there.
   Read `musicgen_theme.py` first for the cue durations / prompt / normalization
   the legacy used, to preserve what listeners hear.
2. **Delete `musicgen_theme.py` + `batch_audiogen_generator.py`** (audiogen =
   dead legacy SFX, node 15 dropped). Remove both from `__init__.py`,
   `_otr_legacy_manifest.LEGACY_AUDIO_NODES`, `legacy_invocation_manifest.json`.
3. **Remove the theme node batch path**: `StableAudioTheme._delegate_batch` +
   the `if interface == "batch"` branch in its `generate()`.
4. **Retire the shared batch dispatch** in `_otr_voice_node_common`:
   `_delegate_batch`, `frozen_batch_widgets`, `_manifest_path`, and the
   `if interface == "batch"` branch in `OTRVoiceNodeBase.generate()`. KEEP
   `build_engine_combo`, `coerce_int_seed`, the `begin_episode` hook,
   `_render_per_line`. (I-3 raw-delegation retired -- the last batch user is gone.)
5. **LEGACY_AUDIO_NODES is now empty** -> delete `nodes/_otr_legacy_manifest.py`
   + `config/legacy_invocation_manifest.json` + tests
   `test_legacy_invocation_manifest.py` + `test_legacy_audio_seeding.py`.
6. **Tests**: convert `test_stable_audio_theme.py` (musicgen batch->clip),
   `test_musicgen_*` (brief_rewire/cache_keys/parity/strict_failure),
   `test_audiogen_*` (cache_keys/ledger/strict_failure/writeback_hardening),
   `test_per_cue_sfx_dur.py`; drop musicgen+audiogen keys from
   `test_audio_byte_identical.py` FIXED_SEEDS (likely empties it -- keep the dict
   or delete the now-dead capture path). KEEP the workflow-JSON denylists in
   `test_full_workflow_v2_audio_wiring.py` + `test_workflow_json_guardrails.py`
   (they name the types as strings to assert NO instance -- still valid). Add a
   `tests/test_batch_dispatch_retired.py` guard: musicgen+audiogen modules gone;
   `_otr_voice_node_common` has no `_delegate_batch`/`frozen_batch_widgets`;
   `OTRVoiceNodeBase.generate` has no batch branch; `_otr_legacy_manifest` gone.
7. Full suite green; commit + push; verify.

## Then 2 -> 3 (still headless), 4 -> 5 (operator/GPU)
- **2**: remove the writer bark voice_preset stamp in `_otr_casting`
  (`python_assign_voice_preset` + `_assert_voice_preset_invariant` + uniqueness
  guard); OTR_CastLock owns casting (bank voice_ref_id); bark draws from the bank.
- **3**: remove R0a legacy seeding (now only the deleted nodes had it -- mostly
  done by 1c) + any remaining `legacy_invocation_manifest` refs; `baseline_v2`
  replaces the render-twice-LEGACY tests (ties to Gate B operator GPU).
- **4**: promotion -- flip full.json engine-widget defaults to the new engines
  per role; retire `OTR_ENABLE_*`. GATED on Wave-3 F GPU pilots (operator).
- **5**: F probes for indextts2 + stable_audio (isolated venvs); operator GPU.

## Resume instructions
Open a fresh window with the project mounted, attach this file, and say:
"Read this handoff + docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md.
Verify HEAD == origin/v2.0-alpha and the suite is green (3731/12/0). Continue the
EXECUTION-PLAN under the CLEAN-BREAK directive, removing legacy in lockstep --
do sprint 1c (musicgen clip + audiogen delete + retire the batch dispatch +
delete the legacy manifest), then 2 -> 3, without pausing to ask; only pause for
operator/GPU-gated steps (Gate B) or a real change to what listeners hear.
Acknowledge when ready."
