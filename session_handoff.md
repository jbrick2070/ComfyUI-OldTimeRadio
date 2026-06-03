# Session Handoff -- OTR v2.0-alpha (audio/voice overhaul) -- 2026-06-02

## DECISION (2026-06-02, operator): CLEAN-BREAK the legacy path -- supersedes every "permanent legacy fallback" line in THIS doc and in the EXECUTION-PLAN
No permanent legacy fallback. The new engine registry is the ONE true audio path. Each
session: continue the EXECUTION-PLAN build AND rip out legacy in lockstep -- do BOTH, every
time. Rule per piece: (1) build the replacement + wire into `full.json`, (2) full suite
green, (3) in the SAME change delete the legacy it replaced + every reference, (4) suite
green again, zero orphan symbols. Never delete legacy before its replacement is proven
green; never close a sprint with the replaced legacy still in the tree; a sprint's
definition of done includes a guard test that FAILS if that legacy reappears. Do NOT defer
legacy removal to an end-of-project cleanup -- deferred legacy rots and sidetracks every
future change.
Supersedes, specifically: I-1 "permanent fallback" (legacy is transitional, gone by
promotion); the "Writer cast/stamp removal" note below -- the legacy bark `voice_preset`
stamp in `_otr_casting.lock_cast` is now a REMOVAL TARGET (OTR_CastLock already owns v2
casting), not something to preserve, and is no longer legacy-baseline-gated; R0a
"render-twice LEGACY baseline" -> capture `baseline_v2` from the NEW engines instead (no
v1.7 baseline to preserve); Promotion "legacy stays the fallback" -> legacy is retired.
KEEP bark/kokoro/musicgen as NORMAL registry adapters (engine options, no "legacy" status).
With no fallback, the F dep-pilot is a HARD prerequisite for a working render (missing
model/dep -> C-7 named error, never a silent fallback).

## Core goal
v2.0-alpha audio + voice-casting overhaul per
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` (the SSOT). Model-
agnostic per-role engine registry + voice casting; the new path is the product,
legacy is the PERMANENT byte-identical fallback (I-1). Everything wires into the
ONE workflow of record, `workflows/otr_scifi_16gb_full.json` (no second json).

## Tech stack & constraints (full set in CLAUDE.md + the EXECUTION-PLAN)
- Python 3.12 + torch, Windows, RTX 5080 16 GB. Branch `v2.0-alpha`, never main.
- ONE json of record = `workflows/otr_scifi_16gb_full.json` (29 nodes, 71 links).
- Tests + git on the WINDOWS HOST via Desktop Commander `cmd`. venv:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Full regression after every change: `python -m pytest -q -p no:cacheprovider`
  (conftest `SystemExit(2)` on ANY new failure). Commit via `.git\COMMIT_EDITMSG`
  + `git commit -F`; `git add` explicit paths, never `-A`. ASCII .py, no BOM.
- Audio is king. I-11: post-engine DSP runs on CPU. `supports_external_generator`
  flips True ONLY after the F GPU pilot verifies a bound `torch.Generator`.

## What's done & decided (this session -- committed + pushed)
- `7d8b758` **I-11 post-engine DSP on CPU.** `scene_sequencer._resample_audio`
  dropped the GPU torchaudio fast path; `audio_enhance.AudioEnhance.enhance`
  forces the whole enhance chain to CPU (removed the cuda move + `_use_cuda`).
  No node surface change. Suite green.
- `9b57e7a` **G1 engine inference bodies.** `eng_chatterbox` / `eng_indextts2` /
  `eng_stable_audio` now implement their documented assumed_call: lazy-import the
  lib inside `load()`/generate, seed locally on top of the `deterministic_inference`
  wrap, bind a per-line `torch.Generator` when the signature accepts one, return
  `{"waveform","sample_rate"}`. New `base.supported_kwargs` drops any kwarg the
  real lib does not accept (an assumed_call name F later corrects cannot crash the
  forward). STILL flag-gated + default-off; `supports_external_generator` stays
  False (operator flips after F). Headless (libs absent) each body fails closed
  with "not installed". +8 tests (`tests/test_audio_engine_bodies_g1.py`).
- Suite GREEN: **3727 passed, 12 skipped, 0 failed**. Wiring guard
  `test_full_workflow_v2_audio_wiring.py` green (16/16); full.json has all 4 v2
  nodes (OTR_CastLock, OTR_BatchCharacterVoices, OTR_AnnouncerVoice,
  OTR_StableAudioTheme).

## NOT safe headless -- needs the 5080 (do NOT do blind)
- **Writer cast/stamp removal (Wave 2a tail).** `_otr_casting.lock_cast` stamps
  the BARK `voice_preset` (`v2/en_speaker_X`) that the LEGACY bark path reads from
  node 62 `out[1]`, guarded by `_assert_voice_preset_invariant`. Removing it
  breaks the permanent legacy fallback (I-1). The v2 `voice_ref_id` casting is
  ALREADY owned by `OTR_CastLock`. Full migration is a legacy-retirement step
  and/or needs the R0a render-twice baseline to prove legacy bytes unchanged.
- **Promotion (flip full.json defaults to the new engines)** must wait for F --
  defaulting to unvalidated inference bodies would break box-fresh renders.

## Immediate next steps (GPU / operator)
1. **F dependency pilot (THE unblock):** install chatterbox / indextts2 /
   stable-audio-tools each in its OWN venv, then
   `...python.exe scripts\otr_audio_dep_pilot.py --json` (or `--python <venv>`).
   Per engine confirm: import clean, torch unchanged, no xformers / flash_attn,
   the real generate/infer/constructor signature, and a bound `torch.Generator`.
   Reconcile the G1 bodies' `GPU-VALIDATE` markers, then flip
   `supports_external_generator=True`.
2. **R0a baseline:** render-twice legacy bit-identity; capture
   `baseline_v2_audio_legacy_{sha,ledger_sha,audio_metadata_sha}`. Unblocks the
   writer cast work + promotion.
3. **Promotion (I):** after F, flip full.json defaults to the best engine per
   role + retire `OTR_ENABLE_*`; legacy stays the fallback.
4. **R0b box-fresh smoke:** launch via `scripts\run_comfy_otr.bat`, load
   full.json, run stub engines, emit a minimal MP4.

## Open questions
- Order of GPU sprints (F pilot vs R0a baseline). F is the G1/promotion unblock.

---
## Resume instructions
Open a fresh window with the project mounted, attach this file, and say:
"Read this handoff and continue the v2.0-alpha audio/voice overhaul. Verify
HEAD == origin/v2.0-alpha. Headless is complete (I-11 + G1 bodies committed,
suite green at 3727, full.json wired 16/16). The remaining work is GPU/operator
-- run the F dependency pilot (scripts/otr_audio_dep_pilot.py) to verify
signatures + flip supports_external_generator, then promotion. Acknowledge when
ready."

## Clean-break progress + F results (2026-06-02, LATEST -- authoritative)
HEAD == origin/v2.0-alpha at 574622d; suite green 3727. Clean-break directive
(top of this doc + the EXECUTION-PLAN) ADOPTED: new engine registry is the ONE
true audio path, no fallback/delegation, F dep-pilot is a HARD render prereq.

F dep-pilot results (isolated venvs created via Desktop Commander under the
Claude packaged-app AppData\Local\otr_pilot_venvs\<engine>):
- chatterbox: PASS. torch 2.6.0+cpu, NO xformers / flash_attn. Real signature
  ChatterboxTTS.generate(text, repetition_penalty, min_p, top_p,
  audio_prompt_path, exaggeration, cfg_weight, temperature) + from_pretrained(device).
  G1 body MATCHES -- supported_kwargs drops the non-existent cfg + generator;
  generate binds NO external generator so supports_external_generator stays False
  (determinism via the deterministic_inference global-seed wrap). Body validated.
- indextts2 / stable_audio / bark / kokoro / musicgen: NOT yet probed.

Engine landscape: nodes/_otr_audio_engines/ has eng_bark/eng_chatterbox/
eng_indextts2/eng_kokoro/eng_musicgen/eng_stable_audio. bark/kokoro/musicgen
currently delegate (interface="batch") to the standalone LEGACY nodes
batch_bark_generator.py / kokoro_announcer.py / musicgen_theme.py /
batch_audiogen_generator.py.

Clean-break removal targets (each a lockstep sprint: build self-contained
adapter -> wire full.json -> suite green -> DELETE the legacy + every ref in the
SAME change -> green + a guard test that FAILS if the legacy reappears):
1. Raw-delegation (I-3): make eng_bark/eng_kokoro/eng_musicgen self-contained
   per_line/clip adapters, then delete batch_bark_generator.py /
   kokoro_announcer.py / musicgen_theme.py / batch_audiogen_generator.py + the
   "batch" dispatch in _otr_voice_node_common + stable_audio_theme. (_otr_bark_lib
   may remain as the bark adapter's inference impl -- verify.)
2. Writer bark voice_preset stamp in _otr_casting (python_assign_voice_preset +
   _assert_voice_preset_invariant + uniqueness guard) -> REMOVE; OTR_CastLock
   owns casting; bark draws its voice from the bank.
3. R0a legacy seeding + config/legacy_invocation_manifest.json -> REMOVE; capture
   baseline_v2 from the NEW engines.
4. C-5/C-6 legacy-first defaults -> flip full.json defaults to the new engines
   (promotion); retire OTR_ENABLE_* gating.
5. Legacy byte-identity tests -> replace with baseline_v2 tests.

NOTE: every removal is gated on first building + proving (suite green) the
self-contained replacement; replacements for bark/kokoro/musicgen need their F
probe (install lib in an isolated venv, confirm import clean + real signature).

Next: continue F (install + probe indextts2/stable_audio/bark/kokoro/musicgen),
then the first lockstep sprint -- eng_bark self-contained -> delete
batch_bark_generator + the batch delegation + refs -> guard test.
