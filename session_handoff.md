# Session Handoff -- OTR v2.0-alpha audio/voice CLEAN-BREAK -- 2026-06-02

## CLEAN-BREAK DIRECTIVE (operator, non-negotiable) -- READ FIRST
The new engine registry is the ONE true audio path. NO permanent legacy
fallback, no raw delegation, no v1.7 byte-identity crutch.
REMOVE EACH LEGACY ITEM IN TANDEM WITH BUILDING ITS REPLACEMENT -- never defer
legacy removal to a later cleanup. RATIONALE (operator): if legacy is not
removed as each replacement lands, you will be SPRINTING INDEFINITELY chasing
legacy removal later; deferred legacy rots and sidetracks every future change.
LOCKSTEP RULE, per piece: (1) build the replacement + wire into full.json, (2)
full suite green, (3) in the SAME change delete the legacy it replaced + every
reference, (4) suite green again with zero orphan symbols, (5) add a guard test
that FAILS if that legacy reappears. Never delete legacy before its replacement
is proven green; never close a sprint with the replaced legacy still in the tree.
Supersedes the EXECUTION-PLAN's I-1 / I-3 / C-5 / C-6 / H / I, plus R0a legacy
seeding + legacy_invocation_manifest + the render-twice-LEGACY baseline (capture
baseline_v2 from the NEW engines instead). Keep bark/kokoro/musicgen as NORMAL
registry adapters (no "legacy" status, self-contained, no delegation). With no
fallback, the F dep-pilot is a HARD render prerequisite; a missing model/dep
raises the C-7 named error, never a silent fallback.

## Core goal
Finish docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md (the SSOT) under
the clean-break above: a model-agnostic per-role engine registry (character
voice / announcer / music) as the SOLE audio path, wired into the ONE workflow
workflows/otr_scifi_16gb_full.json. Every session: continue the build AND rip
legacy in lockstep -- do BOTH.

## Tech stack & constraints (full set in CLAUDE.md + the EXECUTION-PLAN)
- Python 3.12 + torch, Windows, RTX 5080 16 GB. Branch v2.0-alpha, never main.
- ONE workflow of record: workflows/otr_scifi_16gb_full.json (29 nodes, 71 links).
- Tests + git on the WINDOWS HOST via Desktop Commander cmd (NOT PowerShell for
  git; NOT the GitHub connector -- it is context-only). venv python:
  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe.
- Full regression after EVERY change: python -m pytest -q -p no:cacheprovider
  (conftest SystemExit(2) on ANY new failure -- must be fully green). Commit via
  .git\COMMIT_EDITMSG + git commit -F; git add explicit paths, never -A.
  ASCII-only .py, no BOM, never the word "dummy" (use placeholder/test).
- VRAM 14.5 GB ceiling. I-11: post-engine audio DSP runs on CPU. baseline_v2
  (new engines) is THE audio reference now (no v1.7 byte-identity).

## What's done (committed + pushed; HEAD bf10ef7 == origin/v2.0-alpha; suite 3727/12/0)
- 7d8b758 I-11: post-engine DSP forced to CPU (scene_sequencer._resample_audio
  dropped the GPU torchaudio path; audio_enhance.AudioEnhance.enhance is CPU-only).
- 9b57e7a G1: inference bodies for chatterbox / indextts2 / stable_audio written
  to each documented assumed_call; new base.supported_kwargs guard drops any
  kwarg the real lib does not accept; +8 tests (tests/test_audio_engine_bodies_g1).
  Flag-gated, default-off, supports_external_generator False (set True per engine
  after F). Headless (lib absent) each body fails closed with "not installed".
- 4f1a853 clean-break directive adopted in both docs + chatterbox F result.
- bf10ef7 1a-PREP (safe half): relocated the pure Bark per-line helpers
  (_clean_text_for_bark / _chunk_text_for_bark / _generate_single_line) byte-exactly
  into _otr_bark_lib.py; batch_bark_generator.py re-exports them (all import/patch
  targets still resolve); story_orchestrator sources _generate_single_line from the
  lib. Zero behavior change. This is the delegation-free foundation for eng_bark.
  The flip+delete half is GATED -- see below + docs/2026-06-03-bark-cleanbreak-1a__decision-brief.md.

## F dep-pilot results (HARD prerequisite under clean-break)
Harness scripts/otr_audio_dep_pilot.py: offline, subprocess-isolated, --python
<venv>; diffs torch + xformers/flash_attn before/after import; reads assumed_call.
- chatterbox: PASS. Isolated venv created via Desktop Commander under the Claude
  packaged-app AppData\Local\otr_pilot_venvs\chatterbox. torch 2.6.0+cpu, NO
  xformers/flash_attn. Real ChatterboxTTS.generate(text, repetition_penalty,
  min_p, top_p, audio_prompt_path, exaggeration, cfg_weight, temperature) +
  from_pretrained(device). G1 body MATCHES (supported_kwargs drops the nonexistent
  cfg + generator). generate binds NO external generator -> supports_external_
  generator stays False; determinism comes from the deterministic_inference
  global-seed wrap. Body validated against the real API.
- indextts2, stable_audio: NOT yet probed -- need isolated-venv installs + signature
  read; then reconcile the G1 bodies + flip supports_external_generator if they bind a generator.
- bark, kokoro, musicgen: libs ALREADY in the MAIN ComfyUI venv (the legacy nodes
  use them) -> NO isolated venv needed; their clean-break is a refactor of working code.

## Engine landscape (nodes/_otr_audio_engines/)
base.py, registry.py, eng_bark, eng_chatterbox, eng_indextts2, eng_kokoro,
eng_musicgen, eng_stable_audio.
- chatterbox / indextts2 / stable_audio already have self-contained G1 bodies.
- bark / kokoro / musicgen are currently interface="batch" -> delegate verbatim
  to the standalone LEGACY nodes: batch_bark_generator.py, kokoro_announcer.py,
  musicgen_theme.py, batch_audiogen_generator.py. Batch dispatch is at
  _otr_voice_node_common.py:217 (voice) and stable_audio_theme.py:172 (music).
- full.json selects engines by widget value ("bark" L150/L1850, "kokoro" L1916,
  "musicgen" L1982); the legacy nodes are NOT separate graph nodes -- reached
  only via delegation. So converting an adapter to self-contained needs no
  full.json node edit; only the engine-widget defaults change at promotion.

## Clean-break sprints remaining (each LOCKSTEP: build replacement -> wire ->
## suite green -> delete the legacy + every ref in the SAME change -> green +
## guard test that fails if it reappears)
1a. eng_bark -> self-contained per_line body (extract bark inference from
    _otr_bark_lib / BatchBarkGenerator). Delete batch_bark_generator.py + the bark
    "wrapper == legacy byte-identical" tests (convert to baseline_v2). Keep the
    batch dispatch (kokoro/musicgen still use it). Guard test: BatchBarkGenerator gone.
1b. eng_kokoro -> self-contained announcer body. Delete kokoro_announcer.py + refs.
1c. eng_musicgen -> self-contained clip body. Delete musicgen_theme.py +
    batch_audiogen_generator.py, then REMOVE the batch dispatch entirely
    (_otr_voice_node_common:217 + stable_audio_theme:172) -- last batch user, I-3 gone.
2.  Remove the writer bark voice_preset stamp in _otr_casting
    (python_assign_voice_preset + _assert_voice_preset_invariant + uniqueness
    guard); OTR_CastLock owns casting (bank voice_ref_id); bark draws from the bank.
3.  Remove R0a legacy seeding + config/legacy_invocation_manifest.json; capture
    baseline_v2 from the new engines (replace render-twice-LEGACY tests).
4.  Promotion: flip full.json engine-widget defaults to the new engines per role;
    retire OTR_ENABLE_* gating. Gated on F-validation of the chosen defaults.
5.  F probes for indextts2 + stable_audio (isolated venvs); flip
    supports_external_generator + reconcile bodies vs real signatures.

## Immediate next step
1a-PREP is DONE (bf10ef7): bark inference is relocated + delegation-free. The
remaining half (flip bark to per_line + delete batch_bark_generator.py) is BLOCKED
on two operator gates + one settled design, ALL written up with ready-to-apply
code in docs/2026-06-03-bark-cleanbreak-1a__decision-brief.md:
  - GATE A (decide): where the freeze-halt safety gate (BUG-276/300) re-homes.
    It lives ONLY in batch_bark_generator today; no v2 node re-homes it; deleting
    the node drops it. Recommend OTR_CastLock + OTR_BYPASS_FREEZE_HALT env (the
    bypass widget cannot survive -- v2 nodes forbid extra widgets). Appendix C.
  - GATE B (operator GPU): capture render-twice baseline_v2 for bark. The per_line
    path is NOT byte-identical to the batch path, so baseline_v2 is the new
    reference (clean-break directive) and I cannot capture it headless.
  - SETTLED: voice_preset routing via a voice_ref_field adapter attr (appendix A,
    non-breaking for chatterbox/indextts2); eng_bark per_line body (appendix B);
    BUG-096 ledger writeback is REDUNDANT (scene_sequencer.py:768-903 already
    writes authoritative start_s/dur_s) -> safe to drop with the node.
Then: apply A+B+C, delete the node + bark refs (manifest/init/legacy-manifest),
convert the 6 bark test files (list in the brief), add the reappearance guard
test, suite green, capture baseline_v2, commit + push.

## Open questions
- bark/kokoro/musicgen real per-line inference shapes (read _otr_bark_lib etc.).
- indextts2 construction (cfg/model_dir) + stable_audio entry point -- F resolves.

---
## Resume instructions
Open a fresh window with the project mounted, attach this file, and say:
"Read this handoff + docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md.
Verify HEAD == origin/v2.0-alpha and the suite is green (3727). Then continue the
EXECUTION-PLAN under the CLEAN-BREAK directive -- build each replacement AND
delete the legacy it replaces in the SAME change (lockstep), starting with the
bark sprint (1a). Acknowledge when ready."
