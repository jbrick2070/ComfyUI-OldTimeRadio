# Session Handoff -- OTR v2.0-alpha build (audio/voice progression) -- 2026-06-02

## Core goal
This is the ongoing **v2.0-alpha** build of OldTimeRadio -- a long, far-from-
complete progression, not a finished feature. The current focus is the
audio/voice update: a model-agnostic, per-role audio engine registry + voice-
casting subsystem (character voice / announcer / theme music selectable per role;
deterministic voice bank + caster; a post-freeze cast-lock; a frozen
ResolvedVoiceRequest identity/cache contract). **Everything is wired into the ONE
workflow of record, `workflows/otr_scifi_16gb_full.json` -- there is no second /
opt-in json.** Legacy audio stays a permanent byte-identical fallback (the new
nodes delegate to it by default).

## Tech stack & constraints
ComfyUI custom-node package (Python 3.12 + torch, Windows, RTX 5080 16 GB),
branch `v2.0-alpha` (never touch `main`). `CLAUDE.md` auto-loads; the rules that
cause rework if forgotten:
- **ONE json of record = `workflows/otr_scifi_16gb_full.json`.** Wire EVERY change
  into it (CLAUDE.md rule 3). Do NOT create a second / opt-in workflow json -- the
  operator rejected the plan's two-file design this session.
- **Tests + git + any workflow-JSON parsing run on the WINDOWS HOST**, never the
  Linux sandbox (no torch; stale CRLF mount). venv python
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`; git stage/commit/
  push via Desktop Commander `cmd` (never PowerShell for git; never the GitHub
  connector). Full regression after every change (redirect to a file, then read):
  `...python.exe -m pytest -q -p no:cacheprovider`. The conftest known-fail guard
  hard-exits `SystemExit(2)` on ANY new failure -- the suite must be fully green.
- **ASCII-only `.py` source, no em-dash, no BOM.** Audio is king (byte-identical
  legacy fallback; post-engine DSP on CPU, I-11; VRAM ceiling 14.5 GB with single-
  engine residency + teardown-before-done, I-7). Never the word "dummy".
- Commit message via `.git\COMMIT_EDITMSG` + `git commit -F` (cmd mangles `-m`).
  `git add` explicit paths or `-u`, NEVER `-A` (untracked planning docs stay out).

## Spec SSOT (read, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` is the authoritative
wave order, invariants I-1..I-11, ComfyUI C-1..C-7, ResolvedVoiceRequest fields,
per-sprint tests, and re-baseline triggers. This handoff captures only the LIVE
deltas on top of it.

## What's done & decided
- **Operator correction (this session, do not re-litigate):** single json of
  record. The audio lane is wired directly into `full.json`; a separate opt-in
  JSON + a one-shot builder were created then REMOVED. Never reintroduce a 2nd json.
- **`full.json` now carries the audio lane** (commit `a46763b`): CastLock (id 80)
  + BatchCharacterVoices (81) + AnnouncerVoice (82) + StableAudioTheme (83) wired
  in; the legacy bark/kokoro/musicgen/audiogen INSTANCES (old ids 11/13/14/15)
  removed; SFX dropped (lean-alpha = no SFX). Engine widgets default to the legacy
  engines + CastLock `preserve_ledger`, so the lane delegates to the legacy
  generators with the frozen manifest widgets and the produced audio is
  byte-identical by default.
- **Node classes + libraries shipped + registered** (Wave 0 contracts, Wave 1
  nodes 1a-1g, Wave 2a CastLock) -- see git log + `nodes/_otr_class_registry.py`.
- **Guard tests updated in lockstep** with the new canonical wiring: the
  ledger-source invariant now allows the CastLock authority; the bark / musicgen /
  audiogen widget-vector pins were repointed to `config/legacy_invocation_manifest.json`
  (the frozen vectors the lane delegates with), so no drift coverage was lost.
- **Green:** full tests/ 3695 passed, 12 skipped, 0 failed.
- **Rejected:** the two-file opt-in-copy approach; reintroducing SFX (lean-alpha).

## State of the art
- **HEAD = `8585a6c` == `origin/v2.0-alpha`** (verify with `git rev-parse HEAD`
  vs `origin/v2.0-alpha` first). Recent: `8585a6c` handoff, `a46763b` full.json
  consolidation, `8e3d5c2` Wave 2b build (superseded), `4ea3fc2` 2a CastLock.
- **`workflows/otr_scifi_16gb_full.json`** -- 29 nodes, the migrated graph.
  node-62 OTR_LedgerFreezeCascade out[1] script_json -> raw consumers
  {3,12,20,52,55,59,71} + the 3 audio nodes (byte-identical batch delegation);
  out[6] v2_ledger_json -> CastLock(80); CastLock.ledger_json -> {81,82,83} +
  HuMo(51); theme cues by name; done->gate chain 81->82->83; audio_done->FLUX
  loader gate (link 209) preserved.
- **Node files** (import headless; engine libs lazy-imported inside generate):
  `nodes/cast_lock.py`, `batch_character_voices.py`, `announcer_voice.py`,
  `stable_audio_theme.py`, shared base `nodes/_otr_voice_node_common.py`. Libs:
  `_otr_voice_bank.py`, `_otr_delivery_profiles.py`, `_otr_release_gate.py`,
  `_otr_audio_cache.py`, `_otr_engine_profiles.py`, `_otr_resolved_request.py`,
  `_otr_script_prep.py`, and the adapters under `nodes/_otr_audio_engines/`.
- **Wiring guard:** `tests/test_full_workflow_v2_audio_wiring.py` (16 tests).
- **NOT yet implemented (the remaining code):** the opt-in engine adapters' actual
  inference (G1) and the writer cast/stamp removal (Wave 2a tail). Verify the real
  state in the adapter `generate_voice`/`generate_clip` bodies and in
  `OTR_LedgerScriptWriter` before editing -- do not assume.

## Immediate next steps
Continue the progression by coding the remaining sprints, **each one wired into
`full.json` + full regression green + commit (DC cmd)** before moving on.
Headless-codeable now (no GPU needed to write + structure-test):

1. **G1 -- adapter inference (recommended first: additive, does NOT touch the
   legacy byte-identical path).** Fill `generate_voice` (chatterbox, indextts2)
   and `generate_clip` (stable_audio_music) inside the `deterministic_inference`
   context manager, with the plan's seed plumbing (`stable_line_seed_v1`,
   per-engine `engine_seed = _seed_to_int64(engine_name, stable_line_seed)`, a
   bound `torch.Generator`, `model.eval()`, flags/RNG saved+restored in `finally`).
   The opt-in engines stay flag-gated (default-off) until promotion, so this does
   not change the default byte-identical output. Live capture is GPU-gated.
2. **E.3 -- populate delivery profiles** beyond `neutral` in
   `_otr_delivery_profiles.py` (additive; bump DELIVERY_PROFILE_VERSION -> a
   re-baseline trigger, so coordinate).
3. **Wave 2a tail -- writer cast/stamp removal** (move the cast/stamp OUT of
   `OTR_LedgerScriptWriter`, keep only the cheap char_id-subset validator).
   **CAUTION:** this changes the writer's `script_json` bytes, which ARE the legacy
   raw-delegation input -> byte-identity sensitive. The plan says do it WITH the
   R0a render-twice baseline in hand; coding it before R0a means the legacy-audio-
   unchanged claim cannot be verified. Operator decides the timing.

GPU / operator-gated (the operator will NOT queue ComfyUI until the headless
sprints are coded, so these wait): R0a(f) render-twice legacy bit-identity
baseline; R0b live load of `full.json` in ComfyUI Desktop (stub engines); Wave 3
F dep-isolation pilots (IndexTTS2 + Stable Audio 3 are the promotion gate) -> G1
live inference + capture -> I promotion (flip shipped defaults to the best engine
per role, retire the `OTR_ENABLE_*` flags). H (native stereo) deferred.

## Open questions
- Priority order of the headless sprints (G1 adapters vs E.3 vs writer-removal).
- Writer cast/stamp removal timing: code it now (byte-identity unverifiable until
  the R0a GPU baseline) or wait for R0a? Operator call.

---
## Resume instructions
Open a fresh window with the project folder mounted, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps -- continue
the v2.0-alpha audio/voice progression, coding the remaining headless sprints
wired into otr_scifi_16gb_full.json. Verify HEAD + the adapter/writer state first,
then acknowledge when you're ready to start."
