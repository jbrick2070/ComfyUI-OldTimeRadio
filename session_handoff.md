# Session Handoff -- OTR v2.0-alpha (audio/voice progression) -- 2026-06-02

## Core goal
Ongoing **v2.0-alpha** build of OldTimeRadio: a model-agnostic, per-role audio
engine registry + voice-casting subsystem (character voice / announcer / theme
music selectable per role; deterministic voice bank + caster; post-freeze
cast-lock; frozen ResolvedVoiceRequest identity/cache contract). Everything is
wired into the ONE workflow of record, `workflows/otr_scifi_16gb_full.json`.
Legacy audio is a permanent byte-identical fallback (new nodes delegate to it by
default). This session advanced the GPU-gated G1 work as far as it can go without
a GPU.

## Tech stack & constraints (the ones that bite -- full set in CLAUDE.md)
- Python 3.12 + torch, Windows, RTX 5080 16 GB. Branch `v2.0-alpha`, never `main`.
- **ONE json of record** = `workflows/otr_scifi_16gb_full.json`. No second/opt-in
  json (operator rejected the 2-file design). Wire every node-surface change into
  it (CLAUDE.md rule 3).
- **Tests + git run on the WINDOWS HOST**, never the Linux sandbox (no torch;
  stale mount). venv: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
  Git stage/commit/push via **Desktop Commander `cmd`** (never PowerShell for git;
  never the GitHub connector). File I/O on repo files via Desktop Commander too
  (the cowork mount can be stale).
- Full regression after every change: `...python.exe -m pytest -q -p no:cacheprovider`
  (redirect to a file, read the tail). conftest known-fail guard `SystemExit(2)`
  on ANY new failure -- suite must be fully green. Commit msg via
  `.git\COMMIT_EDITMSG` + `git commit -F` (cmd mangles `-m`); `git add` explicit
  paths, NEVER `-A`. ASCII-only `.py` source, no em-dash, no BOM.
- **Audio is king** -- legacy byte-identical output must not change. `base.py`
  rule: `supports_external_generator` flips True ONLY after the F GPU pilot
  verifies an engine's forward binds a `torch.Generator`.

## Spec SSOT (read, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` -- wave order, invariants
I-1..I-11, ComfyUI C-1..C-7, ResolvedVoiceRequest fields, per-sprint tests,
re-baseline triggers. This handoff is only the LIVE deltas on top of it.

## What's done & decided (this session -- all committed + pushed)
- `bf949a9` **G1-prep / F harness:** `scripts/otr_audio_dep_pilot.py` -- headless,
  per-engine subprocess-isolated import probe (chatterbox / indextts2 /
  stable_audio_music). Snapshots torch + xformers/flash_attn before/after import;
  FAILS on a torch swap or banned-dep pull (plan F). Encodes each engine's
  `assumed_call` as the SSOT the adapter `TODO-for-F` comments point at;
  introspects external-generator support; OFFLINE; diagnostic-only (never flips a
  default). All three engine forwards are now CONSISTENT flag-gated stubs that
  raise, each documenting its assumed GPU call as `TODO-for-F`; the chatterbox
  provisional `.generate()` blind call was removed. +10 tests.
- `22de2e7` **G1 determinism wrap:** the opt-in per-line voice forward (character
  + announcer, in shared `nodes/_otr_voice_node_common._render_per_line`) and the
  per-cue theme forward (`nodes/stable_audio_theme._render_clips`) now run inside
  `deterministic_inference(engine_seed, warn_only=True)` -- non-strict so a
  nondeterministic CUDA op can't crash the opt-in render on sm_120; seeds +
  restores all RNG/flags in `finally`. Legacy byte-identical BATCH delegation is
  NOT wrapped (I-3), pinned by a source guard. +6 tests.
- `780c0df` **E.3 lock:** `tests/test_delivery_profiles.py` PINS the neutral-only
  delivery-profile contract (only `neutral`; versions pinned "1"; identity
  projection/overlay; id+version in the IN_KEY). `_otr_delivery_profiles.py` was
  already complete -- populated profiles stay v2.1 + a re-baseline trigger,
  deliberately NOT done. +8 tests.
- **Rejected / not done on purpose:** writing blind GPU inference into the three
  forwards (gated on F); populating delivery profiles (v2.1 + re-baseline);
  touching the legacy byte-identical path.

## State of the art
- **HEAD == origin/v2.0-alpha** (verify: `git rev-parse HEAD` vs
  `git rev-parse origin/v2.0-alpha`). Pushed commits, newest first: this handoff
  refresh, then `780c0df` E.3 lock, `22de2e7` G1 determinism wrap, `bf949a9`
  G1-prep harness, `00c6c1e` prior handoff.
- **Full suite GREEN: 3719 passed, 12 skipped, 0 failed** (was 3695; +24 tests).
- **No node surface changed** this session -> `full.json` (29 nodes) untouched;
  the wiring guard `tests/test_full_workflow_v2_audio_wiring.py` is green.
- Key files: `scripts/otr_audio_dep_pilot.py` (new harness);
  `nodes/_otr_voice_node_common.py` + `nodes/stable_audio_theme.py` (forwards now
  wrapped in `deterministic_inference`); `nodes/_otr_audio_engines/eng_chatterbox.py`
  / `eng_indextts2.py` / `eng_stable_audio.py` (consistent TODO-for-F stubs);
  `nodes/_otr_determinism.py` (`_seed_to_int64`, `deterministic_inference` -- both
  pre-existing); `tests/test_audio_dep_pilot.py` / `test_audio_determinism_wrap.py`
  / `test_delivery_profiles.py` (new).
- **Already in place around the forwards:** per-engine seed derivation
  (`engine_seed = _seed_to_int64(engine, request.stable_line_seed)`) + the
  determinism wrap. So G1 live inference is now "fill the body + flip the flag".

## Immediate next steps (all GPU / operator-gated -- pick one to run)
1. **Run the F dependency pilot on the 5080 (THE G1 unblock):** install
   chatterbox / indextts2 / stable-audio-tools each in its OWN venv, then
   `...\python.exe scripts\otr_audio_dep_pilot.py --json` (or `--python <venv>`
   per engine). Read each verdict: import clean? torch unchanged? no xformers /
   flash_attn? Confirm the `assumed_call` + that the forward binds a
   `torch.Generator`.
2. **G1 live inference (after F):** fill `generate_voice` / `generate_clip` in
   `eng_chatterbox` / `eng_indextts2` / `eng_stable_audio` per the verified
   signatures, flip `supports_external_generator=True`, capture render-twice
   bit-identity. (Seed plumbing + determinism wrap already done.)
3. **R0a baseline (operator GPU):** render-twice legacy bit-identity -- required
   before the writer cast/stamp removal can claim legacy-audio-unchanged.
4. **Writer cast/stamp removal (Wave 2a tail):** byte-identity sensitive (it
   changes the writer's `script_json` = legacy raw-delegation input) -- do it WITH
   the R0a baseline in hand, not before.
5. **E.3 populate -> v2.1:** re-baseline trigger; the lock test forces the version
   bump to be deliberate. Coordinate.

## Open questions
- Order of the GPU sprints (F pilot vs R0a baseline first).
- Writer cast/stamp removal timing (needs the R0a baseline first).

---
## Resume instructions
Open a fresh window with the project folder mounted, attach this file, and say:
"Read this handoff and continue the v2.0-alpha audio/voice progression. Verify
HEAD == origin/v2.0-alpha first. The headless sprints (F harness, G1 determinism
wrap, E.3 lock) are committed + pushed and the suite is green at 3719. The
remaining work is GPU/operator-gated -- tell me which GPU sprint to run (F pilot
is the G1 unblock), or have me prep more headless work. Acknowledge when ready."
