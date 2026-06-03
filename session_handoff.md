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
  into it (CLAUDE.md rule 3). Do NOT create a second / opt-in workflow json.
- **Tests + git + any workflow-JSON parsing run on the WINDOWS HOST**, never the
  Linux sandbox (no torch; stale CRLF mount). venv python
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`; git stage/commit/
  push via Desktop Commander `cmd` (never PowerShell for git; never the GitHub
  connector). Full regression after every change (redirect to a file, then read):
  `...python.exe -m pytest -q -p no:cacheprovider`. The conftest known-fail guard
  hard-exits `SystemExit(2)` on ANY new failure -- the suite must be fully green.
- **ASCII-only `.py` source, no em-dash, no BOM.** Audio is king (byte-identical
  legacy fallback). Never the word "dummy".
- Commit message via `.git\COMMIT_EDITMSG` + `git commit -F` (cmd mangles `-m`).
  `git add` explicit paths or `-u`, NEVER `-A` (untracked planning docs stay out).

## Spec SSOT (read, don't re-derive)
`docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` is the authoritative
wave order, invariants I-1..I-11, ComfyUI C-1..C-7, ResolvedVoiceRequest fields,
per-sprint tests, and re-baseline triggers. This handoff captures only the LIVE
deltas on top of it.

## What's done & decided
- **Operator correction (do not re-litigate):** single json of record. The audio
  lane is wired directly into `full.json`; never reintroduce a 2nd json.
- **`full.json` carries the audio lane** (`a46763b`): CastLock(80) +
  BatchCharacterVoices(81) + AnnouncerVoice(82) + StableAudioTheme(83) wired in;
  legacy bark/kokoro/musicgen/audiogen instances removed; SFX dropped. Engine
  widgets default to the legacy engines so output is byte-identical by default.
- **Node classes + libraries shipped + registered** (Wave 0 contracts, Wave 1
  nodes 1a-1g, Wave 2a CastLock).
- **Headless G1-prep + G1 determinism + E.3 lock landed THIS session (3 commits,
  pushed to origin/v2.0-alpha):**
  - `bf949a9` **G1-prep:** `scripts/otr_audio_dep_pilot.py` -- the headless F
    dependency-pilot harness (per-engine subprocess-isolated import probe;
    snapshots torch + xformers/flash_attn before/after import and FAILS on a
    torch swap or a banned-dep pull; encodes each engine's `assumed_call` as the
    SSOT the adapter TODO-for-F comments point at; introspects external-generator
    support; OFFLINE; diagnostic-only, never flips a default). All three engine
    forwards (chatterbox/indextts2/stable_audio_music) are now CONSISTENT
    flag-gated stubs that raise, each documenting its assumed GPU call as
    TODO-for-F; the chatterbox provisional `.generate()` blind call was removed.
    `supports_external_generator` stays False until F verifies it. +10 tests.
  - `22de2e7` **G1 determinism:** the opt-in per-line voice forward (character +
    announcer, shared dispatch) and the per-cue theme forward now run inside
    `deterministic_inference(engine_seed, warn_only=True)` -- non-strict so a
    nondeterministic CUDA op can't crash the opt-in render on sm_120; seeds +
    restores all RNG/flags. Legacy byte-identical BATCH delegation is NOT wrapped
    (I-3), pinned by a source guard. +6 tests.
  - `780c0df` **E.3 lock:** regression PINS the neutral-only delivery-profile
    contract (only `neutral`; `DELIVERY_PROFILE_VERSION`/`PROJECTION_VERSION`
    pinned "1"; identity projection/overlay; id+version in the IN_KEY). The
    module was already complete -- populated profiles remain v2.1 + a re-baseline
    trigger, deliberately NOT done. +8 tests.
- **Green:** full tests/ **3719 passed, 12 skipped, 0 failed** (was 3695; +24).
- **No node surface changed** -> `full.json` untouched; wiring guard still green.

## State of the art
- **HEAD = `780c0df` == origin/v2.0-alpha** (verify with `git rev-parse HEAD` vs
  `origin/v2.0-alpha`). Recent: `780c0df` E.3 lock, `22de2e7` G1 determinism
  wrap, `bf949a9` G1-prep harness, `00c6c1e` prior handoff.
- **`workflows/otr_scifi_16gb_full.json`** -- 29 nodes, the migrated graph
  (unchanged this session; no node surface touched).
- **Node files:** `nodes/cast_lock.py`, `batch_character_voices.py`,
  `announcer_voice.py`, `stable_audio_theme.py`, shared base
  `nodes/_otr_voice_node_common.py`. Determinism: `nodes/_otr_determinism.py`
  (`_seed_to_int64`, `deterministic_inference`). Adapters under
  `nodes/_otr_audio_engines/` (base + registry + eng_*). New harness:
  `scripts/otr_audio_dep_pilot.py`.
- **NOT yet done (GPU / operator-gated):** the three real engine forward BODIES
  (G1 live inference); R0a render-twice legacy bit-identity baseline; writer
  cast/stamp removal (byte-identity sensitive -- needs R0a first); E.3 populate
  (-> v2.1). The determinism wrap + per-engine seed plumbing are ALREADY in place
  around the forwards, so G1 live inference is "fill the body + flip the flag".

## Immediate next steps
The headless-codeable audio sprints are DONE + pushed. What remains is GPU /
operator-gated:
1. **Run the F dependency pilot on the 5080:** install chatterbox / indextts2 /
   stable-audio-tools each in its OWN venv, then
   `...\python.exe scripts\otr_audio_dep_pilot.py --json` (or `--python <venv>`
   per engine for true venv isolation). Read each verdict: import clean? torch
   unchanged? no xformers / flash_attn? Confirm the `assumed_call` + that the
   forward binds a `torch.Generator`. THIS is the G1 unblock.
2. **G1 live inference (after F):** fill `generate_voice` / `generate_clip` per
   the verified signatures, flip `supports_external_generator=True`, capture
   render-twice bit-identity. (Seed plumbing + the determinism wrap are done.)
3. **R0a baseline (operator GPU):** render-twice legacy bit-identity -- needed
   before the writer cast/stamp removal can claim legacy-audio-unchanged.
4. **E.3 populate -> v2.1** (re-baseline trigger; coordinate; the lock test will
   force the bump to be deliberate).

## Open questions
- Order of the GPU sprints (F pilot vs R0a baseline first).
- Writer cast/stamp removal timing (still needs the R0a baseline in hand).

---
## Resume instructions
Open a fresh window with the project folder mounted, attach this file, and say:
"Read this handoff and continue the v2.0-alpha audio/voice progression. The
headless sprints (F harness, G1 determinism wrap, E.3 lock) are committed +
pushed at HEAD 780c0df. Verify HEAD first; the remaining work (G1 live inference,
R0a baseline, writer-removal, E.3 populate) is GPU/operator-gated -- tell me
which GPU sprint you want to run, or have me prep more headless work."
