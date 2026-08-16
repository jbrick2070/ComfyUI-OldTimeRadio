# Video Lane Preflight

Run this checklist whenever a video lane is added or materially changed.
Format and acceptance protocol follow `SOURCE_BANK_PREFLIGHT.md` (the house
pattern): every hard item receives `PASS`, `FAIL`, or an explicitly allowed
`N/A`, plus evidence - a file and line, test name, validator output, or
receipt path. Save an `ID | status | evidence` matrix; the final receipt
names that matrix and its SHA-256. Any hard `FAIL` stops the lane. Machine
enforcement lives in `tests/test_lane_preflight_matrix.py` (spec S8c); this
document narrates those checks and is never a substitute for running them -
the `vram-recipe-lab/PREFLIGHT.md` rule.

Every gate below exists because a real lane failed it (2026-08-09/10 audits:
16 defects across 18 lanes; receipts in the lab repo and kibitz-runs/).

## Gate 1 -- Weights resolve

- G1.1 Every declared weight resolves via `folder_paths` or a documented env
  pin; no bare `os.path.exists` on a hardcoded default.
  *Origin: wan_i2v shipped dead - default path absent on this box.*
- G1.2 A missing weight produces a NAMED `EngineUnusable` from
  `assert_usable`, never a swallowed import.
  *Origin: registry imports swallow exceptions; a lane can vanish silently.*

## Gate 2 -- Canvas truth

- G2.1 GPU lanes with a fixed render size declare `render_canvas`; both axes
  /32-legal.
- G2.2 The declaration equals what the graph actually emits.
  *Origin: humo_14B_169 requested 1472x832 and rendered 832x480 - 3.07x.*
- G2.3 Every profile canvas either matches the declaration or the dead
  profile channel is documented for that lane.
  *Origin: nine lanes carry profile canvases read by nothing.*
- G2.4 Derived/intermediate canvases (two-stage halves, upscaler inputs) are
  also /32-legal. *Origin: ia2v stage-A 416x240; 240 % 32 == 16.*

## Gate 3 -- Contract matches runtime

- G3.1 `native_fps == target_fps == 25`; a 24 fps model declares 25 and
  converts at delivery (the Veo/H3 pattern), never relabels.
  *Origin: 192 frames labeled 25 fps = 7.68 s against an 8.00 s audio window.*
- G3.2 Discrete menus in FRAMES, boundaries pinned by test at both ends;
  menu arithmetic derived from the installed node's real limits, not a doc's
  rounded seconds. *Origin: the 107-vs-124 floor correction.*
- G3.3 Continuity declared explicitly on every adapter, never defaulted.
  *Origin: default CONTINUITY_NONE refuses chaining silently.*
- G3.4 Multi-clip partition literals for the lane's menu are pinned as test
  literals derived by running the real `partition_beat`.

## Gate 4 -- Admission honesty

- G4.1 The lane has a QUALIFIED cost row / envelope key, OR its receipts say
  "admission NOT enforced" in words, on disk, reachable in the manifest.
  *Origin: a disqualified row enforced on one path and not the other; four
  lanes with no refusal at all; `vram_admission` written but read by nothing.*
- G4.2 The envelope key states engine, recipe/quant, canvas, frame rung, and
  boot lane; a key miss reports unenforced rather than borrowing a number.

## Gate 5 -- Audio law (V-1)

- G5.1 The adapter's canonicalize path runs `validate_silent_clip_contract`
  on its OWN emitted file. A `has_audio: False` literal is not evidence.
  *Origin: H3 natively produces audio; literals lie.*
- G5.1a A DIRECTORY-CLIP lane satisfies G5.1 through the NAMED twin
  `validate_directory_clip`, which proves every frame is really a PNG/EXR
  from its MAGIC BYTES -- a still image has no audio stream to carry, so the
  silence is a fact about the bytes. The gate is taught that name per lane
  (`DIRECTORY_CLIP_AUDIO_LAW`), never widened to accept any validator, and a
  twin assertion checks the named function actually refuses a mis-named
  non-image. *Origin: `mesh_stage` is the only directory-clip lane; its audio
  check read `has_audio` off the dict the adapter itself wrote, while frames
  were accepted by FILENAME EXTENSION -- so a file named `.png` containing a
  WAV counted as proof of silence (lane 10, 2026-08-11).*
- G5.2 A keeps-audio lane (the standalone music runner) declares a NAMED
  exemption here and never registers into episode assembly without a
  standalone-only boundary.

## Gate 6 -- Guards fire early and by name

- G6.1 Sage-sensitive lanes call `assert_sage_not_patched` inside
  `assert_usable`. *Origin: ltx_8gb shipped with no gate on the exact family
  BUG-070 names.*
- G6.2 Boot requirements are declared and probed against the RUNNING
  server's `comfy.cli_args.args` at ShotLock plan time; render-time checks
  are defence in depth only. *Origin: refusals firing after writer/TTS/
  master-freeze/stills were already paid for.*
- G6.3 Module-scope env reads go through the guarded numeric parser; a
  malformed env var must not delete the lane from the registry.
  *Origin: OTR_LTX_AV_RESERVE_VRAM_GB deleted ltx_audio_in, silently.*

## Gate 7 -- Public surface

- G7.1 Exactly one live menu id per internal engine
  (`exact_menu_option_for` proves 1:1); legacy aliases resolve via
  `_LEGACY_ENGINE_ALIASES` and never render as menu options.
- G7.2 Node-87 / variant workflow strings are GENERATED, never hand-typed;
  variants regenerate in the same commit as any profile change.
- G7.3 `ENGINE_MATRIX.md` regenerated in the same commit as ANY
  canvas/contract/registration change (the doc is a live drift gate).
- G7.4 `still_plan` declared and audit-clean; naming states what the lane
  is: audio-conditioned lanes say `audio_in`, portrait lanes say `portrait`,
  the `low`/`high` marker comes from a measurement receipt, never a guess.

## Gate 8 -- Solo smoke

- G8.1 One real render on the lane's declared boot lane: canvas probed,
  frame count exact, silence probed (or audio present for a G5.2-exempt
  lane), VRAM peak receipted, trim ratio logged when tail-trim fired.

## Receipt

`VIDEO_LANE_PREFLIGHT receipt: <lane> | <date> | matrix sha256 <...> |
suite run <test output path> | smoke receipt <path> | verdict PASS/FAIL`

## The family (create each sibling when its subsystem is next touched,
never as an empty paper checklist)

- `SOURCE_BANK_PREFLIGHT.md` - exists (the format authority).
- `VIDEO_LANE_PREFLIGHT.md` - this file; enforced by the S8c suite.
- `TTS_VOICE_PREFLIGHT.md` - exists (2026-08-16); enforced by
  `tests/test_tts_voice_preflight_matrix.py`. Seeded from the cross-engine
  Lemmy work, so its gates are the ones that actually bit: a degraded dropdown
  two engines short, a generator one command from deleting rows it could not
  recreate, and a route tier that would have raised at render time.
- Future, each backed by its own enforcement code before the doc is written:
  `STILL_LANE_PREFLIGHT.md`, `MUSIC_AUDIO_PREFLIGHT.md`,
  `LLM_WRITER_PREFLIGHT.md`, `UPSCALER_PREFLIGHT.md`. Seed their gates from
  this file's shape plus the Bug Bible's per-subsystem entries.
