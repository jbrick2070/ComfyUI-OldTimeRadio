# Session Handoff -- ComfyUI-OldTimeRadio (OTR) -- 2026-06-05 (voice overhaul -> chatterbox)

## Core goal
This session fixed and hardened the **character-voice subsystem** end-to-end:
killed a render-crashing bug, made **IndexTTS2 the actual default voice for every
character (no more silent bark fallback)**, and expanded the CC0 reference bank
from 4 to 36 voices with dynamic gender->voice casting. It is DONE, committed, and
live-validated. **The next mission is a CHATTERBOX engine via a dep-isolated
sidecar venv** (roundtable it, wire it into the workflow, test it) -- with Dia
(Apache-2.0, commercially clean) as the recommended alternative.

## Tech stack & constraints (session-specific; CLAUDE.md + memory auto-load the rest)
- ComfyUI Desktop on :8000, Windows, RTX 5080 16GB (Blackwell sm_120), venv
  `C:\Users\jeffr\Documents\ComfyUI\.venv` (torch 2.10+cu130, numpy 2.4.4).
- `.py` edits need a ComfyUI RESTART to load (module cache). The voice bank JSON
  HOT-RELOADS (load_voice_bank caches by content sha) -- no restart for bank edits.
- Git via Desktop Commander (`git -C <repo> ...` with Start-Process
  `-RedirectStandardOutput` to a file; DC/PowerShell does not capture external-exe
  stdout). **HEAD = 62792fd on v2.0-alpha, [ahead 7] of origin -- NOT PUSHED.**
- `datasets` 5.0.0 was pip-installed this session (+ pyarrow/dill/multiprocess;
  torch/numpy unchanged). HF audio must be read with `Audio(decode=False)` + soundfile
  -- the `torchcodec` decode backend is NOT installed on purpose (it breaks this stack).
- Refs live OUTSIDE git at `C:\ComfyUI-Models\TTS\refs\indextts2\*.wav` (model assets).
- Reuse the existing scratch pattern: write a `scripts/_otr_*.py` that writes its OWN
  result file, run via the venv python, read the file (DC stdout capture is flaky).
  `scripts/_otr_*` is scratch/gitignored EXCEPT the tracked `_otr_indextts2_worker.py`.

## What's done & decided (all on v2.0-alpha, suite 3758/0 throughout)
- **eda8590** mixed-rate crash: `OTR_BatchCharacterVoices` died on
  `pack_audio_batch: mixed sample rates [22050, 24000]` when a cast mixed
  indextts2 (22050) + bark-fallback (24000). Fix: `resample_audio()` in
  `nodes/_otr_audio_utils.py` (scipy.signal.resample_poly -- the I-11 resampler,
  NOT torchaudio) downsamples the bark fallback clip to the primary `sr` in the
  fallback branch of `_otr_voice_node_common._render_per_line`. Plus a
  finally-unload: bark fallback adapter stashed on `self._bark_fallback_active`,
  torn down in `generate()`'s finally.
- **4dbbc3e** the canonical workflow `workflows/otr_scifi_16gb_full.json` node 80
  (OTR_CastLock) flipped to `[default, auto_registry, neutral, true]` -- the old
  `preserve_ledger` default never assigned index refs, forcing bark. (Schema
  default stays `preserve_ledger` for the byte-safe legacy bark path;
  `test_full_workflow_v2_audio_wiring.py::test_widget_vectors_exact` updated to
  expect the node-80 override.)
- **39f384a** gender-agnostic last resort in `_resolve_clone_ref_path`
  (`_otr_voice_node_common.py`): when gender is empty/out-of-bank (writer emits
  `gender='other'`), pick ANY index ref deterministically (seeded by char_id) so a
  clone engine NEVER silently drops to bark. Male/female keep gender-correct picks.
- **382ff86 / c9c8d8a / 8ae8fd6 / 62792fd** -- `scripts/otr_dl_indextts2_refs.py`
  (the reusable downloader: `--source donations|lj-speech|rhasspy`, F0 gender
  tagging, ~12s mono refs, sha, wires `config/voice_reference_bank.json`,
  idempotent + `--dry-run`). Bank is now **36 indextts2 voices: 14 male / 22
  female**, all CC0/PD (kyutai voice-donations CC0 + LJ Speech PD + Rhasspy
  Kathleen/Kerstin CC0 via blob-less GitHub sparse-checkout).
- Rejected: torchaudio resample (codebase uses scipy, I-11); torchcodec install;
  Common Voice (every mirror is script-based/gated/torchcodec on datasets 5.0);
  changing the schema default (only the workflow node was flipped).

## State of the art
- **Live out-of-the-box test PASSED** (ComfyUI restarted 15:52, after the fixes):
  a 6-char cast (3F/3M) + 1 `gender='other'` (HAYES WELLS) + female announcer ->
  every character voiced on IndexTTS2, **`Bark loaded` = 0**, full render to
  `output\otr\episodes\signal_lost_toolwielding_tentacles_20260605_155914\...mp4`
  (68.5s). Headless: a 9-char diverse cast maps 8/9 distinct gender-correct, 0 bark.
- Engines + per-engine I/O contract (the seam for a NEW engine):
  adapter `voice_ref_field` tells the dispatch which cast field feeds it --
  indextts2/chatterbox=`voice_ref_path` (clip), kokoro=`voice_ref_id`,
  bark=`voice_preset`. Rates: indextts2 22050, bark/chatterbox 24000, music 44100,
  SceneSequencer standardizes batches to 48000. `_OTR_CLONE_ENGINES =
  ("indextts2","chatterbox")` is a HARD-CODED tuple in `_otr_voice_node_common.py`.
- **Chatterbox today:** wired (`nodes/_otr_audio_engines/eng_chatterbox.py`, 24000,
  in the engine dropdown + 4 bank entries) but UNUSABLE -- its deps hard-pin
  torch2.6/numpy1.26 which brick the torch2.10/cu130 venv, AND 0 chatterbox ref
  WAVs are installed (bank has 4, disk has 0). Chatterbox license = MIT.
- **Casting-architecture roundtable MUST-FIX backlog (staged, NOT applied)** in
  `docs/2026-06-05-voice-casting-architecture/pass01_plan.md` -- directly relevant
  to adding any new engine: (1) CastLock `_stamp` writes `voice_ref_id` not
  `voice_ref_path`; (2) commercial_clean EFFECTIVE = engine AND ref
  (`eng_indextts2.commercial_clean=False` -- bilibili NON-COMMERCIAL -- vs CC0
  refs=true); (3) gender guaranteed on the cast row; (4) kokoro `ANNOUNCER_VOICE_POOL`
  has 4 voices, only `bm_george` installed; (5) resample EVERY clip not just bark;
  (6) replace `_OTR_CLONE_ENGINES` tuple with adapter metadata
  (`requires_voice_ref` / `voice_ref_kind` / `missing_ref_fallback`).

## Immediate next steps (the chatterbox mission)
1. **Roundtable the chatterbox-sidecar design** (use the `roundtable` skill;
   dry-run estimate first, then live -- panel = GPT+Gemini+Grok+DeepSeek,
   `--reasoning-effort none --max-tokens 12000`, Opus is the judge). Ground it
   against `nodes/_otr_audio_engines/eng_chatterbox.py`, the existing IndexTTS2
   sidecar (`scripts/_otr_indextts2_worker.py` + `docs/indextts2_pathb_setup.md`),
   and the per-engine contract in `nodes/_otr_voice_node_common.py`. Core question:
   isolate chatterbox's torch2.6/numpy1.26 deps in a SEPARATE venv reached by a
   stdin/stdout JSON worker (exactly the IndexTTS2 Path-B pattern, env vars
   `OTR_CHATTERBOX_VENV/_DIR/_WORKER`), so the main torch2.10 venv is never
   touched. Decide worker protocol, ref-clip handling, and whether to first land
   the `_OTR_CLONE_ENGINES`-tuple -> adapter-metadata refactor (casting roundtable
   MUST-FIX #6) so chatterbox (and later Dia) slot in without per-engine `if`s.
2. **Build it**: chatterbox sidecar venv + worker + adapter wiring; then INSTALL
   chatterbox reference WAVs -- the bank already lists 4 chatterbox entries
   (`cc_male_warm/cc_male_gravel/cc_female_warm/cc_female_bright`) but 0 are on
   disk. Either source CC0 chatterbox refs (the kyutai voice-zero / donation WAVs
   work for any clone engine -- re-tag `engine="chatterbox"` copies, or extend
   `otr_dl_indextts2_refs.py` with an `--engine` arg) or generate them.
3. **Wire + test**: set node 81 `engine=chatterbox` (and node 80
   `voice_bank=default`), render a small cast, confirm it voices via the sidecar
   and the main venv stays intact (the live `_otr_index_test.py`-style harness +
   the `_otr_voicewatch.py` console grep are the pattern; recreate them as scratch).
   Run the full suite (`pytest tests -q`) + Bug Bible after any `.py` change.
4. **Strongly consider Dia instead/alongside** (`nari-labs/dia`, 1.6B, Apache-2.0
   -> COMMERCIALLY CLEAN, fixes the IndexTTS2 non-commercial-license liability for
   Jeffrey's films; zero-shot cloning so the 36-voice CC0 bank feeds it directly;
   dialogue-native). Same sidecar + new `eng_dia.py` adapter. If the goal is a
   shippable commercial voice, Dia > chatterbox.

## Open questions
- **Chatterbox vs Dia**: chatterbox is already-wired (MIT) but bricks the venv and
  has 0 refs; Dia is a new adapter but Apache-2.0 (commercial-clean) and reuses the
  bank. Jeffrey asked for chatterbox this session -- confirm priority before the build.
- Apply the casting MUST-FIX backlog (esp. `_OTR_CLONE_ENGINES` -> metadata, and
  commercial_clean-effective) as part of the new-engine work, or separately?
- Push the 7 unpushed commits to origin/v2.0-alpha (Jeffrey hadn't decided).
- The 5 `workflows/GO_FORWARD_PLAN_v7-v11_*.md` deletions are still unstaged
  (pre-existing, not from this session) -- commit-the-deletion vs restore.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
