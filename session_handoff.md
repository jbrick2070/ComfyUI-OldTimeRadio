# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-24

## Core goal
The HuMo VRAM-thrash (BUG-LOCAL-265) is fixed and shipped -- Option C: HuMo-1.7B
as the default, HuMo-17B/14B kept opt-in, via a new `OTR_HuMoTierLoader` node,
plus the Lever-1 pipeline-residue free. The next session's job is to support the
operator running ONE real episode to capture the `PHASE-C-VRAM-PROBE` telemetry
that confirms Lever-1 actually reclaims VRAM, and to act on whatever that run
surfaces.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP + BUG_LOG
auto-load -- not repeated here. Operational rules that bite:
- **Git: Desktop Commander cmd shell only.** Commit message via the file tool to
  `.git\COMMIT_EDITMSG`, then `git commit -F`. Never PowerShell for git.
- **The Linux sandbox (`mcp__workspace__bash`) serves a STALE copy of existing
  repo files** -- this session it served an outdated `batch_humo_render.py` with
  a phantom syntax error. Use Desktop Commander (Windows FS) for all
  `py_compile` / pytest / git / file-state checks. The Read/Write/Edit file
  tools hit the real FS and are fine.
- **pytest:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m
  pytest`, run via Desktop Commander cmd with output redirected to a log
  (`> log 2>&1`), then read the log. Note: cmd expands `%ERRORLEVEL%` at parse
  time, so `& echo %ERRORLEVEL%` after a redirect is meaningless -- read the
  log's summary line instead.
- VRAM-budget / model-choice changes are round-robin gated per CLAUDE.md.

## What's done & decided
- **BUG-LOCAL-265 RESOLVED -- Option C shipped.** HEAD `e619ebb`, `v2.0-alpha`,
  local == origin. Three commits this session:
  - `e981db0` -- smoke-workflow ratio edits + the round-robin problem-statement
    doc (`docs/2026-05-24-humo-model-choice__00_question.md`).
  - `09c2d49` -- the feature: `OTR_HuMoTierLoader` + Lever-1 + workflow rewire +
    estimate-text update + `BUG_LOG.md`.
  - `e619ebb` -- flaky cast-routing test fix (`force_lemmy=False`).
- **New node `OTR_HuMoTierLoader`** (`nodes/_otr_humo_tier_loader.py`, registered
  in `__init__.py`): one upstream loader, three tiers -- `low_vram_default`
  (HuMo-1.7B fp16, 20 steps, cfg 5.0, no distill LoRA -- the shipped default),
  `high_quality` (17B/14B fp8, 6 steps, cfg 1.0, lightx2v distill LoRA -- opt-in),
  `experimental_gguf` (advanced only). Hard auto-downgrade rule: a high tier with
  free VRAM below `vram_safety_threshold_gb` (default 10 GB) downgrades to 1.7B
  or stops with a clear error.
- **Lever 1** (`nodes/_otr_vram_levers.py::free_otr_pipeline_residue`): frees the
  writer-LLM + Bark out-of-band caches `unload_all_models()` cannot see, then the
  ComfyUI unload + CUDA flush. Wired into `BatchHumoRender`'s inter-phase cleanup
  and `OTR_HuMoTierLoader` pre-load. `OTR_UnloadAll` extended to drop Bark.
- **`workflows/otr_scifi_16gb_full.json` rewired:** the 6-node HuMo loader chain
  (nodes 45-50) collapsed into one `OTR_HuMoTierLoader` (node 72) feeding
  `OTR_BatchHumoRender` (node 51); steps/cfg socket-driven; default tier
  `low_vram_default`. The FLUX->UnloadAll->HuMo gate is intact and extended --
  `OTR_UnloadAll.unload_done` (node 24) also gates node 72.
- `batch_humo_render.py` pre-batch estimate updated ~10-12 min -> ~4:23/clip.
- Regression GREEN: full `tests/` walk 2617 passed / 21 skipped / 0 failed;
  Bug Bible 23 passed / 1 skipped / 2 xfailed; `tests/test_humo_tier_loader.py`
  20 passed.
- **Rejected:** 17B-as-default (thrashes in-pipeline); GGUF-as-default (per-step
  dequant tax made it slower than fp8); a model-tier widget *inside*
  `OTR_BatchHumoRender` (tiering goes upstream in the loader -- the renderer
  keeps its clean pre-loaded-inputs surface).

## State of the art
- New files: `nodes/_otr_humo_tier_loader.py`, `nodes/_otr_vram_levers.py`,
  `tests/test_humo_tier_loader.py` -- all committed.
- Modified + committed: `__init__.py`, `nodes/batch_humo_render.py`,
  `visual/unload_all.py`, `workflows/otr_scifi_16gb_full.json`,
  `tests/test_humo_logs_e10.py`, `tests/test_helper_paired_signatures.py`,
  `tests/test_lock_cast_routing.py`, `BUG_LOG.md`, `ROADMAP.md`.
- Production workflow `otr_scifi_16gb_full.json`: 31 nodes / 69 links;
  `last_node_id=72`, `last_link_id=217`. HuMo branch:
  `OTR_HuMoTierLoader` (node 72, `tier=low_vram_default`) -> `OTR_BatchHumoRender`
  (node 51). No `UnetLoaderGGUF` node remains in the JSON.
- `PHASE-C-VRAM-PROBE` telemetry is in `nodes/batch_humo_render.py` -- logs torch
  allocated/reserved, real CUDA free/total, and the ComfyUI-tracked model list
  right after the inter-phase free, before HuMo Phase C. `[VRAMLevers]
  free_otr_pipeline_residue ...` logs the residue free itself.
- **Parked-dirty, do NOT commit:** `docs/s28_diff_tmp.txt` (left over from a
  prior session). `session_handoff.md` is untracked by design.

## Immediate next steps
1. Operator runs ONE real OTR episode end-to-end on the default workflow
   (`workflows/otr_scifi_16gb_full.json`, `OTR_HuMoTierLoader` tier
   `low_vram_default`). This is the BUG-265 verification run. Operator pastes the
   full ComfyUI console output covering the HuMo phase.
2. In that console, find the `[VRAMLevers] free_otr_pipeline_residue
   (BatchHumoRender inter-phase)` line and the `[BatchHumoRender]
   PHASE-C-VRAM-PROBE` line. Confirm: (a) `free_otr_pipeline_residue` ran every
   step (unload_llm, _unload_bark, unload_all_models, soft_empty_cache, cuda
   flush) with no `steps_failed`; (b) the probe shows HuMo getting a clean VRAM
   budget at Phase C entry; (c) the 1.7B HuMo phase renders near the bare-smoke
   pace, not 140-279 s/it.
3. If the probe confirms the ~14 GB residue is actually reclaimed: mark
   BUG-LOCAL-265 fully verified in `BUG_LOG.md` and promote it to the Bug Bible
   (Three-File Contract -- `BUG_BIBLE.yaml` + `README.md` + a regression test in
   the survival-guide repo). If the residue persists (CUDA allocator holding
   reserved blocks despite the unload): that is a separate follow-up VRAM fix,
   and it only affects the 17B opt-in tier -- the 1.7B default is unaffected.
4. (Optional, separate) operator A/B quality call: HuMo-1.7B 20-step output vs
   14B -- `clip_00003_.mp4` vs `clip_00004_.mp4` in the
   `output/old_time_radio/humo_test/` folder.

## Open questions
- Does `free_otr_pipeline_residue` actually reclaim the ~14 GB residue, or does
  the CUDA allocator hold reserved blocks after the unload? Only a real-episode
  `PHASE-C-VRAM-PROBE` answers it. Relevant only to the 17B opt-in tier.
- HuMo-1.7B lip-sync quality with real FLUX reference portraits (the smoke used a
  generic test image) -- expected to improve, unverified.
- BUG-LOCAL-264 (`news_interpreter` `NewsBriefs` schema overrun with gemma-2-2b)
  remains open, unrelated to HuMo -- see BUG_LOG.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
