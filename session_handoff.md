# Session Handoff -- OTR v2.0-alpha stable baseline; next = LoRA (P1) + gender (P2) -- 2026-05-30

## Core goal
OTR Visual Drama Engine, v2.0-alpha. This session landed the first fully clean
end-to-end headless baseline (episode "Elusive Dance") under NORMAL_VRAM --
fixing the FLUX/HuMo VRAM thrash that had blocked every prior run -- verified the
output contract, and tagged it stable. Next work is two phases, in order:
**Phase 1** optimize the LTX LoRA structure (consolidate a duplicate distill
loader); **Phase 2** fix cast name/gender incoherence surfaced by the baseline.

## Tech stack & constraints (live state; CLAUDE.md rules still apply, not repeated)
- **Launch at NORMAL_VRAM.** `C:\Users\jeffr\Documents\ComfyUI\_otr_launch.bat` is now
  `--port 8000 --cuda-malloc` (the `--highvram` flag was REMOVED). `--normalvram`
  is an INVALID flag; "normal" = no vram flag at all. This single change fixed
  BUG-291. Do NOT re-add `--highvram`.
- **Stable tag:** `v2.0-alpha-stable-20260530` -> commit `f4d05b9` (annotated,
  additive, pushed). The plain `v2.0-alpha-stable` was deliberately LEFT at the
  old 2026-05-19 commit `e104056` (Jeffrey's call: dated tag only, no force-push).
- **Output contract (hard):** exactly two valid locations -- final deliverable
  `output/otr/obs/<id>_procgen_blended.mp4`, working tree
  `output/otr/episodes/<id>/`. Base pinned via `OTR_OUTPUT_DIR` set in
  `__init__.py` (node-relative -> `Documents\ComfyUI\output`).
- **`Documents\ComfyUI\output` is invisible to Desktop Commander's directory
  lister AND the Linux sandbox mount** (OneDrive path virtualization). But the
  venv python CAN stat/read it. To inspect outputs, run a python script via DC
  (pattern: `scripts/_otr_check_elusive_dance.py`). The ComfyUI history API
  (`GET http://127.0.0.1:8000/history/<prompt_id>`) gives authoritative
  status + output paths.
- **DC monitoring gotchas:** a running ComfyUI's in-place progress bar trips a
  false "waiting for input" in `read_process_output` -- read with a small
  negative `offset` (tail). Inline `python -c "..."` through cmd MANGLES quotes
  (confirmed again this session) -- always write a script file.
- **Validator gate:** `OTR_WorkflowValidator` (node 63) HARD-RAISES on
  widget-vector drift at queue time. Run a validator-only POST after every
  workflow JSON edit: `scripts/_otr_post_validator.py`.

## What's done & decided this session
- **BUG-291 FIXED** (NORMAL_VRAM / DynamicVRAM streaming): FLUX 1.1-1.3 s/it,
  HuMo 14-18 s/it (was ~193 / ~107 pinned). Lever-1 `free_otr_pipeline_residue`
  reclaims reserved 1760->160 MB between phases; PHASE-C-VRAM-PROBE shows ~14.8
  GB free before HuMo loads.
- **BUG-292 FIXED**: output unified under one `Documents\ComfyUI\output\otr` tree.
- **Baseline "Elusive Dance"** (`signal_lost_elusive_dance_20260530_114408`):
  ComfyUI history `status=success`; 77 words / 5 lines; 63.96 s; 1920x1080 @
  24fps; final blend 54.9 MB in `otr/obs`; full episode tree present
  (audio/composited/upscaled/stills/videos b001-b005). Audio C7 preserved
  (BUG-084 was video 0.44 s longer than audio; `-shortest` trims trailing video).
  Story graded B/B+ (intact arc, SFW, on-theme; thin middle at min word count).
- **LoRA finding (git history, all branches):** workflow nodes 60 & 61 have
  ONLY EVER held the same file `ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors`
  (60 @ 0.5, 61 @ 0.2 == 0.7 additive). True duplicate, never a different
  adapter -> consolidate branch (Path A), NOT "restore an intended 2nd LoRA".
- **Cast finding (Phase 2):** name/gender incoherence -- MALIK HIBBERT (male-coded
  name) char_gender=female -> female voice; PHYLLIS OKAFOR (female-coded name)
  char_gender=male -> male voice. Voices correctly follow the assigned gender;
  the NAMES don't. Likely independent randomization of name vs gender.
- Wrote `docs/GO_FORWARD_PLAN_2026-05-30.md` (9-item hardening + LoRA decision
  tree + execution rule) and `docs/ARCH_AUDIT_2026-05-30.md` (keep-list +
  serialization warnings). DECISION: stabilize / document / lightly harden, do
  NOT redesign; keep all fallback/provider/sidecar/VRAM-lever/mirror surfaces.

## State of the art
- HEAD `f4d05b9` on `v2.0-alpha` (origin synced; this session added the tag +
  a docs commit). Working tree otherwise clean except untracked planning docs.
- **LTX LoRA chain** in `workflows/otr_scifi_16gb_full.json`:
  node 54 `LowVRAMCheckpointLoader` -> 60 `LoraLoaderModelOnly` (@0.5; model in =
  link 87, MODEL out = link 102) -> 61 `LoraLoaderModelOnly` (@0.2; model in =
  link 102, MODEL out = link 103) -> 55 `OTR_BatchLTXRender` (model via link 103).
  Both loaders load the identical distilled lora.
- **Cast gen** assigns char_gender independent of the chosen name. Ledger of the
  baseline: c01 ANNOUNCER gender=female / kokoro bf_lily; c02 MALIK HIBBERT
  gender=female / bark v2/en_speaker_4; c03 PHYLLIS OKAFOR gender=male / bark
  v2/en_speaker_3.
- **Machine-local helpers created** (gitignored `scripts/_*.py`):
  `_otr_check_elusive_dance.py` (output-contract verify + history API),
  `_otr_show_elusive_story.py` (treatment + ledger story dump),
  `_otr_cast_check.py` (cast gender vs voice). Reusable for the next baseline.

## Immediate next steps
**PHASE 1 -- LoRA update (Path A consolidation; code warrants it):**
1. Edit `workflows/otr_scifi_16gb_full.json`: set node 60 `widgets_values[1]`
   0.5 -> 0.7; DELETE node 61; repoint link 103's source from node 61 to node 60
   (node 60 `outputs[0].links` -> `[103]`); remove link 102 from the top-level
   `links` array. Resulting chain: 54 -> 60 (@0.7) -> 55.
2. Validator POST (`scripts/_otr_post_validator.py`) -> expect
   `widget_vector_drift=0`, no raise (LoraLoaderModelOnly = 2 widgets, no
   forceInput/seed -> count unchanged).
3. Fixed-seed A/B smoke vs the Elusive Dance baseline (pin `OTR_CAST_SEED` /
   `OTR_STYLE_SEED` env per the true-randomization memory, or re-use the same
   news seed). Compare LTX frames for parity.
4. If parity holds -> commit (one change; the validator is the re-wire check).
   If LTX instead shows a specific weakness -> **Path B**: keep distill @0.7 on
   node 60 and repurpose node 61 to a DIFFERENT adapter (motion / CRT-period /
   character) @0.15-0.3, additive on top of full distillation (do NOT rob the
   distill budget); A/B vs the consolidated version.
5. Per Jeffrey's plan: if the short LoRA smoke works, run a **110-word baseline**.

**PHASE 2 -- cast name/gender review (AFTER Phase 1):**
6. Locate cast-gen (name + gender draw) -- start in `nodes/OTR_LedgerScriptWriter.py`
   (cast/contract passes) and any cast helper; trace how `gender` is set vs how
   the character name is chosen.
7. Choose the fix: (a) gender-aware naming (draw gender, then pick a fitting
   name) or (b) derive gender from the chosen name. Log `BUG-LOCAL-NNN`.
8. Implement + regress (Bug Bible + core + audio byte-identical) + a smoke;
   confirm a male-named character gets a male voice, etc.

**Also pending (low priority, from GO_FORWARD_PLAN):** items 3/6/7/8/9 --
HuMo->LTX edge naming (`wait_for_humo_clips_dir`), `closing_audio` double-mix
check, `LowVRAMCheckpointLoader` unused-CLIP check, route
`scene_sequencer.py:152 DEFAULT_OUT` through `_otr_paths`, refresh stale workflow
metadata. Plus operator action: retire the duplicate AppData install.

## Open questions
- Path A vs Path B is gated on Jeffrey's visual review of the baseline `otr/obs`
  file: does LTX motion/style look good (-> Path A) or weak (-> Path B)?
- Cast-gen fix direction (gender-from-name vs name-from-gender) -- Jeffrey may
  prefer some non-stereotypical pairings, so confirm intent before enforcing.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
