# OTR Docs

This folder is intentionally small. Current operator planning lives in
`GO_FORWARD_PLAN.md`. There is no separate roadmap: `ROADMAP.md` was
deleted 2026-09-05 because it had gone dangerously stale -- it still listed
the dead-code campaign as pending release-runway row 1 the day after that
campaign finished, and four of the five files it pointed a reader at were
not in the published bundle. Long-horizon work lives in GO_FORWARD's own
design rows; finished campaigns live in git history and `HANDOFF_LOG.md`.

## Current Setup Docs

- `openrouter-setup.md` - optional BYO OpenRouter LLM setup.
- `comfy-credits-setup.md` - optional Comfy Credits / Partner API setup.
- `gemma4-gguf-native-setup.md` and `gemma4/` - local Gemma GGUF notes.

## The video model reference (read both before adding or changing an engine)

- `ENGINE_MATRIX.md` - **every per-model number**: clip window, frame ladder,
  continuity, join mode, segment counts, effective canvas. GENERATED from the
  live registry and DRIFT-GATED - `python tools/engine_matrix.py --check` is a
  suite test, so it cannot disagree with the adapters.
- `2026-08-02-FINAL-all-engine-maths-and-stills.md` - what a generator cannot
  derive: still logic and the local/cloud re-mint split, the fix list with a
  verified per-item status, the open decisions, and the padding rule.

**The rule between them: a hand-maintained doc must never re-type a number the
generated one already owns.** On 2026-08-06 the hand-written tables were found
claiming 3 and 10 segments for HuMo where the live registry said 5 - a ceiling
that had moved four days earlier - while the drift-gated matrix had been correct
throughout. Cite the generated matrix; do not copy it.

## Current Project Docs

- `GO_FORWARD_PLAN.md` - current sprint and next sprint only.
- `multimodal-story-schema/` - active source-pack and story-schema work.
- `model-license-*.md` plus `model-license-audit-targets.txt` - model license
  audit records used by tests.
- `conventions.md` - naming and module-shape rules enforced by tests.
- `MODEL_INVENTORY.md` - the full weight-file list under `C:\ComfyUI-Models`,
  what references each file, and the disk-reclaim analysis. Regenerate it
  when the models root changes materially; it exists so a space audit is
  never re-derived from scratch.

Historical sprint plans, dead smoke harness notes, and one-off setup experiments
should not live here unless a current test, README, or operator handoff points to
them.
