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

## Apple Silicon

A rented Mac mini M4 (16 GB) was used to port this pack to Metal. Start with the
compliance matrix if you want the answer for one engine; start with the guide if
you are setting a Mac up.

- `MAC_COMPLIANCE_MATRIX.md` - **every engine, classified**: PROVEN / LIKELY /
  OOM RISK @16GB / WILL NOT RUN, with weights (auto-ungated, gated, manual) and
  size. Read the legend first: a missing `mps` in a registry row means NOBODY HAS
  RUN IT, not that it cannot run, and gated weights are friction rather than
  failure.
- `MAC_PORTABILITY_GUIDE.md` - the long-form guide: setup, the traps that cost
  real time, measured costs, and a dated corrections ledger. **An OOM on unified
  memory is a machine REBOOT, not a failed render** -- section 2 is only about
  that.
- `MAC_PUNCH_LIST.md` - what closed, what is still open, and what needs the
  other machine.

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
- `multimodal-story-schema/` - the 2026-07 source-pack and story-schema plans
  plus the `schema-examples/` fixtures. Historical plan of record, last
  changed 2026-07-24: `nodes/_otr_source_payload.py`,
  `tests/test_source_payload_chunk3.py` and
  `tests/test_cast_lock_policy_repin.py` still cite it for the source-payload
  contract and the schema examples, but its stage plans also describe a
  per-bank `story_rules` loader that never shipped (there is no
  `nodes/story_rules/` and no `_otr_story_rules.py`) -- read them as history,
  not as the live design. The five runnable banks live in
  `nodes/story_packs/banks.json`; the current source-bank work is the PROPOSED
  My Story bank, scoped in `2026-09-10-my-story-app-scope.md` and tracked as
  sprints 3-4 in `GO_FORWARD_PLAN.md`.
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
