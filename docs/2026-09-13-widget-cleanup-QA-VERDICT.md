# Widget cleanup -- the independent QA verdict, and what it changes

Cursor's verified QA report on `docs/2026-09-13-widget-cleanup-QA-BRIEF.md`,
returned 2026-09-13 with its own contrarian pass folded in. **This file is the
plan of record for the widget tier.** The brief is the question; this is the
answer, and where the two disagree this wins.

**Nothing in the widget tier has been coded.** The QA review changed no source
and no workflow.

---

## The headline: do NOT ship this as one change

Approved, in this order:

1. Remove `perfect_run_spacesaver`.
2. Remove **all five** deprecated `ffmpeg` widgets.
3. Remove `seed_mode` + `request_seed` from `OTR_VideoDirector` **only**.
4. Replace the proposed renames with UI-only `display_name` labels.
5. Consolidate the duplicate title controls, as separate tested work.
6. Reorder the Writer **only after** an exact 36-widget order is pinned.

Rejected or held:

* **Voice-engine consolidation -- NO-GO** under the current design.
* **`custom_source_bank` -- leave unchanged.** The proposed fix does not work.
* **No widget aliases.** No internal renames for labelling reasons.

---

## What it corrected in our own numbers

| the brief said | the truth |
|---|---|
| 47 widget targets in `widget_mapping.json` | 47 managed KEYS, **55 managed targets** |
| 2,074 references a rename would break | 2,074 is the TOTAL widget descriptors in all 17 graphs, not the affected set |
| three deprecated `ffmpeg` widgets | **five** -- this window had already found the same |
| "a trailing widget is nearly free" | **false**, see below |

## The finding that inverts the brief's central assumption

`migrateWidgetsValues` fires exactly when **one** widget is removed, because
the Writer's mask length is `(37 - k) + 1` against 37 saved values. Symbolic
replay, independently reproduced:

| removed | migration | widgets holding a wrong value |
|---|---|---|
| `perfect_run_spacesaver` alone | FIRES | 23 |
| it + `min_p` | no | 27 |
| those + `repetition_penalty` | no | 26 |
| trailing `story_author` alone | **FIRES** | 4 -- corrupts `replay_from` and the three My Story fields |
| last two or three together | no | **0** |

So "remove the trailing one, it is cheap" is the single most dangerous move
available. This is OLD-workflow behaviour; correctly migrated shipped graphs
are fine.

## Two guards that must exist BEFORE any reorder

1. **Live schema-order comparison.** Every graph's ordered widget descriptor
   names against the live `INPUT_TYPES` order. The existing tests prove the 17
   graphs agree with EACH OTHER -- they could all agree on the same wrong order.
2. **Name-paired value migration, with symbolic values.** Real defaults are
   `""`, `False` and `0`, so two swapped widgets can hold equal values and hide
   the corruption. Unique markers per widget, asserted to travel with their name.

## The rename answer: `display_name`, not a rename

Frontend 1.51.10 already separates the UI label from the stable internal key,
and `node_info` forwards the input options:

```python
"source_bank": (choices, {"display_name": "Story Source",
                          "default": "scifi_news_pro"})
```

That gets the wording with none of the blast radius -- internal names, keyword
parameters, API prompt keys, saved descriptors, positional values and profile
mappings all stay put. Update `localized_name` in the canonical for
consistency, regenerate variants, and never touch descriptor `name` or
`widget.name`.

## Why the voice consolidation is a no-go

Not a preference -- five concrete blockers:

* `char_voice_engine` is legitimately stamped literal `"auto"` when CastLock
  resolves nothing, and the render node still needs a concrete engine then.
* `generate()` selects and validates its adapter BEFORE `_render_per_line`
  reads the ledger.
* `IS_CHANGED` fingerprints on the widget engine.
* Profile mapping deliberately patches BOTH CastLock and the render nodes.
* The mismatch guards prevent a real audible wrong voice while credits say
  otherwise.

A future consolidation needs a new contract: CastLock always emits a concrete
resolved character engine, mixed-engine cast rows are settled either way, and
replay / cache / profiles / credits are re-proven.

## Why `custom_source_bank` cannot be fixed as proposed

The dropdown is fed scalar bank IDs by `list_bank_ids()`. The `label` field in
`banks.json` is **not** used as a per-option display label, so editing
`"+ Add Your Own"` changes nothing a user sees. A real fix needs a general
label/value mechanism for banks, which does not exist. Do not build a one-off
for this row.

## On the title consolidation

`OTR_LedgerScriptWriter` is the sole workflow-facing title control, but NOT the
sole producer -- `_otr_ledger_cleanup.py::_complete_prose` fills
`meta.episode_title` when J.5 leaves it empty, deliberately, because credits
need a non-empty title.

The Assembler already has a ledger source: **canonical link 289** carries the
complete `v2_ledger_json` into `replay_descriptor` despite that input's narrow
name. So its title descriptor can go without adding an input. (The QA's own
contrarian claimed otherwise and was overruled on this point.)

`tests/test_video_ledger.py` currently exercises a copied test-only
implementation of SignalLost's title chain. That is inadequate -- extract the
chain into production code and have the test call it.

## One correction to the QA report itself

It lists `tests/test_text_metric_ownership.py` as a still-red baseline guard
that must be cleared first. **It was cleared in `5184e33b`**, before the report
was written -- the visitor now distinguishes a ledger row from ComfyUI's
`ui["text"]` preview dict, and the guard was re-proven against `row["text"]`,
`line["text"]`, `self.row["text"]` and non-atomic updates. Step 1 of its
implementation order is already done.

## Mandatory gates, per stage

```
python scripts/build_variants.py --all && python scripts/build_variants.py --check
python -m pytest -q -p no:cacheprovider \
  tests/test_widget_value_alignment.py \
  tests/test_canonical_widget_input_parity.py \
  tests/test_workflow_link_target_indexes.py \
  tests/test_workflow_live_passes_validator.py \
  tests/test_no_shipped_graph_carries_a_premise.py \
  tests/test_input_types_signature_parity.py \
  tests/test_workflow_apply.py \
  tests/test_capability_profiles.py
```

plus the full suite and the Bug Bible regression, and on the runtime side:
`/object_info` inspection, an API-prompt conversion, and a real canonical run
reaching `RESULT SUCCESS` + `obs_publish OK` with the asset under `otr/obs/`.

## The blocker on step 9

There is still **no exact 36-widget Writer order**. The brief gives a leading
block and broad groups; every model, provider, runtime, source, scaffold,
validation and replay field is unplaced. Coding stops until one explicit list
exists naming every remaining widget exactly once -- and that list becomes the
test.

The settled leading block:

```
episode_title, source_bank, custom_premise, num_characters, act_count,
include_act_breaks, visual_style, creativity, lemmy_cameo,
story_characters, story_plot, story_setting, story_author
```
