# Handoff: the widget rename / reorder tier

Paste the block at the bottom into a fresh window. Everything above it is the
grounding that block refers to, written 2026-09-13 by the window that did the
safe half.

---

## What already landed (do NOT redo it)

Commit `2f11fe88` took the whole **string-and-default** half of an external
widget UI audit. That work is done, pushed, and regression-clean:

* five receipt-only controls (`device_policy`, `dtype_policy` on both
  directors, `seed_mode`, `request_seed`) now say they are receipts, because
  measurement confirmed nothing acts on them;
* tooltips naming retired nodes are fixed, including two that **this repo's own
  09-13 work falsified** (Credits Roll's "Node 93", `draw_scopes` promising
  `OTR_SceneAwareScopes`; nodes 93 and 94 left the canonical that evening);
* the four-widget cross-node voice trap is now named on all four tooltips;
* eleven fresh-node defaults now match the shipped graph, verified by reading
  live `INPUT_TYPES` (worst was both voice nodes defaulting to `indextts2`,
  and `OTR_VideoRenderBatch.mode` defaulting to the diagnostic `soak`);
* `visual_style` leads with the fact that it changes nothing visible under the
  canonical's procedural video lanes.

**Two defaults were deliberately NOT changed** and are the operator's call, not
a defect to fix silently: `llm_quant_policy` (fresh `bnb_nf4` vs canonical
`none`) and `llm_vram_ceiling_gb` (fresh `14.5` vs canonical `10.0`). Matching
the canonical makes a fresh node load the writer UNQUANTISED at a HIGHER
ceiling — correct for the canonical's 4B writer, and capable of OOMing a small
card that `bnb_nf4` would have carried. `act_count` (fresh 3, canonical 1) is
left alone for the same reason: taste, not a defect.

---

## What is left, and why it was not done in the same night

The audit's remaining recommendations are **renames** and a **reorder** of
`OTR_LedgerScriptWriter`'s 37 widgets. Both are positional surgery on the one
file this repo calls its source of truth, and this repo has already corrupted
all 63 workflows once doing exactly this.

### The measured blast radius

Run these yourself before believing any of it:

| What a rename touches | Measured |
|---|---|
| Saved `inputs[].widget.name` references across the 17 shipped graphs | **2,074** |
| `config/profiles/widget_mapping.json` `managed` entries, keyed `[node_type, widget_name]` | **47** |
| `OTR_LedgerScriptWriter` widgets | **37** (its `inputs` array is 38) |

A rename is therefore never just an `INPUT_TYPES` key. It is:

1. the `INPUT_TYPES` key, **and**
2. the node function's **parameter name** (ComfyUI calls by keyword), **and**
3. every caller and test passing that keyword, **and**
4. every `inputs[].widget.name` in every saved graph, **and**
5. `widget_mapping.json` targets, **and**
6. every user's own saved workflow, which you cannot migrate.

Point 6 is the one with no engineering answer. A rename silently breaks graphs
that already exist on other people's machines.

### The reorder hazard, in this repo's own words

`CLAUDE.md` section 0, written after the incident:

> **REMOVING A WIDGET TOUCHES THREE THINGS, NOT ONE.** 1. `widgets_values` —
> drop the value at the widget's index in every saved graph. 2. The `inputs`
> DESCRIPTOR array. 3. **EVERY LINK TARGETING A LATER SLOT.** `dst_slot` is an
> INDEX INTO THAT SAME `inputs` ARRAY, which holds link sockets and widget
> descriptors together. Remove a descriptor and every link past it is off by
> one. This is invisible to a widget-count check.
>
> **Repair by IDENTITY, not arithmetic:** set each `dst_slot` to the index
> whose `inputs[i].link` equals that link's id — self-correcting and impossible
> to double-apply.

A reorder is that hazard once per moved widget. A trailing widget is nearly
free; a mid-list one costs the re-index everywhere.

### The four gates that must all pass

```
python scripts/build_variants.py --all && python scripts/build_variants.py --check
python -m pytest -q -p no:cacheprovider tests/test_widget_value_alignment.py tests/test_canonical_widget_input_parity.py tests/test_workflow_link_target_indexes.py tests/test_workflow_live_passes_validator.py
```

The first three all passed while the links were broken, historically. The
link-target-indexes one is what actually catches it.

### Baseline discipline

The suite is red from causes unrelated to this work. Never read a raw failure
count. Diff against a worktree at the same HEAD:

```
git worktree add --detach C:/Users/jeffr/Documents/ComfyUI/_worktrees/rename_base <HEAD-before-your-work>
```

and compare failure sets, not numbers. `test_w45_campaign_bank_pinning` fails
in a worktree as an artifact (needs an untracked `tmp/`) — not a real failure.

The known-fail guard **hides tracebacks** (it raises `SystemExit(2)` inside
`pytest_sessionfinish`, so the FAILURES section never prints and `-rf` /
`--tb=short` cannot work). Run pytest through a tiny plugin that prints
`report.longreprtext` from `pytest_runtest_logreport`.

---

## The audit's actual recommendations, for reference

**Reorder `OTR_LedgerScriptWriter`** so the strong creative controls are on the
first screen. Today `source_bank` is at position 22 and `visual_style` at 23,
below model selectors, sampling controls, a no-op widget and four cloud
provider bindings. The four My Story fields sit at the very end, 28 widgets
away from `custom_premise`. Proposed order: `episode_title`, `source_bank`,
`custom_premise`, `num_characters`, `act_count`, `include_act_breaks`,
`visual_style`, `creativity`, `lemmy_cameo`, then the My Story four, then the
model pickers, then sampling/runtime, then provider bindings, then
`replay_from`.

**Reorder `OTR_CastLock`** to put `char_voice_engine` / `announcer_voice_engine`
/ `voice_device` above the casting-policy widgets, since the README sends people
there for exactly those.

**Renames proposed** (a sample — the audit lists ~20): `source_bank` →
`story_source`, `creative_writing_model` → `story_model`, `technical_model` →
`structure_model`, `include_act_breaks` → `music_between_acts`, `use_exchange`
→ `grouped_dialogue`, `voice_device` → `audio_device`, `*_model` → `*_engine`
on the Video Director, `fresh_cap` → `max_fresh_stills`.

**Also in that tier**: moving the three image-engine dropdowns off
`OTR_VideoDirector` onto `OTR_ImageDirector` (they are on the wrong-named node
today, while custom image IDs are mapped on the other one), and removing the
genuinely dead widgets (`perfect_run_spacesaver`, the three deprecated `ffmpeg`
widgets, the six inert `OTR_VideoRenderBatch` diagnostic widgets).

---

## TASK ZERO: clear the 14 suite deltas BEFORE any widget work

Added 2026-09-13 night. The full-suite diff against `283abaa6` ended the
session at 63 failures vs 53 -- 14 tests fail that passed at session start,
4 that failed then now pass. **They were never individually cleared.**

This comes FIRST, and the reason is not tidiness: every gate in this document
is a test. Positional widget surgery is verified by running the suite and
comparing failure SETS. If fourteen of those failures are unexplained going
in, you cannot tell your own off-by-one from the noise you inherited -- which
is precisely how the 63-workflow corruption went unnoticed the first time.

The list, with what is already known about two of them:

| test | status |
|---|---|
| `test_legacy_audit_clean::test_no_unclassified_legacy_references` | **known.** Trips on the node TITLES from the canvas relayout -- its audit flags "Director" surfaces and the titles now read "Video Director / Settings". Real, cosmetic. Decide: retitle, or widen the audit's allowlist. |
| `test_canonical_headless_api::test_visual_style_override_does_not_patch_story_fields` | **known: NOT a regression.** Passes in isolation -- order-dependent. Confirm and move on. |
| `test_scope_render_profile` (x4) | expected to follow the node 93/94 removal (`8171e994`). **Confirm, do not assume.** |
| `test_freeze_cascade_title_rename` | unexamined |
| `test_gguf_version_pin_is_documented` (x2) | unexamined |
| `test_google_video_sfx_workflow` | unexamined |
| `test_model_asset_index_drift` | unexamined |
| `test_source_bank_widget_2c::TestAddYourOwnSignpost` | unexamined -- note it failed all night in targeted runs too |
| `test_text_metric_ownership` | unexamined |
| `test_workflow_director_freedom` | unexamined |

**RUN EACH ONE ALONE FIRST.** Two of the fourteen already behave differently
in isolation, so the batch result is not the evidence for any individual row.

For each: it is either (a) a real defect from that session -- fix the code;
(b) an intentional behaviour change -- re-pin the test WITH the reason written
into it, never silently; or (c) an isolation/order artifact -- prove it and
record it. All three outcomes are acceptable. Leaving one unclassified is not.

Four tests also went from failing to PASSING. Confirm they pass for a real
reason and not because something stopped being checked.

---

## PASTE THIS INTO THE NEW WINDOW

```
Pick up the widget audit's remaining tiers on ComfyUI-OldTimeRadio, starting
with TASK ZERO (clear the 14 suite deltas) -- you cannot verify positional
widget surgery against a suite you do not trust. Read
docs/HANDOFF_WIDGET_RENAME_REORDER.md first — it has the measured blast radius
and the gates. Then read CLAUDE.md sections 0 and 0B before touching anything.

Start on main, current: git fetch origin main && git pull --rebase origin main

DO NOT start editing. Your first deliverable is a GO / NO-GO recommendation on
the RENAMES specifically, grounded in the repo, answering one question I could
not resolve and did not want to guess at:

  A rename breaks 2,074 saved inputs[].widget.name references across the 17
  shipped graphs (regenerable), 47 widget_mapping.json targets (fixable), the
  node function's keyword parameter and all its callers (fixable) — and every
  workflow a USER has already saved on their own machine, which we cannot
  migrate and which fails in a way they cannot diagnose.

  Is there a migration path that protects those users? Options to evaluate,
  not assume: a ComfyUI frontend alias/compat map; accepting BOTH the old and
  new keyword in the node function; a graph-load shim that rewrites old names;
  or a documented breaking change gated on a major version. Check whether
  ComfyUI itself offers a supported rename/alias mechanism before designing
  one — grep the frontend package and ComfyUI's own nodes for prior art, and
  note that this pack deliberately REMOVED a _RENAME_ALIASES dict in the
  2026-05-12 clean break (see __init__.py), so find out WHY before proposing
  its return.

  If there is no safe path, say so plainly and recommend we do the REORDER
  only and drop the renames. A reorder does not change any name, so saved
  graphs keep working as long as the positional repair is done correctly.

Then, whichever you recommend, execute in this order and stop at any red:
  1. Take ONE widget as a pilot end to end — I suggest a TRAILING widget so
     the re-index cost is near zero — and prove the whole procedure on it,
     including the four gates and a fresh-worktree baseline diff.
  2. Only after the pilot is green, do the rest in small, individually-pushed
     commits, gates run every time.
  3. Repair dst_slot BY IDENTITY (match inputs[i].link to the link id), never
     by arithmetic. Arithmetic is double-appliable; identity is not.
  4. Regenerate variants with scripts/build_variants.py --all, never hand-edit
     a variant, and confirm with --check.

Constraints that are not negotiable: never git add -A (a subagent's probe file
got swept into a commit that way); commit messages via -F file; push to main
(v2.0-alpha is retired and frozen — do not push to it); and do NOT touch
pyproject.toml, which auto-fires a registry publish and currently has 2.1.0 and
2.1.1 both Pending.

One contrarian reviewer on the finished diff before any push, briefed to
refute, per CLAUDE.md.
```
