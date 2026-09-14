# QA brief: the OTR widget cleanup, BEFORE any code is written

**For an independent reviewer (Cursor / Codex / any second reader).**
Written 2026-09-13 by the Claude window that measured the numbers below.
Nothing in this plan has been coded yet. Your job is to **refute it**.

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
Branch: `main` at `910df1d5`. ComfyUI frontend package `1.51.10`, backend `0.35.1`.

---

## 0. What you are reviewing

An external UI audit asked for five things. In priority order:

1. **REMOVE dead widgets** — `perfect_run_spacesaver` on `OTR_LedgerScriptWriter`;
   the three deprecated `ffmpeg` widgets on `otr_caption_burn.py`,
   `otr_master_audio_mux.py`, `otr_silent_composite.py`; and possibly
   `seed_mode` / `request_seed` on `OTR_VideoDirector`.
2. **ONE OWNER for duplicated controls** — three `episode_title` widgets with
   three different meanings, and the two voice-engine controls that currently
   must agree or the render stops.
3. **REORDER** the writer's 37 widgets so the creative controls come first.
4. **RENAMES** (~20 widgets).
5. **`custom_source_bank`** ("+ Add Your Own") is listed like a runnable source
   and always fails; label it setup-required.

The operator has already ruled on the biggest question: **breaking already-saved
user workflows is ACCEPTABLE.** There is exactly one outside tester (an AMD /
Radeon volunteer) and he will be told to take a new version. So "this breaks
saved graphs" is NOT by itself a reason to reject a step. His one condition was:
*"let's be sure it's working first."*

**So do not spend your review arguing about saved-graph compatibility in the
abstract. Spend it on whether the plan is CORRECT and whether the evidence
below actually says what it is claimed to say.**

---

## 1. The mechanism, measured (please re-derive, do not trust)

### 1a. Widget values restore POSITIONALLY by default

`comfyui_frontend_package/static/assets/settingStore-Bqt2KcA8.js`, inside
`LGraphNode.configure` (original TS recoverable from the adjacent `.js.map`
`sourcesContent`, `src/lib/litegraph/src/LGraphNode.ts`):

```ts
const namedValues = getNamedValues()
if (namedValues && LiteGraph.namedValuesRestore) {
  for (const widget of this.widgets) { ... widget.value = namedValues[widget.name] }
} else if (info.widgets_values) {
  let i = 0
  for (const widget of this.widgets ?? []) {
    if (widget.serialize === false) continue
    if (i >= info.widgets_values.length) break
    widget.value = info.widgets_values[i++]      // <-- pure positional zip
  }
}
```

`Comfy.Workflow.NamedValuesRestore` is declared in `GraphView-Db8WTTzq.js` as
`defaultValue: !1, experimental: !0`, and `LiteGraphGlobal` has
`namedValuesRestore=!1`. The key is absent from the operator's own
`user/default/comfy.settings.json`.

**So restore-by-name is OFF.** The `fallbackWidgetsValuesNames` node-def table
that would otherwise migrate a legacy positional array is also dead on this
stack: ComfyUI 0.35.1's `server.py::node_info` builds a fixed key whitelist that
does not include it, and `grep -rn fallbackWidgetsValuesNames --include=*.py`
over the ComfyUI tree returns zero hits.

**Q1 for you: is that right? Is there any other code path that restores widget
values — an extension, a "fix node" command, a node-def diff on load?**

### 1b. There is a second function nobody expected: `migrateWidgetsValues`

Same bundle. De-minified:

```js
migrateWidgetsValues(defInputs, liveWidgets, savedValues) {
  const names = new Set(liveWidgets.map(w => w.name))
  const mask = Object.values(defInputs)
    .filter(i => names.has(i.name) || i.forceInput)
    .flatMap(i => i.control_after_generate ? [!!i.forceInput, false] : [!!i.forceInput])
  return mask.length === savedValues?.length
    ? savedValues.filter((v, idx) => !mask[idx])
    : savedValues
}
```

It exists to strip values that OLD graphs stored for `forceInput` inputs. Its
only guard is a LENGTH comparison, and the mask is built in **definition order**
while `savedValues` is in **widget order**. `OTR_LedgerScriptWriter` has 38
declared inputs, 37 widgets, and exactly one `forceInput` (`gate_in`, definition
index 32).

So the mask length is `(37 - k) + 1 = 38 - k` for `k` widgets removed, against a
saved array of 37. **They are equal exactly when `k == 1`.**

### 1c. What that costs, simulated against the real `INPUT_TYPES`

Simulation script: `<scratchpad>/sim_migrate.py` (not committed). It loads the
real writer class through `__init__._NODE_MODULES` and replays both functions.

Removing MID-LIST widgets (`perfect_run_spacesaver`, then `min_p`, then
`repetition_penalty`):

| removed | migration fires | widgets holding the WRONG value |
|---|---|---|
| 1 | yes | 23 of 36 |
| 2 | no  | 27 of 35 |
| 3 | no  | 26 of 34 |

Removing TRAILING widgets (`story_author`, `story_setting`, `story_plot`):

| removed | migration fires | widgets holding the WRONG value |
|---|---|---|
| 1 | **yes** | **4 of 36** — `replay_from` and the three My Story fields shift |
| 2 | no | **0 of 35** |
| 3 | no | **0 of 34** |

**The counter-intuitive result: a single TRAILING removal is the only trailing
case that corrupts anything, and it does so because the migration fires.**

**Q2 for you: reproduce this. Is the mask really built in definition order
against a widget-order array? Is `gate_in` really the node's only `forceInput`?
Does `control_after_generate` appear anywhere on this node (it would change the
arithmetic)?**

---

## 2. The plan, and what we think each step costs

### Step 0 (pre-req, already verified, not yet committed)

`tests/test_workflow_link_target_indexes.py` is the dedicated backstop for the
`dst_slot` off-by-one class. It uses `WORKFLOWS_DIR.glob("*.json")` — **non
recursive** — and `workflows/` contains exactly one file, so
`pytest --collect-only` reports **`collected 1 item`**. The 16 variants under
`workflows/variants/` have never been checked by it.

Applying its property to all 17 by hand today: **0 violations**. So widening the
glob to `rglob` is purely protective and lands first, on its own.

**Q3: agree that this must land before any positional surgery? Any reason
`rglob` would pick up something it should not (`workflows/external_examples/`
is `.comfyignore`d but still on disk — does it live under `workflows/`)?**

### Step 1 — removals

Proposed: remove `perfect_run_spacesaver` (writer, widget index 8) and the three
deprecated `ffmpeg` widgets, in ONE change rather than several.

Grounding already done:
* `perfect_run_spacesaver` has **no production consumer**. `rtx_upscale.py`, its
  old reader, no longer exists. `tests/test_perfect_run_spacesaver_deprecated_noop.py`
  currently asserts the widget **MUST REMAIN**, on the reasoning that removing it
  "shifts widget indices 10..34 and breaks saved workflows" — that test was
  written 2026-08-08, before the operator's ruling that breaking saved graphs is
  acceptable, and before this repo learned the three-part removal procedure.
  **That test has to be retired or re-pinned in the same change.**
* The `ffmpeg` widgets say "DEPRECATED and IGNORED (2026-09-04)" in their own
  tooltips. Note there are **FIVE**, not three: the other two are on
  `otr_post_upscale_procgen_blend.py` and `otr_scene_aware_scopes.py`, whose
  nodes 93/94 were removed from the canonical by `8171e994`.

**Q4: is `perfect_run_spacesaver` genuinely dead? Grep for anything that reads
`meta["perfect_run_spacesaver"]` — `nodes/_otr_writer_tail.py` still STAMPS it.
A stamp with no reader is dead; a stamp with a reader is not. Which is it?**

**Q5: should the two `ffmpeg` widgets on the now-unwired nodes 93/94 be removed
too, or does removing a widget from a node that no longer ships buy nothing?**

### Step 2 — one owner

Two sub-items, and this is the one with real DESIGN in it:

* **Three `episode_title` widgets**: `OTR_LedgerScriptWriter` (the real one),
  `OTR_EpisodeAssembler` (log text only), `OTR_SignalLostVideo` (fallback).
* **Voice engines**: today `nodes/_otr_voice_node_common.py` (~lines 1035-1090)
  raises `EngineUnusable` when `OTR_CastLock`'s stamped engine disagrees with the
  render node's own `engine` widget. The audit wants CastLock to be sole owner
  and the render nodes to follow its ledger.

**Q6, and this is the most important question in this document:** the existing
guard has a documented reason — a mismatch makes a real, audible WRONG voice
speak while the ledger and credits say otherwise, because CastLock's bark stamp
clears `voice_ref_id`. AND the code notes that `char_voice_engine` is stamped
literally `"auto"` when CastLock resolved nothing (a preset bank under an auto
request). **If the render node simply follows the ledger, what does it do when
the ledger says `"auto"`?** Does "sole owner" actually hold, or is there a real
case where CastLock does not decide and the widget must?

### Step 3 — reorder the writer's 37 widgets

Current order has `source_bank` at widget index 21 and `visual_style` at 22,
below model selectors, sampling controls, a no-op widget and four cloud provider
bindings. The four My Story fields are at 33-36, twenty-eight widgets away from
`custom_premise` at 4.

A working reorder tool has been proven on a scratch copy
(`<scratchpad>/widget_surgery.py`). It permutes widget DESCRIPTORS among the
slots they already occupy, so the one linked socket (`inputs[32]`, `gate_in`,
carrying link 279) never moves and **zero `dst_slot` repairs are needed**. All
37 values travel with their widget, verified by name-pairing before and after.

**Q7: note that reordering the graph JSON alone changes NOTHING a user sees —
ComfyUI draws widgets in `INPUT_TYPES` declaration order. The reorder is
primarily a Python edit in the node class, with the graph's `widgets_values`
permuted to match. Confirm or refute that, and say what else must move in
lockstep (`config/profiles/widget_mapping.json` has 47 managed entries keyed
`[node_type, widget_name]`, plus an `exempt_widget_names` list naming the
writer's 20 creative widgets).**

### Step 4 — renames (~20)

Under positional restore a rename is **free for values** — the loop never reads
a name. The known casualty is narrower: `ComfyNode.configure`
(`src/services/litegraphService.ts:543`) reconciles the saved `inputs[]` array
BY NAME; a descriptor bearing the old name is not matched and is appended as an
"extra input" carrying its link.

Renames DO break things we own and must fix in the same change: the API-form
prompt JSONs under `scripts/_otr_*prompt*.json` (keyed by name), the 47
`widget_mapping.json` targets, `exempt_widget_names`, the node function's
keyword parameter, and every test passing that keyword.

Note also: this pack deliberately REMOVED a `_RENAME_ALIASES` dict in a
2026-05-12 clean break. **Q8: find that commit, read why, and say whether the
reasoning that retired it argues against a widget-name alias map today.**

### Step 5 — `custom_source_bank`

`nodes/story_packs/banks.json` row: `label: "+ Add Your Own"`, `runnable: false`,
`interpreter: ""`, `fetcher: ""`. It is ALREADY out of the roll pool via
`runnable=false` and a test pins that — **do not change that part.** The ask is
only that the dropdown stop presenting it as a runnable source.

---

## 3. What we want from you

Answer Q1-Q8 above. Beyond those, the standing instruction:

**Default to "refuted" for anything you cannot ground in the real files.** A
claim quoted from minified JavaScript is guilty until you have found that exact
code yourself. The most valuable thing you can find is a correct FACT paired
with a wrong INFERENCE — that is the failure mode this brief is most likely to
contain, and it is the one the author structurally cannot see.

Specific invitations to disagree:

* Is the removal/reorder ordering wrong? Should renames go FIRST (cheapest for
  saved graphs) so that if anything goes wrong the blast radius is smallest?
* Is doing all of steps 1-4 in ONE change better or worse than four commits?
  Given a user's saved graph is wrecked by any mid-list change anyway, is there
  any argument for spreading the damage across four releases?
* Is there a step here that should simply NOT be done? "The audit asked for it"
  is not a reason, and this repo has a standing rule that an inert widget SHOULD
  be removed — but also a standing rule against chasing polish.

## 4. House rules you should know before proposing text

* No curse words in code, comments, logs or commit messages, and never the word
  "dummy".
* UTF-8, no BOM. Plain ASCII hyphens in anything that reaches PowerShell.
* `workflows/otr_canonical.json` is the source of truth; the 16 variants are
  GENERATED by `scripts/build_variants.py --all` and must never be hand-edited.
* Gates that must pass before any push:
  `python scripts/build_variants.py --all && python scripts/build_variants.py --check`
  plus `tests/test_widget_value_alignment.py`,
  `tests/test_canonical_widget_input_parity.py`,
  `tests/test_workflow_link_target_indexes.py`,
  `tests/test_workflow_live_passes_validator.py`,
  `tests/test_no_shipped_graph_carries_a_premise.py`.
