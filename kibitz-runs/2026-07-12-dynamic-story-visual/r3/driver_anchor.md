# r3 driver anchor -- dynamic-story-visual (WIRING)

Written BEFORE the r3 fan-out. Author: Claude (Cowork), docs-only architecture
owner + sole judge. Panel: codex `gpt-5.6-sol` @ ultra, antigravity
`gemini-3.5-pro`. Doc under review: rev 3 (r1 arc + r2 coding plan folded).

Scope of r3: WIRING. The exact `workflows/otr_canonical.json` delta, the node
record, the registration surface, the test pins that break, and the re-validation
procedure (Lesson 7). Not the arc (r1), not the LLM contract (r2).

Every number below was read out of the REAL file (a temp python probe under the
ComfyUI venv, run and deleted; never the Linux mount).

---

## A. Ground truth of the live graph (2026-07-12)

```
last_node_id = 95     last_link_id = 284 is FREE     23 nodes, 57 links
version 0.4, revision 0, config {}, groups []   <- groups is EMPTY: no bounding box constrains placement
every node has mode == 0 (nothing muted/bypassed)
link types in use: STRING x47, AUDIO x8, INT x1, FLOAT x1
```

**Node 62 `OTR_LedgerFreezeCascade`** -- `pos [620, 80]`, `size [379.97, 431.58]`,
`order 3`. Outputs:

| idx | name | type | links |
|---|---|---|---|
| 0 | script_text | STRING | `[]` |
| **1** | **script_json** | STRING | **`[16, 231, 232, 233, 252, 255]`** |
| 2 | news_used | STRING | `[110]` |
| 3 | estimated_minutes | INT | `[]` |
| 4 | freeze_verdict | STRING | `[]` |
| 5 | episode_seed | INT | `[]` |
| 6 | v2_ledger_json | STRING | `[234]` |

Out[1]'s six consumers, decoded:

```
[ 16, 62, 1, 12, 2, "STRING"]  -> OTR_SignalLostVideo (12)         in 2
[231, 62, 1, 81, 0, "STRING"]  -> OTR_BatchCharacterVoices (81)    in 0
[232, 62, 1, 82, 0, "STRING"]  -> OTR_AnnouncerVoice (82)          in 0
[233, 62, 1, 83, 0, "STRING"]  -> OTR_StableAudioTheme (83)        in 0
[252, 62, 1, 90, 0, "STRING"]  -> OTR_ShotLock (90)                in 0   <-- REPOINT
[255, 62, 1, 89, 0, "STRING"]  -> OTR_MetaBriefImagePromptGen (89) in 0   <-- REPOINT
```

**HARD CONSTRAINT:** 16 / 231 / 232 / 233 STAY on node 62. The audio trio is
pinned to the RAW freeze json by `tests/test_full_workflow_v2_audio_wiring.py`
(:220-232), and node 80 `OTR_CastLock` must keep sourcing 62.out[6]
(`v2_ledger_json`, link 234). The direction node is inserted in the VISUAL lane
ONLY -- routing the audio fan through it would be both a test failure and a
design error (the audio branch has no business waiting on a visual LLM).

**Node 89 `OTR_MetaBriefImagePromptGen`** -- `pos [1080, 1040]`,
`size [379.97, 239.97]`, `order 6`, `widgets_values [False]` (len 1).

| in | name | type | link | widget |
|---|---|---|---|---|
| 0 | script_json | STRING | **255** | - |
| 1 | image_policy_json | STRING | 254 | - |
| 2 | gate_in | STRING | null | - |
| 3 | consistency_gate_warn_only | BOOLEAN | null | `{"name": "consistency_gate_warn_only"}` |

Outputs: 0 `image_prompts_json` -> `[258]`; 1 `report` -> `[]`.

**Node 90 `OTR_ShotLock`** -- `pos [1129.39, 1200.0]`, `size [379.97, 239.97]`,
`order 14`, `widgets_values [False]` (len 1).

| in | name | type | link | widget |
|---|---|---|---|---|
| 0 | script_json | STRING | **252** | - |
| 1 | audio_done | STRING | 253 | - |
| 2 | video_policy_json | STRING | 251 | - |
| 3 | image_done | STRING | null | - |
| 4 | gate_in | STRING | null | - |
| 5 | consistency_gate_warn_only | BOOLEAN | null | `{"name": ...}` |

Outputs: 0 `patched_ledger_json` -> `[256]`; 1 `video_revision` (INT) -> `[]`;
2 `shot_report` -> `[]`; **3 `done` -> `[]` (SHIPS UNWIRED -- the precedent for
the new node's own unwired `done`)**; 4 `episode_id` -> `[268]`.

**Node 1 `OTR_LedgerScriptWriter`** -- `widgets_values` has 34 entries;
`visual_style` is at index **24**, value `"sci_fi_radio"`. `inputs[24]` is
`{name: "visual_style", type: "COMBO", link: null, widget: {name:
"visual_style"}}` -- the modern widget-as-input serialization, NOT a forceInput.
The vector is APPEND-ONLY (BUG-LOCAL-097); this feature does not touch it.

Free placement rectangle: x 1060-1440, y 760-1000 is empty (below CastLock's
band at y=470, above 89 at y=1040, right of 87's column which ends at x=1000,
left of 91 at x=1560). Reading order stays left-to-right: 62 (x 620) -> NEW
(x 1060) -> 89/90 (x 1080/1129) -> 91 (x 1560).

---

## B. The exact delta I propose (doc section 8.3, to be made exact in rev 4)

**Minimal-blast-radius rule: REPOINT existing link ids, never renumber.**

1. `last_node_id: 95 -> 96`; `last_link_id: 283 -> 284`.
2. NEW node record id **96**, type `OTR_DynamicStoryDirection`:
   - `pos [1060, 760]`, `size [379.96875, 239.96875]`, `mode 0`, `flags {}`,
     `order` assigned by the frontend's topological sort (it must land AFTER 62
     and BEFORE 89/90; do not hand-author an `order` that contradicts the links).
   - `inputs`: `[0] {name: "script_json", type: "STRING", link: 284}`,
     `[1] {name: "gate_in", type: "STRING", link: null, shape: 7}`.
   - `outputs`: `[0] {name: "patched_ledger_json", type: "STRING", links:
     [252, 255], slot_index: 0}`, `[1] {name: "direction_report", type:
     "STRING", links: []}`, `[2] {name: "done", type: "STRING", links: []}`.
   - `widgets_values: []` (zero widgets -- both inputs are forceInput; the
     precedent is node 95 `OTR_CreditsRoll`, pinned at
     tests/test_workflow_live_passes_validator.py:139 as "zero widgets (two
     forceInputs)").
   - `properties: {"Node name for S&R": "OTR_DynamicStoryDirection"}`.
3. NEW link `[284, 62, 1, 96, 0, "STRING"]`.
4. REPOINT (same ids, new source): `[252, 96, 0, 90, 0, "STRING"]`,
   `[255, 96, 0, 89, 0, "STRING"]`.
5. Node 62 `outputs[1].links`: `[16, 231, 232, 233, 252, 255]` ->
   `[16, 231, 232, 233, 284]`.
6. Nodes 89 and 90: `inputs[0].link` stays **255** and **252** respectively --
   the link ids do not change, only their source. No widget change, no
   `widgets_values` change on either node.

Nothing else moves. No other node's inputs, outputs, widgets, or ids change.

## C. Test pins that BREAK (must change in the SAME commit -- Lesson 7)

| Test | Line | Why | Fix |
|---|---|---|---|
| `tests/test_google_video_sfx_workflow.py` | :41 | `assert wf["last_link_id"] == 283` -- HARD equality | bump to 284 |
| `tests/test_visual_style_widget_3c.py` | :62-66 | `choices == list(vs.list_style_ids())` -- the sentinel is not in the registry | "registry PLUS exactly one sentinel" at the writer surface; keep `list_style_ids()` registry-only |

## D. Generic gates that must stay GREEN (they are the re-validation procedure)

- `tests/test_workflow_live_passes_validator.py:41-47` --
  `validate_workflow_contract(wf, mappings)`. This runs the six checks in
  `nodes/_workflow_validation.py:168-370`: unknown OTR node types (strict only),
  ROGUE SOCKETS (every wired input name must be declared by `INPUT_TYPES`),
  POSITIONAL WIDGET DRIFT, deleted-node tombstones, forbidden input sockets, and
  the LINK-TABLE BATTERY (6-element tuple shape, duplicate link ids,
  `last_link_id == max(link_ids)`, the reserved-id collision set `{111, 112}`,
  and ORPHAN-LINK referential integrity).
  **Caveat that must go in the doc:** this test runs with
  `strict_unknown_types=False`, so if `OTR_DynamicStoryDirection` fails to IMPORT
  in the bare test env, its type is silently SKIPPED here. A green run of this
  test is NOT proof the class registered.
- `tests/test_workflow_graph_integrity_guards.py` -- widget-vector drift across
  ALL nodes (only `OTR_MusicGenTheme` is whitelisted, :44-50); link source-slot
  bounds; and **output-link reconciliation**: a node's `outputs[N].links` list
  must exactly match the central `links[]` table. If the delta repoints 252/255
  in `links[]` but leaves them in `62.outputs[1].links`, THIS is what fires.
- `tests/test_workflow_contract_validation.py:41` -- builds the node-class
  mappings by AST-parsing the LITERAL `_NODE_MODULES` dict in `__init__.py`.
  A node registered only via `nodes/_otr_class_registry.py` (merged at
  `__init__.py:335-349`) is INVISIBLE here -> the new node MUST go in the literal
  dict.
- `tests/test_core.py:410-415` -- `last_node_id >= max(node ids)` and
  `last_link_id >= max(link ids)`.
- `tests/test_workflow_link_target_indexes.py` -- every `inputs[].link` has a
  central row whose `dst_node`/`dst_slot` match.
- `tests/test_full_workflow_v2_audio_wiring.py:149, 220-232` -- the audio trio
  and CastLock stay on the raw freeze outputs.
- `tests/test_otr_workflow_validator.py:288-306` -- transitivity pins
  `87 -> 90.video_policy_json` and `90 -> 91.script_json`. Survives (only 90's
  `script_json` SOURCE changes), but it breaks if the plan touches ShotLock's
  outputs.
- `OTR_WorkflowValidator` node itself (`nodes/_otr_workflow_validator.py:183`,
  `validate()` at :394-480) + `_assert_stamp()` (:296-392): the
  `semantic_master_hash` drift tripwire. `validate_anyway` can NEVER skip it
  (:400-408). Any workflow edit must be followed by a validator run + a JSON
  round-trip + the widget/link audit (CLAUDE.md section 0).

## E. Questions put to the panel

1. Is the REPOINT-don't-renumber delta (section B) correct and complete? Name any
   field of the litegraph node record I omitted that the frontend or the validator
   requires (`order`, `slot_index`, `shape`, `flags`, `properties`, `title`).
2. Does `order` need to be hand-authored, or is it recomputed? If the graph is
   executed via the API (`scripts/otr_api.py`) rather than the frontend, does a
   stale `order` matter?
3. Enumerate EVERY test in the repo that a new node id 96 / link id 284 / the
   62-out[1] fan-out change would fail. I found two hard pins (section C) and the
   generic gates (section D). What did I miss?
4. Is placing the new node's `done` and `direction_report` outputs UNWIRED
   acceptable to the link/widget audit, given ShotLock's own `done` already ships
   with `links: []`?
5. Does anything OTHER than nodes 89 and 90 read `62.out[1]` in a way that would
   need the DIRECTED (post-direction) ledger rather than the raw freeze json?
   Specifically: OTR_SignalLostVideo (12) -- does it touch `meta.visual_style`?
6. What is the correct re-baseline procedure for the canonical workflow after this
   delta (validator, round-trip, widget audit, master hash), in the exact order
   Codex should run it?
