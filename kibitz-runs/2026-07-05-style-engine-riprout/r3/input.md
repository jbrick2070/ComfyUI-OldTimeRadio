# OTR style engine consolidation — 100% rip-out plan v4 (ALL DECISIONS LOCKED)

Operator directive (2026-07-05): ONE internal engine drives the plot/style,
fully automatic, zero manual picker. Full rip-out: no fallback, no
back-compat shims, no trace of the retired system, no negative tests, and
any OTHER dead code found gets removed 100% too (safely verified first).

ALL open decisions are now LOCKED:
- `style` combo AND `style_custom` free-text: **DELETE BOTH ENTIRELY.** No
  repopulated 100-item combo, no `forced_slug`/`label_override` API. Every
  episode's tone/climax/sound-world comes from ONE unchanged call:
  `build_story_contract(cast_seed, script_brief, news_seed, meta) ->
  StoryContract`.
- `story_scaffold` (auto/on/off widget): **KEPT** as a legitimate creative
  mode, unchanged, positional slot untouched aside from reindexing caused
  by the two deletions above.
- `_otr_style_palette.py` + its drift test: **DELETE outright** (confirmed
  dead relative to runtime; safe-removal grep for dynamic/reflective
  access already run and clean).
- Bank/pipeline gating (`science_news` vs `media_archive` vs
  `public_domain_story` vs `custom_source_bank`): **OUT OF SCOPE this
  sprint.** Only `science_news` is runnable today. Leave a plain doc note
  at the `build_story_contract()` call site; no gating code.

## 0. Hard constraints

- One engine only, fully automatic, no manual override anywhere.
- No fallback, no trace, no negative tests.

## 1. Target end-state — FINAL

Both `style` (combo, `widgets_values[8]`) and `style_custom` (STRING,
`widgets_values[9]`) are DELETED together in one edit — adjacent slots, one
combined widget-removal + reindex pass. Everything from old index 10
(`creativity`) through 26 (`visual_style`) shifts down by TWO.
`resolved["style"]`/`style_combo`/`style_custom`/`style_source`/
`style_pending`/`llm_auto` — the entire three-way `_resolve_inputs`
resolver branch for style is DELETED outright, not simplified.
`meta.gen_params_initial`'s four style fields (`style`, `style_combo`,
`style_custom`, `style_source`) are all deleted; the only surviving style
record is `meta.story_contract` (already exists, already
freeze-consistent).

## 1a. `story_scaffold` widget — KEPT

`story_scaffold` (`OTR_LedgerScriptWriter.py:2244-2260`, combo
`["auto","on","off"]`) mutates `OTR_ENABLE_STYLE_GRAMMAR` via
`_apply_story_scaffold_env` (line 1710). KEPT as a real, named creative
mode ("scaffold off" = the writer's own unshaped take) — not a fallback.
`_style_grammar_on` (`OTR_LedgerScriptWriter.py:2819`) stays a real branch.
Old positional index 24 -> NEW index 22 after the two deletions above.

## 1b. Scope: `science_news` bank ONLY this sprint

`build_story_contract()` has zero bank/pipeline awareness and would fire
identically for any of the 4 registered banks in
`nodes/story_packs/banks.json` (`science_news` runnable; `media_archive`,
`public_domain_story`, `custom_source_bank` not yet runnable). Only
`science_news`'s bank config requires the style-pick seams; the others do
not, and `custom_source_bank`'s pipeline (`simple_4_prompt_experimental`)
has no style concept at all. DECIDED: ship for `science_news` only, leave
a plain doc comment at the call site, no gating code — revisit when a
second bank goes runnable.

## 2. Delete outright (zero trace)

- `nodes/_otr_style_picker.py` — the whole file (2-pass LLM inventor).
- `OTR_LedgerScriptWriter.py`: `_STYLE_CHOICES`, `_STYLE_PICKER_SEED_POOL`,
  `_LLM_STYLE_FALLBACK`, `pick_style(...)` call site (~line 2995) + RNG
  plumbing if orphaned, `meta["style_pick"]` stamp, stale NOTE comment
  (~806-815), the entire `style` + `style_custom` widgets/branches.
- A THIRD inline copy of the old 10-slug list hardcoded inside
  `_fetch_rss_seed_or_die` (`OTR_LedgerScriptWriter.py:1160-1175`), with a
  hardcoded fallback to `"mission_control_procedural"` for any unmatched
  slug. Since there is no combo value feeding this function anymore (the
  combo is deleted), strip `style` from the RSS fetch/rerank contract
  entirely — `_fetch_rss_seed_or_die` should no longer take a `style`
  parameter at all.
- Story-pack schema/content seams: `nodes/_otr_story_pack.py:40-43`
  allowlists `style_pick_inventor_system` / `style_pick_inventor_user` /
  `style_pick_chooser_system` / `style_pick_chooser_user`; same strings in
  `nodes/story_packs/banks.json`, `nodes/story_packs/pipelines.json`,
  `nodes/story_packs/science_news/science_news_default.json`.
- The existing fallback violation: `build_story_contract()`'s call site
  swallows exceptions into `contract = None`
  (`OTR_LedgerScriptWriter.py:3357-3362`), and the climax-shape block
  performs a SECOND `select_style()` draw when `contract is None`
  (`OTR_LedgerScriptWriter.py:3587-3596`). Both removed — an invalid
  catalog state must fail loud.
- `_otr_style_palette.py` + `tests/test_style_palette_drift.py` — DELETE
  outright (confirmed dead; safe-removal grep clean).
- Grep sweep (zero hits) across `nodes/`, `tests/`, `nodes/story_packs/*.json`:
  `_otr_style_picker`, `pick_style`, `StylePick`, `StyleGenerationFailedError`,
  `_STYLE_PICKER_SEED_POOL`, `_LLM_STYLE_FALLBACK`, `style_pick`,
  `STYLE_PALETTE`, `_otr_style_palette`, `style_custom`, `_STYLE_CHOICES`.
  Confirmed test-side referencers: `tests/test_otr_style_picker.py`,
  `tests/test_pick_style_routing.py`, `tests/test_helper_paired_
  signatures.py`, `tests/test_audio_byte_identical.py`, `tests/test_
  story_pack_stage1.py`, `tests/test_writer_paired_wiring.py`, `tests/
  test_meta_slot_transitions.py`, `tests/test_style_palette_drift.py`.

## 4. Workflow JSON (positional widgets — TWO adjacent slots removed)

`workflows/otr_scifi_16gb_full.json`, `OTR_LedgerScriptWriter` node: DELETE
`widgets_values[8]` (`style`) and `[9]` (`style_custom`) together. Old
index 10 (`creativity`) onward shifts down by TWO. `story_scaffold` (old
24) -> new 22; `source_bank` (old 25) -> new 23; `visual_style` (old 26)
-> new 24. Update `tests/test_workflow_json_guardrails.py`: delete the
`wv[8] == "let the story decide"` assertion entirely (no combo left);
`wv[24] == "auto"` becomes `wv[22] == "auto"`; the `expected 27`
widgets_values-length assertion becomes `expected 25`. Re-validate:
`OTR_WorkflowValidator` + JSON round-trip + `TestWidgetOrderVsInputTypes`
(the general BUG-LOCAL-097 guard) + link referential integrity.

## 5. Ledger / meta schema

`meta.story_contract` already exists, kept as-is (freeze-consistent via
`_otr_ledger_consistency.py`'s existing matrix row). `meta.style_pick` and
all four `gen_params_initial` style fields are deleted. Already-rendered
episodes on disk keep historical values untouched.

## 7. Sequencing — one atomic cleanbreak sprint (decisions locked)

1. Delete `style` + `style_custom` widgets/inputs, the `_resolve_inputs`
   style resolver branch, and the ledger's four style fields.
2. Delete dead modules/constants/JSON seams (section 2) — grep sweep clean
   across `nodes/`, `tests/`, `nodes/story_packs/`.
3. Delete `_otr_style_palette.py` + its drift test outright.
4. Rewrite tests — positive pins only.
5. Re-validate + re-freeze the workflow JSON (section 4) — two adjacent
   slot deletions, full downstream reindex, both test assertions updated.
6. Add the doc-only bank/pipeline scope note at the `build_story_
   contract()` call site — no gating code.
7. Full regression suite + Bug Bible.
8. Commit AND push to `v2.0-alpha` in the same session.

## 8. Risk / blast radius

- Positional-widget REMOVAL risk: TWO adjacent slots deleted in one pass,
  full downstream reindex of 17 widgets plus two hardcoded test-index
  rewrites — the biggest mechanical risk this sprint.
- C7 determinism: confirm the single-draw contract stays
  cast_seed-keyed/reproducible.
- Bank/pipeline scope is explicitly deferred — make sure the doc-note
  actually lands at the call site.

## 9. Explicit asks for THIS round (r3 — wiring / integration / sequencing)

1. Confirm the exact CURRENT widget order/index for `style` and
   `style_custom` in `INPUT_TYPES()` (are they really adjacent at [8]/[9],
   or has anything shifted since the last kibitz round touched this file?)
   and confirm the full downstream widget list (names + old indices 10
   through 26) so the reindex math above is exactly right, not
   approximate.
2. `_fetch_rss_seed_or_die` currently REQUIRES a `style` parameter
   (`OTR_LedgerScriptWriter.py:1138`). If style is stripped from its
   signature entirely, trace every call site of this function to confirm
   none of them still need to pass something — what does the function do
   with `style` besides the rerank slug normalization, and is removing the
   parameter a clean, self-contained change or does it ripple into
   `story_orchestrator.py`'s `_fetch_science_news` (which also takes a
   `style=` kwarg per the citation)?
2. Sequencing: can steps 2 (delete dead modules) and 3 (delete palette)
   safely run before step 1 (delete widgets/resolver) is fully landed, or
   does the deletion sweep depend on the widget rewire being done first
   (e.g., does `_resolve_inputs` still reference something in step 2's
   deletion list until step 1 completes)?
3. Any hidden dependency between the doc-only bank-scope note (section 1b,
   step 6) and the actual code deletions — i.e., does leaving a comment at
   the `build_story_contract()` call site require the call site to still
   exist in a stable, unchanged form, and does step 1's rewiring touch that
   exact call site in a way that could clobber or misplace the comment?
4. Any other wiring/sequencing gap in sections 1-8.

---

COMFYUI CUSTOM-NODE PROFILE (append to each round prompt)

When the target repo is a ComfyUI custom-node pack, also verify the
domain invariants below. Cite the real node file/class for every claim; if you
cannot see the code, write "verify: <what>" rather than asserting it. These are
ComfyUI-specific failure modes the general rounds do not weight.

1. NODE-CLASS CONTRACT
   - Every exported node class is registered in NODE_CLASS_MAPPINGS (and given a
     label in NODE_DISPLAY_NAME_MAPPINGS). A class that is defined but not mapped
     is dead -- it never loads. Flag any node in the plan that is not wired into
     the mappings.
   - INPUT_TYPES is a @classmethod returning a dict with "required" (and optional
     "optional"/"hidden") keys. Each input is (TYPE, {options}) where TYPE is a
     real type string ("IMAGE", "LATENT", "MODEL", "CONDITIONING", "STRING",
     "INT", "FLOAT", ...) or a list-of-choices for a dropdown.
   - RETURN_TYPES is a tuple (note the trailing comma for a single output),
     length-matched to what FUNCTION actually returns; RETURN_NAMES if present
     must be the same length. CATEGORY and FUNCTION must be set.
   - Widget order is POSITIONAL: appending an optional input is safe; inserting
     one mid-list silently shifts every saved widget value in existing graphs.

2. TENSOR LAYOUT / SHAPE CONVENTIONS
   - ComfyUI IMAGE tensors are float32 in [0,1], shape [B, H, W, C] (channels
     LAST), C usually 3. MASK is [B, H, W]. LATENT is a dict {"samples": tensor}
     with the model's own channel layout (commonly [B, C, H, W]). Flag any node
     that assumes channels-first for IMAGE or forgets the batch dim.
   - Check dtype/device handling: tensors may arrive on cuda or cpu; a node must
     not hard-assume one. Verify .to(device)/.cpu() moves are correct and that
     outputs match the declared RETURN_TYPES layout.

3. VRAM / MODEL MANAGEMENT
   - Heavy models should load through comfy.model_management (residency, offload,
     and eviction are managed there) rather than being pinned in module globals.
     Flag any plan that holds a model resident across runs without an eviction or
     free path, or that bypasses model_management's load/offload.
   - Verify large allocations are freed (or handed to model_management) so a
     long ComfyUI session does not leak VRAM across queued prompts.

4. IS_CHANGED / CACHING CORRECTNESS
   - ComfyUI caches a node's output and re-runs only when inputs change. If a
     node depends on external state (a file on disk, a clock, RNG, a network
     fetch), it must implement IS_CHANGED to return a value that varies when that
     state changes -- otherwise it serves stale cached output. Flag any node with
     hidden external inputs and no IS_CHANGED, and any IS_CHANGED that is more
     conservative/looser than the node's real dependencies.

5. IMPORT ISOLATION (no heavy imports at module top)
   - The module top level is imported at ComfyUI startup. Heavy or optional
     dependencies (torch extras, model libraries, CUDA ext) imported at top level
     slow every boot and hard-crash startup if missing. Move them inside the
     node method (lazy import) so an unrelated missing dep cannot take down the
     whole node pack. Flag top-level imports of optional/heavy packages.
   - Side effects at import time (downloading weights, opening files, mutating
     global state) are a defect -- they run for every user on every boot.
