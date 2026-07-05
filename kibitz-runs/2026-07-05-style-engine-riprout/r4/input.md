# OTR style engine consolidation — 100% rip-out plan v5 (post r1/r2/r3 kibitz)

This is the FINAL convergence round (r4). Three prior rounds already hardened
this plan (r1: killed a stale ~90-cue MusicGen cost item, found the
`story_scaffold` widget; r2: found the RSS-rerank ripple, the existing
fallback-violating exception handler, story-pack JSON seams; r3: found TWO
build-breaking sequencing bugs — `lock_cast()` reading the deleted
`resolved["style"]` before the contract exists, and a circular dependency
between `news_interpreter` and `build_story_contract()` — plus widened the
RSS ripple into `story_orchestrator.py`/`_otr_source_payload.py` and found
`meta.style` has more live readers than assumed).

ALL decisions are LOCKED: one automatic engine only (`build_story_contract`,
unchanged signature, zero new params), both `style` + `style_custom` widgets
DELETED entirely, `story_scaffold` KEPT, `_otr_style_palette.py` DELETED,
bank/pipeline gating OUT OF SCOPE this sprint (doc-note only for
`science_news`).

Your job this round: find anything that survived r1-r3 and still needs
fixing, and catch any regression the r3 fixes themselves introduced. Read
the FULL current plan below — it is the complete, self-contained document.

---

## 0. Hard constraints (non-negotiable per operator)

- One engine only: `nodes/_otr_style_catalog.py` (`STYLE_CATALOG` /
  `build_story_contract` / `select_style`) becomes the SOLE source of both
  the tone/style label AND the climax shape + sound world.
- No fallback, no trace, no negative tests.

## 1. Target end-state — FINAL

**DECIDED: exactly ONE style generator, fully automatic, zero manual
picker.** Both `style` (combo) and `style_custom` (free-text) are DELETED
ENTIRELY. Every episode's tone label, climax shape, and sound-world
grammar come from exactly one call: `build_story_contract(cast_seed,
script_brief, news_seed, meta) -> StoryContract`, unchanged signature,
zero new parameters.

- Delete `style` widget (combo, `widgets_values[8]`) and `style_custom`
  widget (STRING, `widgets_values[9]`) together, in the SAME edit.
- `resolved["style"]`/`style_combo`/`style_custom`/`style_source`/
  `style_pending`/`llm_auto` — the entire three-way `_resolve_inputs`
  resolver branch for style is DELETED, not simplified.
- r3 CRITICAL: `build_story_contract()` must move EARLIER in `run()`.
  Confirmed: `_OTRCAST.lock_cast(..., style=resolved["style"], ...)` fires
  at `OTR_LedgerScriptWriter.py:3193-3198`, ~150 lines BEFORE
  `build_story_contract()` is currently called (`:3337-3345`). Fix: call
  `build_story_contract()` right after `script_brief` and `cast_seed` both
  exist (~line 3174, before `lock_cast`), and thread `contract.label`/
  `.slug` into `lock_cast` and every other caller currently reading
  `resolved["style"]`.
- r3 CRITICAL: `news_interpreter` has a circular dependency with the
  contract. `build_news_briefs()` (via `_otr_source_payload.py:233-259`,
  prompt text at `nodes/news_interpreter.py:719,731-740`) takes a `style`
  param and runs BEFORE `script_brief` exists — but `build_story_contract()`
  needs `script_brief` as an input. Fix: strip `style` from
  `build_news_briefs()`/`news_interpreter.py`'s prompt entirely.
- r3: `meta.style` has more live readers than assumed. The writer also
  stamps `meta["visual_plan"]["style"]` and `meta["style"]`
  (`OTR_LedgerScriptWriter.py:5631-5636`); `nodes/_otr_story_brief.py:565`
  emits `STYLE: {meta.get('style')}`; the freeze validator audits
  `meta.style` (`nodes/_otr_ledger_freeze.py:582-592`). Fix: keep a
  canonical `meta.style` field DERIVED from `meta.story_contract.slug`/
  `.label` (a one-line addition) so every existing reader keeps working.
- Ledger field canonicalization: `meta.gen_params_initial` currently
  stamps `style`, `style_combo`, `style_custom`, `style_source`
  (`OTR_LedgerScriptWriter.py:5505-5508`); ALL FOUR are deleted. The only
  surviving style record is `meta.story_contract`.

## 1b. Scope: `science_news` bank ONLY for this sprint

`build_story_contract()` has ZERO bank/pipeline awareness — it would fire
identically for any of the 4 registered source banks
(`nodes/story_packs/banks.json`: `science_news` [runnable], `media_archive`,
`public_domain_story`, `custom_source_bank` [not yet runnable]).
`science_news`'s bank config requires the (soon-dead) style-pick seams; the
others do not, and `custom_source_bank`'s pipeline
(`simple_4_prompt_experimental`) has no style concept at all.

**DECIDED: out of scope this sprint.** Ship for `science_news` only. Leave
a plain doc note (not code) at the `build_story_contract()` call site: this
engine is the `science_news` default; enabling any other bank must
explicitly decide whether to opt in. No gating code.

r3 sequencing note: since section 1's fix moves the `build_story_contract()`
call EARLIER, this doc comment must land at the call site's NEW resting
place, added AFTER the call site is moved.

## 1a. `story_scaffold` widget — KEPT

`story_scaffold` (`OTR_LedgerScriptWriter.py:2244-2260`, combo
`["auto","on","off"]`) mutates `OTR_ENABLE_STYLE_GRAMMAR` via
`_apply_story_scaffold_env` (line 1710). **DECIDED: KEPT** as an
intentional creative option — "scaffold off" is a legitimate, named,
symmetric story mode, not a silent fallback. `_style_grammar_on`
(`OTR_LedgerScriptWriter.py:2819`) stays a real branch. Old positional
index 24 -> new index 22 after the two widget deletions above.

## 2. Delete outright (zero trace)

- `nodes/_otr_style_picker.py` — the whole file (2-pass LLM inventor).
  r3: TWO import sites — `OTR_LedgerScriptWriter.py:2797` (primary) AND a
  second import inside an in-file smoke-test helper at `:6103-6155`, plus
  the call site (`:2994-3005`) and a phase telemetry stamp (`:5545-5549`).
  Delete ALL in the SAME edit as the file deletion.
- `OTR_LedgerScriptWriter.py`: `_STYLE_CHOICES`, `_STYLE_PICKER_SEED_POOL`,
  `_LLM_STYLE_FALLBACK`, RNG plumbing if orphaned (`_resolve_style_rng_seed`,
  `picker_rng`), `meta["style_pick"]` stamp, stale NOTE comment (~806-815).
- r2+r3: a THIRD inline copy of the old 10-slug list hardcoded inside
  `_fetch_rss_seed_or_die` (`OTR_LedgerScriptWriter.py:1160-1175`), fallback
  to `"mission_control_procedural"`. Confirmed (both Codex and a Sonnet
  subagent, independently) this ripples wider: `_fetch_rss_seed_or_die`
  isn't called directly from the writer — its real caller is
  `nodes/_otr_source_payload.py:219-230`'s `_fetch_science_rss(*, bank,
  style_slug, technical_model)`; `_resolve_inputs` passes `style_slug=` at
  `OTR_LedgerScriptWriter.py:1404-1408`; downstream,
  `nodes/story_orchestrator.py` uses `style` for LLM rank-prompt text at
  FOUR call sites (`:1670-1682`, `:1843-1849`, `:1934-1940`, `:1957-1964`).
  Strip `style` from the RSS fetch/rerank contract entirely, all the way
  down through `story_orchestrator.py`.
- r2: story-pack schema/content carries dead-picker seams:
  `nodes/_otr_story_pack.py:40-43` (`style_pick_inventor_system/user`,
  `style_pick_chooser_system/user`); same strings in
  `nodes/story_packs/banks.json`, `pipelines.json`,
  `science_news/science_news_default.json`.
- r2: existing fallback-violating code: `build_story_contract()`'s call
  site swallows exceptions into `contract = None`
  (`OTR_LedgerScriptWriter.py:3357-3362`), and the climax-shape block
  performs a SECOND `select_style()` draw when `contract is None`
  (`:3587-3596`). Both removed — must fail loud.
- `_otr_style_palette.py` + `tests/test_style_palette_drift.py` — DELETE
  outright (confirmed dead relative to runtime music composition, which
  reads `meta` brief fields via `nodes/_otr_music_prompt.py:76-99`).
  Safe-removal grep for dynamic/reflective access already run, clean.
- Grep sweep (zero hits) across `nodes/`, `tests/`, `nodes/story_packs/*.json`:
  `_otr_style_picker`, `pick_style`, `StylePick`, `StyleGenerationFailedError`,
  `_STYLE_PICKER_SEED_POOL`, `_LLM_STYLE_FALLBACK`, `style_pick`,
  `STYLE_PALETTE`, `_otr_style_palette`, `style_custom`, `_STYLE_CHOICES`.
  Confirmed test-side referencers: `tests/test_otr_style_picker.py`,
  `tests/test_pick_style_routing.py`, `tests/test_helper_paired_signatures.py`,
  `tests/test_audio_byte_identical.py`, `tests/test_story_pack_stage1.py`,
  `tests/test_writer_paired_wiring.py`, `tests/test_meta_slot_transitions.py`,
  `tests/test_style_palette_drift.py`.

## 4. Workflow JSON (positional widgets — TWO adjacent slots removed)

`workflows/otr_scifi_16gb_full.json`, `OTR_LedgerScriptWriter` node: DELETE
`widgets_values[8]` (`style`) and `[9]` (`style_custom`) together. Old
index 10 onward shifts down by TWO. Confirmed live full widget order:
`[8] style, [9] style_custom, [10] creativity, [11]
perfect_run_spacesaver, [12] min_p, [13] repetition_penalty, [14]
max_new_tokens_cap, [15] lemmy_cameo, [16] use_exchange, [17]
enable_production_stage3_validators, [18] news_briefs_required, [19]
openrouter_slot_a_model, [20] openrouter_slot_b_model, [21]
comfy_slot_a_model, [22] comfy_slot_b_model, [23] refine_target_grade,
[24] story_scaffold, [25] source_bank, [26] visual_style`. Post-deletion:
length 25; `story_scaffold` -> [22], `source_bank` -> [23], `visual_style`
-> [24]. Update `tests/test_workflow_json_guardrails.py` (`wv[8] ==
"let the story decide"` deleted, `wv[24] == "auto"` -> `wv[22]`, length
27 -> 25, `_WRITER_STYLE_SLOT = 8` at line 358 updated/removed) AND
`tests/test_otr_api_companions.py:34-214,466`,
`tests/test_source_bank_widget_2c.py:322-323`,
`tests/test_visual_style_widget_3c.py:172-174`,
`tests/test_openrouter_slot_widgets_s2.py:62`,
`tests/test_writer_input_resolve.py` (AST pin on the fetcher's 2nd
positional arg). Re-validate: `OTR_WorkflowValidator` + JSON round-trip +
`TestWidgetOrderVsInputTypes` + link referential integrity.

## 5. Ledger / meta schema

`meta.story_contract` already exists, kept as-is (freeze-consistent via
`_otr_ledger_consistency.py`'s existing matrix row). `meta.style_pick` and
all four `gen_params_initial` style fields are deleted. A canonical
`meta.style` field, derived from `meta.story_contract.slug`/`.label`, is
ADDED so `_otr_story_brief.py` and the freeze validator keep working (r3
finding — this is new, not in the original plan). Already-rendered
episodes on disk keep historical values untouched.

## 7. Sequencing — one atomic cleanbreak sprint

1. Move `build_story_contract()`'s call site EARLIER (before `lock_cast`),
   and strip `style` from `news_interpreter.build_news_briefs()` — r3
   critical fixes, done FIRST.
2. Delete `style` + `style_custom` widgets/inputs and the `_resolve_inputs`
   resolver branch. Thread `contract.label`/`.slug` into `lock_cast` and
   every other former reader of `resolved["style"]`. Add the canonical
   `meta.style` derived field.
3. Strip `style` from the RSS fetch/rerank chain end-to-end
   (`_fetch_rss_seed_or_die`, `_otr_source_payload.py`, `story_orchestrator.py`).
4. Delete the dead modules/constants/JSON seams (section 2) — both
   `_otr_style_picker` import sites, the call site, the telemetry stamp,
   all in the same edit as the file deletion.
5. Delete `_otr_style_palette.py` + its drift test outright.
6. Rewrite tests — positive pins only, across the FULL widget-index test
   list (section 4) and `test_writer_input_resolve.py`'s AST pin.
7. Re-validate + re-freeze the workflow JSON (section 4).
8. Add the doc-only bank/pipeline scope note at the NEW call site (after
   step 1's move).
9. Full regression suite + Bug Bible.
10. Commit AND push to `v2.0-alpha` in the same session.

## 8. Risk / blast radius

- Positional-widget REMOVAL risk: TWO adjacent slots, full downstream
  reindex of 17 widgets plus test-index rewrites across 5+ test files.
- C7 determinism: confirm the single-draw contract stays
  cast_seed-keyed/reproducible.
- Bank/pipeline scope explicitly deferred — the doc-note must actually
  land at the call site.
- Two build-breaking sequencing bugs found in r3 (lock_cast ordering,
  news_interpreter circular dependency) — confirm the fixes above fully
  close both with no remaining gap.

---

COMFYUI CUSTOM-NODE PROFILE (append to each round prompt)

When the target repo is a ComfyUI custom-node pack, also verify the
domain invariants below. Cite the real node file/class for every claim; if you
cannot see the code, write "verify: <what>" rather than asserting it.

1. NODE-CLASS CONTRACT: every exported node class registered in
   NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS. INPUT_TYPES is a
   @classmethod with "required"/"optional"/"hidden" keys. RETURN_TYPES is a
   tuple, length-matched to FUNCTION's actual return. Widget order is
   POSITIONAL: appending is safe, inserting/removing mid-list shifts every
   saved widget value.
2. TENSOR LAYOUT: IMAGE is float32 [0,1], [B,H,W,C]; MASK is [B,H,W];
   LATENT is {"samples": tensor} [B,C,H,W]. Check dtype/device handling.
3. VRAM/MODEL MANAGEMENT: heavy models load through
   comfy.model_management, not pinned in module globals; verify eviction/
   free paths.
4. IS_CHANGED/CACHING: a node depending on external state (file, clock,
   RNG, network) must implement IS_CHANGED correctly.
5. IMPORT ISOLATION: no heavy/optional imports at module top level; lazy-
   import inside the node method; no side effects at import time.

