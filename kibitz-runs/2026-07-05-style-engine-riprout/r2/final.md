# r2 JUDGMENT (Cowork Claude, anchor + judge) -- style engine rip-out. r2: CODING PLAN.

Panel this round: Codex (gpt-5.5, high reasoning) -- OK, full grounded review,
every claim checked against the real files and CONFIRMED, zero misreads.
Antigravity -- again produced NO output (antigravity.log 0 bytes) after
several minutes; process terminated. Two-for-two failure to respond across
r1 and r2. Dropping antigravity for the remainder of this arc; flagging to
the operator as a tooling issue to debug separately (likely a first-run
interactive sign-in `agy` needs that a headless run can't satisfy), not a
plan defect.

Codex's r2 review is exceptionally strong -- it found a real fifth/sixth
style-slug surface neither r1 nor the original analysis caught, and it
correctly identifies that the CURRENT code's own error handling philosophy
(fail-soft, "never break the writer") directly contradicts the operator's
"no fallback" directive for this specific engine. Every citation verified.

## Grounded and ACCEPTED (all verified against the real files)

1. **New style-slug surface #5: `_fetch_rss_seed_or_die` hardcodes a THIRD
   inline copy of the old 10-slug list** (`OTR_LedgerScriptWriter.py:1160-
   1175`), with a hardcoded fallback to `"mission_control_procedural"` for
   ANY slug not in that inline set -- confirmed by direct read. This means
   even after the rip, if a catalog slug from the new 100-set reaches this
   function unchanged, it silently coerces to one hardcoded style for the
   RSS re-rank step -- exactly the "no fallback" violation the operator
   banned, hiding in a function the original plan never looked at. MUST-FIX:
   either strip style entirely from the RSS fetch/rerank contract, or write
   an explicit catalog-slug -> rerank-slug mapping (fail loud on an unmapped
   slug, per the no-fallback rule) before deleting `_LLM_STYLE_FALLBACK`.

2. **New style-slug surface #6: `story_packs` allowlists/JSON reference
   `style_pick_*` seams.** Confirmed: `nodes/_otr_story_pack.py:40-43` lists
   `style_pick_inventor_system` / `style_pick_inventor_user` /
   `style_pick_chooser_system` / `style_pick_chooser_user`; `style_pick`
   also appears in `nodes/story_packs/banks.json`,
   `nodes/story_packs/pipelines.json`, and
   `nodes/story_packs/science_news/science_news_default.json`. Section 2's
   deletion sweep must extend to story-pack schema/content, not just Python
   modules and pytest files -- this is a third category (config/data) the
   plan hadn't covered.

3. **The plan's own "no fallback" constraint is violated by EXISTING code
   the plan proposed to keep.** Confirmed: the try/except around
   `build_story_contract()` swallows any exception and continues with
   `contract = None` (`OTR_LedgerScriptWriter.py:3357-3362`), and the
   climax-shape block then performs a SECOND, independent `select_style()`
   draw as a defensive fallback when `contract is None`
   (`OTR_LedgerScriptWriter.py:3587-3596`, matches what I read directly in
   r1 prep). This is precisely a second engine hiding behind an exception
   handler. MUST-FIX: remove these broad `except Exception: contract = None`
   /defensive-redraw fallbacks for the style path specifically -- an invalid
   catalog/style state must fail loud, consistent with section 0's "no
   fallback," even though this pattern is otherwise a correct, praised
   defensive style elsewhere in this codebase (CLAUDE.md's general "never
   break the writer" principle does not override an explicit operator
   directive for THIS specific engine).

4. **Workflow widget indices confirmed exactly.** `story_scaffold` is
   `widgets_values[24]` (0-indexed; ships `"auto"`), `source_bank` is [25],
   `visual_style` is [26] -- confirmed live in
   `tests/test_workflow_json_guardrails.py:679-736` (both the historical
   comment trail and the literal `assert wv[24] == "auto"`). If section 1a
   resolves to deleting `story_scaffold`, ALL THREE of these positional
   slots need auditing together, not just slot 8 (the `style` combo) as
   originally scoped.

5. **Concrete API contract for the catalog engine, accepted as proposed:**
   `build_story_contract(cast_seed, script_brief, news_seed, meta, *,
   forced_slug=None, label_override=None) -> StoryContract`. When
   `forced_slug` is set, resolve through `get_style()` and RAISE on an
   unknown slug (today `get_style()` returns `None` on a miss -- the new
   fail-loud contract is a real, small, well-scoped code change). Sentinel
   keeps the existing hash-draw path. `style_custom` becomes
   `label_override` only (custom TONE TEXT), never a second slug axis.

6. **Ledger canonical field, accepted as proposed:** `resolved["style"]`
   stays the catalog SLUG (snake_case, freeze-validated, unchanged
   contract with `_otr_ledger_freeze.py`); add `resolved["style_label"]` /
   `meta.style_label` only if a human-readable string is still needed
   downstream. `style_custom`, if kept, populates the label field, never
   the slug field.

7. **CUT recommendation (Codex, stronger than the original plan's "decide
   a or b"): retire `style_custom` as a free-text PRIMARY path entirely.**
   Rationale, verified sound: it is the single biggest source of the
   slug/label ambiguity in finding #6, and a label-override-only design
   (accepted item #5) already gives power users a custom tone string
   without reopening a second slug axis. This is the panel's recommendation
   to the operator, not yet a decision -- surface it plainly at the next
   operator checkpoint rather than quietly picking (a) or (b) from the
   original plan.

8. **Positive-pin test list, accepted as the section 7 test spec:** explicit
   combo slug builds the same `StoryContract.slug`; sentinel hash-draw stays
   deterministic; label override does not change the slug; `meta.
   gen_params_initial.style == meta.story_contract.slug`; no `meta.
   style_pick` anywhere; full workflow positional-slot audit (slots 8, 24,
   25, 26) passes.

9. **Regression cadence:** align with CLAUDE.md's real rule -- regression +
   Bug Bible after every green chunk, not only once at the very end, unless
   the operator explicitly declares this one indivisible atomic edit.
   Accepted; the plan's section 7 phrasing is loosened to allow either,
   operator's call at build time.

10. **`__pycache__` exclusion** for the deletion/grep sweep -- accepted,
    trivial correctness fix (target tracked `.py`/`.json`/workflow files
    only).

## Rejected / not folded

- None. Every Codex claim this round checked out on direct grounding —
  zero misreads, zero hallucinations.

## Convergence statement

r2 substantially deepened the blast radius (two more style-slug surfaces
found: the inline RSS-rerank list, and the story-pack JSON/prompt-seam
allowlist) and surfaced a real contradiction between the operator's "no
fallback" directive and the codebase's existing (otherwise-good) defensive
error handling. This is coding-plan-level progress, not just discovery --
the plan now has a concrete function signature, a concrete field-naming
resolution, and a concrete CUT recommendation on `style_custom`. NOT yet
converged: sections 1 and 1a still need an explicit operator decision
before r3 (wiring) can be meaningfully planned, since wiring depends on
which widgets survive. Carry all ten accepted items into r3.
