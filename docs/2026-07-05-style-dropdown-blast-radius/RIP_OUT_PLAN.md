# OTR style engine consolidation — 100% rip-out plan (draft for kibitz)

Operator directive (2026-07-05, verbatim intent): the story should have ONE
internal engine driving the plot/style. Too many things are influencing it
today. This plan is a full rip-out, not a migration: **no fallback paths, no
back-compat shims, no trace of the retired system left in runtime code, no
negative/back-compat tests.** Companion doc:
`docs/2026-07-05-style-dropdown-blast-radius/ANALYSIS.md` (root-cause map of
the current 4 disconnected style-slug surfaces).

Status: DRAFT, unbuilt. Going through kibitz (Codex + Antigravity local
panel, Cowork Claude as anchor+judge) before any code is touched.

## 0. Hard constraints (non-negotiable per operator)

- One engine only: `nodes/_otr_style_catalog.py` (`STYLE_CATALOG` /
  `build_story_contract` / `select_style`) becomes the SOLE source of both
  the tone/style label AND the climax shape + sound world. Today these are
  two independent draws feeding the same prompt (see ANALYSIS.md) — that
  duplication is itself the bug being killed.
- No fallback. No "if the new path fails, revert to the old list" shim.
  No env kill-switch that reverts to a DIFFERENT engine (see open question
  in section 8 on whether `OTR_ENABLE_STYLE_GRAMMAR` itself must go).
- No trace. Delete the retired modules/constants outright — not deprecate,
  not comment out, not rename with a `_legacy` suffix.
- No negative tests. Delete tests whose sole purpose was pinning the OLD
  system's behavior; write only positive pins on the new single-engine
  behavior. Do not add tests that assert "the old thing is gone" as a
  permanent regression guard — that is not what this repo's test suite is
  for, and it's dead weight the moment the rip lands.

## 1. Target end-state — FINAL (post pros/cons fan-out, 2026-07-05)

**DECIDED: exactly ONE style generator, fully automatic, zero manual
picker.** Both `style` (the combo) and `style_custom` (the free-text box)
are DELETED ENTIRELY — not repopulated, not kept as an override. Every
episode's tone label, climax shape, and sound-world grammar come from
exactly one call: `build_story_contract(cast_seed, script_brief,
news_seed, meta) -> StoryContract`, unchanged signature, zero new
parameters. This supersedes the earlier v2/v3 draft that proposed
repopulating the combo with 100 entries + adding `forced_slug`/
`label_override` — that design is CUT. Rationale (3-way Sonnet fan-out,
see `DROPDOWN_REMOVAL_PROS_CONS.md`): `build_story_contract()` never read
the combo's value even in the old design, the combo's own choices aren't
real catalog slugs, "let the story decide" was already the only value
ever shipped, and skipping `forced_slug`/`label_override` avoids building
a second style-resolution system that would need reconciling with the
first — exactly the class of bug this whole rip-out exists to kill.

- Delete `style` widget (combo, `widgets_values[8]`) and `style_custom`
  widget (STRING, `widgets_values[9]`) together, in the SAME edit — they
  are adjacent, so this is one combined widget-removal + reindex pass, not
  two separate ones. Every widget from old index 10 (`creativity`) through
  26 (`visual_style`) shifts down by two; both hardcoded test indices
  (`wv[8]`, `wv[24]`) must be updated in the same commit.
- `resolved["style"]`/`style_combo`/`style_custom`/`style_source`/
  `style_pending`/`llm_auto` — the entire three-way `_resolve_inputs`
  resolver branch for style is DELETED, not simplified. There is nothing
  left to resolve; `build_story_contract()`'s own `.label` is the tone
  string, full stop.
- **r3 finding (CRITICAL, build-breaking): `build_story_contract()` must
  move EARLIER in `run()`.** Confirmed: `_OTRCAST.lock_cast(...,
  style=resolved["style"], ...)` fires at `OTR_LedgerScriptWriter.py:
  3193-3198`, ~150 lines BEFORE `build_story_contract()` is currently
  called (`:3337-3345`). Deleting `resolved["style"]` per this section
  without moving the contract-build call earlier is an immediate crash at
  `lock_cast`. Fix: call `build_story_contract()` right after
  `script_brief` and `cast_seed` both exist (~line 3174, before
  `lock_cast`), and thread `contract.label`/`.slug` into `lock_cast` and
  every other caller currently reading `resolved["style"]`.
- **r3 finding (CRITICAL): `news_interpreter` has a circular dependency
  with the contract.** `build_news_briefs()` (via `_otr_source_payload.py:
  233-259`, prompt text at `nodes/news_interpreter.py:719,731-740`) takes
  a `style` param and runs BEFORE `script_brief` exists — but
  `build_story_contract()` needs `script_brief` as an input. There is no
  ordering that lets the contract feed `news_interpreter`. Fix: strip
  `style` from `build_news_briefs()`/`news_interpreter.py`'s prompt
  entirely (it's a pre-contract sourcing stage) rather than trying to
  thread a value that structurally cannot exist yet at that point.
- **r3 finding: `meta.style` has more live readers than assumed.** The
  writer also stamps `meta["visual_plan"]["style"]` and `meta["style"]`
  (`OTR_LedgerScriptWriter.py:5631-5636`); `nodes/_otr_story_brief.py:565`
  emits `STYLE: {meta.get('style')}`; the freeze validator audits
  `meta.style` (`nodes/_otr_ledger_freeze.py:582-592`). Fix: keep a
  canonical `meta.style` field DERIVED from `meta.story_contract.slug`/
  `.label` (a one-line addition) so every existing reader keeps working,
  rather than deleting the stamp with no replacement.
- **r4 finding (Codex, confirmed): prompt-facing vs ledger-facing fields
  must use `contract.label` vs `contract.slug` respectively, not
  interchangeably.** Confirmed live: `nodes/_otr_casting.py` (~line 350)
  builds a human casting prompt `f"Style: {style_str}"` — PROMPT-facing,
  wants prose. `nodes/_otr_story_brief.py:565` and the freeze validator's
  snake_case shape check treat `meta.style` as a controlled SLUG —
  LEDGER-facing. Explicit rule: every prompt-facing string (casting
  prompt, `OutlineRequest.style`/outline prompts) threads
  `contract.label`; every ledger/meta-facing field (`meta.style`,
  `meta.visual_plan.style`, `style_descriptor`) threads `contract.slug`.
  Do not mix the two or thread one value everywhere.
- **r4 finding (Codex, confirmed as pre-existing, documented here for
  clarity): `story_scaffold=off` already means "no `meta.style` stamped
  at all," and that is fine.** Confirmed by direct read
  (`OTR_LedgerScriptWriter.py:3330-3362`,
  `tests/test_announcer_kill2_c1.py`'s `TestWriterOffFlagLedgerMeta`):
  this is EXISTING, already-shipped, already-tested behavior, not a new
  gap introduced by this rip-out — `_style_grammar_on=False` means
  `contract` stays `None` by design (per the KILL-2 code comment: "OFF =>
  contract stays None => no meta.story_contract => byte-identical"), and
  the freeze validator already treats a missing `meta.style` as a WARNING
  only (`_otr_ledger_freeze.py:582-586`), not an error. No code change
  needed beyond what this section already does — just don't assume
  `meta.style` is always present.
- **r4 finding (Codex, REJECTED as scope-creep, kept as a documented
  boundary): section 0's "no fallback" rule does NOT extend to
  `_otr_style_catalog.py`'s own internal defensive helpers** —
  `ending_template_for()` (falls back to a default ending tag template
  for an unknown slug), `render_style_grammar()` (returns `""` for an
  unknown slug), and `build_story_contract()`'s documented "never raises
  on a missing style" are the catalog module's OWN intentional, already-
  shipped robustness, unrelated to the competing-selector system being
  killed. `tests/test_announcer_kill2_c1.py::test_missing_style_never_raises`
  pins this and is UNCHANGED — it is not a "negative test of the old
  system," it is a positive pin on the new engine's own contract.
- Ledger field canonicalization: `meta.gen_params_initial` currently
  stamps `style`, `style_combo`, `style_custom`, `style_source`
  (`OTR_LedgerScriptWriter.py:5505-5508`); ALL FOUR are deleted. The only
  surviving style record is `meta.story_contract` (slug/label/ending_tag/
  sound_world — already exists, already freeze-consistent per
  `_otr_ledger_consistency.py`'s existing matrix row).
- **r4 finding (confirmed by direct read): a SECOND freeze-validator block
  reads `meta.gen_params_initial.style` separately from the `meta.style`
  check already covered above.** `nodes/_otr_ledger_freeze.py:594-616`
  (comment: "S25 / MG-6 (BUG-LOCAL-216), relaxed by BUG-LOCAL-240")
  validates `meta.gen_params_initial.style`'s snake_case shape. Confirmed
  NOT build-breaking as-is — once the stamp is deleted,
  `gp_initial.get("style")` returns `None` and the check is a silent
  no-op (the `isinstance(gp_style, str) and gp_style` guard short-circuits)
  — but it becomes a dead check pointed at a field that no longer exists,
  which violates the no-dead-code directive. Fix: delete this block
  (lines 594-616) in the SAME edit as the `gen_params_initial` stamp
  deletion. Locate and update whatever test currently pins this block's
  behavior (search `_otr_ledger_freeze` tests for `gen_params_initial`)
  before landing — it is NOT yet in the section 4 test list because it's
  a freeze-validator test, not a widget-index test.

## 1b. Scope: `science_news` bank ONLY for this sprint (operator, 2026-07-05)

Investigation surfaced that `build_story_contract()` has ZERO bank/
pipeline awareness — it would fire identically for any of the 4 registered
source banks (`nodes/story_packs/banks.json`: `science_news` [runnable],
`media_archive`, `public_domain_story`, `custom_source_bank` [all three
not yet runnable]). `science_news`'s bank config requires the (soon-dead)
style-pick seams; the other two radio-drama banks do NOT list them as
required, and the fourth bank's pipeline (`simple_4_prompt_experimental`)
has no style concept at all — so the 100-catalog engine is not
automatically a universal fit once those banks go live.

**DECIDED: out of scope for this sprint.** Ship the one-engine rip-out for
`science_news` (the only runnable bank today) and do NOT add bank/pipeline
gating logic now — there is nothing else runnable to test it against.
Leave a plain doc note (not a code shim) at the `build_story_contract()`
call site: this engine is the `science_news` default; enabling any other
bank must explicitly decide whether to opt into this engine or build its
own, rather than silently inheriting it. Revisit when a second bank goes
runnable — do not preemptively build gating for banks that don't exist
yet, per the "don't build for a fork you'd never actually take" style
kibitz already applies to subagent fan-outs.

**r3 sequencing note:** since section 1's fix moves the
`build_story_contract()` call EARLIER (before `lock_cast`, not at its old
~3340 location), this doc comment must land at the call site's NEW
resting place, added AFTER the call site is moved — not before, or the
comment gets silently orphaned/misplaced during the rewire.

## 1a. `story_scaffold` widget — a real, already-shipped third control

r1 kibitz grounding surfaced a widget this plan initially missed entirely:
`story_scaffold` (`OTR_LedgerScriptWriter.py:2244-2260`, combo
`["auto","on","off"]`, added 2026-06-24, appended at the end of `optional`
per the BUG-LOCAL-097 positional convention). `_apply_story_scaffold_env`
(line 1710) mutates `OTR_ENABLE_STYLE_GRAMMAR` straight from this widget.
Its own tooltip says `off` = "a story drawn straight from the news seed...
no style catalog, no climax-shape grammar, no grounding gate -- the
writer's own take."

**DECIDED (operator, 2026-07-05): `story_scaffold` is KEPT** as an
intentional, documented creative option — "scaffold off" is a legitimate,
named, symmetric story mode (the writer's own unshaped take), not a silent
fallback to a retired dual-list system. It survives the "no fallback" rule
because it degrades to a clearly-labeled, deliberate alternative a user
picks on purpose, not a hidden revert path. Implication: `_style_grammar_on`
stays a real branch in the writer (it already is — confirmed at
`OTR_LedgerScriptWriter.py:2819`), and sections 2/7's "no fallback" sweep
applies to the STYLE-SELECTION mechanism (killing the picker/inline lists/
silent exception-swallow redraw), not to this widget's on/off duality,
which is a supported creative feature. The widget's positional slot
(`widgets_values[24]`, confirmed) is UNCHANGED by this rip.

## 2. Delete outright (zero trace)

- `nodes/_otr_style_picker.py` — the whole file (2-pass LLM inventor:
  Pass 1 candidate invention, Pass 2 chooser, `StyleGenerationFailedError`,
  `StylePick`, `pick_style`). **r3 finding (confirmed by both reviewers):
  TWO import sites, not one** — `OTR_LedgerScriptWriter.py:2797` (primary
  import) AND a second, easy-to-miss import inside an in-file smoke-test
  helper at `:6103-6155`, plus the call site (`:2994-3005`) and a phase
  telemetry stamp (`:5545-5549`). Delete ALL of these in the SAME edit as
  the file deletion — do not run any intermediate validation between the
  file delete and the writer cleanup, or ComfyUI fails to import the node
  at boot (ImportError on node registration).
- `OTR_LedgerScriptWriter.py`: `_STYLE_CHOICES`, `_STYLE_PICKER_SEED_POOL`,
  `_LLM_STYLE_FALLBACK`, the `pick_style(...)` call site (~line 2995) and
  its surrounding RNG plumbing (`_resolve_style_rng_seed`, `picker_rng`) if
  nothing else calls them, `meta["style_pick"]` stamp, the stale NOTE
  comment block (~lines 806-815) that narrates the now-doubly-dead
  `_generate_style_via_llm` ancestor.
- Any `style_pending` / `llm_auto` branch in `_resolve_inputs`.
- **r2+r3 finding: a THIRD inline copy of the old 10-slug list**, hardcoded
  inside `_fetch_rss_seed_or_die` (`OTR_LedgerScriptWriter.py:1160-1175`),
  with a hardcoded fallback to `"mission_control_procedural"` for any slug
  outside that set — a live "no fallback" violation. **r3 grounding
  (confirmed by BOTH Codex and a Sonnet subagent, independently) found
  this ripples much wider than one function:** `_fetch_rss_seed_or_die`
  isn't called directly from the writer — its real caller is
  `nodes/_otr_source_payload.py:219-230`'s `_fetch_science_rss(*, bank,
  style_slug, technical_model)`, documented as "the S31 B6 slot-label/id
  agreement invariant"; `_resolve_inputs` passes `style_slug=` at
  `OTR_LedgerScriptWriter.py:1404-1408`; downstream,
  `nodes/story_orchestrator.py` uses `style` for LLM rank-prompt text
  (`genre_human = (style or "sci-fi").replace("_", " ")`, ~line 1490) at
  FOUR call sites (`:1670-1682`, `:1843-1849`, `:1934-1940`,
  `:1957-1964`). This is NOT a self-contained parameter removal: the
  fetcher contract, the `_otr_source_payload.py` wrapper, the writer call,
  `story_orchestrator.py`'s ranking/history signatures, AND
  `tests/test_writer_input_resolve.py` (AST-asserts the 2nd positional-arg
  contract) all change together in the SAME edit. Strip `style` from the
  RSS fetch/rerank contract entirely, all the way down through
  `story_orchestrator.py` — do not leave the hardcoded
  `"mission_control_procedural"` default anywhere in that chain.
- **r2 finding: story-pack schema/content also carries the dead picker's
  seams** — `nodes/_otr_story_pack.py:40-43` allowlists
  `style_pick_inventor_system` / `style_pick_inventor_user` /
  `style_pick_chooser_system` / `style_pick_chooser_user`; the same
  `style_pick` strings appear in `nodes/story_packs/banks.json`,
  `nodes/story_packs/pipelines.json`, and
  `nodes/story_packs/science_news/science_news_default.json`. The deletion
  sweep must cover this config/data layer, not just Python + tests.
- **r2 finding: the plan's OWN "no fallback" rule is violated by existing
  code it proposed to keep unchanged.** `build_story_contract()`'s call
  site swallows any exception into `contract = None`
  (`OTR_LedgerScriptWriter.py:3357-3362`), and the climax-shape block then
  performs a SECOND, independent `select_style()` draw as a fallback when
  `contract is None` (`OTR_LedgerScriptWriter.py:3587-3596`). Both must be
  removed for the style-engine path specifically — an invalid catalog
  state must fail loud, per section 0, even though this defensive pattern
  is otherwise a correct, praised style elsewhere in this codebase.
- Grep sweep (must return zero hits before declaring done) across
  `nodes/`, `tests/`, AND `nodes/story_packs/*.json` (r1+r2 kibitz
  grounding found the picker referenced far wider than originally
  assumed): `_otr_style_picker`, `pick_style`, `StylePick`,
  `StyleGenerationFailedError`, `_STYLE_PICKER_SEED_POOL`,
  `_LLM_STYLE_FALLBACK`, `style_pick`.
  Confirmed test-side referencers to fold into this sweep:
  `tests/test_otr_style_picker.py`, `tests/test_pick_style_routing.py`,
  `tests/test_helper_paired_signatures.py`, `tests/test_audio_byte_identical.py`,
  `tests/test_story_pack_stage1.py`, `tests/test_writer_paired_wiring.py`,
  `tests/test_meta_slot_transitions.py`.
- **r4 finding (Codex, confirmed): two more test files were missing from
  the deletion/rewrite list, both verified live.**
  `tests/test_style_randomization.py` imports `_resolve_style_rng_seed`
  directly from `OTR_LedgerScriptWriter` (line 17) and pins its OS-entropy/
  `OTR_STYLE_SEED` contract (BUG-LOCAL-270 guard) — this IS the "nothing
  else calls them" caller of the RNG plumbing this section already marks
  for deletion; delete the whole file along with `_resolve_style_rng_seed`/
  `picker_rng`/`OTR_STYLE_SEED`, it exists solely to pin the retired
  system. `tests/test_news_briefs_required.py:34,43` passes `style_custom=`
  as a kwarg into a resolver call — strip that kwarg from both call sites
  once the resolver branch is deleted.
- **r4 finding (Sonnet, confirmed): a SECOND freeze-validator block reads
  `meta.gen_params_initial.style` separately from the `meta.style` check
  already covered above.** `nodes/_otr_ledger_freeze.py:594-616` ("S25 /
  MG-6 (BUG-LOCAL-216)") validates that field's snake_case shape.
  Confirmed NOT build-breaking as-is (once the stamp is gone,
  `gp_initial.get("style")` returns `None` and the guard short-circuits —
  silent no-op), but it becomes a dead check pointed at a field that no
  longer exists, which the no-dead-code directive forbids. Delete this
  block in the SAME edit as the `gen_params_initial` stamp deletion; find
  its pinning test at build time (grep `_otr_ledger_freeze` tests for
  `gen_params_initial` — no single file could be identified by name
  pattern from analysis alone) and update it there.

## 3. MusicGen palette — CUT (r1 finding: the premise was stale)

Original assumption: `_otr_style_palette.py` `STYLE_PALETTE` (10 slugs) feeds
MusicGen cues by style slug, so covering all 100 catalog slugs with real
opening/closing/interstitial cue triples would be required content work.

**r1 kibitz grounding disproved this.** `compose_music_prompt()`
(`nodes/_otr_music_prompt.py:76-99`) builds every cue prompt from `meta`
brief fields (`story_brief_terms`, `music_mood_terms`, keyword-mined
`script_brief`) — it does not read `STYLE_PALETTE` at all. A grep of
`nodes/*.py` for `STYLE_PALETTE`/`_otr_style_palette` returns only the
palette's own file. The BUG-LOCAL-216-era architecture (style slug -> cue
lookup) was superseded by this brief-driven composer at some later point,
and `_otr_style_palette.py` is now dead relative to runtime, kept alive
only by its own `tests/test_style_palette_drift.py`.

**DECIDED (operator, 2026-07-05, blanket dead-code directive): DELETE
`_otr_style_palette.py` and `tests/test_style_palette_drift.py` outright.**
No cue-authoring work, no "keep it just in case" — confirmed dead relative
to runtime (section 3 grep above), so it is removed 100%, not deprecated.
Safe-removal step (still required, not optional): grep for any dynamic/
string-keyed access to `STYLE_PALETTE`/`KNOWN_STYLE_SLUGS` (e.g.
`getattr`, `importlib`, a string built from parts) before deleting, so a
reflective caller isn't silently orphaned. If that grep is clean — delete
both files outright in the same commit as the rest of section 2's sweep.

## 4. Workflow JSON (positional widgets — TWO adjacent slots removed)

- `workflows/otr_scifi_16gb_full.json`, `OTR_LedgerScriptWriter` node:
  DELETE `widgets_values[8]` (`style`) and `[9]` (`style_custom`) together
  in one edit. Everything from old index 10 (`creativity`) onward shifts
  down by TWO. `story_scaffold` (old index 24, ships `"auto"`, KEPT per
  section 1a) lands at new index 22; `source_bank` (old 25) at new 23;
  `visual_style` (old 26) at new 24.
- Update both hardcoded test assertions in
  `tests/test_workflow_json_guardrails.py`: the `wv[8] == "let the story
  decide"` assertion is DELETED (no more sentinel string to check — there
  is no combo left), and `wv[24] == "auto"` becomes `wv[22] == "auto"`.
  The `expected 27` widgets_values-length assertion becomes `expected 25`.
  Also `_WRITER_STYLE_SLOT = 8` (line 358) needs removing/updating.
- **r3 finding: the index-pinned test scope is wider than just
  `test_workflow_json_guardrails.py`** (confirmed live full widget order,
  `OTR_LedgerScriptWriter.py:1919-2297`, matching the actual
  `workflows/otr_scifi_16gb_full.json` node-1 array exactly): `[8] style,
  [9] style_custom, [10] creativity, [11] perfect_run_spacesaver, [12]
  min_p, [13] repetition_penalty, [14] max_new_tokens_cap, [15]
  lemmy_cameo, [16] use_exchange, [17]
  enable_production_stage3_validators, [18] news_briefs_required, [19]
  openrouter_slot_a_model, [20] openrouter_slot_b_model, [21]
  comfy_slot_a_model, [22] comfy_slot_b_model, [23] refine_target_grade,
  [24] story_scaffold, [25] source_bank, [26] visual_style`. Post-deletion:
  length 25; `story_scaffold` -> [22], `source_bank` -> [23], `visual_
  style` -> [24]. ALSO update: `tests/test_otr_api_companions.py:34-214,
  466`, `tests/test_source_bank_widget_2c.py:322-323`, `tests/
  test_visual_style_widget_3c.py:172-174`, `tests/
  test_openrouter_slot_widgets_s2.py:62`, and `tests/
  test_writer_input_resolve.py` (AST-asserts `_fetch_rss_seed_or_die`'s
  2nd positional-arg contract — must be re-pinned once `style` is
  stripped from that function's signature, section 2).
- Re-validate after edit: `OTR_WorkflowValidator` + JSON round-trip +
  `TestWidgetOrderVsInputTypes` (the general BUG-LOCAL-097 guard that
  derives widget order from `INPUT_TYPES()` and checks the saved JSON is
  an in-order subsequence of it) + link referential integrity.

## 5. Ledger / meta schema

- `meta.story_contract` already exists and is the correct single record —
  keep as-is.
- `meta.style_pick` (old picker stamp) is deleted along with the picker.
  Audit `_otr_ledger_consistency.py` and any other reader of
  `meta.style_pick` before deleting.
- Already-rendered episodes on disk keep their historical
  `meta.style_pick` / old-slug `meta.style` values untouched — this rip
  targets runtime code, not archived output. BUG_LOG.md's historical
  entries (BUG-LOCAL-216, -240, -270, etc.) stay as archival record; they
  describe past incidents, not live code paths.

## 6. Config lever — SUPERSEDED by section 1a

Originally framed as an abstract env-var design fork. r1 kibitz grounding
found the real, concrete decision point: the already-shipped
`story_scaffold` widget (section 1a). Resolve there; this section is kept
only as a pointer so the history of the question is not lost.

## 7. Sequencing — one atomic cleanbreak sprint

Per this repo's standing cleanbreak rule ("no runtime gates inside
cleanbreak sprints; each cleanbreak sprint is the LAST one"): no staged
dual-system interim state. Order within the single sprint:

All open decisions are LOCKED (operator, 2026-07-05): `style` + `style_
custom` = DELETE BOTH (section 1); `story_scaffold` = KEPT (section 1a);
`_otr_style_palette.py` = DELETE (section 3); bank/pipeline gating = OUT
OF SCOPE this sprint, doc-note only (section 1b). r3 kibitz found two
build-breaking sequencing bugs — steps 1-2 below fix those FIRST, before
any deletion work, per r3's judgment.

1. Move `build_story_contract()`'s call site EARLIER (after `script_brief`
   + `cast_seed` exist, before `lock_cast` — r3 critical finding), and
   strip `style` from `news_interpreter.build_news_briefs()` (r3 critical
   finding, the circular-dependency fix). Everything below depends on the
   contract existing at the right point in the sequence.
2. Delete `style` + `style_custom` widgets/inputs and the entire
   `_resolve_inputs` style resolver branch. Thread `contract.label`/
   `.slug` into `lock_cast` and every other former reader of
   `resolved["style"]`. Add a canonical `meta.style` field derived from
   `meta.story_contract.slug`/`.label` (r3 finding) instead of deleting
   the stamp outright.
3. Strip `style` from the RSS fetch/rerank chain end-to-end:
   `_fetch_rss_seed_or_die`, `_otr_source_payload.py`'s
   `_fetch_science_rss`/`_interpret_news`, and `story_orchestrator.py`'s
   ranking/history call sites (r2+r3 finding — one connected change).
4. Delete the dead modules/constants/JSON seams (section 2) — BOTH
   `_otr_style_picker` import sites (line ~2797 AND the smoke-test import
   ~6103), the call site, and the telemetry stamp, all in the same edit as
   the file deletion. Confirm the grep sweep is clean across `nodes/`,
   `tests/`, `nodes/story_packs/`.
5. Delete `_otr_style_palette.py` + its drift test outright (section 3).
6. Rewrite tests — positive pins only, across the FULL widget-index test
   list from section 4 (not just `test_workflow_json_guardrails.py`) and
   `tests/test_writer_input_resolve.py`'s AST pin on the fetcher's
   signature.
7. Re-validate + re-freeze the workflow JSON per section 4 (two adjacent
   slot deletions, full downstream reindex, both test assertions updated).
8. Add the doc-only bank/pipeline scope note at `build_story_contract()`'s
   NEW call site (section 1b) — no gating code, added AFTER step 1's move.
9. Full regression suite + Bug Bible, per CLAUDE.md, after the whole
   chunk — not after each sub-step.
10. Commit AND push to `v2.0-alpha` in the same session (CLAUDE.md section 7).

## 8. Risk / blast radius (carried from ANALYSIS.md + new)

- Positional-widget REMOVAL risk on the workflow JSON: TWO adjacent slots
  (`style`, `style_custom`) deleted in one pass, full downstream reindex
  of 17 widgets (`creativity` through `visual_style`) plus two hardcoded
  test-index rewrites (section 4) — the biggest mechanical risk in this
  sprint, do it once, carefully, not as two separate edits.
- C7 determinism: confirm the single-draw contract stays
  cast_seed-keyed/reproducible exactly like today's `select_style`.
- Sweep `docs/`, `kibitz-runs/`, dashboards for mentions of the deleted
  symbols — archival text only, not a blocker.
- Bank/pipeline scope (section 1b) is explicitly deferred, not solved —
  make sure the doc-note actually lands at the call site so a future
  bank-enablement effort doesn't inherit this engine silently.

## 9. Status: CODE-READY (r4 convergence complete, 2026-07-05)

All four kibitz rounds are complete: r1 (arc-level grounding — killed the
stale MusicGen cost estimate, found `story_scaffold`), r2 (coding plan —
found the RSS/rerank ripple, the existing fallback-violating exception
handler, the story-pack JSON seams), r3 (wiring — found two build-breaking
sequencing bugs: `lock_cast` reading a deleted field before the contract
existed, and the `news_interpreter` circular dependency), and r4
(convergence — found two missing test files, one dead freeze-validator
block, one label-vs-slug threading rule, and correctly rejected one
scope-creep suggestion to touch the catalog module's own unrelated
defensive helpers). See `kibitz-runs/2026-07-05-style-engine-riprout/r4/
final.md` for the full r4 judgment.

Every finding across all four rounds has been folded into this document.
No open forks remain. The plan is code-ready — implementation can proceed
per section 7's sequencing once the operator confirms the other coder
window is clear to edit `nodes/OTR_LedgerScriptWriter.py`,
`workflows/otr_scifi_16gb_full.json`, and the other touched files.

Original r1/r2 exploratory questions below, kept for history:

1. Is collapsing to ONE `build_story_contract()` call (tone + climax +
   sound-world from the same draw) actually sound, or does tying the
   user's explicit hard-pick (a specific catalog slug) to the SAME
   grammar-injection path risk a regression the current split design was
   accidentally protecting against?
2. Section 1a: does `story_scaffold` survive as a legitimate "scaffold off"
   creative mode, or does "no fallback" require deleting it outright?
3. Section 1: does `style_custom` get folded into the single contract, or
   retired?
4. Anything else that reads `meta.style`, `meta.style_pick`, or the old
   10-slug shape that this plan hasn't found yet.
