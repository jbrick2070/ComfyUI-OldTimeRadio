# 2C WIRING PLAN (r1-hardened, 2C-ONLY DELTA) -- source_bank selector on OTR_LedgerScriptWriter

Date: 2026-07-05. Branch v2.0-alpha. Parent: STAGE2_SUBPLAN.md section 4 (v3).
2A+2B are SHIPPED @1d06f5c3 -- this plan is the 2C delta ONLY; 2A/2B = verify unchanged.
Panel note: antigravity hung twice (credit bug, near-zero CPU both attempts) -> DROPPED;
panel = codex; Cowork Claude anchor + judge.

## Scope statement (codex r1 M4, accepted)
2C routes PROMPT RESOLUTION ONLY plus the run-intent gate. The science RSS fetch
(`_fetch_science_news` via `_resolve_inputs`) and `news_interpreter.build_news_briefs`
remain HARDWIRED science machinery in 2C; bank `fetcher`/`interpreter` fields stay
metadata. Source-payload ingestion for the non-science lanes is a LATER stage
(their banks are runnable:false, so the gate makes this honest). State this in the
commit message.

## Changes (one commit, code + JSON + tests together)

1. **Widget** -- `INPUT_TYPES` optional gains `source_bank` APPENDED AFTER
   `story_scaffold` (END; BUG-LOCAL-097). Choices = bank ids from the lazy routing
   registry, called INSIDE INPUT_TYPES; default `"science_news"`. Non-runnable banks
   ARE listed (honest StoryBankNotRunnableError on use -- prior r1 converged decision
   KEPT; codex's runnable-only cut REJECTED, see judgment). A broken registry raises
   out of INPUT_TYPES = node registration fails LOUD. Add a code comment marking this
   as a DELIBERATE exception to the ":1720 INPUT_TYPES must never raise" convention
   (no-fallback law) + a registration-failure test (monkeypatch a broken registry ->
   INPUT_TYPES raises typed StoryRoutingError).

2. **Workflow JSON** -- node 1 `widgets_values` gains slot 25 = `"science_news"`
   (wv 25 -> 26). Widget-only: NO `inputs[]` entry (matches the story_scaffold
   precedent verified in the real JSON). Re-validate: OTR_WorkflowValidator + JSON
   round-trip + link/widget audit, UTF-8 no BOM.

3. **Guardrail test** `tests/test_workflow_json_guardrails.py:673-733` SAME COMMIT:
   len 25 -> 26; slot 24 stays "auto"; slot 25 == "science_news" AND is a registered
   bank id (cross-check via the routing registry -- anchor S2). Plus an INPUT_TYPES
   positional test: `source_bank` is the LAST optional entry.

4. **run() signature** -- `source_bank="science_news"` inserted AFTER
   `story_scaffold="auto"`, BEFORE the `*` keyword-only refine block (verified insert
   point :2498-2502). The refine `_core` locals()-filter (:2538) auto-carries it into
   refine passes; add a refine-lane test that a non-default source_bank survives
   re-entry (anchor S3).

5. **Run-intent gate placement (codex r1 M3 + anchor M2, accepted -- PRECISE):**
   `require_runnable_bank(source_bank)` is called in run() BEFORE the refine gate
   (:2532) and BEFORE `_resolve_inputs()` -- i.e. before ANY side effect (RSS fetch,
   model loads, budget resets). A non-runnable pick fails once, loud, cheap. Test
   pins the ordering (non-runnable pick raises StoryBankNotRunnableError and
   `_fetch_science_news` is NOT called -- monkeypatch sentinel).

6. **Threading path (anchor M1 -- the load-bearing fix; codex r1 S2 folds in):**
   `resolve_creative_system_prompt(repo_id, phase)` gains `source_bank_id: str =
   "science_news"` and passes it to `resolve_story_pack(source_bank_id)` (drop the
   `_SCIENCE_BANK_ID` literal at :127; keep the module constant only as the default).
   There are exactly TWO production callers (not "4"): `_otr_outline.py:1843` and
   `_otr_line_composer.py:2064`. Both must RECEIVE the selection or the widget is
   dead for any future runnable lane:
   - Carrier = one new optional field threaded the same way `creative_repo_id`
     already flows: writer run() -> generate_outline(...) kwarg -> the :1843 call;
     writer run() -> composer entry kwarg -> the :2064 call. Default
     `"science_news"` at every hop keeps all existing callers (production + tests)
     byte-identical.
   - Caller-count pin test updated only if it counts kwargs (verify at build).
   - Science byte-identity: with the shipped JSON (slot 25 = science_news) every
     prompt resolves exactly as today -- pinned by the existing equivalence tests.

## Acceptance
- Suite + Bug Bible + B7 green; test_audio_byte_identical green.
- Widget live in the REAL JSON slot 25; validator + round-trip + widget audit green.
- Non-runnable pick raises BEFORE any fetch/model work (ordering test).
- Non-default source_bank reaches BOTH resolver call sites (threading test) and
  survives a refine pass.
- Zero episode change at defaults.

## Judgment log (r1)
Accepted: codex M2 (2C-only delta reframing -- this doc), M3 (gate before
_resolve_inputs; grounded :2592/:1157), M4 (scope statement; grounded), S1
(UnknownStoryModelError exists :51 -- registry error taxonomy already covers it;
doc mention only), S2 (=anchor M1, "4 callers" false -- 2 production callers),
S3 (=anchor S1, INPUT_TYPES raise is a deliberate convention exception + test);
anchor M1/M2/S1/S2/S3.
Rejected: codex M1 (push policy) -- operator kickoff EXPLICITLY set "commit per
chunk, do NOT push unprompted" for this session; operator directive outranks the
checked-in CLAUDE.md default. Codex CUT-1 (runnable-only dropdown) -- contradicts
the prior converged r1 decision (non-runnable listed, honest error on use); hiding
addressable lanes is its own silent behavior; the loud gate is the design.
No-change: codex CUT-2 (pipeline.executable metadata-only) -- already the law.
Verify-at-build: exact caller-count test mechanics; generate_outline/composer
kwarg insert points.
