# 2C WIRING PLAN (r3-hardened) -- source_bank selector on OTR_LedgerScriptWriter

Supersedes r2/final.md. Panel = codex; Claude anchor+judge.

## Honest threading contract (r3 M1/M3 resolution)
The ONLY pack-routed seam today is `line_composer_system`
(_PHASE_TO_PACK_SEAM in _otr_creative_prompt_router.py). Therefore in 2C the
widget selection governs: (a) the run-intent gate, (b) the line_composer_system
prompt resolution. It does NOT govern: the outline stage prompts (three
hard-wired stage constants at _otr_outline :1870/:1998/:2103; the :1843 resolver
call is only the period-overlay probe -- LEAVE CONSTANT, no outline threading
claim), and the exchange prepass system prompt (hard-coded at
_otr_compose_exchange :384-429; pre-existing, bank-agnostic, science included).
Both are recorded on the LANE-ENABLEMENT CHECKLIST (below) that gates any future
`runnable:true` flip of a non-science bank. Since every non-science bank is
runnable:false in 2C and the gate fires first, no selection can reach an
unthreaded seam -- the contract is honest by construction.

## Changes (one commit: code + JSON + tests together)

1. **Widget**: `source_bank` appended after `story_scaffold` (END of optional);
   choices = registry bank IDS (assert choices == list_bank_ids() order incl.
   non-runnable -- r3 OPT); default "science_news"; registry failure raises out
   of INPUT_TYPES (deliberate convention exception, comment + registration-failure
   test via _clear_caches + monkeypatched registry raising -> INPUT_TYPES raises
   StoryRoutingError).

2. **Workflow JSON**: node 1 wv slot 25 = "science_news" (26 values), NO inputs[]
   entry. Validator + round-trip + link/widget audit, UTF-8 no BOM.

3. **Headless patch surfaces (r3 M4):** add `source_bank` to BOTH
   `CREATIVE_WHITELIST` sets (`nodes/_otr_workflow_apply.py:489` +
   `scripts/otr_api.py:753`); parity test stays green
   (tests/test_workflow_apply.py:258-261). New test: `patch_widget_by_name`
   lands "source_bank" at slot 25 in the updated fixture (r3 S2).

4. **run() + refine**: `source_bank="science_news"` after story_scaffold, before
   `*`. Root-cause the refine `_core` locals() leak (os/_scaffold -> TypeError):
   filter against `inspect.signature(type(self).run).parameters` minus refine
   internals; refine-lane regression test (no TypeError + non-default
   source_bank survives re-entry); BUG_LOG entry.

5. **Gate FIRST:** `require_runnable_bank(source_bank)` is the first statement in
   run() -- before _apply_story_scaffold_env, refine gate, budget resets,
   _resolve_inputs/RSS. Ordering test with sentinels.

6. **Resolved surface (r3 S1):** `_resolve_inputs()` gains/returns
   `source_bank` in the resolved dict (one authoritative value for
   meta/ledger stamping + tests). Stamp `meta.source_bank` on the ledger.

7. **Threading (line_composer seam ONLY):**
   - `resolve_creative_system_prompt(repo_id, phase, source_bank_id="science_news")`
     -> `resolve_story_pack(source_bank_id)`; `_SCIENCE_BANK_ID` survives as
     default only.
   - `compose_line()` AND `compose_line_draft()` gain
     `source_bank_id="science_news"`; draft passes it to the resolver; compose_line
     forwards on the draft call AND on its THREE recursive self-calls
     (:2507, :2664, :2762 -- r3 M2); writer passes it at all three call sites
     (:4581, :4649, :4788).
   - NO generate_outline threading (r3 M1 -- outline stays constant; the r2 claim
     is removed).
   - Threading test: monkeypatch resolve_story_pack; assert the selected id
     reaches the composer resolver call, including through a quality-reroll
     recursion. No AST kwarg pins (r3 CUT).

## Lane-enablement checklist (append to STAGE2_SUBPLAN.md -- gates any future
runnable:true flip of a non-science bank)
1. Outline seams: migrate the three outline stage constants to pack routing
   (outline_macro/phase/beat_system seams exist in the science pack, unconsumed).
2. Exchange seam: make build_exchange_prompt pack-routable or bypass exchange
   for the lane.
3. Source payload: fetcher/interpreter contract (RSS/news_interpreter are
   science-hardwired).
4. Announcer/coda/style seams: audit remaining _PHASE_TO_PACK_SEAM coverage.

## Acceptance
Suite + Bug Bible + B7 + test_audio_byte_identical green; widget in the real JSON
slot 25 + all positional pin tests updated same commit
(test_workflow_json_guardrails 673-733, test_story_scaffold_toggle:50-53,
test_openrouter_slot_widgets_s2:51-63, test_otr_api_companions:148-153/178-204,
new last-optional test); gate-first ordering test; threading test (incl.
recursion); whitelist parity + patcher slot test; refine TypeError regression
test; zero episode change at defaults.

## Judgment log (r3)
Accepted: M1 (outline = constant; drop the outline-threading claim -- resolver
:1843 is overlay-only, stage prompts are 3 constants; pack outline seams
unconsumed CONFIRMED), M2 (recursive compose_line self-calls :2507 CONFIRMED,
:2664/:2762 accepted per pattern), M3 (exchange hard-coded system :384-429
CONFIRMED; resolved as checklist item since non-science banks cannot run in 2C),
M4 (whitelists CONFIRMED :489/:753), S1 (resolved dict + meta stamp), S2
(patcher slot test), OPT (choices == list_bank_ids pin), CUT (no AST kwarg pin).
Rejected: none. The r3 "VERDICT: no" is resolved by scoping the threading
contract honestly rather than widening 2C.
