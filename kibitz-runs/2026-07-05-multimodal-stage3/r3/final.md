# r3 judgment -- Stage 3 wiring round (codex + Sonnet 3-lens fan-out)

Panel this round: codex (gpt-5.5) + THREE Sonnet grounded fan-out lenses
(swallow-hunt / test-blast-radius / byte-identity+caching), per operator
directive. Claude anchor+judge.

CONVERGENCE: all four reviewers independently found the SAME missed seam --
`_compose_mesh_fodder_prompt` (otr_meta_brief_image_prompt.py :1223-1243,
get_era_tail inside a bare except) -- strong evidence the sweep is now complete.

Accepted into v4 (folded into STAGE3_SUBPLAN.md):
- MESH-FODDER seam added to 3A routing + the de-swallow list + a forced-meta
  mesh test (codex M1 = sonnet-1 M2 = sonnet-3 M1; CONFIRMED :1234-1243).
- CONSTANT-OWNERSHIP DECISION (codex M2 + sonnet-2 findings): the 4 tail
  constants REMAIN as literal definitions in _otr_story_brief_helpers.py as
  the EXTRACTION FIXTURE (lazy-safe -- no import-time pack I/O); the
  extraction test pins sci_fi_radio.json == constants byte-for-byte;
  PRODUCTION reads route through the pack; the AST guard bans production
  reads of the 4 constants outside the style module + the helpers'
  definitions themselves + tests/. Consequence: the 5 tail-pin test files
  (test_still_spine_helpers, test_brief_prompt_finishing,
  test_talking_portrait_s4b, test_video_platform_aseam,
  test_era_literals_c2a) keep importing/asserting the constants UNCHANGED --
  they become the byte-identity matrix for free; zero 3A test rewrites for
  tail pins as long as byte-identity holds.
- HELPER SIGNATURES pinned (codex S1): finish_visual_prompt /
  compose_still_prompt / get_era_tail gain `style: VisualStyle | None = None`
  -- None => resolve get_visual_style(meta) internally (fail-loud);
  multi-helper composers resolve ONCE and pass down; render_driver
  style_tail=False path pinned by test.
- AST-pin shape (sonnet-1 S4): the guard must distinguish the INNER
  ImportError-only shim (benign, stays) from the OUTER except-Exception
  (banned around style calls), and also match bare `get_era_tail(` -- not
  just finish_visual_prompt (sonnet-3 M1 scope note). Confirmed-compliant
  call sites (compose_still_word_prompt caller :1716-1718, render_driver
  :1731) enter the guard's positive list so future edits can't wrap them.
- 3C pin list completed (codex M3 + sonnet-2): test_source_bank_widget_2c
  has TWO breaking pins (order[-1]=="source_bank" :61-64 -- rename the test
  -- and the slot-25 patch test len==26 :312-313); test_story_scaffold_toggle
  -3/-2/-1 shift; openrouter-s2 order[26]/len 27; api_companions fixture +
  its MOCK INPUT_TYPES schema must gain visual_style; guardrails :634-745
  len 27 + slot-26 block; whitelist parity tripwire test_workflow_apply
  :258-261 (no edit -- enforcement).
- Gate-order test for get_style(visual_style) beside require_runnable_bank
  (codex S2), same sentinel pattern as 2C.
- Forced-meta mesh_fodder test (codex OPT).
CONFIRMED-SAFE (sonnet-3): prompt hashes computed AFTER finishing from the
composed text (_content_hash :46-49, stamp :1558) -- style change busts
naturally, constant-routing (identical bytes) perturbs nothing; empty-tail
joins filter falsy pieces (no dangling commas) at :518/:555; audio spine has
ZERO imports of the visual constants (frozen invariant safe by construction);
_clear_caches/lazy precedent matches _otr_story_routing exactly.
Verify-at-build kept: otr_image_gen_dispatcher cache-key spot-check
(post-finish prompt text, not a pre-tail draft).
Rejected: none. CUT confirmed kept: no compose-time forbidden-term state.
