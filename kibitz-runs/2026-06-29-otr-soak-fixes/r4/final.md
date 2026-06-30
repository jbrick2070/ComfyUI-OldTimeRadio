# kibitz FINAL judgment -- otr-soak-fixes (r1-r4, CONVERGED 2026-06-29)

Panel: Codex/gpt-5.5 (4 rounds) + Claude anchor + code-grounded judging each round. Antigravity OUT OF
CREDITS after r1 (0-byte output, hung on `agy models`) -> Codex-only panel r2-r4. Agent calls: 5
(r1 codex OK + agy fail; r2/r3/r4 codex OK).

CONVERGED wire-ready plan: `docs/2026-06-29-coverage-soak/SPRINT_PLAN.md`.

## Net deltas the arc produced (all grounded vs the real code)
- BUG-411 is ALREADY implemented (FluxGuidance 3.5 `flux_gen1.py`; cinematic + radio-distress tails +
  STYLE_ANCHOR `otr_meta_brief_image_prompt.py`) -> demoted to a look-QA seed-4242 check.
- NO-FALLBACK is ALREADY enforced at render time (`render_shot` raises loud) -> S-E = scaffolding cleanup
  + DEPRECATE-IN-PLACE of `allow_auto_fallback` (deleting it shifts node-87 wv14..18); NOT a behavior
  change.
- recipe-stamp -> extend the durable `meta.render_engines` payload (NOT `TOP_PRESERVE`; image side already
  durable in `ledger['images']`).
- S-F -> the named `_otr_soak_phase0/1.py` DO NOT EXIST; submit a PRUNED API prompt (render-tail node 92
  only, `/history` == `{92}`/`{63,92}`) with a baked asset BUNDLE (ledger + audio + all referenced stills,
  paths rewritten + preflighted). ComfyUI MCP = the execution tool.
- S-A -> composite-side loop-fill, SIZE-AGNOSTIC (49-frame cap is only `humo_14B_169`; base caps 177);
  assert DELIVERED segment frames == target, keep raw `frame_count` raw, `should_loop` BEFORE the warning.
- labels MUST be `engine_id (...)` (`_engine_id_from_pick`); S-B must pin `OTR_LTX_AV_RENDER_CANVAS`;
  HuMo cfg is NOT a regression; engine-retirement is separable/deferrable (floor constants + soak fixtures
  + tests, bigger than dropdowns).

## Per-round artifacts
r1..r4/{input, anchor, codex, r#_plan}.md (+ r4/final.md = this). Judgment per round at the top of each
r#_plan.md.
