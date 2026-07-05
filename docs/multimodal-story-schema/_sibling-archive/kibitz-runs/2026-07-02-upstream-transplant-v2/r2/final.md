# r2 Final - Coding Plan (synthesized; code landed with this round)

Panel: Codex + Claude Code + driver anchor. Antigravity hit its quota wall
mid-round and was killed after 35 min (skill fallback workflow); its r1
grounding stands. All claims below verified against production_mirror.

The v2 implementation was BUILT during this round and lands with this
synthesis: 41 tests green + validate_lab.py green headless (12 packs, 11
validated specs, drift pins matching the mirror, byte-identical sci_fi_radio
tails). Everything below is folded INTO code, not just prose.

## Folded (verified, now in code or PATCH_PLAN)

1. Per-seam template variables (Codex M1, CONFIRMED at _otr_style_picker.py
   :301/:334): contracts.LABEL_TEMPLATE_VARIABLES + SEAM_RUNTIME_VARIABLES
   (style_pick_inventor: n_required/seed_sample_block/article_excerpt;
   chooser: article_excerpt/candidates_block/story_summary); registry
   validates per seam at load. Answers Claude Code M4.
2. Bridge shapes (Codex M3/M4, CONFIRMED at writer :1500 _build_news_payload,
   :2887, news_interpreter :167): meta_mirrors = {news (all 13 NewsBriefs
   fields, key_terms list-typed), news_seed (all 7 keys)};
   adapter_news_article = the custom-premise article-dict shape (writer
   :1338-1346) - news_used is NOT faked; production derives it from the
   outline. close_brief <-> news_close_brief mapping is explicit in
   bridge.py + facade + tests (Claude Code S1).
3. Fallback inventory completed (Codex M6, all CONFIRMED): six sites in
   transplant_work/PATCH_PLAN.md as a per-lane decision TABLE (Claude Code
   S3) - style padding + chooser first-candidate grandfathered science-only
   (operator directive 2026-06-18), RSS slug substitution science-only,
   title floor + outro floor kept (content-neutral), briefs-degrade n/a via
   bridge. Tests per row at transplant.
4. style/style_custom precedence (Codex M5, CONFIRMED writer :1220/:1871):
   legacy style widgets stay the science tonal-preset lane; non-science
   banks ignore the slug (story_model owns tone); test required proving a
   non-science run never consumes the slug. In PATCH_PLAN item 1.
5. Transplant ADDITIONS clearly labeled (Claude Code M1/M2, CONFIRMED
   missing in mirror): OutlineRequest.outline_system_prompt and pick_style
   kwargs (inventor_system_prompt/chooser_system_prompt/chooser_user_template,
   empty="") are NEW at transplant; exact signatures in PATCH_PLAN.
   compose_source_coda labeled new facade (S2).
6. LlmFns + FetchRequest defined (Claude Code M3): interpreters.LlmFns,
   contracts.FetchRequest. Packs live IN THE REPO as of this landing
   (M5 resolved; combo count now derivable: 11 runnable-bank packs x 5
   styles = 55, asserted in tests, <10s budget).
7. C5 lab-side testing is pure-module only (Codex S1): the four staged
   production modules are dict-in/dict-out with no lab/production imports;
   tested in tests/test_transplant_modules.py.
8. PD path containment via resolve() incl. symlink escape (Codex S4);
   requirements.txt restored with pydantic>=2,<3 (Codex S3);
   PRODUCTION_VISUAL_TAILS name wins everywhere - R1 doc note added
   (Claude Code S4).

## Rejected, with reasons

- Cut per-file provenance hashes (Claude Code C1): kept - computed once at
  bridge emit, and git-hash provenance fails for uncommitted fixture edits
  mid-session; the consumer is the bridge artifact itself.
- Ship resolve() as a flat tuple (Claude Code C3): kept the Resolution
  record - it is implemented, tested, and the bridge artifact carries it;
  downgrading tonight would delete working audit data.
- FakeLLM runner stays simple_4-only (Claude Code C2 confirmed the narrowing;
  the runner REFUSES descriptive pipelines by contract + test).

## Carry to r3 (wiring round, over the landed code)

- Wiring order and socket/widget sequencing for tomorrow's chunks.
- Whether OTR_BridgeArtifactEmit output path convention (bridge_out/) suits
  the production adapter's expected input location.
- Confirm nodes.py registry-discovery keeps ComfyUI IS_CHANGED cache honest.
