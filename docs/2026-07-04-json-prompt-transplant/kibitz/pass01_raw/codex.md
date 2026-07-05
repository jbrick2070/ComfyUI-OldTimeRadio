VERDICT: no. Phase A is underspecified and internally split between “mechanical JSON extraction only” and the full already-built transplant architecture.

MUST-FIX BEFORE BUILD:
1. [Operator scope / Repos to grep] CONFIRMED: baseline is not pinned. Input says production `v2.0-alpha` @ `a7bdc42d` (`input.md:9-12`, `:23-28`), live production is `6d793d40`, and sibling `PRODUCTION_MIRROR_MANIFEST.md` is pinned to `d48a9d76`. Concrete fix: choose one authoritative production baseline, refresh `ComfyUI-OTR-UpstreamStoryLab/production_mirror/`, then update the Phase A doc and drift tests before any extraction.

2. [Operator scope / §4 seams] CONFIRMED: “sci-fi as a JSON profile” is not byte-identical today. Production `nodes/_otr_outline.py:532-558` has the full schema-bearing `_SYSTEM_PROMPT`; sibling `fixtures/story_packs/science_news/science_news_default.json:8-14` contains short paraphrases. Concrete fix: Phase A must either copy exact production strings into JSON and test byte equality per site, or keep science overrides empty like `tests/test_transplant_modules.py:69-77` so production constants remain authoritative until Phase B.

3. [§4 One prompt vocabulary / r1(b,e)] CONFIRMED: the 12-seam list is not complete for byte-identical prompt extraction. Production has active prompt surfaces absent from `TEMPLATE_SEAMS` in `src/upstream_story_lab/contracts.py:25-42`: outline macro/phase/beat prompts at `nodes/_otr_outline.py:1102`, `:1115`, `:1130`, period overlay at `nodes/_otr_outline.py:1826-1850` / `nodes/_otr_period_prompts.py:37`, line composer system prompt at `nodes/_otr_line_composer.py:1174`, and critic prompt at `nodes/_otr_story_critic.py:266`. Concrete fix: replace “12 seams” with an audited in-scope/out-of-scope table; add split seam keys or explicitly defer each omitted prompt with a byte-identity rationale.

4. [Operator scope vs Anchor 1 §§1-7b] CONFIRMED: Phase A says “no bank or pipeline machinery” and “No production code touched” (`input.md:9-17`, `:51-60`), but the anchor plan centers registries, bridge artifact, mirrors, provenance, visual policy, and pipeline simulation (`input.md:99-126`, `:198-226`, `:300-326`). Concrete fix: write a separate Phase A spec with only: JSON profile file format, extraction source map, validator, byte-comparison tests, and destination paths. Move bridge/registry/visual/pipeline concerns to Phase B.

5. [r1(g)] CONFIRMED: smallest viable Phase A API is not decided. `input.md:67-70` asks whether `get_prompt(profile_id, site_key)` is enough, while the live lab already has full `resolve_profile()` merging bank defaults and packs in `src/upstream_story_lab/profiles.py:31-95`. Concrete fix: for Phase A, use a read-only extractor/validator, not the full resolver, unless Phase A also imports bank defaults and labels as first-class JSON.

SHOULD-FIX:
1. [r1(f)] CONFIRMED: Fable’s old 4 must-fixes appear resolved at sibling HEAD `7df7c80`; `profiles.py:1-5` says no Python prose, `registry.py:109-130` fail-loud loads packs/styles, `bridge.py:120-166` emits dual mirrors, and `fixtures/visual_styles/archival_documentary.json:20-24` uses live motion role keys. Fix: update the Phase A doc to say these are resolved background, not active blockers.

2. [§4 seams] CONFIRMED: `style_pick_chooser_user_template` is treated inconsistently. The plan lists it under `style_pick_chooser` (`input.md:181-182`), `contracts.py:35` allows it, but `fixtures/banks.json:55-65` and `:89-99` do not require it for non-science banks. Fix: make it required wherever chooser override is required, or state empty means current production chooser user template remains.

3. [Operator scope] CONFIRMED: “news / cinematic / radio” is not a clean axis vocabulary. Live bank ids are `science_news`, `media_archive`, `public_domain_story`, `custom_source_bank` in `fixtures/banks.json:5-117`; visual style ids include `sci_fi_radio`, `archival_documentary`, `anime`, `cartoon`, `paper_origami`. Fix: use the real ids; do not introduce informal peer categories.

OPTIONAL / NICE-TO-HAVE:
1. Add a generated extraction manifest with `site_key`, source file, line, hash, JSON destination, and “active in Phase A?”.
2. Add a doc note that Phase A has no user-facing behavior if production remains untouched.

CUT THESE (scope / over-engineering):
1. [§5 compatibility mirrors] Cut from Phase A. `NEWS_BRIEFS_FIELDS`, `NEWS_SEED_KEYS`, and `meta_mirrors` are bridge/transplant concerns (`bridge.py:120-166`), not needed for a no-production-write prompt extraction.
2. [§6 visual policy] Cut from Phase A unless visual prompt tails are part of the 12 seams. `PRODUCTION_VISUAL_TAILS` belongs to visual policy/Phase B.
3. [§7b upgrades 2-5] Cut auditable resolution, provenance stamping, cross-product matrix, and pipeline simulation from Phase A. They serve the full registry/bridge architecture, not mechanical byte extraction.
4. [§9 adaptive cleanup] Cut entirely from Phase A. The live `fixtures/pipelines.json:39-42` already says adaptive cleanup is documentation-only.