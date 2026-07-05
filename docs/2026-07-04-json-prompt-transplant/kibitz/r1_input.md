# KIBITZ R1 INPUT -- PHASE A JSON PROMPT EXTRACTION (real anchors)

## Operator scope

Docs-only hardening pass. Anchor the review on the REAL upstream plan
in the sibling repo (paths below); carve out the PHASE A subset the operator
wants shipped first; leave PHASE B for a later sprint.

- **Phase A** (this arc): py-to-JSON extraction of the 12 named seams into
  JSON profile files. Sci-fi lifts as a first-class JSON profile alongside
  news / cinematic / radio. **Byte-identical audio.** No production code
  touched. HEAD `a7bdc42d` on `v2.0-alpha` in the OTR production repo.
- **Phase B** (future sprint): the full architectural transplant described
  in the real R1 doc (4 orthogonal axes source_bank / story_model /
  story_pipeline / visual_style, bank + pipeline + style registries, bridge
  artifact, C1-C5 chunks + 5 Fable structural upgrades). Gated on Phase A
  shipping green.

## Repos to grep

Two repos live side-by-side under `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\`:

1. `ComfyUI-OldTimeRadio` -- **production** OTR node pack. Grep here for
   the actual site file:line targets of the 12 seams (e.g.
   `nodes/_otr_outline.py:532` `_SYSTEM_PROMPT`,
   `nodes/_otr_line_composer.py:3275` `_NEWS_CODA_SYSTEM`,
   `nodes/_otr_style_picker.py:296` `_INVENTOR_SYSTEM`, etc.).
   Branch `v2.0-alpha` @ `a7bdc42d`.
2. `ComfyUI-OTR-UpstreamStoryLab` -- **the lab / transplant workspace**
   containing the R1 plan, the pack fixtures, the compat mirrors, and
   `production_mirror/` (a mirror of the OTR node pack for AST drift
   tests). Branch `main` @ `7df7c80`.

Both anchor docs live in the SIBLING repo at:
- `ComfyUI-OTR-UpstreamStoryLab\docs\R1_ARCHITECTURE_AND_CODING_PLAN_V2.md`
- `ComfyUI-OTR-UpstreamStoryLab\docs\JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md`

Additional supporting docs in the SIBLING repo you should read if useful:
- `docs\FABLE_FINAL_REVIEW_2026-07-02.md` -- Fable already ran once on the
  pre-transplant state; contains 4 MUST-FIX items grounded on real file:line
- `docs\FABLE_FINAL_REVIEW_PROMPT_BEFORE_TRANSPLANT.md`
- `docs\GO_FORWARD_PLAN.md` (in the sibling repo)
- `docs\PACK_AUTHOR_CHECKLIST.md`
- `docs\todays-plan-handoff.md`
- `docs\story-engine-map-brief.md`

## What r1 (arc) is looking for

Focus your review on:

(a) The four-axis architecture (source_bank / story_model / story_pipeline /
    visual_style) -- is it the right decomposition when carved down to
    PHASE A only (just the 12 prompt seams, sci-fi as a profile, no bank
    or pipeline machinery)?
(b) Is the 12-seam list complete and correctly sized for a byte-identical
    Phase A extraction? Any seam that resists mechanical extraction (e.g.
    inline f-string templates) should be flagged.
(c) The compat-mirror pinning (`NEWS_BRIEFS_FIELDS`, `NEWS_SEED_KEYS`,
    `MOTION_ROLE_KEYS`, `PRODUCTION_VISUAL_TAILS`) -- does Phase A
    need any of these mirrors, or are they Phase B?
(d) Sci-fi treatment -- is treating sci-fi as a JSON profile (not ripped)
    the byte-identical-safe path? Any risk?
(e) Any Phase A invariant at risk (byte-identical audio; ROW_KEYED merge;
    `test_period_prompts.py` asserts; critic/reroll seam; env-flag gating)?
(f) Fable's 4 MUST-FIX items from the 2026-07-02 review -- are they
    resolved at `7df7c80` or still open? Do any block Phase A specifically?
(g) The loader/validator/router seam in Python -- is the smallest viable
    Phase A shape a `get_prompt(profile_id, site_key)` function
    returning byte-identical strings, or does the real architecture demand
    more?

Return your review in VERDICT / MUST-FIX / SHOULD-FIX format. Label every
claim CONFIRMED / MISREAD / UNVERIFIABLE against the files you can grep.

---

# ANCHOR 1 -- R1 Architecture + Coding Plan v2

Source: `ComfyUI-OTR-UpstreamStoryLab\docs\R1_ARCHITECTURE_AND_CODING_PLAN_V2.md`
Sibling repo HEAD: `main` @ `7df7c80`

# R1 Architecture + Coding Plan v2 - Upstream Multi-Source Story Engine

Date: 2026-07-02 (overnight run). Author: Claude Fable R1 pass.
Workspace: ComfyUI-OTR-UpstreamStoryLab (transplant workspace).
Baseline: production_mirror @ ComfyUI-OldTimeRadio d48a9d76 (SFX-free).
Prior art: v1 lab (git 41c6512), FABLE_FINAL_REVIEW_2026-07-02.md.
Write scope: THIS FOLDER ONLY. Production is not edited until the explicit
transplant chunk.

Core law (unchanged, now enforced structurally):

```text
JSON owns content and configuration.
Python owns validation, routing, execution, and fail-loud errors.
No fallbacks. No hidden models. No hidden engines.
```

## 1. The architecture in one picture

```text
banks.json ---------+
story packs (JSON) -+-> REGISTRY (loaded, validated, fail-loud)
visual styles (JSON)+        |
pipelines (JSON) ---+        v
                     resolve(source_bank, story_model, story_pipeline, visual_style)
                             |            [every id explicit or declared default;
                             |             unknown/ambiguous = hard error]
source packet (JSON/fetch) --+
                             v
                     SourceInterpreter[bank]   <- declared binding, allowlisted
                             |                    (science = news_interpreter at
                             v                     transplant; fixtures in lab)
                     StoryInputPacket
                             v
                     LedgerWritingSpec  (spec = ids + material + story input
                             |           + prompt profile + visual policy +
                             |           model plan; cross-id validated)
                             v
                     BRIDGE ARTIFACT (one frozen JSON file)
                       - ledger_writing_spec
                       - meta_mirrors: news (NewsBriefs shape), news_seed
                       - lab_state_digest + schema_version + baseline hash
                             v
                     production adapter (tomorrow's transplant chunk)
```

## 2. Axes, precisely separated

Four orthogonal axes; each is data, none may imply another silently:

- `source_bank` - where material comes from and how it is interpreted.
  Declared in `fixtures/banks.json`. Adding a bank touches zero routing code.
- `story_model` - dramatic/tonal writing lane, source-scoped. One JSON pack
  per (bank, model, pipeline). Adding a model = dropping a pack file.
- `story_pipeline` - the LLM pass structure. Declared in
  `fixtures/pipelines.json` as a named sequence of passes with per-pass slot
  roles (creative/technical), seam references, and hard budgets. Python owns
  sequencing, stop conditions, and failure reporting; JSON owns the sequence
  and prompts.
- `visual_style` - render language. One JSON policy per style. Role-keyed
  motion prompts validated against the production role vocabulary.

Bank/model/style defaults are configuration -> they live in `banks.json`
(`default_story_model`, `default_visual_style`), not in Python dicts.

## 3. No hidden models, no hidden engines - made structural

- Every LLM-touching pass in a pipeline declares its slot role
  (`creative` | `technical`) in `pipelines.json`. The spec carries the
  resolved `model_plan` so the ledger can stamp what ran. No pass may call a
  model without a declaration.
- Every Python behavior a bank needs (fetcher, interpreter) is a NAMED
  binding string in `banks.json` (e.g. `"interpreter": "fixture_media_archive"`,
  at transplant `"science_rss_news_interpreter"`). Python resolves bindings
  through one explicit allowlist registry; an undeclared or unknown binding
  is a hard error with the bank id in the message.
- The adaptive-cleanup experiment's approved technical models are a JSON
  allowlist (`pipelines.json`), enforced by Python; the cap
  (`max_cleanup_passes`) is JSON config, the stop condition is Python.
- Ledger stamps at transplant: `meta.source_bank`, `meta.story_model`,
  `meta.story_pipeline`, `meta.visual_style` (+ mirrors). Nothing runs
  unlabeled.

## 4. One prompt vocabulary: seams

v1 had two overlapping content vocabularies (pack `prompt_stages` dict AND
`StoryPromptProfile` fields). v2 defines ONE canonical seam list, matching
the production prompt sites the transplant will parameterize:

```text
interpret            (source brain -> briefs; science keeps news_interpreter)
outline_system
pitch_room_system
story_select_system
dramatic_state_system
line_grounding       (per-line instruction)
casting_brief_seam   (source/casting brief text path; craft stays shared)
coda                 (coda_mode + coda_system + examples)
title_system
style_pick_inventor
style_pick_chooser   (+ chooser_user_template)
labels               (story_form_label, source_material_label,
                      source_develop_verb, source_grounding_label,
                      key_terms_label, close_brief_label, title_form_label)
```

A story pack supplies seam content + guardrails + forbidden lists.
`StoryPromptProfile` is now a RESOLVED VIEW built by validating and merging
(pack + bank defaults). It contains no Python-authored prose. If a seam is
missing and the bank declares it required, loading fails loudly - no default
prose is invented.

Forbidden-term handling stays metadata (leakage tests scan rendered
previews; forbidden phrases are never rendered into live prompts - models
copy negated terms).

## 5. Compatibility mirrors as pinned, drift-proof contracts

`src/upstream_story_lab/compat.py` pins the production shapes:

- `NEWS_BRIEFS_FIELDS` = exact NewsBriefs field list (casting_brief,
  script_brief, news_close_brief, key_terms, source_hash, source_chars,
  prompt_version, schema_version, model_id, decoder_profile, seed, attempts,
  attempt_failures) - cited to production_mirror/nodes/news_interpreter.py.
- `NEWS_SEED_KEYS` = {headline, source, url, date, body_chars, style,
  selected_at} - cited to production_mirror/nodes/_otr_legacy_to_stage1_adapter.py.
- `MOTION_ROLE_KEYS` = {announcer, music_open, music_close, music_inter} -
  cited to production_mirror/nodes/_otr_video_engines/render_driver.py.
- `PRODUCTION_VISUAL_TAILS` = STYLE_TAIL_DEFAULT / IMAGE_GRADE_TAIL /
  RADIO_BROADCAST_TAIL / ERA_TAIL_DEFAULT strings - cited to
  production_mirror/nodes/_otr_story_brief_helpers.py. (Renamed from
  SCI_FI_TAILS in kibitz r2: the constants are genre-neutral; sci_fi_radio
  is the policy that reproduces them. Where this doc and the round finals
  disagree, the latest kibitz final supersedes.)

Drift-proofing (the advanced part): tests AST-parse the mirrored production
files and EXTRACT these shapes (NewsBriefs class fields, the news_seed dict
literal keys, `_LTX_MOTION_PROMPT_BY_ROLE` keys, the tail constants), then
assert the pinned copies match. Re-mirroring after production moves makes
any shape drift a test failure, not a silent bug. `key_terms` is always a
list (freeze invariant, _otr_ledger_freeze.py).

The bridge emits `meta_mirrors = {news: <NewsBriefs shape>, news_seed:
<seed shape>}`. `meta.news = None` degrade semantics stay a production
decision; the bridge always emits a complete mirror.

## 6. Visual policy, production-shaped

`VisualStylePolicy` v2:

- `positive_tail`, `image_grade_tail`, `broadcast_tail`, `era_tail` -
  policy-owned replacements for the four production constants.
- `allow_radio_tails`, `forbidden_terms` (leakage-test fodder, never
  rendered).
- subjects: announcer / music / scene_open / character_portrait /
  character_scene.
- `motion_prompts` keyed ONLY by `MOTION_ROLE_KEYS` (validator rejects
  unknown keys - scene_broll/background_abstract/sfx are dead roles and must
  stay dead).
- `sci_fi_radio` policy must reproduce the production tails byte-identically
  (test compares against the AST-extracted constants) - this IS the science
  visual baseline pin.

## 7. What gets CODED tonight (lab-only)

C1 `src/upstream_story_lab/` v2 package:
   - `contracts.py` - pydantic v2 models, extra=forbid: SourceMaterialPacket,
     PublicDomainSourceManifest, StoryPack (seam-complete), SourceBankSpec,
     PipelineSpec, StoryPromptProfile (resolved view), StoryInputPacket,
     VisualStylePolicy, LedgerWritingSpec (cross-id validators), BridgeArtifact.
   - `compat.py` - pinned production shapes + citations.
   - `registry.py` - loads banks/packs/styles/pipelines from fixtures/,
     duplicate/unknown/missing = hard error, zero content literals,
     binding allowlist for interpreters.
   - `interpreters.py` - fixture interpreters per bank (deterministic,
     network-free), same protocol production brains will implement.
   - `profiles.py` - pack -> profile resolution (validation only).
   - `bridge.py` - spec assembly + bridge artifact emit/refuse + mirrors.
   - `preview.py` - prompt preview rendering + pack-driven leakage scan
     (every non-science bank, not just media archive).
C2 `fixtures/` - banks.json, pipelines.json, 12 story packs (recovered from
   git v1 and extended to seam-complete; diverged v1 Python/JSON lists
   reconciled by union), 5 visual styles (motion keys re-keyed), source
   packets (fixture briefs moved INTO packets), PD source folders,
   custom-bank schema template.
C3 `tests/` - contracts, registry fail-loud/no-fallback, leakage (story +
   visual), mirror drift-proof (AST vs pinned), sci-fi tail byte-pin,
   motion-key validation, bridge emit/refuse, PD manifest safety
   (absolute/.. paths), pipeline sequencing + loud pass failure.
C4 `nodes.py` + `scripts/validate_lab.py` - ComfyUI validator/preview nodes
   and CLI runner rebuilt on v2 (choices discovered from JSON, never
   hardcoded lists).
C5 `transplant_work/` - staged production edits, lab-only:
   - NEW modules as real files (production-ready, no imports from lab):
     `_otr_source_interpreter.py` (facade; science delegates to
     news_interpreter unchanged), `_otr_story_prompt_profile.py`,
     `_otr_visual_style_policy.py`, `_otr_ledger_input_adapter.py`
     (bridge-artifact validator).
   - PATCH SPECS (exact before/after hunks, file+line cited against
     production_mirror) for the big files: OTR_LedgerScriptWriter (routing,
     title, coda call sites, RSS gate to science_news, meta stamps,
     append-only widgets), _otr_line_composer (compose_source_coda facade,
     line grounding), _otr_style_picker (override kwargs, loud non-science
     failure), _otr_story_brief_helpers + otr_meta_brief_image_prompt +
     otr_shot_lock (policy seam reads), news_interpreter (science-only
     confinement), whitelists (scripts/otr_api.py, _otr_workflow_apply.py).
   - `workflows/otr_scifi_16gb_full.json` working copy: widget-append plan
     documented, NOT applied until production tests exist (tomorrow).

Priority order tonight: C1 -> C2 -> C3 green -> C4 -> C5 as far as context
allows. Every chunk committed; single push at end.

## 7b. Fable R1 delta - upgrades the prior rounds did not have

These five are structural, cheap, and compose with the registry design.
They are what makes this architecture verifiable rather than merely tidy.

1. Template-variable validation at load time. Every seam prompt is a
   template with a DECLARED variable set per seam (e.g. line_grounding may
   reference {source_grounding_label}, {scene_premise}). The registry
   string.Formatter-parses every template in every pack and fails loudly on
   an undeclared or misspelled variable at LOAD, not mid-episode. A JSON
   content system without this ships prompt typos silently.
2. Auditable resolution. `resolve()` returns a Resolution record: requested
   ids, resolved ids, which defaults applied, and the source file of every
   decision. "auto" stops being invisible behavior and becomes data the
   bridge artifact carries. Kills the hidden-default class structurally.
3. Provenance stamping with content hashes. The spec/bridge carries
   `provenance = {bank, model, pipeline, visual_style, pack_sha256,
   style_sha256, banks_sha256, pipelines_sha256, lab_state_digest,
   production_baseline}`. Any episode is reproducible and auditable back to
   the exact JSON bytes that shaped it. At transplant this lands in ledger
   meta beside the four id stamps.
4. Cross-product invariant tests. The registry makes (bank x model x
   pipeline x style) enumerable, so tests assert over ALL combos: resolution
   succeeds or raises a typed error (never substitutes); non-science
   previews never contain science/news terms; every template formats
   cleanly against a fixture context; every declared binding exists in the
   allowlist. This is the architectural proof of "no hidden fallback" -
   not example-based, exhaustive.
5. Pipeline simulation with failure injection. A FakeLLM runner executes
   any declared pipeline end-to-end (network-free) and tests inject a
   failure at each pass to assert the exact failing pass is reported and
   nothing falls back. Plus `schema_version` on every fixture; the registry
   refuses versions it does not know. No silent migration.

Considered and rejected (deliberately, for this system's scale): agent-graph
DSLs, content-addressed prompt stores, effect-system-style capability
tokens, pack inheritance/merge trees. Each adds machinery a one-operator
local pipeline does not need; rejection keeps "clean" honest. Single-level
`extends` for packs is noted as a future option if pack count grows past ~30.

## 8. Gates (unchanged from the final review, plus two)

All gates in FABLE_FINAL_REVIEW_2026-07-02.md TEST/VALIDATION GATES, plus:

- AST drift-proof tests pass against production_mirror (new).
- sci_fi_radio visual policy byte-identical to production tails (new).

## 9. Non-goals tonight

- No production writes. No workflow JSON widget edits (plan only).
- No LLM calls in lab tests (fixture interpreters are deterministic).
- No custom_source_bank implementation beyond schema + fail-loud stub.
- No adaptive-cleanup implementation (declared in pipelines.json as
  documented-but-disabled experiment; loading it runs nothing).


---

# ANCHOR 2 -- JSON Content / Python Behavior R1..R4 Rewrite

Source: `ComfyUI-OTR-UpstreamStoryLab\docs\JSON_CONTENT_PYTHON_BEHAVIOR_R1_R4_REWRITE.md`

# R1-R4 Rewrite - JSON Content, Python Behavior

Core law:

```text
JSON owns content and configuration.
Python owns validation, routing, execution, and fail-loud errors.
```

This replaces the fuzzier architecture discussion. The upstream rewrite should
be judged by whether it keeps prompt/story/visual content in editable JSON packs
while Python stays a strict loader, validator, router, and executor.

## R1 - Architecture Rule

The system should treat source/story/visual material as data, not hardcoded
behavior.

JSON owns:

- story prompt stage text
- source-bank examples and manifests
- story model tone rules
- forbidden leakage terms
- visual style prompt tails
- visual forbidden terms
- visual motion prompt language
- default source/story/style configuration when it is pure configuration

Python owns:

- schema contracts
- JSON loading
- source-bank routing
- story-model routing
- visual-style routing
- default resolution
- ledger-writing spec construction
- validation and clear errors
- execution of story/prompt passes

Rule of thumb:

If changing a tone, prompt, style, source pack, forbidden term, or example needs
a Python edit, the design is drifting wrong. If changing the schema or execution
flow needs only JSON edits, the design is also drifting wrong.

## R2 - Coding Plan

Keep the current standalone lab shape and make it stricter.

Content/config files:

- `fixtures/story_packs/**/*.json`
- `fixtures/visual_styles/*.json`
- `fixtures/source_packets/*.json`
- `fixtures/public_domain_sources/**/manifest.json`
- future custom source-bank schema JSON files

Python behavior files:

- `contracts.py`: Pydantic models and consistency validators
- `catalogs.py`: lookup, default resolution, source/style registration
- `preview.py`: source packet -> interpreted story input -> ledger-writing spec
- `nodes.py`: ComfyUI preview/validation nodes and fail-loud UI surface
- `scripts/validate_lab.py`: command-line validation
- `tests/*.py`: automated proof that JSON content routes correctly

Concrete coding rules:

- No prompt meat should be buried inside Python unless it is genuinely generated
  by algorithm.
- No hidden fallback from media archive or public domain back to sci-fi.
- Unknown source bank, story model, story pipeline, or visual style must error.
- New JSON style packs should appear through the loader without editing node UI
  code.
- New JSON story packs should validate through the same `StoryPack` contract.
- The experimental 4-pass pipeline may be JSON-described, but Python must own
  pass sequencing, pass status, and pass failure reporting.

## R3 - Wiring And Transplant Plan

Before touching production OTR, the standalone lab should emit a bridge artifact
that production can consume.

Bridge artifact should include:

- `source_bank_id`
- `story_model_id`
- `story_pipeline_id`
- `visual_style_id`
- validated `source_material`
- validated `story_input`
- validated `prompt_profile`
- validated `visual_policy`
- compatibility mirror for current `meta.news` consumers
- explicit error if any required content/config JSON is missing or invalid

Transplant rule:

Production code should not ask, “Is this sci-fi, media archive, or public
domain?” in scattered conditionals. It should consume the validated spec and
route by declared ids and contracts.

Production edits should focus on:

- replacing hardcoded sci-fi/news prompt strings with profile fields
- replacing hardcoded cinematic/radio visual tails with visual policy fields
- proving `meta.news` compatibility against the real downstream ledger consumer
- adding any new widgets only at the end of production node widget lists
- updating the canonical workflow JSON only after code and validation are green

## R4 - Convergence Gates

The rewrite is ready to transplant only when these are true:

- JSON story packs contain the actual media archive and public-domain prompt
  content.
- JSON visual styles contain the actual archive/anime/cartoon/origami visual
  content.
- Python validates every JSON pack through strict contracts.
- Python can build a ledger-writing spec for science news, media archive, and
  public domain.
- Media archive and public domain default to non-sci-fi visual policy unless
  explicitly overridden.
- The experimental 4-pass path reports the exact failing pass.
- Tests prove media archive/public-domain do not silently fall back to sci-fi.
- Tests prove forbidden sci-fi/news terms do not leak into non-sci-fi prompt
  previews.
- Production `meta.news` compatibility is verified against the actual consumer
  code, not guessed.
- `otr_scifi_16gb_full.json` remains untouched until the transplant chunk.

## One-Sentence Version

Put the creative material in JSON, make Python enforce the contract, and let
production consume one validated ledger-writing spec instead of inheriting more
hidden sci-fi conditionals.
