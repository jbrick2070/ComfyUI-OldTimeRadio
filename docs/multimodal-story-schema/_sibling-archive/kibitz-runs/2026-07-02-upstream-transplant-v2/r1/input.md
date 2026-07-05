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
- `SCI_FI_TAILS` = STYLE_TAIL_DEFAULT / IMAGE_GRADE_TAIL /
  RADIO_BROADCAST_TAIL / ERA_TAIL_DEFAULT strings - cited to
  production_mirror/nodes/_otr_story_brief_helpers.py.

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
