# Fable Final Review - 2026-07-02 (Before Transplant Coding)

Reviewer: Claude (Fable). Method: read every listed lab file, the five audit
docs, and the targeted production sites; ran the lab test suite and validator;
swept ComfyUI-OldTimeRadio for SFX/dead-role remnants; checked git state of the
canonical workflow. Every claim below carries a file/line citation. Items I
could not verify from files are labeled UNVERIFIED and excluded from findings.

Live evidence gathered this review:

```text
pytest (lab)                     5 passed in 0.23s
scripts/validate_lab.py          OK upstream_story_lab / story_packs=12 / pd_manifests=3
workflows/otr_scifi_16gb_full.json  git status clean; last commit touching it:
                                 6bad6e5b 2026-07-01 "rip-sfx-broll: delete the dead
                                 sfx subsystem + scene_broll/background_abstract roles"
grep sfx (production nodes/tests)   all remaining hits are loud-rejection sites,
                                 removal comments, legacy-token scrubbers, and guard
                                 tests; zero sfx strings in the canonical workflow JSON
```

VERDICT:

The lab is real, green, and correctly isolated - but it does not yet obey its
own core law. The ledger-writing spec that production will consume gets its
prompt content from Python literals in `catalogs.py`, not from the JSON story
packs; the packs are loaded, validated, and previewed, then ignored by the
spec. The two copies have already diverged. The legacy `meta.news` mirror does
not match the real production shape. One lab fixture and one plan doc still
reference roles deleted by rip-sfx-broll. Fix the four MUST-FIX items inside
the lab (no production edits required), then the bridge is safe to code. The
production SFX removal itself is verified complete: no fallback, loud
rejection, guard tests, clean workflow.

MUST-FIX BEFORE TRANSPLANT:

- `src/upstream_story_lab/catalogs.py` (get_profile, lines 236-353;
  MEDIA_ARCHIVE_MODELS/PUBLIC_DOMAIN_MODELS, lines 20-172): all
  StoryPromptProfile prompt meat (outline/pitch/select/style-picker system
  prompts, grounding lines, labels) and all story-model tone
  guardrails/forbidden patterns are Python literals. The JSON story packs
  carry the same class of content in `prompt_stages` but nothing routes a pack
  into the profile - `preview.build_spec_from_material` (preview.py:118) calls
  `get_profile()` and never reads the pack. The duplicates have diverged:
  Python forbidden patterns for `media_restoration_adventure` are
  "Star-Trek-style mission plot / Amazing-Stories-style twist anthology /
  spaceship rescue / laboratory containment breach / generic science
  experiment emergency" (catalogs.py:30-36) while the pack JSON says
  "futuristic mission plot / twist anthology ending / generic experiment
  emergency" (fixtures/story_packs/media_archive/media_restoration_adventure.json:23-27);
  the outline system prompts also differ (catalogs.py:284-287 vs pack
  `prompt_stages.outline_system`). Why it matters: this is the exact R1
  failure mode ("if changing a tone, prompt ... needs a Python edit, the
  design is drifting wrong"), and whichever copy the transplant consumes, the
  other silently rots. Exact fix: make the StoryPack JSON the single source -
  extend the pack schema with the profile fields (labels, develop verb,
  coda_mode, grounding instruction, per-stage system prompts already exist as
  `prompt_stages`), have `get_profile()` load + validate the pack and build
  the profile from it, delete the duplicated literals from `catalogs.py`, and
  keep only routing/default-resolution/validation in Python.

- `src/upstream_story_lab/catalogs.py` (_BASE_VISUAL_STYLES lines 356-404,
  _load_visual_style_fixtures lines 410-421, get_visual_styles lines 440-450):
  all five visual styles are fully duplicated as Python literals, and the JSON
  fixtures only "override" them. If `fixtures/visual_styles/` is missing the
  loader returns `{}` and the Python copies silently serve - a hidden
  content fallback in the module whose README forbids hidden fallbacks
  (README.md:24-27). Why it matters: content ownership flips back to Python
  the moment a fixture goes missing, with zero error. Exact fix: delete
  `_BASE_VISUAL_STYLES`; `get_visual_styles()` raises if the fixture dir is
  missing/empty or any required style id (per
  `DEFAULT_VISUAL_STYLE_BY_SOURCE`, catalogs.py:180-184) is absent.

- `src/upstream_story_lab/preview.py` (build_legacy_news_mirror, lines
  136-150): the mirror emits `title`, `headline`, `link` inside a
  `meta.news`-shaped dict, but the real production `meta.news` is
  `NewsBriefs.model_dump()` (stamped at OTR_LedgerScriptWriter.py:2887) whose
  fields are casting_brief, script_brief, news_close_brief, key_terms,
  source_hash, source_chars, prompt_version, schema_version, model_id,
  decoder_profile, seed, attempts, attempt_failures
  (news_interpreter.py:151-186) plus a stamped cache_key
  (news_interpreter.py:588-590). There is no title/headline/link in that
  shape. Headline-and-URL data lives in a different meta key,
  `meta.news_seed = {headline, source, url, date, body_chars, style,
  selected_at}` (_otr_legacy_to_stage1_adapter.py:70-77), which live code
  reads (video_engine.py:2095, adapter above). Note it is `url` there, not
  `link`. Why it matters: the r4 gate "confirm mirror keys against the real
  consumers" fails as-written; a bridge stamping this mirror would feed keys
  nothing reads and starve the news_seed consumers. Exact fix: the bridge
  emits two mirrors - `meta.news` exactly NewsBriefs-shaped (key_terms always
  a list; `_otr_ledger_freeze.py:232-248` hard-errors on null/non-list) and
  `meta.news_seed` in the seed shape - and the lab adds a test pinning mirror
  keys to the NewsBriefs field list so drift fails loudly.

- `fixtures/visual_styles/archival_documentary.json` (motion_prompts, lines
  20-24): keys are `announcer`, `music`, `scene_broll`. `scene_broll` is a
  role deleted by rip-sfx-broll (commit 6bad6e5b; nodes/_otr_shared/role_slots.py
  raises on it), and production motion-prompt vocabulary is `announcer`,
  `music_open`, `music_close`, `music_inter`
  (_otr_video_engines/render_driver.py:546-561). Why it matters: this is the
  one place the upstream plan still depends on the deleted path - the exact
  thing this review was asked to catch (question 8). Exact fix: re-key the
  fixture to the four production roles (plus optional lipsync/character keys
  when the visual stage defines them), and add a contracts.py validator that
  rejects unknown motion_prompt keys against an allowed-role tuple - Python
  validating JSON content, which is the correct ownership.

SHOULD-FIX BEFORE TRANSPLANT:

- `src/upstream_story_lab/preview.py:13-23`
  (FORBIDDEN_MEDIA_ARCHIVE_PROMPT_TERMS): forbidden leakage terms are
  JSON-owned content per R1, and the packs already carry
  `forbidden_leakage_terms`. The preview checker should consume the active
  pack's list (the pattern validate_lab.py:108-117 already uses), keeping only
  the checking mechanics in Python. Also: only media_archive has a preview
  leakage check - add the same negative check for public_domain previews.

- `src/upstream_story_lab/preview.py` (interpret_fixture_material, lines
  26-105): fixture briefs are Python content - e.g. the archive close_brief
  "Archive note: preservation work can change what a community remembers..."
  (lines 44-47) and science key_terms `["science", "sensor", "ocean"]`
  (line 68). Move these into the source-packet JSON so fixture content edits
  never require Python edits. Low risk, same law.

- Missing visual-style leakage tests: PHASE2 map requires
  `test_visual_style_leakage.py` (anime/cartoon/origami emit no
  35mm/film-grain/radio-studio tails; sci_fi_radio preserves tails). The lab
  test suite (tests/test_lab_contracts.py) has no such test. Required before
  the visual transplant stage; cheap to add now while policies are fresh.

- `ComfyUI-OldTimeRadio/docs/2026-07-01-source-bank-visual-style-code-ready/PHASE2_PROMPT_PY_UPDATE_MAP.md`
  (render_driver section, "Policy supplies role motion prompts"): the list
  includes `sfx`. Stale post-rip - the sfx role cannot produce a beat, so a
  policy motion prompt for it is dead weight and an invitation to rebuild the
  deleted path. Drop `sfx` from the list and note the three surviving video
  roles.

- `TRANSPLANT_MANIFEST.md:3` still says "Status: placeholder checklist" while
  r4 declares convergence. Stamp per-gate status (done / not-started /
  evidence link) so the manifest is a live gate list, not a wish list.

DEFER UNTIL AFTER TRANSPLANT:

- Adaptive cleanup experiment (README.md:90-95; experimental pack
  `optional_adaptive_cleanup_decision`): spec-only today, no code path
  depends on it; safe to defer.
- `custom_source_bank` runnable lane: current fail-loud stub
  (nodes.py:369-374) is the correct pre-transplant behavior.
- render_driver deep prompt extraction (motion table, lipsync base,
  char-face fallback): correctly staged as the risky V3 chunk
  (TRANSPLANT_MANIFEST.md:32-47); do not pull it forward.
- `scripts/plan_transplant.py` helper: manifest-driven dry-run design is
  sound; build it during the transplant chunk, not before.

DELETE / RIP OUT:

- `ComfyUI-OldTimeRadio/nodes/story_orchestrator.py:219` and `:2886` - two
  definitions of `_inject_scene_transitions`, a dead injector that appends
  "[SFX: Scene transition - low bass sweep or static crossfade]" into script
  text. Evidence it is dead: no callers anywhere in nodes/scripts/tests (repo
  grep), story_orchestrator survives only as a library for
  `_fetch_science_news` (OTR_LedgerScriptWriter.py:1137-1140) and friends, and
  live transcripts assert the [SFX:] token is gone (guard tests from
  rip-sfx-broll). Replacement: none needed. Fits the existing
  `test_no_orchestrator_legacy_symbols.py` pattern - add the symbol there when
  deleting. Not a transplant blocker; it is the last SFX-emitting text in the
  repo.
- `catalogs.py` duplicated content literals (story-model guardrails, visual
  styles, profile prompt strings) - deleted as the consequence of the first
  two MUST-FIX items, not as a separate action.
- No other SFX remnants require action: remaining production references are
  loud-rejection sites, removal comments, legacy-token scrubbers for old text,
  and guard tests; the canonical workflow JSON contains zero sfx strings and
  its last change was the rip commit itself.

KEEP:

- `contracts.py` exactly as shaped: strict pydantic with `extra="forbid"`
  everywhere and the LedgerWritingSpec cross-id validator (contracts.py:175-202).
  Do not loosen it to make the bridge easier.
- Fail-loud routing: UnknownStoryModelError/UnknownVisualStyleError,
  unknown source packet raises (nodes.py:144), custom_source_bank raises with
  a guide pointer (nodes.py:369-374), PD manifest path safety rejecting
  absolute/`..` paths (nodes.py:155-166).
- Non-sci-fi defaults for archive/PD via DEFAULT_VISUAL_STYLE_BY_SOURCE
  (catalogs.py:180-184) with explicit override only - this correctly answers
  the hidden-fallback question for visuals.
- Experimental pipeline treatment: `simple_4_prompt_experimental` visible as a
  pack, excluded from the narrative story-model dropdown (nodes.py:83-93),
  no fallback to legacy_many_pass (pack tone_guardrails). This is the right
  "visible experiment, not hidden fallback" posture.
- Science lane confinement: `_fetch_rss_seed_or_die` and its sci-fi slugs stay
  the science-only source brain (OTR_LedgerScriptWriter.py:1119-1160, 1368);
  the transplant gates it behind `source_bank == "science_news"` per PHASE2.
- The forbidden-pattern-as-metadata decision (preview.py:166-169): forbidden
  terms counted, not rendered into live prompts, because models copy negated
  terms. Keep this in the production parameterization too.
- The staged, risk-checkpointed visual transplant framing
  (TRANSPLANT_MANIFEST.md:32-47) and the append-only widget + forceInput +
  validator rules for the eventual workflow edit (TRANSPLANT_MANIFEST.md:121-131).

REVISED TRANSPLANT PLAN:

1. Lab content ownership pass: route StoryPack JSON into `get_profile()`,
   delete Python content duplicates, fail-loud visual fixture loading, re-key
   `archival_documentary` motion_prompts to production role vocabulary, move
   preview leakage terms + fixture briefs into JSON. Rerun
   `scripts/validate_lab.py` + pytest until green.
2. Mirror correction: `build_legacy_news_mirror` becomes
   `build_legacy_meta_mirrors()` returning `{news: <NewsBriefs shape>,
   news_seed: <seed shape>}`; add a test pinning `news` keys to the NewsBriefs
   field list and `news_seed` keys to the adapter-documented set.
3. Leakage test completion: public_domain preview negative check + visual
   style leakage tests (anime/cartoon/origami vs 35mm/film grain/radio studio;
   sci_fi_radio preserved).
4. Freeze the bridge artifact - the smallest safe artifact production should
   consume (question 9): one JSON file
   `{schema_version, lab_state_digest, ledger_writing_spec:
   LedgerWritingSpec.model_dump(), meta_mirrors: {news, news_seed}}`,
   emitted by a lab node/script that refuses to write if any content JSON is
   missing or invalid. No production import of lab code; production sees only
   the file.
5. Production-side pure module, no wiring: `nodes/_otr_ledger_input_adapter.py`
   validates a bridge file against production reality (NewsBriefs shape,
   freeze key_terms invariant, spine/quality/dramatic-state reader
   expectations) with tests only - no node registration, no workflow change.
6. Parameterize story/ledger prompt sites from StoryPromptProfile:
   coda facade in `_otr_line_composer.py` (compose_news_coda:3386 stays the
   real_news_report implementation behind compose_source_coda), writer coda
   routing (OTR_LedgerScriptWriter.py:4907-4948), title prompt (:937), line
   grounding (:1642 per audit), style picker overrides
   (_otr_style_picker.py:297, :335 - loud failure instead of first-candidate
   fallback for non-science lanes), outline/pitch/select/dramatic sites per
   LEDGER_PROMPT_AUDIT line map, news_interpreter stays science-only behind a
   source-interpreter facade.
7. Science baseline pin: tests prove science_news prompt previews are
   byte-identical (or intentionally equivalent, diff reviewed) before/after
   parameterization.
8. Visual stage (separate, staged): visual policy catalog module,
   `meta.visual_style` stamped at ShotLock before M4 derivation,
   `finish_visual_prompt`/`compose_still_prompt`
   (_otr_story_brief_helpers.py:455-569) read policy instead of
   STYLE_TAIL_DEFAULT/IMAGE_GRADE_TAIL/RADIO_BROADCAST_TAIL constants
   (:228-251), `sci_fi_radio` reproduces current tails exactly; only then
   render_driver deep prompts one at a time with leakage tests.
9. Runtime routing + widgets: writer resolves
   source_bank/story_model/story_pipeline/visual_style, RSS fetch gated to
   science_news (the empty-custom-premise -> RSS branch at
   OTR_LedgerScriptWriter.py:1348-1372 becomes science-only), meta stamps
   added, new widgets appended only at the end of existing widget lists,
   whitelist updates in `scripts/otr_api.py` and `nodes/_otr_workflow_apply.py`.
10. Workflow JSON last: edit only `workflows/otr_scifi_16gb_full.json`,
    append-only widgets, forceInput sockets for linked policy JSON, then
    validator + JSON round-trip + link referential integrity + widget/input
    audit, all green before commit.

TEST / VALIDATION GATES (must pass before workflow JSON edits):

- Lab: `scripts/validate_lab.py` OK and lab pytest green (baseline today:
  5 passed, story_packs=12, pd_manifests=3).
- Mirror shape test: `meta_mirrors.news` keys == NewsBriefs.model_fields;
  `meta_mirrors.news_seed` keys == {headline, source, url, date, body_chars,
  style, selected_at}; key_terms is a list (never null).
- No-fallback tests: media_archive and public_domain never reach
  `_fetch_rss_seed_or_die`; unknown bank/model/style/pipeline raise; empty
  archive/PD source raises rather than fetching science news.
- Leakage tests: media_archive + public_domain prompt previews clean of
  science/news terms; anime/cartoon/origami visual outputs clean of
  35mm/film-grain/radio-studio; sci_fi_radio tails preserved.
- Science baseline: science_news preview byte-identical pre/post.
- Motion-prompt key validation: fixture motion_prompts keys subset of the
  production role tuple.
- Production: widget-vector append-only test, workflow validator, JSON
  round-trip, link referential integrity audit, widget/input audit,
  otr_api/workflow_apply whitelist parity.
- Git: `workflows/otr_scifi_16gb_full.json` clean until the explicit
  transplant commit.

OPEN QUESTIONS:

- Does anything downstream read `meta.news.cache_key`, or is it
  interpreter-internal? news_interpreter.py:588-590 implies internal (cache
  lookup before stamping), but only a runtime trace or consumer grep at
  transplant time proves non-science mirrors can omit it. UNVERIFIED.
- What should the FreezeCascade `news_used` output socket (slot 2 of 7,
  OTR_LedgerFreezeCascade.py output contract) carry for non-science banks -
  the source packet mirrored into the article dict shape, or empty? Its
  consumers were not enumerated in this review. UNVERIFIED.
- Is `meta.news = None` acceptable for archive/PD lanes (consumers all guard
  with `.get()`, writer already has a None degrade path at :2939), or must the
  bridge always stamp the mirror? Operator decision - both are implementable.
- Which local models belong on the adaptive-cleanup approved-model allowlist?
  Content/config that should live in JSON; not derivable from the files.

ANNEX - direct answers to the ten prompt questions:

1. Core law obeyed? Partially. Schema/routing/fail-loud: yes. Content
   ownership: no - profile and style content is Python-resident (MUST-FIX 1-2).
2. Prompt meat still in Python? Yes: catalogs.py profiles + models,
   _BASE_VISUAL_STYLES, preview.py briefs/leakage terms (MUST-FIX 1-2,
   SHOULD-FIX 1-2).
3. Scattered conditionals? No - routing is centralized in catalogs/preview
   if-chains, which is legitimate Python routing; the problem is the content
   inside the branches, not the branching.
4. Hidden fallback to sci-fi/news prevented? In the lab: yes for routing
   (unknown ids raise; archive/PD default to archival_documentary), with one
   exception - the visual-fixture silent fallback to Python literals
   (MUST-FIX 2). In production: the RSS branch remains reachable whenever
   custom_premise is empty until plan step 9 gates it.
5. Visual content/execution separated? Policy JSON exists and Python
   validates/routes, but the Python duplicate catalog undermines it
   (MUST-FIX 2), motion keys are wrong (MUST-FIX 4), and production has no
   `meta.visual_style` reader yet (grep: zero hits in nodes/) - the seam is
   still transplant work, correctly staged.
6. Experimental path visible, not fallback? Yes (KEEP).
7. Exact production sites: plan steps 6, 8, 9 with file:line citations.
8. SFX-path dependence? Production removal is complete and loudly guarded;
   the only forward-looking dependents found are the lab's `scene_broll`
   motion key (MUST-FIX 4) and the PHASE2 map's `sfx` motion role
   (SHOULD-FIX 4); one dead injector remains in story_orchestrator.py
   (DELETE).
9. Smallest safe bridge artifact: plan step 4.
10. Before workflow JSON is touched: every gate in TEST / VALIDATION GATES.
