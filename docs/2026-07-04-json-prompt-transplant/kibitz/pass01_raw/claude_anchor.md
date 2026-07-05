# CLAUDE ANCHOR REVIEW -- R1 (real anchors)

Written before fan-out synthesis. Grounded against:

- `ComfyUI-OTR-UpstreamStoryLab` @ `main` `7df7c80` (lab / plan)
- `ComfyUI-OldTimeRadio` @ `v2.0-alpha` `a7bdc42d` (production)
- Fable Final Review 2026-07-02 (in sibling `docs\FABLE_FINAL_REVIEW_2026-07-02.md`)
- Prior Explorer-grounded 15-site OTR inventory (this session)

## VERDICT

The real R1 v2 plan is architecturally sound and Fable-vetted. The
lab code has already moved to the target state (catalogs.py deleted;
content lives in `fixtures/story_packs/**/*.json` and
`fixtures/visual_styles/*.json`; `profiles.py:resolve_profile()` routes
(bank, model, pipeline) -> pack -> `StoryPromptProfile` at load time).

The operator's Phase A carve corresponds directly to Fable's REVISED
TRANSPLANT PLAN **steps 4-7**:

- Step 4: freeze the bridge artifact (lab-emitted JSON file)
- Step 5: production-side `_otr_ledger_input_adapter.py` (validator only,
  no wiring)
- Step 6: parameterize the 12 seams at their production call sites from
  `StoryPromptProfile`
- Step 7: science baseline byte-identical pin

Fable's steps 8-10 (visual stage, runtime routing + widgets, workflow
JSON edit) belong in Phase B and stay gated on Phase A soak-green.

Four MUST-FIX items block r1 convergence; five SHOULD-FIX items should
be resolved before r2 elaborates chunk-level diffs.

## MUST-FIX

### MF1. `StoryPromptProfile` seam coverage vs the operator's 12 seams

`src/upstream_story_lab/profiles.py:31` `resolve_profile()` returns a
`StoryPromptProfile`. That struct is defined in `contracts.py` (not read
this pass) and the resolver reads `pack.prompt_stages.get(name, ...)`
for each stage. But the operator's 12 seams are:

```
interpret, outline_system, pitch_room_system, story_select_system,
dramatic_state_system, line_grounding, casting_brief_seam, coda,
title_system, style_pick_inventor, style_pick_chooser, labels
```

Kibitz r2 MUST verify:

(a) All 12 seam names are keys in `StoryPack.prompt_stages` (or covered
    by other pack fields for `labels`, which is not a system prompt).
(b) `StoryPromptProfile` exposes all 12 as attributes.
(c) `interpret` is EITHER a pack field OR explicitly documented as
    "science lane stays Python-owned behind a source-interpreter facade"
    (per Fable step 6). Both are acceptable; ambiguity is not.

If any seam is missing from the resolved profile, Phase A cannot
mechanically parameterize that site.

Label: UNVERIFIABLE (need contracts.py + a pack file inspected in r2).

### MF2. Bridge artifact spec is under-specified for Phase A

Fable step 4 calls for a bridge artifact file emitted by a lab
node/script and consumed by production. The R1 v2 doc section 1 sketches
its contents (`ledger_writing_spec`, `meta_mirrors`, `lab_state_digest`,
`schema_version`, `production_baseline`). But Phase A needs a CONCRETE:

- File path in production where the artifact lives at runtime
- Emit trigger (does the lab CLI emit it? does a Comfy node emit it? does
  the writer's `_fetch_science_news` pull it?)
- Handoff cadence (per-episode? per-workflow-load? cached?)

Without these, `_otr_ledger_input_adapter.py` (step 5) has no known
inputs to validate against.

Label: CONFIRMED (spec cited; Phase-A-actionable details missing).

### MF3. Adding `_otr_ledger_input_adapter.py` to production IS a
production edit

Fable step 5 says the module ships "with tests only - no node
registration, no workflow change." That is nearly-touchless, but it is
still a production file addition. The operator's Phase A directive says
"NO production code touched." Kibitz r2 MUST resolve:

- Does Phase A include this preparatory module (fine, since it has no
  runtime effect until the seam parameterizations in step 6 land)?
- Or does Phase A ship ONLY the lab-side bridge emit (step 4) and the
  production-side changes wait for Phase B?

Recommended: include step 5 in Phase A. It's inert until step 6 wires
it. Skipping it would leave Phase A without a concrete production-side
artifact and pushes the "explicit transplant chunk" indefinitely.

Label: CONFIRMED.

### MF4. `interpret` seam at `news_interpreter.py:704` builds the prompt
via f-string interpolation

`nodes/news_interpreter.py:704` (`_build_user_prompt`) does NOT store the
prompt as a single string literal. It f-string-interpolates
`_MAX_CASTING_BRIEF_CHARS`, `_MAX_SCRIPT_BRIEF_CHARS`, and
`_MAX_NEWS_CLOSE_BRIEF_CHARS` into the user prompt (grounded via
`git grep -n` this session and Explorer's excerpt). Phase A has two
options:

(a) Move the f-string SKELETON into the pack as an `interpret.user_template`
    with `{max_casting_brief_chars}`, etc. slots. Python still owns the
    char-limit constants; the profile resolver would need to validate slot
    presence.
(b) Match Fable step 6: news_interpreter stays "science-only behind a
    source-interpreter facade" -- meaning `interpret` is NOT in the pack
    for science and lives on the interpreter binding instead. Then only
    the media_archive / public_domain / custom banks have a JSON
    `interpret` field.

R1 v2 section 4 seams list includes `interpret` as a seam. R1 v2 section
3 says "science keeps news_interpreter". These are consistent if
`interpret` is a per-bank binding string, not a per-pack prompt. Kibitz
r2 MUST pick one and spell out which pack/bank fields hold the
non-science interpret contract.

Label: CONFIRMED.

## SHOULD-FIX

### SF1. Fable 2026-07-02 line refs may have drifted at OTR HEAD `a7bdc42d`

Fable step 6 cites `_otr_line_composer.py:1642` for line grounding.
My grounding this session cites `:1621` for the grounding rider (and
`:3275` for `_NEWS_CODA_SYSTEM`, `:3407` for the V2 concat). Fable step
6 also cites `_otr_line_composer.py:3386` for `compose_news_coda` --
current is `:3407`. Small drift, but every chunk-level diff needs line
numbers verified at HEAD `a7bdc42d`. Kibitz r2 MUST re-ground all
Fable-cited lines against current HEAD.

### SF2. Story-pack selection mechanism absent from HEAD `a7bdc42d`

R1 v2 axes require a `(source_bank, story_model, story_pipeline)` triple
to route into a pack. OTR HEAD `a7bdc42d` has no such field on the
ledger and no such widget on any node. Bridge artifact must carry these
ids, but Phase A does not add widgets (that's Phase B step 9). So Phase
A operates on a HARDCODED-DEFAULT triple (e.g.
`science_news / sci_fi_radio / legacy_many_pass`) that reproduces
current behavior. Kibitz r2 MUST specify the hardcoded default triple +
the exact call site that supplies it.

### SF3. `test_period_prompts.py` currently asserts on
`OTR_PERIOD_SYSTEM_PROMPT` at `_otr_period_prompts.py:37`. Which pack
carries it post-Phase-A?

The five anchor tokens (`"1940s"`, `"Suspense"`, `"NARRATOR"`,
`"CHARACTER:dialogue"`, `"Family-broadcast safe"`) + the `[SFX:`
prohibition + slang blacklist need to resolve to the pack that plays
the "OTR 1940s radio" role. That is presumably the `sci_fi_radio` visual
style + a specific story pack. Kibitz r2 MUST name the pack.

Extra risk: `test_creative_prompt_router_exact_match.py:60` uses
`assert result is OTR_PERIOD_SYSTEM_PROMPT` -- OBJECT IDENTITY, not
equality. Post-Phase-A the resolved string is a NEW `str` object each
call. Test must switch to `==` or the resolver must return-by-reference
from a module-level cache. Load-bearing but easy fix.

### SF4. Sprint-2 `_STILL_WORD_TYPOGRAPHY` / `_STILL_WORD_BACKDROP` maps
are Python constants that just shipped

Sprint 2 (commit `e821d6fd`, this same day 2026-07-04) added
`_STILL_WORD_TYPOGRAPHY` and `_STILL_WORD_BACKDROP` genre maps in
`nodes/otr_meta_brief_image_prompt.py` per the still-word prompt spec.
These are new Python-owned genre-keyed content. R1 v2 core law says
content should be JSON-owned. Kibitz r2 MUST decide whether these move
in Phase A (they're recent Python literals that fit the pattern) or
defer to Phase B (they're image-side, not dialogue-side, and were
scoped code-only just today).

### SF5. `resolve_profile()` load-time behavior needs an
`IS_CHANGED`-safety review

The lab's `registry.py` loads at import (`registry._load_packs` etc.).
Once transplanted, the registry singleton must:

(a) Not re-load per node call (ComfyUI `IS_CHANGED` cache would break).
(b) Fail-fast on schema errors at ComfyUI node-registration time (not
    at first prompt-resolution time).
(c) Not hold locks that block during LLM calls.

Fable's step 5 is silent on this. Kibitz r3 (wiring) should surface it,
but r1 arc should flag it as a load-bearing concern.

## Grounding table (this session)

| Claim | Source | Label |
|---|---|---|
| Sibling repo exists at custom_nodes\ComfyUI-OTR-UpstreamStoryLab | pwsh Test-Path | CONFIRMED |
| Sibling repo @ main 7df7c80 | git rev-parse | CONFIRMED |
| Sibling has 12 story packs | Get-ChildItem count | CONFIRMED |
| Sibling has 5 visual styles (anime, archival_documentary, cartoon, paper_origami, sci_fi_radio) | Get-ChildItem | CONFIRMED |
| profiles.py has `resolve_profile(registry, source_bank_id, story_model_id, story_pipeline_id)` | git grep | CONFIRMED |
| profiles.py reads `pack.prompt_stages.get("line_grounding", ...)` | git grep | CONFIRMED |
| registry.py has `_load_banks`, `_load_pipelines`, `_load_packs` | git grep | CONFIRMED |
| catalogs.py DELETED at 7df7c80 (was cited in Fable 2026-07-02) | git ls src | CONFIRMED |
| OTR HEAD a7bdc42d on v2.0-alpha | git rev-parse | CONFIRMED |
| 15 OTR prompt sites (12 named + 3 outline hierarchy siblings) | git grep this session | CONFIRMED |
| `interpret` at `news_interpreter.py:704` is f-string interpolation | Explorer excerpt + this session | CONFIRMED |
| `OTR_PERIOD_SYSTEM_PROMPT` at `_otr_period_prompts.py:37` | git grep | CONFIRMED |
| `test_creative_prompt_router_exact_match.py:60` uses `is` identity | git grep | CONFIRMED |
| Fable 2026-07-02 MUST-FIX #1 (catalogs.py get_profile) | catalogs.py deleted at 7df7c80 -> resolved | CONFIRMED-RESOLVED |
| Fable 2026-07-02 MUST-FIX #2 (_BASE_VISUAL_STYLES) | catalogs.py deleted at 7df7c80 -> resolved | CONFIRMED-RESOLVED |
| Fable 2026-07-02 MUST-FIX #3 (build_legacy_news_mirror shape) | not verified at 7df7c80 | UNVERIFIABLE |
| Fable 2026-07-02 MUST-FIX #4 (archival_documentary motion_prompts keys) | not verified at 7df7c80 | UNVERIFIABLE |
| `StoryPromptProfile` attributes vs 12 seams | contracts.py not read this pass | UNVERIFIABLE |
| Bridge artifact emit trigger / production file path | R1 v2 doc silent | UNVERIFIABLE |
| Fable step 6 `_otr_line_composer.py:1642` drift vs my `:1621` | git grep | CONFIRMED-DRIFT |

## Phase A invariants -- risk map

- **Byte-identical audio (I1):** at risk if MF1 unresolved (any missing
  seam falls back to Python literal). Fable step 7 pins science lane;
  kibitz r2 must extend the pin to sci_fi_radio profile explicitly.
- **ROW_KEYED merge (I2):** unchanged (production ledger; not story).
- **`test_period_prompts.py` (I3):** at risk from SF3 (object identity
  in `test_creative_prompt_router_exact_match.py`).
- **Critic / reroll seam (I4):** unchanged.
- **`test_audio_byte_identical` (I5):** downstream of I1.
- **`l3-2026-05-14` ledger schema (I6):** unchanged (no widget adds in
  Phase A per the operator carve).
- **Env-flag gating (I7):** unchanged.
- **`IS_CHANGED` caching (I8):** SF5 flags this.

## What the r1 fanout should focus on

Codex + Antigravity should:

1. Grep both repos for the exact contents of each of the 15 OTR seam
   file:line refs (my grounding table) and confirm/refute each.
2. Read `contracts.py` in the sibling repo and enumerate
   `StoryPromptProfile` fields; map each to one of the 12 seams; flag
   any missing (MF1).
3. Read `bridge.py` in the sibling repo and identify the emit trigger
   / consumer contract; flag if either end is under-specified (MF2).
4. Verify Fable 2026-07-02 MUST-FIX items #3 and #4 are resolved at
   `7df7c80` (my table has these as UNVERIFIABLE).
5. Re-ground the specific line refs in Fable step 6 against current OTR
   HEAD `a7bdc42d` (SF1).

Any agent-proposed change to the axis architecture, the seam list, or
Fable's step ordering is OUT OF SCOPE for Phase A and should be logged
for Phase B, not folded into synthesis.
