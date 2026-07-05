VERDICT: GO-WITH-FIXES -- carve is right, sci-fi-as-profile is right, Fable must-fixes are closed; two byte-identity hazards in the outline/line-composer resolver seam must be fixed before r2.

MUST-FIX (blocks convergence):
1. [`nodes/_otr_line_composer.py:1174` + `:2060-2066`] The 15-site cut misses a 16th profile-branched site: the line-composer creative `_SYSTEM_PROMPT`, routed through the SAME resolver (`phase="line_composer_system"`). `_otr_creative_prompt_router.py:11-13,61-64` shows `OTR_PERIOD_SYSTEM_PROMPT` serves BOTH phases, so Chunk D's claim that outline is "the only site with existing profile-branching" is wrong, and `radio.json` carrying only `outline_system` cannot serve the resolver. Fix: add the site (or explicitly scope it out with the router left dual-source), and correct Chunk D.
2. [`nodes/_otr_outline.py:1847` + `_otr_creative_prompt_router.py:43-46,56-64`] The audio-C7 object-identity contract is unaddressed. Outline detects "modern vs period" via `resolved is _SYSTEM_PROMPT`; the router imports that constant at module level. Chunk B (extract outline) ships before Chunk D (router): deleting the constant breaks the router import; rebinding it to a non-identical object makes the identity check fail silently, prepending the modern prompt as a period overlay -- prompt bytes change, audio breaks. Fix: keep module-level names bound to the loader's singleton strings, pin an object-identity test, or merge Chunks B+D.
3. [anchors/R1_PHASE_A_EXTRACTION.md section 4 vs section 3/5] Spec self-inconsistency r2 will inherit: the schema example includes `news_grounding_rider` as a site key absent from the 15-site table while "unknown keys are a load-time error" (the example rejects itself); and section 5 lists site 2 among "sites without existing profile routing," contradicting table row 2. Fix: canonical 16-key list; correct section 5.

SHOULD-FIX (before r2 elaborates chunks):
1. Phase A site vocabulary diverges from the Phase B seam vocabulary (`contracts.py:217,232` TEMPLATE_SEAMS; `StoryPromptProfile` at `contracts.py:246-279`): `news_coda_system` vs `coda`, `line_grounding_rider` vs `line_grounding`, and "labels" means genre_keywords/typography/backdrop in Phase A but story_form_label-family in the lab. Reconcile names or document the mapping now to avoid a Phase B rename migration.
2. Site 1 (`interpret`) is an f-string with runtime interpolation (`news_interpreter.py:704-712`, `{_MAX_CASTING_BRIEF_CHARS}`); the Phase A loader has no template-variable validation (lab upgrade #1 does). Decide per-site declared variables vs leaving caps as Python format args before R2.
3. [r1_input.md:11 vs extraction section 1 step 3] "No production code touched" contradicts "replace the Python literal with a loader call." Pin the wording to "behavior-preserving mechanical edits only."

Fable 2026-07-02 must-fixes: all four resolved at head -- `catalogs.py` gone from the package; registry fail-loud (`registry.py:40,74,130-133`); bridge emits pinned dual mirrors (`bridge.py:120-166`); archival motion keys re-keyed to the four production roles (`archival_documentary.json:20-25`). None block Phase A.

GROUNDING TABLE:
| claim | source file:line I checked | status |
|---|---|---|
| `_SYSTEM_PROMPT` outline site | `nodes/_otr_outline.py:532,1102,1115,1130` | CONFIRMED |
| `_NEWS_CODA_SYSTEM` + V2 examples + concat | `nodes/_otr_line_composer.py:3275,3297,3407` | CONFIRMED |
| inventor/chooser prompts | `nodes/_otr_style_picker.py:296,329` | CONFIRMED |
| `interpret` inline f-string | `nodes/news_interpreter.py:704-712` | CONFIRMED |
| grounding rider + news fallback | `nodes/_otr_line_composer.py:1621,1631` | CONFIRMED |
| identity check hazard | `nodes/_otr_outline.py:1847`; router `:43-46` | CONFIRMED |
| missing 16th site | `nodes/_otr_line_composer.py:1174,2063` | CONFIRMED |
| 12 packs / 5 styles / 4 bank ids | `fixtures/story_packs/**`, `visual_styles/*`, `banks.json:5,36,70,104` | CONFIRMED |
| env-flag invariant real | `nodes/_otr_pitch_room.py:71-76` | CONFIRMED |
| HEADs `a7bdc42d` / `7df7c80` | not checked (no git run; sandbox lags) | UNVERIFIABLE |
