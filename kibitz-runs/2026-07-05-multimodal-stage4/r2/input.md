# Multi-Modal Story Schema -- STAGE 4 SUB-PLAN (v2, post-kibitz r1)

Date: 2026-07-06 (overnight). Branch: `v2.0-alpha`. Parent: BUILD_PLAN Stage 4
(AMENDED below). Predecessors: Stage 1-3 COMPLETE (3C @c24dc0fa).
Status: v2 (r1 folded; r2+ pending). Arc: `kibitz-runs/2026-07-05-multimodal-stage4/`.

## 0. Premise (BUILD_PLAN AMENDMENT, r1 M2)

Stage 4 builds a **MODULE enforcer, not a graph node** -- the story-content
rules fire INSIDE per-line compose-time gates (mid-writer), where no graph
seam exists; a node could only lint post-hoc. This explicitly AMENDS the
BUILD_PLAN's "enforcer node" wording (edit lands same commit). The
`OTR_StoryRuleReport` node idea is CUT (r1 CUT-1) -- no named consumer.

Law: **JSON owns rule VOCABULARIES (taste: phrase lists, replacement pairs,
banned terms); Python owns rule ENGINES (compilation, matching, severity,
flow control) AND GLOBAL POLICY (SFW).** Structure-encoding regexes
(stage-direction/self-vocative format strips) are Python. SFW profanity
(DEFAULT_PROFANITY_TERMS) is a HARD GLOBAL INVARIANT -- deliberately NOT
pack-tunable (a lane must never be able to weaken SFW); documented
out-of-scope with this reason (r1 M3 disposition).

## 1. Grounded inventory (r1 M3 complete)

MOVES to JSON (taste vocabularies):
- `_otr_line_hygiene.py`: `_CLICHE_RES` :634, `_STAGE_BUSINESS_RES` :657,
  `_ON_THE_NOSE_RES` :751, `_CLICHE_REPLACEMENTS` :700,
  `_BANNED_THESIS_RES` :600 (announcer-close thesis vocab; consumer
  _otr_line_composer :3592), `_PERSONAL_COST_BOILERPLATE_RES` :1135
  (consumer :2341).
- stage3 `banned_phrases` SEED list (locate the writer call-site seed at
  build; validate_banned_phrases :404 stays the engine).
STAYS Python:
- `DEFAULT_PROFANITY_TERMS` (:140 stage3) -- SFW global policy (above).
- `FORBIDDEN_GENERIC_WORDS` (compose_exchange :197 "soft hygiene nudge ONLY
  -- NOT a gate") -- prompt guidance, not validation; OUT of the enforcer
  slice (r1 S1/CUT-3); flagged to the lane-enablement checklist as future
  story-pack PROMPT config.
- Structure regexes, budget maths, schema contracts, cast contract,
  news_interpreter mechanics.

## 2. Design

- **Rules packs at `nodes/story_rules/<source_bank_id>.json`** -- OWN
  directory, OWN sweep; NEVER inside story_packs/ (r1 M1: the Stage-2 router
  sweeps every *.json in a bank dir as a story pack -- _otr_story_routing
  :337/:339 -- and rules.json would fail registration). Path IS the
  coordinate (rules_id == filename == a registered source_bank_id).
- **Why a separate file from the story pack** (r1 S2): story packs own
  PROMPT content with their own loader/cache/seam contract; rules packs own
  VALIDATION vocabulary with a compile step and different consumers. The
  inert story-pack fields (tone_guardrails / forbidden_plot_patterns /
  forbidden_leakage_terms, _otr_story_pack :81) are a future merge candidate
  -- noted, not v1.
- **Schema v1 (exact, unknown key = hard error):** rules_id, label,
  schema_version "v1", cliche_patterns (list[str] regex sources),
  cliche_replacements (list of [pattern, replacement] pairs),
  stage_business_patterns, on_the_nose_patterns, banned_thesis_patterns,
  personal_cost_patterns (each list[str]), banned_phrases (list[str],
  plain substrings). Regex guard (r1 OPT): compile fail-loud at load +
  max pattern length 200 + a fixture corpus test; full ReDoS analysis
  deferred (packs are repo-authored).
- **Loader `nodes/_otr_story_rules.py`** -- the _otr_visual_styles.py
  conventions verbatim: stdlib, LAZY, frozen dataclass (compiled patterns
  as tuples), typed errors (StoryRulesError base + UnknownStoryRulesError,
  StoryRulesValidationError), sweep (every *.json validates + coordinate +
  is a registered bank id), `_clear_caches()`, `get_story_rules(meta)`
  (meta["source_bank"] -> rules pack, fail-loud).
- **One loading contract (r1 M4):** science_news.json REQUIRED (sweep-time:
  the default bank must have rules). Dormant banks: NO rules file until
  their lane-enablement (r1 CUT-2 -- skeletons cut; fake curation frozen in
  is worse than absence); `get_story_rules` on a bank without a pack =
  UnknownStoryRulesError (unreachable in production while runnable:false
  gates them; pinned by test).
- **Byte-identical default:** science_news.json = exact extraction of
  today's vocabularies; constants stay in Python as the extraction fixture
  (3A pattern) + AST production-read guard; every gate keeps its severity +
  flow.

## 3. Chunks
- **4A**: loader + science_news.json extraction + consumers routed
  (hygiene x5 vocab sets via get_story_rules(meta)... consumers receive meta
  or a resolved rules object -- thread like the visual style: resolve ONCE
  per gate entry; VERIFY at build that meta reaches each consumer -- the
  hygiene fns are called from the composer which has req/meta context) +
  stage3 seed extraction + AST guards + flag-parity corpus tests.
- **4B**: BUILD_PLAN amendment text + lane-enablement checklist update
  (rules-pack authoring is a lane-enablement item per bank).

## 4. Invariants
Science-lane behavior byte-identical (same flags/rerolls/raises on a pinned
corpus); audio FROZEN; suite + Bug Bible + B7 green per chunk; UTF-8 no BOM;
commit AND push per green chunk; no `alias` loop vars; regex sources in JSON
must not trip B7 forbidden-sweep markers (diff-visible -- verify).

## 5. Acceptance
- 4A: flag parity on the corpus (every existing hygiene/stage3 test passes
  unchanged); loader fail-loud + lazy + sweep; AST guards green.
- Stage 4 DONE = the inventoried taste vocabularies are JSON-owned; a
  vocabulary edit needs zero Python changes (R1 rule-of-thumb), with SFW +
  structure explicitly and documentedly Python.

## 6. Open questions (r2)
- Q1: hygiene consumers' access to meta -- some helpers are pure(text) with
  no meta param today; threading shape (param vs module-level default-bank
  accessor) needs the r2 coding pass. NO module-global mutable state in a
  resident server.
- Q2: the stage3 banned_phrases seed location (find at build).
