# Multi-Modal Story Schema -- STAGE 4 SUB-PLAN (v1 DRAFT, pre-kibitz)

Date: 2026-07-06 (overnight). Branch: `v2.0-alpha`. Parent: BUILD_PLAN Stage 4.
Predecessors: Stage 1-3 COMPLETE (3C @c24dc0fa -- all 5 visual styles live).
Status: DRAFT -- kibitz arc pending (codex; Sonnet/Fable per operator as needed).

## 0. What Stage 4 actually is (BUILD_PLAN, verbatim intent)

"Moving story-CONTENT validation asserts into JSON needs a NEW declarative-rule
ENFORCER first (today's `_otr_workflow_validator.py` only audits litegraph
structure, not story-content rules). Build the enforcer, THEN move the rules."

Law refinement for rules (mirrors the visual-style law): **JSON owns the rule
VOCABULARIES (phrase lists, banned terms, replacement pairs, thresholds that
are content-tuning); Python owns the rule ENGINES (regex compilation, matching
mechanics, severity/flow control, fail-loud errors).** A regex PATTERN embedding
creative vocabulary is content; the matcher is behavior.

## 1. Grounded inventory -- story-content rule vocabularies living in Python today

- `nodes/_otr_line_hygiene.py`: `_CLICHE_RES` (:634, ~20 purple-prose patterns),
  `_STAGE_BUSINESS_RES` (:657), `_ON_THE_NOSE_RES` (:751),
  `_CLICHE_REPLACEMENTS` (:700, curated swap pairs). These ARE the
  story-quality v2 vocabulary -- pure content-tuning, baked in Python.
- `nodes/_otr_stage3_validators.py`: `validate_banned_phrases` (:404) takes a
  caller-supplied list; the SEED list lives at the writer call site.
- `nodes/_otr_compose_exchange.py`: `FORBIDDEN_GENERIC_WORDS` (grounded via
  its :380 join) -- the exchange prompt's soft-nudge vocabulary.
- Announcer/self-vocative + stage-direction scrub patterns
  (`_otr_line_hygiene` / composer strips) -- SPLIT decision per pattern:
  vocabulary-bearing = candidates; pure-structure regexes stay Python.
- NOT in scope: numeric budget maths, schema contracts, litegraph validation,
  the cast contract (structural), news_interpreter retry mechanics.

## 2. Design (v1 slice)

- **Rules pack:** per-BANK rules JSON: `nodes/story_packs/<bank>/rules.json`
  (the vocabularies are STORY-PATH content -- a public-domain adaptation lane
  should not inherit sci-fi cliche tuning wholesale). Exact v1 schema:
  {rules_id (== "<bank>", path coordinate), schema_version "v1",
  cliche_patterns (list[str], regex source strings), cliche_replacements
  (list[[pattern, replacement]]), stage_business_patterns (list[str]),
  on_the_nose_patterns (list[str]), banned_phrases (list[str]),
  forbidden_generic_words (list[str])}. Unknown key = hard error.
  science_news/rules.json is the BYTE-IDENTICAL extraction of today's Python
  vocabularies (the 3A fixture pattern: constants stay as the extraction
  fixture + AST production-read guard).
- **Enforcer = a MODULE, not a graph node** (proposed; kibitz decides):
  `nodes/_otr_story_rules.py` -- lazy, fail-loud loader (the
  _otr_visual_styles.py conventions verbatim: typed errors, sweep, exact
  schema, _clear_caches, regex-compile-at-load with fail-loud on a bad
  pattern). Consumers (`_otr_line_hygiene`, the stage3 call site,
  compose_exchange) fetch compiled rule sets via
  `get_story_rules(meta)` (meta["source_bank"] -> rules pack; the 2C stamp is
  already there). Rationale for module-over-node: the rules fire INSIDE
  compose-time gates (per-line, mid-writer), not at a graph seam; a graph
  node could only lint post-hoc. The BUILD_PLAN's word "node" is honored by
  the EXISTING OTR_WorkflowValidator remaining the structural node; flag to
  kibitz as the main design fork.
- **No behavior change at default:** compiled rule sets from
  science_news/rules.json are byte/semantics-identical to today's constants;
  every gate keeps its current severity + flow (warn/reroll/raise exactly as
  now). Stage 4 moves VOCABULARY, not policy.
- Banks without a rules.json: HARD ERROR at lane runtime (no fallback), but
  ONLY when a rule consumer actually fires for that bank -- science always
  has one; the 3 dormant banks get skeleton rules.json (copy of science or
  curated-lite) authored addressable like 2B.

## 3. Chunks

- **4A**: `_otr_story_rules.py` + `science_news/rules.json` byte-identical
  extraction + consumers routed (hygiene/stage3-seed/exchange) + AST guards
  (no production reads of the old constants; no bare-except around rule
  loading) + byte-identity tests (same flags on a fixture corpus).
- **4B**: skeleton rules.json for the 3 dormant banks + per-pack exact-key
  tests (2B pattern).
- **4C (only if kibitz confirms need):** an OTR_StoryRuleReport node surface
  (post-writer ledger lint, report-only) -- NOT a gate; deferred unless the
  panel finds a consumer.

## 4. Invariants
Audio spine FROZEN; science lane behavior byte-identical (same flags, same
rerolls, same raises on the same inputs); suite + Bug Bible + B7 green per
chunk; UTF-8 no BOM; commit AND push per green chunk; prod/main gated;
loop vars never named `alias` (B7).

## 5. Acceptance
- 4A: default-lane flag parity on a pinned corpus (every current hygiene test
  passes unchanged); rules load fail-loud; lazy; AST guards green.
- 4B: 4 rules packs addressable; sweep covers them.
- Stage 4 DONE = zero story-content vocabulary edits require a Python change
  (the R1 rule-of-thumb), asserts move complete for the inventoried set.

## 6. Open questions (for the kibitz arc)
- Q1: module vs graph-node enforcer (section 2 rationale -- confirm).
- Q2: per-bank rules vs one global rules.json + per-bank overrides (v1
  proposes per-bank full sets; simpler contract, some duplication).
- Q3: are the scrub/strip patterns (stage-direction, self-vocative) content
  or structure? v1 says: structure (they encode FORMAT, not taste) -- stay
  Python. Confirm.
- Q4: regex-source strings in JSON -- lint at load (re.compile fail-loud) is
  planned; is a complexity cap (ReDoS guard) warranted for pack-authored
  patterns?
