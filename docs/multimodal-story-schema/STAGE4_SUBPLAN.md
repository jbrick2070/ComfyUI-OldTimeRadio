# Multi-Modal Story Schema -- STAGE 4 SUB-PLAN (v4 FINAL -- r3 sequencing fixes folded)

## R3 FIXES (codex, all grounded CONFIRMED -- fold into section 3):
- Writer resolves `story_rules = resolve_story_rules(resolved["source_bank"])`
  ONCE in run() after the gates + _resolve_inputs, BEFORE the beat loop;
  `_w1b_s3_kwargs` (:4689-4703) uses `stage3_banned_phrases=
  list(story_rules.banned_phrases)`; the SAME object passes into compose_line
  as `_story_rules=` (rules exist before compose_line runs -- r3 M1).
- **BUG-LOCAL-417 real fix (r3 M2):** compose_line_draft routes through
  `resolve_creative_system_prompt` when `source_bank_id != "science_news"`
  EVEN IF creative_repo_id is None (today :2061-2068 short-circuits to the
  constant whenever repo is None, so bank threading alone is a NO-OP at the
  reroll/spine sites). Science lane with repo None keeps the constant
  (object identity preserved -- the _otr_outline sentinel + C7 contract).
- story_spine `_recompose` derives `meta = data.get("meta", {}) if
  isinstance(data, dict) else {}` INSIDE the closure (no `meta` in scope
  there -- r3 M3; the broad :177-178 catch would have swallowed a NameError).
- story_spine `_recompose_announcer_tagline` :215 ALSO passes
  `source_bank_id=str(meta.get("source_bank") or "science_news")` (r3 M4).
- Type hints: validate_banned_phrases/validate_line widen to Sequence[str]
  (r3 S1). Scan script: StoryRulesError is FATAL, not a skipped ledger
  (:598-600 catch -- r3 S2). Loader uses object_pairs_hook dup-key rejection
  (the _otr_story_pack :101-108 convention -- r3 S3) + a CONTROL-CHARACTER
  lint on regex sources (kills the JSON `\b`-backspace trap at LOAD time,
  not just at corpus parity -- r3 OPT, accepted).

(v3 text below is the base contract; r2 = codex + 3-lens Sonnet fan-out.)

Date: 2026-07-06 (overnight). Branch: `v2.0-alpha`. Parent: BUILD_PLAN Stage 4 (AMENDED:
module enforcer, not a graph node -- edit lands with 4A). Stage 1-3 COMPLETE.
Status: v3 coding contract (r2 = codex gpt-5.5 + Sonnet lenses threading/blast-radius/
regex-safety -- full agreement on every load-bearing point). Arc:
`kibitz-runs/2026-07-05-multimodal-stage4/` (+ the Sonnet reports in this session's log).

## 0. Law + premise
JSON owns rule VOCABULARIES; Python owns ENGINES + GLOBAL POLICY (SFW profanity
stays Python -- a lane must never weaken SFW). Structure regexes stay Python.
FORBIDDEN_GENERIC_WORDS (compose_exchange :199, "soft hygiene nudge ONLY") is
prompt guidance -- OUT of the enforcer, checklist item for story-pack prompt
config. `OTR_StoryRuleReport` node CUT. ReDoS treatment = pattern length cap
(200) + compile fail-loud + corpus test, FINAL (deferred-analysis language cut).

## 1. Inventory (final)
JSON: _CLICHE_RES :634, _STAGE_BUSINESS_RES :657, _ON_THE_NOSE_RES :751,
_CLICHE_REPLACEMENTS :700 (pairs; REPLACEMENTS carry regex backrefs "\\1" --
schema must preserve backref semantics + the pronoun-preservation engine stays
Python in repair_cliche_span's _do closure :736-740), _BANNED_THESIS_RES :600
(NOTE the ['’] straight+curly apostrophe class -- a JSON-fidelity trap),
_PERSONAL_COST_BOILERPLATE_RES :1135, and stage3 DEFAULT_BANNED_PHRASES
(_otr_stage3_validators :106 -- the REAL seed; there is NO writer call-site
seed: every production call passes stage3_banned_phrases=None and the module
default fires. Extraction alone would be DEAD -- the writer must gain the
producer line, section 3).
Python: DEFAULT_PROFANITY_TERMS :140 (SFW), structure regexes, engines.

## 2. Rules packs + loader (contract)
- `nodes/story_rules/<source_bank_id>.json` -- OWN dir (NEVER story_packs/:
  the Stage-2 sweep loads every bank-dir *.json as a story pack,
  _otr_story_routing :337/:339).
- Schema v1 (exact): rules_id, label, schema_version "v1", cliche_patterns,
  stage_business_patterns, on_the_nose_patterns, banned_thesis_patterns,
  personal_cost_patterns (list[str] regex SOURCES), cliche_replacements
  (list of EXACT two-string [pattern, replacement] pairs; replacements may
  carry "\\1" backrefs), banned_phrases (list[str] plain substrings).
- **JSON ESCAPE RULE (load-bearing, sonnet-3):** JSON files carry the regex
  SOURCE with DOUBLED backslashes (`\\b` in-file -> `\b` to re.compile). A
  single `\b` is VALID JSON that decodes to a BACKSPACE byte -- it compiles
  fine and silently matches nothing (rule goes dead with green tests unless
  the corpus-parity test runs compiled behavior). Therefore the corpus
  flag-parity test (section 5) is LOAD-BEARING, not optional, and parity
  compares REASON STRINGS + matched spans + compose_flags -- not booleans
  (reasons embed matched spans, hygiene :621/:673/:768/:782/:1151).
- **Flags contract:** every pattern set compiles `re.compile(p, re.IGNORECASE)`
  UNIFORMLY -- no per-pattern flag override in v1 (matches all 6 current
  sites; stated as schema law).
- Loader `nodes/_otr_story_rules.py`: stdlib, LAZY, frozen StoryRules
  dataclass (compiled tuples), typed errors (StoryRulesError +
  UnknownStoryRulesError, StoryRulesValidationError), `_clear_caches()`,
  `resolve_story_rules(source_bank_id)` PRIMARY + `get_story_rules(meta)`
  thin wrapper. Sweep (codex r2 M5 exact): reject dirs/non-json under
  story_rules/; every file stem == rules_id == a REGISTERED source_bank_id;
  science_news.json REQUIRED; a missing pack is legal ONLY for
  runnable:false banks; explicit resolve of a missing pack raises
  UnknownStoryRulesError (unreachable in production while the run gate
  holds; test-pinned). Singleton: the Stage-3 lazy pattern inherited
  unchanged (accepted judgment; ComfyUI executor single-threaded in
  practice; disposition noted, no lock added -- verbatim convention).

## 3. Threading (the converged map -- every site named)
- `compose_line`/`compose_line_draft`: resolve
  `rules = resolve_story_rules(source_bank_id)` ONCE at compose_line entry,
  OUTSIDE any try (codex r2 M3: the flag_* wrappers + _quality_flags_for_line
  catch broad Exception and return no-hit -- resolution inside them would
  silently disable fail-loud). Thread as a private `_story_rules=` param
  forwarded on ALL recursive/self calls (:2511-2534 quality, :2669-2690 leak,
  :2768-2790 stage3) AND the re-verify bare calls (:2542 _after_flags,
  :2802-2806) -- a repaired line must be scored by the SAME bank vocabulary.
- `_quality_flags_for_line(cleaned, req, *, rules=None)` +
  `line_quality_defect_score(text, req, *, rules=None)`: keyword-only,
  default resolves the science fixture (sonnet-2: tests call with exactly 2
  positional args -- zero breakage).
- Hygiene wrappers `flag_cliche/flag_stage_business/flag_on_the_nose/
  flag_thesis_close/flag_personal_cost_boilerplate/find_cliche_phrase/
  repair_cliche_span`: gain keyword-only `patterns=`/`rules=` with defaults =
  the module fixture constants (validate_banned_phrases `banned=` :404 is the
  in-repo precedent; sonnet-2 confirms every test uses 1 positional arg).
- `compose_announcer_outro` (:3444): NEW `source_bank_id="science_news"`
  param (it has NO context slot today -- the worst gap); callers pass it:
  OTR_LedgerScriptWriter :5084 (resolved["source_bank"]) + _otr_story_spine
  :215 (meta.get). flag_thesis_close calls :3592/:3616 use the resolved rules.
- **PRE-EXISTING BUG FIXED HERE (BUG-LOCAL-417, sonnet-1):** `_otr_reroll`
  :656 and `_otr_story_spine` :170 call compose_line WITHOUT source_bank_id
  -- a reroll on a non-science bank already routes prompts to science today
  (2C latent). Both gain source_bank_id=meta.get("source_bank",
  "science_news") (meta/led in scope at both sites). BUG_LOG entry.
- **stage3 producer (codex r2 M2):** the writer's `_w1b_s3_kwargs`
  (:4699-4703) gains `stage3_banned_phrases=rules.banned_phrases` -- without
  this the extraction is runtime-dead (the "unwired JSON" failure mode).
- `scripts/story_quality_scan.py`: THIRD resolve site (offline, serialized
  ledgers) -- `rules = get_story_rules(_meta(ledger))` + pass-through to the
  wrappers (it imports the same names; signatures stay compatible).
- `_otr_compose_exchange`: explicitly OUT of scope WITH REASON (dormant,
  use_exchange prepass bypasses ALL hygiene gates by construction today;
  logged on the lane/Build-4 checklist -- a future exchange launch must wire
  the gates or ship ungated KNOWINGLY).

## 4. Guards + invariants
- AST guard A: no production reads of the 7 vocabulary constants outside
  _otr_line_hygiene.py/_otr_stage3_validators.py definitions (offender list
  starts EMPTY -- sonnet-2 verified zero production constant imports; the
  guard needs a SYNTHETIC self-test a la test_b7 catches-reintroduction).
- AST guard B (codex r2 S4): production callers of the wrapper fns must pass
  rules=/patterns= (or be the fixture-default test lane) -- pins the
  threading so constants can't drift back.
- Loop vars: `imp` NOT `alias` -- precedent test_visual_styles_3a.py:340-341
  (CW-6; B7 marker \balias\b is code-context enforced). B7 sweeps *.py ONLY
  (JSON invisible) -- verified against docs/_s28_forbidden_sweep.py; the six
  vocabularies collide with ZERO markers (checked word-for-word).
- Science-lane byte-parity: the 7 pinned test files (sonnet-2 list) pass
  UNCHANGED; corpus parity on reasons/spans/flags; audio FROZEN; suite +
  Bug Bible + B7 green per chunk; UTF-8 no BOM; commit AND push per chunk.

## 5. Chunks + acceptance
- 4A: loader + science_news.json + threading per section 3 + BUG-LOCAL-417
  fix + stage3 producer + guards A/B + corpus parity tests. Accept: all
  existing hygiene/stage3/composer tests pass unchanged; new fail-loud
  matrix; parity corpus green; guards green (with synthetic self-tests).
- 4B: BUILD_PLAN amendment + lane-enablement checklist update (rules pack
  authoring per bank; exchange-gate disposition). Accept: docs land with 4A
  or immediately after, same session.
Stage 4 DONE = vocabulary edits need zero Python; SFW + structure + engines
documentedly Python; no dead extraction (every JSON field has a live
producer AND consumer).
