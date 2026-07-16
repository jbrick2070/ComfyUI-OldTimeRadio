# LLM-First Story Edit Pass -- science_news + media_archive lanes

> **STATUS 2026-07-15 (baseline): STALE -- partially overtaken; Wave-3 excision NOT executed.**
> The pitch room shipped independently (`nodes/_otr_pitch_room.py`) and THE LAW (2026-07-13)
> re-framed the audit surface, but X1 is LIVE WORK: `repair_cliche_span` still rewrites
> spoken lines (`_otr_line_composer.py` ~:2632/:2676) and `cliche_replacements` persists in
> all 8 story_rules JSONs -- a standing violation of the "Python judges, the LLM writes"
> directive. X3 (`OTR_COMPOSER_ACTION_STRIP`) + X4 (`render_few_shot_block`) targets are
> also still live. Re-ground E-items against the 24-lane roster before executing.

- Date: 2026-07-10
- Status: EDIT PLAN (analysis-only; no code changed yet). Supersedes the PLAN section of
  `2026-07-10-scifi-media-archive-story-qa.md` (the findings + codex addendum there remain the
  evidence base; F-numbers below refer to it).

## 0. Operator directive (2026-07-10) -- this reframes everything

The LLM leans in and makes it a good story. Python does not hack the story up after the fact.
Much of the dormant Python guardrail surface should be EXCISED, not wired.

The contract, one line: **Python judges; the LLM writes.**

Python MAY: validate schemas, judge/score drafts, pick the better of two drafts, reroll with a
hint (the LLM does the rewriting), select which prompt/style/seed the LLM gets, and strip
NON-SPOKEN artifacts (stage directions, speaker labels, stop-string tails) so TTS never reads
them. Python may NOT rewrite spoken story text: no regex substitution inside a line, no canned
replacement phrases, no deterministic "repair" of prose.

Consequences:
- Cliche handling becomes detect -> reroll (LLM fixes) -> keep the fewer-defect draft -> if a
  cliche still ships, stamp telemetry and ship it. No substitution.
- The `cliche_replacements` tables and `repair_cliche_span` path are hacksaw -- excise (X1).
- Dormant guardrail surfaces (unconsumed pack fields, dead levers, 0-caller infra) are excised,
  salvaging any authored content INTO prompts first (X2-X4).
- Budget shifts from Python cleanup to LLM passes: the pitch room lights up, the conflict spine
  moves upstream, seams carry the craft.

## 1. Wave 1 -- JSON-only story edits (packs + rules; no writer code, lowest blast radius)

Per codex A6: this whole wave is pack/rules JSON. Gate per codex A1 (JSON parse + registry/
routing tests + prompt-contradiction checks), NOT OTR_WorkflowValidator (no workflow change).

### E1 (F5) media line seam: add the craft block
`nodes/story_packs/media_archive/media_restoration_adventure.json` -> `line_composer_system`,
append:

```
CRAFT:
- Imply more than you state. People rarely say what they mean.
- Match the speaker's stated speech register (the "speaks:" note) exactly --
  a clipped projectionist stays clipped, an ornate curator stays ornate;
  never blur two characters into the same voice.
- Inhabit the mood without naming it. Never announce fear, hope, or wonder.
- Let pressure show through the object at hand: a reel, a splice, a label,
  a catalog card.
- Prefer short and charged over long and explanatory. Stay within plus or
  minus 30% of the requested word count.
```

Why: restores register-matching/subtext/mood-restraint the seam dropped; prevents the
on-the-nose lines the lane's own rules currently catch downstream. Biggest per-line lift, free.

### E2 (F7) media macro seam: priced endings
Same pack -> `outline_macro_system`, add one rule:

```
- The final choice must cost the chooser something concrete they wanted --
  time, credit, a keepsake, the first screening, a tidy catalog -- given up
  willingly. Optimistic endings are paid for, never free.
```

### E3 (F6 + codex A5) seed deck v2: pressure payloads
`nodes/story_packs/media_archive/drama_seeds.json`: schema bump to
`media_archive_drama_seeds_v2`; each seed becomes an object; interpreter renders ALL fields in
the lens block (interpreter change is 1 formatting function -- still Wave 1 adjacent). Example:

```
{ "title": "The China Girl Mystery",
  "want": "put a name to the unnamed woman on a film-lab calibration frame",
  "obstacle": "the lab's records credit machines and studios, never the people",
  "forced_choice": "print an ordinary worker's name on the program, or keep the mystery",
  "paid_cost": "the tidy anniversary program goes to press late",
  "allowed_object_language": "calibration frame, leader, lab report" }
```

Content pass in the same edit: de-duplicate the ~7 "missing media" seeds into genuinely
different want/obstacle shapes; retitle or payload-defuse "The Vanishing Witness" (crime
priming) and keep "China Girl" only WITH the payload above (12B literalization risk).
Deterministic hash pick stays. Tests pin the current 15 titles
(`tests/test_media_archive_interpreter.py:275-278`) -- update deliberately.

### E4 (F8, re-scoped by directive) science rules: detection additions, NO new replacements
`nodes/story_rules/science_news.json` -> `cliche_patterns` (detect -> reroll only), add:

```
\bplaying god\b
\bwhat have (?:we|i) done\b
\bgod help us(?: all)?\b
\bpoint of no return\b
\bon the brink\b
\bfor the (?:greater )?good of (?:humanity|mankind)\b
\b(?:it|we)['’]?s too late\b
\bas you (?:well )?know\b
```

Add `frightened` to the on-the-nose fear alternation. Scope bare `absolutely` (keep
"Absolutely not."). Do NOT add replacement rows -- replacements are being excised (X1).

### E5 (F9) media rules: stop banning the lane's own vocabulary
`nodes/story_rules/media_archive.json`: move `ghost`, `phantom`, `specter`,
`emergency broadcast` out of `banned_phrases` into scoped `cliche_patterns` (reroll class) that
exempt restoration jargon, sketch: `\bghosts?\b(?!\s+image|\s+frame)` (and allow "ghosting").
Exact regexes settled at coding time with codex A4's vocabulary contract test.

### E6 (F14 + F3 seam half) science pack line seam: two one-line fixes
`nodes/story_packs/science_news/science_news_default.json` -> `line_composer_system`:
- "Short and charged beats long and explanatory." -> "Prefer short and charged over long and
  explanatory." (12B misparse).
- 'Generic roles ("the tech", "the lab", "mission control") are fine.' -> "Prefer the proper
  nouns under NAMED ENTITIES; use a generic role only when no named entity fits, and never as
  the scene's focus." (kills the system-vs-user contradiction with the KILL-1 rider; stops
  licensing the exact sameness the lane fights).

### E7 (F13) coda example variety, both packs
Replace the third example in each coda seam with a different syntactic shape so a 12B does not
template-collapse into "Beyond tonight's X:" every episode. Sketch (science): "The laboratory
in our tale was fiction; the one in this report is not:" (media): "Our reel tonight was
fiction; the one in this note still exists:". Keep <=16 words, ends with colon, no stock opener.

### E8 (F16) media phase seam: delete dead tokens
Remove "Favor archive work ... over danger or crime-thriller escalation." from
`outline_phase_system` (its schema is a speaker list; the guidance already lives in the beat
seam). Trim the line seam paragraph's triple "archive object" repetition while there.

### E9 (F11 + codex A3) banks.json: seam coverage
Add `exchange_system` to science_news + media_archive `required_seams` (production runs
use_exchange=True). Codex A3's derived-coverage registry test lands in Wave 3 with the tests.

### E10 (F10 salvage half) inline the authored examples into seams
Move the media pack's priced-dilemma example ("A brittle transcription disc forces ... public
anniversary broadcast or patient restoration") INTO `outline_macro_system` as a worked example
(one "Example premise:" line). Give the science macro seam one worked example of the same
shape. This is the LLM-first use of that content; the dead fields themselves die in X2.

## 2. Wave 2 -- writer/prompt-builder edits (LLM-side architecture)

### E11 (F1 + codex A2) light the pitch room, explicitly
Append a `use_pitch_room` BOOLEAN widget (END of widget surface, positional law), default False
in code, set **True in otr_canonical.json in the same change**. Widen `_GENRES` past 4 while
in there. Full workflow guardrail path: canonical JSON edit + OTR_WorkflowValidator + round-trip
+ link/widget audit. This is the single biggest premise-variety lever, and it is an LLM pass --
exactly where the operator wants the budget.

### E12 (F2 + codex A8) conflict spine upstream, as operational state
Derive dramatic state (opposed wants, dramatic question) BEFORE `generate_outline`; render into
phase + beat user prompts:

```
Central conflict: {A} wants {a_want}; {B} wants {b_want}.
Dramatic question: {dramatic_question}
```

Convert macro `central_tension` into that object (or reject-and-reroll a generic "can they
solve it?" tension) -- the outline consumes the object, not a dangling sentence. Stop asking
the model for fields nobody reads.

### E13 (F3) bank-parameterized riders
Thread `banks.json` defaults into the shared composer tail: `source_grounding_label` ("news
facts" stays the science default -- byte-identical; media renders "archive material") and a
per-lane generic-role example list ("the projectionist", "the front desk", "the stacks" for
media). Fix the two stale "Empty unless L12" comments.

### E14 (F4) arc-phase-conditioned escalation
Resolution beats swap "RAISE THE STAKE ... escalate, never tread water" for "RELEASE the
pressure: show the cost and what changed; do not introduce new conflict." Turn/climax beats
keep escalation. Quiet/unresolved ending_modes can finally land.

### E15 (F15) interpreter briefs ask for drama, both lanes
- science `news_interpreter.py` prompt: script_brief spec gains "WHO wants WHAT, WHAT stands in
  the way"; keep the grounding validators exactly as-is (they validate, they do not write).
- media interpreter: same spine spec + "give the two lead voices one point of friendly friction
  (credit, method, or access)" in casting_brief; temp 0.45 -> 0.6; max_new_tokens 520 -> 640.

### E16 (F12) article-aware style SUBSET (py selects, LLM writes)
Score styles by tag/keyword match vs key_terms+premise, top ~12, seeded pick within subset.
This is selection, not story mutation -- allowed under the directive, and it stops handing the
LLM an incoherent style/premise fusion.

## 3. Wave 3 -- THE EXCISION (the py hacksaw goes away)

### X1 -- cliche auto-substitution
Remove `repair_cliche_span` and both call sites (`nodes/_otr_line_composer.py:2592-2596` on the
kept reroll, `:2635-2640` on the kept original) and the `cliche_replacements` tables from ALL
FIVE `nodes/story_rules/*.json`. Keep: `find_cliche_phrase` detection, the quality-flag ->
reroll-with-hint pass, the defect-score keep-the-better-draft judge, and the
`cliche_shipped_after_reroll` telemetry stamp (now the honest signal instead of a silent
rewrite). Tests + Bug Bible entries that pin `cliche_repaired` behavior get updated
deliberately in the same commit (Three-File Contract).

### X2 -- dormant pack fields
After E10 salvages the examples: delete `examples`, `tone_guardrails`,
`forbidden_plot_patterns`, `forbidden_leakage_terms` from the StoryPack schema
(`nodes/_otr_story_pack.py:85-87`, `:158-159`), all six pack JSONs, and the validators. Their
QA intent is replaced by codex A4's LLM-free lane-vocabulary contract test (which checks the
ACTIVE prompts, not dead metadata).

### X3 -- dormant action-strip lever
`OTR_COMPOSER_ACTION_STRIP` (default OFF, never lit): remove lever + prompt line + strip
branch. The always-on anti-stage-direction rider + stop strings + the format strip pipeline
already cover it, and those stay (format hygiene, not story rewriting).

### X4 -- 0-caller few-shot infra
`render_few_shot_block` is pinned at 0 production callers; examples now live inline in seams
(E10). Delete the infra and its caller-count pin.

### Explicit KEEP list (so the excision never overreaches)
- JSON schema ladders + structured_call retry/repair (format, fail-loud).
- Stage-direction / speaker-label / stop-string strips (non-spoken artifacts; protects TTS).
- Phantom-name gate + `cast_strip` near-miss remap to locked cast spelling (consistency
  hygiene on proper nouns, not prose rewriting).
- Stage-3 validators (ON in production): they REROLL with findings as hints -- the LLM fixes.
- Quality-flag scorer + defect-score draft picker: py judges and chooses; it writes nothing.
- news_interpreter grounding validators V0-V3 + prune-to-floor (they gate facts, not prose).

## 4. Sequencing + gates

1. Wave 1 (one or two commits): JSON edits E1-E10. Gates: JSON parse, registry/routing tests,
   prompt-snapshot updates, byte-identity pins re-pinned deliberately where science seam text
   changed (E6 changes science pack text -- test_story_pack_stage1 pins WILL fire; that is the
   point). Regression suite + Bug Bible.
2. Wave 2: E11-E16, one lever per commit; E11 carries the canonical-workflow edit in the same
   change. Prompt snapshots per codex A7 (shape assertions, not golden prose); live 12B smoke
   after unit gates.
3. Wave 3: X1-X4 excision, one commit per X-item, tests updated in the same commit.
4. Every green chunk: commit AND push v2.0-alpha, same session.

## 5. What "better story" means here (acceptance shape, per codex A7)

On the three fixed source payloads per lane: opposed wants exist before the outline; media
resolutions show a paid cost; no lane-leakage vocabulary in active prompts; coda openers vary
across the fixture set; distinct speech registers per character (spot-audit); zero
`cliche_repaired` stamps (the flag no longer exists) and `cliche_shipped_after_reroll` used
only as telemetry, trending down as the prompts improve.
