# Stage 4 arc CONVERGED (r4) -- BUILD-READY

Plan of record: docs/multimodal-story-schema/STAGE4_SUBPLAN.md v4 FINAL +
these r4 folds:
- StoryRules carries COMPILED cliche_replacements; repair_cliche_span/
  find_cliche_phrase gain rules=; every compose_line repair/find path passes
  the same _story_rules (r4 M1).
- Loader validates REPLACEMENT TEMPLATES fail-loud (re.sub probe against the
  compiled pattern -- bad backrefs die at load, not in the :744 swallow)
  (r4 M2).
- Router `repo_id` widened/documented `str | None` for the BUG-417 routing
  (r4 S1); science repo-None keeps _SYSTEM_PROMPT object identity.
- The test_story_quality_cliche :63 every-pattern-has-a-repair coverage test
  gets a science_news.json twin (r4 S2).
- r4 verify-at-build checklist items 1-7 adopted as the build close-out.

Arc: r1 codex (rules dir collision w/ story-pack sweep; inventory holes;
premise amendment) -> r2 codex + 3-LENS SONNET FAN-OUT (threading map to
every consumer incl. compose_announcer_outro's missing param, the reroll/
spine BUG-LOCAL-417, the DEAD stage3 seed, scan-script 3rd resolve site,
exchange bypass disposition, JSON \b backspace trap, B7 verification) ->
r3 codex (writer-resolve ordering, draft repo-None no-op catch, spine scope
fix, dup-key + control-char lint) -> r4 codex (replacements threading +
template validation). Extraordinary cross-model convergence at every round.

Extraction method decision (judge): science_news.json is GENERATED
programmatically from the live constants (json.dumps of the actual pattern
source strings) -- hand-escape fidelity risk eliminated by construction;
the extraction test then pins the round-trip.
