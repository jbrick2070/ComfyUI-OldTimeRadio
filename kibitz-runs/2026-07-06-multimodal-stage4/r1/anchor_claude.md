# Anchor review (Claude, code-grounded) -- Stage 4 rules-enforcer sub-plan v1 -- r1

VERDICT: SOUND DIRECTION, 3 self-flagged forks for the panel.

Grounding: inventory verified by grep before drafting -- _CLICHE_RES :634 /
_STAGE_BUSINESS_RES :657 / _ON_THE_NOSE_RES :751 / _CLICHE_REPLACEMENTS :700
in _otr_line_hygiene.py; validate_banned_phrases :404 in
_otr_stage3_validators.py (caller-supplied list); FORBIDDEN_GENERIC_WORDS in
_otr_compose_exchange.py (:380 join site).

SELF-FLAGGED:
- F1 (the big fork): module-enforcer vs graph-node. My draft argues module
  (rules fire inside per-line compose gates, mid-writer; a node can only lint
  post-hoc). Panel should verify no OTHER consumer needs a graph-seam surface.
- F2: per-bank full rule sets duplicate ~everything for the 3 dormant banks.
  Alternative: global defaults + per-bank override. I chose per-bank-full for
  contract simplicity; challenge welcome.
- F3: regex-in-JSON is the riskiest law call. Compiled fail-loud at load; the
  question is whether pattern SOURCES are "content". My case: these patterns
  encode taste vocabulary ("shiver down .* spine"), which is exactly what a
  public-domain lane must retune without a Python edit (the R1 rule-of-thumb).
  Structure-only regexes (stage directions) stay Python.
- Verify-at-build: where the stage3 banned_phrases SEED list actually lives
  (writer call site) -- must be found + extracted in 4A.
- B7 discipline: no `alias` loop vars; the new JSON regex strings must not
  trip the forbidden-sweep markers (they are diff-visible).
