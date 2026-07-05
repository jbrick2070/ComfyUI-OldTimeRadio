# pass01_judgment.md -- r1 judgment log

Format: per-source claim, accept/reject/verify-at-build, rationale.

## Codex (raw at ``pass01_raw/codex.md``)

| Codex claim | Status | Rationale / folded into |
|---|---|---|
| MF1 baseline SHA drift | ACCEPT | Folded to MF-C5. |
| MF2 sci-fi not byte-identical (paraphrase) | ACCEPT | Folded to MF-C6 (empty-overrides). |
| MF3 12-seam list incomplete | ACCEPT | Folded to MF-C2 + MF-C3. |
| MF4 operator scope vs anchor scope | ACCEPT | Folded to MF-C7. |
| MF5 smallest viable API undecided | ACCEPT | Folded to MF-C4. |
| SF1 Fable 2026-07-02 resolved | ACCEPT | Status recorded. |
| SF2 chooser_user_template inconsistency | ACCEPT | Folded to SF-C1. |
| SF3 informal vocabulary vs real ids | ACCEPT | Folded to MF-C3 + SF-C1. |
| CUT compat mirrors | ACCEPT | Folded to MF-C7. |
| CUT visual policy | ACCEPT | Folded to MF-C7. |
| CUT upgrades 2-5 | ACCEPT | Folded to MF-C7. |
| CUT adaptive cleanup | ACCEPT | Folded to MF-C7. |

## Fable (raw at ``pass01_raw/fable.md``)

| Fable claim | Status | Rationale / folded into |
|---|---|---|
| MF1 line_composer_system 16th site | ACCEPT | Folded to MF-C2. Verified at ``_otr_line_composer.py:1174`` + router ``:55``. |
| MF2 object-identity audio-C7 contract | ACCEPT | Folded to MF-C1. **LOAD-BEARING.** Verified verbatim at ``_otr_outline.py:1846`` + router comment ``:57-60``. |
| MF3 spec self-inconsistency | ACCEPT | Folded to MF-C8. |
| SF1 vocab alignment lab vs prod | ACCEPT | Folded to SF-C1. |
| SF2 f-string interpret variables | ACCEPT | Folded to SF-C3. |
| SF3 "no production code touched" wording | ACCEPT | Folded to SF-C4. |
| Fable 2026-07-02 all 4 must-fixes resolved | ACCEPT | Confirmed via own grounding. |

## Sonnet (raw at ``pass01_raw/sonnet.md``)

| Sonnet claim | Status | Rationale / folded into |
|---|---|---|
| MF1 TEMPLATE_SEAMS real 14 entries + labels/interpret/casting split | ACCEPT | Folded to MF-C3. Verified at ``contracts.py:25-42``. |
| MF2 adopt lab Registry subset, not new flat helper | ACCEPT | Folded to MF-C4. |
| SF1 ``_INVENTOR_SYSTEM`` variable binding | ACCEPT | Folded to SF-C2. Verified at ``_otr_style_picker.py:295-297``. |
| SF2 compat mirrors Phase B | ACCEPT | Folded to SF-C5 + MF-C7. |

## Claude anchor (raw at ``pass01_raw/claude_anchor.md``)

| Claude claim | Status | Rationale |
|---|---|---|
| MF1 StoryPromptProfile seam coverage | RESOLVED | Panel grounding shows coverage complete (14 + labels + interpret facade + casting content field). Dropped as blocker. |
| MF2 bridge artifact under-specified | DEFERRED | Panel MF-C7 cuts bridge from Phase A entirely. Phase B concern. |
| MF3 ``_otr_ledger_input_adapter.py`` production edit | DEFERRED | Panel MF-C7 cuts adapter from Phase A. |
| MF4 ``interpret`` f-string | RESOLVED | Interpreter binding, not template seam (per Fable step 6 + Sonnet). Not extracted. Folded to SF-C3. |
| SF1 line ref drift Fable review | ACCEPT | Fable review cited ``:1642`` for grounding; live is ``:1621``. Cited ``:3386`` for coda; live is ``:3407``. r2 re-audits Fable step 6 refs. |
| SF2 story-pack selection mechanism | DEFERRED | Phase B (runtime routing widget). Phase A operates on hardcoded default triple. |
| SF3 test_creative_prompt_router_exact_match.py:60 object identity | ACCEPT | Folded to MF-C1 (same root cause). |
| SF4 _STILL_WORD_TYPOGRAPHY / _BACKDROP | DEFERRED | Phase B visual policy scope. |
| SF5 IS_CHANGED caching | DEFERRED | Phase A extractor is read-only + module-cached; concern relaxes. Phase B re-checks. |

## Rejected / verify-at-build

- None. All panel claims accepted, folded, or deferred to Phase B.
- ``PRODUCTION_MIRROR_MANIFEST.md`` presence in sibling repo is
  UNVERIFIABLE this pass. r2 grep confirms or refutes -- if absent,
  document the manifest as future work; do not block Phase A on it.

## Panel diversity dividend

- **Codex** caught the paraphrase risk in ``science_news_default.json``
  and the baseline SHA drift -- pure mechanical grounding.
- **Fable** caught the audio-C7 object identity contract -- narrative-
  judgment adjacent (understood the "why" of the router's identity
  guarantee), spotted the byte-identity cascade the mechanical panels
  didn't (matches the 2026-07-03 Fable reality-exception in CLAUDE.md
  section 9).
- **Sonnet** caught the actual ``TEMPLATE_SEAMS`` vocabulary vs the
  operator's loose "12" and the correct architectural adoption target.
- **Claude anchor** provided the grounding scaffold (15-site inventory,
  Fable review's 4 MUST-FIX resolution status, sci-fi payload catalog)
  that the panel graded against.

Convergence: 4 sources -> 8 MUST-FIX (deduplicated to MF-C1..MF-C8) +
6 SHOULD-FIX (SF-C1..SF-C6). Zero hallucinations. Ready to advance.
