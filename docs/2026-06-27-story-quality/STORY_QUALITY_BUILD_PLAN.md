# OTR Story-Quality Build Plan -- R2 FINAL (judge-grounded synthesis)

Status: R2 convergence complete across all three panels (AntiGravity + panel-2 +
Codex). The judge ground-checked every touch point against the live Windows files;
the weaker plan's 4 grounding errors were corrected in R2 pass 1 and are not
reintroduced. Codex's one residual disagreement -- do NOT add a `speaker_speech_signature`
field or another prompt line for 3.7 -- is ACCEPTED as the grounded position
(`build_voice_card` already emits "speaks:" L1122; the prompt already orders exact
register matching L1177; the real lever is `diversify_speech_signatures` L1410). This
doc is the coder-ready spec and the input to the WIRING roundtable (R3).

Companion: STORY_QUALITY_SYNTHESIS.md (R1 critique + panel-grounding ledger). Hard
constraints unchanged: reuse the existing reroll loop (`_otr_reroll.run_targeted_reroll`
L497; in-line gate in `compose_line` L2270/~L2364); NO ledger-schema change (ride
`meta` + `compose_flags` only); NO workflow-JSON change, no new nodes/widgets;
model-agnostic (lift the weak end, never rewrite the strong); deterministic, never
raise; SFW, UTF-8 no BOM.

## Verified touch points (judge ground-check -- all confirmed in code)

| Item | File / symbol | Line | Confirmed |
|---|---|---|---|
| 3.1 | `_otr_dramatic_state_llm.py` `_TEMPLATES` (ownership tuple) | L187 / L196-199 | yes |
| 3.1 | `_otr_dramatic_state_llm.py` `_pick_templates`, `_fallback_state` (`term=key_terms[0]`) | L238 / L246, L252 | yes |
| 3.1 | `_otr_specificity.py` `derive_central_object` | L82-129 | yes |
| 3.2 | `_otr_line_composer.py` `LineRequest.canon_header` (rendered into prompt) | L774 / L1294 | yes |
| 3.2/3.4 | `compose_line` quality gate (one reroll, no re-verify) | L2270 / ~L2364 | yes |
| 3.3 | `_otr_line_hygiene.py` `detect_stage_business_for_reroll`, `is_third_person_action_clause`, `_NARRATION_VERBS`, `_PRONOUN_ROOTS` | L430 / L384 / L137 / L210 | yes |
| 3.3 | `compose_line_draft` already calls the stage detector | L1992 / L2136 | yes |
| 3.4 | `_CLICHE_RES`, `flag_cliche` | L628 / L649 | yes |
| 3.5 | `_CODA_FACT_MAX=200`, `compose_news_coda` (cap L3162), `validate_news_coda_bridge`, `detect_mojibake` | L3104 / L3149 / L3127 / L871 | yes |
| 3.5 | `story_quality_scan.py` `find_outro_text`, `r2_lever_metrics` | L233 / L266 | yes |
| 3.6 | `_otr_story_quality_l12.py` `_enrich_tail` (`{cost}` in five tails), `_PERSONAL_COST`, `build_sq_data` | L880 / L856,858,860,872,874 / L678 / L723 | yes |
| 3.7 | `_otr_line_composer.py` `build_voice_card` ("speaks:"), register instruction in prompt | L1073 / L1122 / L1177 | yes |
| 3.7 | `_otr_casting.py` `diversify_speech_signatures` (already dedupes exact; called in lock_cast) | L1410 / L1646 | yes |

## Judge build-notes

- BN-1 (3.3 false positive) -- OPEN. The structural rule (leading `-s`/`-ing` participle,
  no 1st/2nd-person pronoun, no quote, <=32 words) would trip terse dialogue like
  `Looks like rain, Watson.` No panel closed this. The build MUST add a dialogue-opener
  guard (allowlist `looks like / sounds like / seems / feels like / smells like`, and/or
  require a comma-separated action chain or a true 3rd-person subject). Highest
  false-positive risk in the plan.
- BN-2 (3.7 cast-time mechanism) -- RESOLVED by Codex. `diversify_speech_signatures`
  (L1410) already exists, draws from a signature pool, dedupes EXACT collisions, and is
  called in `lock_cast` (L1646). Fix = tighten it to collide on NEAR-duplicates. No new
  Python table, no `speaker_speech_signature` field, no extra prompt line.
- BN-3 (3.6 banned-phrase home) -- RESOLVED by Codex. Add a dedicated
  `flag_personal_cost_boilerplate(text)` in `_otr_line_hygiene`; do NOT overload the
  existing announcer-scoped `_BANNED_THESIS_RES` (L594).
- BN-4 (3.5 scan imports) -- OPEN (trivial). `story_quality_scan.py` must import
  `detect_mojibake` and `_TERMINAL_PUNCT` from `_otr_line_hygiene`.

## Step 0 -- BASELINE FIRST (read-only; run before any code change)

CPU-only, no GPU, no :8000.

    python scripts/story_quality_scan.py --ledgers "...\output\otr\episodes\signal_lost_*_20260627_*\audio\*_ledger.json" --label r2-pre --json-out docs/2026-06-27-story-quality/r2_pre.json --md-out docs/2026-06-27-story-quality/r2_pre.md

Record existing fields: `cliche_lines`, `on_the_nose_lines`, `stage_business_lines`,
`leading_stage_dir_lines`, `narration_self_address_lines`, `voice_distinct_ratio`,
`outro_hedge_vs_resolved`. The first implementation pass adds the new counters; rerun
the same command for true pre/post local-vs-frontier numbers (replaces all estimates).

## Shared helper (new, in `_otr_line_composer.py` near the gate ~L2364)

```
def _quality_flags_for_line(cleaned: str, req: LineRequest) -> list[tuple[str, str, str]]:
    # returns [(code, reason, compose_flag), ...]; count via len(), residuals via the flag.
    # Uses flag_cliche / flag_stage_business / flag_on_the_nose / flag_objective_literal
    # + flag_anchor_stuffing + flag_one_breath + flag_personal_cost_boilerplate.
    # Anchor/one-breath/cost apply ONLY when req.speaker_role == "character" (never coda).
```

`compose_line()` gains `_quality_repair_attempted: bool = False`. The existing one-reroll
pattern remains; after the reroll, re-run `_quality_flags_for_line()` for residual
`compose_flags` only -- no second LLM call.

---

## 3.1 Dignity / safety guard (BUILD FIRST)

Defect (frostbite_facility, `dramatic_state_source: "fallback"`):
`character_a_wants: "take sole credit for transgender people"`;
`ending_change: "Control of transgender people passes to whoever is willing to pay the
higher price."` Quality + dignity failure that poisons the whole episode.

Touch: `_otr_dramatic_state_llm._fallback_state` (L246; term=key_terms[0] at L252) +
`_pick_templates` (L238); `_otr_specificity.derive_central_object` (L82-129, secondary).

Signatures:
```
def is_nonownable_story_object(term: object) -> bool: ...
def derive_safe_fallback_term(key_terms: object, cast: object = (), default: str = "the findings") -> str: ...
def _pick_templates(term: str, arc_shape: str = "", *, allow_ownership: bool = True) -> tuple[str, str, str, str]: ...
```

Rule: `_PEOPLE_NOUNS` (people, person, men, women, children, residents, patients,
workers, citizens, population, community, group, humanity, victims) + a protected-identity
/ harm-population set (transgender people, suicide thoughts, violence victims). `_fallback_state`
selects the first key_term that passes `is_nonownable_story_object` (else `default`) BEFORE
`_inject_key_term`/`_pick_templates`; `_pick_templates(..., allow_ownership=False)` omits the
ownership tuple (L196-199) when the term is people-class. `derive_central_object` also skips
people-class terms (secondary; the REQUIRED guard is `_fallback_state`). Stamp freeform meta:
`dramatic_state_fallback_term`, `dramatic_state_fallback_term_replaced`.

Catch: `transgender people` -> no "control/credit transgender people". Leave alone:
`patient records`, `children's health study`, `survey data`, `the decryption machine`
(object heads with people modifiers are fine).

Counters: `nonownable_central_object_count`, `ownership_template_on_nonownable_count`,
`dramatic_state_fallback_replaced_count`. Reuse `wants_default`, `has_central_object`.
Target: 0 ownership templates on people-class objects (frostbite 1 -> 0). Ship always-on.

Tests: `is_nonownable_story_object("transgender people") -> True`;
`is_nonownable_story_object("patient records") -> False`; `_fallback_state` with
key_terms=["transgender people"] yields no "take sole credit"/"control what becomes of"
string.

## 3.2 Anchor-stuffing + one-breath gate (highest-leverage quality lever)

Touch: `flag_anchor_stuffing` / `flag_one_breath` / `extract_specificity_anchors_from_header`
(new, `_otr_line_hygiene`); the shared `_quality_flags_for_line` at the `compose_line` gate
(~L2364); `story_quality_scan.r2_lever_metrics` (L266).

Signatures:
```
def extract_specificity_anchors_from_header(canon_header: object) -> list[str]: ...
def flag_anchor_stuffing(text: object, anchors: object, *, threshold: int = 3) -> tuple[bool, str]: ...
def flag_one_breath(text: object, *, max_words: int = 28, max_clause_markers: int = 3) -> tuple[bool, str]: ...
```

Rule: anchors parsed from `req.canon_header` (the injected "Specificity anchors (when
natural...):" block) -- no `LineRequest` field. Anchor-stuffing trips at >=3 distinct
anchors in one character line. One-breath trips at >28 words, or >22 words with excessive
comma/semicolon/conjunction nesting. CODA/ANNOUNCER is EXEMPT (the science fact is never
trimmed -- operator rule). Reroll via the existing pattern; re-run for telemetry only.

Catch: spindle b008 (30w): `...the disputed record of the rail death now includes this
partial echo of the transgender performer's suicide thoughts and the discrimination behind
harassment and suicide attempts.` Leave alone: bar_chip b002 (11w, 1 anchor): `Name's on
the chip, Steiner. Let's see it before dawn.`

Counters: `anchor_stuffing_lines`, `one_breath_violation_lines`, `quality_residual_lines`,
`quality_retry_lines`. Reuse `cliche_lines`, `on_the_nose_lines`, `stage_business_lines`.
Risk: medium (over-trim frontier specificity) -- recommend env-flag A/B (Q5); capped by the
1-reroll max.

## 3.3 Broaden the stage-action leak detector

Defect: heatwave b008 (EMPTY compose_flags): `snaps off pen's tip, jams it into the
decryption machine's port, turning it into scrap metal`; b011 fused a leading direction
with dialogue. `_NARRATION_VERBS` (L137) has none of snaps/jams/turning/revealing.

Touch: `is_third_person_action_clause` (L384) + `detect_stage_business_for_reroll` (L430);
`compose_line_draft` already routes a hit (L2136).
```
def is_whole_line_stage_action(text: object, *, max_words: int = 32) -> bool: ...
```
Rule (structural, not whitelist): <=32 words, no quotes, 3rd-person action/participle lead,
NO 1st/2nd-person pronoun root (`_PRONOUN_ROOTS` L210). Reroll (do not silently strip) with
`_BARE_STAGE_HINT`. APPLY BN-1 dialogue-opener guard.

Catch: heatwave b008; `steps forward, revealing a keycard...`. Leave alone: links b007
`My thumb keeps catching on the log's corner...` (has "my"); `I snap off the pen's tip.`;
and (post BN-1) `Looks like rain, Watson.`

Counters: `stage_action_leak_lines`; reuse `leading_stage_dir_lines`, `stage_business_lines`,
`narration_self_address_lines`.

## 3.4 Expand cliche floor + re-verify after the reroll

Defect 1: `_CLICHE_RES` (L628) has 6 phrases; the batch shipped `hangs in the balance`
(dialing b002), `over my dead body` (marked), `we're playing with fire` (compass b012).
Defect 2: the gate rerolls once and ships the result UNCHECKED -- `You're playing with
fire, Watson` (dialing b003) shipped despite an exact match.

Touch: `_CLICHE_RES` (L628); `compose_line` re-verify (~L2415).
Rule: add `(?:you|we|i|they)['` curly `]?re playing with fire`, `hangs in the balance`,
`over my dead body`, `not on my watch`, `best left buried`, `running out of time`,
`before it'?s too late`, `safety first`, `go(es)? up in smoke`. After the reroll returns
`_rr`, compare `len(_quality_flags_for_line(cleaned, req))` vs `len(_quality_flags_for_line
(_rr.text, req))` and keep the FEWER-defect draft (original on tie); stamp
`quality_reroll_degraded` when the reroll was worse and the draft was kept.

Catch: `You're playing with fire, Watson.`; `Shut down the lab. Safety first.`. Leave
alone: `The fuse is on fire.`, `The balance arm is stuck.`, `I checked my watch.`

Counters: reuse `cliche_lines`; new `cliche_residual_lines`, `quality_reroll_degraded`.

## 3.6 Stop appending the L12 personal-cost boilerplate

Defect: `_enrich_tail` (L880) glues `{cost}` (from `_PERSONAL_COST`) onto beat intents;
"the trust they will lose either way" rides nearly every beat and leaked into a spoken line
(links b009 `...no matter who trusts either of us`).

Touch: `_otr_story_quality_l12._enrich_tail` (scrub `{cost}` from ALL FIVE tails:
L856,858,860,872,874); new `flag_personal_cost_boilerplate(text)` in `_otr_line_hygiene`
(BN-3), wired into `_quality_flags_for_line` at the composer tail for character lines. Keep
`personal_cost` in the freeform SQ dict for telemetry only.
```
def flag_personal_cost_boilerplate(text: object) -> tuple[bool, str]: ...
```
Catch: `the trust they will lose either way`, `what it costs them to be the one who
decides`. Leave alone: a concrete consequence (`loses access to the observatory archive`).

Counters: `cost_tail_in_intent_count`, `cost_tail_in_dialogue_count`,
`personal_cost_boilerplate_lines`. Target: 0/0.

## 3.5 Coda execution counters (measurement-only; coda fact untouched)

Operator rule: the coda MUST land the real science fact -- execution-only, never trim it.
Touch: `compose_news_coda` (L3149) + `_news_coda_fact_flags` (new); read at scan via
`find_outro_text` (L233) in `r2_lever_metrics`; reuse `detect_mojibake` (L871) +
`_TERMINAL_PUNCT` (BN-4).
```
def _news_coda_fact_flags(raw_fact: str, cleaned_fact: str) -> tuple[str, ...]: ...
```
Rule: flag (do not drop) when the fact was capped by `_CODA_FACT_MAX` (L3162), mojibake
appears, the bridge fell back (`news_coda_fallback`), or a generic bridge shipped
("The real story:" / "The true account:").

Catch: `SaarbrÃ¼cken` (mojibake); a `news_close_brief` over 200 chars (truncated). Leave
alone: a dense but clean coda after a valid bridge (`As the LINK broadcast unfolds: Launch
successful; LINK begins Swift Observatory altitude boost.`).

Counters: `news_coda_truncated_count`, `news_coda_mojibake_count`,
`news_coda_generic_bridge_count`, `news_coda_fallback_count`. Target: truncation + mojibake
0; fallback visible and trending down. (Hardening `compose_news_coda` to reroll on
truncation / vary the bridge is a later pass, not this build.)

## 3.7 Register divergence, not more prompting (demoted; cast-time + measurement)

Defect: signatures already ship (`build_voice_card` "speaks:" L1122; prompt orders exact
match L1177) but the local model ignores it; the two principals read identically because
their signatures are near-duplicates ("measured, precise, weary" vs "measured, concise")
that the EXACT-only `diversify_speech_signatures` (L1410) does not collide.

Touch: `_otr_casting.diversify_speech_signatures` (L1410, called in `lock_cast` L1646);
`story_quality_scan.r2_lever_metrics`; optional line-level overlap reroll in `compose_line`
only when `story_quality_v2_enabled`.
```
def speech_signature_overlap(a: object, b: object) -> float: ...
def flag_low_register_divergence(cast: object, *, threshold: float = 0.67) -> tuple[bool, str]: ...
def flag_line_register_overlap(text: object, speaker: str, ledger_context: object) -> tuple[bool, str]: ...
```
Rule: do NOT add a prompt line or a `speaker_speech_signature` field (Codex divergence,
accepted). Tighten `diversify_speech_signatures` so NEAR-duplicates (token overlap >=
threshold) collide and get reassigned from the existing pool. Add a scan counter for
register overlap; the line-level reroll is optional and flag-gated.

Catch: cast pair `measured, precise, weary` / `measured, concise` (reassign one). Leave
alone: `clipped, procedural` / `warm, rambling`.

Counters: fix/keep `voice_distinct_ratio` (read from `ledger["cast"]` /
`meta["cast_voice_slots"]`); new `speech_signature_near_duplicate_count`,
`register_overlap_lines`.

---

## Build sequence (Codex order, judge-confirmed)

0. Step 0 baseline scan + add the new counters first (true pre/post numbers).
1. 3.1 dignity guard -- safety defect, sits on `_fallback_state`, independent of reroll.
2. 3.2 anchor/one-breath -- builds the shared `_quality_flags_for_line` reroll/recheck path.
3. 3.3 stage-action leak -- draft-stage detector wiring already exists; catch leaks before cleanup hides them.
4. 3.4 cliche floor -- reuses the 3.2 gate + residual measurement.
5. 3.6 cost-tail scrub -- changes intent text; measure after the dialogue floor exists.
6. 3.5 coda counters -- execution-only; coda stays exempt from compression gates.
7. 3.7 register divergence -- real but less urgent than safety / leakage / boilerplate.

Run `story_quality_scan` after each step; commit per green chunk per CLAUDE.md.

## Inputs for the WIRING roundtable (R3)

- This build is pure-python nodes + the scan script: NO litegraph/workflow-JSON wiring,
  NO new nodes/widgets. R3 "wiring" = call-site wiring + the `_quality_flags_for_line` seam
  + scan-counter plumbing + per-item enablement.
- Close BN-1 (3.3 dialogue-opener guard) and BN-4 (scan imports). BN-2/BN-3 resolved above.
- Decide enablement: always-on (3.1, 3.5) vs `story_quality_v2_enabled`-gated (3.2, 3.7).

## Open operator questions

1. One-breath threshold = 28 words (and ~3 clause markers)? Hard reroll or soft warn?
2. Anchor cap = 3 distinct anchors per character line -- confirm (coda exempt is specced).
3. Cliche list governance: keep hand-extending `_CLICHE_RES`, or move phrases to a data file
   you can edit without a code change?
4. 3.1 people-noun / protected-identity list: confirm the starting set and the
   `"the findings"` default term.
5. Ship the new gates always-on, or behind `story_quality_v2_enabled` (per item) for A/B?
6. Promote StoryCritic `arc_verdict == "uneven"` into `_BLOCKING_ARC_VERDICTS`, or keep
   advisory and rely on the per-line gates?

---

## R3 WIRING -- FINAL (converged: Codex + AntiGravity/Opus-4.6, judge-grounded)

Both R3 pass-2 plans converge on structure. The judge resolves three residual diffs
(W-D, W-E) and one budget correction neither panel fully nailed (W-B). All line anchors
verified against live code.

### W-A. Clean-stage gate structure (`compose_line` ~L2364)
Extend the EXISTING block; add `_quality_repair_attempted: bool = False` to `compose_line`
(next to `_stage_dir_repair_attempted` L2295, `_leak_repair_attempted` L2296). Guard:
`if not _stage_dir_repair_attempted and not _quality_repair_attempted:`. Run the existing
always-on flags (cliche/stage_business/on_the_nose) + 3.3-expanded cliche, then the
v2-gated subset (`req.story_quality_v2_enabled and speaker_role=="character"`):
`flag_anchor_stuffing(cleaned, req.canon_header)`, `flag_one_breath`, `flag_banned_personal_cost`,
plus the existing objective-literal. On any hit, ONE recursive reroll; then re-verify
(keep the draft with fewer `_quality_flags_for_line` hits; stamp `quality_reroll_degraded`
if the reroll was not better).

### W-B. Reroll budget + guard propagation (JUDGE CORRECTION)
THREE reroll sites are live: draft stage-dir (`_sd_reroll_done`, in `compose_line_draft`),
the new clean-quality gate (`_quality_repair_attempted`), and leak-floor-v2 (`_leak_repair_attempted`,
L2469, recurses at L2498). Two hard requirements the panels under-specified:
- **Thread all three guards on EVERY recursive `compose_line` call.** The clean-quality
  recursion must pass `_leak_repair_attempted=_leak_repair_attempted` (Opus had this) AND
  the existing leak-floor recursion (L2498-2514) must be UPDATED to pass the new
  `_quality_repair_attempted` (it currently cannot -- add it), or a leak reroll could
  re-open the quality gate.
- **Cap at <=3 generate calls/line:** have the clean-quality recursion ALSO set
  `_leak_repair_attempted=True` (makes quality-reroll and leak-reroll mutually exclusive;
  the deterministic freeze floor backstops any residual leak). Without this, the true worst
  case is 4 (initial + stage-dir + quality + leak), not the 3 both panels claimed. Recommend
  the <=3 cap; flag the trade for the operator (a quality-rerolled line forgoes its one
  leak-floor LLM pass but still gets the freeze-floor strip).

### W-C. Counter source map (single source of truth -- engine and scan use identical strings)
| Counter | Source |
|---|---|
| `anchor_stuffing_lines`, `one_breath_violation_lines`, `fused_action_lines` | scan re-runs the detector on final text (no engine stamp; the reroll hint is not a flag) |
| `quality_reroll_degraded` | engine-stamped `compose_flags` |
| `quality_retry_lines`, `quality_residual_lines` | engine-stamped `quality_residual:*` / `*_retry` flags |
| `news_coda_truncated_count` | engine-stamped `news_coda_truncated` at `compose_news_coda` L3162 (only it knows it capped at `_CODA_FACT_MAX`) |
| `news_coda_mojibake_count`, `news_coda_generic_bridge_count` | scan-derived (`detect_mojibake`; `BRIDGE_GENERIC_OPENERS` prefix) |
| `news_coda_fallback_count` | engine-stamped (existing `news_coda_fallback`, L3193) |
| `dramatic_state_fallback_count`, `ownable_people_object_count` | scan-derived from `meta.dramatic_state_source` / `meta.central_object` |
| `dramatic_state_fallback_replaced_count` | engine-stamped `meta.dramatic_state_fallback_term_replaced` |
| `cost_tail_in_intent_count`, `cost_tail_in_dialogue_count`, `personal_cost_boilerplate_lines` | scan-derived (`beat_intent` + spoken `text`) |
| `speech_signature_near_duplicate_count`, `register_overlap_ratio`, `voice_distinct_ratio` | scan-derived from `cast[].speech_signature` |

### W-D. Hint composition (RESOLVED -- panels disagreed)
Priority (Opus's "speakability first" beats Codex's "leakage first"): `one_breath` >
`anchor_stuffing` > `personal_cost` > `cliche` > `stage_business` > `on_the_nose` >
`objective_literal`. Send TOP-1 only, EXCEPT `one_breath`+`anchor_stuffing` collapse into a
single combined hint ("Rewrite as one spoken beat under ~20 words, using at most one
concrete detail") since they co-occur and share the rewrite. 240-char cap. Order is tunable
from the scan once baseline numbers land.

### W-E. 3.6 tail fix (RESOLVED -- DROP, do not replace)
Codex/judge over Opus: in `_enrich_tail`, DROP the `{cost}` clause entirely (e.g. L856
`"the pressure tightens around {obj}, and {cost} now rides on it"` -> `"the pressure tightens
around {obj}"`), keeping only the `{obj}` enrichment. Opus's "replace with a generic cost
phrase" ("and the cost is now in play") still appends boilerplate to every beat -- which is
the exact homogenization 3.6 exists to kill. `personal_cost` stays in the SQ dict for
telemetry; it is simply never formatted into the tail.

### W-F. v2-flag threading (verified end-to-end)
`meta["story_quality_v2_enabled"]` (writer) -> `LineRequest.story_quality_v2_enabled` (L886,
first-pass) -> reroll rebuild (`_otr_reroll.py`) -> read at `compose_line` L2373 (verified).
Gated by v2: 3.2 + 3.6 only. Always-on: 3.1, 3.3, 3.4, 3.5, 3.7.

### W-G. Tests + verified node-ids
Gate node-ids confirmed in code: `tests/test_b7_forbidden_sweep.py::test_forbidden_sweep_runs_clean`
(L72) and `tests/test_cast_voice_replay_parity.py::test_replay_matches_lock_cast_byte_identical`
(L61) -- the latter MUST pass after 3.7. Update files (names verified to exist):
`test_story_r2_c3_voice_distinct.py` (3.7 near-dup), `test_story_quality_scan_r2.py` (new
counters), plus the cliche/hygiene/dramatic-state/L12 suites. Coder: confirm the exact
`TestDiversify::<method>` name in test_story_r2_c3_voice_distinct.py before pinning it.

### W-H. Re-baseline + green gates (run before code, and after each green chunk)

    $env:PYTHONUTF8='1'
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\story_quality_scan.py --ledgers "...\output\otr\episodes\signal_lost_*_20260627_*\audio\*_ledger.json" --label r3-pre --json-out docs\2026-06-27-story-quality\r3_pre.json --md-out docs\2026-06-27-story-quality\r3_pre.md

Green: `ownership_template_on_nonownable_count==0`, `stage_action_leak_lines==0`,
`personal_cost_boilerplate_lines==0`, `news_coda_mojibake_count==0`,
`news_coda_fact_truncated_count==0`, `speech_signature_near_duplicate_count==0`,
`quality_residual_lines` not increasing after the local-vs-frontier split. Regression: focused
new tests -> full suite -> Bug Bible -> B7 sweep -> cast replay parity, before commit.

---

## R4 FINAL POLISH -- coder cautions (GO; Codex R4 grounded by judge)

R4 converged. **GO for coder handoff** with six must-fix implementation cautions, all verified
against live code. No residual disagreement with R3.

- **MF-1 (NEW -- budget/correctness; supersedes W-B "three sites").** There are FOUR recursive
  `compose_line` sites: clean-quality (~L2415), leak-floor-v2 (L2498), AND the Stage-3 validator
  repair (L2592) -- which today passes ONLY `_stage3_repair_attempted=True` (L2605) and omits
  the rest, so a stage-3 repair re-opens draft/clean/leak/quality. RULE: every recursive
  `compose_line` call threads ALL guards (`_stage_dir_repair_attempted`, `_quality_repair_attempted`,
  `_leak_repair_attempted`, `_stage3_repair_attempted`). Fix L2592 + L2498 + the clean-gate
  recursion. (Stage-3 is opt-in via `enable_stage3_validators`, off by default, but propagation
  must be correct regardless.)
- **MF-2 (coda truncation).** `clean_one_line(text, max_chars)` treats `max_chars<=0` as no
  truncation (L2798/L2806). Stamp `news_coda_truncated` only when
  `clean_one_line(brief, 0) != clean_one_line(brief, _CODA_FACT_MAX)` -- never raw `len>200`.
- **MF-3 (anchors).** `flag_anchor_stuffing` / `extract_specificity_anchors_from_header` must
  `re.escape` each anchor (key_terms carry "41.3 degrees C", "837/835", hyphens, quotes) and
  build the reroll hint from canon-header order, not set iteration (determinism).
- **MF-4 (scan import).** Keep new detectors stdlib-only; the guarded import at
  `story_quality_scan` ~L70 catches any Exception and no-ops ALL R2 metrics on one bad import.
  Add a test that FAILS if the guarded import fell back (assert `_HAS_R2_HELPERS` / real
  detection) so a typo can't silently zero the metrics.
- **MF-5 (single source of truth).** `_quality_flags_for_line` is the only scorer for the gate
  AND the re-verify, applying the SAME v2 gating (3.2 + 3.6 v2-gated; cliche/stage/nose
  always-on; coda never enters). Prevents the re-verify rejecting a reroll for a flag the gate
  never raised.
- **MF-6 (flag-once).** Hold retry/residual/degraded flags in local tuples until the final
  result is chosen; append once. Follow the existing Stage-3 dedupe pattern (the `_seen` set at
  L2608) so no duplicate `cliche_retry` / `quality_reroll_degraded`.

Nice-to-have: defensive `candidates = filtered or tmpls` in `_pick_templates`; surface
`_HAS_R2_HELPERS` in the scan md; keep flag IDs short ASCII, no punctuation beyond ":" so
`aggregate_compose_flags` (L681) groups stably.

CONFIRMED-CLEAN (judge-verified): DramaticState validators compatible with the "the findings"
default; `diversify_speech_signatures` seed-rotated only, no rng; no node
INPUT_TYPES/widget/workflow-JSON/OTR_WorkflowValidator touch (restart Comfy to load the edits --
module cache); ASCII logs / UTF-8 no BOM; scan is read-only. **Verdict: GO.**

### R4 reconciled (Codex + Opus-4.6, judge-adjudicated) -- CONVERGED, GO

Both R4 panels returned GO; this is convergence (R4 is the final round). Adjudications where
they differed or refined:

- **Budget -> accept <=4 generate calls/line (Opus), NOT the force-<=3 hack (Codex).** The
  binding requirement is guard PROPAGATION across all FOUR recursion sites -- clean-quality
  (~L2415), leak-floor (L2498), Stage-3 (L2592), and the internal draft -- so each fires at
  most once. <=4 equals the existing status-quo ceiling (no regression) and keeps the
  leak-floor LLM recompose on quality-rerolled lines. This SUPERSEDES the W-B "<=3" line.
  OPERATOR CHOICE (Q7): to shave the worst case to <=3, set `_leak_repair_attempted=True` on
  the clean-quality recursion (1 kwarg) -- trade-off: a quality-rerolled line forgoes its
  leak-floor LLM recompose but still gets the deterministic freeze-floor strip. Default = <=4.
- **Coda truncation (MF-2 refined):** stamp `news_coda_truncated` iff
  `len(clean_one_line(brief, 0)) > _CODA_FACT_MAX` (hygiene-only clean exceeds the cap;
  `max_chars=0` disables truncation, L2798). Avoids whitespace/quote false-positives.
- **Anchors (MF-3 resolved):** use plain `phrase in text.casefold()` substring matching (the
  planned design) -- NO `re.escape` needed ("41.3 degrees C", "837/835" match literally).
  Escape only if a future impl switches to `re.search`. Build the hint in canon-header order.
- **Re-verify (MF-5 concrete):** `_quality_flags_for_line(text, canon_header, v2_enabled)`
  takes the v2 flag and skips anchor/one-breath/personal-cost when off, so keep/discard
  mirrors exactly what the gate raised.
- **Empty-template belt:** keep the 1-line `tmpls = filtered or <unfiltered>` in
  `_pick_templates` (not triggerable today; cheap insurance).
- **Scan generic-bridge:** mirror the engine's `validate_news_coda_bridge` `startswith` test
  (L3142), not a broad `in`, so a mid-sentence "meanwhile" doesn't false-count.
- **CORRECTION to Opus item 8 (module-cache):** Opus's "no restart needed" is WRONG. A
  long-running ComfyUI server caches imported modules; `.py` edits do NOT take effect on the
  next compose -- a Comfy/module RESTART is required before live validation (documented OTR
  behavior). Keep the restart step.
- **Confirmed-clean by both (judge-verified):** DramaticState validators OK with the default
  term; `diversify_speech_signatures` seed-rotated only; no node
  INPUT_TYPES/widget/workflow/validator touch; ASCII logs (`%r` escapes non-ASCII under
  cp1252) / UTF-8 no BOM; scan read-only.

**FINAL VERDICT: GO for coder handoff.** All must-fixes are local one-liners folded at coding
time -- no design change, no build-order change. One operator decision outstanding (Q7: budget
cap <=4 default vs <=3 shave).
