# OTR Story-Quality Update Plan -- DRAFT (kibitz input, 2026-06-28)

Author: Claude (Cowork), grounded against the real Windows files + the overnight
all-visualizer soak batch. This is the r1 INPUT for the 3-way kibitz (Codex +
Antigravity + Claude as code-grounded panelist AND judge). It will be hardened
r1 (arc) -> r2 (coding) -> r3 (wiring/sequencing) -> r4 (convergence).

## 0. Scope, non-goals, invariants

- **Lane:** writer / composer / critic / dramatic-state CONTENT only. **CPU.
  NO workflow JSON. NO GPU. NO node/widget changes.** (Operator-directed
  exception to the otherwise-parked story-pipeline.)
- **Length: re-entered scope 2026-06-28 (operator) -- but via gate-tuning, NOT
  padding.** Initial steer was "don't chase length"; the 2000w/1340w probes then
  proved `target_words` is a near-no-op (grok @ 720w->213 words, @ 1340w->263, @
  2000w->ERROR over the ~1363 structural ceiling). Root cause is G1 (the
  `one_breath`/`anchor_stuffing` gates cap every line at ~22-28 words). So length
  is addressed as a SIDE EFFECT of fixing G1 -- lines get fuller because the gate
  stops over-compressing them. NEVER pad to hit a word count; never add a length
  normalizer.
- **Default-OFF / byte-identical-when-able.** Every change rides the existing
  `story_quality_v2` gate (or a new sub-flag); flag-OFF must be byte-identical
  (`test_audio_byte_identical` stays green).
- **Reuse the existing reroll loop** -- `_otr_line_composer.compose_line` in-line
  gate (~L2364) + `_otr_reroll.run_targeted_reroll`. No new reroll engine.
- **NO ledger-schema change** -- new signals ride the freeform `meta` dict + the
  freeform `compose_flags` list (already arbitrary strings).
- **Builds ON the shipped 2026-06-27 gate-seam cluster (3.1-3.7).** Do not
  regress it (proof it works in section 2).
- **Future-idea (PARKED, not in scope):** an agentic-CLI "showrunner" writer
  (codex/agy) that reads the downstream gates and writes pre-cleared lines
  (operator brainstorm 2026-06-28). Real but a separate architecture lane;
  collides with seed-keyed determinism. Recorded, not built here.

## 1. Evidence base

- **Overnight all-visualizer soak (2026-06-28):** 27 successful episodes (1 leg
  errored), writers mistral-nemo + gemma-4-E2B, 420w, creativity sweep
  (balanced / wild&rough / maximum chaos / safe&tight), $0 (100% local).
- **+ 2 enrichment renders (this session, operator-directed):** mistral-720 +
  frontier grok-4.3-720, all-visualizer -- a higher-word-room sample + a frontier
  "north-star" reference.
- **Census:** `scripts/story_quality_scan.py` over the ledgers ->
  `docs/2026-06-28-story-quality-kibitz/scan_soak.json`.

**Enrichment findings (mistral-720 `artifacts_breath` + grok-720
`plancks_vanishing_horizon`, both read in full):**
- **Under-length is a HARD STRUCTURAL CAP, confirmed 3 ways.** mistral-720 (209
  voiced words) and grok-720 (213) both produced exactly 18 lines / ~210 words --
  identical to the 420w batch. Even a frontier model cannot lengthen it; the
  ~14-character-line outline skeleton caps it regardless of writer or target.
  -> length is correctly out of scope (section 0).
- **Coda verify RESOLVED:** grok-720 got a real premise-derived bridge
  (`news_coda_bridge`, NOT fallback). So the 18/29 fallback rate is a WEAK-LOCAL
  bridge-generation gap, not a too-strict validator -> S2 = lift the floor, do not
  loosen the gate.
- **Seed verify RESOLVED:** grok-720 is richly ON-premise (Planck papers /
  retraction / academic betrayal woven through every beat on a vivid storm-gondola
  set); local is intermittent (artifacts_breath stayed on-theme, dance_of_keys
  drifted) -> S1 = lift LOCAL toward frontier; it is variance, not a constant.
- **NEW (G1 -- LEAD): the line-compression gates cap length AND degrade craft.**
  Nearly every grok-720/1340 line is stamped `anchor_stuffing_retry,
  one_breath_retry, body_gate_reroll`, and the rerolls compress grok's rich lines
  into noun-salad / ungrammatical fragments (e.g. b03 "Storm the helm Quasimodo
  protects his paradox self to bury the merger disclosure"). Grounded mechanism:
  `_otr_line_hygiene.flag_one_breath(max_words=28, soft 22 + >3 clauses)` +
  `flag_anchor_stuffing` reroll any line over ~22-28 words SHORTER. The all-local
  420w batch HID this (terse local lines rarely trip the gates); the frontier legs
  exposed it.
- **Length probes (operator-directed): `target_words` is a near-no-op, and G1 is
  why.** grok @ 720w = 213 voiced words; @ 1340w (max-fit) = **263** (budget
  allocated up to 80 words/beat, but every line came out 8-28 words); @ 2000w =
  ERROR (`InvalidEpisodeBudgetError`, over the ~1363-word ceiling at 3 acts; the
  budget cap is `BEAT_WORD_HARD_MAX=80` x 14 beats). So the 14-beat skeleton x the
  ~22-28-word one_breath line-cap hard-bounds an episode at ~210-310 voiced words
  regardless of target. Length is a SIDE EFFECT of G1 -- fix the gate, lines fill
  toward the budget. (Operator: this IS worth chasing; see section 0 + G1.)

## 2. What the 2026-06-27 gate-seam cluster FIXED (grounded -- do NOT regress)

Aggregate over the soak ledgers (these were the dominant 2026-06-27 defects):

| counter | total | reading |
|---|---|---|
| anchor_stuffing_total | 1 | was the dominant frontier failure -> ~0 |
| one_breath_violation_total | 3 | low |
| personal_cost_boilerplate_total | 0 | 3.6 holding |
| ownable_people_object_total | 0 | 3.1 dignity holding |
| ownership_template_on_nonownable_total | 0 | 3.1 holding |
| dramatic_state_fallback_total | 0 | every episode's LLM dramatic_state succeeded |
| news_coda_truncated_total | 0 | 3.5 holding |
| news_coda_mojibake_total | 0 | 3.5 holding |
| speech_signature_near_duplicate_total | 0 | 3.7 string-collision fix holding |

The gate-seam cluster worked. The remaining defects are DIFFERENT and are the
target of this plan.

## 3. The NEW dominant story-quality defects (grounded, file-cited)

### G1 (LEAD) -- The line-compression gates over-correct: cap length + degrade craft
- **Defect:** `_otr_line_hygiene.flag_one_breath(max_words=28, soft
  _ONE_BREATH_SOFT_WORDS=22 + >3 clause markers)` + `flag_anchor_stuffing` reroll
  any character line over ~22-28 words into a shorter, denser one. Built 2026-06-27
  to stop FRONTIER overstuffing; they now over-fire and (a) cap every episode at
  ~210-310 voiced words regardless of `target_words` (the `_otr_episode_budget`
  allocator gives up to 80 words/beat), and (b) compress developed lines into
  noun-salad / ungrammatical fragments.
- **Evidence:** grok-720 (213w) + grok-1340 (263w at a 1340-word budget) -- nearly
  every line stamped `one_breath_retry, anchor_stuffing_retry, body_gate_reroll`,
  all 8-28 words; grok b03 "Storm the helm Quasimodo protects his paradox self to
  bury the merger disclosure", b10 "Restore the quantum beacon cells or the loan
  covenant buries your pension with those papers". The reroll hint ("use ONE
  detail, one breath") forces COMPRESSION, not SIMPLIFICATION.
- **Fix lane (CPU/content; reuse the existing gate seam; v2-gated so flag-OFF is
  byte-identical):** (a) raise the one_breath/anchor thresholds toward the per-beat
  budget -- scale `max_words` with the budget's `words_per_beat_range` eff_hi (a
  CLEAN line may run ~50-60 words; KEEP the >3-clause run-on guard); (b) change the
  reroll hint to favor SPLIT-INTO-TWO-SENTENCES / simplify, never "cram into fewer
  words"; (c) re-verify after the reroll and keep the cleaner + grammatical draft,
  not merely the shorter one (extend the 3.4 keep-better logic to the
  one_breath/anchor rerolls); (d) do NOT touch `BEAT_WORD_HARD_MAX` / the Stage-3
  Beat schema this pass -- the ~1363-word structural ceiling stays (>1363w still
  errors, acceptable).
- **Measure:** `length_ratio` (expect rise from ~0.5 toward ~0.7+ WITHOUT padding),
  `one_breath_violation_lines` / `anchor_stuffing_lines` (gate still catches true
  run-ons), a new `gate_reroll_degraded_grammar` counter, and the per-line
  word-count median (should rise from ~15 toward the budget).
- **Risk:** reintroducing the overstuffing the gates were built to stop.
  Mitigation: keep the run-on/clause guard; the keep-better-draft re-verify; TUNE
  thresholds, do not remove gates. (Subsumes the earlier "S0" enrichment finding.)

### S1 -- Seed abandonment: the drama is not ABOUT its premise (HIGH ceiling)
- **Evidence:** `dance_of_keys` (mistral, balanced): `dramatic_question` = "Will
  the pursuit of American dominance in quantum innovations compromise its moral
  integrity?"; the entire script is a generic "desert satellite-decay, pull the
  breakers" emergency -- quantum / NSF / morality appear NOWHERE in the dialogue,
  only in the coda. (Same disease as 2026-06-27 `dialing_shadows`: "seed only in
  the coda.")
- **Root (grounded):** the per-line composer/brief never binds dialogue to the
  `dramatic_question` + the two wants; weak local models default to generic genre
  action. `inject_central_object_into_brief` (`_otr_specificity.py` L199) returns
  the brief UNCHANGED by design, so the central object never shapes the body
  either.
- **Direction:** thread `dramatic_question` + `character_a_wants`/`b_wants` into
  the per-line composer as a persistent constraint ("this exchange advances THIS
  question"); add a deterministic off-premise-drift detector (does the line
  reference ANY seed key_term / central anchor?); on a generic-action scene with
  zero seed anchors, reroll once with a seed-anchored hint.
- **Measure:** new `seed_anchor_absent_lines` + `off_premise_episode` counters.

### S2 -- Coda bridge collapse: the drama->news payoff is broken in the majority (HIGH)
- **Evidence:** 18/29 `news_coda_fallback` + 9 `news_coda_generic_bridge`.
  `dance_of_keys`: "The real story: In a move to secure U.S. scientific
  leadership..." ; `ledger_ink_runs_dry`: "From tonight's headlines: UCLA
  Health..." -- both stamped `news_coda_fallback, news_coda_bridge_invalid`.
- **Root (grounded):** `compose_news_coda` (`_otr_line_composer.py` L3278) gives
  the local LLM TWO attempts to write a bridge clause; `validate_news_coda_bridge`
  (L3239) rejects a generic-opener / over-`_BRIDGE_MAX_CHARS` / bracketed bridge;
  two failures -> the deterministic `NEWS_CODA_POOL` canned-prefix fallback
  ("The real story:" / "From tonight's headlines:"). Weak local models fail the
  validator twice ~62% of the time.
- **Direction:** (a) strengthen the bridge prompt for weak models (1-2 in-context
  premise->bridge examples in `_NEWS_CODA_SYSTEM`); (b) raise attempts 2 -> 3;
  (c) replace the generic `NEWS_CODA_POOL` prefixes with PREMISE-DERIVED bridges
  (deterministic from the intro/premise nouns) so even the fallback bridges FROM
  the tale instead of "The real story:". (No change to the news FACT -- that stays
  appended deterministically.)
- **Measure:** `news_coda_fallback_count` + `news_coda_generic_bridge_count` down.

### S3 -- gemma body-gate-reroll degradation: run-ons + ALL-CAPS roster-name leak (HIGH on worst episodes)
- **Evidence:** `ledger_ink_runs_dry` (gemma): 8/14 character lines stamped
  `body_gate_reroll`; several broken:
  - [04] "...what they agreed to when CLARISSE GORDON claim this whole arrangement
    was a miracle" -- speaker's OWN name shouted in caps + broken grammar.
  - [13] "...VICTOR STENDAHL; the turning of the page is what matters now." --
    addressee full-name in caps mid-line.
  - [03] / [12] run-ons with no internal punctuation.
- **Root (grounded):** the `body_gate_reroll` path (`OTR_LedgerScriptWriter.py`
  ~L4537) on the weak gemma model emits run-ons + ALL-CAPS roster full-names; the
  leak-floor "roster vocative" rule catches a TRAILING vocative, not an ALL-CAPS
  full roster name embedded MID-clause; the reroll output does not re-pass the
  leak-floor / a run-on check.
- **Direction:** (a) extend the roster-vocative leak rule to detect+repair an
  embedded ALL-CAPS roster full-name ANYWHERE in the line; (b) re-run the
  leak-floor + a terminal-punctuation/run-on check on `body_gate_reroll` output;
  (c) keep the cleaner of the two drafts (gemma reroll frequently degrades).
- **Measure:** new `roster_name_caps_leak_lines` + `run_on_lines` counters.

### S4 -- Cliche floor not landing on local reproductions (MED-HIGH, low effort)
- **Evidence:** `dance_of_keys` shipped "Over my dead body, Lemmy" [04] + "Not on
  my watch, Pim" [11] -- BOTH are in the 3.4 `_CLICHE_RES` expansion, yet shipped.
- **Root (grounded):** the 3.4 mechanism rerolls once and keeps the fewer-defect
  draft (original on tie); a weak local model reproduces the cliche in BOTH drafts
  -> it ships. Detection works; replacement does not.
- **Direction:** on a cliche hit, give the reroll a TARGETED hint ("avoid the
  exact phrase 'X'; say it plainly") + a 2nd attempt; if still cliche, a
  deterministic de-cliche rewrite that drops the phrase. Strong lines pass on the
  first pass.
- **Measure:** `cliche_shipped_after_reroll` -> 0 (counter already specced in 3.4).

### S5 -- Interchangeable voices (MED)
- **Evidence:** `dance_of_keys` is pure threat<->counter-threat; both principals
  read identically. 3.7 fixed the near-dup `speech_signature` STRING collision,
  not the generated text; the line-level register reroll was deferred.
- **Direction:** thread each speaker's `speech_signature` into the per-line prompt
  as a persistent style directive + a deterministic register-divergence reroll
  (the deferred 3.7 half), reusing the existing loop.
- **Measure:** `register_overlap_ratio` down toward the frontier level.

### S6 -- Phantom invented entities (LOW)
- **Evidence:** `dance_of_keys` "SAT" / "Q-SAT" stamped `phantom_name` but shipped.
- **Direction:** reroll on `phantom_name` when the entity is a bare invented
  acronym absent from cast/seed. Low priority.

## 4. Ranking + proposed build order

| # | defect | impact | effort | primary touch |
|---|---|---|---|---|
| G1 | line-compression gates over-correct (length + craft) | **Very high** (every episode, both axes) | Med | `_otr_line_hygiene.py`, `_otr_line_composer.py` |
| S2 | coda bridge collapse | High | Low-Med | `_otr_line_composer.py` |
| S3 | gemma reroll degradation + roster-caps | High (worst eps) | Low-Med | `_otr_line_hygiene.py`, `OTR_LedgerScriptWriter.py` |
| S4 | cliche replacement | Med-High | Low | `_otr_line_hygiene.py`, `_otr_line_composer.py` |
| S1 | seed fidelity (weak-local; intermittent) | High ceiling | Med-High | composer brief seam, `_otr_specificity.py` |
| S5 | voice divergence | Med | Med | `_otr_casting.py`, `_otr_line_composer.py` |
| S6 | phantom entities (+ false positives: Atlantic, SAT) | Low | Low | `_otr_line_hygiene.py` |

**Proposed order:** **G1 first** -- it dominates (it sets length AND craft on every
episode, and S3/S4/S5 all touch the same reroll seam, so G1-first de-risks them)
-> S2 (coda floor) -> S3 (gemma roster-caps + reroll keep-better) -> S4 (cliche)
-> S5 (voices) -> S1 (seed -- highest ceiling but riskiest) -> S6. The kibitz will
pressure-test this ordering.

## 5. Measurement + discipline

- Extend `scripts/story_quality_scan.py` with the new counters; re-scan the soak
  + the 2 enrichment episodes pre/post each chunk; targets per defect above.
- Every change flag-gated; flag-OFF byte-identical. Full suite + Bug Bible + B7
  sweep BEFORE each commit. Commit+push per green chunk to `v2.0-alpha` ONLY.
  UTF-8 no BOM. prod/main + tags GATED.

## 6. Open questions for the operator

1. **S1 aggressiveness:** soft per-line seed hint, or a hard off-premise reroll
   gate? (Risk: over-constraining the weak local model into stiffness.)
2. **S2 fallback bridges:** OK to replace the generic `NEWS_CODA_POOL` prefixes
   with premise-derived bridges? (Changes only the bridge text, never the news
   fact.)
3. **S4 cliche governance:** keep extending `_CLICHE_RES` in code, or move the
   phrase list to a data file editable without a code change?
4. **S5 register push:** how hard to push voice divergence before it reads robotic?
5. **Frontier north-star:** use the grok-720 enrichment episode to set the "good"
   bar that S1/S5 aim the local writers toward?

## 7. Appendix -- ComfyUI domain invariants (scope-weighted)

This plan is **pure-python CPU story-content**: no node INPUT_TYPES/RETURN_TYPES
change, no tensor/VRAM work, no `workflows/otr_scifi_16gb_full.json` edit. So of
the ComfyUI custom-node invariants, the in-scope ones are:
- **Import isolation (#5):** the story modules (`_otr_line_composer`,
  `_otr_line_hygiene`, `_otr_specificity`, `_otr_casting`, `_otr_dramatic_state*`,
  `OTR_LedgerScriptWriter`) import at ComfyUI boot -- new helpers must stay
  pure-python, no heavy/top-level imports, no import-time side effects.
- **Determinism:** every reroll/hint must be seed-keyed and replay-stable
  (cast/style RNG draws OS entropy per episode; the gates must not introduce
  non-determinism). Flag any proposed change that is non-deterministic.
- **No widget drift (#1):** no INPUT_TYPES widget is added/reordered (would shift
  saved `widgets_values` in the canonical JSON, BUG-LOCAL-097) -- this plan adds
  none; flag any item that implies one.
Out-of-scope here (no need to weight): tensor layout (#2), VRAM/model-management
(#3), IS_CHANGED caching (#4) -- this lane touches none of them.

Reviewers: cite the real node file/class for every claim; the soak ledgers are at
`output/otr/episodes/<ep>/audio/*_ledger.json` and the census at
`docs/2026-06-28-story-quality-kibitz/scan_soak.json`. If you cannot see code,
write "verify: <what>".
