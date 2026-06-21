# Signal Lost -- Story-Engine Coding Sprint Plan (v2, post roundtable 2)

Executable sprint plan for the 10 grounded fixes in `STORY_ENGINE_IMPROVEMENT_PLAN.md`, hardened by roundtable 2 (see `roundtable/pass02_judgment.md`). F9 and F10 are DEFERRED out of this plan (panel-unanimous). Line numbers are approximate (commit `f99af26`) -- **grep the function name before every edit; do not trust the line number** (the code may have moved).

## Invariants (hold on every commit)
- **Ledger intact (C2):** `l3-2026-05-14` schema unchanged; only additive `meta.*` keys; `lines[]` ordering semantics preserved; `test_audio_byte_identical` holds where text is unchanged.
- **No QA-only round (C3):** no scoring/reject/reroll PASS. A deterministic detector that triggers the EXISTING single-line recompose seam is a hygiene repair, not a QA gate -- allowed.
- **No architecture change (C4):** all story logic stays inside node 1 `OTR_LedgerScriptWriter` and the internal modules it calls (`_otr_line_composer`, `_otr_line_hygiene`, `_otr_story_spine`, `_otr_dramatic_state*`, `_otr_casting`, `_otr_style_picker`). These are NOT separate ComfyUI nodes. No new node, no DB/RAG, no training.
- **Build discipline:** EACH `.py`-touching task is its own green chunk -- run the suite (`$env:PYTHONUTF8=1`; venv python; `pytest -q -p no:cacheprovider`) + the Bug Bible regression, then commit AND push to `v2.0-alpha`, then verify HEAD==origin / no 0-byte / no BOM / AST-parse. No batching multiple edits before a commit.
- **Wiring:** v1 needs ZERO `otr_scifi_16gb_full.json` edits (see WIRING_PLAN.md); F8 stores `arc_shape` in additive `meta`, exposing no widget.

## Measurement contract (Sprint 0 builds this; every later sprint uses it)
- **Fixed smoke:** exactly **12 legs** at `target_words=864`, with a FIXED input/seed list pinned via `OTR_CAST_SEED` + `OTR_STYLE_SEED` (the existing C7 reproducibility env vars) so before/after are apples-to-apples despite the OS-entropy RNGs. The 12 seeds are recorded in `SPRINT_BASELINE.md` and reused unchanged for every sprint.
- **`scripts/story_quality_scan.py`** (committed tooling) reports, per leg and aggregate:
  - `length_ratio` = CHARACTER word total / target (EXCLUDES announcer + music lines).
  - `length_pass_fired` (bool) -- did the post-script length normalizer activate.
  - `episode_valid` (from `slot_drama_contracts_audit`).
  - `outro_hedge_vs_resolved` -- outro contains a phrase from the fixed HEDGE_LIST `["remains to be seen","only tomorrow will tell","open question","remains unknown","time will tell","yet to be seen"]` AND `dramatic_state.ending_change` is in the RESOLVED set (categorized: a success/closure ending vs an open/ambiguous one -- the scan classifies via a small keyword rule documented in the script).
  - `narration_self_address_lines` -- count of lines flagged by the F7 detector.
- **Rounding:** percentages computed over the 12-leg set; ">=90%" means `>= 0.90 * 12` i.e. >=11/12; ">=85% ratio" means the aggregate mean rounds to >=0.85.

## Acceptance targets (on the 12-leg fixed smoke)
| Metric | Today | Target |
|---|---|---|
| `length_ratio` mean | 0.70 | **>= 0.85** AND `length_pass_fired` not the cause (<=2/12 firings) |
| `episode_valid` | 24% | **>= 90%** (>=11/12) |
| `outro_hedge_vs_resolved` | several | **0/12** |
| `narration_self_address_lines` | >0 | **0** across the smoke |
| full suite / Bug Bible | green | **stays green**; `test_audio_byte_identical` holds |

---

## Sprint 0 -- measurement harness (committed tooling, no engine edits)
- **T0.1** Build/confirm `scripts/story_quality_scan.py` with every field above. Commit as tooling (own green chunk).
- **T0.2** Reset-before-headless (selective CIM kill; port 8000/8011 clear; VRAM at ~1.5GB baseline). Run the FIXED 12-leg smoke on CURRENT code -> write before-numbers + the exact seed list + VRAM/port state to `SPRINT_BASELINE.md`.
- **T0.3** Commit `SPRINT_BASELINE.md`.
**Exit:** baseline recorded; harness reproducible.

## Sprint 1 -- ship-first (F1, F2, F3, F6) -- 4 commit chunks
- **T1.1 [PROMPT] F1 length tail tracks the per-beat band.** In `_otr_line_composer._build_user_prompt` (near the always-emitted tail): replace literal "about 20-30 words" with `about {beat_lo}-{beat_hi} words` derived from the beat's allocated band, OR drop the number. Rename the cap input to `beat_target_words`; guard None with a numeric fallback; raise/confirm `max_new_tokens = min(200, max(40, beat_target_words*4))` and `_MAX_NEW_TOKENS_PER_LINE`. Update `_build_user_prompt` Test 3 to assert the NEW instruction (band or no-number), not the stale literal. Add tests: None target, 864 episode case, no mid-sentence truncation.
  - *Accept:* `length_ratio` >= 0.85; <=2/12 `length_pass_fired`.
- **T1.2 [LOGIC] F2 bind costly-choice (single source of truth).** In `pick_costly_choice_slot` (`_otr_dramatic_state.py`) + the dramatic_state wire (`OTR_LedgerScriptWriter.py ~:2785`): build the costly-slot candidate list from CHARACTER-only voiced beats (exclude announcer/music) AND ensure the picked slot receives a `must_turn` contract from the SAME list the audit checks (`_otr_slot_drama_contract.validate_episode_contracts`). Empty-character case: force the contract onto the chosen last character slot; never leave the pointer on `d001`=announcer. Keep `^d\d{3}$` shape.
  - *Accept:* `episode_valid` >= 11/12; unit test asserts `picked_slot_id in must_turn_contract_slot_ids` and the zero-eligible-beat path.
- **T1.3 [PROMPT] F3 ending-aware outro (prompt-first + bounded recompose).** In `compose_announcer_outro` (`_otr_line_composer.py`): thread `meta.dramatic_state.ending_change` (always present) into the user prompt; add the final character line ONLY with a null-guard after verifying it exists at outro-compose time (else use the final-beat summary). System rule: "if the question resolved, state the outcome; do not hedge." Add a deterministic post-check: if the outro contains a HEDGE_LIST phrase while `ending_change` is RESOLVED, recompose the outro once (logged), else keep.
  - *Accept:* `outro_hedge_vs_resolved` = 0/12; outro still passes its char-band validation.
- **T1.4 [PROMPT] F6 unconditional anti-decorative rider.** Ungate the "perform indirectly / the situation must be different after this line" rider in `_build_user_prompt` so it lands on every CHARACTER beat (not intro/outro/music paths). (Shares the file region with F1 -- land after T1.1.)
  - *Accept:* rider present in every character-line prompt (unit test); no `length_ratio` regression vs T1.1.
**Exit:** ratio >= 0.85, valid >= 11/12, 0 contradicted closes, suite+Bug Bible green. **Ship-first milestone -- stop here if only Tier 1 is wanted.**

## Sprint 2 -- craft + integrity (F4, F5, F7) -- 3 commit chunks
- **T2.1 [PROMPT] F4 gender/pronouns reach the line composer.** Prefer the smaller change: pass normalized `{gender}`/pronouns into the compose_line CHARACTER context if available; else require gender+consistent pronouns in the `_otr_casting` CHARACTER VISUAL CONTRACT. Tests for male/female/unknown pronoun consistency (name-independent).
  - *Accept:* no "Mister <female>"-class mismatch in a targeted re-render; pronoun-consistency unit test.
- **T2.2 [PROMPT] F5 speech-register as an artifact.** Add a `speech_signature` (<=5 words) to each character card; thread it into the composer prompt (via existing `all_voice_cards`). 
  - *Accept:* every card has a nonempty `speech_signature` AND the composer prompt includes it (unit test). Subjective distinctness = nice-to-have.
- **T2.3 [LOGIC] F7 narration/self-address hygiene (prompt-first + bounded recompose).** Add the negative line "Never narrate in third person; never speak your own name" to compose_line OUTPUT FORMAT. Add a deterministic detector to the EXISTING `_otr_line_hygiene.py` (line opens with He/She/They/<speaker-name> + narration verb, or is a 3rd-person summary -- FIRST-PERSON self-narration is excluded, that's craft not breakage). On a hit, route to the existing `_hy_recompose` single-line seam (NOT strip-in-place), logged, fallback to original. Move the shared narration regexes into one module to avoid an import cycle.
  - *Accept:* `narration_self_address_lines` = 0 on the smoke; tests for empty-result, speaker-name-substring, and a legitimate 3rd-person reference (must NOT false-positive). C3: this is a recompose seam, not a reject gate.
**Exit:** gender bug gone, cards carry register, hygiene clean, no Tier-1 regression.

## Sprint 3 -- arc variety (F8 only) -- 1-2 commit chunks
- **T3.1 [PROMPT/LOGIC] F8 arc-shape variety via additive `meta`.** A small SEEDED pre-step picks `arc_shape` from a set (`setup_complication_resolution | investigation_without_answer | slow_dread | heist | betrayal`); pass it as CONTEXT into the macro + dramatic_state prompts; record `meta.arc_shape` (additive -- macro JSON schema `{title,premise,setting,time_of_day,central_tension}` UNCHANGED, C2). Add matching structural templates in `_otr_dramatic_state_llm._TEMPLATES`; keep the post-validator's key-term + opposed-wants checks.
  - *Accept:* `arc_shape` distribution across the 12-leg smoke is not single-valued; macro post-validator still passes; suite green.
**Exit:** variety up; all hard acceptance targets still met.

---

## Deferred (documented, NOT built this pass)
- **F9 -- condition outline on dramatic_state (reorder).** Cut by all 4 panelists: large structural change, C4-spirit/regression risk; F2 already fixes binding. Revisit only if F8's variety proves insufficient, with a `lines[]`-order + `test_audio_byte_identical` gate on the reordered path.
- **F10 -- persistent anti-repeat list.** Deferred: statefulness/determinism/CI-contamination risk for the softest target. The real lever is RSS-source dedup (code out of scope). Revisit with explicit test-isolation + a deterministic opt-out flag.

## Sequencing
F1 -> F6 (shared file region); F4+F5 together; F2 and F3 independent; Sprint 1 is the gate for Sprint 3. Each `.py` task = its own push to `v2.0-alpha`.

## Suggested first commit
Sprint 0, then T1.1 + T1.2 + T1.3 as three consecutive green chunks (length, the 76% audit, the contradicting outro) -- all single-file/near-single-file, ledger-safe, zero workflow-JSON change.
