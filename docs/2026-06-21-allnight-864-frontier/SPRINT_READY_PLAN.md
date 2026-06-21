# Signal Lost -- Story-Engine Sprint-Ready Plan (FINAL, post 4 roundtables)

Converged across roundtables 1-4 (problem -> improvement -> sprint -> wiring -> bugs) + three grounded code audits. Bug-pass fixes are folded in. This is the document `/otr-handoff` picks up. Line numbers are approximate (commit `f99af26`) -- **grep the function before editing**. Judgments: `roundtable/pass0{1,2,3,4}_judgment.md`.

## Scope
IN: F1 length, F2 costly-choice binding, F3 ending-aware outro, F4 gender/pronouns, F5 speech register, F6 (split) rider, F7 narration hygiene, F8 arc-shape variety. DEFERRED (not built): F9 outline reorder, F10 persistent anti-repeat list.

## Invariants (every commit)
- **C2 ledger:** schema `l3-2026-05-14` unchanged; only additive `meta.*`/`cast[].*` keys; `lines[]` order preserved; `test_audio_byte_identical` holds where text unchanged.
- **C3 no QA gate:** no scoring/reject/reroll PASS. A deterministic detector triggering the EXISTING single-line recompose seam (max 1 attempt, distinct log marker, fallback to original/template) is a hygiene repair -- allowed.
- **C4 no arch change:** all logic inside node 1 `OTR_LedgerScriptWriter` + internal modules (`_otr_line_composer`, `_otr_line_hygiene`, `_otr_story_spine`, `_otr_dramatic_state*`, `_otr_casting`, `_otr_style_picker`). No new node/DB/training.
- **Freeze CRITICAL invariants (do not regress, `_otr_ledger_freeze.py`):** 7 top-level lists present+list-typed (`cast,lines,beats,scenes,shots,music,clips`); unique non-empty `line_id`; `speaker_role` in `{character,announcer,music_open,music_close,music_inter,sfx}`; voiced lines keep non-empty `char_id` + referenced cast keep `char_id`/`name`/`voice_preset`; skipped line has `text==""` + non-empty `tts_skip_reason`.
- **Build:** each `.py` task = own green chunk -- suite (`$env:PYTHONUTF8=1`; venv python; `pytest -q -p no:cacheprovider`) + Bug Bible (`cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`; venv python; relative `tests\bug_bible_regression.py`), then commit AND push to `v2.0-alpha`, verify HEAD==origin / no 0-byte / no BOM / AST-parse.
- **Wiring:** ZERO `otr_scifi_16gb_full.json` edits in v1 (verified against consumers; see WIRING_PLAN.md). v1 forbids any widget/INPUT_TYPES/link edit.

## Measurement contract (Sprint 0 builds `scripts/story_quality_scan.py`; reused every sprint)
- Fixed **12-leg** smoke at `target_words=864`. `SPRINT_BASELINE.md` records the EXACT 12 news inputs/invocations AND `OTR_CAST_SEED`+`OTR_STYLE_SEED` per leg (verify these env vars pin BOTH cast and style RNGs on the baseline run). Reuse unchanged every sprint.
- Scan reports per-leg + aggregate:
  - `length_ratio` = ALL VOICED words (character + announcer; EXCLUDE music) / `target_words`. (Matches the 0.70 baseline, which was total voiced / target.)
  - `length_pass_fired` -> reported as a COUNT of legs where the post-script length normalizer activated.
  - `episode_valid` = `freeze_valid AND dramatic_contract_valid` (freeze = `_otr_ledger_freeze` CRITICAL pass; dramatic = `_otr_slot_drama_contract.validate_episode_contracts`). Define this exact boolean in the scan so baseline and acceptance agree.
  - `outro_hedge_vs_resolved` = outro contains a HEDGE_LIST phrase AND `is_resolved_ending_change(ending_change)` is true.
  - `narration_self_address_lines` = count flagged by the F7 detector (post-recompose).
- HEDGE_LIST = `["remains to be seen","only tomorrow will tell","open question","remains unknown","time will tell","yet to be seen"]`.
- `is_resolved_ending_change()` is ONE shared helper used by both the scan and the F3 composer repair (no divergence).

## Acceptance targets (12-leg fixed smoke)
| Metric | Today | Target |
|---|---|---|
| `length_ratio` mean | 0.70 | >= 0.85, with `length_pass_fired` <= 2/12 |
| `episode_valid` | 24% | >= 11/12 |
| `outro_hedge_vs_resolved` | several | 0/12 (guaranteed by the deterministic fallback template) |
| `narration_self_address_lines` | >0 | 0/12 after the single recompose |
| full suite / Bug Bible / `test_audio_byte_identical` | green | stays green |

## Sprint 0 -- harness (committed tooling, no engine edits)
Build `scripts/story_quality_scan.py` (all metrics above, incl. the shared `is_resolved_ending_change`). Reset-before-headless (kill ONLY processes whose CommandLine matches the Comfy/headless server + soak harness via CIM, plus port 8000/8011 owners by PID; never blanket-kill). Run the fixed 12-leg smoke on CURRENT code; write before-numbers + the 12 exact inputs + seeds to `SPRINT_BASELINE.md`. Commit.

## Sprint 1 -- ship-first (order: T1.1 -> T1.4 -> T1.2 -> T1.3)
- **T1.1 [PROMPT] F1 length.** In `_otr_line_composer._build_user_prompt` tail: **drop the literal "about 20-30 words"** (default, no scope dependency); only interpolate `about {beat_lo}-{beat_hi} words` IF those values are confirmed in scope at build. Rename cap input `beat_target_words`; guard exactly: `max_new_tokens = 200 if beat_target_words is None else min(200, max(40, int(beat_target_words)*4))`. Update `_build_user_prompt` Test 3 to assert the NEW tail; add tests for None and the 864 case. (No-mid-sentence-truncation is checked via `length_pass_fired`/spot-read, not a hard unit gate.) *Accept:* `length_ratio`>=0.85, `length_pass_fired`<=2/12.
- **T1.4 [PROMPT] F6 (SPLIT).** Make "perform indirectly; do not summarize the objective or explain the turn" UNCONDITIONAL on every CHARACTER beat. Keep "the situation must be different after this line" GATED to turn/costly beats (do NOT apply it to every line -- over-acting risk). Lands right after T1.1 (same region). *Accept:* the indirect-performance clause present in every character-line prompt (test); the situation-change clause only on turn beats; no `length_ratio` regression.
- **T1.2 [LOGIC] F2 binding.** Build the costly-slot candidate list from CHARACTER-only voiced beats AND create its `must_turn` contract from the SAME list the audit checks (`_otr_dramatic_state.pick_costly_choice_slot` + wire `OTR_LedgerScriptWriter.py ~:2785` + `_otr_slot_drama_contract.validate_episode_contracts`). **Zero-eligible (all-announcer/empty-cast): do NOT emit the contract** (episode is invalid -- acceptable+rare), never index a non-existent slot, never put `must_turn` on announcer/music/sfx. Decide the id namespace once (slot id vs line_id) and convert in one place. *Accept:* `episode_valid`>=11/12; tests assert `picked_slot_id in must_turn_contract_slot_ids`, plus all-announcer and empty-cast paths.
- **T1.3 [PROMPT] F3 ending-aware outro.** In `compose_announcer_outro`: thread `meta.dramatic_state.ending_change` (null-guard: missing -> treat as unresolved, skip repair) + the final character line (null-guarded; only if available at outro-compose time -- verify order -- else use the final-beat summary). System rule: "if resolved, state the outcome; do not hedge." Post-check: if a HEDGE_LIST phrase appears while `is_resolved_ending_change()` is true, recompose once; if it STILL hedges, emit a deterministic fallback outro template that states `ending_change` with no HEDGE_LIST phrase. *Accept:* `outro_hedge_vs_resolved`=0/12; tests: first-compose-hedges-then-recompose-hedges -> fallback; missing `ending_change` -> no crash; final-line-unavailable -> summary path.
**Exit:** ratio>=0.85, valid>=11/12, 0 contradicted closes, suite+Bug Bible green. **Ship-first milestone.**

## Sprint 2 -- craft + integrity
- **T2.1 [PROMPT] F4 gender/pronouns.** Inject the speaker's gender/pronouns from the EXISTING `cast[].gender` field into the compose_line CHARACTER block at prompt assembly (do NOT change the casting contract -> no schema/C2 impact). Name-independent pronoun-consistency tests (male/female/other). *Accept:* no "Mister <female>"-class mismatch.
- **T2.2 [PROMPT] F5 speech register.** Add `speech_signature` (<=5 words) to the JSON the cast LLM already returns; deterministic backfill ("plain spoken") for any missing/legacy card; thread into the composer via `all_voice_cards`. *Accept:* every card has nonempty `speech_signature` (incl. backfill) AND the composer prompt includes it (tests).
- **T2.3 [LOGIC] F7 narration hygiene.** Add OUTPUT-FORMAT negative constraint ("never narrate in third person; never speak your own name"). Add a detector LOCAL to `_otr_line_hygiene.py` (duplicate a small NARRATION-VERB set with a test -- do NOT refactor the composer's regexes now): fire only on a line describing the SPEAKER's own physical action in third person (He/She/They/<speaker-name> + a narration/stage-direction verb like paces/stops/gazes/contemplates, or a 3rd-person beat-summary). EXCLUDE first-person and legitimate 3rd-person references to OTHERS ("He is lying", "They know the code"). On hit -> existing `_hy_recompose`, exactly 1 attempt, distinct log marker, fallback to original. *Accept:* `narration_self_address_lines`=0 post-recompose; mandatory tests BEFORE wiring: first-person allowed, legit-3rd-person-reference allowed, speaker-name-substring no-trigger, true self-narration triggers, empty-recompose -> original, no-multiple-retries.

## Sprint 3 -- arc variety
- **T3.1 [PROMPT/LOGIC] F8 arc-shape.** A SEEDED pre-step (tied to the same reproducibility seed; record `meta.arc_shape`) picks `arc_shape` from `{setup_complication_resolution, investigation_without_answer, slow_dread, heist, betrayal}` (start with 3 if pressured); pass as CONTEXT into macro + dramatic_state prompts; macro JSON schema UNCHANGED (strip any stray `arc_shape` the LLM emits). Add matching templates in `_otr_dramatic_state_llm._TEMPLATES`. **Branch the post-validator:** only confrontation shapes require opposed wants; non-confrontation shapes (investigation/slow_dread) use a shape-appropriate check so valid generations are not rejected (prevents a generation STALL). Keep beat/act counts as-is for v1. *Accept:* `arc_shape` distribution not single-valued across the smoke; no generation rejections; suite green.

## Deferred (documented)
- F9 outline reorder -- cut by all panelists; revisit with a `lines[]`-order + `test_audio_byte_identical` gate.
- F10 persistent anti-repeat list -- deferred (statefulness/CI risk); real lever is RSS-source dedup (out of scope); revisit with test-isolation + deterministic opt-out.

## First commit
Sprint 0, then T1.1 + T1.4 + T1.2 + T1.3 as consecutive green chunks. Each `.py` task = its own push to `v2.0-alpha`.

## Verify-at-build (gate Sprint 1 on these; record in SPRINT_BASELINE.md)
1. `OTR_CAST_SEED`/`OTR_STYLE_SEED` pin BOTH cast and style RNGs for a reproducible 12-leg smoke.
2. `beat_lo`/`beat_hi` availability in `_build_user_prompt` (if absent, F1 uses the drop-the-number default).
3. Final character line exists at `compose_announcer_outro` time (else F3 uses the final-beat summary).
4. The costly-choice id namespace (slot id vs `line_id`) that `validate_episode_contracts` checks.
