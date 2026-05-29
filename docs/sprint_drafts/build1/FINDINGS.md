# Build 1 — Findings (measurement integrity)

## Task 1 — Critic key verdict
**Same scoring basis, different ledger key.** There is exactly one whole-episode critic; it is stamped only under `meta.stage7_shadow_critic`. The key `meta.whole_episode_critic` is **never written** anywhere in the codebase (repo-wide grep for the assignment returns nothing). It was only ever a handoff/doc label.

Evidence:
- `nodes/_otr_freeze_cascade.py:815-820` — sole call: `critic_result = run_whole_episode_critic(...)` then `meta["stage7_shadow_critic"] = critic_result.to_dict()`.
- Rubric: `_otr_critic_rubric.py:54` (10 axes, ordinal 1..10), each axis 1-5; `compute_verdict_locally` sets `mean = sum(scores)/len(scores)` over all 10 (`_otr_whole_episode_critic.py:339`); ship threshold mean >= 3.5 (`_otr_critic_rubric.py:72`).
- Confusion source: `CriticResult.to_dict()` docstring says "Serialize for meta.whole_episode_critic ledger stamp." (`_otr_whole_episode_critic.py:187`) but the live call site writes `stage7_shadow_critic`.
- 3.70 baseline came from this exact pass: `_otr_freeze_cascade.py:863-865` logs "Stage 7 whole-episode critic returned verdict=ship mean=3.70"; escalation reads `meta.get("stage7_shadow_critic")` (`:887`).

**Canonical key for the 3.70 comparison: `meta.stage7_shadow_critic.mean_score`.** The 3.7 this run scored IS apples-to-apples with the 3.70 baseline.

## Task 2 — Stage-1 shadow bug verdict
**Inert / diagnostic-only — NOT in the active render path. Not plateau-relevant.** The shadow pass is measurement-only; the legacy outline+dialogue path is the sole producer of the rendered ledger. The discarded plan never reaches audio, dialogue, the committed ledger, or the critic.

Evidence:
- `OTR_LedgerScriptWriter.py:2433-2445` — header: "MEASUREMENT-ONLY ... does NOT drive the rest of the pipeline ... the shadow pass is the only consumer of Stage 1."
- `_shadow_plan` read only at `:2627-2629` (INFO log) and `:2643` `audit_cast(_shadow_plan)` (pure validation -> `meta["stage1_cast_audit"]`). Never assigned into cast_rows, beats, continuity, or composer input.
- Discard path: `:2677-2698` stamps `stage1_shadow_plan_present=False`, logs "Existing pipeline continues unaffected."

Root cause: `tension: Optional[int] = Field(default=None, ge=1, le=5)` at `_otr_stage1_plan.py:260-265`. Constrained generator binds `Stage1Plan` (`OTR_LedgerScriptWriter.py:2465-2467`); model emits `tension=0` for low-stakes/opening beats -> pydantic ValidationError -> parse_failed (`_otr_stage1_call.py:330-351`) -> 2-attempt budget exhausted (`_STAGE1_MAX_ATTEMPTS=2`). Same shape as BUG-LOCAL-282 (which relaxed `length_target_words` ge=5->ge=0); `tension` was missed.

Worth fixing for measurement integrity (corrupts the Stage-1 first-attempt-valid soak gate, wastes ~2 LLM calls/run) but will NOT move the 3.70 mean.

## Open questions for integration
1. Critic key: rename ledger key `stage7_shadow_critic` -> `whole_episode_critic`, or fix the stale `to_dict` docstring instead? Pick one, pin it in the wiring test.
2. `tension=0`: blanket allow (recommended) or carve-out to non-voiced/bookend beats only (like the MUSIC carve-out for `length_target_words`)?
3. If any later build wires `tension` into the writer prompt, re-evaluate whether 0 should be a sentinel or clamped at consumption time.
