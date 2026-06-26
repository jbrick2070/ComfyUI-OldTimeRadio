<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The stated goal is “NO DRIFT” whole-story integrity, but the current arc still treats failed/absent story accuracy as `strong`, ships known quality failures, and has no defined deterministic cross-stage assertion map.

MUST-FIX BEFORE BUILD:

1. [Grounded surface 1 / Question A / `_otr_story_critic.py: StoryCriticReport.clean` / `run_story_critic`] Fail-open is still architecturally central: any critic failure returns an all-empty report with `arc_verdict="strong"`, and the caller cannot distinguish “verified strong” from “critic did not run.” This directly contradicts the round’s ask that “a failed or absent critic must not read as strong.” Concrete fix: add a caller-stamped status outside the frozen report schema, e.g. `meta.story_critic_status = {"ran": bool, "validated": bool, "failure": "...", "positive_evidence": {...}}`, and stop using bare `arc_verdict == "strong"` as proof. If schema changes are allowed, migrate `StoryCriticReport` to include `critic_status`; otherwise keep the report shape and stamp status in `meta`.

2. [Question A / `_otr_freeze_cascade.py: Sprint 5C targeted reroll loop` / A2 repair-then-ship block] The cascade story is contradictory: the escalation branch labels `EscalationScope.EPISODE` as “Structural failure -- the arc is broken,” creates a `needs_full_rerun` reroll disposition, then the A2 block explicitly “SHIPPING the best candidate” and falls through to freeze. That makes a whole-episode story failure neither a blocker nor a clear warning class. Concrete fix: split terminology and verdicts: deterministic unrenderable defects remain `structural` and block at Phase 10; story-quality failures become `quality_warn_through` / `quality_needs_regeneration` and must be surfaced as non-clean freeze metadata. Do not call an arc/critic failure “structural” if the policy is to ship it.

3. [Question B / Grounded surface 2] The plan asks for “minimal deterministic assertions” but never defines the source-of-truth matrix. Without that, the next `sound_palette`-class drop is not preventable. Concrete fix: create a small explicit contract table before build: `field`, `source`, `derived ledger/canon path`, `normalizer`, `required/optional`, `assertion timing`. At minimum include `sound_palette <- StoryContract.sound_world`, title, premise, setting, time_of_day, style, cast ids/names/roles, protagonist/antagonist if present, outline beat ids, line `beat_id`, and `meta.line_dramatic_frame` coverage. Implement as offline CI tests plus a pre-freeze deterministic assertion.

4. [Question A / `_otr_story_critic.py: _critic_character_lines` / `_render_critic_user_prompt`] The “whole-story” critic is not actually whole-story: it filters to `speaker_role == "character"` and explicitly excludes announcer/music/SFX beats. Continuity and canon drift can live in locked structural content: location SFX, announcer framing, title/setup narration, music cues, or non-character exposition. Concrete fix: give the critic read-only context for all story-bearing lines while restricting actionable `reroll_targets` to character lines. Deterministic validators should still reject reroll targets for non-rerollable lines.

5. [Question A / `_otr_story_critic.py: _make_critic_post_validator`] The only post-validator checks unknown `line_id`s. It does not require positive evidence that the critic completed the requested whole-story pass. Example: `render_priority` is prompted to include every shown line, but an empty or partial list validates and can be mistaken for benign fallback. Concrete fix: on full-episode critic runs, require `render_priority` to be an exact permutation of the shown critic-scope line ids, or stamp `critic_status.validated = false`. For scoped reroll runs, require exact permutation of the scoped ids if `render_priority` is consumed; otherwise ignore it explicitly.

6. [Grounded surface 3 / `_otr_freeze_cascade.py: successful freeze verdict mapping`] `frozen_with_warns` and `frozen_with_doctor_edits` are allowed to ship, but the plan does not classify which warning classes are acceptable quality noise versus accuracy defects. Concrete fix: define a warning taxonomy before build: `structural_error` blocks, `story_accuracy_warning` ships only as non-clean with operator-visible metadata, `cosmetic_warning` may ship clean-with-warns. Wire Phase 10/gap audit warnings and critic findings into that taxonomy instead of relying on counts.

7. [Question D / Grounded surface 4] Widget positional drift is identified as a known workflow-ledger drift vector, but the plan leaves open whether the validator runs in CI. That is not build-ready. Concrete fix: add a mandatory offline CI test that loads the saved workflow JSON, extracts `widgets_values` order for the relevant node, compares it to live `INPUT_TYPES`, and fails on any non-append-compatible mismatch. Verify: exact validator name and workflow path, because no validator code is shown here.

8. [Question D / Grounded surface 5 / Invariants] Schema-version evolution is acknowledged but not designed. “Old ledgers silently read as missing -> default -> wrong” is exactly the drift class the round is about. Concrete fix: define a migration/compat policy for `l3-2026-05-14`: old ledger fixture tests, explicit default provenance, and a fail/repair path for fields whose default changes semantics. Do not add optional fields that affect behavior unless a migration or deterministic derivation exists.

SHOULD-FIX:

1. [Question C / Already SETTLED / Invariants] The operator asks for “multiple LLMs on the BINARY decisions,” but the invariants correctly require deterministic/offline CI guards for accuracy guards. The plan needs to reject multi-LLM voting for hard binary gates, not leave it as an open possibility. Concrete fix: state that LLMs may produce advisory findings or repair hints, but binary pass/fail gates must be deterministic or must degrade to explicit “unchecked,” never “pass.”

2. [Grounded surface 1 / `_otr_story_critic.py: ArcVerdict`] `strong` is overloaded as both a real verdict and the safe fallback default. Even with external status metadata, this is a semantic trap. Concrete fix: treat `strong` as valid only when accompanied by `story_critic_status.ran == true`, `validated == true`, and `positive_evidence.render_priority_complete == true`; otherwise downstream readers must display `unchecked`.

3. [Grounded surface 1 / `_otr_story_critic.py: StanceIssue`] Stance consistency is Section 7 in the prompt but the surrounding docstrings and user prompt repeatedly say “6-section rubric.” It is also telemetry-only and “does NOT change any verdict.” That makes the story arc incoherent: the plan claims whole-story accuracy while explicitly excluding an identified coherence failure from repair/gating. Concrete fix: either cut stance from the integrity path, or classify stance reversals as `story_accuracy_warning` with operator-visible status and optional regeneration hint.

4. [Question B / Grounded surface 2] The plan focuses on canon/contract drift but does not explicitly cover outline-to-line intent drift. The critic sees `beat_intent`, but there is no deterministic assertion that each outline beat exists, is represented once or intentionally split, and maps to valid line ids. Concrete fix: add an offline outline coverage guard: every required outline beat has at least one ledger line; every line `beat_id` exists upstream or is marked generated; arc phases/tension frames cover character lines.

5. [Question B / Grounded surface 2] Cast consistency is mentioned, but the current cited fix in `_otr_freeze_cascade.py` is a late coercion sweep of `speaker_role` based on `char_id`. That prevents one symptom but does not prove ledger cast matches CastLock. Concrete fix: add a deterministic CastLock equivalence check: cast ids, display names, roles, voice ids/style-critical fields if present, and line `char_id` membership. [ASSUMPTION] Exact CastLock shape is not shown.

6. [Question D / `_otr_freeze_cascade.py: _persist_cascade_meta`] Persistence is treated as best-effort and never raises. That is compatible with audio-first, but not with forensic ledger integrity if disk readers are part of the contract. Concrete fix: stamp `meta.persistence_status` on save failure and make CI/test mode fail on persistence failure, while production remains non-crashing.

7. [Question E] The plan needs a priority order. Right now it lists critic fail-open, contract drift, freeze warns, widget order, and schema migration as peers. Concrete fix: sequence build as: 1) critic status/no false strong, 2) deterministic source-of-truth assertion table, 3) CI drift guards for widgets/schema, 4) warning taxonomy, 5) optional quality improvements.

OPTIONAL / NICE-TO-HAVE:

- [Question A] Add a small deterministic “critic input completeness” report: number of total lines, character lines, non-character story lines, missing beat intents, missing tension frames.
- [Question B] Keep golden fixtures for the `sound_palette` regression and at least one old schema ledger.
- [Question C] Add a dashboard label that distinguishes `frozen_clean`, `frozen_with_quality_warns`, and `frozen_with_structural_warns` instead of forcing users to inspect nested meta.

CUT THESE (scope / over-engineering):

1. [Question C] Cut multi-LLM voting for binary gates at this stage. It conflicts with the invariant that guards be deterministic/offline/CI-runnable, adds cost and model-dependence, and still cannot prove correctness.

2. [Grounded surface 1 / `_otr_story_critic.py: Section 7 stance`] Cut or demote stance telemetry unless it becomes actionable. A telemetry-only coherence detector that “does NOT change any verdict” adds apparent coverage without closing drift.

3. [Grounded surface 1 / `_otr_story_critic.py: render_priority`] Cut render-budget prioritization from the integrity-critical critic path if time is limited. It is not a ledger-faithfulness guard; it can remain an advisory render feature after fail-open and cross-stage assertions are fixed.

4. [Question B] Cut broad “assert everything in canon matches everything upstream” ambitions. Safe minimal cut: only assert fields that are deterministic derivations or identity-preserving copies. Do not build semantic LLM canon comparison as a guard.

5. [Question D] Cut ad-hoc/manual validators for widget order and schema compatibility. If it is not CI-runnable and offline, it does not satisfy the stated invariant and will drift again.