# OTR Better Story — Execution Plan v9 (build-ready, Claude's best judgment)

**Date:** 2026-05-28
**Status:** CANONICAL execution plan. Supersedes v7 (Claude draft) and v8 (synthesis + annotations).
**Thesis:** Don't ask the model for a better story. Make each ledger slot carry a dramatic obligation, then refuse to commit lines that don't discharge it.

This is the version we build from. Judgment calls are made and marked **[DECISION]**; I'm not re-opening them unless a run disproves them.

---

## The one big call

The plateau is fixed at the **commit gate**, not the critic. The chain becomes:

```
obligation (contract)  →  exchange-level writing  →  deterministic craft-floor  →  one targeted repair  →  commit
```

The critic/editor stop being responsible for quality. They become a postmortem.

---

## Build order — REORDERED from v8 [DECISION]

v8 put `compose_exchange` last. **I'm moving it ahead of the semantic validator.** Reason: the validator's semantic checks (turn, costly-choice, exposition-dump) assume exchange-level writing. Hard-failing them against single-line drafts would false-red every run and force the repair loop constantly. Write at the scene level first, *then* judge at the scene level.

Final order:

| # | Step | Risk | Gate to pass before next step |
|---|------|------|-------------------------------|
| 0 | **N=3 baseline on current build** | none | 3 episode means + per-axis recorded |
| 1 | De-exposition prompt block (composer + Story Room writer) | low | regression green; forbidden words absent in N=2 |
| 2 | `slot_drama_contract` generation per voiced slot | low-med | contract present + sane on N=2; LLM call tagged (rule 6) |
| 3 | Story Room emits `d###|SPEAKER` rows; one-in/one-out | med | extract+commit still 8/8, no fallback, slot IDs intact |
| 4 | `compose_exchange` (2–3 voiced beats per call) | med-high | VRAM ≤ 14.5 GB; commit clean; subjective read improves |
| 5 | **Tier-A** deterministic validator (hard-fail) pre-commit | low | no false-reds on a known-good draft |
| 6 | **Tier-B** semantic checks (warn-only → promote) | med | warn logs look right for ≥3 eps before hard-fail |
| 7 | One targeted repair pass (failed slots only) | med | repair touches only failed slots; fail-loud otherwise |

## The validator is two tiers [DECISION — do not blur them]

**Tier A — truly deterministic, hard-fail from day one:**
- slot count / order / speaker exact match (one-in/one-out)
- empty voiced line
- word-count floor per line
- forbidden-word hit (lexical)
- ≥1 required concrete noun from the slot's `concrete_detail_required` is present

**Tier B — semantic, NOT deterministic. Warn-only first, promote per-check only after it proves stable:**
- costly-choice slot contains a decision / refusal / confession
- `must_turn` line changes the situation
- `EXPOSITION_DUMP` (states the dramatic state directly)

Tier B cannot be judged in pure Python without an LLM, and an LLM judge reintroduces the jitter/looping we're escaping. Approximate with crude proxies where possible (e.g. costly-choice beat must contain a first-person commitment verb + a concrete consequence noun) and accept they're heuristics. **Never let `EXPOSITION_DUMP` silently become an LLM taste call.**

## Lever weighting [DECISION]
The **concrete-detail requirement** is the primary lever; the forbidden-word list is secondary (it's whack-a-mole and over-fires on legit sci-fi like "anomaly"). Keep both, weight the requirement.

---

## Per-phase detail

### Phase 0 — Baseline (do first, ~1 evening)
Run 3 episodes on the current build (heartbeat already shipped, `2b3e708`). Record `mean_score` + the 10 axis scores for each. This is the number every later phase must beat past ±0.2–0.4 jitter. Focus axes: **naturalness, emotional_arc, specificity** (the persistent 3s).

### Phase 1 — De-exposition guardrail
Inject the banned-language + concrete-detail block into the line composer (`compose_line_multiturn` path) and the Story Room writer prompt. *Locate exact file/surface first* (`nodes/_otr_story_room.py` + the composer module). No JSON surface change. Fast single-axis nudge for naturalness/specificity.

### Phase 2 — `slot_drama_contract`
Before the room writes, generate per voiced slot: `line_job`, `hidden_pressure`, `concrete_detail_required`, `forbidden_words`, `state_before`, `state_after`, `must_turn`. This is the obligation the floor enforces. It's a new LLM pass — tag creative/technical per rule 6, budget latency, and sanity-check the contract before trusting any floor built on it (garbage contract → garbage floor).

### Phase 3 — Slot-formatted Story Room output
Writer outputs exactly `d001|ANNOUNCER: ...`, one row per slot. Hard rule: one row in, one row out, same slot IDs, same speakers, no added/skipped slots. This protects the clean commit we already have (8/8, no fallback) and makes extraction trivial.

### Phase 4 — `compose_exchange`
Replace isolated `compose_line(dNNN)` with `compose_exchange(d002,d003,d004)` — write a 4–6 line exchange satisfying 2–3 slots, prior committed lines in context, scored at the *scene* level. This is where subtext/turn/avoidance become expressible. Watch context length vs the 14.5 GB ceiling; truncate against `context_cap`; never `force_vram_offload()` between LLM phases.

### Phase 5–6 — Validator (Tier A then Tier B) before commit
Wire Tier A hard-fail. Add Tier B warn-only; read the warn logs for ≥3 episodes; promote individual Tier-B checks to hard-fail only once their warnings are trustworthy.

### Phase 7 — One repair pass
On failure, send only the failed slots back with the exact failure reason; allow exactly one repair pass; keep slot IDs + speakers; fix only the listed failure. Still failing → fail loud or fall back to legacy. **No multi-cycle vague editor loops** — that's the churn we're killing.

---

## Discard / Keep

**Discard:** more editor cycles; vague rubric scoring; "make it more dramatic"; single-line best-of-N; trusting DramaticState alone; music interludes papering over a missing middle.

**Keep:** banned generic language; concrete-detail requirement; exchange-level writing; slot drama contracts; deterministic pre-commit floor; one targeted repair.

## Non-negotiables (project rules)
- **Audio is king** — every phase is text-side; if commit→audio destabilizes, revert. Re-run the audio byte-identity gate each phase.
- **14.5 GB VRAM ceiling**; `_flush_vram_keep_llm()` not `force_vram_offload()`; truncate to `context_cap`.
- **Run Bug Bible + core + audio regression after every change** (CLAUDE.md), and **N=3 means** before declaring any phase a win.
- **No "dummy"; UTF-8 no BOM; SFW; full arc.**
- Wire any node/widget/socket change into the workflow JSON (PD3); most phases here are internal and add no node surface.

## Definition of done
Across N=3 episodes, **mean ≥ 4.0** with **naturalness, emotional_arc, and specificity each ≥ 4**, SFW still 5, commit still clean, and no new editor-loop churn. That's "off the plateau."

## Loose threads (separate side-quests — log, don't bundle)
- **Stage-1 shadow validation bug:** plan emits `tension=0`, schema floor is 1 → shadow plan thrown away every run. Widen floor to 0 or clamp the generator. → `BUG_LOG.md`.
- **Critic key discrepancy:** ledger has `stage7_shadow_critic.mean_score` (3.7) but not `whole_episode_critic` (the baseline key). Confirm same critic vs two passes so the baseline is provably apples-to-apples.
