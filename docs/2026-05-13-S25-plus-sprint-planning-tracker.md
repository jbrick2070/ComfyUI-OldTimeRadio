# S25+ Sprint Planning — Master Tracker for Everything Not Yet Locked

**Date:** 2026-05-13
**Branch:** `v2.0-alpha` at `f529812`
**Purpose:** consolidate every deferred item, every pending IMP-*, every scoped-out sub-task across all batches into one prioritized list so Jeffrey can lock down clean sprints to close them. The two batch QA docs (`2026-05-13-voice-path-cleanbreak-S15.5-S19-qa.md` and `2026-05-13-S24-fix-sprint-qa.md`) cover what shipped + ideas surfaced; this doc covers what's **left to ship**.

Tiers are by ship-priority, not chronology. Each entry names the source (which batch / plan / IMP), the unblock condition, and a suggested sprint package.

---

## Tier 1 — Architectural decisions blocking v2.0 lock

These are items where the design call itself hasn't been made (or the prior plan's premise was wrong) and a real choice has to land before the implementation sprint can start.

### T1.1 — C8: CastContract quarantine — **DECISION NEEDED**
- **Source:** S24 / C8 plan, deferred 2026-05-13 (commit `bb689f2`).
- **Status:** deferred; premise corrected.
- **What's needed:** pick one of three unblock options:
  1. **Extract helpers** — move just the helpers `cast_repair` needs from `_otr_cast_contract.py` into a new `_otr_cast_helpers.py`, then quarantine cast_contract clean. Smallest scope; needs an audit of what cast_repair actually imports.
  2. **Quarantine the chain** — move `cast_contract + cast_repair + apply_deterministic_cast_repairs` to `experimental/` together as one big sprint. Cleanest separation; largest scope.
  3. **Drop the quarantine plan** — accept that cast contract is production-wired; reframe C8 as "audit + harden cast_contract as a production module" instead.
- **Why HIGH:** the QA doc flagged this as the most architecturally consequential question in the S24 batch. Touches the writer-time invariant chain.
- **Suggested sprint:** S25 architecture decision + execute. Round-robin (ChatGPT + Gemini) is worth running given the trade-off shape.
- **Detailed entry:** `docs/cleanbreak-deferred.md` §C8.

### T1.2 — S14.2 implementation — **SCHEDULED**
- **Source:** original S14.2 plan from S15.5-S19 batch; decision ADR landed S24 / C12 (commit `d35aa71`).
- **Status:** decision locked (Option B: opt-in `OTR_WorkflowValidator` first-node). Implementation deferred to S25+.
- **What's needed:** build the validator node. Estimated scope per ADR:
  - `nodes/_otr_workflow_validator.py` (~150 LOC) — INPUT_TYPES + execute() that reads the workflow JSON from disk + calls `validate_workflow_contract(strict_unknown_types=True)`.
  - `workflows/otr_scifi_16gb_full.json` — insert the validator as position-0 in `nodes[]`.
  - `tests/test_otr_workflow_validator.py` — cover canonical workflow + adversarial broken workflow fixture.
- **Why HIGH:** closes the runtime gap for hand-edited workflows; the canonical workflow is already CI-covered.
- **Suggested sprint:** S25 implementation sprint. Standalone (no design call needed).
- **ADR:** `docs/2026-05-13-S14_2-active-validation-ADR.md`.

---

## Tier 2 — Pending-gate (waiting on time / usage cycles, no design work needed)

### T2.1 — S19.3 — Survival-guide promotion of known-failures hook
- **Source:** S15.5-S19 plan; S19.1 + S19.2 shipped in commit `32f62eb`.
- **Status:** deferred; gated on 2-3 clean sprints of S15.3-based usage. Two clean sprints have now passed (S15.5-S19 + S24).
- **What's needed:** in the sibling repo `comfyui-custom-node-survival-guide/`, create `patterns/known-failures-hook.md` documenting the pattern (hook structure, 80% subset threshold, setup/call/teardown tracking, PROMOTABLE banner mechanic) and cross-link from OTR's `docs/known-failures.md`.
- **Why MEDIUM:** the pattern has proven out; the gate condition is met. Pure doc work in a sibling repo.
- **Suggested sprint:** S25 add-on; can ship alongside any other sprint. Manual sibling-repo commit per the project rule.
- **Detailed entry:** `docs/known-failures-promotion-pending.md`.

### T2.2 — IMP-33b — Cross-version ComfyUI stability
- **Source:** S24 / C9 follow-up (commit `af7e7b1`).
- **Status:** tracked in `_LLMTimeoutWorkflowPause` class docstring.
- **What's needed:** on the next ComfyUI major bump, verify the queue-halt assumption still holds. The mock-based smoke (C9) doesn't catch a silent ComfyUI behavior change.
- **Why LOW (no-ETA):** triggered by external event (ComfyUI release), not work we plan.
- **Suggested sprint:** N/A — bookmark in `docs/2026-05-13-imp33-queue-halt-test-decision.md`.

---

## Tier 3 — Improvements that close known gaps

### T3.1 — IMP-43 — Widget-vector alignment check in S14.2 validator
- **Source:** S24 QA doc / new this batch.
- **Status:** open; awaiting S14.2 implementation.
- **What's needed:** should the S25+ `OTR_WorkflowValidator` also check widget-vector length against `INPUT_TYPES` declared slot count? Catches BUG-LOCAL-210-class drift (cleanbreak deleted a required input but a saved workflow still carries the stale widget value) in user-edited workflows.
- **Why MEDIUM:** closes a runtime gap that fired on production AudioGen during S24/C3. The C6 test covers the *canonical* workflow; this would extend it to ANY workflow the validator runs on.
- **Suggested sprint:** fold into S14.2 implementation sprint (T1.2) -- adds ~30 LOC + 1-2 tests, fits in scope.

### T3.2 — IMP-42 — AST meta-test for `EXCLUDED_PATHS` justification comments
- **Source:** S24 / C11 follow-up.
- **Status:** open.
- **What's needed:** extend `tests/test_legacy_audit_clean.py` with a meta-test that AST-walks the `EXCLUDED_PATHS` frozenset literal in its own source and asserts each entry has a preceding `# justification:` comment within N lines. Mechanizes the discipline rather than relying on PR review.
- **Why MEDIUM:** the rule exists (C11) but enforcement is human. AST check makes it CI-enforced.
- **Suggested sprint:** S25 add-on. Self-contained ~30 LOC test addition.

### T3.3 — IMP-40 — Cleanbreak playbook step: "shrink widget vectors"
- **Source:** S24 BUG-LOCAL-210 lesson.
- **Status:** open; doc-only.
- **What's needed:** add a step to the cleanbreak playbook (wherever the playbook lives — probably in the `comfyui-custom-node-survival-guide` repo): "When a cleanbreak deletes a REQUIRED `INPUT_TYPES` entry, every saved-workflow `widgets_values` vector MUST be trimmed at the same index in lockstep." Cite BUG-LOCAL-210 as the example.
- **Why MEDIUM:** prevents the next BUG-LOCAL-210-class instance in a future cleanbreak.
- **Suggested sprint:** S25 add-on; couple lines of doc.

### T3.4 — IMP-34 — Audit context-window structural marker
- **Source:** S15.5-S19 QA / surfaced 2026-05-13.
- **Status:** open.
- **What's needed:** the current `tests/test_legacy_audit_clean.py` uses a 5-line context window lookback to handle multi-line forensic comment blocks. If a refactor moves a marker line out of a block, downstream lines silently break. A structural marker (e.g., "is this line inside a class whose name ends in `_LEGACY_TOKENS`" or "is this inside a frozenset literal named `DELETED_NODE_TYPES`") would be more robust.
- **Why LOW-MEDIUM:** the current heuristic works in practice; this is forward-looking robustness.
- **Suggested sprint:** S26+ when scope allows. Could be folded into a broader "audit-test hardening" sprint with T3.2 + IMP-46.

### T3.5 — IMP-35 — S16.4 workflow JSON surgical patch alternative
- **Source:** S15.5-S19 QA.
- **Status:** open.
- **What's needed:** S16.4 used `json.dumps(indent=2)` for the FluxPortrait wiring writeback, which reformatted the entire workflow JSON's whitespace. Surgical link-by-link patches would preserve whitespace at the cost of more complex writeback code. Decide which trade-off is right going forward.
- **Why LOW:** cosmetic; the data structure is preserved correctly. Decision is "ongoing workflow-edit convention," not a fix.
- **Suggested sprint:** decide via comment in `CLAUDE.md` rather than a sprint. ~5 minutes of design call.

---

## Tier 4 — Forward-compat / nice-to-have

### T4.1 — IMP-39 — `_fallback/` directory growth policy
- **Source:** S24 / C2 follow-up.
- **Status:** open.
- **What's needed:** the C2 AudioGen short-output fallback saves to a sibling `<cache_dir>/_fallback/` dir indefinitely. Worth a rotation policy (delete entries >N days), single-file rotation, or accept unbounded growth on kB-scale wavs?
- **Why LOW:** disk impact is small (procedural + sfx wavs are kB-scale); the fallback is rare (only fires on transformers AudioGen regression).
- **Suggested sprint:** S26+ if disk fills up; otherwise N/A.

### T4.2 — IMP-41 — README Node Reference schema-version tagging
- **Source:** S24 / C1 follow-up.
- **Status:** open.
- **What's needed:** tag each row in the README's Node Reference table with the schema version it targets (currently `l3-2026-05-14`). Makes schema bumps surface in the README diff.
- **Why LOW:** maintenance discipline trade-off; benefit is small unless schema bumps are frequent.
- **Suggested sprint:** S26+ doc cleanup.

### T4.3 — IMP-44 — `"skipped"` enum slot decision
- **Source:** S24 / C5 follow-up.
- **Status:** open.
- **What's needed:** `ALLOWED_SFX_RENDER_STATUS` includes `"skipped"` but no producer stamps it today. Drop the slot or document the expected future producer.
- **Why LOW:** forward-compat trade-off; no consequence either way.
- **Suggested sprint:** N/A — fold into a future status-enum cleanup sprint if one happens.

### T4.4 — IMP-45 — ProcSFX alias-chain intent pin
- **Source:** S24 / C4 follow-up.
- **Status:** open.
- **What's needed:** the resolver's "Additional semantic aliases" `if/elif` chain has FIRST-MATCH-WINS semantics. The matched-flag test (C4) doesn't pin "keyword loop is primary; alias chain is fallback before radio_tuning default." Worth a test?
- **Why LOW:** the matched flag is sufficient for the contract C4 introduced; this is intent documentation, not a contract gap.
- **Suggested sprint:** N/A.

### T4.5 — IMP-46 — Actually-deleted LFC-era class names spot-check
- **Source:** S24 / C10 follow-up.
- **Status:** open.
- **What's needed:** C10 deferred LFC audit en masse because LFC = live system. But if there was an earlier `OTR_LFC_Cascade` or similar that was retired in favor of the current `OTR_LFCPhase4/5/6`, that specific name would qualify for the audit. Worth a targeted spot-check.
- **Why LOW:** speculative; may turn up nothing.
- **Suggested sprint:** S26+ during a slow week. ~30 minutes of `git log --diff-filter=D` work.

### T4.6 — IMP-33a — Real ComfyUI subprocess smoke
- **Source:** S24 / C9 follow-up.
- **Status:** skipped stub in `tests/test_llm_timeout_queue_halt_smoke.py`.
- **What's needed:** spin up real ComfyUI in a subprocess, force a ScriptWriter timeout, assert ComfyUI's actual queue halts. Highest-fidelity test of the `_LLMTimeoutWorkflowPause` assumption.
- **Why LOW (no-ETA):** ComfyUI doesn't ship a CI-friendly subprocess harness today. Waits on external tooling.
- **Suggested sprint:** N/A until ComfyUI harness exists.

---

## Tier 5 — Rule conflicts (won't ship unless Jeffrey opts in)

### T5.1 — S21.3 — Workflow preset split
- **Source:** S15.5-S19 plan; deferred 2026-05-13.
- **Status:** conflicts with standing rule `feedback_minimum_json_files` ("keep workflow JSONs to minimum, don't create variants").
- **What's needed:** Jeffrey explicitly opts in by overriding the standing rule. Otherwise stays deferred.
- **Why N/A:** rule conflict, not a technical question.
- **Suggested sprint:** never, unless rule changes.

---

## Tier 6 — Stretch / non-blocking (skipped)

### T6.1 — S20 stretch tasks
- **Source:** S15.5-S19 plan §S20.
- **Status:** marked non-blocking by the original plan; skipped during execution.
- **Content:** three stretch items in the original plan: (a) validator on non-OTR third-party nodes; (b) `LINE_ID_CONSUMERS` registry doc; (c) subset-coverage instrumentation on the 80% threshold.
- **Why STRETCH:** quality-of-life improvements, not contract gaps.
- **Suggested sprint:** S27+ during a slow week, or fold individual items into adjacent sprints.

---

## Closed in S24 (for reference — don't re-open)

| Item | Closing commit | Notes |
|---|---|---|
| S23.10 — README rewrite | `cf8eb96` (C1) | Closed. Forensic anchors preserved throughout. |
| IMP-31 — AudioGen `_cache_key` alias | `493ab8c` (C7) | Closed. Matches MusicGen S17.1. |
| IMP-32 — `sfx_render_status` enum check | `2bfab7f` (C5) | Closed. 8-value frozenset. |
| IMP-33 — Queue-halt smoke | `af7e7b1` (C9) | Closed (mock-based). Real-subprocess = T4.6; cross-version = T2.2. |
| IMP-36 — README schema-version tags | — | Reframed as T4.2 (was previously the README-rewrite item itself, now satisfied by C1; the schema-version tagging is a separate forward-looking IMP). |
| IMP-37 — LFC audit | `4e972c7` (C10) | Rejected; LFC is live architecture. |
| IMP-38 — EXCLUDED_PATHS justification | `f9f5aa7` (C11) | Closed. Per-entry comments + module docstring rule. |

---

## Suggested S25+ sprint packaging

Each sprint groups items that share scope / consequence. Sprints are sized to land in a single session each.

### S25 — Implementation sprint
**Headline:** ship the S14.2 validator + close the highest-priority decision.

- **T1.2** S14.2 implementation (~150 LOC node + tests + workflow wiring)
- **T3.1** IMP-43 widget-vector check folded into the validator (~30 LOC + tests)
- **T2.1** S19.3 survival-guide promotion (manual sibling-repo commit at end)

### S26 — Decision sprint
**Headline:** resolve the C8 CastContract architecture question.

- **T1.1** C8 round-robin (ChatGPT + Gemini) on the 3 unblock options
- Execute whichever option lands

### S27 — Audit-test hardening sprint
**Headline:** strengthen the legacy-audit machinery + close the smaller doc-only IMPs.

- **T3.2** IMP-42 AST meta-test for EXCLUDED_PATHS justifications
- **T3.3** IMP-40 cleanbreak playbook step (sibling repo or CLAUDE.md)
- **T3.4** IMP-34 audit context-window structural marker
- **T3.5** IMP-35 workflow JSON surgical-patch convention decision (or `CLAUDE.md` note)
- **T4.5** IMP-46 actually-deleted LFC-era names spot-check

### S28+ — Forward-compat / cleanup (opportunistic)
**Headline:** non-blocking; fold into adjacent sprints when time allows.

- **T4.1** IMP-39 `_fallback/` rotation
- **T4.2** IMP-41 README schema-version tagging
- **T4.3** IMP-44 `"skipped"` enum slot decision
- **T4.4** IMP-45 ProcSFX alias-chain intent pin
- **T6.1** S20 stretch items (if any prove worth doing)

### Never (unless rule changes)
- **T5.1** S21.3 workflow preset split

### External-event-triggered (no sprint)
- **T2.2** IMP-33b cross-version stability (on next ComfyUI major bump)
- **T4.6** IMP-33a real ComfyUI subprocess smoke (when ComfyUI ships a CI harness)

---

## Cross-references

| Doc | Purpose |
|---|---|
| `docs/2026-05-13-voice-path-cleanbreak-S15.5-S19-qa.md` | S15.5-S19 batch QA |
| `docs/2026-05-13-S24-fix-sprint-qa.md` | S24 batch QA |
| `docs/cleanbreak-deferred.md` | per-item deferral entries (C8 / C10 / S14.2) |
| `docs/2026-05-13-S14_2-active-validation-ADR.md` | S14.2 design decision (Option B) |
| `docs/2026-05-13-imp33-queue-halt-test-decision.md` | IMP-33 design decision (mock-based) |
| `docs/known-failures-promotion-pending.md` | S19.3 handoff marker |
| `docs/manual-smoke-tests.md` | S22.2 manual procedure |
| `ROADMAP.md` "CURRENT WORK" section | latest batch summary |
| `BUG_LOG.md` "Bible candidates pending promotion" | promotable bugs awaiting v2.0 ship |

---

## Tracker maintenance

Update this doc whenever a tracked item closes (link to the closing commit + move to "Closed" section), or a new item enters (add to the right Tier with source + unblock condition). Doing this in-batch keeps the tracker honest; deferring it lets items go stale silently.

Items that close should be moved to the "Closed in <batch>" section so the tier lists stay clean.
