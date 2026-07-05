# 2C WIRING PLAN -- FINAL (r4 CONVERGED, BUILD-READY)

Arc: r1 -> r2 -> r3 -> r4, 2026-07-05. Panel = codex (gpt-5.5, reasoning=high);
antigravity DROPPED (credit bug, two hung attempts at near-zero CPU); Cowork
Claude anchor + judge every round. r4 verdict: yes, converged, no residual
build blockers.

THE PLAN OF RECORD = r3/final.md, plus the two r4 items folded in:
- Acceptance gains: "STAGE2_SUBPLAN.md lane-enablement checklist appended"
  (same commit).
- Meta-stamp test named explicitly: assert
  ledger.meta.source_bank == resolved["source_bank"] on the default path AND a
  runnable-stubbed non-default path.
- r4's 7-item VERIFY-AT-BUILD checklist is adopted verbatim as the build
  closeout list (r4/codex.md items 1-7).

## Arc summary (what each round bought)
- r1 (codex): 2C-only delta reframing; gate before _resolve_inputs (RSS);
  honest scope statement (fetch/interpreter stay science-hardwired);
  "4 callers" corrected to 2. Rejected: runnable-only dropdown (contradicts the
  converged honest-error design); push-policy change (operator session directive
  governs).
- r2 (codex): LATENT BUG FOUND -- refine _core locals() capture leaks
  os/_scaffold -> TypeError on any refine-enabled run since 2026-06-24; fix at
  root via signature-filtered capture + regression test + BUG_LOG entry. Gate
  moved to the VERY first statement (env mutation precedes the old spot). Full
  compose_line()/compose_line_draft() chain named. All positional widget-pin
  tests enumerated (story_scaffold_toggle, openrouter_s2, api_companions,
  guardrails).
- r3 (codex): threading contract made HONEST -- only line_composer_system is
  pack-routed; outline stays constant (resolver call is overlay-only); exchange
  prepass prompt is hard-coded (pre-existing, bank-agnostic) -> both moved to a
  LANE-ENABLEMENT CHECKLIST gating any future runnable:true flip. Recursive
  compose_line self-calls (:2507/:2664/:2762) threaded. Headless
  CREATIVE_WHITELIST x2 + patcher slot test. source_bank into _resolve_inputs
  resolved dict + ledger meta stamp.
- r4 (codex): converged; doc-acceptance + meta-stamp test naming only.

Agent calls: 4 codex + 2 antigravity attempts (both hung, killed). $0 cloud spend
(local CLIs).
