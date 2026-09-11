# Reported diagnostic corrections

Base59a40131; code on v2.0-alpha. Independent CLI QA: CursorGrok4.6High;
Euler verified the small accepted test/docstring follow-ups. Root grounded both.

- O1: explicit cast-preflight phase logs deferred missing stills at INFO.
  The request is identical to the render-phase request; actual render-time
  absence and unrelated exceptions retain their existing checks.
- O3: successful active-episode path reconciliation logs INFO in mux, image
  dispatcher and clip persistence. Prior warnings, failed identity/freeze
  checks and downstream PCM proof remain unchanged. Path matching alone does
  not claim byte identity.
- Updated a stale mux-canonical test expectation for My Story's already-shipped
  script_json link291. Existing audio link278/fanout checks remain. No workflow edit.

Final focused:203passed/1skipped. Full:14135passed,53pre-existing failures,
183skipped,1xfailed. Versus boundary baseline, exactly the stale canonical test
changes from failure to pass; no new failure IDs or changed assertion contents.
One existing AMD dictionary assertion changes display order only. Full comparison:
diagnostics_full_comparison.json.

Bug Bible:30passed/10unchanged failures/11skipped/3xfailed. It ran against a fresh
detached59a40131 checkout plus the exact9reviewed files; diagnostics_snapshot.json
records source/candidate file hashes. diagnostics_bible_comparison.json has no
outcome or failure-payload differences. No new duplicate PBUG/Bible rule is
invented for warning-only outcomes; WIRE-W2 and PBUG-20260721-14 / BUG-12.66 retain
their established contracts.

Canonical validator, JSON roundtrip and link/widget audit pass:23nodes/63links/
37writerwidgets. SHA256 is unchanged:
d586a286aaee4c039e410ae9a10014c5c7f4ab82d00eac0e9e1cc0564415057c.

No new media run or source-fidelity qualification. O2 LTX intent, grammar/P1,
source preservation, cleanup and Mac measurement remain in GO_FORWARD. The
cross-machine architecture arc has now converged; adaptive briefing requires
actual native capacity before implementation. Review receipts are under
kibitz-runs/2026-09-10-my-story-cross-machine/ and my-story-diagnostics-qa/.
