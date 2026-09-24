# My Story treatment prompt clarification

The supplied incident recovered on its existing lower-temperature retry. This
change clarifies the two field roles that the prompt left asymmetric; it does
not establish that overlap caused the loop or that the new wording eliminates
stochastic repetition. The raw reported server log is unavailable locally.

Task list:
- [x] Read project lessons, relevant production history and Bible 12.100.
- [x] Trace shipped pack resolution, treatment schema and retry/repair delivery.
- [x] Clarify the existing pack instructions; keep the worked example and schema.
- [x] Scoped regressions: 342 passed (18.81 s).
- [x] Bug Bible: 24 passed, 36 skipped, 3 xfailed (2.27 s).
- [ ] Full-suite result and baseline comparison where needed.
- [ ] Push the scoped green change, independent QA, and final integrity checks.

Runtime scope: every My Story treatment using the shipped pack on Windows,
Mac or pod. The existing graph resolves this pack dynamically; no new node or
widget is introduced and no canonical-workflow edit is required. Sampling,
schema constraints and the liveness guard remain unchanged.

This is prompt hardening, not live model qualification. No new render or paid
model call is needed to verify prompt delivery; no claim is made about the
reported episode's completion or its monitor.
