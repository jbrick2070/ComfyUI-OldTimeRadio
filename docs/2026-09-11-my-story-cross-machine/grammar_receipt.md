# Grammar state and My Story P1 implementation receipt

Scope: M1 static state retention, implicit grammar limits, live S1/PBUG-20260910-05
binder routing. Model weights/tokenizer preprocessing remain resident; each
actual generate call owns fresh parser/prefix history, including min_p retry.
Weak-reference tests exercise success/error/retry and dynamic schema release.
Both LMFE initialization and post-builder configuration remove only implicit
array limits; explicit maxItems remains effective. Provider-capacity prose
disables open-string length monitoring while token-cycle detection stays active.
Message contract survives normalization and typed string repair.

P1 binds StoryTreatment once; all three attempts use it. P2/P3 retain their
original callables. Failed transport raw_completion is durable evidence, never
a returned proposal or source of accepted/proposed cast and act counts.

Validation:
- Focused 197 passed, including final CLI fixture corrections.
- Full 14,154 passed / 53 unchanged failures / 183 skipped / 1 xfailed.
  No new failure IDs, changed payloads or quarantines against e07476eb baseline.
- Controlled candidate Bible 30 passed / 10 unchanged failures / 11 skipped /
  3 xfailed; clean e07476eb baseline 29/11/11/3 under the same extended guard.
- New Bible 11.63 metadata valid; legacy validator has 146 issues on both
  baseline and candidate, zero newly introduced issues.
- Canonical validator, JSON round-trip and link/widget audit pass: 23 nodes,
  63 links, 37 writer widgets. SHA256:
  d586a286aaee4c039e410ae9a10014c5c7f4ab82d00eac0e9e1cc0564415057c.
- Single finished-diff CLI reviews: Gemini 3.8 Flash (High) for OTR;
  Cursor Grok 4.6 High for the separate Bible promotion. Claims grounded by root.

Machine-readable comparisons and candidate hashes are adjacent grammar_*.json.
The full run preceded two test-fixture-only corrections; production hashes did
not change. Final focused tests cover those corrections. The controlled Bible
candidate was resynchronized for the fixture correction and rerun afterward.

No GPU/model/media run in this chunk. No complete source qualification, native
capacity repair, cleanup conservation, visual source repair, or Mac OS-kill
cure is claimed. Continue A2 then the remaining A1R work in GO_FORWARD_PLAN.md.
