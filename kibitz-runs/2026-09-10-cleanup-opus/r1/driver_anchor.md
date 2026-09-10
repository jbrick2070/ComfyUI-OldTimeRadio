VERDICT: yes-with-fixes. Three bounded candidates are code-grounded; proof instructions need final executable fixture/command mapping before an Opus handoff.

MUST-FIX BEFORE BUILD
1. CONFIRMED [4/P3]: identify concrete current real-cascade fixture fields and inline/producer policy selectors. Existing B1 tests patch the orchestrator, so they cannot prove its callback is unused. Source _otr_freeze_cascade.py:676-706 independently proves the retired callback sink; add a meaningful runtime proof, not a parallel fake cascade.
2. CONFIRMED [4/P4]: resolve exact drift/variant/validator commands and preserve the existing 54-failure set. The current full suite is not green; equal before/after failure sets are a bounded regression statement only.

SHOULD-FIX
1. CONFIRMED [1/C3]: comments at freeze node 265 and 336 imply live model work. Correct nearby stale prose without refactoring the first/post-finally serialization fallback chain (343, 404, 413).
2. CONFIRMED [3]: audio fixture uses blank VideoDirector policy and ReplayDescriptor and synthetic upstream voices/music. Its scope is explicitly non-foley/non-replay, not end-to-end production qualification.

CUT THESE
1. CONFIRMED [2]: no offset extraction, common-engine abstraction, old-harness blanket purge or model-loader change is needed. The current inline loop handles persisted clips and zero-offset music markers differently.

Evidence inspected at Windows HEAD 1f27e4123a9567400ab39ef698d2a8a394601284: exact seven Scene stores; G8/per-line duplicate ownership and phase callers; canonical registrations; freeze early exits, acquisition, retired callback, unload and serialization; existing G8/policy/unload tests; new audio script and launcher. No production edits made. Astra remains sole judge.
