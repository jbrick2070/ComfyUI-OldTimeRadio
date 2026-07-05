# Claude anchor review -- r1 (arc / creative coherence)

VERDICT: SHIP-with-nits. The C1 arc is coherent and correctly scoped; the
extraction core is already shipped and grounded. The producer-wiring fork is
real but low-stakes TODAY.

Grounded facts (CONFIRMED against the real tree):
- CONFIRMED: `nodes/_otr_audio_motion.py` is import-clean (numpy + lazy
  soundfile; no torch/comfy) and read-only; 14 tests green; full suite 6628/0.
- CONFIRMED: node 92 OTR_VideoRenderBatch already receives patched_ledger_json
  (link 260) + master_audio_path (link 264) and already slices the frozen
  master per beat via render_driver._slice_master_audio (read-only, cache-keyed
  on master_audio_sha256). => Option B needs NO new inputs.
- CONFIRMED: nothing consumes `audio_motion_profiles` yet (C2 deferred). So the
  ONE robustness advantage of Option A (profile present on procgen-only /
  still-only episodes that never hit node 92) buys nothing until a consumer
  exists.

MUST-FIX (arc): none blocking.

SHOULD-FIX:
- State the consumer story explicitly: because C2 is the first reader and it is
  deferred, "always-runs producer" is not load-bearing now -> B is the
  proportionate build; A is promoted when C2 lands AND needs the profile on
  non-motion episodes (operator present to eyeball the JSON then).
- Flag onset_s / speech_vs_music as documented v1 heuristics so C2 does not
  over-trust them.
- Whichever option: the stamp must be fail-soft (never block a render) and
  read-only (byte-identity), matching the shipped core's contract.

Recommendation: B now, A when C2 needs it. Await panel grounding before locking.
