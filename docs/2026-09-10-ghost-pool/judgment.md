# Finished-diff review disposition

One independent local CLI lane: `agy`, exit 0, verdict **NO BLOCKERS**.
The lane self-reported **Gemini 3.8 Flash (High)**; this is reviewer-reported
identity, not an independently queried model setting. Raw output is preserved
in `agy_review.md`. No second reviewer or arc was run.

The driver checked the claims against the frozen code and behavioral tests:

- `deterministic_leaf` records the latest chronological occurrence of each
  signature and chooses the oldest eligible one. Stable seed/beat probe order
  resolves equal ages; signatures reserved only by future replay have no past
  occurrence.
- `deterministic_batch` visits the original ordinals, contributing frozen rows
  to history at their actual position. The previous emitted signature and the
  next frozen signature are excluded. Fresh/fresh adjacency is protected when
  the second row is allocated.
- ShotLock passes an ordinal map for replay, supplies all replay signatures to
  writer admission, and passes each allocation disposition into that fresh
  beat's existing fallback_reason. Model-failure text survives the join.
- Frozen fallback objects keep their original reasons. The pre-existing
  writer_llm -> replay source transition is still present; the review's broad
  phrase "without modification" must be read with that existing transition.
- No stored field, template hash, recipe, profile, workflow, v3 render composer,
  env read or process call changed. Structural errors remain loud.

The review's mixed-replay test line citation was stale: the actual test is
`test_shotlock_preserves_per_beat_reuse_reasons_and_mixed_replay` at
`tests/test_ghost_prompt_v2_lane.py:528`. The referenced behavior was verified
by name and execution, not accepted from the line number. Its cited test
counts came from the driver's run, not a second test campaign by the reviewer.

The two nonblocking observations require no code change: replay's keys are
constructed from the same specs, and production ordinals cover the complete
ghost sequence. No public API accepts arbitrary sparse/overlapping ordinal
maps here. No new design decision was introduced.

The review clears the coding gate. It does not clear the five-act canonical
CUDA publication gate. `live_gate.json` records `requires_cuda` for both
ghost-authoring lanes under the actual Mac profile. No publication is claimed.
