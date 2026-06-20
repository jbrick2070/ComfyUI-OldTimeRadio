# Audio Segment Normalization -- Operator Spec (Jeffrey, verbatim)

> This file preserves Jeffrey's authored spec exactly as given. It is the
> source of truth for scope. The architecture / test-plan / sequence docs in
> this folder elaborate on it but never widen it.

---

## Authored spec

```
Audio normalization optimization path:

- Design and implement an optional segment-level normalization stage before audio concat/join.
- Normalize each generated segment independently so character turns and scene transitions arrive at the assembly stage with more consistent loudness.
- Keep this default-OFF and fully gated.
- Frozen-audio-spine rules remain in force:
    * Disabled path must remain byte-identical to current output.
    * New tests proving legacy parity when disabled.
    * Metadata stamp indicating normalization enabled/disabled.
- First implementation target:
    1. Measure segment loudness.
    2. Apply per-segment normalization.
    3. Join segments.
    4. Emit final episode loudness report (LUFS/RMS/peak metrics if available).

Do NOT perform global audio re-baselining, loudness-target policy changes, LUFS standard selection, or RMS architecture changes yet. Those remain roundtable-gated.

Success criteria:
* Audible reduction in segment-to-segment volume swings.
* No clipping introduced.
* No regression to Bark rendering pipeline.
* No impact to frozen output when feature disabled.

Deliverables:
* Architecture note.
* Test results.
* Before/after loudness measurements.
* Example episode comparison.
* Runtime impact measurement.
```

---

## Hard-NO list (operator explicit)

- **No global audio re-baselining** -- the frozen master stays frozen.
- **No loudness-target policy changes** -- LUFS standard selection deferred to roundtable.
- **No RMS architecture changes** -- deferred to roundtable.
- **No touching the Bark rendering pipeline itself** -- segment normalization is a
  downstream post-process; Bark output bytes stay untouched.
- **No disturbing the frozen master for the disabled path** -- byte-identical is a hard gate.

## Stop conditions

- Audio byte-identical breaks on the disabled path -> `git reset --hard` uncommitted + ping
  immediately. **This is the canary; protect it absolutely.**
- Clipping introduced on the enabled path -> log + fix on the spot.
- Bark rendering pipeline shows any regression -> log + ping (out of scope to fix; means the
  design crossed the line).
- Real ambiguity in scope (something that might be "global re-baselining" / "LUFS policy" /
  "RMS architecture" rather than per-segment normalization) -> log + skip + surface.
- File-lane collision with another active session -> coordinate via GO_FORWARD_PLAN + ping.
- $20 cost ceiling.
- Destructive op needing approval -> log + skip.

## Sequencing dependency (do not violate)

1. **Phase 0 (NOW, doc-only):** these docs. No code edits -- the story-quality soak
   (`local_5c212bd6`) is exercising the audio spine; editing spine code mid-soak risks
   corrupting its audio-byte-identical proof.
2. **Phase 1 (CPU code):** starts only AFTER the story-quality soak completes AND the
   qwen_image / hidream_i1 promotion (`local_608386ee`) lands.
3. **Phase 2 (GPU verify):** single-episode smoke with the feature ON; waits for BOTH the
   story soak and the image session's GPU work to free the 5080.
4. **Phase 3 (deliverables):** test results, loudness measurements, runtime impact, example
   episode comparison.
