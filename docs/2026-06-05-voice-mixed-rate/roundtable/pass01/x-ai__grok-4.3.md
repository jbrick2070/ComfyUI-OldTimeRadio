<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Candidate directions leave the mixed-rate path in _otr_voice_node_common.py:401-407 unhandled while pack_audio_batch:122-128 still asserts; no concrete resampler or test matrix is specified.

MUST-FIX BEFORE BUILD:
1. [Candidate fix direction A] No resample call exists after _bark_fb.generate_voice at _otr_voice_node_common.py:401-407; clips list therefore mixes 22050/24000 tensors before the pack_audio_batch(sr) call at line 423. Add deterministic torchaudio.functional.resample (CPU, warn_only path) immediately before clips.append in the bark fallback branch only.
2. [_otr_voice_node_common.py:273 and :307] empty_audio_batch(sr) paths use the adapter/profile rate but the fallback branch never updates sr; a zero-line episode after a bark fallback still emits the wrong rate. Ensure the fallback path forces sr = 22050 (primary) before any empty_audio_batch.
3. [pack_audio_batch in base.py:122-128] The ValueError is the only guard; direction B would remove it without an opt-in path. Keep the raise but document it as the tripwire for non-fallback mixed-rate bugs.

SHOULD-FIX:
1. [Direction D] "Render whole episode on bark" changes casting semantics for every character when one ref is missing; this violates the per-line intent of profile.sample_rate and the "indextts2 default" contract in eng_indextts2.py:58. Drop D.
2. [Invariants section] No regression test matrix is listed for (a) all-bark, (b) single fallback line, (c) mono_safe vs channels, (d) zero in-role lines after fallback. Add at least those four cases before the C7 fixture baseline moves.
3. [Direction C] Canonical rate change touches every engine adapter and the byte-identical bark legacy path; the document gives no migration plan for existing C7 audio fixtures.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line rate comment next to every engine's sample_rate = N declaration.
- Expose target_sr as an optional kwarg to pack_audio_batch for future non-fallback callers only.

CUT THESE (over-engineering):
1. Direction C: full rate unification is heavier than a single conditional resample in the fallback branch and would require re-baselining every bark test that pins 24000 output.
2. Direction B: changing the shared primitive contract for one fallback case adds risk to unrelated call sites that currently rely on the strict single-rate precondition.

[ASSUMPTION] torchaudio is already importable in the Blackwell venv without extra pip deps (verify against the real environment; the grounding excerpts never import it).