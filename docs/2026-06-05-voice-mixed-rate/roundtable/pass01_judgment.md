# Pass 01 judgment log

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro (via OpenRouter,
~latest aliases). Spend this pass: ~$0.17. Judge: Claude (Opus), grounded
against the real files.

## Verdicts
- GPT-5.5: yes-with-fixes. Gemini: yes-with-fixes. DeepSeek: build-ready with
  fixes. Grok: no (doc lacked a concrete resampler + test matrix). All four
  agreed on the DIRECTION (A) and on cutting B/C/D.

## CONFIRMED (grounded true, folded in)
- The bark fallback appends a 24000 clip into a 22050 batch; `pack_audio_batch`
  raises (`_otr_voice_node_common.py:401-407,423` + `base.py:122-128`). [all 4]
- Fix in the fallback branch only; downsample bark to primary `sr`, never upsample
  index (C7 bit-exact). [all 4]
- Keep `pack_audio_batch`'s raise as a tripwire; reject default-resample. [all 4]
- Reject B/C/D as over-engineered/semantics-changing. [all 4; DeepSeek's point
  that D wouldn't even clear the fallback at 6 chars/4 refs is correct.]
- `_bark_fb.unload()` is not exception-safe (`:418-422` after the loop, not in
  `finally`). [GPT only] -- verified real; folded in as companion hardening.
- Test matrix: all-index / mixed / all-fallback / single-line / mono / zero-line /
  determinism. [GPT + DeepSeek + Grok]

## CORRECTED (panel consensus overridden by ground truth)
- **Resampler = scipy.signal.resample_poly, not torchaudio.** All 4 said
  `torchaudio.functional.resample`. I verified torchaudio 2.10.0+cu130 is present
  and deterministic here (so it would work), BUT `scene_sequencer.py:109` already
  uses `scipy.signal.resample_poly` and comment **I-11** removed the torchaudio
  path for determinism. Reuse the project's own resampler. (Panel's was a
  reasonable generic answer; it just didn't know the codebase precedent.)

## MISREAD (grounded false; no code change)
- "Downstream SceneSequencer gets mixed rates" [Gemini MUST-FIX, GPT #8]:
  SceneSequencer already standardizes every batch to 48000 via `_resample_audio`
  (`scene_sequencer.py:592,686`); the 09:33 smoke assembled kokoro+bark+SA3. Kept
  only as a confirming test note.
- "Zero-line after fallback emits wrong rate" [Grok MUST-FIX #2]: zero-line
  short-circuits at `:306-307` before the loop; `sr` is already primary. Test, no
  code change.

## UNVERIFIABLE -> RESOLVED
- torchaudio availability [all 4 flagged ASSUMPTION]: resolved -- present +
  deterministic on the venv. Moot given the scipy decision above.

## Convergence
Stop after pass 1. The fix is a single localized branch + one helper + tests;
the panel was unanimous on direction and grounding closed every open item. A
second fan-out would only re-confirm. Offer a pass 2 only if the synthesized
plan itself needs adversarial review before build.
