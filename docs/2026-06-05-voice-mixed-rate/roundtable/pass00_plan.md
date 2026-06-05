# OTR voice bug: mixed sample rates crash `pack_audio_batch` (indextts2 22050 + bark fallback 24000)

Panel: review the real code (appended as grounding) and pressure-test the fix.
You are critiquing a fix DIRECTION for a concrete crash, not writing the patch.
Be concrete, cite file:line, and flag anything that breaks an invariant below.

## Symptom (reproduced live, 2026-06-05)

A full-render combo C1 (60 words, 6 speaking characters, writer gemma-4-12b,
char-voice `engine=indextts2`, `voice_bank=default`, `allow_voice_reuse=False`)
crashed at runtime in ComfyUI node 81 `OTR_BatchCharacterVoices`:

```
OTR_BatchCharacterVoices: pack_audio_batch: mixed sample rates [22050, 24000];
resample to one rate before packing
```

The episode wrote a valid script and cast; the failure is purely in the
character-voice audio packing stage. It is deterministic for any episode where at
least one speaking character gets an indextts2 reference and at least one does
not (6 characters vs ~4 installed reference WAVs guarantees it).

## Root cause (exact control flow, with real code appended as grounding)

The character-voice node renders one clip per dialogue line, then packs all clips
into the single Bark AUDIO-batch contract `{"waveform": [B,C,T], "sample_rate"}`.

1. The primary engine is indextts2, whose native rate is **22050 Hz**
   (`eng_indextts2.py`: `sample_rate = 22050`). The per-line loop sets
   `sr = profile.sample_rate` (22050) and will pack at that rate.

2. indextts2 / chatterbox are voice-CLONING engines: each character needs a
   per-character reference WAV. When a cast row has no usable reference,
   `_otr_voice_node_common.py:385-407` renders THAT line with the **bark
   fallback** ("audio is king -- never hard-fail on a missing ref"). Bark's
   native rate is **24000 Hz** (`eng_bark.py`: `sample_rate = 24000`). The
   fallback clip is appended to `clips` AS-IS:

   ```python
   # _otr_voice_node_common.py ~401-407 (bark fallback branch)
   bark_seed = _seed_to_int64("bark", request.stable_line_seed)
   with deterministic_inference(bark_seed, warn_only=True):
       audio = _bark_fb.generate_voice(prepared, voice_preset or "v2/en_speaker_6", None, bark_seed)
   clips.append(audio)          # <-- 24000 Hz clip, NOT resampled to sr (22050)
   continue
   ```

3. After the loop, all clips are packed at the primary rate:

   ```python
   # _otr_voice_node_common.py ~423
   packed = pack_audio_batch(clips, sample_rate=sr, mono=mono)   # sr = 22050
   ```

4. `pack_audio_batch` ASSERTS a single rate and raises -- it does not resample:

   ```python
   # _otr_audio_engines/base.py ~122-128
   sr = int(sample_rate) if sample_rate else rates[0]
   mismatched = {r for r in rates if r != sr}
   if mismatched:
       raise ValueError(f"pack_audio_batch: mixed sample rates {sorted(set(rates))}; "
                        f"resample to one rate before packing")
   ```

So the episode mixes 22050 (indextts2 lines) and 24000 (bark-fallback lines) in
one batch, and the packer -- correctly refusing to silently concatenate
different rates -- aborts the whole render.

## The design tension

Two correct-in-isolation behaviors collide:

- **"Audio is king" (PD1):** a missing reference WAV must NOT hard-fail the
  episode; fall back to bark so it always renders.
- **Single-rate batch contract:** `pack_audio_batch` packs into one
  `[B,C,T]` tensor with one `sample_rate`, and refuses mixed rates rather than
  produce a batch whose clips play at the wrong speed downstream.

The fallback (a feature) violates the packer's precondition (a safety). Today the
safety wins and the whole episode dies -- the worst outcome, because it defeats
the very "always render" goal the fallback exists to serve.

## Candidate fix directions (critique these; propose better if you have one)

**A. Resample the bark-fallback clip to the primary `sr` at the append site.**
In the fallback branch, after `_bark_fb.generate_voice(...)`, resample `audio`
from 24000 to `sr` (22050) before `clips.append(audio)`. Smallest blast radius:
indextts2 clips stay bit-identical; only the already-degraded fallback line is
resampled. Keeps `pack_audio_batch`'s strict single-rate precondition intact.

**B. Make `pack_audio_batch` resample to a target rate instead of raising.**
Move the responsibility into the packer: resample every clip to `sample_rate`
(or to `max(rates)`), drop the raise. More general (catches any future
mixed-rate source) but changes a widely-shared primitive and weakens a contract
that currently catches real bugs; risks masking genuine rate mistakes elsewhere.

**C. Project-wide canonical voice rate.**
Force every char-voice engine to emit one rate (e.g. normalize indextts2 and bark
to 24000, or to 48000). Cleanest conceptually; largest change; touches the
byte-identical bark legacy path and every engine adapter; would require
re-baselining C7 audio fixtures.

**D. Avoid the mix entirely (engine consistency).**
If any character lacks a ref, render the WHOLE episode on bark (uniform 24000),
or auto-enable `allow_voice_reuse` / `cast_voice_policy=auto_registry` so every
character gets an indextts2 ref. Avoids resampling but changes casting semantics
and can silently demote an all-indextts2 intent to bark for the whole episode.

## Invariants the fix MUST NOT break

- **Byte-identical bark legacy path.** The pure-bark batch route
  (`interface="batch"`) is pinned byte-for-byte by tests. A fix must not alter
  output when no fallback occurs (an all-indextts2 or all-bark episode).
- **Audio is king (PD1).** The episode must still always render; the fix cannot
  reintroduce a hard-fail on a missing reference.
- **C7 determinism.** Voice render is seed-reproducible. Any resample must be
  deterministic (same input -> same bytes); no nondeterministic GPU resampler.
- **AUDIO contract.** Output stays `{"waveform": tensor[B,C,T], "sample_rate"}`,
  mono when `mono_safe`. `pack_audio_batch` still returns one rate.
- **No import-time IO/CUDA** (C-5); **16 GB VRAM** ceiling (Blackwell sm_120,
  torch 2.10 + cu130). Resampling should be CPU/torchaudio-light, not a model.
- **Empty-batch rate.** `empty_audio_batch(sr)` is used at `:273`/`:307`; the
  zero-line and zero-clip paths must keep a coherent rate.

## Questions for the panel

1. Which direction (A/B/C/D or a hybrid) is correct AND safest given the
   invariants? Name the single best place to resample and why.
2. For direction A, what is the right deterministic resampler on a torch
   2.10 + cu130 / numpy 2.x stack with no extra pip deps -- `torchaudio.functional.resample`,
   `torchaudio.transforms.Resample`, a polyphase (`scipy.signal.resample_poly`),
   or a hand-rolled kernel? Quality vs determinism vs dependency trade-off?
3. Is downsampling the bark fallback 24000 -> 22050 acceptable, or should the
   batch instead be packed at 24000 (upsampling the indextts2 22050 clips)?
   Which choice better preserves quality and downstream timing (SceneSequencer
   unbinds dim 0 and concatenates; HuMo lip-sync consumes the result)?
4. Are there OTHER mixed-rate sources in the same pipeline we should fix at the
   same time -- announcer (kokoro) vs character (indextts2/bark), or music
   (stable_audio_3 vs musicgen vs stable_audio_music) feeding a later mux?
5. What edge cases must a regression test cover (all-fallback episode,
   single-line, mono vs the `mono_safe` policy, zero in-role lines, a 1-female-ref
   cast forcing many fallbacks)?
6. Does fixing only at the append site (A) leave `pack_audio_batch`'s raise as a
   useful tripwire, or should the packer ALSO gain an opt-in `target_sr` resample
   so the precondition can never crash a render again? Argue the contract design.

Grounding appended below: `_otr_voice_node_common.py` (dispatch + fallback +
pack call), `_otr_audio_engines/base.py` (the packer), `eng_indextts2.py` (22050),
`eng_bark.py` (24000), `_otr_audio_utils.py` (canonical_audio / mono_safe), and a
`voice_nodes.json` excerpt of the four voice nodes (80/81/82/83) from the live
workflow.
