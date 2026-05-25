# Round-Robin Problem Statement -- Per-Clip Audio Loudness Normalization

- **Date:** 2026-05-24
- **Topic:** per-clip-audio-normalization
- **Stage:** 00 -- question (pre-consultation)
- **Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha`
- **Gating:** audio path -- round-robin gated per CLAUDE.md ("Audio is king";
  "Definitely use it for ... anything touching the audio path").

---

## The observation

During episode playback the operator noticed that some Bark dialogue clips
are audibly quieter than others -- line-to-line loudness is inconsistent
within a single episode. The narration still plays, but the mix feels
uneven: one character line sits at a comfortable level, the next is notably
softer.

## Current normalization architecture (confirmed in code 2026-05-24)

Three normalization touch-points, **all peak-based**:

1. **BatchBark** -- `nodes/batch_bark_generator.py` (~L611-619). Every Bark
   dialogue clip gets a per-clip **peak** normalize to **-3 dBFS** before
   stitching. The code comment states the intent explicitly: "so quieter
   voices (e.g. Bark speaker variance) don't get buried in the mix."

   ```python
   peak = clip_t.abs().max()
   if peak > 1e-6:
       clip_t = clip_t * (10 ** (-3.0 / 20) / peak)
   ```

2. **AudioEnhance** -- `nodes/audio_enhance.py`. A `_normalize` (peak) helper
   exists but Step 7 is **deferred/skipped**: "Normalizing here (before
   crossfades) caused clipping during segment overlaps."

3. **EpisodeAssembler** -- final **peak** normalize to **-1.0 dBFS**, applied
   post-crossfade on the fully assembled track.

Announcer lines render on a separate bus (`KokoroAnnouncer`); their level is
set by Kokoro, not by the BatchBark -3 dBFS pass.

## The problem

The line-to-line loudness inconsistency persists *despite* the BatchBark
per-clip -3 dBFS peak normalize. This is expected behaviour, not a bug in the
existing code: **peak normalization does not equalize perceived loudness.**
Two clips peak-normalized to the same -3 dBFS can differ substantially in
loudness because Bark output has a variable crest factor (peak-to-RMS ratio)
-- a clip with one sharp consonant transient peaks early while its sustained
speech sits low; a clip with even delivery sits much louder for the same peak
value. The current code targets the right goal ("quieter voices don't get
buried") with a mechanism (peak) that cannot achieve it.

Operator's proposed direction: normalize each clip on a **loudness** basis
(RMS / LUFS) rather than peak -- and the question is whether per-clip loudness
normalization fixes the imbalance, and how it should interact with the
existing final-stage pass.

## Constraints the consultation must weigh

- **Prime Directive 1 -- "Audio is king."** Any change re-baselines
  `tests/test_audio_byte_identical.py`; the operator re-blesses the new
  baseline. Expected and acceptable -- but the change must be deliberate.
- **Silent-clip guardrail (BUG-LOCAL-031).** Clips below -28 dBFS RMS are
  silent-skipped (the 2026-05-24 verification run's line b003 measured
  -48.92 dBFS). A loudness-normalize pass MUST gate on a speech-RMS floor:
  RMS-normalizing a near-silent clip up to the dialogue target would amplify
  pure hiss by ~40 dB.
- **Crossfade clipping.** AudioEnhance already learned that boosting levels
  before crossfades causes overlap clipping. Per-clip loudness boosting feeds
  louder material into the crossfades; the final -1 dBFS pass catches the
  overall peak, but a crossfade of two loud clips can momentarily sum above
  0 dBFS.
- **Two voice buses.** Bark (dialogue) and Kokoro (announcer) have
  independent output levels. Fixing only Bark clip-to-clip consistency may
  still leave an announcer-vs-dialogue mismatch.
- **Music / SFX out of scope.** Music beds intentionally sit below dialogue;
  this question is about dialogue clip-to-clip consistency, not the
  dialogue-vs-music balance.
- **Platform.** 100% local, open-source, offline; Windows, single RTX 5080.
  A LUFS dependency (e.g. `pyloudnorm`) is pip-installable and offline-capable
  but adds a dependency vs. dependency-free RMS.

## Questions for the round-robin

1. **Metric.** Peak (current), plain RMS, gated RMS (ignore inter-word
   silence, EBU-style), or full LUFS (ITU-R BS.1770 / EBU R128)? Is
   dependency-free gated RMS a good-enough perceived-loudness match for a
   radio-drama dialogue use case, or is LUFS worth the `pyloudnorm`
   dependency?
2. **Target + ceiling.** Loudness normalization can push peaks above 0 dBFS.
   What target level, and how to handle the ceiling -- a true-peak limiter,
   normalize-then-peak-cap, or fixed headroom?
3. **Placement.** Replace the BatchBark -3 dBFS peak pass in place, or add a
   unified loudness-match pass downstream that spans BOTH buses (Bark
   dialogue + Kokoro announcer) before SceneSequencer assembly?
4. **Silent / borderline clips.** Confirm the speech-RMS floor gate. What
   target for a clip that contains real but quiet speech sitting near the
   -28 dBFS BUG-031 threshold?
5. **Crossfade interaction.** Does per-clip loudness matching meaningfully
   raise the crossfade-overlap clipping risk? If so, should the final
   EpisodeAssembler pass become a limiter rather than a plain peak normalize?
6. **Keep or drop the final pass.** With per-clip loudness matching in place,
   is the final -1.0 dBFS peak normalize still needed (as a ceiling/limiter)
   or redundant?

## Operator's leaning

Per-clip **loudness** (not peak) normalization, applied **in addition to** --
not instead of -- a final ceiling pass.

---

## Round-robin process

Per CLAUDE.md "Round-Robin Consultation":

1. **ChatGPT** (gpt-4.1) -- first opinion + critique of this problem
   statement. Output -> `__01_chatgpt.md`.
2. **Gemini** (gemini-2.5-pro) -- given ChatGPT's answer + this question;
   asked for agreement, corrections, additions. Output -> `__02_gemini.md`.
3. **Claude** -- synthesize, flag disagreements, decide the grounded answer.
   Output -> `__03_synthesis.md`.
4. Loop step 2 if the externals disagree on something material.

No code change lands before the synthesis. The change, when it lands, is
gated on a re-blessed `test_audio_byte_identical.py` baseline.
