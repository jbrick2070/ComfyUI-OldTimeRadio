# HuMo audio-concat mode — design doc

**Filed:** 2026-04-29 PM
**Author:** locked with Jeffrey
**Goal:** mechanically guarantee lip-sync by routing audio FROM the
HuMo clips themselves, with full-timeline visual coverage via radio
bookend gap-fills.

## Insight

Each HuMo clip has perfect lip-sync to its OWN audio (HuMo was
conditioned on that exact dialogue waveform). The drift we worry
about happens when VideoComposite OVERWRITES HuMo's audio with
master_mix audio at the `-map 0:a` step — master_mix has music
ducking, SFX layering, EQ, ITU loudness norm, all of which slightly
shifts phonemes vs. what HuMo was trained on.

**If we just keep each HuMo clip's native audio + concat in order,
lip-sync is mechanically perfect.** Nothing ever de-syncs.

But we lose music, SFX, ANNOUNCER, and gap audio if we go pure
concat. Those are not background — they ARE the radio talking
(Jeffrey 2026-04-29 PM: "MUSIC ANNOUNCER SFX ALL TALKING RADIO").

## Architecture

Hybrid: HuMo clips for dialogue windows, radio-bookend image +
master_mix slice for gap windows. Concat all in time order. Master
audio = sum of (HuMo per-clip audio) + (master_mix slices in gaps).

```
Timeline windows for episode duration D:
  [0,                clip[0].start_s]   <- gap (opening music + announcer)
  [clip[0].start_s,  clip[0].end_s ]    <- HuMo clip 0 (its native video + audio)
  [clip[0].end_s,    clip[1].start_s]   <- gap (transition + music bed)
  [clip[1].start_s,  clip[1].end_s ]    <- HuMo clip 1
  ...
  [clip[N].end_s,    D                ] <- gap (closing music)

For each gap window [a, b]:
  ffmpeg -i radio_bookend.png -i master_mix.wav \
         -ss a -to b -c:v libx264 -t (b-a) gap_<i>.mp4
  with master_mix audio sliced to [a, b]

For each HuMo window:
  copy clip<i>.mp4 verbatim (video + native audio)

Final assembly:
  ffmpeg concat demuxer joins gap_0.mp4 + clip_0.mp4 + gap_1.mp4 +
  clip_1.mp4 + ... + gap_N.mp4 -> episode.mp4
```

## VideoComposite changes

- New widget: `audio_source` (dropdown):
  - `"master_mix"` (default, current behaviour, no change to anyone)
  - `"humo_concat"` (new, opt-in)
- When `audio_source == "humo_concat"`:
  1. Read `ledger.clips[]` sorted by `start_s`
  2. Read `ledger.radio_bookend_path` -- if missing, fall back to
     first HuMo clip's middle frame extracted via ffprobe
  3. Build segment list (alternating gap + HuMo)
  4. For each gap: ffmpeg-create static image + master_mix slice mp4
  5. Concat-demuxer all segments -> final mp4
  6. Skip the existing 2-layer filter graph entirely
- When `audio_source == "master_mix"`: existing path runs unchanged

## Failure modes

- **Bookend missing:** fall back to procgen frame for gaps. Ugly but
  doesn't crash. Logs warning.
- **Master_mix audio shorter than expected:** zero-pad with silence.
- **HuMo clip count == 0:** error early with clear message.
- **Gap window 0 length:** skip, no segment generated.
- **Crossfade artifacts at concat boundaries:** apply 30 ms audio
  crossfade via concat-with-filter mode (instead of pure demuxer).

## Cost

Per episode: 1 ffmpeg pass per gap window (typically 5-15 gaps for a
12-line episode) + 1 final concat pass. Wall-clock < 30 sec.

vs. current: 1 filter_complex ffmpeg pass, ~5-10 sec.

Net: +20 sec per episode, in exchange for mechanical lip-sync.

## When to land

After BUG-LOCAL-113 is fixed (bookend rendering reliably). The new
mode falls back to procgen gap-fill if bookend missing, but the
talking-radio identity needs the bookend to be the right shape. Land
this in two commits:

1. Bookend reliability fix (BUG-113) -- diagnostic just shipped
   tonight in commit pending; tomorrow's first task is to interpret
   the next run's bookend log line and patch the cause.
2. VideoComposite `audio_source = humo_concat` mode -- after bookend
   is reliable, ship this with a smoke test at 100-word smoke run
   so we can compare master_mix vs humo_concat on the same script.

## Test plan

- `tests/test_video_composite_audio_modes.py` -- new file
- Mock a 3-line ledger with 3 HuMo clips + 4 gaps
- Assert master_mix mode: audio = procgen.0:a passthrough
- Assert humo_concat mode: timeline segments add to episode duration
  within 1 frame, audio at HuMo clip boundaries comes from clip's
  native track, audio in gaps comes from master_mix slice
- Lip-sync regression: byte-compare HuMo clip audio sample at clip
  start_s before vs after composite -- must match within float
  tolerance

## Open questions for Jeffrey

1. **Crossfade duration at boundaries:** 30 ms hard, or 100 ms
   smoothest? 30 is invisible, 100 may smear consonants.
2. **Default mode:** keep `master_mix` as default (safer rollout)
   or flip to `humo_concat` as default (faster validation)? My
   recommendation: keep master_mix default until 5+ runs have been
   compared.
3. **What goes in the gap visual?** Pure radio bookend, or radio
   bookend with subtle motion overlay (CRT scanlines + audio-reactive
   flicker layered on top)? My pick: radio bookend + procgen-CRT
   lighten layer at 0.3 opacity, same as current dialogue overlay.
   Keeps the broadcast aesthetic continuous between dialogue and gap.
