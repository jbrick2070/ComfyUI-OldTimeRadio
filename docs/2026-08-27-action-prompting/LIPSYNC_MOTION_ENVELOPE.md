# The audio-in motion envelope -- ANSWERED, from human-reviewed lab results

**This closes `LIPSYNC_MOTION_PROBLEM_STATEMENT.md`.** The question was "how
much motion can the local lip-sync lanes actually do, based on research not
assumption". The answer came back grounded in the VRAM lab's own human PASS/FAIL
rulings, not analogy, and it is now baked into each lane's default.

## THE RULING

**Target moderate UPPER-BODY motion, never locomotion.** The practical ceiling:

> ONE main motion -- a small lean, a restrained body shift, or a very slow
> camera push -- PLUS one minor motion such as a nod or a small below-chin
> gesture.

Face medium-close, mostly frontal, continuously visible, unobstructed.
**Walking, strong profile turns, active tracking and broad hand acting are
beyond the locally proven range.** Do not combine the lean AND the camera push
on a first pass; they are separate rungs.

## PER LANE, WITH THE EVIDENCE

| lane | locally demonstrated | proven upper edge |
|---|---|---|
| **HuMo 14B FP8** (`humo`, `humo_14B_169`) | 3.88 s human PASS, bake-off winner (`HUMO_BAKEOFF.md:97`) | Highest-confidence push: controlled lean/nod plus one small chest-level gesture, camera LOCKED. A tiny push-in may substitute for the lean. **Hands or props near the face break first.** |
| **HuMo 1.7B** (`humo_1.7B`, `humo_1.7B_169`) | 5.16 s human PASS, accurate and stable | Slight head drift, nod, shoulder movement, small lean. Camera locked, gestures restrained. |
| **MiniMax H3 REF2VA** (`minimax_h3_audio_in`) | 5.17 s PASS on seed 43; the IDENTICAL prompt FAILED on seed 42 (`ENVELOPE_LADDERS.md:30`) | One weight shift or forward lean, slight head tilt, possibly a very slow small push-in. **Stop before strong yaw/profile -- the jaw, chin and collar begin reshaping.** Generate multiple seeds; one bad take is not a verdict. At 8 s it was judged "dub-grade". |
| **LTX 2.3 IA2V** (`ltx_audio_in`) | Visible articulation and gradual push, but **NO explicit phoneme-sync verdict** (`results/comparisons/ltx_audio_ia2v_official_template_20260821.md:33`) | EXPERIMENTAL. Medium-close single take, frontal speaker, slow push only. Its recorded failure is the subject TURNING AWAY. |
| **LTX 2.5 A2V** (foley/mime siblings) | **No human-reviewed motion-plus-sync result at all** | Do not use it to define the boundary. Face/head micro-motion only. Its rapid-tracking test was well beyond the safe edge. |

## MODEL-SPECIFIC PRACTICE

* **HuMo** -- favour the native 97-frame / 25 fps envelope (3.88 s); official
  guidance warns past 97 frames degrades, and names stronger audio guidance as
  the audio-motion sync control. Vendor demos show walking-while-speaking, but
  that is a vendor stretch, NOT proof for this local FP8 workflow.
* **H3** -- put the exact transcript inside `<d>[English]...</d>`, reference
  `<Audio 1>` explicitly, choreograph pauses and the final lip closure, and
  inspect two or three seeds. Judge against the ORIGINAL source audio, never
  H3's own reconstructed audio.
* **LTX** -- one continuous take, chronological physical actions, exact quoted
  dialogue, camera movement described relative to the speaker.

## HOW THIS LANDED IN CODE (2026-08-27)

It CORRECTED a first cut that was too aggressive, written before this research
existed:

* `eng_humo.py` -- was speaker+framing only; now states the safe envelope
  (medium-close, frontal, mouth and jaw visible, slight lean, one nod, camera
  locked) so an under-specified talking head cannot drift into a still.
* `eng_minimax_h3.py` audio-in -- **"shoulders and hands moving with the words"
  REMOVED.** Hands cross the face, which is the documented danger; replaced by
  one slight forward weight shift and a small head tilt, staying frontal.
* `eng_ltx_av.py` (both build paths) -- **"hands moving with the words"
  REMOVED.** This lane has no sync verdict and its failure is turning away, so
  it gets one continuous medium-close frontal take with a very slow push.

Every one of these is the NO-BEAT-TEXT FLOOR. When the beat states movement,
that text arrives as `text_prompt` and governs -- which is what HuMo's own
directive ("add no movement the beat does not state") requires.

## THE LADDER, IF THE EDGE IS EVER PUSHED

One variable at a time: (1) face and head motion only; (2) add one below-chin
gesture; (3) return to baseline and add a slow push-in INSTEAD; (4) only after
both pass, one measured step with compensating camera movement.

**Stop when:** the mouth becomes partly hidden, the head approaches profile, the
jawline reshapes, the hands cross the face, or pauses and the final lip closure
stop matching.

**For production today: HuMo 14B at the moderate-motion envelope is the best
edge. Do not cross into walking plus active camera movement.**
