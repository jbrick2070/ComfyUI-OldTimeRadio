# Round D — Jeffrey's external code-review pass (2026-05-01)

Reviewer read `nodes/batch_humo_render.py` and `nodes/video_composite.py` directly and pushed back on the round-robin's framing in two places, confirmed Option 1, and added concrete implementation specifics. Verbatim review below.

---

I went and read the relevant bits of `batch_humo_render.py` and `video_composite.py`. Short version: agree with Option 1, agree with most of the second-opinion correction, two pushbacks, and a Question 2 path I think the round-robin missed.

## Bottom line

Yes, ship Option 1. The empirical evidence is strong enough on its own — radio still routed correctly, two unrelated generic faces emerge (`batch_humo_render.py:1195-1199` confirms the dispatch did its job). Whether the theoretical mechanism (`reference_latents` as soft attention hint, generation latent starts at zeros) is the full story matters less than the operational truth: HuMo's weights produce faces. You don't ship around that, you accept it.

The second-opinion correction about VideoComposite is the most important point in this whole exchange and the brief underweights it. I'll come back to that under Q5.

## Q1 — read of `WanHuMoImageToVideo`

The OTR-side code is doing what the brief says (`batch_humo_render.py:1466-1477` passes `ref_image` into the WanHuMoImageToVideo call as a kwarg, no other shaping). I haven't been given `comfy_extras/nodes_wan.py` source, but the snippet you quoted is the standard Wan2.1 I2V conditioning shape and matches what I've seen of that family. So: read is consistent with the code I can see, and consistent with the empirical output.

One honesty note worth surfacing: "the model's restriction to faces is in the weights, not the code" is probably right but isn't airtight without a reproducible test (e.g. ref_image of a radio with CFG swept 1→9, seeds varied, prompt stripped of "speaks", "face", "expressions"). The two-blonde-women observation is consistent with weights-driven face prior; it's also consistent with text-prompt dominance at this CFG scale washing out a weakly-weighted reference. Either way, the operational conclusion is the same — don't ship a fragile thing — but if anyone in the future wants to revisit, that's the experiment that would actually settle it. Not a blocker for Option 1.

## Q2 — fourth path the round-robin missed

There's one. Not generative video, but legitimately makes the radio "perform" deterministically: **ffmpeg audio-reactive filter passes over the static still.**

`showwaves`, `showspectrum`, `avectorscope`, and `showvolume` filters take the master audio and produce a per-frame visualization video. Compose that as an overlay on the radio still (small inset showing the oscilloscope traces "live", or a subtle needle-meter on the radio dial driven by RMS). Pure ffmpeg, frame-deterministic, runs on CPU, zero GPU contention with HuMo.

Example invocation skeleton (untested for your exact spec, but the filter graph shape):

```
-loop 1 -i radio_bookend.png -i master.wav \
-filter_complex "[1:a]showwaves=s=200x80:mode=line:rate=25[wv]; \
                 [0:v][wv]overlay=x=...:y=...:shortest=1" \
-frames:v <round(dur_s*25)> -r 25 -c:v libx264 -pix_fmt yuv420p -an out.mp4
```

This isn't a replacement for Option 1's routing rule — it's an enrichment of the static-video path that makes non-dialogue segments feel motion-alive without adding a model. It also matches the OTR aesthetic better than a frozen still: 1940s radios had needle meters and oscilloscope-style indicators that responded to signal level. So the visual idiom is correct to the period.

Worth investigating after Option 1 lands. Don't block on it.

## Q3 — C7 mux pattern safety

Gemini's pattern is correct AND **it's already what `_render_master_mix_per_clip_mux_mode` does.** Read `video_composite.py:552-630`:

1. Pillarbox each HuMo clip with `-an` (line 438) → silent segments
2. Concat-demux with `-c copy` (lines 602-610) → `silent_combined.mp4`
3. Single final mux: silent_combined + procgen with `-c:v copy -c:a copy -shortest` (lines 619-630)

So the static-video segments just need to slot into step 1 as additional silent inputs. The mux pattern is already in place.

Specific gotchas I'd preempt with concrete invocations rather than leaving to discovery:

**Timebase mismatch.** ffmpeg's concat-demuxer with `-c copy` is intolerant of mismatched container timebases. HuMo clips will have whatever timebase ffmpeg picked at save time (probably tbn=12800); a `-loop 1 -i still.png` static clip will default to tbn=15360 or similar. They'll concat but you may see frame-timing drift. Force it explicitly on every segment:

```
-video_track_timescale 12800
```

Same value on HuMo clips and static clips.

**Frame count rounding.** `-t <dur_s>` rounds based on timestamps and can produce off-by-one. For static segments where you know exactly how many frames you want, prefer:

```
-frames:v <int(round(dur_s * 25))>
```

This makes per-segment frame count exactly deterministic and per-clip drift bounded to ±0.5 frame at the seam, not accumulating.

**Audio source.** The "current per-clip mux copies audio from the procgen video, not directly from a WAV master" point in the second opinion is correct (line 622: `-i procgen` then `-map 1:a`) — but I'd push back on calling that a C7 violation. Your C7 contract per the docstring at lines 477-478 is "0 audio re-encodes downstream of SignalLostVideo's CreateVideo step." That is, byte-identity is measured from procgen.mp4's audio stream, not from a separate WAV baseline. `-c:a copy` from procgen does preserve byte-identity downstream of procgen. The hash-the-extracted-audio test is still a good sanity check (and you should run it once to verify) but the framing in the second opinion oversells the risk.

**`-shortest` truncation (BUG-128).** This is the real audio-loss bug. Currently the tail-pad in per_clip_mux (lines 559-563) is hardcoded to the original timeline's last index. If `pb_failures` pruned the genuine last clip, the surviving last clip has no tail pad, `silent_combined` ends a few frames before procgen audio ends, `-shortest` truncates trailing audio. Fix: compute `last_idx` from `pillarboxed[-1]` after the loop, not before. Or simpler: drop `-shortest` and post-trim with ffprobe-measured durations.

## Q4 — concurrent SFX

The plan is right: SFX overlaid on dialogue should never cause a visual cut. Stay on the speaking character's HuMo clip, mark `is_concurrent_with_dialogue=True`, static-video path skips it.

One refinement: I'd surface this in the schema rather than just the flag. Distinguish two kinds of `ledger.lines[]` entries:

* **anchor lines** — drive a video segment (one clip per range; HuMo for dialogue/announcer, static for music/SFX-standalone)
* **overlay lines** — audio-only events folded into the master mix; never drive video; never become a render target

Anything that walks `ledger.lines[]` to construct a video timeline iterates only over anchor lines. Mix construction iterates over both. This makes the contract explicit and resistant to a future regression where someone double-counts overlay lines into the video timeline.

If schema churn is too expensive at v2.0-alpha, the boolean flag works — just be aware it's a load-bearing flag and every consumer needs to honor it.

## Q5 — sequencing

I'd flip it. Land BUG-128 (mux) first, then BUG-129 (routing). Three reasons:

1. BUG-128 is a contained patch to one function (`_render_master_mix_per_clip_mux_mode`). Small blast radius, easy to test in isolation. Once landed, every downstream change inherits the fixed mux baseline.
2. BUG-129 is multi-node (BatchHumoRender role policy + VideoComposite static-segment generation + possibly SceneSequencer schema). Bigger surface. You want the audio path provably clean before you start changing the video routing on top of it.
3. If you land BUG-129 first and there's still a sync issue in the resulting episode, you can't cleanly attribute it — routing change vs. latent mux truncation are mixed.

Counter-argument that's defensible: BUG-129 is the visible problem (your eyes see two blondes; nobody sees the audio bytes). If product correctness ranks above engineering hygiene, ship the visible fix first. That's a real call; not strictly wrong. But from a pure risk-isolation standpoint, mux first.

One thing both bugs share: neither is fully fixed without the VideoComposite contract change the second opinion flagged. Right now per_clip_mux at `video_composite.py:511-519` silently skips clips not on disk. If BatchHumoRender stops rendering music/SFX clips (BUG-129's BatchHumoRender half), per_clip_mux will silently skip those rows, the video timeline collapses (concat with no gap-filling), and the master audio retains its full duration. Result: cumulative audio-leads-video drift across the episode. The static-video helper isn't optional — it's the integral half of BUG-129 that makes the other half safe.

So actually a cleaner sequencing might be **three commits not two**:

1. **BUG-128**: tail-pad on actual surviving last clip + verify `-c:a copy` byte-identity with a hash test.
2. **BUG-129a (VideoComposite)**: Add static-segment generation for non-character ledger rows. Keep BatchHumoRender unchanged — it'll still render HuMo for everything. VideoComposite now CAN handle missing HuMo clips by generating a static segment instead of skipping. Test against current renders; should produce identical output (no missing clips means static path doesn't fire).
3. **BUG-129b (BatchHumoRender)**: Stop rendering HuMo for music/SFX/announcer-as-radio; flip ANNOUNCER to a real cast member. Now VideoComposite's static-segment path actually fires.

This makes each commit independently revertible and each one provably safe at land time. The order of 129a-then-129b means at no point is there a "broken between commits" state where lines have no video coverage.

## What I'd actually argue with in the second opinion

Two things, both small.

**The C7 framing.** "MP4 + copied AAC from procgen is not the same thing as byte-identical to a 48 kHz mono WAV baseline" — true at face value, but your C7 rule per docstring is byte-identity downstream of procgen, not vs. a WAV master. If the canonical baseline really is a WAV, that's worth surfacing as a separate concern, but it's a different rule from what's currently coded. Don't conflate.

**Calling Option 1 "not done until VideoComposite is updated".** Correct as a code-correctness statement, slightly understated as a risk statement. Half-done Option 1 doesn't just leave timeline-coverage gaps — it produces visually-shorter-than-audio output that's then time-compressed by `-shortest`, so every dialogue clip will lip-sync drift. This is louder than "you'll have a coverage problem." That's why I want VideoComposite landed before BatchHumoRender role changes.

So: agree with Option 1, agree with the second opinion's correction (and would emphasize it harder), suggest mux-first sequencing in three commits not two, and suggest the audio-reactive ffmpeg overlay as a Q2 enrichment after the architectural commit lands. The blonde-women problem is real and Option 1 is the right answer to it.
