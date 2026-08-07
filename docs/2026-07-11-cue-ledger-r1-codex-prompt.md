# R1 prompt -- Codex (GPT-5.6 sol): broad architecture pass on the Timeline Cue Ledger

**RETIRED 2026-08-06** with its parent design (`2026-07-11-timeline-cue-ledger.md`)
under the operator's "rip out SFX 100%" ruling -- see
`docs/2026-08-06-BUILD-SPEC-rip-sfx.md`. Historical record only; the code
symbols it cites (e.g. `compile_sfx_bed_from_manifest`) no longer exist.

Paste everything below the line into Codex. It is an **R1 pass**: high-level arc, architecture,
alternatives. Not a coding plan, not wiring -- those are R2/R3.

---

You are doing an **R1 architecture review** of a design roadmap for OTR, a local ComfyUI custom-node
pipeline that generates complete old-time-radio-style episodes end to end. R1 means: **step back and
attack the shape of the idea.** Do not write code, do not produce a sprint plan, do not do wiring. Later
rounds do that. Your job is to find the wrong *approach*, not the wrong *line*.

## Repo

`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`

Read these before you say anything:

- `docs/2026-07-11-timeline-cue-ledger.md` -- **the plan under review**
- `CLAUDE.md` -- operating rules. Note especially: the canonical workflow
  (`workflows/otr_canonical.json`) is the source of truth and unwired code is dead code; never "dummy";
  fix at root cause, no shims.
- `nodes/scene_sequencer.py` -- `SceneSequencer.sequence()` (~625-1170) and `OTR_EpisodeAssembler`
  (~1174-1891). This is where the plan wants to cut.
- `nodes/otr_master_audio_mux.py` -- the terminal mux; `compile_sfx_bed_from_manifest`.
- `nodes/_otr_voice_node_common.py` (~382-629) -- the per-line TTS render loop.
- `nodes/otr_shot_lock.py` -- clip budget / frame math on the final audio timeline.
- `workflows/otr_canonical.json` -- the real graph.

**Ground every claim in the actual files, with `file:line`.** If you assert something about the code,
you must have read it. Prior review passes on this doc were burned by confident claims about code that
did not exist -- one of them argued from pacing constants that are assigned and never read. Do not
repeat that. If you are unsure, say "unverified."

## What the pipeline does

LLM writes a script -> per-line TTS (in-memory AUDIO batches) -> `SceneSequencer` concatenates dialogue +
music cues + a room-tone bed into one waveform -> `AudioEnhance` (period/tape DSP) -> `EpisodeAssembler`
prepends/appends themes, runs master loudness, writes `<ep>_master.wav` -> video chain renders against
that master (shot budgets computed from cumulative audio samples) -> `MasterAudioMux` muxes and publishes.

## The problem the plan solves

**Sound effects.** Two prior attempts failed:

1. **Pre-render SFX** -- cues authored as pseudo-dialogue rows *before any audio existed*. The writer had
   to guess where a sound would land in a performance it had never heard. Ripped; a guard test now fails
   loud if it returns.
2. **Byproduct SFX (current state)** -- the cloud video engine happens to return video-with-audio; that
   stem is extracted and mixed as an episode-length bed. **Nobody chose those sounds.**

Root cause of both: *you cannot place a sound against a performance that does not exist yet.*

## The plan's core bet (attack this)

Invert the order: **render the episode, transcribe/align it, let an LLM spot cues against the real
performance, then re-cut the timeline to make room for them -- before video renders.**

Specifically:
- Persist per-line TTS stems (they are currently written to temp and deleted); force-align them to get a
  word-and-gap map.
- **Two-pass spotting.** Pass 1 = intent: reads the script, emits *events in the fiction* with a dramatic
  function (`establish | punctuate | story_beat`) and explicit deliberate silences. **Sees no timecodes.**
  Pass 2 = placement: mechanical, anchors events to words/gaps, **forbidden from adding events.**
- **The LLM never authors a timecode.** A deterministic resolver does.
- **Splice, don't duck.** Rather than overlaying effects on finished dialogue, *carve a real gap* so the
  effect has its own "nest" -- via one new `OTR_CueSplicer` node at the raw scene-audio seam (between
  `SceneSequencer` and `AudioEnhance`), where audio is pre-loudness/pre-theme and the ledger is still in
  scene-audio space. Video renders afterward, so frame budgets absorb the lengthened timeline for free.
- **Curated CC0 period SFX library**, not per-cue generation -- the same door every slam *is the show's
  door*; generation gives a different door every time.
- Cue decisions therefore see **no picture at all** (the clip manifest and ShotLock are both downstream).
  Claimed to be fine because the picture is generated *from the same brief* the cue director reads.

## The named danger

**The noun detector:** an LLM reading a transcript cues a door sound every time someone says "door." Real
radio effects were dramaturgical stage directions -- events in the fiction -- not naturalistic
sync-to-word. The plan's whole gate (C1) is a blind A/B against a budget-matched noun-detector control
arm: if the operator cannot tell them apart, the project **stops**.

## Your R1 questions

1. **Is the inversion the right frame at all?** Is "render, then spot" actually the correct decomposition,
   or is there a third shape neither of the two failed attempts nor this plan has considered? Think about
   what a *human* radio production actually does -- rehearsal, marked scripts, a sound man in the room --
   and whether some structurally different mapping onto this pipeline is better than post-hoc spotting.
   Divergent thinking is the point of this round.

2. **Is the splice sound, or is it a trap?** Carving gaps changes episode duration, which changes video
   frame budgets, which changes render cost. Is "splice at the raw seam, let everything downstream
   recompute" genuinely free, or is there a load-bearing assumption that breaks? Is there a cheaper
   mechanism that gets the same perceptual result (an effect that isn't masked by dialogue)?

3. **Is the two-pass, timestamp-blind design actually a defense, or theater?** Pass 1 still reads the
   script -- the nouns are right there in the text. Blindness protects *placement*, not *selection*. Is
   there a stronger structural defense against the noun detector than a budget + a blind A/B?

4. **Is C1 the right gate, and can it be gamed?** It is the single point where this project is supposed to
   be able to die. Does it actually have the power to kill a bad idea, or will it rubber-stamp?

5. **Library vs generation.** Is committing to ~30-60 curated CC0 one-shots right for a 1940s show, or is
   that a false economy that caps the ceiling? What breaks when an episode needs a sound the library
   doesn't have?

6. **No picture context.** The plan argues cue decisions need no picture because both lanes are
   conditioned on the same brief. Is that reasoning sound, or a rationalization of a wiring constraint?

7. **What is the plan not thinking about?** Sequencing, failure modes, what happens on a bad align, what
   an operator does when a cue is wrong, how this interacts with an imminent release. Anything the doc is
   silent on that a senior architect would flag before a line of code is written.

## Output

A direct critique. **Lead with anything that changes the plan's shape.** Concrete alternatives, not
praise. If a section is right, say so in one line and move on. Rank your findings by whether they would
change what gets built. Flag explicitly anything you could not verify in the code.
