# EXECUTIVE DECISION: the video path forward

**Judge: Claude (Opus 5), 2026-08-02.** Inputs: four kibitz rounds, an archive
sweep, a Fable quality ruling, and the external deep-research report
(`2026-08-02-HUMO-LTX-DEEP-RESEARCH-REPORT.md`). This is a decision, not a menu.

## THE HEADLINE: the research disproved the plan we were about to build

Fable ranked the lip-sync pad first and codex specified it down to the frame
count (4 frames / 160 ms, pre-pad the audio, strip `pad_frames` from the decoded
batch). **We must not build that yet**, because the research shows the fix only
works for one of two possible defects, and we have never determined which one we
have:

* **Onset-only cold start** -> pre-roll + equal trim works.
* **Constant lag** -> pre-roll + equal trim is **algebraically a no-op**.
  Pre-padding by `p` and trimming `p` leaves exactly the same delay `d`.

So the specified fix has a coin-flip chance of costing render time and changing
nothing. **Measurement gates the fix.** That is the single most valuable thing
this research bought us, and it cost zero GPU hours.

## WHAT IS NOW SETTLED -- stop debating these

| Question | Answer | Grade |
|---|---|---|
| Does orientation change VRAM at equal pixels? | **No.** 480x832 and 832x480 both give a 30x52 / 52x30 DiT grid = **1,560 tokens per latent time**. Square patching `[1,2,2]`, global attention window `[-1,-1]`, RoPE reshaped to `f*h*w`. | A |
| Is the 49-vs-177 orientation split defensible? | **No.** No architectural basis, no published counterexample. Delete the orientation rule. | A |
| Does that make 177 safe? | **No.** The portrait 177 never had a peak trace either. **Both** numbers are unqualified. | A |
| Can HuMo segments be frame-chained? | **No.** The reference is appended to the END of the sequence *specifically* to prevent first-frame continuation. JUMP is correct. | A |
| Is our 33-frame floor a model law? | **No.** The lattice is `1+4k` from 1. 33 is our own policy. | A |
| Fixed seed per beat for identity? | **Wrong.** Seed gives repeatability, not continuity, and a repeated seed makes the pose reset MORE visibly repetitive. Vary it per segment. | A |
| IP-Adapter / PuLID / InstantID for identity? | **Incompatible.** Those are SD1.5/SDXL/FLUX-family. HuMo already has native reference conditioning and we already feed it. | A |
| Does flat-to-81 license flat-to-177? | **No.** Kijai's HuMo `tiled_decode` is SPATIAL -- it hands all temporal latents to each tile. `vae_temporal=16` bounds nothing unless the path genuinely chunks latent time. | A/D |
| Cut placement at silences? | **Solved, locally.** Silero VAD (MIT, offline, CPU) + forced alignment; cut at the silence midpoint nearest a 25 fps boundary. | A |

**And a provenance lead worth recording:** the phantom "~15.9 GB" closely matches
the native Comfy HuMo fp8 checkpoint's ~17.1 GB **file size on disk** expressed
as GiB. A disk-weight figure may have been copied into a VRAM table. Unprovable
now that the source document is gone -- but it fits, and it is the best
explanation anyone has offered for a number that gated a production engine.

## WHAT IS STILL UNKNOWN -- and only our GPU can answer

1. The **actual safe frame cap** for HuMo-14B fp8 on this box. Not published for
   this checkpoint at either orientation, by anyone.
2. Whether our lip-sync error is **onset-only or constant**. Gates the fix.
3. **LTX-2.3 449-frame** feasibility under 14.5 GB. Not published.

## THE DECISION

### Phase 1 -- code, no GPU (do this first, it makes Phase 2 safe)

1. **Wire the admission guard.** One boundary in `render_driver`, before
   `BeatSession.prepare()`, reading pre-load free VRAM once. A missing cost row
   FAILS qualification rather than falling back to `_DEFAULT_FRAME_COST`. This is
   what makes a frame-ladder test safe to run at all -- without it the ladder is
   an OOM generator.
2. **Fix the one-segment coverage bypass.** A stamped one-segment plan carrying
   `trim_tail` currently skips coverage execution entirely.
3. **Delete the orientation-specific HuMo cap** and set BOTH 14B routes to the
   same conservative value pending Phase 2. Do not encode 177 by inference; the
   research explicitly refuses to license it.
4. **Collapse cap authority to one:** ledger-stamped `video.max_render_frames`;
   env twins must be absent or equal before planning; `render.frame_budget` is
   diagnostic only.
5. **Vary the video seed per segment** -- `(shot_seed, segment_index)` for N>0,
   preserving the old seed for single-segment shots. Research-supported, and it
   fixes the stale "different seed" rationale in `mouth_policy.py:144` at the
   same time.
6. **Fix the campaign roster and acceptance.** It runs six engines while claiming
   all local ones, and its acceptance would not reject a mirror.
7. **Correct the record:** the 33 floor is our policy not a model law; the stale
   mirror-extend comment at `eng_humo.py:61`; the `word_razzle` matrix row.

**Do not ship the 65 cap.** It derives from a cost row that is itself
unqualified.

### Phase 2 -- three measurements, in this order

**M1. Lip-sync classification (cheapest, gates the biggest win).**
Generate clear frontal lines with sharp plosive onsets, mux a zero-based
CFR-25 / 16 kHz diagnostic, and read SyncNet offset in **early, middle and late**
speech windows -- after validating SyncNet's sign on a deliberately shifted
control clip. Early-only error means Case A (pre-roll + trim). Equal error
throughout means Case B (advance the 25 Hz conditioning features by `d`). A
growing error means a rate/timestamp bug, not a pad.
Run a matched no-LoRA control: Kijai reports the Lightx2v distill is **not fully
compatible with HuMo**, so the defect may belong to our LoRA, not to HuMo.

**M2. Paired orientation ladder.** `49 -> 65 -> 81 -> 97` in BOTH orientations,
everything else identical, cold and warm recorded separately (Windows compile
caches make first and second runs different facts). Per-phase peaks after
Whisper, after DiT sampling, after VAE decode -- one end-of-run number cannot
identify the scaling phase. NVML high-frequency sampling, not point-in-time
`nvidia-smi`.
**Predicted result: the two orientations match at every rung.** If they do not,
we have a real bug in tiling or kernel selection and that becomes the next
investigation. Stop at the first rung that breaches 14.5 GiB cold.
**97 is the quality-supported target** -- HuMo was trained at 97 frames and its
authors warn longer generation degrades. A memory pass at 125+ is not a quality
pass.

**M3. LTX-2.3 449 frames** at the lowest supported canvas first, per-phase peaks,
with a host-RAM trace (offloading a 29-46 GB checkpoint can fail outside VRAM).
Check first whether our graph is one-stage or the official two-stage path: the
two-stage helper requires /64 dimensions, and **neither 832x480 nor 512x288 is
valid for it.**

### Phase 3

Implement the correction M1 selects. Set the single HuMo cap from M2. Then run
the 30-45 word randomizer across every local engine, with a recorded random seed
and the real acceptance test.

## WHAT THIS CHANGES ABOUT FABLE'S RANKING

Fable's judgment holds where it was a judgment: identity drift is worse than pose
reset, ten cuts is a stutter rather than a style, and the defects concentrate on
the one face beat per episode. The research confirms the mechanism behind the
snapback -- fresh noise every call, no motion phase carried, similar openings
from a repeated reference.

What changes is only the **order**: the lip-sync fix cannot be built before it is
classified. Fable ranked by audience impact and was right to; the research added
a precondition Fable had no way to know about. Its second item -- requalify the
HuMo cap -- is now the best-supported action in the whole plan, because we know
in advance what the answer should look like.

## THE ONE-LINE VERSION

**Measure two things (is the lip-sync error constant, and where does the frame
ladder actually break), fix the guard and the coverage bypass while the GPU is
idle, and stop treating any cap in this repo as qualified until it has a trace
attached.**
