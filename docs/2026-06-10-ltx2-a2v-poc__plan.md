# LTX-2 Audio-to-Video Proof of Concept -- small plan (2026-06-10)

## What we are testing

LTX-2 (Lightricks, Jan 2026) is a joint audio-visual model: 14B video + 5B
audio streams coupled by cross-attention. Its **audio-to-video (A2V) pipeline
takes an audio file as the PRIMARY conditioning** and generates matching
video -- the sound drives motion, pacing, and (per the docs) lip sync -- and
returns the original audio UNTOUCHED. The operator question: feed it one of
OUR per-beat voice slices and SEE what it actually does on this box. Maybe
real lip-sync, maybe just audio-stylized motion -- the PoC exists to find out,
not to assume.

Why it fits OTR if it works: "returns the original audio unmodified" is
exactly the V-1 frozen-master contract; FP8 reportedly runs in ~12GB (16GB
cards confirmed by testers); the OFFICIAL ComfyUI-LTX plugin ships an A2V
node, which likely retires the 2026-06-01 "Blackwell dep risk" decline.

## Hard scope (this is a PROBE, not an integration)

- NO OTR graph changes, NO new engine adapter yet, NO production-workflow
  edits. Scratch ComfyUI graph only (V-12: if the plugin's deps touch the
  main venv beyond ComfyUI-native, STOP and report -- sidecar question).
- The 13 unpushed production commits + the acceptance test stay the active
  mission; this runs AFTER the operator's eyeball, on operator GO.
- Spend: $0 (open weights, local).

## P0 -- install + smoke (one evening)

1. Install the official ComfyUI-LTX plugin (Manager -> "LTXVideo"); let the
   LTX-2 FP8 weights auto-download into C:\ComfyUI-Models (set the env so
   nothing lands on C: bare). Record exact versions + dep deltas.
2. Build the smallest A2V scratch graph: audio in -> A2V pipeline -> mp4 out.
3. Inputs: ONE existing per-beat slice from a rendered episode
   (otr\tmp slice cache or re-cut from a master, read-only) + a short text
   prompt ("a 1940s radio actor speaking, head and shoulders, warm light").
4. Render ~5s at 720p FP8. Capture: peak machine-NVML, wall time, and
   whether the process stays under the 14.5GB ceiling.

GATES: weights verified on disk; one clip renders without OOM; the output
mp4's audio stream is byte-identical to the input slice (prove the
"unmodified audio" claim with a hash).

## P1 -- the eyeball matrix (same evening)

Render 4 clips, same seed, varying ONE thing each:
  a) voice slice + face-forward prompt  -> does the MOUTH track the words?
  b) voice slice + wide scene prompt    -> does motion/pacing still follow audio?
  c) music slice + scene prompt         -> does it cut/move on the beat?
  d) same prompt, NO audio (T2V)        -> the control: what does audio add?

Operator eyeballs a/b/c against d. Verdict vocabulary: LIPSYNC (mouths
track), STYLIZED (motion follows energy/pacing but not phonemes), or INERT
(audio changes nothing). Any of the three is a finding; none is a failure.

## P2 -- only on operator GO after P1

If LIPSYNC or STYLIZED: a new engine adapter `eng_ltx2_a2v` on the existing
MotionEngineBase pattern -- family audio_driven (or a new audio_stylized
family), roles per the verdict (character_video + announcer_visual if
LIPSYNC; scene/announcer/music if merely STYLIZED), required_inputs
(text_prompt, audio_ref), default-OFF behind OTR_ENABLE_LTX2_A2V, full-frame
canvas, the standard fallback chain, dropdown-selectable like every engine.
Sizing: ~1 coder session + the usual suite/Bug Bible/byte-identical gates.
If INERT: write the finding into docs/, close the lane, lose nothing.

## Risks / verify-at-build

- 14B+5B on 16GB: FP8 + offload claims need OUR measurement (P0 gate), and
  in-process co-residency with HuMo/FLUX is a later VRAM-lease question --
  the probe runs solo.
- "Accurate lip sync" is the vendor's claim at 4K/50fps cloud scale; local
  FP8 720p may degrade to STYLIZED -- that is precisely what P1 measures.
- Plugin dep hygiene on torch 2.10/cu130/sm_120: if it imports beyond
  ComfyUI-native ops, evaluate the cu128 sidecar pattern instead (the
  latentsync precedent).
- 20-second clip claims (LTX-2.3) vs our ~4-10s beats: beats fit easily.

## References (research 2026-06-10)

arxiv.org/abs/2601.03233 (LTX-2 paper); ltx.io/model/capabilities/audio-to-video
(A2V); ltx.io/blog/a-guide-to-ltx-2-3-audio (audio sync); the ComfyUI local-run
+ GGUF guides (codersera.com, dev.to) for the 16GB FP8 recipes.
