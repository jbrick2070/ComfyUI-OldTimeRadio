# Carry-over problem statement for the VIDEO engine plan (lesson from the audio overhaul)

## Problem statement
The 2026-06-01 audio-engine plan was thorough on everything INSIDE the ComfyUI
graph (node contracts, byte-identity, fail-closed dispatch, determinism) but
UNDER-planned the integration reality of each model. Three gaps caused a cascade
of unplanned follow-up sprints for chatterbox + IndexTTS2:

1. **Outside-the-graph dependencies were deferred to a late "pilot," not designed
   up front.** Dependency conflicts were treated as a pass/fail check, not as an
   architecture fork. They turned out irreconcilable (chatterbox torch 2.6 /
   IndexTTS2 torch 2.8 vs the main cu130 venv), which forced an entirely
   unplanned architecture -- a per-engine isolated venv + subprocess worker +
   JSON bridge -- plus per-engine Blackwell torch (cu128), model-weight
   downloads, reference-audio sourcing, and commercial-licensing handling. None
   of that was a sprint in the plan; all of it became one later.

2. **Architectural principles were stated but not test-enforced.** The plan said
   "model-agnostic, no per-engine ladders," but the build hard-coded an
   engine-name tuple that violated the principle -- and undoing it became a whole
   refactor sprint.

3. **Library APIs were assumed ("assumed_call"), verified only at a final
   pilot.** So the real behaviors (torchaudio.save routing through torchcodec,
   exact generate() kwargs, Dia's transcript requirement) surfaced as RUNTIME
   bugs after the build, not during planning.

## How it was fixed (2026-06-05)
Built the Path-B isolated-venv sidecar pattern for both engines; landed the
adapter-metadata refactor AND added tests that fail if dispatch hard-codes an
engine name; wired the delivery vector end-to-end; sourced + mirrored the
reference bank; and ran a LIVE GPU smoke that caught the real bug
(torchaudio.save -> soundfile) before shipping. Result: chatterbox proven
rendering on the RTX 5080; full suite 3786/0, Bug Bible green.

## MANDATE for the video planner (bake into the INITIAL plan -- do NOT defer)
1. **Cover ALL dependencies up front -- in-graph AND outside-the-graph.** For
   every model (HuMo, FLUX, LTX, ...): a dependency matrix against the main
   cu130 venv, the in-graph-vs-isolated-sidecar decision, the Blackwell
   torch/CUDA flavor (sm_120/cu128) + any save/load codec quirks, model-weight +
   conditioning-data sourcing, and commercial licensing -- each a NAMED sprint,
   not a "detail" or a late pilot.
2. **Every principle is enforced by a test.** If the plan says "model-agnostic /
   no hard-coded model names," ship a test from sprint 1 that FAILS when a model
   name is hard-coded in dispatch. A capability sprint is not "done" until it is
   wired into the live graph and validated end-to-end (no build-and-shelve).
3. **No assumptions -- verify on the real GPU early.** A tiny live
   one-frame/one-clip GPU smoke per model is an EARLY sprint, not a final pilot.
   Ban "assumed API" items: the docs lie, the GPU is the source of truth.
