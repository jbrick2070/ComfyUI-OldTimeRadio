# PASS 01 REVIEW FOCUS: ARCHITECTURE

You are one panelist in an adversarial review of the plan below. THIS pass is
the ARCHITECTURE pass. Confine your must-fix findings to architecture; note
other issues in one line each at most.

Pressure-test exactly these:
1. Family token (Q1): new `audio_conditioned_video` family vs reusing
   `audio_driven_face` vs TWO adapters (talking-head vs music-reactive)
   sharing one engine core. Judge against the registry protocol and
   role_compat in the grounding. One engine spanning lip-sync AND
   music-reactive motion is the design tension -- is it a smell?
2. Isolation (Q4): in-process (the eng_ltx_video precedent) vs cu128 sidecar
   (the latentsync precedent). What is the explicit STOP rule if the LTX-2.3
   audio-conditioned path drags deps beyond ComfyUI-native into the cu130
   venv (V-12)?
3. Fallback chain (Q8): ltx_av -> humo -> latentsync -> still_kenburns is 4
   hops and humo pillarboxes a portrait while ltx_av renders full-frame
   landscape -- is a mid-chain ASPECT change an acceptable LOUD degrade, or
   should the chain be ltx_av -> still_kenburns direct? Check
   fallback.py's cycle/hop machinery in the grounding.
4. Adapter shape: does MotionEngineBase fit an audio-conditioned engine
   without mutation? Does anything in the render lifecycle
   (assert_usable/prepare/render_clip/canonicalize/teardown) need a NEW
   member for audio conditioning, and is that additive?
5. The Yvann-Nodes zero-new-model lane: parallel cheap lane for music_visual
   only, or distraction? Decide and defend in 5 lines.
6. Registry/schemas touch list: is the additive file list complete and
   minimal? Reject any delta that mutates existing engine behavior.

Rules: every claim about the code must cite the grounding excerpts; flag
anything you cannot verify as VERIFY-AT-BUILD; do not parrot the corrected
dims claim ("/32+1" is wrong for W/H); the V-1 audio invariant and the 14.5GB
NVML ceiling are non-negotiable; eng_ltx_video.py and the 13 unpushed commits
are out of scope for edits. Output: numbered MUST-FIX items (with the file
they touch), then SHOULD-CONSIDER, then OPEN-QUESTIONS. Be specific and
terse.
