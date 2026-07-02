VERDICT: no. The concept is coherent, but the plan hides unpinned cloud dependencies, feeds Kling lipsync the wrong kind of input, and expands F2 beyond the agreed no-rig 3D POC.

MUST-FIX BEFORE BUILD:
1. [§2 F1-a / §7] F1 says it needs “S1 + kling row only,” but F1-a depends on “LTX 2.3 outpaint cloud template” (C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/kibitz-runs/2026-07-02-creative-formats/r1/input.md:44-51). The cloud S1 final rows are only recraft / flux_pro / nano_banana_2 (C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md:176-180), and the checked-in pin has no LTX row. Concrete fix: either add the LTX outpaint adapter/pin as an explicit F1 prerequisite, or rewrite F1 board mint to use only pinned S1 still rows.

2. [§2 F1-c / §1 prereq 3] F1-c implies a still polaroid crop can go directly to `kling_lipsync` (input.md:55-59), but the pinned `KlingLipSyncAudioToVideoNode` requires `audio`, `video`, and `voice_language`, not an image crop (C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_shared/partner_nodes.yaml:135-164). Concrete fix: add an explicit local still-crop -> silent base video clip step before Kling, or use a different pinned node that accepts image+audio.

3. [§3 F2-a / §1 prereq 2] F2-a depends on IdeogramV4 (input.md:70-73), but the current cloud rows do not include Ideogram; S1 final rows are recraft / flux_pro / nano_banana_2 (pass04_plan.md:176-180). Concrete fix: either make Ideogram a new pinned concept-stills row with pricing/license/cache coverage, or make `tin_toy_v1` run on an already-pinned S1 still engine.

4. [§3 F2-b / §4 V2] F2 reopens rigging/animation as part of the first build (input.md:74-81, 97-98), contradicting the current 3D POC appendix, which explicitly says `NO rigging` and “Rig/animate (Meshy) stays future-lane” (pass04_plan.md:247-254). Concrete fix: cut MeshyRigModelNode/MeshyAnimateModelNode from F2 MVP; render static/turntable/dolly GLB plates in Blender first, then treat rigged idle as a later phase after V2.

5. [§2 F1-d / §3 F2-e] The format switch is architecturally undecided: “widget on the VideoDirector policy or an episode-level toggle -- decide at wiring round” (input.md:60-63, 85-86). Current `OTR_VideoDirector` policy emits `video_models`, `image_models`, canvas, seed, fallback, and duration, but no visual-format field (C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_video_director.py:325-342). Concrete fix: choose the routing model now: either formats are selectable engine ids per role, or append one explicit `visual_format` widget/policy field and route all renderers from it.

6. [§2 F1-a] The board cache story contradicts “per-episode board assembly”: the board is keyed by “board prompt + cast portrait hashes” and “re-billed only when the cast changes” (input.md:46-51). If pinned clues are episode-specific, same-cast episodes can incorrectly reuse the wrong board. Concrete fix: split static cast-board cache from episode-evidence overlay, or include an `episode_evidence_hash` / clue hash in the request key.

SHOULD-FIX:
1. [§4 V4 / §5] V4 checks only crop/paste coordinate landing within +/-2px (input.md:99-102), while §5 claims portrait-hash identity remains the source for both formats (input.md:106-113). Concrete fix: add an identity/face-similarity acceptance check on the post-Kling pasted crop, not just geometry.

2. [§6] Cost posture excludes the unpinned/new rows it relies on: LTX outpaint, Ideogram, and the 3D adapter chain (input.md:117-121). Concrete fix: add format-specific estimate rows only after those rows are explicitly pinned/priced, or mark the current estimates as [ASSUMPTION].

3. [§4 V1] The Tin-Toy kill-switch tests only whether Kling forces skin texture onto metal (input.md:93-96). [ASSUMPTION] The bigger user-facing risk is whether small painted-metal mouth motion is readable at actual shot size. Concrete fix: add a readability probe at planned crop/full-frame sizes.

OPTIONAL / NICE-TO-HAVE:
- Add one explicit “golden 30-second sample” acceptance for each format before full-episode wiring.
- Record the exact canonical asset types for board PNG, crop base clip, Kling crop output, pasted segment, and final silent composite.

CUT THESE (scope / over-engineering):
1. [§3 F2-b / §4 V2] Cut rig + idle animation from F2 MVP. It is safe because pass04 already defines a no-rig 3D POC path and keeps Meshy rig/animate future-lane (pass04_plan.md:247-254).

2. [§3 F2-e] Cut “whole episode” Tin-Toy mode from the first build. Character-beat style is the stated goal (input.md:20-25, 85-86); whole-episode mode multiplies routing and cache cases before the mouth/mesh look is proven.

3. [§2 F1-a] Cut 8K as a hard first target. Keep the board concept, but prove stitch/crop/lipsync at the real render canvas first; 8K outpaint is an external dependency and cost multiplier until the LTX row is pinned.