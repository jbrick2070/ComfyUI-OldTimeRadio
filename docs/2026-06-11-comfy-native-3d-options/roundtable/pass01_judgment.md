# Pass-01 judgment — comfy-native 3D options (Claude judge; 3-model panel + Claude panelist)

Panel: openai/gpt-5.5-20260423, google/gemini-3.1-pro-preview-20260219,
x-ai/grok-4.3-20260430, + review_claude_panelist.md (written before reading the panel).
Spend: ~$0.12. Operator pin honored: ADDITIVE ONLY — nothing leaves the 3D plan; the
toolkit lane (TripoSG/ARKit/Rhubarb/Blender; hunyuan3d_talk/trellis_talk scaffolds) is
untouched. This roundtable only picks the easiest extra on-ramp.

## Convergence call: CONVERGED after one pass
All four reviewers independently ranked the same winner (Class F: camera-orbit preset on
an engine we already run) and the same runner-up (Class A: depth-parallax), and the same
NO-GO set for a first test (C/D/E; B not-first). GPT's must-fixes are "turn the question
into a build contract" items — folded into the synthesis. No unresolved material dispute
remains, so a second pass would buy accents, not facts.

## Grounded claim log (CONFIRMED / MISREAD / UNVERIFIABLE)

- CONFIRMED (all panel + repo): splat/TRELLIS rasterizers + flash-attn class deps have no
  sm_120/cu130/Win wheels; registry already pins hunyuan3d_talk/trellis_talk to
  required_toolchain=cu128_toolkit. Class D NO-GO for the first test (toolkit lane covers
  real 3D later, unchanged).
- CONFIRMED (Gemini + GPT + Claude): mesh -> mp4 needs an offscreen renderer ComfyUI does
  not ship (Load 3D = interactive preview). Gemini overstates "impossible without compiled
  rasterizer" (pyrender/OpenGL is wheel-only) but the conclusion stands: NOT the easiest
  path. Class B = "first real 3D ASSET on-ramp, later" with pyrender as verify-at-build.
- CONFIRMED (Gemini): SV3D/Zero123 class = square low-res output (canvas-contract
  violation) + NC-tier licenses. Class C NO-GO.
- CONFIRMED (repo): schemas.FAMILIES already has `static_motion`; the parallax engine
  reuses it (Gemini/Grok) — no new family token (rejects my pass00 hedge and Grok's
  optional `static_motion_3d`).
- CONFIRMED (repo, GPT): cheap-family default canvas is 832x480 when request canvas absent
  — new adapters must set 1472x832 + target_frame_count explicitly.
- CONFIRMED (GPT): flux_still rides flux.1-dev (known non-commercial-grade license,
  pipeline-wide, pre-existing) — label, not a new blocker.
- PARTIAL-MISREAD (Gemini): "DepthAnythingV2 is CC-BY-NC" — only Base/Large/Giant are;
  DA-V2-SMALL is Apache-2.0 (Claude panelist). Fix kept: pin the Small variant (or
  Marigold/DA-V1, both Apache) and record the choice in model_requirements.
- MISREAD (Gemini): "torch 2.10 does not exist (latest 2.6)" — stale model knowledge; the
  machine runs torch 2.10.0+cu130 today. Discarded; does not change wheel conclusions.
- MISREAD-as-blocker (Grok): "registry contradiction" for Class B — hunyuan3d_talk's
  cu128 row is the TALKING toolkit lane, not core hy3d-2mv mesh-gen; no contradiction.
  His fix (a new row would need required_toolchain=None + an existence-gated adapter) is
  still the right shape if B is ever built.
- MISREAD-ish (GPT #8): "no other_beats_visual role on cheap adapters" — the
  slot->role mapping exists one layer up (sweep SLOTS maps other_beats_visual ->
  character_video; profile layer resolves it). Kept as a wiring reminder only.
- UNVERIFIABLE -> verify-at-build: existence/quality of a specific PUBLISHED camera-orbit
  LoRA for LTX at our resolution (Grok demanded an exact filename — right; v0 therefore
  ships prompt-only orbit with LoRA as an optional v1 upgrade); Trellis2 wrapper "wheels"
  claim vs our stack (expect NO-GO).

## Rejected panel items (with reason)
- GPT "cut cloud-lane footnotes from the build spec": kept as ONE labelled line in the
  recommendation (operator awareness), excluded from the build contract — split the baby.
- Grok should-fix 1 "extend StillKenBurnsFamily instead of a new registry entry": the
  operator wants a visible dropdown option; a NEW engine id (still_parallax) in the same
  static_motion family gives the dropdown row without duplicating family code (GPT #6
  agrees). Rejected as written, honored in spirit (reuse the family base class).
- Gemini "kill Class B entirely": softened to "not-first; verify-at-build" — operator pin
  forbids removing 3D options, and B is the only no-toolchain path to a REAL 3D asset.
