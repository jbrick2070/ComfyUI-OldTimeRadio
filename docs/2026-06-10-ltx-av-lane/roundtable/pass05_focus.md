# PASS 05 REVIEW FOCUS: COMFYUI-NATIVE TESTING

You are one panelist in an adversarial review of the plan below. THIS pass
is the TESTING pass. Pass01-04 decisions are LOCKED -- one-line flags only.

The suite is currently 3815/0 with Bug Bible green at every commit; the
new lane must land the same way: every M1-M3 commit keeps the FULL suite
green, byte-identical green, and the engine dark by default.

Pressure-test exactly these against the grounding (existing test patterns
are in the grounding files -- mirror them, do not invent new frameworks):

1. TEST ENUMERATION: produce the DEFINITIVE list of new test files /
   cases for M1-M3, each with (a) what it proves, (b) the pattern file it
   mirrors. Cover at minimum: registry/dropdown additive presence; role
   compat (music audio_ref supply; talk slot fit); schema family
   round-trip; av_dims unit cases (W/H/frames, nearest-valid hints,
   1472x832 passes, 1450x832 raises, frames snap-up cases incl. T=25,
   T=26, cap); chain termination (5-hop talk chain, 3-hop music chain,
   trail retention); dark-lane GOLDEN FIXTURES (existing-engine requests
   bit-identical with the lane registered-but-dark); flag-off render
   -time degrade; force-map role guard; announcer portrait alias; the
   fake-AV-mp4 strip + zero-audio-stream ffprobe assert; AST
   no-brief-import; cold-import (V-12) with the new module; identity
   stamps in CanonicalClip/manifest; pad-tail marker emission;
   _render_one request_template pass-through (TypeError guard).
2. EXISTING-TEST FALLOUT: which existing tests MUST change when two
   engines register (engine counts, dropdown enumerations, fallback
   -chain sweeps, ENGINE_FAMILY assertions, b7 forbidden sweep)? Name
   the files from grounding where possible; the b7 sweep's AST loop var
   must be `imp` (repo gotcha). What existing test would FAIL TODAY if
   the coder forgets each touch-list edit (one per edit -- the "forgot
   it" detector matrix)?
3. GPU-GATED VS CPU TESTS: the suite runs headless/CPU; HuMo/LTX forwards
   are GPU-smoke scripts, not pytest. Define the exact split for the new
   lane: what is CPU-provable (everything above) vs what lives in the M0
   /M4 GPU scripts (real render, NVML ceiling, wall time, lip-sync
   eyeball). Should the M0 sheet be a CHECKED-IN artifact (e.g.
   docs/.../M0_RESULTS.md) that a later test asserts exists + parses?
4. BYTE-IDENTICAL GUARD: test_audio_byte_identical is the crown jewel.
   Does the new lane need a DEDICATED variant (episode rendered with
   ltx_av forced -> master hash unchanged), and can that run CPU-only
   via the existing prune-to-node-7 trick (audio path without video
   cost), or is it M4-GPU-only? Specify.
5. DESKTOP-VS-HEADLESS NODE GATE: PR #13111 nodes may exist in one build
   and not the other. Where is that gate TESTED -- assert_usable unit
   with a mocked NODE_CLASS_MAPPINGS missing one class (CPU), plus an M0
   checklist row per build? Anything else?
6. REGRESSION DISCIPLINE: Bug Bible flow for this lane -- which existing
   BUG-IDs are at risk of regression (BUG-070 Sage, BUG-291 reclaim,
   BUG-265 family) and does any new lane behavior deserve a NEW Bug
   Bible row at ship (e.g. the silent-rounding dims trap)?

Rules: cite grounding or VERIFY-AT-BUILD; mirror existing patterns; no
new test frameworks; CPU determinism (no network, no GPU in pytest).
Output: numbered MUST-FIX (file + what), SHOULD-CONSIDER, OPEN-QUESTIONS.
Terse.
