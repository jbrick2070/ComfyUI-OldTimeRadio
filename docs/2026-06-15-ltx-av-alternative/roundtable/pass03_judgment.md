# LTX-AV Alternative Path -- Convergence Judgment

Loop: pass00 (operator starter + OTR refresh) -> pass01 (Claude synthesis, no live panel -- credits out) ->
pass02 (live GPT-5.5+Gemini-3.1-pro+DeepSeek-v4, 15 must-fixes) -> pass03 (live panel, CONVERGED). Total
panel spend ~$0.39. Claude = panelist (code-grounded) + judge throughout.

## CONVERGED -- why we stop here
pass03's three reviews agree the plan is SOUND and reduce to TWO themes, both folded into pass03_plan_FINAL:
1. **DEFER to the locked 8-pass sprint plan** (`docs/2026-06-10-ltx-av-lane/LTX_AV_SPRINT_PLAN.md`) for the
   internal contracts. My pass01/02 had DRIFTED from it in 3 places; the panel pulled them back:
   frame math = snap-UP `next_8n1` (NOT Lane A's snap-DOWN); NO Director/JSON edit (V-6 auto-dropdown);
   canonicalize trims/pads to exactly T. All three were the panel CORRECTING my synthesis toward the
   already-converged spec -- a strong convergence signal, not a new hole.
2. **M0 is the universal gate.** The LTX-2.3 A2V graph topology / terminal decode node / audio loader /
   viable low-VRAM artifact are UNKNOWN until M0 probes them on THIS 5080. So M1+ are gated behind M0 GO;
   OOM -> PARK. No engine code can be finalized before M0.

The remaining pass03 items are precision/verify-at-build (exception types, NVML helper name, schemas type,
fallback IDs, singleton state machine) -- all covered by "defer to the sprint plan + M0 captures the rest."
A pass04 would re-confirm, not discover; STOP per the convergence rule.

## NET (the refresh deltas over the 6/10 spec -- accepted)
re-ground vs HEAD 9633e1e; M0 as a graph-spike probe-or-park (don't preselect node names); model/VRAM
reality (A2V is 22B-class, GGUF-Q3/offload, NVFP4 cut, <=14500 or park); Lane A prod / Lane B dark with the
lip-sync-vs-HuMo graduation bar; boomerang isolation; singleton-core render_talk/render_music dispatch +
poison-on-OOM; pre-M0 CPU-skeleton sequencing.

## REJECTED / CUT
my snap-DOWN + JSON-edit + "canonicalize-not-needed" drift (pulled back to the spec); NVFP4; the bundled
slice-cache/instrumentation/golden-projection scope (separate tickets); the optical-flow hard gate for
announcer/music.

## STATUS
**PARKED but build-ready PENDING M0.** Lane A (today's shipped motion) stands as production. Lane B starts
with the M0 graph spike whenever the operator green-lights it; if M0 parks, nothing is lost.
