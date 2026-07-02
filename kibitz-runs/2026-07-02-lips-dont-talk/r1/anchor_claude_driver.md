# ANCHOR REVIEW (Claude driver) -- r1 high-level arc
Doc: docs/2026-07-02-canonical-ia2v/LIPS_DONT_TALK_PROBLEM.md
Round focus: is the problem framed right; is the hypothesis ranking sound; is
the probe plan the fastest path to the killer delta.

## VERDICT
The doc's controlled-A/B framing is correct and the deliverables are right.
The RANKING needs rework based on evidence gathered after it was written; two
hypotheses weaken, one strengthens decisively, and two NEW deltas are missing.

## Grounded findings (real files, this session)

1. CONFIRMED (killer candidate #1) -- base-pass resolution. Every production
   clip logged `canvas=512x288` (proof_server.log PLAN lines) -> ia2v base
   pass = 256x144 (`base_w, base_h = width//2, height//2`,
   eng_ltx_av._build_graph_ia2v). Canonical base = 640x360. That is 6.25x
   fewer pixels; in /32 latent space 8x4.5 cells vs 20x11.25. A radio-face
   mouth spanning ~40% of frame height is ~2 latent rows in production vs ~4-5
   in canonical; a CHARACTER face in a wide scene still is FAR smaller. The
   model cannot articulate structure it cannot represent.
2. CONFIRMED (weakens hypothesis #3/prompt) -- the M4 character prompts
   ALREADY carry talking language: `prompt source=m4 ... "face visible,
   speaking to camera, 40s, Lead Astronomer..."` (proof_server.log b002-b004)
   -- and those beats still do not talk. Prompt language may matter at the
   margin (bookends' scene-open prompts lack it) but cannot be the sole cause.
3. CONFIRMED (kills hypothesis #6/negative) -- `_LTX_DEFAULT_NEGATIVE`
   (eng_ltx_av.py:66) = "low quality, ... static, frozen pose, still image,
   watermark, text". It PENALIZES freezing; it does not suppress mouth motion.
4. CONFIRMED (new spread) -- production frame counts span 97..305 @25fps
   (PLAN lines: 241/233/105/97/145/305). SHORT beats (97/105 ~= canonical's
   121) also fail => clip length alone cannot explain the failure, though the
   241-305 bookends may additionally degrade past the trained window.
   Hypothesis #2 demotes to co-factor.
5. NEW DELTA (missing from doc) -- audio-length vs frame-length alignment at
   LTXVConcatAVLatent: canonical trims audio to EXACTLY fps*seconds+1 frames
   (TrimAudioDuration 0-5 + 121f). Production encodes the full per-beat slice
   (e.g. 9.293s) against next_8n1(target_frame_count) frames (233f=9.32s) --
   close but not exact; behavior of the concat under mismatch (pad? truncate?
   stretch?) is UNVERIFIABLE from our repo (node internals in ComfyUI-LTXVideo)
   -- verify-at-build or probe.
6. NEW DELTA (missing from doc) -- still CONTENT per role: canonical + the
   working smoke conditioned on a FACE-FORWARD close-up. Production character
   beats condition on wide SCENE stills (small faces); bookends on the
   radio-face still (face-forward, closest to canonical). Interacts
   multiplicatively with (1): face pixels = canvas pixels x face fraction.
7. UNVERIFIABLE in-repo -- LTX-2.3 fps sensitivity (24 vs 25) and trained
   max audio-coupled window; treat as probe variables, not assumptions.

## MUST-FIX for the plan (r1)
- Reorder hypotheses: (1) base resolution, (2) face fraction of frame (per
  role), (3) AV length alignment, (4) long windows (bookends only),
  (5) prompt language, (6) fps 24v25. Drop the negative-prompt hypothesis.
- The probe matrix MUST use the working isolation harness and mutate ONE
  variable per run, ~3 min each, in this order:
  P1 canonical-everything + base 256x144 (expect: kills articulation =>
     resolution proven);
  P2 canonical-everything + our production still for a CHARACTER scene (wide
     shot) at canonical res (tests face-fraction);
  P3 canonical + 241f @ 25fps (tests window/fps);
  P4 canonical + production bookend prompt text (tests prompt);
  P5 canonical + untrimmed 9.3s audio vs 121f (tests AV alignment).
- Add the fix-shape decision to the plan: if P1 confirms, the production fix
  is a per-recipe render canvas (ia2v derives its REFINE target from the
  deliverable canvas, base = half of that), NOT a global
  OTR_LTX_AV_RENDER_CANVAS bump (VRAM: refine at 832x480 /w 241f must be
  re-measured against the 14.5GB ceiling; the isolation smoke's 1280x720x121
  peaked 14.4GB).
- Keep the constraint set verbatim (no fallbacks, byte-identical audio,
  suite/Bible green, workflow JSON only on node/widget change).

## SHOULD-FIX
- Note the composited-final measurement caveat: my correlation metric on the
  blended obs final under-reads (procgen open blend, captions, off-center
  radio) -- probes should score the RAW clips (episodes/<ep>/clips/*.mp4).
- Record per-beat slice durations vs frame counts from the ledger in the
  probe log so hypothesis 5 is judged on numbers, not vibes.
