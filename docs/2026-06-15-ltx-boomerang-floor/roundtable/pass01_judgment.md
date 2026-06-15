# Pass01 Judgment (Claude, grounded vs eng_ltx_video.py + render_driver.py)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4. Spend $0.10. Converged in ONE pass.

## ACCEPTED (folded into pass01_plan.md)
- **Half-length math + freeze shortfall** (GPT#1, Gemini#2, DeepSeek#3). CONFIRMED by arithmetic: target
  169 -> naive half 85 -> 8n+1 snap 81 -> `2*81-1=161 < 169` -> composite freeze. Fix: round up
  `(target+1)//2`, then `while 2*src-1 < target: src += 8`. The strongest catch of the pass.
- **No global floor change; boomerang-only source length w/ hardcoded 97** (GPT#3/#4/cut1, DeepSeek#1/#5).
  CONFIRMED the code says the 169 floor governs both paths "do NOT touch". New helper + env
  `OTR_LTX_LOOP_MIN_DECODE_FRAMES=97`, global `_ltx_frame_length` untouched.
- **`frames[-2::-1]` correct; "midpoint" wording wrong** (GPT#2). Verified `[0,1,2,3]->[0,1,2,3,2,1,0]`;
  the dropped frame is the turnaround/LAST, not a midpoint. Wording fixed; index unit-test added.
- **LtxOrbitEngine inherits render_clip -> boomerang leaks** (GPT#6). CONFIRMED L804 `class
  LtxOrbitEngine(LtxVideoEngine)` with no render_clip override. Gate via a class attr (ltx_orbit OFF).
- **Class-resolution ordering** (GPT#7). Compute src AFTER dims/normalize, BEFORE graph build; keep i2v
  class path unchanged. Matches the code (length@L706, graph@L716/719).
- **Guards/asserts** (GPT-SHOULD1/2/6, DeepSeek#4): `len(frames)<2` LOUD skip; assert final `%8==1`; env
  false/invalid parsing. Cheap, accepted.
- **Ledger stamp plumbing** (GPT#5, DeepSeek-SHOULD): return-dict + canonicalize path; sink exists
  (b005 ledger has the field). SHOULD, verify at build.

## REJECTED / DOWNGRADED (grounded out)
- **DeepSeek#2 "disable i2v / text-only for the half-render; unconditioned reversed first frame seam"** --
  MISREAD. The reverse half is the SAME decoded frames played backward (no re-conditioning). b005 was
  i2v+loop and is the GOOD target. i2v stays ON; the loop join lands on the bookend = seamless. Kept only
  as a build-time eyeball of the turnaround.
- **Gemini Option C (full-render 169 + slice first half + mirror)** -- REJECTED. It yields SHALLOWER motion
  (only the first half of the render, never reaching peak drift) which DEFEATS the "more motion" goal, and
  wastes compute. Gemini's valid sub-point (don't touch the global floor / guarantee decode) is HONORED by
  the accepted 97-floor boomerang-only path.
- **Probe 73/49 for a lower min** (my pass00 / cut by GPT#2-cut, DeepSeek-cut) -- CUT. Use the proven 97;
  a lower default can come later from a build-time smoke, not a runtime probe.
- **Canvas-aware GLOBAL floor (Option A)** -- CUT for this patch (GPT-cut1, Gemini-cut1). Orthogonal; if ever
  wanted, do it separately for non-loop renders, not entangled with the boomerang.

## CORRECTED FACTS (pass00 imprecision the panel caught)
- `OTR_LTX_RENDER_CANVAS` is read in `render_driver.py` L816 (default 832x480), NOT in the engine; the
  engine gets 832x480 via `request.canvas` in `_dims`. Outcome correct, mechanism reworded.

## CONVERGENCE
No invariant-breaking proposals survived; all three models agree on the core (boomerang-only source length
+ in-tensor mirror, leave the global floor alone). One material new must-fix (the freeze shortfall) was
found and folded. The spec is build-ready; a 2nd pass would spend money to hear "looks good." STOP.
