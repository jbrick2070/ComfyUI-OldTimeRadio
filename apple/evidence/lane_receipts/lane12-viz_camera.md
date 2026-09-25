# VIDEO_LANE_PREFLIGHT receipt -- lane 12, `viz_camera`

`VIDEO_LANE_PREFLIGHT receipt: viz_camera | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane12_viz_camera/ | verdict PASS -- 7/7`

**The short lane, and that is the ledger working.** Both of its red gates are
the ones lane 11 solved the day before, and both answers transferred -- but the
PREMISE was re-checked on this engine's own render path rather than assumed,
which is L19's own runnable check and the only thing that made the transfer
legitimate.

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | PASS | PASS | no weights on this lane |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT |
| G3 contract vs runtime | **RED** | **PASS** | `continuity=CONTINUITY_NONE` at this lane's own contract |
| G4 / G6 | n/a | n/a | exempt -- procedural CPU lane, no model in VRAM, no boot contract |
| G5 audio law (V-1) | PASS | PASS | already probed its own emitted mp4 |
| G7 public surface | PASS | PASS | `ENGINE_MATRIX.md` unchanged (correct -- nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.

## G2 -- INERT, and why that is not just "same as last time"

L19 says a lane must not declare a `render_canvas` for a number that is really
an operator lever's default, and lane 11 closed `viz_green` that way. The rule
lane 19 also carries is that you copy the REASONING, not the shape -- so the
premise was verified here:

`eng_viz_camera.render_clip` builds `paint_golden_camera_frame`, the scanline
table, the vignette and the encoder call **all from the request's own `w, h`**.
There is no latent grid, no trained input size and no canvas-dependent constant
anywhere in the path. So the 1472x832 an episode hands this lane is the default
of `OTR_VIDEO_LANDSCAPE_CANVAS`, not a fact about the engine, and because
`declared_render_canvas` is applied LAST a declaration would silently make this
the one visualizer that ignores that lever.

Four profiles set `render.canvas_w/h` on this lane (`16gb_full`,
`otr_w45_viz_camera`, and two untracked `otr_sbcov_*`). The channel is declared
INERT in `PROFILE_CANVAS_DOCUMENTED_DEAD` with the mechanism written out --
inert, not dead (L18): the number IS carried profile -> applier -> node-87
director widgets -> `request["canvas"]`, and is then overwritten. As in lane 11,
that also means the untracked sbcov profiles are irrelevant to this row rather
than blocking it.

## G3 -- one line, this lane only

`frame_contract` never passed `continuity=`, so `CONTINUITY_NONE` was a dataclass
default while the comment above it had claimed "CONTINUITY none" since the
engine was written. True here for a stateable reason: `render_clip` paints every
frame from the beat's own audio analysis and a per-beat rng key, and reads no
predecessor frame, so no terminal state exists for a successor to inherit.

`viz_mxc_cpu` and `viz_mxc_mandala` are untouched and keep both their rows --
each visualizer owns its own contract, so nothing here reaches them.

## What the QA pass found, and why it is bigger than this lane

The post-coding Sonnet QA on this diff caught a **tautology in the continuity
test I had just written -- and in lanes 10's and 11's, already pushed.**

The assertion was `"continuity=" in inspect.getsource(<engine class>)`, meant to
prove the keyword was DECLARED rather than defaulted. But the comment added
directly above each declaration, explaining why the value is NONE, contains that
same literal. So the test passed on the comment alone and would have kept
passing with the real keyword deleted. **Gate G3.3 itself had the identical
hole** -- it was the same substring search over `_mro_source`.

Asserting the resolved VALUE could never have caught it either: `CONTINUITY_NONE`
is the dataclass default, so a lane that never considered chaining and a lane
that reasoned its way to NONE are byte-identical at runtime. The only readable
difference is whether the keyword was passed -- a fact about syntax.

Fixed at the root: `frame_contract.declares_continuity_kwarg` parses the AST for
a `FrameContract(...)` call carrying a `continuity` keyword. Comments are not
AST nodes. It lives in the ENGINE module so the gate and all three lanes' tests
ask one question through one reader, and it is guarded by
`test_g3_cannot_be_satisfied_by_a_COMMENT_about_continuity`, which feeds it a
class that documents the rule without following it and asserts refusal.

Recorded as **L20: documenting a rule can disable the check that enforces it.**
Three lanes wrote the same tautology without noticing, which is the argument for
the QA pass existing.

## The solo smoke -- LIVE PASS

Stock `default` boot; box reset per CLAUDE.md section 4 first (no resident
server, port 8000 clear, VRAM 1,422 MiB at boot). Real audio: the same 4-second
mono 24 kHz slice lane 11 cut from a published episode, copied into this lane's
own smoke directory, so the visualizer runs REACTIVE rather than idle.

| Item | Value |
|---|---|
| Harness | `_otr_single_engine_smoke.py --engine viz_camera --frames 100 --audio <slice>` |
| Prompt id | `cd6a4b44-7b2c-4ffc-ba82-5ed5e706810f` |
| Wall time | **2.0 s** |
| Canvas PROBED | **832x480** -- `render_single`'s wide default, which is the shipped path for a lane that declares nothing |
| Frames PROBED | **100**, exactly the ask; 100/25 = 4.000 s against the 4 s slice |
| Rate / codec | 25/1, h264, yuv420p, bt709 |
| Audio | **zero audio streams** -- silence proved on the emitted file |
| Artifact | `.../_lane_smokes/lane12_viz_camera/lane12_viz_camera_smoke_f100.mp4` |
| sha256 | `0c9f7e2edca876fc741e43f6dcc8f06d171868d4545d03fb005c460a8ecdbfda` |

**Only ONE leg here, deliberately.** Lane 11 ran a second leg with
`OTR_VIDEO_RENDER_CANVAS` moved to prove the operator lever reaches a
declaration-free visualizer live, and that property is now covered CPU-side for
this lane by `test_the_landscape_lever_still_reaches_this_lane`, driving the
real `build_request_from_shot`. Re-proving the same driver behaviour on a second
engine would have been GPU time spent on a fact already in evidence.

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/ffmpeg lane, G4 exempt, `vram_peak_mb:
null` is correct rather than a gap.

**No profile, variant, workflow or `ENGINE_MATRIX.md` change.** Nothing was
declared, so nothing downstream moved; this lane's whole diff is one import, one
keyword, two gate-table entries, three tests and docs.

**`viz_mxc_cpu` (lane 13) and `viz_mxc_mandala` (lane 14) were not touched.**
Their answers are probably these two again -- but "probably" is exactly what
L19 says to check against each engine's own render path before reusing.
