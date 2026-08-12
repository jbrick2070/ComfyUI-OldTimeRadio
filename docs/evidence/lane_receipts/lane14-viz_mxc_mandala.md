# VIDEO_LANE_PREFLIGHT receipt -- lane 14, `viz_mxc_mandala`

`VIDEO_LANE_PREFLIGHT receipt: viz_mxc_mandala | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane14_viz_mxc_mandala/ | verdict PASS -- 7/7`

**The last visualizer. All four are now closed, and every one of them closed by
declaring its profile canvas channel INERT rather than by declaring a canvas.**

## Matrix row

| Gate | Before | After | What moved |
|---|---|---|---|
| G1 weights resolve | PASS | PASS | no weights on this lane |
| G2 canvas truth | **RED** | **PASS** | profile canvas channel declared INERT |
| G3 contract vs runtime | **RED** | **PASS** | `continuity=CONTINUITY_NONE` at this lane's own contract |
| G4 / G6 | n/a | n/a | exempt -- procedural CPU lane |
| G5 audio law (V-1) | PASS | PASS | already probed its own emitted mp4 |
| G7 public surface | PASS | PASS | `ENGINE_MATRIX.md` unchanged (correct -- nothing declared) |

Pre-lane `scripts/build_variants.py --check`: **46 variants, 0 failures**.

## S8b-16's pycairo half was ALREADY DONE -- verified, not rebuilt

The corpus assigns this lane "the pycairo half of S8b-16 (a NAMED dependency
refusal)". It is already in place and was checked rather than re-implemented:
`assert_usable` probes `import cairo` and raises `EngineUnusable(MISSING_MODEL,
"viz_mxc_mandala needs pycairo (pip install pycairo)")` BEFORE the ffmpeg probe,
with a separate message for each, and `load()` carries the same pair. Coverage
exists too -- `test_assert_usable_missing_cairo_fails_loud` forces the ImportError
via `monkeypatch.setitem(sys.modules, "cairo", None)`, so it runs even on a box
where pycairo IS installed rather than skipping into silence.

Recorded as verified-already-green rather than claimed as this lane's work.

## G2 -- the premise re-checked with MORE suspicion, and it still held

This was the visualizer most likely to have needed a real declaration: the only
lane in the family with a NAMED external dependency, and the only one painting
through a graphics library rather than numpy. L19 says check the premise per
engine, so it got a harder look, not a lighter one.

It holds:

* `render_clip` allocates `cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)` from
  the request's own dimensions;
* `paint_mandala(ctx, w, h, ...)` lays the mandala out from the same pair;
* `mandala_surface_to_rgb(surface, w, h)`, `build_scanlines(w, h)`,
  `build_vignette(w, h)` and `apply_crt_post_rgb` all take them;
* `encode_silent_mp4` is handed them last.

**Cairo imposes no canvas of its own** -- an `ImageSurface` is whatever size you
ask for. So the 1472x832 an episode hands this lane is
`OTR_VIDEO_LANDSCAPE_CANVAS`'s default, an operator lever, and since
`declared_render_canvas` is applied LAST a declaration would overrule it for this
lane alone. Channel declared INERT in `PROFILE_CANVAS_DOCUMENTED_DEAD`.

Seven profiles select this lane, including the `otr_amd16_rocm`,
`otr_amd8_rocm` and `otr_mac_mps` portability tiers; as in lanes 11-13 the
untracked `otr_sbcov_3` is irrelevant to the row rather than blocking it.

## G3 -- the one that looked stateful and is not

`continuity=CONTINUITY_NONE` passed at this lane's own contract. The reason is
worth the extra sentence, because this is the one lane where NONE needed
checking rather than assuming: **the cairo surface and context ARE reused across
frames.** That is an allocation optimisation, not state -- `paint_mandala`
repaints the full field every frame from that frame's own audio analysis, and
nothing reads a predecessor frame's PIXELS. So no terminal state exists for a
successor segment to inherit, and NONE is honest.

Its test uses the AST reader `frame_contract.declares_continuity_kwarg` from the
start, so L20's tautology is not re-introduced here.

**With this lane, all four visualizers' G3 rows are closed** -- four separate
one-line declarations in four modules, because this family shares no base (lane
10's `_CheapFamilyBase` fix reached the still shelf instead).

## The solo smoke -- LIVE PASS

Stock `default` boot; box reset per CLAUDE.md section 4 first. pycairo 1.29.0
confirmed present in the venv before submitting, so a fail-closed refusal and a
missing dependency could not be confused (lane 7's lesson 4, applied to a
library instead of a node class). Real audio: the same 4-second slice, so the
mandala runs REACTIVE rather than idle.

| Item | Value |
|---|---|
| Harness | `_otr_single_engine_smoke.py --engine viz_mxc_mandala --frames 100 --audio <slice>` |
| Prompt id | `f7cfc7a4-7a25-427a-9e8b-7fd187f6de45` |
| Wall time | **2.2 s** |
| Canvas PROBED | **832x480** -- `render_single`'s wide default, the shipped path for a lane that declares nothing |
| Frames PROBED | **100**, exactly the ask; 100/25 = 4.000 s against the 4 s slice |
| Rate / codec | 25/1, h264, yuv420p, bt709 |
| Audio | **zero audio streams** -- silence proved on the emitted file |
| Artifact | `.../_lane_smokes/lane14_viz_mxc_mandala/lane14_viz_mxc_mandala_smoke_f100.mp4` |
| sha256 | `e242137e1532ca0e557cc091df9b6f865280c96ae8aad397345ae9b1c7c2c9fa` |

## What the visualizer family looks like now

| Lane | Engine | G2 closed by | G3 |
|---|---|---|---|
| 11 | `viz_green` | channel INERT | declared |
| 12 | `viz_camera` | channel INERT | declared |
| 13 | `viz_mxc_cpu` | channel INERT | declared |
| 14 | `viz_mxc_mandala` | channel INERT | declared |

Four for four. **If a future procedural lane finds itself wanting to declare a
canvas, that is the anomaly and it needs the L19 argument made explicitly, not
the precedent.**

## Deliberately NOT done here

**No VRAM number, no cost row** -- CPU/pycairo/ffmpeg lane, G4 exempt.

**No profile, variant, workflow or `ENGINE_MATRIX.md` change** -- nothing
declared, so nothing downstream moved.

**pycairo was not added to the main requirements.** The module docstring says it
deliberately is not, so a box without system libcairo never breaks any other
engine's install; the NAMED refusal is what makes that safe. Untouched.
