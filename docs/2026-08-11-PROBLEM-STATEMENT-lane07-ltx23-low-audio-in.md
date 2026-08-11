# Lane 7 problem statement -- `ltx23_low_audio_in` (`ltx_audio_in`)

Video lane build, packet 7 of 21. Written 2026-08-11 as the panel's grounding
document. **Facts and forks only -- no recommendation is stated here**, so a
cold reviewer forms its own view before the driver anchor frames one.

Companions the reviewer should read: `docs/LANE_BUILD_LESSONS.md` (the ledger),
`docs/VIDEO_LANE_PREFLIGHT.md` (the gates), `docs/GO_FORWARD_PLAN.md` item 5
(the lane queue), `CLAUDE.md` (hard operator rules).

---

## 1. What this lane is

`ltx_audio_in` is the unified LTX-2.3 22B **audio-conditioned** video lane
(`nodes/_otr_video_engines/eng_ltx_av.py`, 1363 lines). Family
`audio_conditioned_video`. It serves three roles -- `announcer_visual`,
`music_visual`, `character_video` -- and is the default engine for the two
bookend roles.

Its public menu id today is `ltx23_16gb_audio_in`
(`nodes/_otr_shared/public_engines.py:32`). The 21-lane build renames it to
`ltx23_low_audio_in` under the `<model><version>_<low|high>_<capability>`
convention adopted at lane 1; `low`/`high` is a **measured VRAM bucket**, not a
quality claim (`public_engines.py:34-47`).

Its declared frame contract (`eng_ltx_av.py:1350-1357`): `min_frames=9`,
`max_frames=497`, `quantum=8` (8n+1), `native_fps=25`, `allow_tail_trim=True`,
`continuity=CONTINUITY_SOFT_REFERENCE`.

Preflight matrix today (`tests/test_lane_preflight_matrix.py`):

| Gate | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|---|
| `ltx_audio_in` | ok | **RED** | ok | ok | ok | **RED** | ok |

Both RED rows are `EXPECTED_RED` entries owned by this lane
(`tests/test_lane_preflight_matrix.py:188-196`). **Both must be deleted in the
commit that closes the lane**, or the strict unexpected-pass gate fails the
suite.

---

## 2. The four defects this lane owns

### 2.1 S8b-9 -- an env typo deletes the lane from the menu (gate G6)

`eng_ltx_av.py:177`:

    _LTX_AV_RESERVE_VRAM_GB = float(os.environ.get("OTR_LTX_AV_RESERVE_VRAM_GB", "4.0"))

A bare module-scope `float()`. Every adapter import in
`nodes/_otr_video_engines/__init__.py` is wrapped in
`try: ... except Exception: pass`, so a malformed value raises during import,
the guard swallows it, and the engine is simply **absent from the ComfyUI
dropdown with nothing in the log** (reproduced live: registry 27 -> 26). Worse,
`frame_contract.frame_contract_for` resolves an adapter it cannot reach to
`SINGLE_ONLY` through its own broad `except Exception`
(`frame_contract.py:243-247`), so the lane silently reverts to unbounded
single-clip behaviour.

The guarded parser this file already owns is `_env_num`
(`eng_ltx_av.py:59-90`), applied to four other constants at `:95, :100-102`.
Line 177 is the one module-scope numeric read it was never applied to.

`tests/test_ltx_av_env_import_safety.py` states in its own docstring that it
covers every module-scope environment read, and its `MALFORMED` table
(`:38-43`) lists exactly four variables -- omitting this one.

### 2.2 S8b-10 -- the ia2v stage-A base latent is not /32-legal (gate G2)

`_build_graph_ia2v` (`eng_ltx_av.py:778-...`) is the canonical two-stage
lip-sync graph. Stage A renders motion at **half canvas**
(`eng_ltx_av.py:819`):

    base_w, base_h = int(width) // 2, int(height) // 2  # exact canonical halving

At the lane's live ia2v canvas of 832x480 that is **416x240**, and
`240 % 32 == 16`. LTX's spatial multiple is 32
(`nodes/_otr_shared/av_dims.py:59-80`, `assert_ltx_dims`), but that helper is
only ever called on the FULL canvas (`eng_ltx_av.py:587, :1040`) -- never on
the derived stage-A latent.

Two places assert the illegal value is legal:

* `nodes/_otr_video_engines/render_driver.py:2542` -- the comment
  "832x480 = 2.6x the old pixels, a 1.77x deliverable upscale, base 416x240
  (all /32)". `240` is not /32.
* `tests/test_ltx_av_ia2v_canonical.py:62-63` **pins** `416` and `240` as the
  expected stage-A width/height.

**The arithmetic that generalises the defect:** because stage A is an exact
halving and stage B upsamples it x2 (`LTXVLatentUpsampler`, then
`LTXVCropGuides` against the base latent), the stage-A latent is /32-legal
**if and only if the full canvas is /64-legal on both axes**. Checked against
the three canvases in play:

| Full canvas | /64 on both axes? | Stage-A half | /32-legal? |
|---|---|---|---|
| 832x480 | 832 yes, 480 **no** | 416x240 | **no** (240 % 32 = 16) |
| 512x288 | 512 yes, 288 **no** | 256x144 | **no** (144 % 32 = 16) |
| 1024x576 | 1024 yes, 576 yes | 512x288 | yes |

(512x288 only reaches stage A if a single-pass recipe were ever routed through
the two-stage builder; today it is not -- see 2.3.)

Snapping the half UP to 416x256 is not a free fix: stage B's x2 upsample would
then produce 832x512 against a declared 832x480, and `LTXVCropGuides` is wired
to the base latent, so the two stages would disagree geometrically.

### 2.3 S3 -- the lane declares no `render_canvas`, and its canvas is RECIPE-DEPENDENT (gate G2)

This is the fork.

**What G2 wants.** `gate_g2_canvas` (`tests/test_lane_preflight_matrix.py:379-430`)
requires, for a local GPU lane: (G2.1) a `render_canvas` declaration, /32-legal
on both axes; (G2.2) a per-lane drift test pinning it; (G2.3) every profile
that sets `render.canvas_w/h` either matches the declaration, or the lane is
listed in `PROFILE_CANVAS_DOCUMENTED_DEAD` with the dead channel written down.

**What the lane does today.** It declares nothing, so the canvas is decided by
an inline branch in the driver (`render_driver.py:2534-2553`) and is
**recipe-dependent**:

    _av_default = ("832x480"
                   if _ia2v_talking_register_active("ltx_audio_in")
                   else "512x288")
    _avc = os.environ.get("OTR_LTX_AV_RENDER_CANVAS", _av_default)

* `ia2v_canonical` recipe active -> **832x480**
* any other recipe (single-pass) -> **512x288**
* `OTR_LTX_AV_RENDER_CANVAS` overrides either, with no refusal on conflict.

`_ia2v_talking_register_active` (`render_driver.py:1341-1359`) asks the ENGINE
which recipe is live; the recipe itself is selected by `OTR_LTX_AV_RECIPE`
(default `auto`, derived from the installed unet family,
`eng_ltx_av.py:442-490`).

`declared_render_canvas` (`render_driver.py:227+`) is a **static per-class
attribute applied LAST** in `build_request_from_shot` and it overrules every
branch above -- "nothing can clobber this and this clobbers nothing" (B5,
2026-07-27). The driver's own comment at `:2566` records the current scoping
decision: "`ltx_audio_in` keeps its env branch until the general resolver
lands."

**So a static declaration and a recipe-dependent canvas cannot both be true.**

**What S3 orders.** `docs/2026-08-09-SPEC-lab-findings-into-otr.md:195-206`:
`OTR_LTX_AV_RECIPE` is a graph-TOPOLOGY namespace and neither canvas nor frame
count is reachable from it, so ship `render_canvas = (1024, 576)` "on the HQ
lane (both axes /32-legal) plus a named profile supplying 1024x576 and 193
frames, with the measured envelope (warm 7.36 GiB / 585.3 s) documented at the
declaration. H1/H2 alternates CUT."

`docs/2026-08-10-FINAL-QA-video-build-corpus.md:626` assigns lane 7 "the
1024x576x193 HQ profile, legal stage-A canvas".

**The three shipped profiles all say 832x480.** Precisely (corrected after
kibitz r1 -- the first draft said the channel was "read by nothing", which is
wrong): the profile canvas IS consumed upstream, by the profile applier
(`nodes/_otr_workflow_apply.py:540-544`) and emitted through `OTR_VideoDirector`
(`nodes/otr_video_director.py:260-262`). What it is not is the FINAL authority
on the adapter's render, because `build_request_from_shot` overwrites it later.
Dead as render authority, live as a configuration channel -- which is why the
right answer is to make the profiles agree with the declaration rather than to
delete the field. The three:
`config/profiles/otr_16gb_ltx_audio_in.json:65-66`,
`otr_g4_ltx_audio_in.json:71-72`, `otr_w45_ltx_audio_in.json:71-72` -- each
`render.canvas_w/h = 832x480`, `frame_budget: 25`, `beats: 40`.

**Evidence on the table, with its surface stated (lesson L7):**

| Config | Number | Surface / cache | Provenance |
|---|---|---|---|
| LTX AV GGUF Q3, 832x480x97 | 7.2-7.5 GiB | lab **warm** | `docs/2026-08-09-SPEC...:171` |
| LTX AV HQ, 1024x576x193 | 7.36 GiB / 585.3 s | lab **warm** | `...:172`; corpus `:375` marks it SUPPORTED against `results\ltx_audio_hq_h3_1024x576_193f_run2.json` |
| ltx_audio_in at **1280x704** | **14,716 MB -- BREACHED the 14.5 GB ceiling** | **full production pipeline** | `render_driver.py:2539-2541` (proof6), which adds: "the isolation probes carried less resident state" |
| ltx_audio_in at 1280x720 | fails the /32 grid gate | live | `render_driver.py:2538` (proof5b) |
| measured receipt coverage | **stops at f97** | -- | corpus `:398`, `docs\VIDEO_RECIPE_ATTEMPTS.md:10,22-25`. "19.88 s is model-legal, not envelope-qualified." |

Pixel ratios: 1024x576 is **1.48x** the pixels of 832x480 and **0.65x** of
1280x704.

**Two facts that pull the other way, both from the repo:**

1. **832x480 is 26:15, not 16:9.** `docs/2026-07-26-8gb-1080p-arc-judgment.md:27-28`:
   "512x288 and 1024x576 are the only exact-16:9 rungs that are also /32-clean.
   `832x480` is 26:15 -- delivered it becomes 1872x1080 with side bars inside
   the 1920 frame." That judgment ruled "Pillarbox: never (all three)."
2. **The same judgment picked the SMALL canvas for a motion reason**
   (`:20-25`): "At 512x288 a beat plays as ONE continuous shot. At 1024x576 the
   same beat becomes four or five stitched clips -- a motion restart roughly
   every 1.5 seconds... The 4x pixel advantage is largely nominal." That ruling
   was made for `ltx_8gb` (LTX 0.9.8 2B, `otr_8gb_ltx.json`), a **different
   lane** with a much smaller frame ceiling -- but the mechanism (a canvas
   raise lowers the machine-qualified frame ceiling, which raises the segment
   count, which restarts motion) is lane-agnostic.

And the standing operator directive it must be squared with (`CLAUDE.md`,
`GO_FORWARD_PLAN.md`): **"The recipes are not on the table"** -- no VRAM, speed
or cap finding justifies a recipe change; measurement runs the SHIPPED recipe
unchanged. The current 832x480 ia2v canvas is itself an operator quality catch
(`render_driver.py:2535-2544`: the 512x288 default "upscaled ~2.9x to the
deliverable = the 'really low quality' the operator flagged").

### 2.4 No `ContractEnvConflict` refusal (not gated; the transplant plan requires it)

`OTR_LTX_AV_MAX_FRAMES` silently caps the render at `eng_ltx_av.py:1036-1039`,
and the mismatch surfaces AFTER the GPU work as a segment mismatch. The
declared contract deliberately hardcodes the literal `497` rather than the
constant, precisely so the environment cannot rewrite a declaration the image
phase already planned against (`eng_ltx_av.py:1341-1349`).

The sibling lane already has the refusal to copy:
`eng_ltx_video.py:160-222` raises `ContractEnvConflict` for
`OTR_LTX_MAX_FRAMES`, `OTR_LTX_MIN_DECODE_FRAMES` **and**
`OTR_LTX_RENDER_CANVAS` -- the last one on the reasoning that "the environment
disagreeing with a declaration is a REFUSAL, not a quiet re-plan."
`ContractEnvConflict` is defined at `frame_contract.py:69`.

---

## 3. The forks a reviewer is asked to break

**F1 -- the canvas.** Given that `render_canvas` is static and the lane's
canvas is recipe-dependent, what ships?

* (a) Declare `(1024, 576)` per S3. Fixes stage-A legality for free (512x288
  half), gives exact 16:9 with zero pad area, and is the only rung with a
  lab receipt at a long frame count. Costs: it overrules BOTH live branches,
  raises production pixels 1.48x on a lane with a live in-pipeline breach
  receipt 0.65x further up, and the 7.36 GiB figure is lab-warm on a surface
  the repo has already caught reading low.
* (b) Declare `(832, 480)` -- what ia2v renders today, what all three profiles
  say. Costs: kills the 512x288 single-pass branch by fiat, keeps a 26:15
  canvas the 07-26 judgment called a pillarbox source, and leaves S8b-10 to be
  fixed some other way, which the /64 arithmetic in 2.2 suggests may not exist
  without changing the two-stage geometry.
* (c) Declare nothing and register the lane in
  `PROFILE_CANVAS_DOCUMENTED_DEAD` with the recipe-dependence written down.
  Costs: contradicts S3 and the corpus, and G2.1 still fails for a local GPU
  lane, so this is only viable if G2.1 itself is taught about
  recipe-dependent lanes.
* (d) Something else -- e.g. make the declaration recipe-aware at its root
  (a per-recipe canvas map read by `declared_render_canvas`), which is a
  change to a shared driver mechanism serving every lane.

**F2 -- does F1 need a live measurement before it ships?** Lanes 8 and 9 are
explicitly "lab-first measurement before naming". Lane 7 is not marked that
way, but the 1024x576 evidence is lab-warm and the only in-pipeline datapoint
nearby is a breach. If a measurement is required, what exactly is measured, at
what frame count, on which boot lane, and does the lane close in two commits
(measurement, then root fix) as the build law allows?

**F3 -- the frame ceiling.** The contract declares 497 (19.88 s). The only
measured rungs are f97 (832x480) and f193 (1024x576). If the canvas moves, does
the declared max stay 497? A canvas raise that lowers the qualified ceiling
raises segment counts, which is the exact motion defect the 07-26 judgment
named.

**F4 -- what else in the tree assumes 832x480 or 512x288 for this lane?** The
naming rename alone broke tests in three of the last six lanes (lessons L8 /
lane 3): tests hardcoding the aspect suffix, tests asserting a bare internal id
as a saved widget value, and tests using a lane as a "declares nothing"
negative control.

---

## 4. Hard constraints any answer must satisfy

* One lane open at a time; its registration, public id, alias, node-87
  strings, profile/variant, `ENGINE_MATRIX.md` row and canonical-workflow
  delta land **atomically** with the lane.
* A rename **MOVES** the old public id into `_LEGACY_ENGINE_ALIASES`; it never
  adds a second row. Two public ids on one internal id trips the module-scope
  bijection assert (`public_engines.py:68-72`) at IMPORT time and empties most
  of the ComfyUI node menu (lesson L5).
* Node-87 strings are GENERATED by `exact_menu_option_for`, never typed.
* Both `EXPECTED_RED` rows leave `tests/test_lane_preflight_matrix.py` in the
  same commit.
* Root-cause fixes only -- no shims, no band-aids.
* Every numeric claim carries the L7 shape: surface, cache state, boot lane,
  receipt path, and a commit that contains the receipt.
* `workflows/otr_canonical.json` is the workflow source of truth; any wiring
  change lands in it in the same change as the code.
