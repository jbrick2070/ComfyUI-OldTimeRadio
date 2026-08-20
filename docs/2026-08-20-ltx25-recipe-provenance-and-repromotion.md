# LTX 2.5: where our recipe came from, how it differs from the official
# template, and exactly how to re-promote when the lab settles it

**Written 2026-08-20 while hunting the delivered-video grid.** Nothing here
changes a value. It records a chain of custody that took a day to establish, so
the next window inherits it instead of re-deriving it.

## THE CHAIN OF CUSTODY

    official ComfyUI LTX-2.5 template
              |  (diverges -- see the table)
              v
    vram-recipe-lab/recipes/ltx_2_5_golden_i2v_foley.json
              |  (faithful transcription -- verified node by node)
              v
    nodes/_otr_video_engines/ltx25_recipe.py  +  eng_ltx25.py

**OTR IS A FAITHFUL COPY OF THE LAB'S GOLDEN. Verified 2026-08-20, node by
node:** anchor `LTXVImgToVideoInplace` strength 1.0, decode `512/64/33/4`,
`LTXVScheduler` fed from the ImgToVideo output, dual CFG 1.0/1.0. Every value
agrees.

**So when our lane differs from the official ComfyUI template, OTR is not the
place that drifted -- the lab's golden is.** That matters for where a fix goes:
changing `ltx25_recipe.py` directly would put OTR out of step with its own
source of truth and destroy the ability to attribute any result.

## HOW THE LAB GOLDEN DIFFERS FROM THE OFFICIAL TEMPLATE

Established by diffing against `video_ltx2_5_i2v.json` and
`video_ltx2_5_flf2v.json`, the official ComfyUI templates.

| | lab golden -> OTR | official template | status |
|---|---|---|---|
| sigma source | **`LTXVScheduler`** computed: `steps 8, max_shift 2.05, base_shift 0.95, stretch, terminal 0.1` | **`ManualSigmas`** fixed list | **UNTESTED -- live hypothesis** |
| resolved sigmas | `[1, .970339, .931954, .880331, .807194, .695557, .504157, .1, 0]` | `[1, .99375, .9875, .98125, .975, .909375, .725, .421875, 0]` | see below |
| decode spatial | 512 / 64 | 512 / 64 | **identical** |
| decode temporal | 33 / 4 | 64 / 16 | **UNTESTED** |
| conditioning | `LTXVImgToVideoInplace` strength **1.0** | 2x `LTXVAddGuide` strength **0.7** | DELIBERATE, see below |
| sampler | `euler_ancestral_cfg_pp`, eta 1.0 (default) | `euler_ancestral`, eta **0** | **TESTED -- exonerated** |
| latent x2 upscaler | banned in-graph | applied before decode | DELIBERATE, OOMs |
| transformer | `Q3_K_M` GGUF | int8 convrot | DELIBERATE, Q5 breached the clamp |

**`terminal: 0.1` is why our last sigma is 0.1** rather than 0.421875. That is
the computed curve doing what it was told, not drift.

**The schedules differ in SHAPE, not just endpoint.** Ancestral injection per
step, computed:

| step | ours | official |
|---|---|---|
| 0-3 | 0.23 - 0.32 | **0.11 flat** |
| 4-5 | 0.35, 0.35 | 0.33, 0.44 |
| 6 | **0.098** | **0.343** |
| total | 1.90 | 1.55 |

We are 2-3x noisier through the FIRST FOUR steps and much quieter at the end, on
a model distilled for the official shape. That is the live hypothesis.

## DELTAS THAT ARE DELIBERATE -- do not "fix" these without an operator decision

Each has a recorded reason in `ltx25_recipe.py`. They are decisions, not drift:

* **Anchor strength 1.0** (vs the official 0.7). Chosen to fight *"a character's
  face changing between beats"*, a live correctness defect named in `CLAUDE.md`.
  The recipe itself notes the honest framing: **"0.7 vs 1.0 has never been A/B'd
  on this model"**, not "the lab chose 1.0".
* **No in-graph 2x latent upscale.** Forces the video VAE to decode 1664x960x97
  and hard-OOMs. May be used as a SEPARATE offline pass.
* **`Q3_K_M`.** `Q5_K_M` measured 15.58 GiB, over the clamp; the Q5 file is
  quarantined.
* **No 161-frame multishot.** Spikes to 18-20 GiB at this canvas.

## WHAT IS ALREADY ELIMINATED AS A CAUSE OF THE GRID

All by measurement, not argument. Full detail in
`vram-recipe-lab/LTX25_GRID_PROBLEM_STATEMENT_v2.md`.

1. decode tiling -- identical scores in a VAE-only round trip
2. the VAE itself -- delta periodicity 1.7x, no carrier
3. the pre-VAE still resize -- lanczos/area/bicubic identical
4. delivery-side filtering -- 7 ffmpeg chains + Real-ESRGAN, nothing works
5. canvas -- 1024x576 was worse and cost +44% wall clock
6. **sampler eta** -- eta 0 vs 1.0, seeded A/B, lattice unchanged at 2.2

## THE RE-PROMOTION PROCEDURE -- when the lab settles the golden

**Do not edit `ltx25_recipe.py` from a lab RESULT. Edit it from the lab's
updated GOLDEN JSON**, so the chain of custody above stays intact and
`tests/test_ltx25_recipe_matches_lab_golden.py` keeps meaning something.

1. **The lab updates `recipes/ltx_2_5_golden_i2v_foley.json`** and states which
   node/value changed and what proved it.
2. **Run the drift gate first:**
   `pytest -q tests/test_ltx25_recipe_matches_lab_golden.py`. It reads the lab's
   ACTUAL workflow file and will now FAIL -- that failure is the signal, and it
   names the exact constant that is out of step.
3. **Transcribe into `ltx25_recipe.py`**, one constant per changed node, keeping
   the comment discipline: say what the value is, what it was, and what proved
   the change.
4. **If the change touches VRAM** (temporal decode does), re-measure peak with
   `torch.cuda.max_memory_allocated()` on the production I2V lane and record it
   against the 15.445 GiB control. **There is no absolute gate** -- production
   already sits above the 14.5 GiB target, so the comparison is relative.
5. **Full suite + the Bug Bible regression**, then the acceptance leg: one live
   canonical render that reaches `otr/obs/`. Per the operator's standing rule a
   leg that does not publish did not pass.
6. **Push, then verify HEAD == origin**, no 0-byte files, no BOM, AST parse.

**The workflow JSON is NOT expected to change.** These are engine-internal
recipe constants, not node wiring, so `workflows/otr_canonical.json` stays as it
is -- but re-run `OTR_WorkflowValidator` anyway if any node class or widget
count moves.

## WHAT IS QUEUED IN THE LAB

* **The sigma test** -- swap the computed `LTXVScheduler` for the official
  `ManualSigmas` list, changing nothing else. The strongest remaining candidate.
* **The FLF2V probe** -- feasibility and cost of first-and-last-frame
  conditioning, for continuity across per-beat renders. NOT a grid fix; a
  separate capability the operator asked for.
