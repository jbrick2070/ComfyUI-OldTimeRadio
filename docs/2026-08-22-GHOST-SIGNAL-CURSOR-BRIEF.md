# CURSOR BRIEF -- pressure-test and improve the Ghost Signal plan

You have read access to this repository. **Read the real files before you claim
anything.** Every assertion in your answer must cite a path and, where it
matters, a line. If you cannot verify something, say "unverified" rather than
asserting it -- a confident wrong claim costs more here than a gap.

Repo root: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`

## What this is

We are adding a new video lane to a fully local, offline, open-source
old-time-radio episode generator. Episodes are assembled from ~2 s "beats" and
published to `otr/obs/`. The new lane -- **Ghost Signal** -- is a deliberately
lo-fi, artistic, text-to-video lane built on AnimateDiff (SD1.5-era motion
modules) for people with very little VRAM. Its instability is a STYLE, not a
defect.

A design panel has already run (Fable cold + 3 rounds; Codex blind + a developed
pass). The spec is settled in shape. **Your job is to break it or improve it,
not to admire it.**

## Read these first

1. `docs/2026-08-22-GHOST-SIGNAL-SPEC.md` -- the spec of record. Start here.
2. `docs/2026-08-22-lofi-video-lane-PLAN.md` -- the engineering anchor: wiring
   surfaces, the operator rulings with their reasoning, the VRAM section.
3. `docs/VIDEO_LANE_PREFLIGHT.md` -- the 8 gates every new video lane must pass,
   machine-enforced by `tests/test_lane_preflight_matrix.py`.
4. The "HOW TO ADD YOUR OWN VIDEO ENGINE" docstring at the top of
   `nodes/_otr_video_engines/__init__.py`.
5. `CLAUDE.md` at the repo root -- hard operator rules.

Code you will need: `nodes/_otr_video_engines/registry.py` (the `VideoEngine`
Protocol), `motion_common.py` (`MotionEngineBase`), `eng_ltx_8gb.py` (the
current cheapest lane, `render_canvas = (512, 288)`), `otr_silent_composite.py`
(`_scale_filter`, line ~172), `_otr_visual_styles.py`, `_otr_motion_clause.py`,
`_otr_shared/public_engines.py`.

## THE TWO CONSTRAINTS IN TENSION -- this is the whole assignment

The operator, verbatim:

> **"I love full 1080p but need to keep under 4gb end to end."**

**Constraint A -- FULL 1920x1080, edge to edge, no black bars.** This is his
stated problem statement: *"I hate black bars."* Non-negotiable.

**Constraint B -- UNDER 4 GB VRAM, END TO END.** The hardest constraint in the
design. It outranks every aesthetic preference.

### The arithmetic that makes this hard, and the first thing you should attack

Approximate fp16 weights for the proposed stack:

| Component | Approx |
|---|---|
| SD1.5 UNet | ~1.7 GB |
| AnimateDiff motion module | ~1.7 GB |
| CLIP text encoder | ~0.25 GB |
| VAE | ~0.16 GB |
| **Weights alone** | **~3.8 GB** |

**That is essentially the entire ceiling before a single activation.** These are
ESTIMATES made without the package installed -- correct them if you can find
real figures.

**Attack this first. The honest question is whether the lane can exist at all
under 4 GB, and if so how.** Things worth investigating and costing:

- Sequential load/unload staging (encode text -> free; sample -> free; decode ->
  free). Does the existing `MotionEngineBase` teardown (V-4 patcher detach --
  and note `unload_all_models` is forbidden here) actually reclaim between
  stages, or does something retain?
- ComfyUI `--lowvram` / `--novram` block-swapping: what does it really buy, and
  what does it cost in wall clock?
- Quantized or pruned SD1.5 (GGUF, fp8, pruned ~1.6 GB checkpoints). The repo
  already ships `ComfyUI-GGUF` in `custom_nodes/`. Is a GGUF SD1.5 + AnimateDiff
  path viable, and does AnimateDiff-Evolved tolerate a quantized UNet?
- Motion-module SIZE as a primary selection criterion. The operator proposed
  `mm-p-0.5`; alternatives are v2/v3, AnimateLCM, AnimateDiff-Lightning. Which
  is smallest, and what does each cost elsewhere? (See the CFG interlock below
  -- this is not a free choice.)
- Activation cost of the context window: at 512x288 the latent is 64x36 = 2304
  tokens/frame x 16 context frames; at 384x216 it is 48x27 = 1296. Under a 4 GB
  ceiling, does the canvas choice actually bind? The spec commits to 512x288 and
  I am NOT confident that survives 4 GB.

### Resolve an ambiguity in "END TO END"

Does "under 4 GB end to end" mean (a) this lane's own path -- render, enlarge,
composite -- never exceeds 4 GB at any stage, or (b) the WHOLE episode pipeline
(writer LLM, TTS, music, video) fits under 4 GB?

Reading (b) is much harder and has a concrete consequence you can check:
`config/profiles/otr_4060_nano.json` declares `"vram_ceiling_gb": 6.8` for the
LLM writer. Under reading (b) that profile already breaches the ceiling before
video runs at all, and the lane would need a different writer/TTS/music tier
alongside it. **Say which reading the evidence supports, flag it if it needs an
operator answer, and design for the strict reading where you can.**

## SETTLED -- do not relitigate, design within these

| # | Ruling |
|---|---|
| R1 | No still, ever. `text_to_video`, prompt plus noise. `accepts_still = False` DECLARED (the G3.6 gate polices silence, not the value). |
| R2 | The lane fills ALL THREE roles -- announcer, music, character. The prompter differentiates them. No role may be refused. |
| R3 | No ping-pong, no mirror, no loop-fill. Original seconds of render for every beat. Already law: `_otr_video_engines/acceptance.py` `DELIVERABLE_EXTENSION_MODES = ("none",)`, graded on all beats by `grade_no_mirror`. |
| R4 | 12.5 fps native, uniform hold-2, exact 25 fps delivery. |
| R5 | Full frame, no black bars. |
| R6 | Low resolution is ACCEPTED as identity, not tolerated as a defect. |
| R7 | Under 4 GB VRAM, end to end. |
| R8 | ENLARGE (invents nothing), never UPSCALE (hallucinates detail). The operator's own word. |

**R8 has a repo-grounded reason you should verify and can build on:**
Real-ESRGAN x2plus is a live suspect in an unresolved artifact the operator
personally reported -- see `docs/GO_FORWARD_PLAN.md` around line 1136 (mesh at
00:37 of `beneath_the_silvery_boughs`; raw 832x480 render clean, mesh in the
1920x1080 composite, unpinned because the ledger never records which upscale
engine ran).

## THE ASK -- improve the overall plan, and BE CREATIVE

This is deliberately open. Do not treat the list below as a checklist to tick;
treat it as the shape of the problem, then go wherever you think the plan is
weakest or where you can see something better. **We want your angle, not a
confirmation of ours.** A genuinely better idea we had not considered is worth
more than a thorough audit of the ideas we already have.

Two areas matter most.

### 1. THE PROMPTING ARCHITECTURE -- the operator's own priority

His words: *"the key part is prompting -- how many animations per beat, how do
we take the ledger and visual styles to prompt the animations."*

This is where the lane lives or dies, because with no still image **the prompt
carries the entire picture**: subject, style, and motion. Every other lane in
this repo is a still-carried lane where the prompt only had to describe what
MOVES, so the whole existing prompt architecture is the wrong shape here and was
built for a different problem.

The spec's current answer is an eight-slot composer with a 320-character budget
derived from the CLIP 77-token window. **Attack it. Improve it. Replace it if
you have something better.** Genuinely open questions, and you should feel free
to reframe them:

- Is a fixed slot order even the right abstraction, or should composition be
  driven by what the beat actually contains?
- Is one prompt per beat right? Per animation? Should the prompt EVOLVE across a
  long beat, and what would that cost in coherence?
- How should the nine style packs in `nodes/visual_styles/` and the story ledger
  actually combine? Read the packs -- they are richer than a naive reading
  suggests, and the existing `motion_registers` were written for a different
  consumer.
- What in the ledger is worth spending scarce characters on, and what is a trap?
  (A no-still lane paints literally whatever it is told.)
- How do you get a recognizable CHARACTER out of pure noise, twice in a row,
  with no image anchor? This is the hardest problem in the design and the
  operator has ruled the character role must ship.
- Is there a prompting idea here nobody has tried -- something that exploits
  what AnimateDiff is actually good at rather than fighting what it is bad at?

### 2. SURVIVING 4 GB WHILE DELIVERING FULL 1080p

See the arithmetic above: ~3.8 GB of weights against a 4 GB ceiling. **Be
creative here too.** Staging, quantization, module choice, canvas, offload
strategy, or an approach we have not thought of. If the honest answer is "this
cannot fit as specified", say so plainly and say what has to give -- that is a
useful answer, not a failure.

### Known soft spots, offered as starting points only

Mentioned so you do not waste time rediscovering them. Follow them or ignore
them:

- **The Context Options claim.** The coverage design says one beat = one
  AnimateDiff timeline, with sliding Context Options bounding VRAM by CONTEXT
  LENGTH rather than beat length. This is an **unverified upstream doc claim
  about a package that is not installed**, and the entire coverage architecture
  rests on it.
- **The canvas.** Committed to 512x288 with 384x216 as a reserve. Note
  1080 = 2^3 x 3^3 x 5, so no integer-multiple enlargement can also land on a
  /32-legal height -- that option does not exist.
- **The CFG interlock.** The spec claims this is the first lane whose negative
  prompt genuinely BINDS (real CFG ~7) versus the distilled LTX lanes at CFG 1.0
  where negatives are inert -- and that a cheap AnimateLCM backbone at CFG 1-2
  would make it inert again. Both halves are worth checking, and it collides
  with the VRAM question since AnimateLCM is the cheap option.
- **Wiring.** Spec section 9 lists open questions, including a
  `character_video` vs `character_visual` role-key mismatch and whether
  `MotionEngineBase` is the right parent (it inherits `accepts_still = True` and
  a ping-pong extension path that R3 forbids).

## What to deliver

A written review that IMPROVES the plan -- ideas first, audit second. For each
point: what you propose, why it is better, and the evidence (path + line) where
it touches this repo.

Loosely separate:
- **BETTER IDEAS** -- the part we actually want. Creative improvements to the
  plan and especially to the prompting architecture. Argue for them.
- **BLOCKERS** -- this cannot ship as specified.
- **CORRECTIONS** -- the spec states something false about this repo.
- **UNVERIFIABLE** -- what you could not check and what it would take.

Do not write code. Do not modify files. Do not relitigate the settled rulings --
though if you think one is a mistake, say so once, briefly, at the end; the
operator decides, not the reviewer.
