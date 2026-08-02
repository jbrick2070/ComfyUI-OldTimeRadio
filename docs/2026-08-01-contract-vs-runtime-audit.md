# A FrameContract that overstates its adapter is worse than no contract

**Status: one instance found and fixed on live evidence. The question this plan
asks is whether there are OTHERS, before three more legs pay to find out.**

## 1. WHAT HAPPENED (measured, not theorized)

The 45-word campaign leg for `ltx_video` died 11.8 minutes in, after the writer
and the whole audio phase had already been paid for:

    shot shot_music_opening_001 segment 1 rendered 169 frame(s) but its plan
    asked for 89 (a surplus of 80). NO FALLBACK -- the plan's count is what this
    segment's audio slice was cut against, so assembling a segment of any other
    length makes the beat drift against its own audio.

Root cause, grounded in `nodes/_otr_video_engines/eng_ltx_video.py`:

* the adapter DECLARED `frame_contract = FrameContract(min_frames=9, ...)`
* its runtime `_ltx_frame_length` (line 156) RAISES every ask below
  `_LTX_DECODE_FLOOR_DEFAULT = 169` up to 169, because "the wrapper VAEDecode
  fails outside its tiled band at this canvas"
* `_LTX_MAX_FRAMES_DEFAULT` is ALSO 169

So the decode floor EQUALS the cap: this adapter can only ever produce exactly
one length, 169. It declared a range of 9..169 step 8, i.e. 21 legal lengths, 20
of which do not exist. Live log, same run:

    [eng_ltx_video] frame ask 89 below the decode floor 169 -- raising
    [eng_ltx_video] frame ask 97 below the decode floor 169 -- raising

The planner is not at fault and neither is the refusal. `render_beat_coverage`
refused precisely as designed. What it was checking against was a lie.

**Why this class is expensive:** a contract exists so the planner never asks for
a length the adapter cannot deliver. An UNDERSTATED contract merely wastes
capability. An OVERSTATED one converts a plannable engine into a GUARANTEED late
failure -- and it fails late, after the writer, the cast, the TTS and the music
have all been rendered. `ltx_video` also passed for months in single-clip mode,
where nothing ever asked it for a non-169 length; only coverage planning exposed
the lie.

## 2. THE FIX APPLIED

`eng_ltx_video.frame_contract` now declares `min_frames=169, max_frames=169`.

Deliberately LITERALS, not `_LTX_DECODE_FLOOR_DEFAULT`. A first draft derived
them from the constants "so they cannot drift" and
`test_the_LTX_ceilings_do_not_silently_follow_their_env_overrides` rejected it
for a better reason: a `FrameContract` is STATIC because stills are minted
against it before the render phase begins, so it must never track a value that
can move underneath it. Drift is meant to fail a TEST, visibly; derivation would
absorb it silently.

Proven, plan vs engine, every beat size:

    beat 89   1 seg  169->169   visible=89
    beat 250  2 seg  169->169, 169->169   visible=250
    beat 530  4 seg  169->169 x4          visible=530
    ALL SEGMENTS RENDER EXACTLY WHAT THE PLAN ASKED: True

Two tripwires added in `tests/test_engine_contract_roster.py`:
`test_a_declared_MINIMUM_is_a_length_the_adapter_can_actually_render` (feed the
declared min through the adapter's own resolver; it must come back unchanged)
and `test_the_ltx_video_declaration_still_matches_its_runtime_decode_floor`
(equality with the constant, never identity). `docs/ENGINE_MATRIX.md` regenerated
via `tools/engine_matrix.py` -- it now reads `169-169 step 8`.

## 3. THE DRIVER'S OWN AUDIT (anchor -- ground this, do not trust it)

Grepping every `eng_*.py` for runtime floor/cap sites that could contradict a
declaration:

| adapter | floor/cap hits | driver's read |
|---|---|---|
| `eng_ltx_video` | 14 | CONFIRMED the real defect. Fixed above. |
| `eng_ltx_8gb` | 1 | MISREAD-RISK: the single hit is a COMMENT at line 112, not code. No decode floor. Its leg PASSED live (14.2 min, coverage held, `delivered_engine=ltx_8gb`). |
| `eng_ltx_av` | 1 | MISREAD-RISK: the hit at line 580 is a BYTE-size check on a model FILE, not a frame floor. |

Only two adapters define their own length resolver at all: `eng_ltx_video`
(`_ltx_frame_length`) and `eng_wan_ti2v` (`_floor_length` / `_planned_length`).

**The driver's UNVERIFIED belief, which is what the panel is for:** that this
defect is isolated to `ltx_video`. That belief rests on a grep for the STRING
patterns of a floor ("below the ... floor", "-- raising", "exceeds cap"). An
adapter that constrains its length by a different idiom -- a `max()`, a `min()`,
a silent quantize, an assertion in `render_clip`, a graph-node limit, a VAE
temporal stride, a canvas-dependent bound -- would not appear in that grep at
all. A string grep is not an audit.

## 4. WHAT THE PANEL MUST BREAK

1. **Find the contract lies the grep missed.** For EVERY registered engine,
   compare the declared `frame_contract` (min, max, quantum, discrete_frames)
   against what the adapter's `render_clip` path will ACTUALLY produce for that
   ask. Name any adapter where a legal declared length cannot be rendered.
   `humo`, `wan_ti2v`, `wan_i2v` and the still/viz families have NOT yet been
   proven on a live 45-word leg this round.
2. **Is `min_frames == max_frames` a legitimate contract shape?** Does
   `coverage_plan.partition_beat` handle a single-legal-length ladder correctly
   in BOTH join modes -- and specifically under `JOIN_CHAIN`, where each
   successor also drops a head frame? Show the arithmetic for a beat that is not
   a multiple of 169.
3. **Did narrowing `ltx_video` to exactly 169 break anything upstream?** Stills
   are minted against the contract BEFORE the render phase. A beat under 169
   frames now renders 169 and trims. Does `jump_still_requests` /
   `segment_render_window` / the audio slice arithmetic still hold?
4. **Is 169 even right?** The floor is `_env_int("OTR_LTX_MIN_DECODE_FRAMES", ...)`
   and canvas-dependent by its own comment ("fails outside its tiled band AT THIS
   CANVAS"). A static contract now encodes a canvas-dependent number. Is that a
   second lie waiting, at a different canvas?
5. **The general guard.** Should the roster test that feeds a declared minimum
   through its adapter's own resolver be generalized to EVERY engine rather than
   just `ltx_video`? What would that require of adapters that have no separate
   resolver function?

## 5. CONSTRAINTS (non-negotiable)

- Every second of audio gets video. No mirror / ping-pong / re-used frames.
- Fail loud, never silently degrade. `wan_ti2v`'s frozen RECIPE does not move.
- The only workflow JSON is `workflows/otr_canonical.json`.
- 16 GB RTX 5080 laptop; models at `C:\ComfyUI-Models`.
- **A GPU campaign is running right now -- do NOT launch renders or boot a
  server.** Read the files; reason about them.

---

## 6. THE CLOUD LANES -- DEFERRED, NOT DISMISSED (operator ruling 2026-08-02)

**Operator:** "once we get local only we need to be sure all cloud models get
the same treatment so they work -- but they usually have more frame
flexibility."

Both halves are right, and the second half changes the FIX SHAPE. Measured from
the live registry:

| engine | declared |
|---|---|
| `cloud_kling_avatar` | range 50-7500 step 1 (up to 300 s) |
| `google_omni_video` | range 75-250 **step 1** (any integer) |
| `cloud_seedance_2` | range 100-375 step 25 |
| `google_veo_video` | menu (100, 150, 200) |
| `word_razzle` | menu (125, 200) |
| -- local, for contrast -- | |
| `ltx_video` | 169 only |
| `wan_ti2v` / `fastwan_8gb` | 17-177 step 4 |

So cloud lanes really do have far more frame range than anything local.

**But the cloud defect is the INVERSE of the local one, and copying the local
rule across would not fix it.** Local adapters declare MORE than they can render
(`ltx_video` claimed 21 rungs, had 1). The cloud adapters declare a length
range they do not actually CONTROL: `google_omni_video` declares any integer
75-250, yet its request payload sends no duration at all and canonicalization
merely reports whatever duration the provider returned -- while `render_driver`
requires the measured length to equal `segment.render_frames` EXACTLY. The
declared flexibility is unbacked: the PROVIDER picks, not us.

`google_veo_video` fails a third way -- its menu is conditional at RUNTIME.
Reference images, or 1080p/4K, force eight seconds, so planning can select 100
or 150 while execution requests 200.

**Therefore the cloud fix is a different mechanism, not the same one:**
1. A **provider-variable contract** that plans only after delivery, or
2. a **deterministic guaranteed length** plus exact canonical trimming, or
3. **splitting engine identities by capability** so a configuration that forces
   8 s is a different engine id from one that does not.

The env-conflict refusal built for `ltx_video` is still necessary here (the
cloud adapters prioritize duration env vars over `timing.target_frame_count`,
and `word_razzle` already raises `ContractEnvConflict` for exactly this) -- but
it is not SUFFICIENT, because refusing a bad env does not make an adapter
control a length the provider owns.

**Sequencing (operator's, and correct):** local first. Cloud is not touched
until the six local engines are proven on live legs. Recorded here so the
findings survive the wait -- they came from the r2 panel and are grounded, and
re-deriving them later would cost the same review again.

Source: `kibitz-runs/2026-08-01-contract-vs-runtime/r2/` (codex `gpt-5.6-sol`
high MUST-FIX 3/4, SHOULD-FIX 2; grounded and deferred in `judgment.md` R1).

---

## 7. THE OPEN QUESTION r3 SURFACED: THE FLOOR MAY NOT BELONG AT THIS CANVAS

Found by the antigravity lane, 2026-08-02, and it is the most valuable single
claim either panel produced -- because it says the CURRENT fix, while honest, is
more restrictive than the hardware requires.

Read `_LTX_DECODE_FLOOR_DEFAULT`'s own comment (`eng_ltx_video.py:140`):

> at the **1472x832** landscape canvas the installed wrapper's VAEDecode
> survives ONLY in its tiled band -- 169f and 233f decode clean, 121f and 137f
> raise the tensor 256-vs-128 (dim 1) mismatch

And ten lines below, the loop path's own note:

> 97f decodes clean at **832x480** in 12s

So the two canvases behave DIFFERENTLY, and the 169 floor was measured at
1472x832 -- **not** at 832x480, which is where all three shipped profiles and the
live leg actually render. At the production canvas there is direct evidence that
a 97-frame decode is clean, i.e. the floor is very likely not required there at
all.

But `_ltx_frame_length` applies the floor UNCONDITIONALLY, whatever the canvas.

**Consequences, separated carefully:**

* The shipped fix is still CORRECT. A declaration must describe what the runtime
  DOES, and the runtime does raise every ask to 169 at every canvas. Declaring
  169 is what stops the plan-vs-render disagreement that killed the leg.
* The shipped fix is also more restrictive than the hardware. Every ltx_video
  beat now renders 6.76 s and trims back to its audio, which on a short beat is
  a lot of discarded work.
* The real improvement is a CANVAS-AWARE floor: 169 at 1472x832, something
  lower (97 is evidenced, but only for the loop path) at 832x480. That would let
  the contract widen and cut the waste.

**Deliberately NOT done tonight.** It needs its own decode measurements at
832x480 -- which lengths actually survive the band at that canvas -- and those
cost GPU time on a box that is mid-campaign. Guessing a floor is exactly how the
original defect got written. Tracked here so the next session inherits the
evidence instead of re-deriving it.

**Corrected in the same pass:** a comment and a test name added earlier on
2026-08-02 both claimed the 169 floor was measured at 832x480. That was false --
they conflated the canvas production renders at with the canvas the floor was
measured at. Both now say which is which.
