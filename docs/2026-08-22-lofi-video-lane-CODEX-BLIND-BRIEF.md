# BLIND CREATIVE BRIEF -- a lo-fi artistic video lane for very low VRAM

**You are giving a cold, independent creative opinion.** Another model has been
asked the same question in parallel. You are deliberately NOT being shown its
answer, its vocabulary, or the driver's own leanings, so that the two can be
compared honestly. Do not ask what the other said; invent your own.

Everything below is either an operator ruling (binding fact) or a grounded
repo fact with a file path you can open. Nothing below is a suggestion.

Repo root: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`

---

## 1. What the project is

A fully local, offline, open-source pipeline that generates old-time-radio
episodes: a written script, TTS voices, music, and a VISUAL track rendered over
the audio. Finished episodes publish to `otr/obs/`, which is the operator's
success signal -- a leg that does not reach it did not pass.

The visual track is assembled from **beats**. A beat is one unit of the episode
with its own audio slice; beat length follows the dialogue, so beats vary. The
delivery canvas is 1920x1080 at 25 fps.

There are **three visual roles**: `announcer_visual`, `music_visual`, and the
character role.

## 2. The ask

The operator wants a NEW video lane, in his words:

> *"a new video lane ... think creative artistic for people with really low
> vram, yeah its low res, its experimental"*
> *"especially a new 'video' dropdown entry that could occupy announcer, music,
> or character beats"*

The candidate engine is **AnimateDiff** (Kosinkadink's
ComfyUI-AnimateDiff-Evolved) -- SD1.5-era motion modules, small weights, sliding
context windows, famously loose and unstable motion. **It is NOT currently
installed**, so any claim about its node classes, context-window size or native
resolution is a guess and must be flagged as one.

The operator's framing is that AnimateDiff's instability is a **style, not a
defect**. The lane is supposed to look low-resolution and hand-made on purpose.
His concrete starting recipe: **SD1.5 + the `mm-p-0.5` motion module, 256x256,
16 frames, 8 fps, 8-12 steps.**

The repo ALREADY ships zero-VRAM motion (`still_pan`, `still_motion` in
`nodes/_otr_video_engines/cheap_families.py`) and a cheap GPU lane
(`eng_ltx_8gb.py`, `render_canvas = (512, 288)`). So a new lane must justify
itself against those.

## 3. Operator rulings -- BINDING, do not relitigate

1. **NO STILL.** The lane is text-to-video: prompt plus noise. It must not try
   to generate or consume a still image. He accepts the cost in his own words:
   *"greater abstraction, subject mutation, and flicker -- which may suit your
   low-quality experimental-art aesthetic."*
2. **THE GOLDEN RULE:** *"a 'video' lane should fill all 3 [roles] with the
   prompter prompting for each."* All three roles ship. The PROMPTER is what
   differentiates them. Refusing a role is not available.
3. **NO PING PONG. "Original seconds of render for every beat."** Every beat's
   frames are genuinely rendered for its real duration. This is already law:
   `nodes/_otr_video_engines/acceptance.py:257` sets
   `DELIVERABLE_EXTENSION_MODES = ("none",)` under his 2026-08-06 ruling
   *"there is no mirror or ping pong unless for credits"*, graded on ALL beats
   by `grade_no_mirror`. No mirroring, no looping, no padding to fill a beat.
4. **The road to 1080p** is nearest-neighbour or Lanczos. Note that Lanczos
   ALREADY ships: `nodes/otr_silent_composite.py:172` `_scale_filter()` runs
   `scale=...:flags=lanczos` + `unsharp` for a real engine clip. Nearest-
   neighbour does not exist yet.

## 4. THE CENTRAL QUESTION -- the prompter

The operator's own words for what matters most:

> *"the key part is prompting -- how many animations per beat, how do we take
> the ledger and visual styles to prompt the animations ... all have some
> impact if possible."*

**Read these real files before answering:**

- `nodes/_otr_visual_styles.py` -- the `VisualStyle` dataclass and pack loader.
  Note `positive_tail`, `era_tail`, `image_grade_tail`, `portrait_look`,
  `scene_instruction_look`, `announcer_subject_face`, `motion_registers`, and
  `_MOTION_REGISTER_MAX_CHARS = 240`.
- `nodes/visual_styles/*.json` -- the 9 shipped style packs. Read at least
  three of them to see a pack's real shape.
- `nodes/_otr_motion_clause.py` -- the per-beat, story-driven motion clause. A
  batch pass fills `ledger['video']['shots'][i]['motion_clause']` for every
  shot; `CLAUSE_MAX_CHARS = 130`; the render path is READ-ONLY via
  `resolve_motion_clause_text`.
- `nodes/_otr_video_engines/render_driver.py` around line 1327 --
  `_LTX_MOTION_PROMPT_BY_ROLE` (the existing per-role static motion text) and
  `_LTX_MOTION_PROMPT_MAX = 240`, the composed-prompt budget the clause shares
  with the story core.

**The architectural problem.** Today's 240-character budget is sized for lanes
where the STILL carries the subject, so the prompt only has to say what MOVES.
With no still, this lane's prompt must carry the subject, the style AND the
motion. The existing composition may simply be the wrong shape.

## 5. What to deliver

A design document. No code. Be opinionated and concrete; a recommendation beats
a survey. Flag every guess about AnimateDiff or about code you did not open.

**A. THE LOOK.** Name what this lane looks like on screen, in specific visual
language a person would recognize. Give it ONE identity, not a menu of moods.
Say what it must never look like, and why that failure is the dangerous one.

**B. WHY IT BELONGS TO RADIO.** Radio is an audio form; the picture accompanies
it. Argue what an unstable, low-resolution, dreaming image gives a radio drama
that a clean stable one does not. **If the honest answer is "nothing, and the
zero-VRAM `still_pan` lane already wins", say that plainly.** A real verdict is
worth more than an agreeable one.

**C. THE PROMPT COMPOSITION.** What exactly goes into one prompt for this lane,
in what order, and where does each part come from -- style pack, per-role table,
per-beat ledger clause, story core? Give the actual assembly. If 240 characters
is the wrong budget for a no-still lane, say what it should be and what must
change (that constant is the project's own, not an engine limit).

**D. HOW MANY ANIMATIONS PER BEAT.** 16 frames at 8 fps is 2.0 seconds. Beats
vary with dialogue and may be much longer, and may NOT be padded, mirrored or
looped. Does a long beat become one long render or several chained real
renders? If chained, what stops them looking like unrelated images?

**E. WHAT THE LEDGER SHOULD CONTRIBUTE.** The story ledger knows the scene, the
speaker, the emotional beat. Name the fields worth spending prompt characters
on, in priority order -- and name the traps a no-still lane will fall into,
because it paints literally whatever it is told.

**F. STYLE PACKS.** Nine packs exist and the operator picks one per episode.
Does this lane's look OVERRIDE the chosen pack, COMPOSE with it, or ship as a
tenth pack? Justify it against the fact that a lane whose surface is freely
configurable arguably has no identity of its own.

**G. THE THREE ROLES, BY PROMPT ALONE.** No still is available for any of them.
AnimateDiff's weakest case is a held human face -- identity drifts across a
context window -- and the character role must still ship (ruling 2 above).
**Concretely: what does the prompter do to make a character beat hold together
for two seconds of pure noise?** This is the hardest thing in the design.

**H. THE SCALE TO 1080p.** Nearest-neighbour (hard chunky pixels, the image
declares its own poverty) or Lanczos (smooth, reads as degraded or defocused)?
Is the scaler part of the lane's identity or a delivery detail? Note that a
256x256 square render composites into 1920x1080 pillarboxed, because
`_scale_filter` uses `force_original_aspect_ratio=decrease` then pads black.

**I. THE NAME.** What is this lane called in the operator's dropdown, and what
do people call it out loud? Repo convention is
`<model><version>_<low|high>_<capability>`, and the low/high token must come
from a real measurement receipt, so leave that token out.
