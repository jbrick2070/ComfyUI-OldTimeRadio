<!-- independent panelist: claude-fable-5 (the session agent) -- written BEFORE
     reading the API panel reviews (independence preserved; also the judge,
     which is why this critique is deliberately adversarial to its own plan) -->

VERDICT: yes-with-fixes. The spine is the right architecture, but the statement
underestimates THREE seams I have personally watched break tonight, and one
scope trap.

MUST-FIX BEFORE BUILD:

1. [S1/S3 ordering] The statement implies stills mint where portraits mint
   (post-ShotLock image phase) and the driver consumes by beat_id. But the
   OPEN beat (b000) is SYNTHETIC -- it exists only after ShotLock injects it,
   and its beat_id is not a ledger line. The image phase iterates CAST, not
   beats. Concrete fix: the scene-still mint must iterate the PLANNED SHOT
   rows (video.shots, post-ShotLock), not cast rows or lines -- and store by
   the same beat_id the driver resolves (`_beat_id_for_shot`). Otherwise the
   open still never exists for the one beat the operator cares most about.

2. [S3 LTX] Do NOT gate the whole spine on LTX img2vid. The installed wrapper
   exposing LTXVImgToVideo is UNVERIFIED, and tonight taught us this wrapper
   has version-band quirks (the 121f VAEDecode failure). The build order must
   be: kenburns-over-still (zero risk, the June-5 look literally) ->
   wan_i2v(init=still) (native i2v, ckpt on disk, but operator-gated env +
   ~207s/clip cost at 14B) -> LTX img2vid LAST as a probe. The acceptance
   render must be able to PASS with kenburns alone, or the spine ships
   blocked.

3. [S2 storage] "Move stills into the episode folder" has a hidden consumer:
   eng_humo stages init_image via `stage_into_comfy_input(path)` and the
   portrait_ledger/AS-5 hash-stamps GLOBAL paths; the image-gate (image_done)
   and the 3D plan's routing expect ledger images[] paths to be stable.
   Concrete fix: write-TWICE is wrong, junction/copy is fragile -- the right
   move is: episode folder is the CANONICAL save target, ledger images[]
   carries the episode-local absolute path, and the global stills/ pool is
   retired ONLY after a grep proves zero readers assume it (the b7-style
   sweep). Pre-existing episode dirs must not break replay tooling.

4. [S1 prompts] The June-5 macro look came from FLUX-quality STILLS, but the
   statement says "prompts composed June-5-style" without pinning WHERE the
   subject table lives. Round-5 just moved open-subject wording INTO
   render_driver (_is_open branch). If the image node grows its own copy, the
   two drift -- the exact disease the gap audit just cured. Concrete fix: ONE
   shared helper (in `_otr_story_brief_helpers`, the only dep-free shared
   home) returning (subject, clauses) per role; BOTH the driver text path and
   the still-prompt node call it. Add a parity test: driver text prompt and
   still prompt for the same beat share the same leading subject.

5. [S4] Wiring the M4->HuMo seam here is scope creep INTO a working lane --
   but NOT wiring the scene still into the trace observability is a repeat of
   tonight's blindness. Split it: S4 stays a SEPARATE ticket; this spine only
   stamps `init_image`/`init_source` on trace rows for every beat (the
   mechanical face/still acceptance check), which is 10 lines and makes both
   projects auditable.

SHOULD-FIX:

6. [S5] Era-tail diet: trim is right but make it PER-SURFACE, not global --
   stills keep atmosphere line + palette top-2 + lighting top-2 (~120 chars);
   capped video prompts already self-trim. A global trim would change the
   portrait hashes mid-campaign for no visual reason.

7. [VRAM] Scene stills add N extra FLUX calls per episode (N = text-engine
   beats, 3-4 tonight). flux_gen1 at 20 steps is ~25s/still -- fine -- but the
   mint must run inside the image phase (before heavy video engines load),
   never lazily from the render driver mid-episode (lease contention with a
   resident HuMo/LTX would breach the 14.5GB ceiling).

8. [Determinism] Still seeds must come from the same request-hash scheme as
   shots (V-7), or render-twice stops being byte-comparable at the still
   layer and the soak determinism gate weakens.

9. [Acceptance] "operator eyeball: macro-radio June-5-style" needs a
   mechanical pre-check too: the still prompt leads with the shared subject
   helper output (string assert), the still file exists in the episode dir
   BEFORE the video phase (ordering assert), and at least one trace row
   carries init_source=scene_still.

CUT THESE (over-engineering):

10. Per-beat stills for EVERY beat. Only text-engine beats (open/announcer/
    outro) + portraits earn stills in v1; scene_broll/background can reuse
    the open still or stay text-only. Minting 6+ stills/episode buys latency,
    not look.
11. Any new workflow WIDGET for the spine. Policy lives in ImageDirector
    JSON + env; the json gets relinked only if a new node must enter the
    graph (and then in-place, per the operator's directive).

[ASSUMPTION] wan_i2v accepts an arbitrary 1472x832 (or 832x1216) still as
init without dimension snapping that crops the radio subject -- verify at
build with one probe clip before wiring the default.
