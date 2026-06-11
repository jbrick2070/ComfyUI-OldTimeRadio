# OTR Video Platform -- HANDOFF -- THE STILL-SPINE BUILD (2026-06-10 night)

> **CANONICAL LOCATION:** this in-repo file (`docs/VIDEO_BUILD_HANDOFF.md`) is
> the SINGLE git-tracked source of truth for the video build.

## ACTIVE MISSION (the only active build)

Build the **2D-still spine** from the BUILD-READY sprint plan
`docs/2026-06-10-still-image-spine/STILL_SPINE_SPRINT_PLAN.md` (tickets
ST-0..ST-8, seam map W1-W8, gotcha checklist). Panel record:
`docs/2026-06-10-still-image-spine/roundtable/` (2 passes, CONVERGED; Fable
sat as the 4th panelist). GOAL: the next 30-word production render opens on
6/5-quality macro-radio stills + shot-accurate in-character portraits, ALL
saved under `episodes/<ep>/stills/`, FEEDING the video engines as inputs
(kenburns + wan_i2v v1; LTX stays text-only by design), with `init_source`
trace proof -- expandable to 3D consumers later. Current step: **ST-0 probes**
(kenburns external-init; the render node's image_done gate), then ST-1..ST-8
in order, commit+push per green chunk.

## HARD RULES (copy verbatim)

- Do NOT start / resume / "continue" any other sprint -- NOT story-spine, NOT
  story-pipeline, NOT any audio sprint, NOT 3D (PARKED), NOT switchable-
  workflow (GO pending), NOT whiny-voice (GO pending), NOT the LTX-AV lane
  (P4, ANOTHER window owns it; never during this build).
- The audio refactor is SHIPPED; the audio script ledger is FROZEN
  (read-only). Byte-identical master audio + mux-LAST;
  `test_audio_byte_identical` stays GREEN at every step.
- Ignore any stale `session_handoff.md` and any memory / ROADMAP entry
  implying other "active" work.
- **GIT POLICY (operator, 2026-06-10 -- supersedes all older handoff lines):
  ONE branch v2.0-alpha; COMMIT AND PUSH together, every green chunk,
  immediately. The operator eyeball gates TAGS/promotions
  (v2.0-alpha-stable, prod, main) -- NEVER pushes. After every push verify
  HEAD==origin, no 0-byte files, no BOM, AST parse.**
- Invariants: single resident heavy engine <= 14.5 GB machine-NVML; BUG-291
  detach reclaim; LOUD fallbacks (log + ledger restamp, never silent);
  fail-soft never fail-episode; V-12 isolation; engine-agnostic (no model is
  "primary"); no new widgets beyond the planned json relinks (IN PLACE in
  `workflows/otr_scifi_16gb_full.json`, never a runner patch); UTF-8 no BOM;
  SFW. Suite (3863/0 + 28 skip baseline) + Bug Bible green at every commit.

## WHERE WE ARE (2026-06-10 night session; nothing invented)

- **Round 5 SHIPPED + PUSHED** (HEAD == origin @ `64e9411`; the old 13-commit
  push gate is DEAD -- everything is on GitHub): LTX frame cap + DECODE FLOOR
  169 (the installed wrapper's VAEDecode fails below ~169f at 1472x832 --
  tensor 256-vs-128; 169/233 proven), per-beat brief+beat LTX prompts +
  prompt sha8 trace + diversity gate (node-92 report), talking-head subject
  anchor + authored-prompt person gate, writer self-vocative ATTRIBUTION
  repair (last pre-freeze slot), shot-row char_id join + announcer id
  normalization, manifest positioned-mode start_s fallback, concrete
  radio-subject opens (narrative loglines render as murk -- operator catch),
  portrait GEAR SCRUB + three-quarter framing (negations planted the mic).
  Commits: 5b2012e, bdef529, 7ae7782, 379dd41, d087bfa (+1351d78 git policy,
  64e9411 still-spine docs).
- **Acceptance renders tonight**: ticking_lab (diversity proven; exposed the
  decode band), shattered_silencing (operator caught the logline murk),
  **alien_frequencies = ZERO fallbacks, ltx:4/humo:2, diversity 4/4 distinct,
  all nine gates green, byte-identical 6146e49a1c6b** -- awaiting the
  operator eyeball (gates the v2.0-alpha-stable TAG only).
- **Suite 3863/0** + Bug Bible green (canonical invocation). The optional
  `--pack-dir` deep scan carries 5 PRE-EXISTING findings (not from these
  commits).
- **Known open seams** (tickets, NOT this build): M4 creative prompts never
  reach HuMo requests in the live graph (cast beats render on the default
  studio prompt + portrait init -- the new trace observability proves it);
  era-tail full diet on video surfaces; global stills pool retirement; LTX
  img2vid probe.
- **The 6/5 reference** (operator north star): preserved legacy composer at
  `docs/2026-06-10-brief-downstream-gaps/legacy_otr_video_plan_e74a3ce.py.txt`;
  input-history comparison at
  `docs/2026-06-10-flux-and-ltx-input-comparison__last-week-vs-today.md`.
- Episode palette note: per-episode color (e.g. Mars = red) comes from the
  brief's era tail BY DESIGN; the still profile trims it, never deletes it.

## FIRST ACTIONS for the next session (then STOP for operator go)

1. Read `docs/2026-06-10-still-image-spine/STILL_SPINE_SPRINT_PLAN.md` FULLY
   (tickets + the W1-W8 seam map + the gotcha checklist), then the roundtable
   `pass01_plan.md` + `pass02_judgment.md` (the 7 folded items are binding).
2. `git status -sb` (expect clean, HEAD==origin on v2.0-alpha) + run the full
   suite + Bug Bible; confirm 3863/0 + green BEFORE coding.
3. Give the operator a 5-line summary of ST-0..ST-2 (the probes + helpers +
   schema) with their pass/fail asserts to prove comprehension.
4. No code until the operator confirms. Then build in ticket order,
   commit+push per green chunk, and finish with the ST-8 acceptance render +
   eyeball frames + STOP.

## PARKED -- not now

M4->HuMo seam (own ticket); LTX img2vid probe; global-pool retirement; 3D
toolkit (own plan); switchable-workflow S0-S6 (GO pending); whiny-voice P0-P4
(GO pending); LTX-AV lane (P4 -- other window); MuseTalk; RTXUpscale.
