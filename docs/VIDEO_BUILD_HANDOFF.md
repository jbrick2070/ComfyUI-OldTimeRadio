# OTR Video Platform -- HANDOFF -- CS-4 RESOLVED (1.7B default @ 955f134); SWEEP CLEAR TO RESUME; 0-E PHASE B AWAITS GO (2026-06-11 night)

> **CANONICAL LOCATION:** this in-repo file (`docs/VIDEO_BUILD_HANDOFF.md`) is
> the SINGLE git-tracked source of truth for the video build.

## ACTIVE MISSION (the only active build)

The forward order is `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` **section 0**,
which now carries LIVE STATUS + OPEN TICKETS + RUNWAY TO DONE (one-plan rule:
everything tracks THERE; read it first). **CS-4 IS RESOLVED** (2026-06-11 night,
coder window): NO code regression -- the umt5 TE stays 5,248 MB resident through
HuMo sampling (DEBUG VBAR evidence), which starves only the 16.5 GB 14B; the
operator flipped the default character tier to **humo_1.7B** (BUG-265 Option C
restored) @ `955f134`; ACCEPTANCE leg PASS 38 min on fixed HEAD (histogram
{ltx_video:3, humo_1.7B:3}, audio byte-identical, render-phase peak 10,305 MB).
**CURRENT STEP: resume item 4** (planner window) -- the sweep remainder on the
1.7B default (registry now enumerates 31 options / 27 runnable incl. the 0-E
engines: re-derive the leg list) + BOTH latentsync re-legs + the supervised wan
batch; then create `scripts\_otr_0e_gpu_go.txt` to release 0-E Phase B; then the
operator gates (look-QA, license sign-off, S-3D-0 green light), item 5 (3D
sprints), closing S3-S6. The 14B humo leg is OPERATOR-DEPRIORITIZED (opt-in only;
CS-4-open = the lazy post-encode TE-detach ticket). OPS GUARDRAIL: no heavy
parallel agent work (suites / multi-GB downloads / Blender) while a timed GPU
leg renders. Whiny-voice P0 audition stays an operator GPU sitting (plan v3.1
sec 5-7).

## HARD RULES (copy verbatim into the handoff on HAND OFF)

- The forward order is 3D-plan **section 0**. Do NOT start/resume/"continue"
  any OTHER sprint -- NOT story-spine, NOT story-pipeline, NOT the broader
  audio stack, NOT any other ROADMAP item. PARKED.
- The audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no
  `-shortest`); `test_audio_byte_identical` stays GREEN. The ONLY sanctioned
  audio work is the character-voice "whiny" fix
  (`docs/2026-06-10-character-voice-whiny-fix__problem-statement.md`) --
  UPSTREAM TTS only.
- EVERY session (planner AND coder) UPDATES the `otr-build-tracker` (content;
  preserve gauge + lanes styling). Never tell a window "don't touch the
  tracker". It is the durable roadmap.
- Ignore any stale `session_handoff.md` / memory / ROADMAP "active" entry.
  Section 0 + the tracker are the source of truth until the operator says
  otherwise.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100%
  local/offline; determinism (seed-keyed); every in-render fallback LOUD;
  UTF-8 no BOM; SFW; V-12 dependency isolation.
- v2.0 PRODUCTION / main is GATED until all the operator's work is done; a
  `v2.0-alpha-stable` tag on `v2.0-alpha` is fine; prod/main is NOT.
- GIT POLICY (operator 2026-06-10): ONE branch v2.0-alpha; commit AND push
  together per green chunk; the operator eyeball gates TAGS/promotions only;
  verify HEAD==origin / no 0-byte / no BOM / AST after every push.
- COORDINATION (operator 2026-06-11): ONE coder window in the repo's code at a
  time; the 0-E agent's Phase B + any new coder window serialize via the
  planner's GO file. Never run two coders in overlapping files.

## WHERE WE ARE (2026-06-11 night, post-CS-4 coder session; nothing invented)

- **CS-4 RESOLVED -- no code regression; 1.7B default shipped @ `955f134`.**
  Mechanism (DEBUG VBAR + the operator's BUG-291/265 pointer): the umt5 TE stays
  5,248 MB resident through HuMo sampling -- by design for the 1.7B stack (fits
  whole; 9.9-26 s/it flat), fatal for the 16.5 GB 14B (~10 GB budget -> per-step
  paging, 46->119 s/it idle, 153->1,788 under the 0-E agent's parallel box load;
  the 12-14 "healthy" class was always the 1.7B's). The legacy BUG-291 FLUX class
  is NOT re-opened (FLUX VBAR = 0 at HuMo entry; greps for the old EXIT-eviction /
  PHASE-C probe lines come back empty in every current-era log). 14B = selectable
  opt-in; fallback chain unchanged; CS-4-open = lazy post-encode TE-detach ticket.
  Full record: `docs/2026-06-11-coverage-sweep-triage__tickets.md`.
- **Sweep (item 4) CLEAR TO RESUME on the 1.7B default**: run 0 legs 1-6 PASS
  stand + the humo_1.7B leg = PASS x2 (33 min pre-flip; 38 min ACCEPTANCE on
  fixed HEAD, histogram {ltx_video:3, humo_1.7B:3}); 14B leg
  OPERATOR-DEPRIORITIZED. Registry enumerates 31 options / 27 runnable now (0-E
  engines aboard) -- re-derive the leg list. CS-1 = BOTH latentsync legs still
  re-run; CS-2 = phase attribution still open (note: the 1.7B leg's render-phase
  peak reads 10,305 MB vs the 14B-era ~16 GB pins -- partial answer); CS-3 =
  supervised wan batch post-resume.
- **0-E on-ramp**: CPU side SHIPPED @ a05dbda/3b535c7/1daaa6a (suite 4096/0;
  selectable-not-default; LICENSE_RECORD.md gates default-on). The follow-through
  agent is LIVE in Phase A; Phase B (E-1 probe, E-6 renders, per-engine sweep
  legs) HELD on the GO file.
- **One-plan consolidation**: section 0-E + LIVE STATUS/OPEN TICKETS + RUNWAY TO
  DONE (~6-9 sprints; S-3D-0/T2b shortcut forks -> ~2-3) live in section 0.
  Roundtable evidence: `docs/2026-06-11-comfy-native-3d-options/` (2 passes,
  ~$0.22, grounded on the live install -- hy3d-2mv core nodes verified present).
- **Git**: origin/v2.0-alpha @ `955f134` (CS-4 flip) + the wrap docs commit after
  it; HEAD==origin verified per push (no 0-byte / no BOM / AST clean). The planner
  docs edits that were working-tree rode the `955f134` chunk as directed.
- Track 3 = CLOSED (230fe4e..1571e0f); GATE B S0-S2 COMPLETE (6a1b716..230fe4e);
  stale-server shim RETIRED (fresh processes load tonight's code).

## FIRST ACTIONS for the next session (then STOP for operator go)

1. Read 3D_TOOLKIT_PLAN.md **section 0** (LIVE STATUS + RUNWAY first) + this
   handoff; skim the otr-build-tracker; `git log --oneline -12` + `git status`.
2. CS-4 is DONE (1.7B default @ 955f134; acceptance leg PASS 38 min). Resume
   item 4: re-derive the leg list from the registry (31 options / 27 runnable
   now), run the remainder on the 1.7B default -> BOTH latentsync re-legs ->
   the supervised wan batch; then create `scripts\_otr_0e_gpu_go.txt` to release
   0-E Phase B. Skip the 14B humo leg (OPERATOR-DEPRIORITIZED). Keep the box
   quiet during timed legs (the CS-4 ops guardrail).
3. State the CURRENT STEP per section 0 in <=5 lines; STOP for operator GO.

## PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale;
LTX-AV lane (own plan, gated); switchable S3-S6 (closing phase, AFTER 3D);
3D GPU lanes (T/G/W) until S-3D-0 + the operator green light.
