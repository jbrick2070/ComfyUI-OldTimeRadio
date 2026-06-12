# OTR Video Platform -- HANDOFF -- CS-4 CRITICAL PATH; SWEEP PAUSED 6/16; 0-E AGENT LIVE (2026-06-11 late)

> **CANONICAL LOCATION:** this in-repo file (`docs/VIDEO_BUILD_HANDOFF.md`) is
> the SINGLE git-tracked source of truth for the video build.

## ACTIVE MISSION (the only active build)

The forward order is `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` **section 0**,
which now carries LIVE STATUS + OPEN TICKETS + RUNWAY TO DONE (one-plan rule:
everything tracks THERE; read it first). **CURRENT STEP: fix CS-4** -- new-code HuMo
sampling thrash, the critical path blocking item 4's sweep remainder, the wan batch,
0-E Phase B, AND the operator look-QA. Then finish item 4 (9 legs + BOTH latentsync
re-legs + the supervised wan batch), then the operator gates (look-QA, license
sign-off, S-3D-0 green light), then item 5 (3D sprints) and closing S3-S6.
CORRECTION vs the prior handoff: the sweep is NOT finished -- the `done:true` was a
per-batch artifact; real state = 6/16 PASS (run 0, old code), runs 1-2 VOIDED by CS-4.
PARALLEL: the 0-E follow-through agent runs Phase A (Blender 4.5.10 pinned + cube
self-test PASSED + ckpt/depth fetches) and idles polling
`scripts\_otr_0e_gpu_go.txt`; the PLANNER creates that file ONLY after CS-4 + sweep +
wan land. Whiny-voice P0 audition stays an operator GPU sitting (plan v3.1 sec 5-7).

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

## WHERE WE ARE (2026-06-11 day/late planner session; nothing invented)

- **CS-4 (CRITICAL): post-overnight code thrashes HuMo sampling** -- 153->1,788
  s/it vs the 14-18 baseline; repro'd on a CLEAN env + idle box; r5d same-install
  baseline (7.2 s/it) pins it to the code era; WanTE/WanVAE staging = HuMo-normal
  (run-1's superset attribution RETRACTED). Bisect set + repro in
  `docs/2026-06-11-coverage-sweep-triage__tickets.md` (suspect #1 = 1ef6786, the
  63->87 master-json gate wire; the json edit travels WITH the code checkout --
  A/B them together). A thrash-era headless server may be on :8000 for repro.
- **Sweep (item 4) PAUSED 6/16**: run 0 (old code) legs 1-6 PASS; CS-1 = run-0's
  latentsync "PASS" was fallback-only (stale code; attempts trace proves the LOUD
  chain worked) -- BOTH latentsync legs re-run post-fix; CS-2 = machine NVML
  ~16 GB pin vs the 14.5 ceiling while driver-phase reads 3.1-3.5 GB (phase
  attribution needed); CS-3 rescoped = wan+humo 16gb coexistence unproven
  (supervised wan batch post-fix).
- **0-E on-ramp**: CPU side SHIPPED @ a05dbda/3b535c7/1daaa6a (suite 4096/0;
  selectable-not-default; LICENSE_RECORD.md gates default-on). The follow-through
  agent is LIVE in Phase A; Phase B (E-1 probe, E-6 renders, per-engine sweep
  legs) HELD on the GO file.
- **One-plan consolidation**: section 0-E + LIVE STATUS/OPEN TICKETS + RUNWAY TO
  DONE (~6-9 sprints; S-3D-0/T2b shortcut forks -> ~2-3) live in section 0.
  Roundtable evidence: `docs/2026-06-11-comfy-native-3d-options/` (2 passes,
  ~$0.22, grounded on the live install -- hy3d-2mv core nodes verified present).
- **Git**: origin/v2.0-alpha @ 1daaa6a (+ anything the live agents have pushed
  since). Planner docs edits (3D plan section-0 blocks, triage doc, this handoff,
  roundtable docs) are WORKING-TREE; they ride the next coder commit+push.
- Track 3 = CLOSED (230fe4e..1571e0f); GATE B S0-S2 COMPLETE (6a1b716..230fe4e);
  stale-server shim RETIRED (fresh processes load tonight's code).

## FIRST ACTIONS for the next session (then STOP for operator go)

1. Read 3D_TOOLKIT_PLAN.md **section 0** (LIVE STATUS + RUNWAY first) + this
   handoff; skim the otr-build-tracker; `git log --oneline -12` + `git status`.
2. Check CS-4: if the fix landed (humo smoke in the 14-18 s/it class + one sweep
   leg PASS <= 50 min), resume item 4 (9 legs -> latentsync re-legs -> supervised
   wan batch), then create `scripts\_otr_0e_gpu_go.txt` to release 0-E Phase B.
   If NOT fixed: the CS-4 kickoff (triage doc / tracker foot) goes to a FRESH
   coder window -- never alongside another coder in the same files.
3. State the CURRENT STEP per section 0 in <=5 lines; STOP for operator GO.

## PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale;
LTX-AV lane (own plan, gated); switchable S3-S6 (closing phase, AFTER 3D);
3D GPU lanes (T/G/W) until S-3D-0 + the operator green light.
