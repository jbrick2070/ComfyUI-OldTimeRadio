# R1: bug fixes first, then the randomizer proof, SFX on the fence

**Operator direction, 2026-08-02 morning:** "bug fixes are my big thing, if they
can be readily chased in story and video gen and general -- then confirm the
randomizer works -- and I'm on the fence about SFX."

State this plan starts from: 4 of 6 local engines PASS with published episodes
(`ltx_8gb`, `ltx_audio_in`, `humo`, `fastwan_8gb`). The two blockers are NOT
video-engine defects: one story-writer bug and one design collision.

---

## ARC 1 -- STORY-GEN BUG (the one blocking both remaining legs)

**The silent second character (PBUG-20260802-02).** The writer casts two
characters, writes dialogue for one; c03's rows ship with empty text. Killed
`wan_ti2v` at the parser (`CAST_MEMBER_SILENT`) and `ltx_video` at the freeze
gate (proof-coverage mismatch naming `shot_00N_b2/b4`) -- same fault, two doors.

Fix shape (root, not shim):
1. Composition must FILL the slots the skeleton allocated, or the skeleton must
   not allocate slots the composition won't fill -- find which side owns it.
2. A named gate right after composition: every cast member with an allocated
   slot has >= 1 non-empty line, failing as "character c03 never got dialogue"
   instead of a coverage mismatch five stages later.
3. Repair-ladder attempt on failure (it already retries 4x -- make the retry
   prompt SAY which character is silent).

Evidence to reuse: the two dead ledgers (`pending_20260802_024714`,
`pending_20260802_015532`) are on disk with the exact failure shape.

## ARC 2 -- VIDEO-GEN BUG (needs the operator ruling in Q1 below)

**The load-bearing mirror on `wan_ti2v`.** Cost model prices it at 24
affordable frames while the contract allows 177 single-clip, so the 25-177 beat
band renders ONLY via ping-pong -- which the no-mirror ruling forbids. Two
fixes, not exclusive:
- **A. Coverage-plan it** (matches the standing no-mirror ruling): add to
  `PLANNING_CAP_ENGINES` with a measured ceiling, rewrite the 3 protective
  tests, live leg. Precedent: fastwan, bit-identical base weights, benched FLAT.
- **B. Measure + correct the cost model** first: `(7000, 185)` predicts 22 GB
  where the bench measured 6.5 GB flat. A short 3-point measurement leg
  (17/49/81 frames, read NVML peaks) replaces the guess with a curve. If the row
  is honest, the predictor may afford whole beats and A shrinks or vanishes.

Then: `ltx_video` needs NO video fix -- re-run it once Arc 1 lands.

## ARC 3 -- GENERAL BUGS (known, sized, no ruling needed)

1. **Beat-hoist churn**: 391 model loads for 62 renders, teardowns "detached 0"
   -- the session holds nothing. Biggest render-time win available.
2. **Scratch hygiene**: 893 files / 5.9 GB in `episodes/_shared/tmp`; janitor
   only sweeps after a successful publish, so failed legs leak forever. The test
   suite also writes real mp4s there.
3. **Survival-guide push** (operator: `git pull --rebase && git push`).

## ARC 4 -- RANDOMIZER PROOF (cheap, mostly already earned)

Live evidence tonight: 4 episodes rolled 3 distinct banks + 4 distinct styles
(shakespeare x2, scifi_news, original / anime, storybook_engraving, video_art,
archival_documentary). The roll works. What's unproven is COVERAGE:
- CPU-only audit: enumerate the eligible pools, then N seeded rolls asserting
  every eligible bank/style is reachable and weights aren't degenerate.
- No GPU legs needed unless a specific bank has never once been rolled live.

## ARC 5 -- SFX (fence; not planned until Q4 answers)

If IN: scope = per-beat foley/ambience via the existing audio bed, no new
model downloads, no cloud. If OUT: park with a one-line note in the ledger spec
so downstream consumers stop reserving fields for it.

## OPERATOR RULINGS (2026-08-02 morning, all four questions answered)

1. **Arc 2 (wan_ti2v):** kibitz FIRST for the solution -- a proven workflow
   recipe -- THEN measure the coverage. Panel before code, measurement before
   any constant.
2. **Arc 1 (story):** root fix + named gate. GO.
3. **Arc 4 (randomizer):** CPU audit only. GO.
4. **Arc 5 REVISED:** SFX decision is deferred BEHIND a new arc -- **a 45-word
   all-CLOUD-videos campaign** -- which itself only runs after we have a good
   grounding on the cloud engines' multi-clip beat math ("in this new
   'each video path gets its own still logic' as well"). So the ladder is:
   local 6/6 -> ground cloud multi-clip beat math + per-path still logic ->
   45-word all-cloud campaign -> THEN the SFX call. The r2 panel's cloud
   findings (contracts declaring ranges the provider controls) are the starting
   evidence for that grounding, banked in the contract-vs-runtime audit doc.

## ORDER AND GATES

Arc 1 -> re-run `ltx_video` + `wan_ti2v` legs -> Arc 2's ruling applied ->
6/6 proven -> Arc 3 in parallel where GPU-free -> Arc 4 audit -> Arc 5 if in.
Every arc: kibitz per standing order; suite + Bug Bible green per chunk;
commit-and-push per green chunk. The 6/6 live proof remains the bar ("local
humming like a beauty, no errors") before any cloud work.
