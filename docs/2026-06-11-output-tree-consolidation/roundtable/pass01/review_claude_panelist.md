# Claude panelist review (independent, pre-panel)

**Placement: Option A — `otr/.system/{cache,tmp,state}` + the Windows HIDDEN attribute.**
Defense: a dot-prefix alone does NOT hide a folder in Explorer — set `attrib +h` on
`.system` in the migration so the operator's Explorer view literally shows episodes +
obs only (the directive's spirit). A beats B (underscore dirs inside episodes/ pollute
the episode listing AND every "enumerate episode dirs / find newest ledger" codepath
must learn to skip them = regression surface). A beats C on two hard facts: the stills
cache is content-addressed FOR cross-episode/leg reuse (C re-renders = real GPU cost),
and OTR_TMP is a LAUNCH-TIME env consumed by ffmpeg children — it cannot be
per-episode (no episode exists at boot). Same-volume os.replace preserved (same tree).
OBS untouched (obs/ flat).

**Enforcement (4 layers, all LOUD):**
1. pytest: top-level entries under otr/ must be ⊆ {episodes, obs, .system}; zero
   top-level files.
2. The capstone/sweep HYGIENE GATE adds the same assertion per leg (fail the leg).
3. _otr_paths.py contract test: every exported helper's resolved path must fall under
   the sanctioned set (parametrize over __all__).
4. Bypass audit: extend the existing b7-style forbidden-sweep AST test to ban
   hardcoded `otr/<dir>` joins outside _otr_paths.py (the single-authority rule made
   testable).

**Migration order:** (m1) ONE code chunk: add .system helpers; repoint otr_state_dir /
stills-cache helper / OTR_TMP (launch cmd) ; update both hygiene gates same chunk.
(m2) Janitor: stale-tmp sweep (age >48h) at server boot + post-episode publish; cache
files MOVE (names preserved — content-addressed so a rebuild is also acceptable);
state files MOVE (7 files). (m3) Debris: move the dead set (audio 7.7GB, videos,
script_gates, blend_test, _legacy_stills, qa_*, portraits, aship*, _lane1) to
`output/otr_attic_<date>/` OUTSIDE otr/ — operator eyeballs, deletes at leisure;
NOTHING auto-deletes.

**Sequencing:** drain the running queue first (sweep legs + wan + 0-E Phase B baseline
hygiene against the CURRENT tree; mid-queue path flips invalidate legs and violate the
quiet-box guardrail). Land OH-1 immediately after, BEFORE item 5 — the 3D sprints add
new writers and must be born under the locked contract. Size: 1 coder chunk (paths +
gates + launch env) + 1 ops chunk (janitor + migration + attic). Risks to verify at
build: any reader that LISTS otr/ top-level (assumes old dirs); the ledger
stills_dir re-resolve path; OTR_GPU_LEASE_DIR consumers.

**Ticket cut:** OH-1 contract+helpers+gates; OH-2 launch env + hygiene gates update;
OH-3 janitor + stale-tmp policy; OH-4 attic migration + operator-approved delete;
OH-5 bypass-audit AST test; OH-6 section-0/tracker/docs. (OH-1/2/5 = one coder chunk
realistically; OH-3/4 = the ops chunk.)
