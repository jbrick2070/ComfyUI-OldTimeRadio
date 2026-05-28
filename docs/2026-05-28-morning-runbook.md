# Morning Runbook -- 2026-05-28

Overnight autonomous session delivered:

1. **Refined `docs/OTR_story_quality_build_plan.md`** -- audited against
   real source, corrected schema-field drift (Stage1Beat has no
   `speaker_role`; voicedness is `speaker != "MUSIC"`; field names are
   `length_target_words` + `emotional_register`). Each sprint is now a
   self-contained subagent contract with REVIEW / CODE / WIRE / REGRESS
   / COMMIT steps, file paths, function names, schema fragments, test
   paths, and a done-when boolean.
2. **`docs/2026-05-27-otr-quality-baseline.md`** -- Sprint 0 baseline
   doc. Three numbers to capture during the 5-episode soak are spec'd;
   the autonomous session could not read the live `pending_*` /
   `episodes/*` ledger files because they were not in the mounted
   workspace (only the 2026-05-19 pre-Wave-3 fixture was visible).
3. **Schema-drift finding** -- `init_lines_from_outline` lives in
   `nodes/production_ledger.py` (line 671), not `_otr_ledger.py`. It
   uses `getattr(beat, X, default)` so the Stage1Beat <-> outline.Beat
   field-name gap doesn't crash, but Sprint 1 has to wire
   `dialogue_slot_id` through this surface explicitly. The plan now
   documents two adapter options; the Sprint 1 owner picks.

**Not done autonomously** -- by design, because:
- No Desktop Commander cmd shell available in this session (so no
  `git push` per the CLAUDE.md cmd-only rule).
- No Windows `.venv` Python available (Linux sandbox can run pure-
  pydantic tests but not the torch / ComfyUI integration paths).
- No live ComfyUI run capability (Sprint 1's commit gate per the plan
  is "5/5 episodes prove full commit" -- operator-driven only).
- The Stage1Beat <-> ledger adapter question deserves a paragraph in
  the Sprint 1 commit body, not a blind autonomous edit.

---

## Morning order of operations

### Step 1 -- Pull the refined plan and confirm green baseline

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git status
git pull origin v2.0-alpha
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
```
Expected: pytest 3597 passed / 21 skipped / 0 failed, forbidden sweep
exits 0. If anything fails before any new code lands, hand the failures
back -- that's a pre-existing breakage.

### Step 2 -- Commit the refined plan + Sprint 0 baseline + this runbook

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git add docs\OTR_story_quality_build_plan.md docs\2026-05-27-otr-quality-baseline.md docs\2026-05-28-morning-runbook.md
```
Write to `.git\COMMIT_EDITMSG` via file tool:
```
docs: Sprint 0 baseline + refined Sprint 1-5 plan (subagent-executable)

- OTR_story_quality_build_plan.md re-audited against live source at
  bcfe8a5: corrected Stage1Beat field-name drift, mapped surfaces with
  line numbers, added a subagent-contract section per sprint.
- 2026-05-27-otr-quality-baseline.md scaffolds the three Sprint 1 soak
  numbers (rows_skipped, fallback_to_legacy count, rubber-stamp rate).
- 2026-05-28-morning-runbook.md captures the overnight findings + the
  morning order of operations.

Audit also surfaced a Sprint 1 adapter question: init_lines_from_outline
lives in production_ledger.py (not _otr_ledger.py) and reads beats via
getattr defaulting -- so wiring dialogue_slot_id needs a one-line getter
add in that method, NOT a separate _otr_ledger.py change. Plan documents
two adapter options; Sprint 1 owner picks and stamps the rationale in
the Sprint 1 commit body.

No code changes. Plan + baseline + runbook only.
```
Push:
```cmd
git commit -F .git\COMMIT_EDITMSG
git push origin v2.0-alpha
git log -1 --format=%H%n%s
```
Verify HEAD matches origin and no 0-byte files.

### Step 3 -- Execute Sprint 1 (start of the sprint cascade)

Read `docs/OTR_story_quality_build_plan.md` Sprint 1 section in full.

Decisions to make at the top:
1. **Adapter option A vs B** (see Sprint 1 CODE step 3 in the plan).
   Read `init_lines_from_outline` callers
   (`grep -rn init_lines_from_outline nodes/`). Pick the simpler one and
   stamp the rationale in the commit message.
2. **`MUSIC`-speaker beats not on ledger lines** -- confirm whether the
   ledger today has `lines[*]` rows for music_inter beats (Stage1Beat
   with `speaker == "MUSIC"`). If yes, the new
   `dialogue_slot_id` column is None for those rows. If no, no extra
   handling. The 2026-05-19 fixture has no music_inter rows; the live
   18-line episodes presumably do.

Then execute the Sprint 1 plan's CODE -> WIRE -> REGRESS steps top
to bottom. After regression passes, follow the COMMIT block verbatim.

### Step 4 -- Sprint 1 live soak (5 episodes)

For each episode, paste the three numbers into
`docs/2026-05-27-otr-quality-baseline.md` (or a successor doc dated
2026-05-28). If any episode shows `rows_skipped > 0` or
`fallback_to_legacy == true`, revert Sprint 1 and open a Bug Bible
candidate. Do NOT proceed to Sprint 2.

### Step 5 -- Sprint 2 through Sprint 5

Each sprint is its own self-contained section in the build plan. The
"subagent contract" block at the top of each sprint section describes
inputs, outputs, and acceptance criteria; a focused subagent can be
dispatched per sprint with that section as the entire spec.

Order is non-negotiable: 1 -> 2 -> 3 -> 4 -> 5.

---

## Open questions carried into the morning

These are flagged in the plan but worth surfacing here:

1. **Writer halt on news-brief exhaustion** -- Sprint 2 sub-section
   per Jeffrey's 2026-05-27 evening direction.
2. **`_otr_outline.Beat` mirror** -- whether to add `dialogue_slot_id`
   to Path A's Beat in Sprint 1 (for symmetry) or wait for Sprint 4's
   best-of-N to motivate it.
3. **`MUSIC`-speaker beats in lines[*]** -- handled defensively (None
   on music rows), but worth confirming during Sprint 1 soak.
