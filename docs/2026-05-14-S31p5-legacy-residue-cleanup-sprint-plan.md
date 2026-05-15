# S31.5 — Legacy residue cleanup (Cowork loop, pytest-only)

> **What this is:** Post-S31 cleanup sprint. Sweeps the legacy residue revealed by S31's clean break — stale test files, vestigial naming, leftover delegation wrappers, dead comments referencing deleted symbols. Nothing functional changes; the codebase just stops carrying ghosts.

**Status:** PLANNED.
**Branch:** `s31p5-legacy-residue-cleanup`. Cut from `s31-loader-clean-break` @ B8 (`2b0b6dc`).
**Sequencing:** S31 → **S31.5 (this sprint)** → S32 → S33 → Sprint C → Sprint A.
**S32 B0 revert required at B0:** S32 B0 already landed at `655dd6a` on `s32-helper-per-subpass-routing`. Before cutting S31.5, revert that single commit so S32 picks up cleanly from a fully-clean baseline once S31.5 closes. Single revert, trivial.

**Loop per commit:** review → code → wire (none, mostly) → pytest → commit → push. No ComfyUI execution. No operator gates.

---

## Hard rules (continuity from S31+S32)

1. **Audio C7 byte-identical (pytest proxy)** must hold at every commit boundary. S31.5 changes no behavior; if C7 regresses, you accidentally changed runtime code instead of cleanup. STOP, revert.
2. **No legacy back-compat reintroduced.** S31.5 deletes residue; it does not resurrect anything S31 deleted.
3. **No new generate or lifecycle surfaces.** Hard rules #5 and #6 from S31 stay in force.
4. **Bug Bible regression** 23 / 1 / 2xf at every commit boundary.
5. **No separate change logs.** Updates flow to `BUG_LOG.md` and `ROADMAP.md`.
6. **Tests written before fixes** for any structural defect. Red-on-parent, green-on-fix.
7. **Forbidden-pattern sweep** stays at 0 runtime hits.
8. **No version-label bumps.**

---

## Canonical pytest run

Same as S31+S32 canonical subset, plus any new test files created in this sprint. Match the existing pattern in `docs/2026-05-14-S31-S32-cowork-execution-plan.md`.

After every commit:
```cmd
git diff <parent-branch> -- "*.py" | Out-File -Encoding utf8 docs\s31p5_diff_tmp.txt
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
```

**Pytest baseline:** 243/7/2 (S31 B8 close). Target at S31.5 B7 close: same or slightly LOWER if BUG-LOCAL-227 triage deletes orphaned tests. Test deletion is allowed and expected here; the count matters less than the surface getting cleaner.

---

## Inventory of residue to clean (gathered from S31 final QA review)

| # | Item | Source | Action |
|--:|---|---|---|
| 1 | 25 LFC test failures latent at S30 B8 (PRE-EXISTING, BUG-LOCAL-227) | Forward work, S31 QA | Triage in B1: delete stale tests referencing deleted Phase 3/4/5/6, refactor to test surviving surface, or add to `EXPECTED_FAILED_NODEIDS` with rationale |
| 2 | `tests/test_b4b_rss_rewire.py` — filename references S30 B4b, content collapsed to deletion guards | Deviation #5, S31 QA | Fold remaining assertions into `tests/test_no_orchestrator_legacy_symbols.py`; delete the file |
| 3 | `tests/test_unload_synchronize_guard.py` — collapsed to 1 deletion guard, original BUG-LOCAL-073 purpose retired | Deviation #5, S31 QA | Fold the single guard into `tests/test_no_orchestrator_legacy_symbols.py`; delete the file |
| 4 | `_vram_cleanup_via_loader` wrapper in `story_orchestrator.py` | Deviation #3, S31 QA | Audit: can `register_vram_cleanup` take `_otr_model_loader.unload_llm` directly? If yes, eliminate wrapper. If no, document why and keep |
| 5 | Stale comments in `nodes/_otr_model_loader.py` referencing `_LLM_CACHE` rebind workaround (added at B2, made obsolete at B4) | Deviation #1, S31 QA | Comment sweep; remove `_LLM_CACHE`-rebind doc fragments |
| 6 | Stale comments in `nodes/story_orchestrator.py` referencing deleted symbols (B4b history, post-B4 cleanup) | inferred from B4 deletion scope | Comment sweep |
| 7 | Stale docstring in `nodes/_otr_model_loader.load_llm` mentioning the cache-hit short-circuit (deleted at B4 per Deviation #2) | Deviation #2, S31 QA | Docstring sweep |

---

## Commit structure (B0 → B7)

### B0 — S32 B0 revert + branch cut + plan landing (~0.25 d)

**Review.** Confirm `s31-loader-clean-break` @ `2b0b6dc` is the parent (S31 B8 close). Confirm `s32-helper-per-subpass-routing` @ `655dd6a` is the orphan S32 B0 to be reverted. Confirm clean working tree.

**Code (in order).**

1. **Revert S32 B0** on `s32-helper-per-subpass-routing`:
   ```cmd
   git checkout s32-helper-per-subpass-routing
   git revert 655dd6a --no-edit
   git push origin s32-helper-per-subpass-routing
   ```
   The S32 branch now sits one commit ahead of S31 B8 (the revert commit). When S31.5 closes and S32 picks back up, S32 B0 gets re-landed fresh against the now-clean codebase.

2. **Cut S31.5 branch** from S31 B8:
   ```cmd
   git checkout s31-loader-clean-break
   git checkout -b s31p5-legacy-residue-cleanup
   ```

3. **Land this plan** at `docs/<date>-S31p5-legacy-residue-cleanup-sprint-plan.md`.

4. **Push branch** to origin.

**Wire / Pytest.** Baseline-roll recorded in commit message (canonical pytest count from S31 B8: 243/7/2).

**Commit subject.** `B0: S32 B0 revert + S31.5 branch cut + plan landing`

---

### B1 — BUG-LOCAL-227 triage: 25 LFC stale tests (~1 d)

**Review.** Run wide pytest walk to surface the 25 failures:
```cmd
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests\ -q --tb=no 2>&1 | findstr "FAILED"
```

For each of the 25 failures, classify into one of three buckets:
- **Bucket A — references deleted Phase 3/4/5/6 / VisualLLMSelector / LedgerScriptReviewer / Director:** test is dead-symbol-collateral; DELETE the test (or the whole test file if all tests in it are dead).
- **Bucket B — references a surviving phase but assertion is stale:** REFACTOR to test the current surface.
- **Bucket C — genuinely flaky / environment-dependent / out-of-scope:** add to `EXPECTED_FAILED_NODEIDS` (or pytest skip marker) with a comment explaining why.

Document the classification in the commit message — each of the 25 failures gets a one-line decision.

**Code.** Apply the classifications. Most likely outcome: 18-22 tests delete (Bucket A dominates, since S30 deleted 4 phases + 1 node class + 1 reviewer-style node), 2-5 refactor (Bucket B), 0-3 skip (Bucket C).

**Wire.** None.

**Pytest.** Wide tests/ walk after cleanup: confirm count drops to 0 unexpected failures. Canonical subset: confirm still green. Audio C7: must hold (test deletion can't perturb runtime).

**Commit gate.** Wide pytest walk shows 0 unexpected failures. Canonical subset still green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.

**BUG_LOG update.** Mark BUG-LOCAL-227 as `[FIXED <B1 hash> <date>]` with a one-line note: "Triaged: N deleted (Bucket A), M refactored (Bucket B), K skipped with rationale (Bucket C)."

**Commit subject.** `B1: BUG-LOCAL-227 triage — N+M+K classifications applied to 25 latent LFC failures`

---

### B2 — vestigial test file consolidation (~0.5 d)

**Review.** Open `tests/test_b4b_rss_rewire.py` and `tests/test_unload_synchronize_guard.py`. Confirm their current contents are entirely deletion guards / safe-invalidation assertions (no original-purpose tests remain).

**Code.**

1. Move any surviving assertions into `tests/test_no_orchestrator_legacy_symbols.py` under a clearly-labeled section (e.g., `# Folded in from test_b4b_rss_rewire.py @ S31.5 B2 — S30 B4b deletion guards`).
2. Delete `tests/test_b4b_rss_rewire.py`.
3. Delete `tests/test_unload_synchronize_guard.py`.
4. Update any `pytest.ini` / `pyproject.toml` collection patterns if they enumerate these files explicitly (probably not — pytest usually discovers).

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| (folded assertions retained) | `tests/test_no_orchestrator_legacy_symbols.py` | Same deletion-guard assertions, just relocated |
| `test_b4b_rss_rewire_file_does_not_exist` | new at end of `test_no_orchestrator_legacy_symbols.py` | Sanity: `not (REPO_ROOT / "tests" / "test_b4b_rss_rewire.py").exists()` |
| `test_unload_synchronize_guard_file_does_not_exist` | same | Same shape for second file |

**Commit gate.** Folded tests green. Two new "file does not exist" guards green. Canonical pytest count adjusts by -2 files (counts the assertions, not the file count, so should net flat or +small). Audio C7 holds. Forbidden sweep clean.

**Commit subject.** `B2: consolidate vestigial test files (test_b4b_rss_rewire, test_unload_synchronize_guard) into test_no_orchestrator_legacy_symbols`

---

### B3 — `_vram_cleanup_via_loader` wrapper audit (~0.5 d)

**Review.** Open `nodes/story_orchestrator.py` and find `_vram_cleanup_via_loader` (added at S31 B4 per Deviation #3). Read `register_vram_cleanup` — what's its callable signature requirement?

Two outcomes possible:

**Outcome A — `register_vram_cleanup` can take `_otr_model_loader.unload_llm` directly:** the wrapper is pure delegation overhead. ELIMINATE.

```python
# OLD (B4 state)
def _vram_cleanup_via_loader():
    try:
        from . import _otr_model_loader
        _otr_model_loader.unload_llm()
    except Exception as exc:
        log.debug(...)

register_vram_cleanup(_vram_cleanup_via_loader)

# NEW (if Outcome A)
from . import _otr_model_loader as _otr_loader_mod
register_vram_cleanup(_otr_loader_mod.unload_llm)
```

**Outcome B — `register_vram_cleanup` needs a no-arg callable with try/except wrapping (e.g., because it's called from signal handlers or cleanup callbacks that must never raise):** the wrapper is structurally required. KEEP, document the requirement in a docstring.

**Code.** Whichever outcome lands.

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_register_vram_cleanup_path` | new or extend existing | If Outcome A: no `_vram_cleanup_via_loader` symbol in `story_orchestrator`. If Outcome B: symbol exists with the try/except pattern. |

**Commit gate.** Test green. Audio C7 holds. Forbidden sweep clean.

**Commit subject (Outcome A).** `B3: eliminate _vram_cleanup_via_loader wrapper — register_vram_cleanup takes _otr_model_loader.unload_llm directly`

**Commit subject (Outcome B).** `B3: document _vram_cleanup_via_loader structural requirement — register_vram_cleanup needs no-raise callable`

---

### B4 — stale comment + docstring sweep (~0.5 d)

**Review.** Pre-grep for stale references:

```cmd
findstr /s /n "_LLM_CACHE rebind" nodes\*.py
findstr /s /n "B4b" nodes\*.py
findstr /s /n "cache-hit short-circuit" nodes\*.py
findstr /s /n "legacy _load_llm" nodes\*.py
findstr /s /n "delegates to story_orchestrator" nodes\*.py
findstr /s /n "ported from" nodes\*.py
```

Each hit either:
- (a) describes current behavior accurately → KEEP
- (b) describes deleted/changed behavior → DELETE the stale comment/docstring fragment
- (c) historically useful for understanding → REWRITE to current tense

**Code.** Apply comment / docstring edits. Common targets:
- `nodes/_otr_model_loader.load_llm` docstring: drop `_LLM_CACHE`-rebind workaround language (Deviation #1)
- `nodes/_otr_model_loader.load_llm` docstring: drop cache-hit short-circuit references (Deviation #2)
- `nodes/story_orchestrator.py` module-level comment about LLM stack: update to "all LLM lifecycle lives in `_otr_model_loader`"
- Inline comments referencing `_LLM_CACHE`, `_load_llm`, `_unload_llm`, `_generate_with_llm` as live symbols → delete or rewrite

**Wire.** None.

**Pytest.** None new — comment edits don't need tests. Run canonical subset to confirm nothing broke.

**Commit gate.** Canonical subset green. Audio C7 holds. Forbidden sweep clean (the deleted-symbol names in comments are forensic per the sweep's tokenize classifier, so they should already be marked forensic — but verify post-sweep that runtime hits stay 0).

**Commit subject.** `B4: stale comment + docstring sweep — remove references to deleted symbols, B4b history, cache-rebind workarounds`

---

### B5 — forbidden-pattern sweep verification (~0.25 d)

**Review.** Confirm S31 B5's 6 extinction markers still hold post-S31.5 changes. No new markers needed unless B1-B4 deleted symbols not already covered.

**Code.** Likely zero changes to `docs/_s28_forbidden_sweep.py`. If B3 eliminated `_vram_cleanup_via_loader`, optional: add a sweep marker for it (`\b_vram_cleanup_via_loader\b`) — judgment call, since the name is so specific that accidental reintroduction is unlikely.

**Pytest.** Manual sweep: confirm 0 runtime hits across the S31.5 diff.

**Commit gate.** Sweep clean. Bug Bible 23/1/2xf.

**Commit subject.** `B5: S31.5 forbidden-pattern sweep verification — 0 runtime hits`

---

### B6 — round-robin integration buffer (~0.5 d, variable)

Same A/B/C/D rules as S31 B7 and S32 B7. If no findings, empty/skipped.

**Commit subject.** `B6: round-robin integration — <summary>` or `B6: empty, skipped`

---

### B7 — sprint close (~0.5 d)

**Review.** Mirror S31 final QA format. File at `docs/<date>-S31p5-final-qa-review.md`.

**Code.** Final QA review file. ROADMAP refresh — mark S31.5 closed. BUG_LOG confirm BUG-LOCAL-227 marked FIXED at B1.

**Wire / Pytest.** Full clean canonical run.

**Commit gate.** All acceptance rows green. Audio C7 held at every B1-B5 boundary. Branch pushed.

**Commit subject.** `B7: Sprint S31.5 close — legacy residue cleared, BUG-LOCAL-227 FIXED, codebase ghost-free`

---

## S31.5 acceptance table

| # | Check | Target |
|--:|---|---|
| 1 | Canonical pytest count | green (count may shift due to deletions in B1, B2) |
| 2 | Wide `pytest tests/` walk | 0 unexpected failures (BUG-LOCAL-227 triaged) |
| 3 | Bug Bible regression | 23 / 1 / 2 |
| 4 | Audio C7 byte-identical (pytest proxy) | holds B1 → B7 |
| 5 | Forbidden sweep | 0 runtime hits |
| 6 | `tests/test_b4b_rss_rewire.py` exists | False (deleted at B2) |
| 7 | `tests/test_unload_synchronize_guard.py` exists | False (deleted at B2) |
| 8 | Surviving assertions from #6/#7 present in `test_no_orchestrator_legacy_symbols.py` | True |
| 9 | `_vram_cleanup_via_loader` audit complete | Outcome A (eliminated) OR Outcome B (documented) |
| 10 | Stale comments referencing `_LLM_CACHE`/`B4b`/cache-hit short-circuit | 0 in nodes/*.py runtime code |
| 11 | `load_llm` docstring describes current (post-B4) behavior accurately | manual review confirms |
| 12 | BUG-LOCAL-227 | FIXED at B1 with classification log |
| 13 | New sweep markers added (if any) | optional, ≤ 1 |
| 14 | ROADMAP refreshed | S31.5 marked closed |

---

## Out of scope for S31.5

These are listed in S31 forward work but are NOT in S31.5 scope:

- **Loader API consolidation** — splitting `_otr_model_loader.py` into `{__init__.py, bnb_profiles.py, generation.py}`. This is a structural refactor, not residue cleanup. Keeps its own sprint.
- **UNGATED_PASS_RECOMMENDATION post-soak landing** — blocked on ungated PASS-tier soak validation. S31.5 doesn't touch the catalog.
- **Audio-intentional sprint** — model-author `generation_config.json` respect. Independent theme.

## Out of scope (long-term)

S33 (editor-only cleanup phases — retire cascade Phase 1 + Phase 9 auditors) sequences AFTER S32 closes, per the locked sprint sequence: S31 → S31.5 → S32 → S33.

**After S31.5 closes:** Cowork resumes S32 by re-landing S32 B0 fresh against the clean baseline, then proceeds B1 → B8.

---

## Sources

- `docs/2026-05-14-S31-final-qa-review.md` — S31 close + Deviations #1-5 + Forward work (BUG-LOCAL-227 + register_vram_cleanup wrapper + comment debt).
- `docs/2026-05-14-S31-S32-cowork-execution-plan.md` — format mirror + Hard rules continuity.
- `BUG_LOG.md` — BUG-LOCAL-227 entry (carried from S31 B1 surfacing).
- `CLAUDE.md` — standing rules.
