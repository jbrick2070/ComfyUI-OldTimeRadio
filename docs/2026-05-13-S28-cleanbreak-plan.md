# S28 Cleaner Break — autonomous extinction sprint

**Branch:** `s28-cleaner-break` cut from `s27-cleanbreak-tail` HEAD `4277952`
**Posture:** Zero tolerance. No legacy paths. No deferrals. No mid-sprint hand-backs. Cowork ships every phase autonomously.
**Closure rule:** every surface in this plan extincts inside S28. `docs/cleanbreak-deferred.md` empty at close.
**Arbiter:** `tests/v2/test_audio_byte_identical.py` (Prime Directive 1). If byte-identity holds at every commit, the cleanbreak is correct. If a single commit breaks it, that commit reverts and the producer trace goes one level deeper.

---

## Surfaces

| Phase | Target | Files | Risk |
|---:|---|---|---|
| 0 | Baseline capture | repo-wide grep | LOW |
| 1 | `otr_legacy_audio_dir()` + legacy flat-layout ledger walker | 8 nodes + `_otr_paths.py` + `_otr_ledger.py` | LOW |
| 2 | `req.budget is None` legacy paths | `_otr_outline.py` + ~20 test fixtures | MEDIUM |
| 3 | `_otr_line_composer.py` legacy caller-shape tolerance (4 sites) | `_otr_line_composer.py` + producer side | MEDIUM |
| 4 | `_otr_ledger_freeze.py` legacy ledger-shape tolerance (4 sites) | `_otr_ledger_freeze.py` + producer side | HIGH (audio) |
| 5 | Final static verification | full repo | gate |

---

## Auto-decision policy (Cowork applies these without asking)

Every decision Cowork encounters inside S28 resolves under one of these rules.

**A. Should field X be always populated by its producer?**
YES. The v2.0 contract is fully-shaped data. If a consumer guards
against absent X, the producer is leaking — fix the producer first,
then delete the consumer guard.

**B. Should the freeze cascade emit field X for non-dialogue lines (music, sfx-as-line, narration)?**
YES. Uniform shape across all line types. No type-specific absence.

**C. A test asserts that a consumer tolerates a legacy shape on purpose.**
The assertion is wrong. Either flip it (assert the consumer rejects
the legacy shape) or delete the test if redundant with the
producer-side contract guarantee. Never keep the legacy-tolerance
assertion.

**D. Producer fix would be a one-line edit.**
Apply it. Commit it as `fix(s28-pN-producer-M): <producer> always populates <field>`.

**E. Producer fix would touch the audio path.**
Apply it. Run audio-byte-identical immediately. If byte-identity
holds, proceed. If it breaks, revert and trace one level further
upstream.

**F. `audio-byte-identical` fails after a deletion commit.**
Revert that commit. Trace the producer ONE level further up the
chain. Fix the deeper producer. Re-run the deletion. Repeat until
byte-identity holds. The worst case is the leak chains back to
`OTR_LedgerScriptWriter` itself — fix it there, then proceed.

**G. A producer fix would require touching `_otr_ledger.py` core
(`set_lines`, `set_music`, `set_clips`).**
In scope. Apply the fix. The cleaner break extinguishes legacy on
both sides of every contract, including the ledger core.

**H. Pytest fails for a reason that is not audio byte-identity.**
Read the failure. If it's a test asserting legacy behavior (rule C),
flip or delete. If it's a real production regression, fix the
production code in the same commit that caused it. Do not advance
to the next phase with red tests.

---

## Phase 0 — Baseline

```
git checkout v2.0-alpha
git pull origin v2.0-alpha
git checkout s27-cleanbreak-tail
git log -1 --format=%H                 # must equal 4277952
git status --short                     # must be empty
git checkout -b s28-cleaner-break

pytest tests/ -q --tb=line > docs/2026-05-13-S28-baseline-pytest.txt
# expect: 2145 passed, 8 skipped, 0 failed

git grep -nE 'otr_legacy_audio_dir|def otr_legacy_audio_dir' nodes/ \
  > docs/2026-05-13-S28-baseline-footprint.txt
git grep -nE 'd\.glob.*_ledger\.json' nodes/_otr_ledger.py \
  >> docs/2026-05-13-S28-baseline-footprint.txt
git grep -nE 'req\.budget is None|budget is None' nodes/_otr_outline.py \
  >> docs/2026-05-13-S28-baseline-footprint.txt
git grep -nE 'back-compat|legacy fallback|legacy shape' nodes/_otr_line_composer.py \
  >> docs/2026-05-13-S28-baseline-footprint.txt
git grep -nE 'back-compat|legacy fallback|legacy shape' nodes/_otr_ledger_freeze.py \
  >> docs/2026-05-13-S28-baseline-footprint.txt
```

Commit: `docs(s28): baseline pytest + footprint`.

---

## Phase 1 — `otr_legacy_audio_dir()` + flat-layout walker

### Step 1.1 — Migrate 13 caller sites

For each file: drop `otr_legacy_audio_dir` from the import line, drop
the call-site fallback entry, replace with one forensic comment.

| File | Lines | Edit |
|---|---:|---|
| `nodes/_otr_ledger.py` | 328 | `[_P.otr_episodes_root(), _P.otr_legacy_audio_dir()]` → `[_P.otr_episodes_root()]` |
| `nodes/audio_enhance.py` | 434 | inline import + audio_dirs list — drop both |
| `nodes/batch_audiogen_generator.py` | 33 | top import — drop the symbol |
| `nodes/batch_bark_generator.py` | 33 | top import — drop the symbol |
| `nodes/batch_humo_render.py` | 65, 2829 | import + call site — drop both |
| `nodes/batch_ltx_render.py` | 82, 2090 | `audio_dirs = [otr_audio_dir(), otr_legacy_audio_dir()]` → `[otr_audio_dir()]` |
| `nodes/scene_sequencer.py` | 879, 1123 | two function-local imports + lists — drop |
| `nodes/video_composite.py` | 90, 396 | import + call site — drop both |

**Do not touch:** `BUG_LOG.md`, `docs/**`, forensic comments from
prior sprints, `tools/validate_workflow_links.py` `FORBIDDEN_PATTERNS`
catalogue entry, `docs/2026-05-13-S26-audit-results.md` enumeration list.

**Targeted regression battery (run after each file's commit):**

```
pytest tests/test_otr_ledger_consumers.py \
       tests/test_batch_humo_render.py \
       tests/test_batch_ltx_render.py \
       tests/test_video_composite.py \
       tests/test_scene_sequencer.py \
       tests/test_audio_enhance.py \
       tests/test_batch_audiogen_generator.py \
       tests/test_batch_bark_generator.py \
       tests/v2/test_audio_byte_identical.py \
       -q
```

Commit per file: `cleanbreak(s28-p1-N): drop otr_legacy_audio_dir from <file>`.

### Step 1.2 — Delete the function (only after Step 1.1 cross-check)

Precondition:

```
git grep -nE 'from \._otr_paths import.*otr_legacy_audio_dir|otr_legacy_audio_dir\(' nodes/
```

must return zero hits. If any remain, finish Step 1.1 first.

Edits:

- `nodes/_otr_paths.py:201` — delete `def otr_legacy_audio_dir(...)`.
- `nodes/_otr_paths.py:524` — delete `"otr_legacy_audio_dir"` from `__all__`.
- Replace with one consolidated forensic comment.

Cross-check:

```
git grep -n 'otr_legacy_audio_dir' nodes/ tests/
```

Only forensic comments + `FORBIDDEN_PATTERNS` catalogue entry. Zero
imports, zero calls.

Regression battery (Step 1.1 battery + `tests/test_otr_paths.py` if it exists).

Commit: `cleanbreak(s28-p1-fn): delete otr_legacy_audio_dir function + __all__`.

### Step 1.3 — Strip legacy flat-layout walker

`nodes/_otr_ledger.py:340-376` — `find_most_recent_ledger`:

```python
# Pre-S28
candidates.extend(d.glob("*_ledger.json"))          # legacy flat (broken)
candidates.extend(d.glob("*/audio/*_ledger.json"))  # per-episode workspace
# Post-S28
candidates.extend(d.glob("*/audio/*_ledger.json"))  # the only contract
```

Update the docstring: drop the "Walks each given dir at TWO levels"
framing. Forensic comment replaces the removed walk.

Cross-check:

```
git grep -nE 'd\.glob.*_ledger\.json' nodes/_otr_ledger.py
```

One hit (per-episode workspace), zero flat-layout walks.

Regression: `pytest tests/test_otr_ledger.py tests/v2/test_audio_byte_identical.py -q`.

Commit: `cleanbreak(s28-p1-walker): strip flat-layout walk from find_most_recent_ledger`.

---

## Phase 2 — `req.budget is None` extinction

### Step 2.1 — Add `standard_budget` fixture

`tests/conftest.py`:

```python
@pytest.fixture
def standard_budget():
    """v2.0 production-shape EpisodeBudget for outline + composer tests.
    Mirrors OTR_LedgerScriptWriter defaults so prompt-shape tests
    exercise the production code path.
    """
    from nodes._otr_episode_budget import EpisodeBudget
    return EpisodeBudget(
        total_words=2400,
        beat_count=8,
        words_per_beat_min=240,
        words_per_beat_max=360,
        music_inter_count=2,
    )
```

(Fields read live from `OTR_LedgerScriptWriter` defaults at edit time
to keep fixture in sync.)

Commit: `test(s28-p2-fixture): add standard_budget fixture`.

### Step 2.2 — Migrate every `budget=None` test

Sweep:

```
git grep -nE 'budget\s*=\s*None|OutlineRequest\(.*budget=None|OutlineRequest\([^)]*\)' tests/
```

Every `OutlineRequest(...)` call gets `budget=standard_budget`. Tests
asserting the bare-format prompt re-assert against the budget-block-
present prompt.

Cross-check:

```
git grep -nE 'budget\s*=\s*None|OutlineRequest.*budget=None' tests/
```

Zero hits.

Regression: `pytest tests/test_otr_outline.py tests/test_episode_budget.py tests/test_outline_validators.py -q`.

Commit: `cleanbreak(s28-p2-tests): migrate outline tests to standard_budget`.

### Step 2.3 — Delete production fallbacks

`nodes/_otr_outline.py` edits:

- `:289-310` — `budget: object = None` field. Drop the default; let
  `__post_init__` raise `ValueError("budget required for v2.0 contract")` if missing.
- `:471-484` — delete the `if budget_block:` guard. Always append the budget block.
- `:725` — delete the no-op-when-None branch.
- `:1026` — delete the no-op-when-None branch in Phase 2A validators.
- `:1243-1258` — Test 11 in the inline harness: keep only the rich render path; delete the bare-format Test 11a.

Each deletion gets a one-line forensic comment.

Cross-check:

```
git grep -nE 'req\.budget is None|budget is None' nodes/_otr_outline.py
```

Zero hits.

Regression: same module set as Step 2.2 plus audio-byte-identical.

Commit: `cleanbreak(s28-p2-delete): drop budget=None fallbacks from _otr_outline.py`.

---

## Phase 3 — `_otr_line_composer.py` caller-shape extinction

### Step 3.1 — Producer audit

Write `docs/2026-05-13-S28-producer-audit-b4.md`. For each consumer site:

| Site | Field read | Producer | Auto-policy rule |
|---|---|---|---|
| `_otr_line_composer.py:468` | allowed-terms set default | `_otr_outline.py.__post_init__` | A |
| `_otr_line_composer.py:856` | `allowed_people` / `allowed_things` | `_otr_outline.py` outline builder | A |
| `_otr_line_composer.py:1215` | `generate_fn` fallback | `OTR_LedgerScriptWriter` call sites | A |
| `_otr_line_composer.py:1492` | caller-built field | call sites | A |

For any producer that doesn't always populate: apply rule D or E,
commit the producer fix.

Commit (audit doc): `docs(s28-p3-audit): producer audit b4`.
Commit(s) (producer fixes if any): `fix(s28-p3-producer-N): <producer> always populates <field>`.

### Step 3.2 — Delete consumer-side fallbacks

After the audit confirms producers always populate (or producer fixes
land), delete the 4 fallbacks. Each gets a forensic comment.

Cross-check:

```
git grep -nE 'back-compat|legacy fallback|legacy shape' nodes/_otr_line_composer.py
```

Only forensic comments.

Regression: `pytest tests/test_otr_line_composer.py tests/test_line_composer_contract.py tests/v2/test_audio_byte_identical.py -q`.

Commit: `cleanbreak(s28-p3-delete): drop _otr_line_composer caller-shape fallbacks`.

---

## Phase 4 — `_otr_ledger_freeze.py` ledger-shape extinction (audio-critical)

### Step 4.1 — Producer audit

Write `docs/2026-05-13-S28-producer-audit-b5.md`:

| Site | Field read | Producer | Auto-policy rule |
|---|---|---|---|
| `_otr_ledger_freeze.py:279` | `meta.outline.beats` | `OTR_LedgerScriptWriter` post-outline | A, B |
| `_otr_ledger_freeze.py:356` | `skip=True` flag | writer per-line | A — drop skip flag uniformly |
| `_otr_ledger_freeze.py:482` | `speaker_role` | writer per-line | A, B |
| `_otr_ledger_freeze.py:669` | `dur_s` per line | `find_clip_durations` / upstream timing | A, B |

Apply rule D or E per producer leak. Producer fixes commit first.

Commit (audit): `docs(s28-p4-audit): producer audit b5`.
Commit(s) (producer fixes): `fix(s28-p4-producer-N): <producer> always populates <field>`.

### Step 4.2 — Delete consumer-side tolerance, ONE SITE PER COMMIT

Each deletion is its own commit. Audio-byte-identical runs after each.
Rule F governs failures (revert single commit, trace one level deeper).

1. Site 1 — `meta.outline.beats` fallback at `:279`.
   ```
   pytest tests/test_otr_ledger_freeze.py tests/v2/test_audio_byte_identical.py -q
   ```
   Commit: `cleanbreak(s28-p4-site1): delete meta.outline.beats legacy fallback`.

2. Site 2 — legacy `skip=True` tolerance at `:356`.
   Same regression.
   Commit: `cleanbreak(s28-p4-site2): delete legacy skip flag tolerance`.

3. Site 3 — `speaker_role` substitute at `:482`.
   Same regression.
   Commit: `cleanbreak(s28-p4-site3): delete speaker_role legacy substitute`.

4. Site 4 — `dur_s` absent/None tolerance at `:669`.
   Same regression.
   Commit: `cleanbreak(s28-p4-site4): delete dur_s absent tolerance`.

Cross-check after Site 4:

```
git grep -nE 'back-compat|legacy fallback|legacy shape' nodes/_otr_ledger_freeze.py
```

Only forensic comments.

---

## Phase 5 — Final static verification

```
pytest tests/ -q --tb=line > docs/2026-05-13-S28-final-pytest.txt
# zero unexpected failures

pytest C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -v
# expect: 23 passed, 1 skipped, 2 xfailed

python tools/validate_workflow_links.py workflows/*.json \
  > docs/2026-05-13-S28-link-integrity-report.txt
# every JSON: TOTAL violations: 0

# known-fail delta
diff docs/2026-05-13-S28-baseline-known-fail-nodeids.txt \
     docs/2026-05-13-S28-final-known-fail-nodeids.txt \
  > docs/2026-05-13-S28-known-fail-delta.txt
# empty

# forbidden-pattern sweep
git --no-pager diff s27-cleanbreak-tail..HEAD -- '*.py' \
  | grep -E '^\+' \
  | grep -E 'DeprecationWarning|back-compat|legacy fallback|legacy shape|\bshim\b|\balias\b|\botr_legacy_audio_dir\b|budget is None' \
  | grep -vE '^\+\s*#' \
  | grep -vE '^\+\s*"""' \
  | grep -vE '^\+\s*r"' \
  > docs/2026-05-13-S28-new-forbidden-hits.txt
# empty

# final audio-byte-identical
pytest tests/v2/test_audio_byte_identical.py -q
# PASS
```

Write `docs/2026-05-13-S28-final-qa-review.md` (synthesis, verdict,
sign-off) and `docs/2026-05-13-S28-audit-results.md` (per-phase result
log).

Commit: `docs(s28): final QA review + hand-off artifacts`.

Push: `cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio && git push origin s28-cleaner-break`.

---

## Acceptance criteria

- [ ] `git status --short` empty at sprint open AND close
- [ ] `git grep -n 'otr_legacy_audio_dir' nodes/ tests/` — only forensic comments + catalogue entry
- [ ] `git grep -nE 'def otr_legacy_audio_dir' nodes/` — zero hits
- [ ] `git grep -nE 'd\.glob.*_ledger\.json' nodes/` — only per-episode workspace glob
- [ ] `git grep -nE 'req\.budget is None|budget is None' nodes/` — zero non-comment hits
- [ ] `git grep -nE 'back-compat|legacy fallback|legacy shape' nodes/_otr_line_composer.py` — only forensic comments
- [ ] `git grep -nE 'back-compat|legacy fallback|legacy shape' nodes/_otr_ledger_freeze.py` — only forensic comments
- [ ] Full pytest: zero unexpected failures, math explained
- [ ] Known-fail delta empty
- [ ] Bug Bible: `23 passed, 1 skipped, 2 xfailed`
- [ ] All 5 workflow JSONs: `TOTAL violations: 0`
- [ ] Forbidden-pattern sweep: empty file (only forensic + catalogue allowed)
- [ ] Audio-byte-identical PASSES at every Phase 4 site boundary AND final
- [ ] `docs/cleanbreak-deferred.md` empty. Zero items. Zero carve-outs.
- [ ] All `docs/2026-05-13-S28-*` artifacts written
- [ ] `git push origin s28-cleaner-break` succeeded, local HEAD == origin HEAD

---

## Hand-off artifacts (all under `docs/2026-05-13-S28-*`)

- `baseline-pytest.txt`
- `baseline-known-fail-nodeids.txt`
- `baseline-footprint.txt`
- `producer-audit-b4.md`
- `producer-audit-b5.md`
- `final-pytest.txt`
- `final-known-fail-nodeids.txt`
- `known-fail-delta.txt`
- `forbidden-pattern-sweep.txt`
- `new-forbidden-hits.txt`
- `link-integrity-report.txt`
- `audit-results.md`
- `final-qa-review.md`

---

## Out of scope (forward work, not S28)

- Sync drift
- LTX clip metadata
- Gaussian splat rendering
- SIGNAL LOST narrative layer
- B Two-Model Selector
- C `meta.story_brief` v2
- A downstream verification
- Three-File Contract promotion of BUG-LOCAL-221/222/223 (waits on v2.0 ship)
- Post-cleanbreak ComfyUI runtime smoke (Jeffrey's sanity check at S28 close, not a sprint gate)

---

## Sizing

- Phase 0: 15 min
- Phase 1: 1-2 hr
- Phase 2: 2-3 hr
- Phase 3: 2-3 hr
- Phase 4: 3-4 hr (audio-critical, per-site)
- Phase 5: 30 min
- **Total: 9-13 hr autonomous Cowork. One swim-run, end-to-end.**

---

## Why this is the LAST cleanbreak sprint

After S28, every legacy path from the S24 → S25 → S26 → S27 chain is
extinct. The v2.0 contract is the only contract. Producers respect
their own contracts; consumers trust producers; no fallbacks, no
defensive guards, no "tolerance for older shapes."

If a future audit finds a missed surface, it's a `BUG-LOCAL-NNN` with a
single-commit fix — not a sprint name. **100% means 100%. The cleaner
break ends the chain.**
