# PHASE A INVARIANT CONTRACT

**Companion to** `R1_PHASE_A_EXTRACTION.md`.
**Sprint:** 2026-07-04 JSON Prompt Transplant, **Phase A only**.
**Branch:** `v2.0-alpha`. **Anchor HEAD:** `a7bdc42d`.

This document is the invariant contract for Phase A. It enumerates the
properties Phase A must preserve, and specifies the harness test that
proves each one. Any chunk that violates an invariant is reverted before
push.

---

## 0. What this document is (and is not)

Is:
- A checklist of behaviors that MUST NOT change between HEAD `a7bdc42d`
  (pre-Phase-A) and the tip of Phase A (post-extraction).
- A per-invariant harness command that a coder window can run to prove
  each one.

Is not:
- A design doc. Design lives in `R1_PHASE_A_EXTRACTION.md`.
- A statement about Phase B. Phase B is out of scope; see `PHASE_B_STUB.md`.
- A statement about content. Phase A does not change any prompt content.

---

## 1. Invariant I1: byte-identical rendered prompt strings

**Property.** For every one of the 15 sites listed in R1_PHASE_A_EXTRACTION
section 3, the string that gets sent to the LLM at runtime post-Phase-A
must equal the string that was sent pre-Phase-A, byte-for-byte, on a
fixed-input replay.

**Harness.** New test module `tests/test_prompt_profile_extraction.py`.

Steps (executed by coder before each chunk push):

1. Before Phase A begins, run a capture script over HEAD `a7bdc42d`:
   for each site key, call the current Python code path (or extract the
   Python constant directly) and write the result to
   `tests/snapshots/prompt_profiles/pre_phaseA/<profile>__<site>.txt`.
   Commit the snapshots as part of Chunk A.
2. Post-Chunk B/C/D/E, the test:

   ```python
   def test_site_bytes_match_snapshot(profile_id, site_key):
       resolved = get_prompt(profile_id, site_key)
       expected = read_snapshot(profile_id, site_key)
       assert resolved.raw_bytes == expected  # not str.strip, not str.lower
   ```

3. Any mismatch fails the chunk.

**Notes.**
- Snapshot comparison is byte-level (`bytes.__eq__`). Whitespace, trailing
  newlines, and unicode form all matter.
- The JSON round-trip in the middle (Python literal -> JSON-escape ->
  file -> JSON-decode -> str) is the only intermediate transform, and the
  snapshot proves it is lossless.

**Failure mode.** Chunk reverted, JSON-escape logic re-audited.

---

## 2. Invariant I2: byte-identical audio output

**Property.** Running the regression episode at post-Phase-A HEAD produces
audio files with checksums identical to those produced at pre-Phase-A HEAD
`a7bdc42d`, given the same input, the same seed, and the same environment.

**Harness.** Existing suite `test_audio_byte_identical` (or the closest
existing suite name -- kibitz r3 confirms). The suite:

1. Loads a fixed regression episode fixture.
2. Renders end-to-end with a fixed seed.
3. Compares audio checksums against a committed baseline.

**Command (canonical run):**

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
python -m pytest tests -k audio_byte_identical -x --tb=short
```

**Cadence.** Runs at the end of every chunk (A through F). Cannot be
deferred. If red, revert the chunk.

**Failure mode.** Chunk reverted. Cause is almost certainly Invariant I1
having been violated somewhere -- a prompt string is different, so an LLM
output is different, so an audio output is different. Debug via the I1
snapshot diff first.

---

## 3. Invariant I3: ROW_KEYED merge invariants

**Property.** The ledger row merge behavior in
`nodes/OTR_LedgerScriptWriter.py` (and any helper it calls) is unchanged.
No new merge keys, no changed merge precedence, no changed
`ROW_KEYED` semantics.

**Rationale.** Phase A moves PROMPT CONTENT out of Python. It does NOT
touch merge logic. Any diff that appears in the merge code path during
Phase A is a bug.

**Harness.** Existing ledger merge tests under `tests/`. Kibitz r3 confirms
the exact test names. Canonical run:

```powershell
python -m pytest tests -k "ledger or row_keyed or merge" -x --tb=short
```

**Cadence.** Runs at the end of every chunk.

**Failure mode.** Chunk reverted. This should never fail unless a chunk
diff accidentally hit a merge-adjacent line -- diff review will catch it.

---

## 4. Invariant I4: ledger schema `l3-2026-05-14` untouched

**Property.** The ledger schema version stays `l3-2026-05-14`. No fields
added, renamed, removed, or repositioned.

**Rationale.** The ledger schema is on-disk state used by soak episodes and
downstream tooling. Phase A moves prompt strings; it does not move ledger
fields.

**Harness.** Grep + AST check. Canonical run:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git grep -n 'l3-2026-05-14' -- '*.py'
# Compare hits to pre-Phase-A baseline (captured in Chunk A commit message).
```

If any hit changes line, changes file, or disappears, the chunk is
reverted.

**Cadence.** End of every chunk. Automated in the CI script for the
sprint if one exists (kibitz r3 confirms).

**Failure mode.** Chunk reverted. Ledger schema change is a Phase B
concern; if it appears in Phase A, the diff is wrong.

---

## 5. Invariant I5: critic / reroll seam untouched

**Property.** No changes to `nodes/_otr_story_critic.py`,
`nodes/_otr_ledger_reviewer.py`, or the reroll orchestration in
`OTR_LedgerScriptWriter.py`. These are not in the 15-site set and are not
in scope for Phase A.

**Harness.** Grep. Canonical run:

```powershell
git diff --name-only $(git merge-base HEAD origin/main)..HEAD -- \
    nodes/_otr_story_critic.py \
    nodes/_otr_ledger_reviewer.py
```

Expected output: empty. Any file listed = the chunk touched a critic /
reroll file = revert.

**Cadence.** End of every chunk.

**Failure mode.** Chunk reverted. Critic and reroll are Phase B concerns.

---

## 6. Invariant I6: `test_period_prompts` assertions unchanged

**Property.** The five anchor tokens (`"1940s"`, `"Suspense"`, `"NARRATOR"`,
`"CHARACTER:dialogue"`, `"Family-broadcast safe"`), the `[SFX:` prohibition,
and the modern-slang blacklist in `tests/test_period_prompts.py` all
continue to pass after Chunk D (radio wiring).

**Rationale.** These tests currently assert on the Python constant
`OTR_PERIOD_SYSTEM_PROMPT`. Post-Chunk D they assert on
`get_prompt("radio", "outline_system").system`. Assertion set is
unchanged; only the source of the string changes.

**Harness.**

```powershell
python -m pytest tests/test_period_prompts.py -x --tb=short
```

**Cadence.** Runs at the end of Chunk D (and every chunk after). Green
across all subsequent chunks.

**Failure mode.** Chunk D revert. Cause: the JSON-escape of
`OTR_PERIOD_SYSTEM_PROMPT` altered a token the assertions look for.
Re-audit escaping; the snapshot from Invariant I1 will show the diff.

---

## 7. Invariant I7: no changes to `IS_CHANGED`, VRAM, or model management

**Property.** No node's `IS_CHANGED` returns a different value pre- vs
post-Phase-A on the same inputs. No changes to VRAM budgeting, model
management, or single-resident-heavy discipline.

**Rationale.** Phase A moves strings. Cache keys and VRAM residency are
independent of prompt-content storage location.

**Harness.** ComfyUI cache-hit smoke: run the regression episode twice,
back to back, and confirm that the second run hits the LLM cache path for
every prompt-owning node. Canonical check: grep the log for
`cache_hit=true` counts matching pre-Phase-A totals.

**Cadence.** End of every chunk.

**Failure mode.** Chunk reverted. Almost certainly caused by the loader
being re-imported per node call instead of module-cached; audit loader
import discipline.

---

## 8. Invariant I8: env flags unchanged

**Property.** `OTR_ENABLE_PITCH_ROOM`, `OTR_GROUNDING_LEVER`, and every
other env flag currently gating prompt selection behavior at HEAD
`a7bdc42d` continues to gate identically post-Phase-A. The gating
condition stays in Python; only the text of the branches moves to JSON.

**Harness.** Grep. Canonical run:

```powershell
git grep -n 'os.environ.get\|getenv' -- 'nodes/*.py' | \
    Where-Object { $_ -match 'OTR_' }
```

Compare hit list pre- vs post-Phase-A. Must be identical (same file,
same line, same env var name).

**Cadence.** End of every chunk.

**Failure mode.** Chunk reverted; env-flag semantics are behavior, not
content, and never move.

---

## 9. Bug Bible: canonical per-chunk verification block

Copy-paste at the end of each chunk before commit:

```powershell
# Bug Bible -- Phase A per-chunk verification
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

# I1: prompt bytes match snapshot
python -m pytest tests/test_prompt_profile_extraction.py -x --tb=short

# I2: audio byte-identical
python -m pytest tests -k audio_byte_identical -x --tb=short

# I3: ledger merge behavior unchanged
python -m pytest tests -k "ledger or row_keyed or merge" -x --tb=short

# I4: schema version unchanged
$expect = 12  # placeholder -- replace with count captured in Chunk A commit
$actual = (git grep -c 'l3-2026-05-14' -- '*.py' | Measure-Object -Line).Lines
if ($actual -ne $expect) { throw "Ledger schema hits changed: $actual != $expect" }

# I5: critic / reroll untouched
$diff = git diff --name-only origin/v2.0-alpha..HEAD -- `
    nodes/_otr_story_critic.py nodes/_otr_ledger_reviewer.py
if ($diff) { throw "Critic/reroll seam touched: $diff" }

# I6: period prompt asserts still green (runs from Chunk D onward)
python -m pytest tests/test_period_prompts.py -x --tb=short

# I8: env flags unchanged
$flags = git grep -l 'os.environ.get\|getenv' -- 'nodes/*.py'
# Compare against Chunk A baseline; identical or throw.

Write-Host "Bug Bible: chunk verification GREEN"
```

All 8 invariants pass -> commit -> push. Any red -> revert -> debug.

---

## 10. Phase B gate reminder

Phase B does not start until:

- All 6 chunks (A through F) are on `origin/v2.0-alpha`.
- All 8 invariants are green on a clean rebuild.
- A soak run of at least one full episode ships with audio byte-identical
  against the pre-Phase-A baseline.

Then, and only then, Phase B planning begins in a new sprint with its own
kibitz + roundtable arc.

Phase A does not encroach on Phase B. See `PHASE_B_STUB.md`.
