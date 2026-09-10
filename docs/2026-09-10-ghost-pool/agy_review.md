### Review Summary & Verdict

**Reviewer Identity / Lane**: Antigravity CLI (`agy`) / Gemini 3.8 Flash (High)  
**Task**: Read-only finished-diff review for Row 2.2 (`docs/2026-09-10-ghost-pool/driver_anchor.md` and `/Users/rentamac/Documents/otr-mac/logs/ghost_pool/review.diff`)  
**Verdict**: **NO BLOCKERS**

---

### Detailed Review Findings

#### 1. True Least-Recent Reuse (LRU) & Tie-Breaking
- **Evidence**: [`nodes/_otr_video_engines/ghost_signal_author.py:L1156-1206`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1156-L1206)
- **Mechanism**:
  - `history = [str(u) for u in used if str(u)]` maintains chronological order.
  - `last_used = {sig: i for i, sig in enumerate(history)}` maps each distinct signature to its latest appearance index in history.
  - In `deterministic_leaf`, `min(eligible, key=lambda item: last_used.get(item[1], -1))` selects the candidate whose most recent usage was farthest in the past (or `-1` if present in `reserved` but never yet used in `history`).
  - Ties are broken deterministically by Python's stable `min()` returning the first candidate in `eligible`, which inherits the `(start + step) % len(pool)` probe order seeded by `episode_seed` and `beat_id`.
  - Verified by [`tests/test_ghost_signal_author.py:L1066-1090`](file:///Users/rentamac/Documents/otr-mac/repo/tests/test_ghost_signal_author.py#L1066-L1090) (`sigs[capacity:] == sigs[:-capacity]` over 60 beats).

#### 2. Both Replay Neighbors Excluded
- **Evidence**: [`nodes/_otr_video_engines/ghost_signal_author.py:L1158-1160`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1158-L1160), [`L1224-1235`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1224-L1235), [`nodes/otr_shot_lock.py:L2634-2645`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/otr_shot_lock.py#L2634-L2645)
- **Mechanism**:
  - `_ghost_replayed_signatures` maps `spec["ordinal"] -> sig`.
  - In `deterministic_batch`, ordinals are iterated in sorted order over `set(pending) | set(fixed)`.
  - **Preceding Neighbor ($k-1$)**: When beat $k$ evaluates, `history[-1]` is either fixed replay row $k-1$ (appended during iteration $k-1$) or fresh row $k-1$ (appended when allocated). `forbidden.add(history[-1])` excludes the backward neighbor.
  - **Succeeding Neighbor ($k+1$)**: `deterministic_batch` passes `avoid=(fixed.get(ordinal + 1, ""),)`. If ordinal $k+1$ is a fixed replay row, it is added to `forbidden` and excluded from `eligible`. If ordinal $k+1$ is a fresh row, beat $k$'s signature will become `history[-1]` when $k+1$ is processed, excluding $k$ from $k+1$.
  - Both neighbors are strictly excluded for all transitions (fixed–fresh, fresh–fixed, and fresh–fresh).
  - Verified by [`tests/test_ghost_prompt_v2_lane.py:L267-305`](file:///Users/rentamac/Documents/otr-mac/repo/tests/test_ghost_prompt_v2_lane.py#L267-L305) and [`tests/test_ghost_signal_author.py:L1104-1118`](file:///Users/rentamac/Documents/otr-mac/repo/tests/test_ghost_signal_author.py#L1104-L1118).

#### 3. Deterministic Seed Behavior & Short Allocations
- **Evidence**: [`nodes/_otr_video_engines/ghost_signal_author.py:L1192-1197`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1192-L1197)
- **Mechanism**:
  - `start = _hash_int(episode_seed, spec.get("beat_id"), mode, GHOST_AUTHOR_VERSION) % len(pool)` is unchanged.
  - For batches within the 18-clause capacity (no exhaustion), `_free(candidate)` checks `sig not in spent and sig not in forbidden`. Since previous beats are in `spent`, the first unspent candidate in probe order is selected, exactly as in pre-change allocations.
  - Short allocations remain byte-identical in leaf choice.

#### 4. Exact Stored Schema & Immutability of Replay Objects
- **Evidence**: [`nodes/_otr_video_engines/ghost_signal_author.py:L1249-1310`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1249-L1310), [`nodes/otr_shot_lock.py:L2899-2912`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/otr_shot_lock.py#L2899-L2912), [`L2940-2946`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/otr_shot_lock.py#L2940-L2946)
- **Mechanism**:
  - `GHOST_PROMPT_FIELDS` (10 fields) is unaltered. No schema extension or version bump was introduced.
  - Existing replayed objects populated in `out` during `_author_ghost_prompts` (lines 2908–2911) are preserved without modification. Only `spec in needs` are constructed and written to `out` (lines 2940–2946).
  - Allocation dispositions are carried out-of-band in `reuse_dispositions` dict keyed by `spec["id"]`, and formatted into `fallback_reason` via `"; ".join(filter(None, (reason, reuse_dispositions.get(spec["id"], ""))))` only when `source == "deterministic_fallback"`.

#### 5. Loud Failures for Structural Corruption
- **Evidence**: [`nodes/_otr_video_engines/ghost_signal_author.py:L1171-1174`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1171-L1174), [`L1199-1202`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1199-L1202)
- **Mechanism**:
  - In `_free(candidate)`, if `not sig`, `GhostAuthorError` is immediately raised rather than silently treating the candidate as unconstrained.
  - If a corrupted or artificial pool has no candidate outside `forbidden`, `deterministic_leaf` raises `GhostAuthorError("the %s Ghost clause pool has no nonadjacent finalized prompt...")` loudly.
  - Only genuine exhaustion of non-forbidden candidates recovers into LRU reuse.
  - No new environment variables are read, no processes are spawned, and the v3 render composer is untouched.

---

### Non-Blocking Observations & Test Boundary Analysis

1. **Caller Contract on `_ghost_replayed_signatures`**:
   In [`nodes/otr_shot_lock.py:L2638`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/otr_shot_lock.py#L2638), `spec = spec_for[beat_id]` performs a direct lookup where `spec_for` is built from `specs`. In `_author_ghost_prompts`, `replayed` is `out`, which is strictly a subset of `specs`, so `KeyError` cannot occur in normal execution. If `_ghost_replayed_signatures` is called directly with mismatched inputs in custom harnesses, an unknown `beat_id` will fail loudly with `KeyError` (consistent with fail-loud design).
2. **Ordinals with Gaps**:
   In [`nodes/_otr_video_engines/ghost_signal_author.py:L1224`](file:///Users/rentamac/Documents/otr-mac/repo/nodes/_otr_video_engines/ghost_signal_author.py#L1224), `deterministic_batch` sorts `set(pending) | set(fixed)`. In canonical production, ordinals are consecutive `0..N-1`. If an artificial spec list had ordinal gaps (e.g. 2, then 5), beat 5 would avoid beat 2 via `history[-1]`. Because beat 5 immediately follows beat 2 in sequential presentation, avoiding the immediate prior beat is functionally sound.

---

### Verification & Environment Limits

- **Focused Suite**: 162 passed, 1 skipped.
- **Bug Bible Baseline**: 10 failed, 29 passed, 11 skipped, 3 xfailed.
- **Environment Limits**: Execution ran on the rented Mac M4. Haunted lane MPS exclusion is maintained; the pod remains stopped and the 5080 production loop is untouched. Model-network access is disabled and 3 Bark audition tests requiring Windows evidence files are deselected.

**Conclusion**: The implementation satisfies all specifications in `driver_anchor.md`. There are no correctness or safety blockers to push.
