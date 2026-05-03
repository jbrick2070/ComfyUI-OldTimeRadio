# Question -- 2026-05-02

# Phase B consult — production_ledger.py rename + os.replace hardening

## Context

OTR (Old Time Radio) ComfyUI node pack on Windows, single 5080 16GB GPU. Audio is king (Rule C7: byte-identical audio output across runs). ComfyUI queue is strictly sequential — no concurrent runs of the pipeline at the OS-process level, but inside one pipeline multiple nodes touch the ledger via two write paths:

1. **`Ledger` class** in `nodes/production_ledger.py` — write-only, in-memory dict, stamped to disk via `save()`. Owns the canonical `<ep>_ledger.json`.
2. **`_otr_ledger.save_ledger_safe()`** — schema-l3 helpers from audio nodes (Bark, MusicGen, AudioGen, AudioEnhance, EpisodeAssembler) that read the JSON, mutate it, write it back. They run *between* `Ledger.save()` calls.

Phase A just landed: spacesaver in `rtx_upscale.py` no longer uses a global `find_most_recent_ledger` walker — it derives the episode dir directly from the `src` mp4 path. That fix exposed two adjacent bugs in `Ledger.rename_episode` that we need to close in Phase B (atomic rename path):

### Finding 2 — treatment file rename gap
`Ledger.rename_episode(new_id)` does:
- Move parent dir `otr/episodes/pending_<ts>/` → `otr/episodes/<new_id>/`
- Rename `pending_<ts>_ledger.json` → `<new_id>_ledger.json` inside the moved dir

But it does NOT rename `pending_<ts>_treatment.txt` → `<new_id>_treatment.txt`. The treatment file is written by `OTR_LLMScriptWriter` early in the run (before the title is finalized), so its name still has the `pending_<ts>` prefix. After rename, it sits in the new dir with the old prefix. Phase A's spacesaver currently keeps it via `glob("*_treatment.txt")` (defensive), but that defensive measure is paid for by Phase B closing the gap. End-state goal: exactly one ledger + one treatment per episode dir, both with the canonical `<new_id>_*` prefix.

### Finding 3 — `os.replace` fallback leaves split state
Current code (excerpt):
```python
moved_dir = False
if (os.path.exists(old_ep_dir)
        and not os.path.exists(new_ep_dir)
        and old_ep_dir != new_ep_dir):
    try:
        os.replace(old_ep_dir, new_ep_dir)
        moved_dir = True
    except Exception as exc:
        log.warning("[Ledger] per-episode dir move failed (%s -> %s): %s; "
                    "falling back to file-only rename", old_ep_dir, new_ep_dir, exc)

# Step 2: update in-memory state (episode_id + out_dir to point
# at the new per-episode audio dir).
self.episode_id = new_id
self.data["episode_id"] = new_id
if moved_dir:
    self.out_dir = new_audio_dir

# Step 3: rename the ledger file ...
```

If `os.replace(old_ep_dir, new_ep_dir)` fails (Windows Defender lock, indexer holding a handle, EACCES on a file inside, partial dir state from a previous crash, etc.), the code logs a warning and **continues**:
- `self.episode_id` updates to `new_id`
- `self.data["episode_id"]` updates to `new_id`
- `self.out_dir` stays pointing at the OLD audio dir (`moved_dir=False` branch)
- The next `self.save()` writes the ledger to the OLD path with the NEW id baked in
- But every other downstream node (BatchHumoRender, VideoComposite, RTXUpscale) uses the NEW id to build their own paths — they will look for things in `otr/episodes/<new_id>/...` which doesn't exist
- Pipeline crashes downstream with confusing "file not found" errors that don't point at the rename failure

## The proposed fix

**Treatment rename:** in `rename_episode`, also walk the audio dir for `pending_<ts>_treatment.txt` (or any `pending_<ts>_*.txt` we own) and rename to the canonical `<new_id>_*.txt` form, AFTER the dir move succeeds.

**Hard-fail on dir-move failure:** add a Windows retry loop (3 attempts, 0.5s sleep) around `os.replace(old_ep_dir, new_ep_dir)`. If all 3 attempts fail, raise `RuntimeError` with a clear message AND do not update any in-memory state. The pipeline crashes here, immediately, with an actionable error pointing at the rename failure.

```python
import time as _time
moved_dir = False
last_exc = None
if (os.path.exists(old_ep_dir)
        and not os.path.exists(new_ep_dir)
        and old_ep_dir != new_ep_dir):
    for attempt in range(3):
        try:
            os.replace(old_ep_dir, new_ep_dir)
            moved_dir = True
            last_exc = None
            break
        except OSError as exc:
            last_exc = exc
            log.warning(
                "[Ledger] per-episode dir move attempt %d/3 failed "
                "(%s -> %s): %s",
                attempt + 1, old_ep_dir, new_ep_dir, exc,
            )
            _time.sleep(0.5)
    if not moved_dir and last_exc is not None:
        raise RuntimeError(
            f"[Ledger] rename_episode: per-episode dir move failed after "
            f"3 attempts ({old_ep_dir} -> {new_ep_dir}): {last_exc}. "
            f"In-memory state NOT updated. Fix the underlying issue "
            f"(file lock, permissions, partial dir) and re-queue."
        )

# Only past this point if dir-move succeeded OR there was nothing to move.
self.episode_id = new_id
self.data["episode_id"] = new_id
if moved_dir:
    self.out_dir = new_audio_dir
    # Rename pending_<ts>_treatment.txt -> <new_id>_treatment.txt
    for tx in Path(new_audio_dir).glob("pending_*_treatment.txt"):
        canon = Path(new_audio_dir) / f"{_slugify(new_id, limit=120)}_treatment.txt"
        try:
            os.replace(str(tx), str(canon))
            log.info("[Ledger] treatment moved %s -> %s", tx.name, canon.name)
        except Exception as exc:
            log.warning("[Ledger] treatment rename failed (%s -> %s): %s",
                        tx, canon, exc)
# ... existing ledger file rename logic unchanged ...
```

## What I want your opinion on (please be specific)

**Q1: Retry semantics.** Is 3 attempts × 0.5s the right Windows-defender / indexer dance? Some sources say exponential backoff is better (0.1, 0.5, 2.0). What's your call given:
- Windows 11, NTFS, OTR has 16-32 GB RAM headroom, RTX 5080 laptop
- The rename operation moves a directory containing several wav files, several png stills, and a few txt files (typically 10-50 MB total)
- A real defender lock typically clears in 200-800ms in our environment

**Q2: Hard-fail vs split-state recovery.** The proposed fix raises `RuntimeError` on irrecoverable dir-move failure. The alternative is a "split state" recovery mode where we accept the dir didn't move, leave `out_dir` pointing at the pending dir, and let the next `save()` write a finalized-id ledger into the pending dir. Then a startup janitor on the next run reconciles. Which is safer? My instinct says hard-fail because Phase A's spacesaver guard (depth-1 `relative_to(otr_episodes_root())`) would refuse to act on the pending dir anyway, so split state would silently bypass cleanup forever. Confirm or push back.

**Q3: Treatment rename ordering.** Should treatment rename happen BEFORE or AFTER the ledger file rename inside the moved dir? Consider: if treatment-rename fails after ledger-rename succeeds, the next save updates the (correctly named) ledger but leaves a `pending_<ts>_treatment.txt` orphan. If treatment-rename happens first, a failure leaves the ledger still under its old name (recoverable on next save). I'm leaning AFTER ledger rename so the ledger represents the new state correctly even if treatment cleanup fails — treatment loss is recoverable from regenerated text, but ledger inconsistency cascades.

**Q4: What about other `pending_<ts>_*` files I might be missing?** Audit point: are there other writers in OTR that may have created `pending_<ts>_*` files in the audio dir before rename fires? I know about treatment.txt. Anything else?

**Q5: Concurrent writer interleaving.** While `Ledger.rename_episode` is mid-flight (between `os.replace` of dir and the in-memory state update), is there any path where another node could call `_otr_ledger.save_ledger_safe()` and write to either the old or new path? ComfyUI is sequential per workflow execution, but `save_ledger_safe` resolves its own path via `find_most_recent_ledger` (different code path from `Ledger.path`). Could the walker race with the dir move?

**Q6: What hardening tests should I add to `tests/test_core.py` (or a new `tests/test_ledger_rename.py`)?** Specific cases:
- Happy path: pending → finalized rename, both ledger and treatment land at canonical names
- Dir-move fails 2 of 3 attempts then succeeds (mock `os.replace` to raise OSError twice)
- Dir-move fails all 3 attempts → expect `RuntimeError`, in-memory state unchanged, on-disk state unchanged
- Treatment rename fails after ledger move succeeds → expect log warning, ledger still at canonical name
- Multiple `pending_<ts>_*` patterns (if any) all get renamed
- Reject re-rename: calling `rename_episode` twice (idempotency)

**Q7: Anything else in this rename path that's a footgun on Windows specifically?** I want a "boring" rename that always works. Single-process, sequential, NTFS.

## Hard constraints

- C7: audio output bytes must remain identical run-to-run. The rename path doesn't touch wav bytes, but if your suggestion changes when/how files land, double-check.
- ASCII-only Python source, no BOM, UTF-8 file encoding.
- No new dependencies. Standard library only (os, time, pathlib, logging).
- Windows-first (Defender, indexer, locking semantics).
- One commit, one bug log entry, single-file change preferred.

Please structure your reply: (1) Direct answers to Q1-Q7, (2) Any factual errors in my proposed fix, (3) Recommended final shape, (4) Things I should verify before merging.
