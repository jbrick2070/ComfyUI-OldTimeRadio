# Synthesis -- 2026-05-02

**Question:** # Phase B consult — production_ledger.py rename + os.replace hardening

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

---

## ChatGPT (gpt-5.5)

## 1) Direct answers to Q1-Q7

### Q1: Retry semantics

Your proposed `3 attempts x 0.5s` is reasonable for this specific workload and environment.

Given your observed Defender/indexer lock window of roughly `200-800ms`, attempts at:

```text
t = 0.0s
t = 0.5s
t = 1.0s
```

should catch the common case without making the failure feel hung. Directory rename on NTFS is metadata-only when staying on the same volume, so the 10-50 MB artifact size mostly does not matter unless something is actively scanning/opening one of the files.

My recommendation:

- Keep it simple.
- Use 3 attempts.
- Sleep only between attempts, not after the final failed attempt.
- Catch `OSError`, not broad `Exception`, for the directory move.
- Log every failed attempt with attempt count and source/destination.

I would write it as either:

```python
for attempt in range(3):
    try:
        os.replace(old_ep_dir, new_ep_dir)
        moved_dir = True
        last_exc = None
        break
    except OSError as exc:
        last_exc = exc
        log.warning(...)
        if attempt < 2:
            _time.sleep(0.5)
```

If you want a slightly more “Windows boring” version, I would use 4 attempts over roughly 2 seconds:

```python
delays = (0.2, 0.5, 1.0)
```

That gives attempts at about `0.0s`, `0.2s`, `0.7s`, `1.7s`. But that is a policy call. For your reported environment, `3 x 0.5s` is acceptable and is the smaller change.

Do not chase exponential backoff here unless you are seeing real failures after the third attempt. This is not a throughput path, and the main production need is a clear deterministic fail point.

---

### Q2: Hard-fail vs split-state recovery

I strongly agree with hard-fail.

The current fallback creates exactly the kind of poison state that makes later failures misleading:

- in-memory `episode_id` becomes finalized,
- `data["episode_id"]` becomes finalized,
- `out_dir` remains pending,
- downstream nodes construct finalized paths,
- spacesaver/cleanup guards likely refuse to touch the pending directory,
- the real root cause is buried behind later “file not found” errors.

For this pipeline, split-state recovery is worse than immediate failure. You are single-process, sequential, local-only, and the rename is a structural invariant. If the episode directory cannot be renamed, the pipeline should stop at `Ledger.rename_episode()` with an actionable message.

Your instinct is correct: do not accept split state and hope a future janitor fixes it. A janitor can be useful later for diagnostics, but it should not be part of the normal success path.

One important addition: hard-fail not only when `os.replace(old_ep_dir, new_ep_dir)` fails, but also when both directories exist.

This case is currently dangerous:

```python
old_ep_dir exists
new_ep_dir exists
old_ep_dir != new_ep_dir
```

Your proposed condition skips the move because `not os.path.exists(new_ep_dir)` is false, then proceeds to update in-memory state. That can still create split or ambiguous state.

Treat that as a conflict unless you have explicit recovery logic.

---

### Q3: Treatment rename ordering

I would do:

1. Move episode directory.
2. Update in-memory state only after the directory is known to be in its final location.
3. Rename the ledger file to canonical name.
4. Rename treatment/owned txt sidecars.
5. Let normal `save()` update the canonical ledger.

So yes: ledger before treatment.

Reasoning:

- The ledger is the canonical coordination artifact.
- Downstream nodes are much more likely to depend on the ledger path/name than the treatment file name.
- A stale `pending_<ts>_treatment.txt` is ugly but recoverable.
- A stale or missing canonical ledger cascades into many misleading failures.

However, your sample code currently places treatment rename before the comment:

```python
# ... existing ledger file rename logic unchanged ...
```

If inserted exactly there, treatment rename would happen before ledger rename. I would move the treatment sidecar block after the ledger-file rename block.

Also: if ledger rename itself fails, I would consider that fatal or at least retry it too. Directory move is the big one, but ledger rename failure can still leave:

```text
<new_id>/
    pending_<ts>_ledger.json
```

or duplicate ledgers after the next `save()`.

For Phase B minimum change, directory-move hard-fail is the priority. But if you are touching this function anyway, retrying/hard-failing the ledger-file rename would be a defensible small hardening.

---

### Q4: Other `pending_<ts>_*` files you might be missing

I cannot audit the repo from the excerpt, so I would not claim there are or are not other writers.

Known from your description:

```text
pending_<ts>_ledger.json
pending_<ts>_treatment.txt
```

Possible suspects to grep for:

```bash
rg "treatment" nodes
rg "episode_id" nodes
rg "_ledger.json" nodes
rg "_treatment.txt" nodes
rg "out_dir" nodes
rg "pending_" nodes
rg "save_ledger_safe" nodes
```

Also grep for filename construction patterns:

```bash
rg "f\".*episode_id.*\"" nodes
rg "f'.*episode_id.*'" nodes
rg "_script.txt|_summary.txt|_metadata.txt|_prompt.txt|_notes.txt" nodes
```

For the code itself, I would not glob every `pending_*_treatment.txt`. Use the actual old episode id prefix.

Prefer:

```python
old_prefix = f"{old_id}_"
for tx in Path(new_audio_dir).glob(f"{old_prefix}*.txt"):
    ...
```

not:

```python
Path(new_audio_dir).glob("pending_*_treatment.txt")
```

The broad glob can accidentally rename unrelated pending treatment files if they ever land in the same directory due to a bug or manual copy.

Whether to rename all `old_id_*.txt` or only `old_id_treatment.txt` depends on your audit. My practical recommendation:

- Phase B minimal: rename exact `old_id_treatment.txt`.
- Slightly more future-proof: rename `old_id_*.txt` to `new_id_*.txt`.
- Do not rename wav/png/mp4 by wildcard unless you verify downstream expects canonical names for those. Do not risk touching audio bytes or replacing media artifacts unnecessarily.

Renaming txt sidecars does not affect C7 audio byte identity.

---

### Q5: Concurrent writer interleaving

Given your stated model — ComfyUI queue strictly sequential and no concurrent pipeline OS process — there should not be another node interleaving between:

```python
os.replace(old_ep_dir, new_ep_dir)
```

and:

```python
self.episode_id = new_id
```

within the same `rename_episode()` call.

A normal synchronous Python node call is not preempted by another workflow node in the middle of a function. So `_otr_ledger.save_ledger_safe()` should not run “between lines” unless you have:

- background threads,
- async callbacks,
- subprocesses,
- file watchers,
- UI preview callbacks writing ledger state,
- or another ComfyUI process pointed at the same output root.

The `find_most_recent_ledger()` walker could theoretically observe an in-between state only if something else runs concurrently. In your stated architecture, it should not.

That said, the walker is another reason hard-failing on rename failure is correct. Once Phase A removed the global walker from `rtx_upscale.py`, the remaining walker use in `_otr_ledger.save_ledger_safe()` is less exposed, but it still benefits from a clean invariant:

```text
An episode has exactly one canonical directory and one canonical ledger.
```

---

### Q6: Hardening tests to add

I would put these in a new file if the existing `tests/test_core.py` is already broad/noisy:

```text
tests/test_ledger_rename.py
```

Specific tests I would add:

#### 1. Happy path

Setup:

```text
episodes/
    pending_123/
        pending_123_ledger.json
        pending_123_treatment.txt
```

Call:

```python
ledger.rename_episode("final_ep")
```

Assert:

```text
episodes/
    final_ep/
        final_ep_ledger.json
        final_ep_treatment.txt
```

Assert no old pending files remain:

```python
assert not (final_dir / "pending_123_ledger.json").exists()
assert not (final_dir / "pending_123_treatment.txt").exists()
assert not old_dir.exists()
```

Assert in-memory state:

```python
assert ledger.episode_id == "final_ep"
assert ledger.data["episode_id"] == "final_ep"
assert Path(ledger.out_dir) == expected_new_audio_dir
```

#### 2. Directory move fails twice, then succeeds

Mock `os.replace` so only the directory move raises `OSError` on the first two calls, then succeeds.

Important: `rename_episode()` also uses `os.replace()` for ledger/treatment renames, so your mock should distinguish source/destination paths.

Assert:

- no `RuntimeError`,
- 3 directory move attempts,
- final directory exists,
- in-memory state updated,
- canonical ledger exists.

Also monkeypatch sleep to avoid slowing tests:

```python
monkeypatch.setattr(production_ledger._time, "sleep", lambda _: None)
```

If you import `time as _time` inside the function, this is harder to patch. Prefer module-level:

```python
import time as _time
```

#### 3. Directory move fails all attempts

Mock directory `os.replace()` to always raise `OSError`.

Assert:

```python
with pytest.raises(RuntimeError, match="per-episode dir move failed"):
    ledger.rename_episode("final_ep")
```

Then assert in-memory state unchanged:

```python
assert ledger.episode_id == "pending_123"
assert ledger.data["episode_id"] == "pending_123"
assert Path(ledger.out_dir) == old_audio_dir
```

Assert on-disk state unchanged:

```python
assert old_dir.exists()
assert not final_dir.exists()
assert (old_dir / "pending_123_ledger.json").exists()
assert (old_dir / "pending_123_treatment.txt").exists()
```

#### 4. Destination directory already exists

This is important and not in your current list.

Setup:

```text
episodes/
    pending_123/
        pending_123_ledger.json
    final_ep/
        some_file.txt
```

Call:

```python
ledger.rename_episode("final_ep")
```

Assert:

- raises `RuntimeError`,
- in-memory state unchanged,
- both directories remain untouched.

This catches the “partial dir from previous crash” case.

#### 5. Old missing, new exists

This is the idempotent/recovery-ish case.

Setup:

```text
episodes/
    final_ep/
        final_ep_ledger.json
```

No `pending_123/`.

Depending on your desired semantics, either:

A. Accept as already moved and update `out_dir` to final dir, or  
B. Raise because the source disappeared unexpectedly.

For production, I would accept only if the canonical ledger exists. Otherwise raise.

Test whichever behavior you choose.

#### 6. Treatment rename fails after ledger rename succeeds

Mock `os.replace()` so:

- directory move succeeds,
- ledger rename succeeds,
- treatment rename raises `OSError`.

Assert:

- no crash if you choose warning-only,
- warning logged,
- canonical ledger exists,
- pending treatment still exists,
- no canonical treatment exists unless pre-existing.

This matches your proposed tolerance.

#### 7. Multiple owned txt sidecars

Only add this if you implement `old_id_*.txt` sidecar renaming.

Setup:

```text
pending_123_treatment.txt
pending_123_notes.txt
pending_123_prompt.txt
unrelated_pending_999_treatment.txt
```

Assert:

```text
final_ep_treatment.txt
final_ep_notes.txt
final_ep_prompt.txt
unrelated_pending_999_treatment.txt
```

This proves you used the exact old prefix, not broad `pending_*`.

#### 8. Reject or no-op re-rename

Two cases:

```python
ledger.rename_episode("final_ep")
ledger.rename_episode("final_ep")
```

This should be a no-op or clean idempotent success.

But:

```python
ledger.rename_episode("final_ep")
ledger.rename_episode("other_ep")
```

I would probably reject unless your system intentionally supports post-finalization renames. Re-renaming a finalized episode can confuse downstream filenames and cleanup.

---

### Q7: Other Windows-specific footguns

Main ones:

#### Destination directory exists

`os.replace(src_dir, dst_dir)` cannot safely replace a populated existing destination directory on Windows. If `new_ep_dir` already exists, do not silently continue. Raise a conflict error unless you have explicit reconciliation logic.

#### Source missing but destination exists

This can happen after a previous partial success or manual fix. Your current/proposed code would leave `self.out_dir` pointing at the old path because `moved_dir` remains false.

If `old_ep_dir` is missing and `new_ep_dir` exists, you likely want:

```python
self.out_dir = new_audio_dir
```

after validating the destination contains the expected ledger or episode structure.

#### Case-insensitive paths

Windows paths are case-insensitive by default. Compare normalized absolute paths when deciding whether old and new dirs are “the same”:

```python
os.path.normcase(os.path.abspath(old_ep_dir)) == os.path.normcase(os.path.abspath(new_ep_dir))
```

This avoids weirdness if only case changes.

#### Long paths

You already mention slug limits. Keep them. Windows long-path support varies by system policy and Python/runtime behavior.

#### Invalid filename characters

Make sure `new_id` has already been slugified/sanitized before it reaches file naming. If `new_id` can contain `:`, `?`, `*`, quotes, etc., Windows will fail.

#### Broad globs

Avoid:

```python
glob("pending_*_treatment.txt")
```

Prefer exact old id prefix.

#### Sleeping after final attempt

Do not sleep after the final failed attempt. It only slows failure and tests.

#### Open handles

Explorer preview panes, editors, antivirus, and indexer can all hold handles. Retrying helps. Hard-fail message should explicitly mention file locks, permissions, and pre-existing destination.

#### Ledger finder by mtime

Directory/file renames can alter mtimes. If any remaining code uses “most recent ledger” discovery, a rename operation may affect selection order. In your sequential single-run model this is probably fine, but it is another reason to reduce global ledger walking over time.

---

## 2) Factual errors or issues in the proposed fix

### Issue 1: Destination-exists case still creates bad state

Your proposed guard:

```python
if (os.path.exists(old_ep_dir)
        and not os.path.exists(new_ep_dir)
        and old_ep_dir != new_ep_dir):
```

does nothing when both old and new dirs exist.

Then the code proceeds to:

```python
self.episode_id = new_id
self.data["episode_id"] = new_id
```

That is still dangerous.

You should explicitly handle:

```python
old exists, new exists, old != new
```

as a hard error.

---

### Issue 2: Old missing, new exists leaves `out_dir` stale

If a previous attempt already moved the directory but crashed before updating state, then:

```text
old_ep_dir does not exist
new_ep_dir exists
```

Your code skips the move and `moved_dir` remains false, so this block does not run:

```python
if moved_dir:
    self.out_dir = new_audio_dir
```

But in that case `self.out_dir` should probably become `new_audio_dir`.

So distinguish “directory is already at final path” from “nothing happened.”

---

### Issue 3: Treatment block placement in sample is before ledger rename

Your sample says:

```python
if moved_dir:
    self.out_dir = new_audio_dir
    # Rename pending_<ts>_treatment.txt -> <new_id>_treatment.txt
    ...
# ... existing ledger file rename logic unchanged ...
```

That means treatment rename happens before ledger rename.

Based on your own Q3 reasoning, move the treatment rename block after the ledger-file rename logic.

---

### Issue 4: `_slugify(new_id, limit=120)` may not match existing ledger naming

You wrote:

```python
canon = Path(new_audio_dir) / f"{_slugify(new_id, limit=120)}_treatment.txt"
```

If the ledger file rename uses raw `new_id`, or uses a different slug limit/rule, you can create mismatched names.

Use the same canonical episode id string/prefix that the ledger path uses. Ideally:

```python
new_prefix = new_id
```

if `new_id` is already the canonical slug. Do not independently slugify unless that is already what `rename_episode()` does for ledger naming.

---

### Issue 5: Broad treatment glob

This:

```python
Path(new_audio_dir).glob("pending_*_treatment.txt")
```

is broader than needed.

Prefer exact old id:

```python
Path(new_audio_dir).glob(f"{old_id}_treatment.txt")
```

or, if you intentionally handle owned txt sidecars:

```python
Path(new_audio_dir).glob(f"{old_id}_*.txt")
```

---

### Issue 6: Treatment warning means invariant is best-effort

If treatment rename failure is only a warning, then the end-state goal:

```text
exactly one ledger + one treatment per episode dir, both canonical
```

is not guaranteed.

That may be fine. I think warning-only is acceptable for treatment. But be clear in the bug log: Phase B guarantees canonical directory and ledger; treatment canonicalization is best-effort unless you choose to make treatment rename fatal.

---

## 3) Recommended final shape

In `nodes/production_ledger.py`, inside `Ledger.rename_episode()` near the excerpted directory move logic, I would structure it like this.

Pseudo-code, not drop-in exact because I do not have your full path variables:

```python
import time as _time
from pathlib import Path
```

Helper:

```python
def _replace_with_retry(src, dst, label, attempts=3, delay=0.5):
    last_exc = None
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last_exc = exc
            log.warning(
                "[Ledger] %s move attempt %d/%d failed (%s -> %s): %s",
                label, attempt + 1, attempts, src, dst, exc,
            )
            if attempt < attempts - 1:
                _time.sleep(delay)

    raise RuntimeError(
        f"[Ledger] rename_episode: {label} move failed after "
        f"{attempts} attempts ({src} -> {dst}): {last_exc}. "
        f"In-memory state NOT updated. Fix the underlying issue "
        f"(file lock, permissions, pre-existing destination, partial dir) "
        f"and re-queue."
    )
```

Then in `rename_episode()`:

```python
old_id = self.episode_id
```

Normalize path comparison:

```python
same_ep_dir = (
    os.path.normcase(os.path.abspath(old_ep_dir))
    == os.path.normcase(os.path.abspath(new_ep_dir))
)
```

Directory state handling:

```python
old_exists = os.path.isdir(old_ep_dir)
new_exists = os.path.isdir(new_ep_dir)
dir_ready = False
moved_dir = False

if same_ep_dir:
    dir_ready = True
elif old_exists and not new_exists:
    _replace_with_retry(old_ep_dir, new_ep_dir, "per-episode dir")
    moved_dir = True
    dir_ready = True
elif not old_exists and new_exists:
    # Previous move or manual repair. Accept only if this is a known-safe state.
    dir_ready = True
elif old_exists and new_exists:
    raise RuntimeError(
        f"[Ledger] rename_episode: cannot rename episode directory because "
        f"both source and destination exist ({old_ep_dir} -> {new_ep_dir}). "
        f"In-memory state NOT updated. Resolve the partial directory state "
        f"and re-queue."
    )
else:
    raise RuntimeError(
        f"[Ledger] rename_episode: neither source nor destination episode "
        f"directory exists ({old_ep_dir} -> {new_ep_dir}). "
        f"In-memory state NOT updated."
    )
```

Only after this point:

```python
self.episode_id = new_id
self.data["episode_id"] = new_id
self.out_dir = new_audio_dir
```

Note: set `self.out_dir = new_audio_dir` when the final directory is ready, not only when `moved_dir` is true.

Then ledger-file rename should happen before treatment sidecars.

Conceptually:

```python
# Rename ledger file inside the final dir.
# Existing logic here, but preferably with retry/hard-fail too.
```

Then treatment/sidecar rename:

```python
ep_dir = Path(new_audio_dir)

old_prefix = f"{old_id}_"
new_prefix = f"{new_id}_"

# Minimal exact treatment only:
sidecars = list(ep_dir.glob(f"{old_id}_treatment.txt"))

# Or if audit confirms owned txt sidecars:
# sidecars = list(ep_dir.glob(f"{old_prefix}*.txt"))

for tx in sidecars:
    suffix = tx.name[len(old_prefix):]
    canon = ep_dir / f"{new_prefix}{suffix}"

    if tx == canon:
        continue

    try:
        os.replace(str(tx), str(canon))
        log.info("[Ledger] sidecar moved %s -> %s", tx.name, canon.name)
    except OSError as exc:
        log.warning(
            "[Ledger] sidecar rename failed (%s -> %s): %s",
            tx, canon, exc,
        )
```

For the one-commit/single-file version, my recommended behavior matrix is:

| State | Behavior |
|---|---|
| old exists, new missing | retry `os.replace`; hard-fail after attempts |
| old exists, new exists | hard-fail conflict |
| old missing, new exists | accept as already moved, set `out_dir` to new |
| old missing, new missing | hard-fail |
| ledger rename fails | preferably retry/hard-fail, or at minimum do not hide it |
| treatment rename fails | log warning, continue |

This gives you the big payoff: no silent split state.

---

## 4) Things to verify before merging

1. **Exact ledger naming rule.**  
   Confirm whether canonical filenames use raw `new_id` or `_slugify(new_id, limit=120)`. Use the same prefix for treatment.

2. **Actual `rename_episode()` ordering.**  
   Ensure treatment sidecar rename is after ledger-file rename if you want ledger canonicalization to take priority.

3. **Destination-exists handling.**  
   Add explicit hard-fail for:

   ```text
   old dir exists + new dir exists
   ```

4. **Old-missing/new-existing behavior.**  
   Decide whether this should be accepted as already moved. I recommend accepting it only if the final directory contains the expected ledger or recognizable episode contents.

5. **Grep for other pending sidecars.**  
   Run the grep audit before deciding between exact treatment-only rename and `old_id_*.txt`.

6. **No audio-byte changes.**  
   This change should not touch wav bytes. Still, for C7, do a quick before/after run and compare hashes of final audio outputs.

7. **Remaining global ledger walkers.**  
   Confirm `_otr_ledger.save_ledger_safe()` cannot run concurrently during `rename_episode()` and that remaining `find_most_recent_ledger()` use is sequential.

8. **Windows manual smoke test.**  
   On Windows 11, run one episode with Explorer open in the episode root. Then run another with Defender active. Confirm failure messages are clear if you intentionally create a destination conflict.

9. **Test monkeypatching.**  
   If tests mock `os.replace`, remember directory move, ledger rename, and treatment rename all call the same function. Make the mock path-sensitive.

10. **Bug log wording.**  
   Record the invariant you are enforcing:

   ```text
   rename_episode either completes with canonical episode dir + canonical ledger,
   or raises before mutating in-memory episode state.
   ```

That invariant is the heart of Phase B.

---

## Gemini (gemini-3.1-pro-preview-customtools)

**1. State whether you AGREE, PARTIALLY AGREE, or DISAGREE**
I AGREE with the core recommendation to hard-fail on directory move failures and to sequence the treatment rename *after* the ledger rename, but I advise keeping the retry logic inline rather than extracting a new helper function to strictly honor your "smallest change" preference.

**2. FACTUAL ERRORS in the ChatGPT answer**
*   **Windows `os.replace` directory semantics:** ChatGPT states `os.replace(src_dir, dst_dir)` "cannot safely replace a populated existing destination directory on Windows." This is slightly inaccurate and understates the danger. On Windows, `os.replace` (which calls the Win32 API `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`) will **always fail** with an `OSError` (usually `[WinError 5] Access is denied` or `[WinError 145] The directory is not empty`) if the destination is an existing directory, *even if it is completely empty*. You must absolutely check `if os.path.exists(new_ep_dir)` and hard-fail before attempting the replace.

**3. IMPORTANT THAT WAS OMITTED**
*   **Human-held file locks:** ChatGPT focuses entirely on system locks (Windows Defender, Search Indexer) which clear in milliseconds. It omits the most common solo-dev Windows footgun: *you* having `pending_<ts>_treatment.txt` open in Notepad, or a pending `.wav` open in VLC/Media Player. 1.5 seconds of retry will not outwait a human. Your `RuntimeError` message must explicitly tell the user to "close external media players or text editors."
*   **Pathlib vs OS module mixing:** Your snippet mixes `os.replace`, `os.path.exists`, and `Path(new_audio_dir).glob`. Since ComfyUI runs on Python 3.9+, `Path.replace()` and `Path.exists()` are available and cleaner. Sticking to one paradigm reduces cognitive load, though `os.replace` is perfectly fine if you want the absolute smallest diff.
*   **VRAM and Rule C7 Confirmation:** ChatGPT didn't explicitly close the loop on your constraints. I will: Because this is purely a control-plane OS metadata operation happening sequentially between GPU node executions, this change costs **0 bytes of VRAM** and mathematically cannot affect the audio byte-identity (Rule C7). 

**4. Give your own short recommendation (3-6 bullets)**
*   **Implement the Hard-Fail:** Adopt the 3 attempts × 0.5s inline retry loop. If it fails, raise `RuntimeError` immediately. Do *not* update `self.episode_id`, `self.data`, or `self.out_dir`. Dead-stopping the ComfyUI queue is vastly preferable to poisoning the ledger state.
*   **Strict State Matrix:** Adopt ChatGPT's logic for checking `old_exists` and `new_exists`. If `new_ep_dir` already exists, raise a `RuntimeError` immediately. Do not attempt to merge or overwrite directories.
*   **Order of Operations:** Enforce this strict sequence in `nodes/production_ledger.py`: 1) Move Directory -> 2) Update in-memory state (`self.episode_id`, `self.out_dir`) -> 3) Rename Ledger File -> 4) Rename Treatment/Sidecar files. 
*   **Targeted Sidecar Globbing:** Use `Path(new_audio_dir).glob(f"{old_id}_*.txt")`. This safely catches `_treatment.txt` and any future text sidecars (like prompts or notes) without the risk of the overly broad `pending_*` glob.
*   **Keep it inline:** Don't build the `_replace_with_retry` helper ChatGPT suggested. You only do this in one or two places. Keep the `for attempt in range(3):` loop inline inside `rename_episode` to minimize the blast radius of your diff.

**5. Note any items where you are uncertain and would want to verify**
*   **Slugification mismatch:** I am uncertain if `new_id` is *always* pre-slugified before being passed to `rename_episode(new_id)`. Your snippet does `canon = Path(...) / f"{_slugify(new_id, limit=120)}_treatment.txt"`. If the ledger file rename logic uses raw `new_id` and the treatment uses `_slugify(new_id)`, you will end up with mismatched prefixes (e.g., `My Episode_ledger.json` vs `my-episode_treatment.txt`). Verify exactly how the ledger filename is constructed and mirror it identically.
*   **Downstream Node File Handles:** I am uncertain if any of your custom audio nodes (Bark, MusicGen) or ComfyUI preview nodes accidentally leak open file handles to the pending `.wav` files *during* the workflow execution. If a node writes a file but forgets to close the handle, the directory move will consistently fail on Windows. Verify your audio nodes use `with open(...)` or properly close their file descriptors.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### 1. Direct Answers to Q1-Q7  

**Q1: Retry semantics**  
**Agree with Gemini's correction of ChatGPT.**  
- Use **3 attempts × 0.5s** inline (no helper function).  
- **Enhance error message** to include: *"Check for manually locked files (e.g., open in editors/players)."*  
- **Why:** Defender/indexer locks clear quickly, but human-held locks (e.g., Notepad, VLC) require user intervention. Retries won’t resolve these, so the error message must guide the user.  

**Q2: Hard-fail vs split-state recovery**  
**Agree with both.** Hard-fail is correct.  
- **Add explicit check:** If `new_ep_dir` exists, raise `RuntimeError` immediately.  
- **Why:** Split state leads to downstream chaos. Immediate failure is safer.  

**Q3: Treatment rename ordering**  
**Agree: Ledger before treatment.**  
- **Why:** Ledger is the source of truth. A failed treatment rename is recoverable; a failed ledger rename is not.  

**Q4: Other `pending_<ts>_*` files**  
**Audit required.**  
- **Recommend:** Use `old_id_*.txt` glob (not `pending_*`).  
- **Why:** Avoid accidental renames of unrelated files. If `old_id` is the original episode ID, this ensures only owned files are renamed.  

**Q5: Concurrent writer interleaving**  
**No risk in current model.**  
- **Why:** ComfyUI is strictly sequential. No background threads or parallel processes. `_otr_ledger.save_ledger_safe()` cannot interleave during `rename_episode()` because it’s part of the same sequential workflow.  

**Q6: Hardening tests**  
**Add these tests:**  
1. **Happy path:** All files renamed, state updated.  
2. **Dir move fails 2/3 attempts then succeeds.**  
3. **Dir move fails all 3 attempts:** `RuntimeError`, state unchanged.  
4. **Destination exists:** Hard-fail.  
5. **Old missing, new exists:** Treat as already moved.  
6. **Treatment rename fails after ledger succeeds:** Warning only.  
7. **Re-rename idempotency:** Second call to `rename_episode` with same `new_id` is no-op.  

**Q7: Windows-specific footguns**  
**Key risks:**  
- **Destination directory exists:** Check before retrying.  
- **Human-held file locks:** Error message must warn about this.  
- **Case-insensitive paths:** Normalize with `os.path.normcase()` when comparing paths.  

---

### 2. Factual Errors in Previous Answers  

**ChatGPT’s `os.replace` claim:**  
- **Error:** Stated that `os.replace` can replace an empty destination directory on Windows.  
- **Correction (Gemini):** On Windows, `os.replace` **always fails** if the destination directory exists, even if empty.  

**ChatGPT’s helper function suggestion:**  
- **Issue:** Extracting a helper (`_replace_with_retry`) increases diff size.  
- **Fix:** Keep retry logic inline to minimize changes.  

---

### 3. Important Omissions by Both  

1. **Slugification consistency:**  
   - Must ensure `new_id` is pre-slugified **or** apply the same slug function used for the ledger to treatment/sidecars.  
   - **Risk:** Mismatched names (e.g., `My_Episode_ledger.json` vs `my-episode_treatment.txt`).  

2. **Open file handles in nodes:**  
   - If audio/text nodes (e.g., `AudioGen`, `OTR_LLMScriptWriter`) leave files open, directory moves will fail on Windows.  
   - **Action:** Audit nodes for proper file closure (e.g., `with open(...)`).  

3. **Idempotency of `rename_episode`:**  
   - Calling `rename_episode(new_id)` twice should either no-op or fail.  
   - **Test:** Ensure second call detects the directory is already renamed.  

---

### 4. Prioritized Recommendations  

1. **Hard-fail directory move with retry:**  
   - Inline 3 attempts × 0.5s.  
   - **Error message:** Include advice to check for manually locked files.  

2. **Strict directory state handling:**  
   - If `new_ep_dir` exists → `RuntimeError`.  
   - If `old_ep_dir` missing but `new_ep_dir` exists → treat as already moved.  

3. **Sequence operations:**  
   - **Order:** Move dir → Update state → Rename ledger → Rename sidecars.  

4. **Precise sidecar renaming:**  
   - Use `old_id_*.txt` glob.  
   - Avoid `pending_*` to prevent accidental renames.  

5. **Slug consistency:**  
   - Ensure `new_id` is slugified **before** `rename_episode` is called, or apply the same slug function to both ledger and treatment.  

---

### 5. Uncertainties Needing Verification  

1. **Slugification workflow:**  
   - Is `new_id` pre-slugified by the caller, or does `rename_episode` need to slugify it?  

2. **Open file handles:**  
   - Do nodes like `AudioGen` or `OTR_LLMScriptWriter` properly close files?  

3. **Idempotency:**  
   - Does `rename_episode(final_ep)` called twice behave correctly?  

4. **Path normalization:**  
   - Are paths compared case-insensitively using `os.path.normcase()`?  

---

### Final Recommendation Summary  

- **Adopt hard-fail with retry** (inline, 3×0.5s).  
- **Enforce strict directory state checks** (existing dirs = error).  
- **Rename ledger before treatment.**  
- **Use `old_id_*.txt` for sidecars.**  
- **Verify slug consistency and file handle closure in nodes.**  
- **Add tests for edge cases (existing dirs, idempotency).**

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
