# Round A -- ChatGPT (gpt-5.5) elapsed=150.8s

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
