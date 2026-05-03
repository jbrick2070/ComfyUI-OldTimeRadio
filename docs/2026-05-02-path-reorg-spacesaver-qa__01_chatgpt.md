# Round A -- ChatGPT (gpt-5.5) elapsed=105.7s

Below is a QA pass focused only on runtime risks in the shipped path-reorg / spacesaver work. I’m not relitigating the `otr/obs/` vs `otr/episodes/<ep>/` layout.

I don’t have your actual repo line numbers from the excerpts, so I’ll cite grep targets / functions.

---

## 1. `Ledger.rename_episode` dir-move safety

- **Mostly sound for the happy path, but I would harden it before calling it final.**
- Grep target: `production_ledger.py`, `def rename_episode`.

### Windows `os.replace(old_ep_dir, new_ep_dir)` behavior

- On the same filesystem, a directory rename is effectively atomic: it should either succeed or fail, not half-move the tree.
- On Windows, failure is plausible if:
  - any file under the tree is open with incompatible sharing flags,
  - antivirus / indexer briefly holds a handle,
  - `new_ep_dir` already exists,
  - source/destination are on different volumes,
  - path length / permission issue.
- Worst case should generally be **no directory move**, not half-moved files.
- So: **yes, add a retry loop with short backoff**. Not because of halfway moves, but because transient Windows locks are common.

Suggested shape:

```python
for attempt in range(5):
    try:
        os.replace(old_ep_dir, new_ep_dir)
        moved_dir = True
        break
    except PermissionError:
        time.sleep(0.10 * (attempt + 1))
    except OSError:
        ...
```

### Bigger issue: if `new_ep_dir` already exists, you silently fall back to file-only rename

Current logic:

```python
if os.path.exists(old_ep_dir) and not os.path.exists(new_ep_dir):
    os.replace(old_ep_dir, new_ep_dir)
```

If `new_ep_dir` exists, you do **not** move or merge the workspace. Then:

```python
self.episode_id = new_id
...
if moved_dir:
    self.out_dir = new_audio_dir
```

So if `moved_dir == False`, you now have:

- `ledger.data["episode_id"] == new_id`
- ledger file possibly renamed inside `pending_<ts>/audio/`
- assets still under `pending_<ts>/`
- final episode id says `<new_id>`

That is survivable, but confusing and unsafe for later cleanup/discovery.

**Recommendation:** if `new_ep_dir` exists, treat as a collision and generate a disambiguated episode id, e.g. `<new_id>_2`, or fail loudly before updating `self.episode_id`.

Do not silently leave a finalized episode in a `pending_*` workspace unless that is an explicit recovery mode.

### Possible missing rename: treatment file

Your rename function only renames:

```python
<_old_slug>_ledger.json -> <_new_slug>_ledger.json
```

But your final layout and spacesaver keep-list expect:

```text
<episode_id>_treatment.txt
```

If the treatment is originally written as:

```text
pending_<ts>_treatment.txt
```

and nothing else renames it, then spacesaver will delete it.

Grep targets:

- `production_ledger.py`, `rename_episode`
- treatment writer, likely `*_treatment.txt`
- `rtx_upscale.py`, `_spacesaver_cleanup_if_flagged`

This is a concrete failure mode:

> `perfect_run_spacesaver=True` can wipe the treatment because keep-list uses final episode id but the file may still be named with the pending id.

If treatment is already renamed elsewhere, fine. If not, add treatment rename to `Ledger.rename_episode`.

### Slug mismatch risk

In `rename_episode`, ledger filenames use:

```python
_slugify(old, limit=120)
_slugify(new_id, limit=120)
```

But spacesaver keep-list uses raw `ep_id`:

```python
audio_dir / f"{ep_id}_ledger.json"
audio_dir / f"{ep_id}_treatment.txt"
```

If episode ids can contain punctuation, spaces, quotes, Unicode oddities, etc., then spacesaver may fail to recognize the real ledger/treatment and delete them.

**Fix:** use the actual discovered `ledger_path` as the ledger keep file, and use the same slug helper for treatment, or discover treatment by metadata.

Example:

```python
keep_files = {ledger_path.resolve()}
treatment = audio_dir / f"{_slugify(ep_id, limit=120)}_treatment.txt"
keep_files.add(treatment.resolve())
```

---

## 2. MusicGen + AudioGen `_cache_dir()` race condition

- Grep targets:
  - `musicgen_theme.py`, `def _cache_dir`
  - AudioGen equivalent, `def _cache_dir`
  - `_otr_ledger.py`, `find_most_recent_ledger`

### Filename collision: yes, possible

Current key:

```python
ts_ms = int(time.time() * 1000)
return f"{cue_id}_{digest}_{ts_ms}.wav"
```

Two calls in the same millisecond with same cue id / prompt / duration / seed can produce the same filename.

On a single ComfyUI queue this may be unlikely. With concurrent queued workflows or parallel node execution, it is plausible.

**Minimum fix:** add a nonce with more entropy.

```python
import uuid
return f"{cue_id}_{digest}_{ts_ms}_{uuid.uuid4().hex[:8]}.wav"
```

However, see C7 note below.

### Bigger race: `_cache_dir()` can choose the wrong ledger

The real concurrency bug is not only filename collision. It is this:

```python
find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])
```

If two workflows run concurrently, both MusicGen/AudioGen nodes may discover the same “most recent” ledger. Then they write to the wrong episode’s audio dir.

The timestamp suffix reduces filename overwrite risk, but it does not fix wrong-episode association.

For destructive or episode-associated outputs, “most recent ledger globally” is inherently race-prone.

**Best fix:** pass the current ledger path / episode workspace through `project_state` or node inputs instead of rediscovering globally.

**Smallest acceptable hardening if you don’t want to refactor now:**

- Add a run/workflow id to the pending ledger early.
- Have MusicGen/AudioGen search for a ledger matching that id if available.
- Otherwise fall back to most-recent with a warning.

### C7 / byte-identical warning

Rule C7 says audio output must remain byte-identical between runs.

Timestamp/UUID filenames are nondeterministic. That is okay only if filenames never affect:

- prompt ordering,
- cue ordering,
- ledger serialization that later drives assembly,
- glob sorting,
- mux metadata,
- cache hit/miss behavior,
- final audio selection.

If any downstream stage glob-sorts these files and uses filename order, timestamp/UUID suffixes can alter output selection/order.

For C7, safer pattern:

- use deterministic content key for the semantic cue:
  - `cue_id + prompt + duration + episode_seed + cue_index`
- add a collision-safe write method:
  - write to temp file,
  - atomically replace target,
  - or if target exists, verify it is the expected file and reuse it.

If you truly need unique per-render files, UUID is operationally safer, but less C7-friendly.

---

## 3. Spacesaver cleanup safety

- Grep target: `rtx_upscale.py`, `_spacesaver_cleanup_if_flagged`.

### I would not ship the current destructive cleanup as-is

The destructive cleanup currently locates the ledger by:

```python
find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])
```

For a destructive operation, global “most recent ledger” is too risky.

Concrete failure mode:

> RTXUpscale finishes episode A, but episode B has a newer pending/final ledger mtime. Spacesaver reads B’s ledger and wipes B’s intermediates instead of A’s.

This is the highest-risk bug in the excerpts.

**Fix:** derive the episode workspace from the actual `src` being upscaled/composited, not global recency.

For example:

- If `src` is under `output/otr/episodes/<ep>/...`, resolve `<ep>`.
- Then find ledger under:
  - `output/otr/episodes/<ep>/audio/*_ledger.json`
- Only if that direct lookup fails, consider fallback discovery, and do not wipe on fallback unless explicitly safe.

### Current guard is too weak

Current guard:

```python
parts_lower = [p.lower() for p in ep_dir.parts]
if "episodes" not in parts_lower or "otr" not in parts_lower:
    return
```

This allows many unintended paths, for example:

```text
C:\tmp\otr\episodes\somewhere
D:\scratch\not-output\otr\episodes\foo
C:\Users\...\otr\episodes\foo
```

It proves the path contains components named `otr` and `episodes`; it does not prove it is inside your actual ComfyUI output tree.

**Fix:** resolve against canonical root:

```python
root = otr_episodes_root().resolve()
ep_dir_real = ep_dir.resolve()

try:
    ep_dir_real.relative_to(root)
except ValueError:
    refuse
```

Also require:

```python
audio_dir.name == "audio"
ep_dir_real.parent == root
```

That prevents wiping nested accidents like:

```text
output/otr/episodes/foo/audio/bar/audio/
```

### Symlinks / junctions

You’re right to worry.

`shutil.rmtree()` behavior around symlinks/junctions is safer in modern Python than it used to be, but I would not rely on it for destructive project cleanup.

Before deleting sibling dirs, check that each target resolves inside the episode dir:

```python
sub_real = sub.resolve()
try:
    sub_real.relative_to(ep_dir_real)
except ValueError:
    refuse / skip
```

Also consider refusing to delete symlinks/junctions outright:

```python
if sub.is_symlink():
    sub.unlink()
    continue
```

On Windows, junction detection is awkward. If you want to be extra conservative, do not allow reparse points in spacesaver targets.

### Windows file locks

Current code appears to have no per-file exception handling in the destructive loops:

```python
child.unlink()
shutil.rmtree(sub, ignore_errors=True)
```

Problems:

- `child.unlink()` can raise `PermissionError` and abort the rest of cleanup.
- `shutil.rmtree(..., ignore_errors=True)` hides partial failures.
- You may think cleanup succeeded when it didn’t.

**Fix:** collect failures and log/stamp them.

Pseudo-shape:

```python
deleted = []
failed = []

try:
    child.unlink()
    deleted.append(str(child))
except Exception as exc:
    failed.append((str(child), repr(exc)))
```

For dirs, avoid `ignore_errors=True`; use `onerror`/`onexc` and collect failures.

### Yes, stamp cleanup result into the ledger

Add:

```json
"meta": {
  "perfect_run_spacesaver": true,
  "spacesaver_executed": true,
  "spacesaver_executed_at": "...",
  "spacesaver_deleted_counts": {
    "files": 123,
    "dirs": 4
  },
  "spacesaver_failed": [],
  "spacesaver_target_ep_dir": "..."
}
```

If skipped:

```json
"spacesaver_executed": false,
"spacesaver_skip_reason": "ledger_not_found"
```

Caveat: write the stamp before deleting anything else, or ensure the ledger is in the keep-list. Since the ledger is supposed to survive, post-cleanup stamp is also okay, but do it carefully with atomic JSON write.

### Silent skip is safe but bad for diagnosis

This is safe:

```python
if ledger_path is None:
    return
```

But it will hide bugs. At least append to `log_lines`:

```text
spacesaver skipped: ledger discovery failed
```

Do not turn this into a hard failure unless the owner expects spacesaver to be strict. Final MP4 should not be lost because cleanup bookkeeping failed.

---

## 4. Back-compat with existing episodes

- Grep target: `_otr_ledger.py`, `find_most_recent_ledger`.

### The two-layer glob is basically sound for read compatibility

Current:

```python
candidates.extend(d.glob("*_ledger.json"))
candidates.extend(d.glob("*/audio/*_ledger.json"))
```

For the intended roots:

```python
[otr_episodes_root(), otr_legacy_audio_dir()]
```

this is fine for:

- old flat legacy ledgers,
- new `episodes/<ep>/audio/` ledgers.

### Edge cases

- If `otr_legacy_audio_dir()` contains a directory shaped like:

```text
output/otr/audio/something/audio/foo_ledger.json
```

then the second glob will pick it up.

That is not inherently corrupting, but it broadens discovery more than the docstring implies when used on legacy roots.

### Race in `max(candidates, key=lambda p: p.stat().st_mtime)`

If another process deletes/moves a candidate between glob and `stat()`, this can raise.

For normal read code, rare. For your rename and spacesaver timing, plausible.

Harden:

```python
scored = []
for p in candidates:
    try:
        scored.append((p.stat().st_mtime, p))
    except FileNotFoundError:
        continue
return max(scored)[1] if scored else None
```

### Pending ledgers can win

Because you pick newest by mtime, a `pending_*_ledger.json` may win over a final episode ledger.

That is okay for “find the current in-flight ledger” but dangerous for “find the ledger for the episode I’m currently finalizing.”

So I’d keep this function for broad discovery, but avoid it in destructive cleanup.

---

## 5. Ledger schema

### Should you bump schema version?

I would bump or at least add explicit layout metadata.

The ledger format has gained runtime semantics:

- per-episode workspace,
- pending-to-final rename,
- spacesaver destructive mode,
- final OBS path split from intermediate episode assets.

Even if the JSON shape is backward compatible, the meaning of paths changed.

Current schema:

```text
l3-2026-04-28
```

I’d move to something like:

```text
l4-2026-05-02
```

or keep schema but add:

```json
"meta": {
  "layout_version": "otr-workspace-v2"
}
```

If you have existing consumers that key off schema and may break, the lower-risk path is:

- keep schema version,
- add `meta.layout_version`.

But architecturally, yes, this deserves a schema/layout marker.

### Recommended meta fields

I would add these:

```json
"meta": {
  "layout_version": "otr-per-episode-workspace-v2",
  "output_obs_path": "output/otr/obs/<episode_id>.mp4",
  "episode_workspace": "output/otr/episodes/<episode_id>",
  "audio_dir": "output/otr/episodes/<episode_id>/audio",

  "dir_moved_from_pending": "pending_...",
  "dir_move_succeeded": true,
  "dir_move_error": null,

  "perfect_run_spacesaver": false,
  "spacesaver_executed": false,
  "spacesaver_executed_at": null,
  "spacesaver_skip_reason": null,
  "spacesaver_deleted_counts": null,
  "spacesaver_failures": []
}
```

For post-run forensics, the highest-value ones are:

1. `meta.layout_version`
2. `meta.dir_moved_from_pending`
3. `meta.dir_move_succeeded`
4. `meta.spacesaver_executed`
5. `meta.spacesaver_skip_reason` / `meta.spacesaver_failures`

### Avoid divergent duplicate flags

You mentioned `meta.perfect_run_spacesaver` and also nesting it under `meta.gen_params_initial`.

That is okay as a historical record, but the canonical control flag should be one place. I’d define:

- canonical runtime flag: `meta.perfect_run_spacesaver`
- original user input snapshot: `meta.gen_params_initial.perfect_run_spacesaver`

Do not let later code read both and disagree.

---

## 6. Gaps for final hardening tests

Your proposed tests are exactly the right direction. I would add these.

### `Ledger.rename_episode` functional tests

Temp-dir test:

- Create:

```text
episodes/pending_123/audio/pending_123_ledger.json
episodes/pending_123/audio/pending_123_treatment.txt
episodes/pending_123/audio/opening_x.wav
episodes/pending_123/videos/foo.mp4
```

- Call:

```python
ledger.rename_episode("final_ep")
```

Assert:

```text
episodes/final_ep/audio/final_ep_ledger.json exists
episodes/final_ep/audio/final_ep_treatment.txt exists   # if you implement treatment rename
episodes/final_ep/videos/foo.mp4 exists
episodes/pending_123 does not exist
ledger.out_dir == episodes/final_ep/audio
ledger.data["episode_id"] == "final_ep"
```

Also test:

- destination exists,
- move fails via monkeypatched `os.replace`,
- old ledger missing,
- slugified episode id,
- Windows-ish unsafe chars if your slugifier handles them.

### `find_most_recent_ledger` tests

Create:

```text
legacy_audio/old_ledger.json
episodes/ep1/audio/ep1_ledger.json
episodes/ep2/audio/ep2_ledger.json
```

Assert it finds the newest.

Also test deletion race by monkeypatching `Path.stat` or deleting a candidate before stat.

### Spacesaver keep-list tests

Create full fake workspace:

```text
episodes/ep/audio/ep_ledger.json
episodes/ep/audio/ep_treatment.txt
episodes/ep/audio/ep.mp4
episodes/ep/audio/opening.wav
episodes/ep/audio/director_dump_1.txt
episodes/ep/stills/a.png
episodes/ep/portraits/a.png
episodes/ep/videos/a.mp4
episodes/ep/composited/a.mp4
```

Set:

```json
meta.perfect_run_spacesaver = true
```

Run cleanup.

Assert only ledger/treatment remain.

Also test:

- flag false leaves everything,
- ledger missing skips,
- invalid JSON skips/logs,
- spacesaver stamp is written,
- locked file produces failure entry, not full abort.

### Spacesaver guard tests

Assert cleanup refuses:

```text
C:/tmp/otr/episodes/ep
output/not_otr/episodes/ep
output/otr/episodes/ep/audio/../../other
```

Use resolved `relative_to(root)` guard.

### Wrong-ledger regression test

This is important.

Create two episodes:

```text
episodes/A/audio/A_ledger.json  # spacesaver true
episodes/B/audio/B_ledger.json  # newer mtime, spacesaver true
```

Call RTX cleanup for a `src` belonging to A.

Assert A is cleaned and B is untouched.

This test will likely fail with current global `find_most_recent_ledger` behavior. That’s the clearest proof of the destructive cleanup bug.

### Concurrent workflow smoke test

Even if you do not fully support concurrency, add a test documenting behavior:

- Two pending ledgers.
- MusicGen `_cache_dir()` resolves one globally.
- Show that without explicit project state it can choose wrong.

If you decide single-queue only is supported, add an explicit warning/log or workflow lock.

### C7 determinism test

Given same seeds and same script inputs:

- run the audio-producing path twice into clean temp dirs,
- hash final WAV/muxed audio stream,
- assert byte-identical.

Also inspect whether timestamped filenames leak into:

- ledger fields used by assembler,
- cue ordering,
- ffmpeg concat lists,
- output metadata.

---

## 7. Anything else I’d flag

### Highest-priority reject/fix items

I would not reject the overall layout work, but I would block final hardening on these:

1. **Destructive spacesaver must not use global most-recent ledger.**  
   Grep: `rtx_upscale.py`, `_spacesaver_cleanup_if_flagged`, call to `find_most_recent_ledger`.

2. **Spacesaver keep-list should use slug-consistent / actual ledger path.**  
   Grep: `keep_files = { audio_dir / f"{ep_id}_ledger.json", ... }`.

3. **Treatment rename must be verified.**  
   Grep: `rename_episode`, `_treatment.txt`.  
   If treatment remains `pending_*_treatment.txt`, spacesaver will delete it.

4. **MusicGen/AudioGen global ledger discovery is not concurrency-safe.**  
   Grep: `musicgen_theme.py`, `_cache_dir`; AudioGen equivalent.

### Add an episode/workflow lock if concurrency is unsupported

Since this is a single Windows workstation and concurrent queues are “unlikely but possible,” the smallest safe policy may be:

- add a local lock file under `output/otr/.otr_generation.lock`,
- acquire at script-writer start,
- release at final RTXUpscale / terminal node,
- if lock exists, fail with a clear message or wait.

This is much smaller than refactoring every node to carry explicit workspace state, and it prevents most “most recent ledger” race bugs.

Use care with stale locks:

```json
{
  "pid": 1234,
  "created_at": "...",
  "episode_id": "pending_..."
}
```

On Windows, use a real file lock if possible, not just existence.

### Atomic ledger writes

For any new meta stamps, write ledger atomically:

```python
tmp = ledger_path.with_suffix(".json.tmp")
tmp.write_text(...)
os.replace(tmp, ledger_path)
```

Do not partially write JSON right before spacesaver.

### Be careful with mtime-based discovery after adding timestamps

You now create many files in `audio/`, but `find_most_recent_ledger` keys off ledger mtime only. If later nodes update the ledger after MusicGen/AudioGen, okay. If not, stale ledgers can win/lose unexpectedly.

Consider storing:

```json
meta.last_touched_at
```

and update it deliberately when the episode is active, or avoid global mtime discovery for anything important.

### Bypass mode final copy

You mentioned RTXUpscale bypass mode copies source to `otr/obs/`.

Make sure spacesaver executes after the copy completes and after file handles are closed. On Windows, deleting `composited/` while ffmpeg/copy still has an open handle will fail or partially skip.

### OBS final should be protected

Spacesaver appears to only wipe under `episodes/<ep>/`, which is correct. Add a test asserting it never deletes:

```text
output/otr/obs/<episode_id>.mp4
```

### Log clarity

Add one high-signal log line when spacesaver runs:

```text
[OTR_RTXUpscale] spacesaver: ep=<ep_id> root=<ep_dir> deleted_files=N deleted_dirs=N failures=M
```

And when skipped:

```text
[OTR_RTXUpscale] spacesaver skipped: reason=<reason> src=<src>
```

---

## Bottom line

- **Path reorg is directionally sound.**
- **Back-compat walker is acceptable for non-destructive discovery.**
- **`Ledger.rename_episode` is close, but needs retry/collision handling and likely treatment rename.**
- **MusicGen/AudioGen timestamp suffix should be made collision-proof, but the deeper issue is global most-recent ledger selection.**
- **Spacesaver needs one important fix before I’d trust it: derive the target episode from the actual RTXUpscale source/final context, not from global most-recent ledger.**

If you only make one hardening change, make it that spacesaver target resolution change. That has the largest safety payoff.
