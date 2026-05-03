# Synthesis -- 2026-05-02

**Question:** # Phase D consult — MusicGen + AudioGen cache key fix

## Context

OTR (Old Time Radio) ComfyUI node pack. Phase D of a 5-phase QA pass on the path-reorg / spacesaver work. Touches Rule C7 (audio output bytes must remain identical run-to-run), so per CLAUDE.md cadence this requires a round-robin before merge.

Hardware: Windows, RTX 5080 16GB, single GPU, ComfyUI sequential queue.

## The bug

Two files, identical pattern:

**`nodes/musicgen_theme.py`** (line 119):
```python
def _cache_key(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Format: <cue_id>_<sha8>_<timestamp_ms>.wav"""
    import time as _time
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    ts_ms = int(_time.time() * 1000)
    return f"{cue_id}_{digest}_{ts_ms}.wav"
```

**`nodes/batch_audiogen_generator.py`** (line 79):
```python
def _cache_key(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Format: sfx_<safe_prompt_prefix>_<sha8>_<timestamp_ms>.wav"""
    import time as _time
    payload = f"{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:20]).lower()
    ts_ms = int(_time.time() * 1000)
    return f"sfx_{safe_name}_{digest}_{ts_ms}.wav"
```

Both are then used in the call site like:
```python
cache_path = os.path.join(cache_dir, _cache_key(cue_id, prompt, duration, episode_seed))
cached = _load_cached_wav(cache_path)
if cached is not None:
    # cache hit
else:
    # generate, save to cache_path
```

**Result:** `_cache_key` returns a path with the *current* millisecond timestamp baked in. `_load_cached_wav` checks if that exact path exists — which it never does because the filename has the *now* timestamp. Cache miss every single run. ~22s of wasted MusicGen rendering per episode + N seconds of wasted AudioGen rendering per SFX cue.

The docstrings explicitly say this is intentional ("guaranteed unique across episodes"), but it defeats the entire purpose of the cache. It also violates Rule C7 (byte-identical audio output across runs) because every run writes a *new* file with a different timestamp, and downstream FFmpeg muxing may embed those filenames into mp4 metadata.

## Phase A context (already landed, won't change)

The spacesaver in `rtx_upscale.py` was rewritten to derive the episode dir directly from the `src` mp4 path. It can no longer wipe the wrong episode's cache.

## Phase B context (already landed, won't change)

`Ledger.rename_episode` now hard-fails on `os.replace` failure with retry. The per-episode workspace lives at `output/otr/episodes/<ep>/audio/` after the pending → finalized rename.

## The proposed fix

Separate the LOOKUP from the WRITE:

```python
def _cache_key_for_lookup(cue_id, prompt, duration_sec, episode_seed):
    """Returns the deterministic prefix for cache discovery via glob."""
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{cue_id}_{digest}"  # NO timestamp

def _cache_key_for_write(cue_id, prompt, duration_sec, episode_seed):
    """Returns the unique-per-write filename. Only used when cache MISS."""
    import time as _time
    prefix = _cache_key_for_lookup(cue_id, prompt, duration_sec, episode_seed)
    ts_ms = int(_time.time() * 1000)
    return f"{prefix}_{ts_ms}.wav"

def _find_cached(cache_dir, prefix):
    """Returns newest path matching <prefix>_*.wav, or None."""
    from pathlib import Path
    matches = sorted(Path(cache_dir).glob(f"{prefix}_*.wav"),
                     key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return str(matches[0]) if matches else None
```

Call site becomes:
```python
prefix = _cache_key_for_lookup(cue_id, cue["prompt"], cue["duration_sec"], episode_seed)
cache_path = _find_cached(cache_dir, prefix)
if cache_path is not None:
    # cache hit -- load
    cached = _load_cached_wav(cache_path)
else:
    # cache miss -- generate, write to fresh timestamped name
    cache_path = os.path.join(cache_dir, _cache_key_for_write(cue_id, ...))
    # ... render and save_wav(cache_path)
```

## What I want your opinion on (please be specific)

**Q1: Is this the right shape?** Two functions (`_cache_key_for_lookup` returning prefix, `_cache_key_for_write` returning filename) plus a `_find_cached` helper. Or would you collapse it differently — maybe one function that takes a `mode='lookup'|'write'` arg?

**Q2: Newest-mtime selection on multiple matches.** If two files exist with the same `<role>_<sha>_*.wav` prefix (e.g., a re-run wrote two), I'm picking newest mtime. Is that right? The alternative is "first lexicographically" since timestamps sort lexicographically, but mtime is more robust to clock drift. Or should I treat duplicate matches as a corruption signal and warn?

**Q3: Rule C7 compatibility.** Phase D restores cache hits, which means the same input on a re-run loads the same wav file (byte-identical). But the *first* time a sha is generated, the file is written with `_<ts_ms>` suffix. Subsequent runs hit that exact file. The mp4 muxer downstream embeds the input wav filename into mp4 metadata streams. With the fix, the same sha → same wav file → same mp4 metadata bytes. Without the fix, every run gets a fresh ts → different wav file → different mp4 bytes. **Confirm: the fix actually makes C7 hold, where the current code violates it.** I want to make sure I'm not missing a subtlety.

**Q4: Mutation tests.** I plan to add a 5-mutation suite, one per dimension that should produce a fresh sha:
1. seed (`episode_seed`) changes → fresh sha
2. prompt changes → fresh sha
3. duration_sec changes → fresh sha
4. (audiogen only) safe_name changes when prompt[:20] sanitization differs — should this even matter? It's a display thing, not a content thing
5. (musicgen only) cue_id changes → fresh sha

Anything else I should mutate? `model_revision` and `decode_mode` aren't currently in the cache key. Should they be?

**Q5: Backward compatibility with existing cache files.** Existing `.wav` files on disk follow the format `<role>_<sha>_<ts>.wav`. The new lookup glob `<role>_<sha>_*.wav` will match them. So existing caches transparently start hitting after the deploy. Confirm this back-compat or flag if I'm missing a concern.

**Q6: What about the old `_cache_key` function signature?** Other code may import `_cache_key` from these modules. I plan to keep the old name as a thin alias to `_cache_key_for_write` for back-compat. Or should I just rename it `_cache_filename_for_write` and grep-fix any external callers (I haven't seen any)?

**Q7: Concurrent writers.** ComfyUI is sequential per workflow, but if two SFX cues in the same episode happen to have identical prompts + duration + seed, they'd resolve to the same prefix. Today (broken) they get different timestamped files (collision-resistant). With the fix, both look up the same file — second cue finds first cue's file. Is that correct behavior (yes, identical inputs should share output) or do I need to guard against same-episode collisions some other way?

**Q8: Anything else this misses?** The QA pass plan says "5 mutation tests, one per dimension {seed, model_revision, decode_mode, sample_rate, prompt}." `sample_rate` and `model_revision` and `decode_mode` aren't in the current cache key. Should they be added? Argument for: a model revision change SHOULD invalidate the cache. Argument against: it's a separate scope expansion.

## Hard constraints

- **C7 byte-identity must hold.** Same prompt + same seed + same model + same duration → byte-identical wav → byte-identical mp4.
- ASCII-only Python source, no BOM, UTF-8.
- No new dependencies. Standard library + already-imported (hashlib, time, pathlib, soundfile).
- Single commit per file change. Single bug log entry.
- Existing on-disk cache files must continue to work (no force-regen).

Please structure your reply: (1) Direct answers to Q1-Q8, (2) Any factual errors in my proposed fix, (3) Recommended final shape (code), (4) Things I should verify before merging.

---

## ChatGPT (gpt-5.5)

## (1) Direct answers to Q1-Q8

### Q1: Is this the right shape?

Yes on the separation of concerns, but I would slightly change the final shape.

I would use separate helpers, not a `mode="lookup"|"write"` argument. The bug exists because one function currently mixes two incompatible jobs:

- deterministic cache identity
- unique filename generation

So make that separation explicit.

Recommended names:

```python
_cache_prefix(...)
_cache_filename_for_write(...)
_find_cached(...)
```

However, for Rule C7, I would strongly consider making `_cache_filename_for_write()` deterministic too, e.g.:

```python
return f"{prefix}.wav"
```

not:

```python
return f"{prefix}_{ts_ms}.wav"
```

Your proposed lookup/write split fixes the cache-miss bug in `nodes/musicgen_theme.py:119` and `nodes/batch_audiogen_generator.py:79`, but if the first generation of a clean cache still writes a timestamped filename, then clean-cache runs can still produce different downstream container metadata if FFmpeg records input filenames.

So my preferred shape is:

- lookup:
  1. check deterministic canonical file: `<prefix>.wav`
  2. fallback to legacy timestamped files: `<prefix>_*.wav`
- write:
  - write deterministic canonical file: `<prefix>.wav`

That gives you:

- existing cache compatibility
- no forced regeneration
- no new timestamp in future outputs
- stronger C7 behavior

### Q2: Newest-mtime selection on multiple matches

I would not use newest mtime as the primary selector.

For C7, mtime is a weak selector because it can change when files are copied, restored from backup, touched by tooling, extracted from archives, etc. It can also differ between machines or filesystems.

Better order:

1. Prefer canonical deterministic file: `<prefix>.wav`
2. If absent, look for legacy timestamped files: `<prefix>_*.wav`
3. If multiple legacy matches exist:
   - warn
   - select deterministically by filename timestamp, not mtime

Since the legacy filenames contain millisecond timestamps, use that suffix as the ordering source. If parsing fails, fall back to filename sort.

This is more C7-friendly than mtime because the chosen file is stable as long as the directory contents are stable.

I would treat duplicate matches as a warning, not corruption. Existing broken runs may have produced many timestamped files, so failing hard would be unfriendly and would violate your “existing on-disk cache files must continue to work” constraint.

### Q3: Rule C7 compatibility

Mostly yes, with one important caveat.

Your proposed fix makes C7 hold for reruns against the same existing cache:

- first run writes `cue_abcd1234_1712345678901.wav`
- second run discovers that exact file via prefix lookup
- downstream muxing sees the same WAV path
- audio bytes and likely MP4 bytes stabilize

Current code violates that because every run calculates a brand-new filename at lookup time, misses the cache, renders again, and writes a new timestamped file.

However, if the first run after a clean cache still writes a timestamped filename, then two independent clean-cache runs can still differ at the MP4/container level if FFmpeg embeds the input WAV filename.

So I would phrase it this way:

- Your proposed fix restores cache hits and fixes C7 for same-cache reruns.
- To make C7 stronger for clean-cache runs too, stop writing timestamped filenames going forward.

Given the hard constraint:

> Same prompt + same seed + same model + same duration -> byte-identical wav -> byte-identical mp4.

I recommend deterministic write names.

### Q4: Mutation tests

Your mutation list is close, but I would remove the AudioGen `safe_name` mutation as a cache identity dimension.

`safe_name` is display/path cosmetics. It should not decide content identity. The prompt itself already feeds the digest, so if the prompt changes, the digest changes regardless of whether the first 20 sanitized characters change.

For AudioGen, test prompt changes in two ways instead:

1. prompt changes within first 20 chars
2. prompt changes after first 20 chars

That catches accidental reliance on `safe_name`.

Recommended mutation dimensions:

#### MusicGen

- `episode_seed`
- `prompt`
- `duration_sec`
- `cue_id`
- model/render params if variable: `model_revision`, `decode_mode`, `sample_rate`, etc.

#### AudioGen

- `episode_seed`
- `prompt`
- `duration_sec`
- model/render params if variable: `model_revision`, `decode_mode`, `sample_rate`, etc.

For the QA plan dimensions `{seed, model_revision, decode_mode, sample_rate, prompt}`: yes, those are valid cache-key dimensions if they can affect generated WAV bytes.

If `model_revision`, `decode_mode`, or `sample_rate` are currently variable inputs/config values, they should be in the cache identity. If they are hardcoded constants today, not including them is less urgent, but I would still leave a clear TODO or bug-log entry.

### Q5: Backward compatibility with existing cache files

Yes, your compatibility claim is correct if the digest payload remains unchanged.

Existing files:

```text
<role>_<sha>_<ts>.wav
```

will be found by:

```text
<role>_<sha>_*.wav
```

For MusicGen, this means the new prefix must remain:

```text
<cue_id>_<digest>
```

For AudioGen, this means the new prefix must remain:

```text
sfx_<safe_name>_<digest>
```

Important caveat: if you add `model_revision`, `decode_mode`, or `sample_rate` to the digest immediately, the SHA changes and legacy files will no longer match.

If you need both:

- add new dimensions
- preserve old cache files

then you need a two-level lookup:

1. lookup new/v2 prefix
2. fallback to legacy/v1 prefix

But that introduces policy questions around whether it is safe to reuse legacy files when model/sample/decode settings may have changed.

For the smallest safe Phase D fix, I would keep the existing digest payload unchanged and only fix the timestamp/cache lookup behavior. Then do a deliberate v2 cache-key migration later.

### Q6: What about the old `_cache_key` function signature?

Keep it as a thin compatibility wrapper for now.

These are underscore-prefixed helpers, so external imports are unlikely, but keeping the name costs almost nothing and avoids surprise breakage.

I would do:

```python
def _cache_key(...):
    """Backward-compatible wrapper. Returns the write filename."""
    return _cache_filename_for_write(...)
```

But I would also grep the repo. If there are no other callers, you can still keep the alias for one release/phase and remove later.

Do not keep the old timestamp semantics under `_cache_key`; that preserves the bug pattern.

### Q7: Concurrent writers / identical SFX cues

If two AudioGen cues have identical:

- prompt
- duration
- episode seed
- model/render params

then sharing the cached WAV is correct.

That is exactly what a content cache should do.

If the intended behavior is “same prompt but different random variation per cue,” then the cue identity or cue index needs to be part of the generation seed/cache key. But that is a higher-level design choice. The current AudioGen key already treats same prompt + duration + seed as the same content identity, so your fix is consistent with the current model.

On concurrency: ComfyUI sequential queue makes this acceptable. If you later introduce parallel cue generation, deterministic filenames create a possible race where two workers both miss and both try to write `<prefix>.wav`. That can be handled later with temp-file + `os.replace()` or lock files. I would not expand Phase D for that unless you already have parallel writers.

### Q8: Anything else this misses?

The big missing piece is cache identity completeness.

If these affect output bytes, they belong in the cache key:

- `model_revision`
- `decode_mode`
- `sample_rate`
- guidance/CFG settings
- temperature/top-k/top-p/sampling settings
- stereo/mono/channel layout
- normalization/postprocessing settings
- any explicit negative prompt
- any model checkpoint/path/revision value

Given your hard C7 rule, the ideal key identity is:

```text
all inputs that can affect rendered WAV bytes
```

But adding dimensions now may invalidate existing cache files unless you add a legacy fallback. So I would separate this into two decisions:

#### Phase D minimal fix

Fix the timestamped lookup bug without changing the digest payload.

#### Later cache-key v2

Add model/render/sample-rate dimensions with a versioned key and optional legacy fallback.

If the QA pass explicitly requires `{seed, model_revision, decode_mode, sample_rate, prompt}` mutations in Phase D, then you probably need to add a v2 prefix now. But that is a scope expansion beyond the timestamp bug.

My candid recommendation: do the small timestamp/cache-hit fix now, but write the code so a v2 digest can be added cleanly.

---

## (2) Factual errors or risks in the proposed fix

### 1. “mtime is more robust to clock drift”

I would not rely on that.

mtime is vulnerable to copy/restore/touch/archive behavior. For C7, deterministic filename selection is safer than mtime.

Use canonical file first, then deterministic legacy filename/timestamp ordering.

### 2. Timestamp-on-write still leaks nondeterminism

Your proposed `_cache_key_for_write()` still writes:

```python
<prefix>_<ts_ms>.wav
```

That is better than the current code because lookup no longer uses a fresh timestamp, but it is not the strongest C7 form.

If downstream MP4 metadata embeds the WAV filename, timestamped writes can still produce different MP4 bytes on clean-cache runs.

### 3. `Path.glob(f"{prefix}_*.wav")` can be unsafe if prefix contains glob metacharacters

If `cue_id` can contain characters like `[` or `*`, `Path.glob()` treats them as pattern syntax.

Maybe cue IDs are controlled and boring. I do not know. But the safer standard-library approach is to iterate directory entries and use `str.startswith()`.

### 4. AudioGen `safe_name` is not an identity dimension

It is filename decoration. Prompt is the identity input; `safe_name` is derived display text.

### 5. Adding model/sample/decode to the digest breaks legacy lookup unless handled explicitly

If you change the payload from:

```python
f"{duration_sec}|{prompt}|{episode_seed}"
```

to include more fields, old files will not be found by the new prefix.

That may be correct long-term, but it conflicts with “existing on-disk cache files must continue to work” unless you add fallback.

---

## (3) Recommended final shape

Below is the shape I would use. This keeps source ASCII-only and uses only standard library.

### MusicGen helper shape

For `nodes/musicgen_theme.py`, replacing the current timestamped `_cache_key()` around line 119:

```python
def _cache_prefix(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Deterministic cache identity prefix.

    Legacy timestamped files used:
        <cue_id>_<sha8>_<timestamp_ms>.wav

    New canonical files use:
        <cue_id>_<sha8>.wav
    """
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{cue_id}_{digest}"


def _cache_filename_for_write(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Deterministic filename for new cache writes."""
    return f"{_cache_prefix(cue_id, prompt, duration_sec, episode_seed)}.wav"


def _cache_key(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Backward-compatible wrapper. Returns the write filename."""
    return _cache_filename_for_write(cue_id, prompt, duration_sec, episode_seed)


def _find_cached(cache_dir: str, prefix: str) -> str | None:
    """Find canonical or legacy cached WAV for prefix.

    Preference:
      1. <prefix>.wav
      2. newest legacy timestamp by filename: <prefix>_<timestamp_ms>.wav
    """
    from pathlib import Path

    base = Path(cache_dir)

    canonical = base / f"{prefix}.wav"
    if canonical.is_file():
        return str(canonical)

    if not base.exists():
        return None

    legacy_prefix = prefix + "_"
    matches = []
    for path in base.iterdir():
        name = path.name
        if path.is_file() and name.startswith(legacy_prefix) and name.lower().endswith(".wav"):
            matches.append(path)

    if not matches:
        return None

    if len(matches) > 1:
        print(f"[OTR] Warning: multiple cache files for prefix {prefix}; using newest filename timestamp")

    def legacy_sort_key(path):
        name = path.name
        suffix = name[len(legacy_prefix):-4]
        try:
            return (1, int(suffix), name)
        except ValueError:
            return (0, 0, name)

    matches.sort(key=legacy_sort_key, reverse=True)
    return str(matches[0])
```

Call site:

```python
prefix = _cache_prefix(cue_id, prompt, duration, episode_seed)
cache_path = _find_cached(cache_dir, prefix)

if cache_path is not None:
    cached = _load_cached_wav(cache_path)
else:
    cache_path = os.path.join(
        cache_dir,
        _cache_filename_for_write(cue_id, prompt, duration, episode_seed),
    )
    cached = None

if cached is not None:
    # cache hit
    ...
else:
    # generate and save to cache_path
    ...
```

### AudioGen helper shape

For `nodes/batch_audiogen_generator.py`, replacing the current timestamped `_cache_key()` around line 79:

```python
def _cache_prefix(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Deterministic cache identity prefix.

    Legacy timestamped files used:
        sfx_<safe_prompt_prefix>_<sha8>_<timestamp_ms>.wav

    New canonical files use:
        sfx_<safe_prompt_prefix>_<sha8>.wav
    """
    payload = f"{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:20]).lower()
    return f"sfx_{safe_name}_{digest}"


def _cache_filename_for_write(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Deterministic filename for new cache writes."""
    return f"{_cache_prefix(prompt, duration_sec, episode_seed)}.wav"


def _cache_key(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Backward-compatible wrapper. Returns the write filename."""
    return _cache_filename_for_write(prompt, duration_sec, episode_seed)


def _find_cached(cache_dir: str, prefix: str) -> str | None:
    """Find canonical or legacy cached WAV for prefix."""
    from pathlib import Path

    base = Path(cache_dir)

    canonical = base / f"{prefix}.wav"
    if canonical.is_file():
        return str(canonical)

    if not base.exists():
        return None

    legacy_prefix = prefix + "_"
    matches = []
    for path in base.iterdir():
        name = path.name
        if path.is_file() and name.startswith(legacy_prefix) and name.lower().endswith(".wav"):
            matches.append(path)

    if not matches:
        return None

    if len(matches) > 1:
        print(f"[OTR] Warning: multiple cache files for prefix {prefix}; using newest filename timestamp")

    def legacy_sort_key(path):
        name = path.name
        suffix = name[len(legacy_prefix):-4]
        try:
            return (1, int(suffix), name)
        except ValueError:
            return (0, 0, name)

    matches.sort(key=legacy_sort_key, reverse=True)
    return str(matches[0])
```

Same call-site pattern:

```python
prefix = _cache_prefix(prompt, duration_sec, episode_seed)
cache_path = _find_cached(cache_dir, prefix)

if cache_path is not None:
    cached = _load_cached_wav(cache_path)
else:
    cache_path = os.path.join(
        cache_dir,
        _cache_filename_for_write(prompt, duration_sec, episode_seed),
    )
    cached = None
```

### Why deterministic write is my recommendation

This gives you:

```text
old file: cue_abc12345_1712345678901.wav   # still found
new file: cue_abc12345.wav                 # deterministic
```

and:

```text
old file: sfx_door_creak_abc12345_1712345678901.wav
new file: sfx_door_creak_abc12345.wav
```

That is a cleaner C7 story than continuing to write timestamps.

---

## (4) Things to verify before merging

1. **Grep for `_cache_key` imports/callers**

   Confirm no external code depends on the old timestamped filename behavior.

2. **Verify existing legacy cache hit**

   Create or use an existing file like:

   ```text
   cue_theme_abcd1234_1712345678901.wav
   ```

   Confirm `_find_cached()` returns it.

3. **Verify canonical cache preference**

   If both exist:

   ```text
   cue_theme_abcd1234.wav
   cue_theme_abcd1234_1712345678901.wav
   ```

   Confirm canonical wins.

4. **Verify duplicate legacy selection is deterministic**

   With:

   ```text
   cue_theme_abcd1234_1712345678901.wav
   cue_theme_abcd1234_1712345678902.wav
   ```

   confirm it chooses `...8902.wav` consistently and emits only a warning.

5. **Verify no fresh timestamped files are created**

   After a cache miss, new output should be:

   ```text
   <prefix>.wav
   ```

   not:

   ```text
   <prefix>_<now>.wav
   ```

6. **Verify rerun byte identity**

   Run same episode twice against same workspace/cache.

   Compare:

   - WAV bytes
   - final MP4 bytes
   - any intermediate muxed files if applicable

7. **Verify clean-cache behavior**

   If practical, run the same episode twice from clean cache into two separate output roots.

   If final MP4 bytes differ, inspect whether the difference is:

   - FFmpeg metadata
   - creation timestamp
   - input filename/path
   - nondeterministic model output
   - filesystem ordering

8. **Verify model/render parameters**

   Decide whether Phase D is only fixing timestamp cache misses or also expanding cache identity.

   If the latter, add tests for:

   - `model_revision`
   - `decode_mode`
   - `sample_rate`

   But if adding those to the digest now, implement legacy fallback deliberately.

9. **Verify AudioGen prompt edge cases**

   Test prompts where:

   - first 20 chars differ
   - first 20 chars are same but later chars differ
   - prompt sanitizes to many underscores
   - prompt is empty or very short, if allowed

10. **Verify Windows path behavior**

   Especially with ComfyUI on Windows:

   - cache directory exists before lookup
   - `Path.iterdir()` handles the directory
   - no issue with case-insensitive `.wav`/`.WAV`

Bottom line: your diagnosis is right. The timestamp in the lookup key is the core bug in `nodes/musicgen_theme.py:119` and `nodes/batch_audiogen_generator.py:79`. I would take the fix one step further and stop writing timestamped cache filenames going forward, while retaining legacy timestamp lookup for existing files.

---

## Gemini (gemini-3.1-pro-preview-customtools)

### 1. Verdict
I **STRONGLY AGREE** with ChatGPT's core recommendation to drop the timestamp entirely on new writes, as it is the only way to strictly satisfy your Rule C7 (byte-identical MP4s) for clean-cache runs.

### 2. Factual Errors in ChatGPT's Answer
ChatGPT’s answer is exceptionally solid, but there is one minor technical nuance regarding ComfyUI:
*   **Concurrency/Race Conditions:** ChatGPT states, *"If you later introduce parallel cue generation, deterministic filenames create a possible race..."* ComfyUI's core execution engine (`execution.py`) is strictly single-threaded. Nodes execute sequentially. Even if you queue 50 prompts or have a workflow with 10 AudioGen nodes, they will never evaluate concurrently. You do not need to worry about race conditions between ComfyUI nodes, only between entirely separate ComfyUI instances (which you aren't running).

### 3. Important Omissions
*   **Corrupted Cache Hits (The Interruption Problem):** By moving to deterministic filenames (`<prefix>.wav`), you introduce a new risk. If you cancel the ComfyUI queue *while* the model is saving the WAV file, you will leave a half-written, corrupted `<prefix>.wav` on disk. On the next run, `_find_cached` will see it exists, assume a cache hit, and pass garbage to downstream nodes, breaking the run permanently for that seed/prompt. **You must use atomic writes.**
*   **Model Name in Cache Key:** You mentioned `model_revision` and `decode_mode`, but didn't explicitly mention the base model name (e.g., `audiogen-medium` vs `musicgen-small`). If the model name isn't in the digest payload, switching models in the ComfyUI node will incorrectly load the cached output of the previous model. 

### 4. Short Recommendation
1.  **Adopt the deterministic write:** Change new writes to `<prefix>.wav`. This is the highest-payoff change for Rule C7, ensuring FFmpeg metadata remains identical even on a completely fresh machine/cache.
2.  **Implement atomic writes:** When generating a cache miss, save the audio to `<prefix>.tmp`, then use `os.replace(tmp_path, final_path)`. Since you already handled `os.replace` retries in Phase B, you know how Windows file-locking behaves. This prevents corrupted cache hits.
3.  **Use ChatGPT's lookup logic:** Its `iterdir()` + `startswith` approach is much safer than `Path.glob` (which chokes on brackets `[]` in prompts) and avoids the fragility of `mtime` sorting.
4.  **Defer Cache Key V2:** Do not add `model_revision`, `sample_rate`, etc., to the hash payload in Phase D. It violates your "smallest change" constraint and forces you to write complex fallback logic for legacy files. Log it as a Phase E or v2 ticket.

### 5. Uncertainties to Verify
*   **Does your `_load_cached_wav` or `save_wav` already handle atomic writes?** If `save_wav` is just a wrapper around `torchaudio.save` or `soundfile.write`, it is *not* atomic. You will need to implement the `.tmp` -> `.wav` rename at the call site in `musicgen_theme.py` (line ~125) and `batch_audiogen_generator.py` (line ~85).
*   **Are there any other callers of `_cache_key`?** You mentioned you haven't seen any, but a quick `Find in Files` for `_cache_key` across your custom nodes directory is mandatory before changing its signature or behavior.
*   **Is `episode_seed` truly deterministic?** Ensure `episode_seed` isn't accidentally being cast from a float or changing format (e.g., `12345` vs `12345.0`) upstream, as this would silently change the string payload and break the cache hash.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### (1) Direct Answers to Q1-Q8 (Synthesizing Agreement/Divergence)

**Q1: Cache Key Splitting**  
**Agree with ChatGPT/Gemini:** Splitting into `_cache_prefix`, `_cache_filename_for_write`, and `_find_cached` is the right shape. Deterministic writes (no timestamp) are critical for Rule C7. Gemini’s emphasis on atomic writes is a necessary addition.

**Q2: Multiple Matches Selection**  
**Agree with ChatGPT:** Prefer canonical file first, then legacy files ordered by filename timestamp (not mtime). Gemini’s note on ComfyUI’s single-threadedness means mtime isn’t a concurrency risk, but filename sorting is more deterministic.

**Q3: Rule C7 Compliance**  
**Agree with Both:** The fix restores cache hits and stabilizes MP4 metadata for reruns. To fully satisfy C7 (including clean-cache runs), deterministic writes (no timestamp) are required. ChatGPT’s concern about timestamped writes affecting MP4 metadata is valid.

**Q4: Mutation Tests**  
**Agree with ChatGPT:** Remove `safe_name` from AudioGen tests (it’s cosmetic). Add tests for `model_revision`, `decode_mode`, and `sample_rate` *if* they affect output. Gemini’s point about model name inclusion is crucial if models are swappable.

**Q5: Backward Compatibility**  
**Agree with Both:** Existing files (with timestamps) are compatible with the new glob-based lookup. No force-regen needed.

**Q6: Old `_cache_key` Function**  
**Agree with ChatGPT:** Keep it as a thin alias for `_cache_filename_for_write` temporarily. Gemini’s reminder to grep for callers is essential.

**Q7: Concurrent Writers**  
**Agree with Gemini:** ComfyUI’s sequential execution means identical cues sharing a cache entry is safe. No concurrency handling needed now.

**Q8: Missing Elements**  
**Agree with Gemini:** Atomic writes are critical to prevent corrupted cache hits on interruptions. ChatGPT’s push for model/render params in the cache key is valid but should be phased (see Recommendations).

---

### (2) Factual Errors or Risks in Proposed Fix

1. **ChatGPT’s Initial mtime Suggestion**  
   - **Risk:** mtime is fragile (affected by file copies, restores, etc.).  
   - **Fix:** Use filename timestamp for legacy sorting (as ChatGPT later revises).

2. **Gemini’s Model Name Omission in ChatGPT’s Answer**  
   - **Risk:** If model name isn’t in the digest, switching models could load incorrect cached files.  
   - **Fix:** Include model name in the cache key if models are configurable.

3. **Atomic Writes Not Originally Addressed**  
   - **Risk:** Interruptions during writes could leave corrupted `<prefix>.wav` files.  
   - **Fix:** Implement temporary files + `os.replace()` (Gemini’s key addition).

---

### (3) Recommended Final Shape (Code)

#### MusicGen (`nodes/musicgen_theme.py:119`)
```python
def _cache_prefix(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{cue_id}_{digest}"

def _cache_filename_for_write(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    return f"{_cache_prefix(cue_id, prompt, duration_sec, episode_seed)}.wav"

def _cache_key(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    return _cache_filename_for_write(cue_id, prompt, duration_sec, episode_seed)

def _find_cached(cache_dir: str, prefix: str) -> str | None:
    base = Path(cache_dir)
    canonical = base / f"{prefix}.wav"
    if canonical.is_file():
        return str(canonical)
    if not base.exists():
        return None
    legacy_prefix = prefix + "_"
    matches = [
        path for path in base.iterdir()
        if path.is_file() and path.name.startswith(legacy_prefix) and path.name.lower().endswith(".wav")
    ]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[OTR] Warning: multiple cache files for prefix {prefix}; using newest filename timestamp")
    def sort_key(path):
        name = path.name
        suffix = name[len(legacy_prefix):-4]
        try:
            return (1, int(suffix), name)
        except ValueError:
            return (0, 0, name)
    matches.sort(key=sort_key, reverse=True)
    return str(matches[0])
```

#### AudioGen (`nodes/batch_audiogen_generator.py:79`)
```python
def _cache_prefix(prompt: str, duration_sec: float, episode_seed: str) -> str:
    payload = f"{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:20]).lower()
    return f"sfx_{safe_name}_{digest}"

def _cache_filename_for_write(prompt: str, duration_sec: float, episode_seed: str) -> str:
    return f"{_cache_prefix(prompt, duration_sec, episode_seed)}.wav"

def _cache_key(prompt: str, duration_sec: float, episode_seed: str) -> str:
    return _cache_filename_for_write(prompt, duration_sec, episode_seed)

def _find_cached(cache_dir: str, prefix: str) -> str | None:
    # Same implementation as MusicGen's _find_cached
```

#### Atomic Write Call Site (Example)
```python
# In musicgen_theme.py/batch_audiogen_generator.py
temp_path = os.path.join(cache_dir, f".tmp_{prefix}.wav")
try:
    # Generate and save to temp_path
    save_wav(temp_path, audio_data)
    os.replace(temp_path, cache_path)  # Atomic rename
except:
    os.remove(temp_path)  # Cleanup on failure
    raise
```

---

### (4) Things to Verify Before Merging

1. **Atomic Write Implementation**  
   - Confirm `save_wav` doesn’t already handle this. If not, modify the call site.

2. **Model Name Inclusion**  
   - If models are configurable, add `model_name` to the digest payload:  
     ```python
     payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}|{model_name}".encode("utf-8")
     ```

3. **Existing Cache Hits**  
   - Test with legacy files (e.g., `cue_theme_abcd1234_1712345678901.wav`) to ensure `_find_cached` returns them.

4. **Deterministic Selection**  
   - Verify multiple legacy files are sorted correctly by filename timestamp.

5. **Clean-Cache Byte Identity**  
   - Run two clean-cache episodes and compare MP4 hashes.

6. **ComfyUI Callers**  
   - Grep for `_cache_key` to ensure no external dependencies.

7. **Windows Path Behavior**  
   - Test case-insensitive `.wav` handling and directory iteration.

---

### (5) Uncertainties

- **Performance Overhead of Atomic Writes:** Temporary files and `os.replace()` may add minor latency. Measure impact on 22s MusicGen renders.
- **Model Name Variability:** If model names aren’t standardized (e.g., `audiogen-medium` vs `audiogen_medium`), hash collisions could occur.
- **Legacy File Cleanup:** Should old timestamped files be deleted automatically? (Out of scope for Phase D.)

---

### Prioritized Recommendations

1. **Enforce Deterministic Writes**  
   Remove timestamps from new cache files to guarantee Rule C7 compliance.

2. **Implement Atomic Writes**  
   Use temporary files + `os.replace()` to prevent corrupted cache hits.

3. **Include Model Name in Cache Key (If Applicable)**  
   If models are swappable, add `model_name` to the digest payload.

4. **Verify Existing Callers of `_cache_key`**  
   Ensure no external code relies on the old timestamped behavior.

5. **Test Mutation Scenarios**  
   Focus on `model_revision`, `decode_mode`, and `sample_rate` if included in the key.

6. **Document Cache Key Policy**  
   Clarify which parameters affect identity (e.g., model name, sample rate) for future phases.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
