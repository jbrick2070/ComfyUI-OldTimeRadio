# Round A -- ChatGPT (gpt-5.5) elapsed=108.1s

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
