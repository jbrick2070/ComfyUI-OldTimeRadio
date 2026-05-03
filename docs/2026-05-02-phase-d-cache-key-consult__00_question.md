# Question -- 2026-05-02

# Phase D consult — MusicGen + AudioGen cache key fix

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
