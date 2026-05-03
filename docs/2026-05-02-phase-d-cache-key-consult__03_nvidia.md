# Round C -- NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5) elapsed=127.0s

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
