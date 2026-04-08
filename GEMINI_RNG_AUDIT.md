# ComfyUI-OldTimeRadio — RNG Audit for External QA
**Repo:** jbrick2070/ComfyUI-OldTimeRadio  
**File under audit:** `nodes/gemma4_orchestrator.py`  
**Issue being investigated:** Lemmy easter-egg character (11% RNG roll) appears to fire on every run despite toggle being OFF

---

## The System in Plain English

`gemma4_orchestrator.py` is a ComfyUI custom node that:
1. Fetches real science news via RSS
2. Seeds a deterministic RNG from an episode fingerprint (sha256 of title+genre+minutes+chars+temp)
3. Uses Gemma 4 LLM to write a sci-fi radio drama script
4. Has an easter-egg character "LEMMY" who should appear in ~11% of runs

---

## The Known Bug History

### Bug 1 (fixed): Lemmy frozen by deterministic seed
`random.seed(seed)` at line 1650 was called BEFORE the Lemmy roll.
For any fixed widget config (same title/genre/minutes), `random.random()` always returns the same value.
If that value happened to be `< 0.11`, Lemmy fired on 100% of runs. If `>= 0.11`, he never appeared.

**Fix applied:** Replaced module-level `random.random()` with `_LEMMY_RNG = SystemRandom()` at line 33.
`SystemRandom` uses OS entropy and is NOT affected by `random.seed()`.

### Bug 2 (still occurring?): Lemmy still firing every run despite toggle=OFF
After the SystemRandom fix was deployed, user reports Lemmy is still appearing on every run.

---

## Full RNG Map — Every `random.*` Call in the File

| Line | Call | Pre/Post seed? | Intent | Verdict |
|------|------|----------------|--------|---------|
| 26 | `from random import SystemRandom` | module load | import | — |
| 33 | `_LEMMY_RNG = SystemRandom()` | module load | OS RNG for Lemmy | ✅ correct |
| 410 | `random.Random(f"{episode_seed}_char_{idx}")` | pre-seed | isolated RNG per character | ✅ isolated instance |
| 475 | `random.Random(f"{episode_seed}_announcer")` | pre-seed | isolated RNG for announcer | ✅ isolated instance |
| 676 | `random.shuffle(shuffled_feeds)` | pre-seed | randomize news feed order | ✅ pre-seed, fine |
| 701 | `random.shuffle(pool)` | pre-seed | randomize news pool | ✅ pre-seed, fine |
| 1458 | `random.Random(f"{episode_fingerprint}_voices")` | post-seed | isolated voice pool RNG | ✅ isolated instance |
| 1650 | `random.seed(seed)` | — | **THE SEED CALL** | intentional |
| 1721 | `random.choice(_FALLBACK_SEEDS)` | post-seed | fallback news seed | ✅ deterministic ok |
| 1750 | `_LEMMY_RNG.random() < 0.11` | post-seed | **LEMMY ROLL** | ✅ should be free |
| 1855 | `random.choice("ABCDEFGHIJKL")` | post-seed | Story Arc type selection | ✅ deterministic intended |
| 2119 | `random.sample("ABCDEFGHIJKL", 3)` | post-seed | Open-Close arc selection | ✅ deterministic intended |
| 2545 | `random.choice("ABCDEFGH")` | post-seed | Chunked outline arc | ✅ deterministic intended |

---

## Relevant Code Blocks

### Block 1 — Module-level LEMMY_RNG (line 26-33)
```python
import random
from random import SystemRandom

# OS-backed RNG for the Lemmy easter-egg coin flip.
# We can't use the seeded module-level `random` because it's seeded per-episode
# from the fingerprint (for reproducible Gemma behavior), which would freeze the
# 11% roll into "always on" or "always off" for any given widget config.
# SystemRandom is unaffected by random.seed() and gives a true ~11% per run.
_LEMMY_RNG = SystemRandom()
```

### Block 2 — Deterministic seed (lines 1644-1651)
```python
seed = int(episode_fingerprint, 16) % (2**31 - 1)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
random.seed(seed)
_runtime_log(f"ScriptWriter: SEED {seed} (from fingerprint {episode_fingerprint})")
```

### Block 3 — Lemmy roll (lines 1747-1758)
```python
# Use _LEMMY_RNG (SystemRandom) instead of seeded `random` so the 11%
# is actually 11% per run, not frozen by the per-episode fingerprint seed.
_natural_roll = _LEMMY_RNG.random() < 0.11
lemmy_roll = force_lemmy or _natural_roll
if force_lemmy:
    _lemmy_source = "🔧 Lemmy was summoned by the boss (force toggle ON)"
elif _natural_roll:
    _lemmy_source = "🎲 Lemmy rolled in on his own (lucky 11%)"
else:
    _lemmy_source = "💤 Lemmy stayed in the garage tonight"
log.info(f"[Gemma4ScriptWriter] {_lemmy_source}  [force={force_lemmy}, rng_hit={_natural_roll}]")
```

### Block 4 — Widget definition (line 1374)
```python
"summon_lemmy": ("BOOLEAN", {
    "default": False,
    "tooltip": "🔧 Summon Lemmy! Drags the grizzled engineer out of the garage and into every episode. Leave OFF for the rare 11% surprise."
}),
```

### Block 5 — write_script signature (line 1540-1541)
```python
def write_script(self, ..., summon_lemmy=False):
    force_lemmy = summon_lemmy  # internal alias for clarity below
```

---

## Possible Remaining Causes (Please Investigate)

### Hypothesis A: ComfyUI module cache
ComfyUI loads Python modules ONCE at startup. If the `_LEMMY_RNG` fix was committed after the last ComfyUI restart, the old code (`random.random() < 0.11`) is still running in memory. The fix only takes effect after a full ComfyUI restart.

**Test:** Check the boot log for `[OldTimeRadio] ✓ All 9 nodes loaded successfully`. If this line appears BEFORE the git pull that included the SystemRandom fix, old code is running.

### Hypothesis B: Widget state cache in browser
ComfyUI caches widget values in the browser session. If `summon_lemmy` was set to `True` in the UI at any point, that value persists even if the JSON on disk has `false`. The widget state in the browser takes precedence over the JSON default for the current session.

**Test:** Hard-refresh the browser tab (Ctrl+Shift+R), reload the workflow JSON from disk via File → Open. Confirm `summon_lemmy` reads `false` in Node 1.

**Critical check:** The workflow JSONs previously had only 15 `widgets_values` entries (missing `summon_lemmy`). ComfyUI maps widget values positionally. If a 16th value was never saved, it uses the session-cached value — which may have been `True` from a prior test run.

### Hypothesis C: `_LEMMY_RNG` not truly free from seeding
`SystemRandom` wraps `os.urandom()` and should be immune to `random.seed()`. However, on some Python builds or platforms, there have been edge cases. 

**Test:** Add this temporary debug line immediately after the `_natural_roll` line:
```python
import os
log.info(f"[LemmyDebug] _LEMMY_RNG type={type(_LEMMY_RNG).__name__}, os.urandom(4)={os.urandom(4).hex()}")
```
If `os.urandom` returns the same 4 bytes on every run, something is wrong at the OS level (virtually impossible on Windows).

### Hypothesis D: Fingerprint collision across test runs
If the user always runs with the same episode title ("The Last Frequency"), same genre, same target_minutes — the fingerprint is identical every run. The `_natural_roll` via `_LEMMY_RNG` should still vary (it's SystemRandom), but worth confirming the log shows different `rng_hit` values across runs.

**Test:** Look at 5 consecutive boot logs. The line:
```
[Gemma4ScriptWriter] {_lemmy_source}  [force={force_lemmy}, rng_hit={_natural_roll}]
```
Should show `rng_hit=True` in ~1 out of 9 runs, not every run.

---

## What Good Logs Look Like

After a clean restart with `summon_lemmy=OFF`, expect to see:
```
# Most runs (~89%):
[Gemma4ScriptWriter] 💤 Lemmy stayed in the garage tonight  [force=False, rng_hit=False]

# Rare runs (~11%):
[Gemma4ScriptWriter] 🎲 Lemmy rolled in on his own (lucky 11%)  [force=False, rng_hit=True]

# Only when toggle is ON:
[Gemma4ScriptWriter] 🔧 Lemmy was summoned by the boss (force toggle ON)  [force=True, rng_hit=False/True]
```

If you see `force=True` with the toggle OFF, the widget value is being passed incorrectly (Hypothesis B).
If you see `force=False, rng_hit=True` every single run, SystemRandom is not working correctly (Hypothesis C).

---

## Files to Review

- `nodes/gemma4_orchestrator.py` — full orchestrator (primary file)
- `workflows/old_time_radio_scifi_full.json` — check `widgets_values` array for Node 1 (id=1), position 15 (0-indexed) should be `false`
- `workflows/old_time_radio_test.json` — same check
- `workflows/old_time_radio_scifi_lite.json` — same check

---

## Requested Actions from Reviewer

1. Confirm `_LEMMY_RNG = SystemRandom()` is truly immune to the `random.seed(seed)` call at line 1650 on Python 3.12 / Windows
2. Identify any other code path that could force `lemmy_roll = True` regardless of the toggle and RNG
3. Check if there is any way `force_lemmy` could receive `True` despite `summon_lemmy=False` being passed from ComfyUI
4. Confirm no other `random.*` call post-seed should be converted to `SystemRandom` for correctness
