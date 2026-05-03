# Synthesis -- 2026-05-02

**Question:** # Phase G consult — path-reorg blast-radius audit (mid-soak crash repro)

## Context

OTR ComfyUI node pack. Just landed Phases A→E (per-episode workspace at `output/otr/episodes/<ep>/audio/...`, hard-fail rename invariant, schema bump with meta.paths) and Sprint 1 (LLM OOM + parse fixes). Live soak surfaced two real bugs that A→E and Sprint 1 missed. Fact-checking the audit before shipping fixes — explicitly want to catch other latent path-reorg holes before user re-queues another 12-minute soak.

## Confirmed bug #1 — `nodes/video_engine.py:1443-1452`

`SignalLostVideoRenderer` (procgen mp4 writer) hardcodes the legacy flat output path:

```python
# Line 1443-1452 (current, broken):
out_dir = os.path.join(
    os.path.expanduser("~"), "Documents", "ComfyUI",
    "output", "otr", "audio"        # <- LEGACY flat layout
)
os.makedirs(out_dir, exist_ok=True)
ts = _time.strftime("%Y%m%d_%H%M%S")
safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in episode_title)
safe_title = safe_title.strip().replace(" ", "_").lower()[:40]
out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")
```

**What breaks downstream:** `BatchHumoRender._load_ledger_with_path` (line 920, 1861) takes the procgen mp4 path as input and does a stem swap (`.mp4 → _ledger.json`) in the SAME directory. The ledger is now at `output/otr/episodes/<ep>/audio/<ep>_ledger.json` (per Phase B). The mp4 is at `output/otr/audio/<file>.mp4`. Stem swap looks for `output/otr/audio/<file>_ledger.json` which doesn't exist. Result: hard crash mid-pipeline:

```
RuntimeError: BatchHumoRender: derived ledger from .mp4 not found:
  C:\...\output\otr\audio\signal_lost_cramped_cargo_bay_vibrating_20260502_220824_ledger.json
```

**Proposed fix:**
```python
from .production_ledger import get_ledger
ep_id = get_ledger().episode_id  # "pending_<ts>" at this point
out_dir = os.path.join(
    os.path.expanduser("~"), "Documents", "ComfyUI",
    "output", "otr", "episodes", ep_id, "audio"
)
os.makedirs(out_dir, exist_ok=True)
# ... rest unchanged
```

After write, `SignalLostVideoRenderer` triggers `Ledger.rename_episode(<new_id>)` which moves the parent dir from `episodes/pending_<ts>/` to `episodes/<new_id>/`. Since the mp4 lives INSIDE that dir, it moves along with everything else. Post-rename: mp4 at `episodes/<new_id>/audio/<filename>.mp4`, ledger at `episodes/<new_id>/audio/<new_id>_ledger.json`. BatchHumoRender's stem swap finds it.

## Confirmed bug #2 — `visual/batch_flux_render.py:641`

Radio bookend writer uses the global mtime walker:

```python
ledger_p = _OTRL.find_most_recent_ledger(
    [otr_episodes_root(), otr_legacy_audio_dir()]
)
# ...
episode_id = (led.get("episode_id") or "episode").strip()
# Then writes: output/otr/_legacy_stills/radio_bookend_<episode_id>.png
# And stamps ledger.radio_bookend_path
```

**Repro from live log:** during a `signal_lost_cramped_cargo_bay_vibrating_20260502_220824` run, BatchFluxRender stamped the radio bookend to `signal_lost_signal_abyss_20260426_161737` (a 6-day-old episode) because that ledger's mtime was apparently more recent at the moment of the call. Wrong episode entirely.

This is the EXACT same bug shape as BUG-LOCAL-014 (Phase A spacesaver wrong-episode wipe via global mtime scan). Phase A killed it for `rtx_upscale.py` only; this site was missed.

**Proposed fix:**
```python
from .production_ledger import get_ledger
_led = get_ledger()
ledger_p = Path(_led.path)  # canonical path of in-flight ledger
episode_id = _led.episode_id or "episode"
# Then read from disk for schema-l3 merge semantics:
led = _json.loads(ledger_p.read_text(encoding="utf-8"))
```

The `production_ledger._CURRENT` singleton tracks the in-flight episode by construction (set by `new_ledger()` in LLMScriptWriter, advanced by `rename_episode()` in SignalLostVideoRenderer). ComfyUI sequential queue guarantees no cross-episode race. No mtime scan needed.

## Pattern audit — other `find_most_recent_ledger` users

Same wrong-episode shape exists in 6 more sites:

| File | Line | Purpose |
|---|---|---|
| `nodes/scene_sequencer.py` | 920 | ledger path discovery for write-back |
| `nodes/scene_sequencer.py` | 1257 | same |
| `nodes/batch_bark_generator.py` | 703 | same |
| `nodes/batch_audiogen_generator.py` | 58, 511 | cache_dir + ledger write-back (Phase D added one; both should be reviewed) |
| `nodes/musicgen_theme.py` | 98, 494 | same |
| `nodes/audio_enhance.py` | 436 | ledger write-back |
| `nodes/post_audio_video_pipeline.py` | 126 | scan |

**Proposed sweep fix:** replace every `find_most_recent_ledger(...)` call in audio-side write-back paths with `Path(get_ledger().path)` from the production_ledger singleton.

For the cache_dir lookups in musicgen/audiogen specifically: those need the ledger's audio dir as the cache root. Same fix applies — use `get_ledger().out_dir` instead of walking by mtime.

## Other suspects flagged for your review

| File | Line | Concern |
|---|---|---|
| `nodes/scene_sequencer.py` | 147 | `DEFAULT_OUT = .../output/otr/audio` legacy hardcoded default — only matters if it's ever the actual write target. Likely safe but worth eyeballing. |
| `nodes/batch_humo_render.py` | 1773 | uses `otr_legacy_audio_dir()` in some path — defensive fallback or live write path? |
| `nodes/batch_ltx_render.py` | 300 | `otr_stills_dir()` with NO episode_id — same wrong-episode shape if there are multiple episodes' stills on disk |
| `nodes/batch_ltx_render.py` | 846 | `[otr_audio_dir(), otr_legacy_audio_dir()]` with NO ep_id — same |
| `nodes/video_composite.py` | 282 | `otr_legacy_audio_dir()` in scan path — defensive or active? |
| `nodes/story_orchestrator.py` | 6276 | hardcoded `output/otr/audio/` — what writes here? |

## Hard constraints

- C7: audio output bytes must remain identical run-to-run. None of the path-reorg fixes touch audio bytes; verify this assumption.
- ASCII-only Python source, no BOM, UTF-8.
- Single commit per logical fix cluster preferred.
- Existing on-disk artifacts from previous runs (in legacy locations) must continue to load.

## Questions

**Q1:** Is the singleton-based `get_ledger().path` lookup the right pattern, or is there a hidden interaction with `Ledger._merge_with_disk` (the BUG-LOCAL-108 schema-l3 merge logic) that I'm missing? Specifically: does the in-memory singleton's `.path` always match the on-disk file location after `rename_episode` advances it?

**Q2:** Phase A spacesaver derives ep_dir from `src.parent.parent` (the input mp4 path) rather than the singleton. Why prefer one over the other for different code paths? My read: spacesaver runs in a destructive context where the singleton might have advanced past the actual src episode (e.g., test scenarios), so deriving from input is safer there. Audio-side write-backs are inside the in-flight pipeline so the singleton IS the in-flight episode by definition. Confirm or push back.

**Q3:** For `batch_ltx_render.py:300` and `:846` — those call `otr_stills_dir()` and `otr_audio_dir()` with NO episode_id argument. The path helpers fall back to legacy `_legacy_stills/` / `_legacy_audio/` dirs. Is that intentional (fallback for files that didn't make it to per-episode dirs) or another wrong-episode bug?

**Q4:** Anything in the proposed fixes that would break existing on-disk artifacts (radio bookend PNGs, MusicGen/AudioGen cache wavs, Bark wavs) that are currently in legacy locations from runs before the path reorg landed?

**Q5:** Are there sibling bugs I'm missing? Search for: any node that writes a file path into the ledger, then a downstream node that reads that path expecting a specific layout. The radio_bookend_path stamping is one such pattern — what else?

**Q6:** Should `Ledger.save()` (which already stamps `meta.paths` per Phase E) ALSO sanity-check that the actual on-disk path matches `meta.paths.ledger_path` and warn / refuse-to-write if they diverge? That would catch this whole class of bug at the source.

**Q7:** Is there a risk that fixing video_engine.py to write into the per-episode dir breaks the existing `[Video] Saved` log message format that anything downstream parses?

Please structure your reply: (1) Direct answers to Q1-Q7, (2) Any factual errors in my proposed fixes, (3) Bugs I missed in the audit, (4) Recommended final fix shape, (5) What I should verify before merging.

---

## ChatGPT (gpt-5.5)

## (1) Direct answers to Q1-Q7

### Q1 — Is `get_ledger().path` the right pattern? Any `_merge_with_disk` trap?

Yes, for **in-flight write-back nodes** inside the normal ComfyUI episode pipeline, `Path(get_ledger().path)` is the right replacement for `find_most_recent_ledger(...)`.

The important distinction is:

- `find_most_recent_ledger(...)` answers: “What ledger on disk has the newest mtime?”
- `get_ledger().path` answers: “What ledger belongs to the currently executing episode?”

For write-backs, you want the second answer.

The `_merge_with_disk` logic should not be a blocker as long as the save path is the singleton ledger’s current path. The merge should merge the current in-memory ledger with the JSON at `self.path`. That is exactly the path you want.

But verify one thing before trusting this globally:

> After `Ledger.rename_episode(<new_id>)`, the singleton must update all three in-memory values:
>
> - `ledger.episode_id`
> - `ledger.out_dir`
> - `ledger.path`

If `rename_episode()` physically moves the directory but leaves `_CURRENT.path` pointing at `episodes/pending_<ts>/audio/pending_<ts>_ledger.json`, then every proposed singleton fix becomes unsafe after rename.

So Q1 answer is: **yes, correct pattern, assuming `rename_episode()` advances `_CURRENT.path` to the final on-disk ledger path. Verify that directly in `production_ledger.py`.**

I would add a tiny regression assertion/test:

```python
led = get_ledger()
old = Path(led.path)
led.rename_episode(new_id)
assert led.episode_id == new_id
assert Path(led.path).exists()
assert new_id in str(led.path)
assert not Path(old).exists()
```

If that fails, fix `rename_episode()` first.

---

### Q2 — Why singleton for audio-side write-backs but derive from `src.parent.parent` for spacesaver?

Your read is right.

For destructive or semi-destructive cleanup code, deriving from the explicit input artifact is safer than using global process state.

Spacesaver gets an input mp4 path. Its safest source of truth is therefore:

```python
ep_dir = src.parent.parent
```

That makes spacesaver operate on the episode that owns the mp4 it was handed, even if:

- the singleton has advanced,
- someone manually re-runs the node,
- a test invokes the node outside the normal graph order,
- an old artifact is being processed.

For active audio-side write-backs like `batch_flux_render.py:641`, `scene_sequencer.py:920`, `batch_bark_generator.py:703`, etc., the node is part of the in-flight generation chain and is mutating the currently active ledger. In that context, the singleton is the right source.

General rule I would use:

1. **If the node receives an explicit artifact path** and is modifying/deleting/deriving beside that artifact, derive episode identity from that path.
2. **If the node is doing an in-flight ledger write-back** and has no explicit ledger/artifact path, use `get_ledger().path`.
3. **If the node can be run standalone**, accept an explicit ledger path if possible; fall back to legacy discovery only for read-only compatibility.

---

### Q3 — `batch_ltx_render.py:300` and `:846`: intentional fallback or wrong-episode bug?

Based on the pattern you described, I would treat both as suspicious until proven otherwise.

- `batch_ltx_render.py:300` uses `otr_stills_dir()` with no `episode_id`.
- `batch_ltx_render.py:846` uses `[otr_audio_dir(), otr_legacy_audio_dir()]` with no `episode_id`.

If those helpers fall back to shared legacy dirs such as:

```text
output/otr/_legacy_stills/
output/otr/_legacy_audio/
```

then they are acceptable only for **read-only legacy fallback**.

They are not safe for active per-episode writes or episode selection.

For new runs, active LTX rendering should resolve paths from one of:

```python
get_ledger().out_dir
Path(get_ledger().path).parent
episode_id = get_ledger().episode_id
otr_stills_dir(episode_id)
otr_audio_dir(episode_id)
```

If `batch_ltx_render.py:300` writes new stills into `_legacy_stills`, that is another path-reorg bug.

If `batch_ltx_render.py:846` scans global audio dirs to locate the current episode audio/ledger, that is the same wrong-episode class as BUG-LOCAL-014 and the radio bookend bug.

So Q3 answer: **legacy fallback is okay for loading old artifacts; using no-episode helpers in active generation is likely another wrong-episode bug.**

---

### Q4 — Could proposed fixes break existing legacy artifacts?

They should not, if you keep this rule:

> New writes go to the current episode dir. Old reads may still fall back to legacy locations.

Specific risks:

#### Radio bookend PNGs

If older ledgers contain:

```json
"radio_bookend_path": "output/otr/_legacy_stills/radio_bookend_....png"
```

then loaders should continue to honor that path if it exists.

The fix should only change where newly generated radio bookends are written/stamped. It should not remove support for reading already-stamped legacy paths.

#### MusicGen / AudioGen cache wavs

This is more sensitive.

Changing cache root from global mtime-selected dirs to `get_ledger().out_dir` is correct for new runs, but it can cause cache misses if older cache wavs only exist in legacy dirs and the code expects to rediscover them by scanning a cache directory.

To preserve compatibility:

1. Prefer explicit ledger-stamped paths if present and existing.
2. Then check current per-episode cache dir.
3. Then check legacy cache dirs read-only.
4. Never write new cache files into the legacy dirs.

One C7-related caution: moving the cache root can change whether a node reuses a cached wav or regenerates it. If MusicGen/AudioGen generation is not fully deterministic, that can affect audio bytes. The path fix itself does not modify audio bytes, but a cache-hit/cache-miss change can. So verify final wav hashes after this sweep.

#### Bark wavs

Same principle. If ledger has absolute/relative Bark wav paths from older runs, honor them. New Bark writes should use the current episode audio dir.

So Q4 answer: **the proposed direction is compatible, but only if fallback is retained for reads and explicit ledger paths remain authoritative.**

---

### Q5 — Sibling bugs to look for

Yes. The class is broader than `find_most_recent_ledger(...)`.

Search for any of these patterns:

```text
find_most_recent_ledger
otr_audio_dir()
otr_stills_dir()
otr_legacy_audio_dir()
otr_legacy_stills_dir()
output/otr/audio
output\\otr\\audio
_legacy_audio
_legacy_stills
glob("*ledger*.json")
rglob("*ledger*.json")
mtime
getmtime
stat().st_mtime
replace(".mp4", "_ledger.json")
with_suffix(...)
stem + "_ledger.json"
```

Path-reorg bugs usually fall into one of these buckets:

### A. Active write to legacy/global dir

Confirmed example:

- `nodes/video_engine.py:1443-1452`

Likely suspects:

- `nodes/story_orchestrator.py:6276` hardcoded `output/otr/audio/`
- `nodes/scene_sequencer.py:147` `DEFAULT_OUT = .../output/otr/audio`, if that default is ever used for live writes
- `nodes/batch_ltx_render.py:300` if `otr_stills_dir()` without `episode_id` is used for writes

### B. Global mtime scan for active write-back

Confirmed example:

- `visual/batch_flux_render.py:641`

Other listed sites should be swept:

- `nodes/scene_sequencer.py:920`
- `nodes/scene_sequencer.py:1257`
- `nodes/batch_bark_generator.py:703`
- `nodes/batch_audiogen_generator.py:58`
- `nodes/batch_audiogen_generator.py:511`
- `nodes/musicgen_theme.py:98`
- `nodes/musicgen_theme.py:494`
- `nodes/audio_enhance.py:436`
- `nodes/post_audio_video_pipeline.py:126`

For normal in-flight execution, these should not use global mtime discovery.

### C. Stem-swap assumptions

Your confirmed crash is here:

- `nodes/batch_humo_render.py:920`
- `nodes/batch_humo_render.py:1861`

Stem swap from:

```text
foo.mp4
```

to:

```text
foo_ledger.json
```

only works if the mp4 stem intentionally matches the final episode id.

This is okay if the procgen mp4 is named:

```text
<episode_id>.mp4
```

or:

```text
signal_lost_<title>_<timestamp>.mp4
```

where that stem exactly equals the final `episode_id`.

But if video filename and final episode id can diverge, stem-swap will keep failing even after moving the mp4 into the episode dir.

Safer BatchHumo behavior would be:

1. If input mp4 is inside `episodes/<ep>/audio/`, look for `episodes/<ep>/audio/<ep>_ledger.json`.
2. Else try legacy stem swap for old artifacts.
3. Else accept explicit ledger path if available.

### D. Return-path-after-rename bugs

This is the biggest hole in the proposed `video_engine.py` fix.

If `SignalLostVideoRenderer` writes the mp4 to:

```text
episodes/pending_<ts>/audio/<file>.mp4
```

then calls:

```python
Ledger.rename_episode(<new_id>)
```

the file moves to:

```text
episodes/<new_id>/audio/<file>.mp4
```

If the node returns the original `out_path` string computed before rename, downstream nodes will receive a stale path into the deleted/moved `pending_<ts>` directory.

Before shipping the `video_engine.py:1443-1452` fix, verify the node recomputes and returns/logs the final path after rename.

The fix should look conceptually like:

```python
pending_out_path = Path(out_dir) / filename

# write pending_out_path

new_id = ...
ledger.rename_episode(new_id)

final_out_path = Path(ledger.out_dir) / pending_out_path.name
assert final_out_path.exists()

return str(final_out_path)
```

or `rename_episode()` should return a path mapping.

Do not return the stale pending path.

### E. Ledger field path stamping

Search for ledger fields that are stamped with file paths. Examples to audit:

- `radio_bookend_path`
- Bark voice wav paths
- MusicGen theme/music paths
- AudioGen ambience/effect paths
- enhanced audio path
- final mix path
- scene manifest path
- still image paths
- LTX/Hunyuan/Humo output paths
- composite mp4 path
- subtitle/transcript/caption paths, if any

For each field, verify:

1. New writes go under `episodes/<episode_id>/...`.
2. Existing legacy paths are still loaded if present.
3. No code derives another episode’s path by mtime.

---

### Q6 — Should `Ledger.save()` sanity-check `meta.paths.ledger_path`?

Yes, but I would be careful about how hard it fails.

`Ledger.save()` should treat `self.path` as canonical and ensure `meta.paths.ledger_path` matches it before writing.

Recommended behavior:

```python
actual = Path(self.path).resolve()
meta_path = Path(meta.paths.ledger_path).resolve() if present else None

if meta_path and meta_path != actual:
    # For new schema/current run: warn or hard-fail depending on strictness.
    # Then rewrite meta.paths.ledger_path = actual.
```

Given your project already has hard-fail rename invariants, I would enforce strongly for new in-flight ledgers, but keep compatibility for old ledgers.

Suggested policy:

- For schema-l3/current ledgers: hard-fail if save target is outside the expected episode dir.
- For legacy ledgers: normalize/warn, do not refuse to load.
- Always stamp `meta.paths.ledger_path` from `self.path` immediately before save.

Also check these invariants in `save()`:

```text
Path(self.path).name == f"{self.episode_id}_ledger.json"
Path(self.path).parent == Path(self.out_dir)
Path(self.out_dir).name == "audio"
Path(self.out_dir).parent.name == self.episode_id
```

Maybe allow exceptions for legacy mode, but for new runs those invariants are valuable.

Q6 answer: **yes, add the check. It will catch this class early, but avoid breaking legacy loads.**

---

### Q7 — Could changing `video_engine.py` path break `[Video] Saved` log parsers?

Unlikely if the log format stays the same and only the path value changes.

For example, keep:

```text
[Video] Saved: <path>
```

Do not change it to:

```text
[Video] Procgen output finalized at <path>
```

unless you have searched for downstream log parsing.

However, the bigger risk is not the log line. The bigger risk is the node return value.

After moving the mp4 into the pending episode dir and then renaming the episode dir, make sure the returned path is the final path, not the stale pending path.

So Q7 answer: **path change should be fine; preserve log format; verify return value and downstream value passing.**

---

## (2) Factual issues / risks in the proposed fixes

### Issue 1 — `video_engine.py` proposed fix is incomplete unless return path is updated after rename

You proposed:

```python
ep_id = get_ledger().episode_id  # "pending_<ts>" at this point
out_dir = ... episodes/ep_id/audio
```

That part is right.

But if `SignalLostVideoRenderer` later calls `Ledger.rename_episode(<new_id>)`, then the original `out_path` no longer exists after the directory move.

You need to recompute the final mp4 path after rename.

Minimum safe pattern:

```python
from .production_ledger import get_ledger

ledger = get_ledger()
out_dir = Path(ledger.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

pending_out_path = out_dir / f"signal_lost_{safe_title}_{ts}.mp4"

# write pending_out_path

# rename episode
ledger.rename_episode(new_id)

final_out_path = Path(ledger.out_dir) / pending_out_path.name
if not final_out_path.exists():
    raise RuntimeError(f"VideoRenderer: final mp4 missing after episode rename: {final_out_path}")

out_path = str(final_out_path)
```

If the node logs `[Video] Saved` before rename, either log again after rename or log only the final path.

---

### Issue 2 — BatchHumo stem swap only works if mp4 stem equals final episode id

You wrote:

> Post-rename: mp4 at `episodes/<new_id>/audio/<filename>.mp4`, ledger at `episodes/<new_id>/audio/<new_id>_ledger.json`. BatchHumoRender's stem swap finds it.

That is true only if:

```text
<filename> == <new_id>
```

In your live example, the mp4 filename stem appears to equal the episode id:

```text
signal_lost_cramped_cargo_bay_vibrating_20260502_220824.mp4
signal_lost_cramped_cargo_bay_vibrating_20260502_220824_ledger.json
```

So the current naming may make it true.

But do not rely on “same directory” alone. Stem swap still requires matching stems.

I would either enforce the procgen mp4 stem to equal `ledger.episode_id`, or change BatchHumo ledger discovery to derive from the episode directory first.

Safer BatchHumo lookup:

```python
mp4 = Path(input_mp4)

# New layout
if mp4.parent.name == "audio" and mp4.parent.parent.parent.name == "episodes":
    ep_id = mp4.parent.parent.name
    candidate = mp4.parent / f"{ep_id}_ledger.json"
    if candidate.exists():
        return candidate

# Legacy fallback
candidate = mp4.with_name(mp4.stem + "_ledger.json")
if candidate.exists():
    return candidate
```

That would decouple ledger lookup from mp4 filename.

---

### Issue 3 — Relative imports may differ between `nodes/` and `visual/`

You proposed in `visual/batch_flux_render.py`:

```python
from .production_ledger import get_ledger
```

That may or may not be correct depending on package layout.

If `production_ledger.py` is under `nodes/`, and `visual/` is a sibling package, then `.production_ledger` from `visual` will fail. The existing `_OTRL` alias may already be importing the ledger module through the correct package path.

So the concept is right, but verify the import path in `visual/batch_flux_render.py`.

Possible forms might be:

```python
from ..nodes.production_ledger import get_ledger
```

or:

```python
from nodes.production_ledger import get_ledger
```

or continuing to use `_OTRL.get_ledger()` if `_OTRL` is already the correct module.

Do not introduce a second module instance via a different import path. That would create two singletons, which would be disastrous.

This is important:

> All code must import `production_ledger` through the same module identity, or `_CURRENT` may split.

Before merging, print/check:

```python
import sys
[m for m in sys.modules if m.endswith("production_ledger")]
```

You should not see duplicate logical imports of the same file under different names.

---

### Issue 4 — Cache-dir changes can indirectly affect C7

You said:

> None of the path-reorg fixes touch audio bytes.

Mostly true, but cache root changes can alter cache hit/miss behavior.

If a node previously hit a cache wav and now regenerates, the output bytes may change unless generation is perfectly deterministic.

So the code changes do not intentionally alter audio, but C7 verification still needs final audio hash checks.

---

## (3) Bugs I think you may have missed or should promote in priority

### 1. Stale pending path after `video_engine.py` rename

As above, this is the highest-priority sibling bug because your proposed video fix creates it if the return path is not recomputed.

Relevant location:

- `nodes/video_engine.py:1443-1452`
- nearby call to `Ledger.rename_episode(<new_id>)`

Audit both the log and node return value.

---

### 2. BatchHumo should not rely only on mp4 stem swap

Relevant locations:

- `nodes/batch_humo_render.py:920`
- `nodes/batch_humo_render.py:1861`

Even after moving procgen mp4 into the episode dir, a stem swap is brittle. The new layout gives you a better invariant:

```text
episodes/<ep>/audio/<ep>_ledger.json
```

Use that first. Keep stem-swap as legacy fallback.

---

### 3. `post_audio_video_pipeline.py:126` global ledger scan

This one deserves priority because post-pipeline nodes often run late, after many files have been touched. Mtime skew is especially likely there.

If `post_audio_video_pipeline.py:126` selects the ledger to update/process via scan, it has the exact same bug shape as the radio bookend issue.

Use singleton or explicit upstream path for active runs. Keep scan only for standalone/read-only legacy mode.

---

### 4. `batch_ltx_render.py:300` and `:846`

These are likely real bugs if active rendering uses them.

- `batch_ltx_render.py:300` with `otr_stills_dir()` no episode id can write/read the wrong stills.
- `batch_ltx_render.py:846` with `[otr_audio_dir(), otr_legacy_audio_dir()]` no episode id can select wrong audio/ledger.

New LTX outputs should be episode-scoped.

---

### 5. `story_orchestrator.py:6276` hardcoded `output/otr/audio/`

This is a red flag.

I cannot say whether it is live without seeing the surrounding code, but hardcoded `output/otr/audio/` after the path reorg should be treated as guilty until proven read-only legacy compatibility.

If it writes, fix it.

If it reads, make it explicit legacy fallback.

---

### 6. `scene_sequencer.py:147` `DEFAULT_OUT`

If that value is merely a UI default that gets overridden by ledger state, it is low risk.

If any execution path uses `DEFAULT_OUT` as the actual output directory, it is a live bug.

Given this is a ComfyUI node pack, defaults often become real widget values. I would not assume it is safe without tracing it.

---

### 7. Duplicate singleton import risk

If some modules import `production_ledger` as:

```python
import production_ledger
```

and others as:

```python
from .production_ledger import get_ledger
```

or under different package roots, Python may load the same file twice with two `_CURRENT` values.

That would make singleton fixes appear flaky.

This is worth a one-time grep and runtime check.

---

## (4) Recommended final fix shape

I would do this in four small commits or one commit with four clear sections.

### Fix cluster 1 — `video_engine.py` procgen path and post-rename final path

At `nodes/video_engine.py:1443-1452`, replace hardcoded legacy dir with the current ledger audio dir.

Prefer:

```python
ledger = get_ledger()
out_dir = Path(ledger.out_dir)
```

over reconstructing from `~Documents/ComfyUI/...`, because `ledger.out_dir` is already the canonical Phase B/E location.

Then after `rename_episode()`:

```python
final_out_path = Path(ledger.out_dir) / pending_out_path.name
```

Verify it exists and return/log that final path.

Also ensure the mp4 filename stem either equals final `episode_id`, or update BatchHumo to not require it.

---

### Fix cluster 2 — Remove global mtime scans from active write-back paths

Replace active `find_most_recent_ledger(...)` calls at:

- `visual/batch_flux_render.py:641`
- `nodes/scene_sequencer.py:920`
- `nodes/scene_sequencer.py:1257`
- `nodes/batch_bark_generator.py:703`
- `nodes/batch_audiogen_generator.py:58`
- `nodes/batch_audiogen_generator.py:511`
- `nodes/musicgen_theme.py:98`
- `nodes/musicgen_theme.py:494`
- `nodes/audio_enhance.py:436`
- `nodes/post_audio_video_pipeline.py:126`

with a common helper, not copy/paste.

For example, in `production_ledger.py` or path utilities:

```python
def current_ledger_path() -> Path:
    led = get_ledger()
    p = Path(led.path)
    if not p.exists():
        raise RuntimeError(f"Current ledger path does not exist: {p}")
    return p
```

And:

```python
def current_audio_dir() -> Path:
    led = get_ledger()
    p = Path(led.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

This reduces future drift.

Do not use `find_most_recent_ledger` for write-back.

Keep `find_most_recent_ledger` only for explicit legacy/standalone commands where the user did not supply a ledger.

---

### Fix cluster 3 — Make BatchHumo ledger derivation layout-aware

At:

- `nodes/batch_humo_render.py:920`
- `nodes/batch_humo_render.py:1861`

Use new-layout lookup first:

```python
mp4 = Path(mp4_path)

# New layout:
# output/otr/episodes/<ep>/audio/<something>.mp4
if mp4.parent.name == "audio":
    ep_dir = mp4.parent.parent
    if ep_dir.parent.name == "episodes":
        ep_id = ep_dir.name
        ledger_p = mp4.parent / f"{ep_id}_ledger.json"
        if ledger_p.exists():
            return ledger_p

# Legacy:
legacy_p = mp4.with_name(mp4.stem + "_ledger.json")
if legacy_p.exists():
    return legacy_p
```

This preserves old artifacts and makes new layout robust.

---

### Fix cluster 4 — New writes require episode id/current ledger; legacy dirs read-only

Audit these:

- `nodes/batch_ltx_render.py:300`
- `nodes/batch_ltx_render.py:846`
- `nodes/video_composite.py:282`
- `nodes/story_orchestrator.py:6276`
- `nodes/scene_sequencer.py:147`
- `nodes/batch_humo_render.py:1773`

Policy:

- If active write: use current ledger/episode id.
- If active read for current run: use current ledger/episode id first.
- If compatibility read: fallback to legacy.
- If destructive operation: derive from explicit input artifact path.

---

### Recommended helper semantics

I would centralize this to avoid future reintroductions:

```python
def require_current_ledger() -> Ledger:
    led = get_ledger()
    if not led or not getattr(led, "path", None):
        raise RuntimeError("No active production ledger for in-flight write-back")
    return led


def require_current_ledger_path() -> Path:
    led = require_current_ledger()
    p = Path(led.path)
    if not p.exists():
        raise RuntimeError(f"Active ledger path missing on disk: {p}")
    return p


def require_current_audio_dir() -> Path:
    led = require_current_ledger()
    p = Path(led.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

Then code reads clearly:

```python
ledger_p = require_current_ledger_path()
audio_dir = require_current_audio_dir()
```

---

## (5) What to verify before merging

### A. Static grep

Run searches for:

```text
find_most_recent_ledger
output/otr/audio
output\\otr\\audio
otr_audio_dir()
otr_stills_dir()
otr_legacy_audio_dir
otr_legacy_stills_dir
_legacy_audio
_legacy_stills
st_mtime
getmtime
rglob("*ledger")
glob("*ledger")
_ledger.json
```

For every hit, classify it:

- active write,
- active read,
- standalone read,
- legacy fallback,
- destructive cleanup.

Only legacy/standalone read paths should use global scans or legacy dirs.

---

### B. Singleton import identity

Verify `production_ledger.py` is not imported twice under different module names.

During a run, log or inspect:

```python
import sys
mods = [k for k, v in sys.modules.items() if k.endswith("production_ledger")]
print(mods)
```

You want one logical module.

---

### C. Rename correctness

Run a minimal episode and assert:

```text
No returned path contains /episodes/pending_
No ledger meta path contains /episodes/pending_
No stamped artifact path contains /episodes/pending_
Final mp4 exists under /episodes/<new_id>/audio/
Final ledger exists under /episodes/<new_id>/audio/<new_id>_ledger.json
```

Especially verify `SignalLostVideoRenderer` returns the final mp4 path after rename.

---

### D. Wrong-mtime repro

Create or touch an old ledger so it has a newer mtime than the active episode.

Then run the pipeline section that previously failed.

Expected:

- `visual/batch_flux_render.py:641` writes/stamps radio bookend into the active episode ledger.
- No active write-back touches the old episode.
- No spacesaver/upscaler deletes or modifies the old episode.

This directly validates the class of bug.

---

### E. Legacy artifact compatibility

Test opening/processing an old run where artifacts live in:

```text
output/otr/audio/
output/otr/_legacy_audio/
output/otr/_legacy_stills/
```

Expected:

- Existing ledger-stamped paths load.
- Legacy mp4-to-ledger stem swap still works.
- No new outputs are written to legacy dirs unless explicitly running a legacy migration/tool mode.

---

### F. C7 audio hash check

Do not assume.

Run the same deterministic prompt/config twice after the path fixes and hash:

- Bark wavs
- MusicGen/AudioGen wavs
- enhanced wav
- final mix wav

Expected:

```text
hash run 1 == hash run 2
```

Pay special attention to MusicGen/AudioGen cache behavior. A path change can cause cache misses, and cache misses can expose nondeterminism.

---

### G. Ledger invariant check

For a fresh run, inspect final ledger JSON:

```json
"episode_id": "<ep>",
"meta": {
  "paths": {
    "ledger_path": ".../output/otr/episodes/<ep>/audio/<ep>_ledger.json",
    "audio_dir": ".../output/otr/episodes/<ep>/audio",
    ...
  }
}
```

Verify every stamped path that belongs to the current run is under:

```text
output/otr/episodes/<ep>/
```

except explicitly intentional shared assets.

---

### H. Log format

Search for parsing of:

```text
[Video] Saved
```

If none, just preserve the prefix and format.

If there is parsing, make sure it receives the final post-rename path.

---

## Bottom line

Your audit direction is correct. The two confirmed bugs are real path-reorg misses.

The main correction I would make is this:

> Moving `SignalLostVideoRenderer` output into `episodes/pending_<ts>/audio` is necessary but not sufficient. After `rename_episode()`, recompute and return the final mp4 path, and ensure BatchHumo does not rely solely on mp4 stem swap unless that stem is guaranteed to equal `episode_id`.

For the rest: replace active mtime ledger discovery with the production ledger singleton, preserve legacy fallback only for reads, and add `Ledger.save()` invariants so this class fails early next time.

---

## Gemini (gemini-3.1-pro-preview-customtools)

### 1. Verdict

**PARTIALLY AGREE.** ChatGPT’s pathing logic and directory restructuring are correct, but replacing the global mtime scan with a global Python singleton (`_CURRENT`) in ComfyUI is a ticking time bomb due to ComfyUI's node execution cache.

### 2. Factual Errors in the ChatGPT Answer

*   **Fatal misunderstanding of ComfyUI execution (The Singleton Trap):** ChatGPT states, *"ComfyUI sequential queue guarantees no cross-episode race."* This ignores **ComfyUI's node caching**. If the node that initializes the singleton (e.g., `LLMScriptWriter`) has unchanged inputs, ComfyUI will cache it and *not execute it*. The `_CURRENT` singleton will either be uninitialized (crashing the run) or stale from the previous run (silently overwriting the wrong episode). The mtime scan you are trying to replace was likely a dirty hack originally written to bypass this exact caching issue.
*   **Ignored VRAM constraints on cache misses:** ChatGPT correctly notes that changing the AudioGen/MusicGen cache directory will cause cache misses, but misses the hardware implication. On your RTX 5080 (14.5 GB VRAM ceiling), an unexpected cache miss mid-pipeline will force ComfyUI to load the audio models into VRAM. If a heavy visual model is already loaded, this unexpected model load will instantly trigger an OOM crash.
*   **Stem swap assumption:** ChatGPT assumes the mp4 stem and `episode_id` match in your live log. However, looking at `nodes/video_engine.py:1449`, you have `safe_title = ...[:40]`. Because the title is truncated to 40 characters, the resulting mp4 stem will almost certainly *not* match the exact `episode_id` if the title is long. Stem swapping is mathematically broken here, not just brittle.

### 3. Important Things That Were Omitted

*   **The DAG is the only safe state:** In ComfyUI, the only safe way to pass state (like the current episode ID or ledger path) is through the graph (node outputs connected to node inputs). Global state in ComfyUI is an anti-pattern that always ends in race conditions or cache-invalidation bugs.
*   **C7 (Byte-Identical Audio) Guarantee:** If Bark, MusicGen, or AudioGen have *any* non-determinism (e.g., floating-point drift on different batch sizes, unseeded samplers), a cache miss permanently violates Rule C7. To guarantee C7, you *must* hit the legacy cache.

### 4. Recommended Final Fix Shape (Smallest Change, Largest Payoff)

1.  **Fix `video_engine.py` Post-Rename Return:** As ChatGPT noted, you must recompute and return the final path after `rename_episode`. Returning the `pending_<ts>` path will break downstream nodes.
2.  **Bust the ComfyUI Cache for the Singleton (The High-Payoff Hack):** If you don't want to rewire all your ComfyUI node ports to pass `ledger_path` explicitly (which is the "correct" but massive architectural fix), you must force the node that creates `new_ledger()` to execute every single time. Add a `classmethod` to that node:
    ```python
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN") # Forces ComfyUI to never cache this node
    ```
    This ensures `_CURRENT` is actually updated per queue item.
3.  **Layout-Aware Ledger Discovery (No Stem Swaps):** Update `BatchHumoRender` and others to derive the ledger from the directory structure, completely ignoring the mp4 filename.
    ```python
    # If mp4 is at episodes/<ep_id>/audio/<anything>.mp4
    ep_dir = mp4_path.parent.parent
    ledger_path = mp4_path.parent / f"{ep_dir.name}_ledger.json"
    ```
4.  **Strict Cache Fallback (Protect VRAM & C7):** For MusicGen/AudioGen (`batch_audiogen_generator.py`, `musicgen_theme.py`), the cache lookup *must* check `_legacy_audio` first before attempting to generate.
    ```python
    if legacy_cache_path.exists():
        return legacy_cache_path # Preserves C7, prevents VRAM spike
    # Else generate and save to new episodes/<ep>/audio/ dir
    ```

### 5. Uncertainties to Verify Before Merging

*   **Is `LLMScriptWriter` (or whatever creates the ledger) currently being cached by ComfyUI?** Test this by queuing two identical prompts in a row. If the second run finishes instantly without hitting the LLM, your singleton is stale.
*   **Are your audio generators 100% deterministic?** If you delete the cache and re-run an old prompt, do the output bytes match exactly? If not, Rule C7 relies entirely on the cache surviving this path reorg.
*   **Does `rename_episode` update `_CURRENT.path`?** Verify that the singleton's internal path string is actually updated when the directory is moved, otherwise `get_ledger().path` points to a ghost directory.

---

## NVIDIA (mistralai/mistral-nemotron)

### 1. Agreement / Disagreement

**Agreement:**
- **`video_engine.py` Post-Rename Path Fix:** Both agree that recomputing and returning the final path after `rename_episode` is necessary. This is correct.
- **Busting ComfyUI Cache for Singleton:** Gemini’s `IS_CHANGED` method is a critical addition to ensure the singleton is updated per queue item. ChatGPT missed this entirely.
- **Layout-Aware Ledger Discovery:** Both agree on moving away from stem swaps, but Gemini’s directory-based approach is more robust.
- **Strict Cache Fallback:** Both emphasize preserving legacy cache paths for VRAM and C7 compliance.

**Disagreement:**
- **Singleton Use in ComfyUI:** ChatGPT advocates for using a singleton (`_CURRENT`), while Gemini warns against it due to ComfyUI’s node caching. Gemini is correct here—global state in ComfyUI is dangerous.
- **Path Reorg Impact on C7:** ChatGPT assumes path changes won’t affect audio bytes, but Gemini rightly points out that cache misses can violate C7 if audio generation isn’t deterministic.

**Third Path:**
- **Hybrid Approach:** Use the singleton (`_CURRENT`) but enforce cache invalidation (via `IS_CHANGED`) to ensure it’s always fresh. This balances ChatGPT’s simplicity with Gemini’s safety concerns.

### 2. Factual Errors

**ChatGPT:**
- Assumes ComfyUI’s sequential queue prevents cross-episode races. **False:** ComfyUI’s node caching can stale the singleton.
- Overlooks VRAM implications of cache misses in AudioGen/MusicGen.
- Assumes mp4 stem matches `episode_id`, which is broken due to title truncation (`safe_title = ...[:40]`).

**Gemini:**
- No factual errors, but could clarify that `IS_CHANGED` is a ComfyUI-specific method to force node re-execution.

### 3. Important Omissions

**Both Missed:**
- **Ledger Path Validation:** Neither suggests adding a sanity check in `Ledger.save()` to ensure `meta.paths.ledger_path` matches `self.path`. This would catch path-reorg bugs early.
- **Legacy Artifact Handling:** Neither fully addresses how to handle existing artifacts (e.g., radio bookend PNGs) in legacy locations. A migration script might be needed.
- **Downstream Node Parsing:** Neither verifies if any downstream nodes parse `[Video] Saved` log messages. This could break if the path format changes.

### 4. Prioritized Recommendations

1. **Fix `video_engine.py` Post-Rename Path:**
   - Recompute and return the final path after `rename_episode` to avoid stale `pending_<ts>` paths.
   - Example:
     ```python
     final_out_path = Path(ledger.out_dir) / pending_out_path.name
     assert final_out_path.exists()
     return str(final_out_path)
     ```

2. **Bust ComfyUI Cache for Singleton:**
   - Add `IS_CHANGED` to the node that creates `new_ledger()` to ensure the singleton is always fresh.
   - Example:
     ```python
     @classmethod
     def IS_CHANGED(cls, **kwargs):
         return float("NaN")  # Forces ComfyUI to never cache this node
     ```

3. **Layout-Aware Ledger Discovery:**
   - Replace stem swaps with directory-based ledger discovery in `BatchHumoRender` and others.
   - Example:
     ```python
     ep_dir = mp4_path.parent.parent
     ledger_path = mp4_path.parent / f"{ep_dir.name}_ledger.json"
     ```

4. **Strict Cache Fallback:**
   - Ensure MusicGen/AudioGen check `_legacy_audio` first to preserve C7 and prevent VRAM spikes.
   - Example:
     ```python
     if legacy_cache_path.exists():
         return legacy_cache_path  # Preserves C7, prevents VRAM spike
     # Else generate and save to new episodes/<ep>/audio/ dir
     ```

5. **Ledger Path Validation:**
   - Add a sanity check in `Ledger.save()` to ensure `meta.paths.ledger_path` matches `self.path`.
   - Example:
     ```python
     actual = Path(self.path).resolve()
     meta_path = Path(meta.paths.ledger_path).resolve() if present else None
     if meta_path and meta_path != actual:
         raise RuntimeError(f"Ledger path mismatch: {meta_path} != {actual}")
     ```

6. **Verify Audio Determinism:**
   - Test if deleting the cache and re-running an old prompt produces identical audio bytes. If not, Rule C7 is violated.

### 5. Uncertainties for Verification

- **Is `LLMScriptWriter` being cached by ComfyUI?** Test by queuing two identical prompts. If the second run finishes instantly, the singleton is stale.
- **Does `rename_episode` update `_CURRENT.path`?** Verify that the singleton’s internal path string is updated when the directory is moved.
- **Are audio generators deterministic?** If not, Rule C7 relies entirely on the cache surviving the path reorg.
- **Downstream Log Parsing:** Check if any nodes parse `[Video] Saved` log messages. If so, ensure the path format remains compatible.

### Final Answer

**Prioritized Fixes:**
1. **Fix `video_engine.py` Post-Rename Path** to avoid stale `pending_<ts>` paths.
2. **Bust ComfyUI Cache for Singleton** using `IS_CHANGED` to ensure the singleton is always fresh.
3. **Layout-Aware Ledger Discovery** to replace stem swaps with directory-based lookups.
4. **Strict Cache Fallback** for MusicGen/AudioGen to preserve C7 and prevent VRAM spikes.
5. **Ledger Path Validation** in `Ledger.save()` to catch path-reorg bugs early.
6. **Verify Audio Determinism** to ensure Rule C7 compliance.

**Uncertainties to Verify:**
- ComfyUI cache behavior for `LLMScriptWriter`.
- `_CURRENT.path` updates in `rename_episode`.
- Audio generator determinism.
- Downstream log parsing for `[Video] Saved`.

**Key Takeaway:** Use the singleton (`_CURRENT`) but enforce cache invalidation (via `IS_CHANGED`) to ensure it’s always fresh. This balances simplicity with safety in ComfyUI.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
