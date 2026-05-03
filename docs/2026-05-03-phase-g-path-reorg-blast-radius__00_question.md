# Question -- 2026-05-02

# Phase G consult — path-reorg blast-radius audit (mid-soak crash repro)

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
