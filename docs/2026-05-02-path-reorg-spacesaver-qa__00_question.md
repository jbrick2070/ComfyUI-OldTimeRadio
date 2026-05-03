# Question -- 2026-05-02

# OTR path-reorg + spacesaver QA -- review my work for bugs / safety / schema gaps

## Context

I just shipped 4 sequential commits to v2.0-alpha that consolidate every project output under `output/otr/` in a per-episode workspace, plus added a new `perfect_run_spacesaver` toggle on the main script-writer node that wipes intermediates after the final mp4 lands. Asking for an external QA pass on:

- **bugs introduced** in this work
- **race conditions / safety holes** (especially the destructive cleanup + the rename-pass dir move)
- **back-compat soundness** (existing episodes on disk should stay readable)
- **ledger schema** -- should we bump the version? add new meta fields?
- **gaps for a final hardening phase**

Hardware / context: ComfyUI on Windows 11, RTX 5080 Laptop, 16 GB VRAM, single GPU, no cloud. Project is a 1940s-radio-drama generator with a long pipeline (LLM script -> Director -> MusicGen + Bark + Kokoro + AudioGen + AudioEnhance + EpisodeAssembler -> SignalLostVideo procgen -> BatchFluxRender -> BatchHumoRender -> BatchLTXRender -> VideoComposite -> RTXUpscale).

## The 4 phases shipped

| Phase | Commit | Change |
|---|---|---|
| A | 79ce42b | Split intermediate vs final: `otr/obs/<ep>.mp4` (FLAT, OBS-watched) vs `otr/episodes/<ep>.mp4` (intermediate). RTXUpscale writes to `obs/`; bypass mode COPIES the source there. |
| B | f3be5a4 | Per-episode workspace: `otr/episodes/<ep>/{audio,stills,portraits,videos,composited}/`. 4 path helpers refactored to take `episode_id`. Auto-pick walker handles both per-episode tree AND legacy flat layout for back-compat. MusicGen + AudioGen filenames get `_<timestamp_ms>` suffix for uniqueness. |
| C | 4d92ca3 | MusicGen + AudioGen wavs land in per-episode audio dir via `Ledger.rename_episode` extending to move the parent dir. `_default_out_dir(episode_id)` returns `otr/episodes/<ep>/audio/`. The pending ledger lives at `otr/episodes/pending_<ts>/audio/pending_<ts>_ledger.json` until SignalLostVideo finalizes the title. |
| D | 92fd67a | New `perfect_run_spacesaver` BOOLEAN widget on `OTR_LLMScriptWriter`. Stamped into `ledger.meta.perfect_run_spacesaver`. RTXUpscale reads it after writing the final mp4 and wipes per-episode intermediates, keeping only the ledger + treatment. |

## Final layout for new runs

```
output/otr/
├── obs/
│   └── <episode_id>.mp4                   # ★ FINAL, FLAT, OBS-watched
└── episodes/
    └── <episode_id>/                      # everything for THIS episode here
        ├── audio/
        │   ├── <episode_id>.mp4                   # procgen base
        │   ├── <episode_id>_ledger.json           # production ledger
        │   ├── <episode_id>_treatment.txt         # treatment
        │   ├── opening_<sha8>_<ts>.wav            # MusicGen
        │   ├── closing_<sha8>_<ts>.wav            # MusicGen
        │   ├── interstitial_<sha8>_<ts>.wav       # MusicGen
        │   ├── sfx_<prompt>_<sha8>_<ts>.wav       # AudioGen (one per cue)
        │   └── director_dump_<ts>.txt             # LLMDirector raw dumps
        ├── stills/                                # FLUX environments + radio bookend
        ├── portraits/                             # PASS1 character portraits
        ├── videos/                                # ✂ per-line clip pieces
        └── composited/                            # 832x480 intermediate
```

## Code excerpts to audit

### 1. `Ledger.rename_episode` -- extends to move the parent per-episode dir (production_ledger.py)

```python
def rename_episode(self, new_id: str) -> None:
    """Rename the episode id and atomically move BOTH the parent
    per-episode dir AND the ledger file inside.

    Per-episode workspace extension (2026-05-02 EVENING): the dir
    otr/episodes/pending_<ts>/ ALSO gets renamed to otr/episodes/<new_id>/
    so EVERY per-episode asset (Bark wavs, MusicGen wavs, AudioGen wavs,
    ledger, director dumps) moves together with the rename.
    """
    old = self.episode_id
    if old == new_id:
        return

    old_audio_dir = self.out_dir   # otr/episodes/pending_<ts>/audio
    old_ep_dir = os.path.dirname(old_audio_dir)  # otr/episodes/pending_<ts>
    new_ep_dir = os.path.join(os.path.dirname(old_ep_dir), new_id)
    new_audio_dir = os.path.join(new_ep_dir, os.path.basename(old_audio_dir))

    # Step 1: move the per-episode parent dir if it exists and the
    # destination isn't already there.
    moved_dir = False
    if (os.path.exists(old_ep_dir)
            and not os.path.exists(new_ep_dir)
            and old_ep_dir != new_ep_dir):
        try:
            os.replace(old_ep_dir, new_ep_dir)
            moved_dir = True
            log.info("[Ledger] per-episode dir moved %s -> %s", ...)
        except Exception as exc:
            log.warning("...; falling back to file-only rename", ...)

    # Step 2: update in-memory state.
    self.episode_id = new_id
    self.data["episode_id"] = new_id
    if moved_dir:
        self.out_dir = new_audio_dir

    # Step 3: rename the ledger file inside.
    if moved_dir:
        old_path = os.path.join(new_audio_dir, f"{_slugify(old, limit=120)}_ledger.json")
    else:
        old_path = os.path.join(old_audio_dir, f"{_slugify(old, limit=120)}_ledger.json")
    new_path = self.path
    if old_path != new_path and os.path.exists(old_path):
        try:
            os.replace(old_path, new_path)
        except Exception as exc:
            log.warning("[Ledger] BUG-108 file move failed (%s -> %s): %s", old_path, new_path, exc)
    log.info("[Ledger] renamed %s -> %s (dir_moved=%s)", old, new_id, moved_dir)
```

### 2. MusicGen `_cache_dir()` -- writes wavs alongside the in-flight ledger (musicgen_theme.py)

```python
def _cache_dir() -> str:
    """Per-episode MusicGen output dir.

    Walks the in-flight ledger via `find_most_recent_ledger` to find the
    current episode's `otr/episodes/<ep>/audio/` dir, then writes
    MusicGen wavs alongside the ledger. Falls back to legacy
    `models/musicgen_cache/` when no in-flight ledger is found.
    """
    try:
        ledger_path = _OTRL_PATHS.find_most_recent_ledger(
            [otr_episodes_root(), otr_legacy_audio_dir()]
        )
        if ledger_path is not None:
            base = str(ledger_path.parent)
            os.makedirs(base, exist_ok=True)
            return base
    except Exception as exc:
        log.warning("[MusicGenTheme] per-episode cache_dir lookup failed: %s", exc)

    # Legacy fallback: shared models/musicgen_cache/.
    try:
        import folder_paths
        base = os.path.join(folder_paths.models_dir, CACHE_SUBDIR)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, "..", "..", "..", "models", CACHE_SUBDIR))
    os.makedirs(base, exist_ok=True)
    return base
```

(AudioGen `_cache_dir()` follows the identical pattern.)

### 3. MusicGen + AudioGen filename uniqueness via `_<timestamp_ms>` suffix

```python
def _cache_key(cue_id, prompt, duration_sec, episode_seed) -> str:
    """Per-render cache filename, guaranteed unique across episodes.

    Format: <cue_id>_<sha8>_<timestamp_ms>.wav
    """
    import time as _time
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    ts_ms = int(_time.time() * 1000)
    return f"{cue_id}_{digest}_{ts_ms}.wav"
```

### 4. `find_most_recent_ledger` -- handles both layouts (back-compat) (_otr_ledger.py)

```python
def find_most_recent_ledger(audio_dirs: Iterable[Path]) -> Optional[Path]:
    """Walks each given dir at TWO levels:
      1. <dir>/*_ledger.json                       (legacy flat layout)
      2. <dir>/<episode_id>/audio/*_ledger.json    (per-episode workspace)
    """
    candidates: list[Path] = []
    for d in audio_dirs:
        try:
            d = Path(d)
            if not d.exists():
                continue
            candidates.extend(d.glob("*_ledger.json"))           # legacy
            candidates.extend(d.glob("*/audio/*_ledger.json"))   # per-episode
        except Exception as exc:
            log.warning(...)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
```

### 5. Spacesaver cleanup -- destructive wipe at end of pipeline (rtx_upscale.py)

```python
def _spacesaver_cleanup_if_flagged(*, src: Path, log_lines: list[str]) -> None:
    """If `meta.perfect_run_spacesaver` is True, wipe per-episode
    intermediates EXCEPT the ledger + treatment.

    Keep:  otr/episodes/<ep>/audio/<ep>_ledger.json
           otr/episodes/<ep>/audio/<ep>_treatment.txt
    Wipe:  audio/<ep>.mp4 + *.wav + director_dump_*.txt + pending_*_ledger.json
           stills/  portraits/  videos/  composited/
    """
    try:
        import _otr_ledger as _OTRL
        from _otr_paths import otr_episodes_root, otr_legacy_audio_dir
        ledger_path = _OTRL.find_most_recent_ledger(
            [otr_episodes_root(), otr_legacy_audio_dir()]
        )
    except Exception as exc:
        return
    if ledger_path is None:
        return
    led = _json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if not bool((led.get("meta") or {}).get("perfect_run_spacesaver")):
        return

    audio_dir = Path(ledger_path).parent
    ep_dir = audio_dir.parent
    ep_id = led.get("episode_id") or ep_dir.name

    # Sanity guard: refuse to wipe if ep_dir doesn't look like
    # a per-episode dir under otr/episodes/.
    parts_lower = [p.lower() for p in ep_dir.parts]
    if "episodes" not in parts_lower or "otr" not in parts_lower:
        log.warning("[OTR_RTXUpscale] spacesaver: refusing to wipe %s ...", ep_dir)
        return

    keep_files = {
        audio_dir / f"{ep_id}_ledger.json",
        audio_dir / f"{ep_id}_treatment.txt",
    }

    # Walk audio/: keep only ledger + treatment
    for child in audio_dir.iterdir():
        if child in keep_files:
            continue
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

    # Wipe sibling subdirs entirely
    for sub_name in ("stills", "portraits", "videos", "composited"):
        sub = ep_dir / sub_name
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)
```

### 6. New BOOLEAN widget on OTR_LLMScriptWriter

Placed AFTER `optimization_profile` and BEFORE the `project_state` socket (BUG-LOCAL-027 ordering rule says project_state must remain the tail socket-only entry). Default OFF. Stamped to ledger meta:

```python
_early_led.data.setdefault("meta", {})["perfect_run_spacesaver"] = bool(perfect_run_spacesaver)
```

## Specific questions

1. **`Ledger.rename_episode` dir-move safety.** Is there a race condition or partial-failure mode I'm not handling? E.g., what if MusicGen is mid-write to the audio dir when SignalLostVideo's rename pass fires? Windows `os.replace` on a directory should be atomic at the filesystem level but I'm not 100% sure. What's the worst-case if `os.replace(old_ep_dir, new_ep_dir)` fails halfway? Should I add a retry loop with backoff?

2. **MusicGen + AudioGen `_cache_dir()` race condition.** Both nodes call `find_most_recent_ledger` to locate the in-flight ledger and write caches alongside. If two ComfyUI workflows queue concurrently (unlikely but possible), both would resolve the SAME ledger and write to the same dir -- but with `_<timestamp_ms>` filename suffixes, would they collide? The timestamp is millisecond resolution; 2 calls within 1ms would produce identical filenames. Should I add a uuid4 suffix instead of (or alongside) timestamp?

3. **Spacesaver cleanup safety.** The sanity guard requires `episodes` AND `otr` in the path before wiping. Adequate? I'm worried about: (a) symlinks pointing outside the tree, (b) Windows file locks (in-flight reads from another process), (c) what if RTXUpscale runs but the ledger discovery fails -- spacesaver silently skips, which is the right answer but might hide a bug. Should the cleanup also stamp `meta.spacesaver_executed = true/false` into the ledger so we know post-run whether it ran?

4. **Back-compat with existing episodes.** Walker scans BOTH `<dir>/*_ledger.json` (legacy) AND `<dir>/<ep>/audio/*_ledger.json` (new). Is there a glob pattern that could collide on disks where a future legacy episode happens to have a directory inside `otr/audio/`? Or is the two-layer glob safe?

5. **Ledger schema.** I added `meta.perfect_run_spacesaver` as a top-level meta key (in addition to nesting it under `meta.gen_params_initial`). Should I bump the schema version (currently `l3-2026-04-28`)? What other new meta fields would help post-run forensics:
   - `meta.layout_version` so future migrations can tell which layout an episode used?
   - `meta.spacesaver_executed` so a post-run scan can tell which episodes were wiped?
   - `meta.dir_moved_from_pending` recording the original pending_<ts> stem so we can correlate logs across the rename?

6. **Gaps for a final hardening phase.** What QA tests should I add to lock this down? My current regression covers AST + workflow JSON shape (`tests/test_workflow_zod_shape.py`) + bug bible regression. Missing:
   - Functional test for `Ledger.rename_episode` dir-move (would need temp dir setup)
   - Test that find_most_recent_ledger actually finds ledgers in the new layout
   - Test that spacesaver wipe respects the keep list
   - Test that the sanity guard refuses bad paths

7. **Anything else you'd flag.** Cosmetic, architectural, or correctness-related issues I missed.

## Format requested

- Bullet list per question
- If the work is sound on a question, say so plainly so I can move on
- If you'd reject it, name the specific failure mode (with grep target / log line if applicable)
- Do not relitigate the layout choices (otr/obs/ vs otr/episodes/<ep>/) -- those are settled by Jeffrey directive. Only flag concrete runtime risks.
