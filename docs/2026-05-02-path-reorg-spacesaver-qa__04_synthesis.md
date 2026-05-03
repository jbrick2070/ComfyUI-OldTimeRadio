# Synthesis -- 2026-05-02

**Question:** # OTR path-reorg + spacesaver QA -- review my work for bugs / safety / schema gaps

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

---

## ChatGPT (gpt-5.5)

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

---

## Gemini (gemini-3.1-pro-preview-customtools)

### 1. Assessment
I PARTIALLY AGREE: ChatGPT correctly identified a catastrophic bug in the spacesaver's target discovery and the orphaned treatment file, but it hallucinated concurrency risks that don't exist on your hardware and completely missed that your timestamp suffix destroys both caching and your Rule C7 determinism constraint.

### 2. Factual Errors in ChatGPT's Answer
*   **Concurrency panic is hardware-impossible here:** ChatGPT recommends UUIDs and workflow locks to prevent concurrent ComfyUI runs from colliding. You are running a massive pipeline (Flux, LTX, AudioGen, etc.) on a single 16 GB VRAM laptop. ComfyUI's queue is strictly sequential, and your VRAM ceiling physically prevents concurrent execution of this pipeline. You do not need workflow locks.
*   **Misunderstanding of `_cache_key`:** ChatGPT says the `ts_ms` timestamp is "nondeterministic but okay" and suggests adding a UUID. It completely missed that the function is called `_cache_key`. If you append a timestamp (or UUID) to a cache filename, **you force a cache miss every single run**. 
*   **Rule C7 Violation:** ChatGPT failed to definitively flag that changing filenames every run via `ts_ms` threatens Rule C7. If FFmpeg embeds the input `.wav` filenames into the final `.mp4` metadata streams (which it often does by default), your final output bytes will change between identical runs.

### 3. Important Omissions
*   **ComfyUI Socket Architecture:** ChatGPT tells you to "derive the episode workspace from the actual `src`" in `rtx_upscale.py`, but misses the ComfyUI-native way to do this. You mentioned a `project_state` socket in your prompt. `RTXUpscale` should be receiving the exact `episode_id` or workspace path via this `project_state` dictionary, completely eliminating the need to guess paths or parse the `src` string.
*   **The `os.replace` fallback bug:** In `production_ledger.py`, if `os.replace` fails, you catch the exception and log a warning, but you *continue executing* and update the in-memory `self.episode_id`. This means your ledger thinks it moved, but the files are still in `pending_<ts>`. The pipeline will immediately crash on the next node that looks for the new path.

### 4. Short Recommendation
*   **Fix the Spacesaver Footgun:** Do not use `find_most_recent_ledger` in `rtx_upscale.py`. If I queue Episode A, then queue Episode B, Episode B writes its pending ledger. When Episode A reaches RTXUpscale, it will find Episode B's ledger and wipe Episode B's intermediates while it's rendering! Extract the `episode_id` from the `project_state` socket instead.
*   **Drop the Timestamp in Cache Keys:** In `musicgen_theme.py` and AudioGen, change the return to `return f"{cue_id}_{digest}.wav"`. This restores actual caching, guarantees Rule C7 byte-identity, and naturally prevents collisions.
*   **Fix the Orphaned Treatment:** In `production_ledger.py` (`rename_episode`), you must rename `pending_<ts>_treatment.txt` to `<new_id>_treatment.txt`. As written, it gets left behind with the old name, and spacesaver will delete it because it doesn't match the keep-list.
*   **Add a Windows Retry Block:** Wrap the `os.replace(old_ep_dir, new_ep_dir)` in a short `time.sleep` retry loop (3 attempts, 0.5s delay). Windows Defender *will* occasionally lock these files right after they are written, causing transient permission errors.
*   **Hard-fail on Dir Move Failure:** If `os.replace` fails after retries, raise a `RuntimeError`. Do not fall back to a "file-only rename" that leaves your workspace split between a pending folder and a finalized in-memory state.

### 5. Uncertainties to Verify
*   **When is the treatment written?** I am assuming the treatment is written *before* `SignalLostVideo` triggers the rename pass. If it's written *after*, you don't need to rename it. Check the node execution order.
*   **Does `project_state` carry the final ID?** I am assuming your `project_state` socket (mentioned in Phase D) carries the updated `episode_id` by the time it reaches the tail of the graph. If it only carries the `pending_<ts>` ID, you'll need to ensure the node that finalizes the ID (SignalLostVideo) updates `project_state` before passing it downstream.

---

## NVIDIA ()



---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
