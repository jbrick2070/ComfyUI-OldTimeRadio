# Question -- 2026-05-02

# OTR Sprint 3 BUG-LOCAL-011 fix review -- production workflow attached

## What we're asking

Did we fix BUG-LOCAL-011 the right way, or is there a sharper / safer / more maintainable alternative? Specific concerns at the bottom.

## Failure observed in production

Live run on ComfyUI Desktop using `workflows/otr_scifi_16gb_full.json` (commit `7c4dfd4`, the Sprint 3 mega-sprint commit). LLM phase passed (Gemma-4 E2B), audio cascade passed (Bark + Kokoro + MusicGen + AudioGen + AudioEnhance + EpisodeAssembler), SignalLostVideo passed (procgen mp4 saved 52.2 MB / 113s / 2712 frames), BatchFluxRender passed (5 cast portraits + radio bookend at 1248x720 -> Lanczos 832x480), BatchHumoRender passed (4 character clips l002-l005 in ~40 min wallclock). Then the LowVRAMCheckpointLoader -> OTR_BatchLTXRender chain fired (sequencing edge worked), but BatchLTXRender raised:

```
File "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\batch_ltx_render.py", line 446, in execute
    raise RuntimeError(
RuntimeError: BatchLTXRender: ledger could not be loaded from inline JSON or path

Prompt executed in 00:58:53
```

## Root cause

`OTR_SignalLostVideo.0` (the STRING input feeding `BatchLTXRender.ledger_json` via link 90) emits the **mp4 path**, not the `_ledger.json` path. Sister nodes `BatchHumoRender._load_ledger_with_path` and `OTR_VideoComposite._load_ledger_with_path` both have a multi-tier stem-fallback that swaps `.mp4` -> `_ledger.json` (BUG-LOCAL-118 hardening from a prior session). The new `BatchLTXRender._load_ledger` skipped that fallback -- it called `load_ledger_safe(.mp4)` directly, got `None`, returned `(None, None)`, raised.

## ORIGINAL (broken) code -- nodes/batch_ltx_render.py @ commit 7c4dfd4

```python
@staticmethod
def _load_ledger(arg: str) -> tuple[dict | None, Path | None]:
    """Accept inline JSON OR a path to *_ledger.json. Empty arg
    triggers auto-pick of the most recent ledger under
    otr_audio_dir / otr_legacy_audio_dir."""
    import json as _json
    try:
        from . import _otr_ledger as _OTRL  # type: ignore
    except Exception:
        _OTRL = None  # type: ignore

    if not arg or not isinstance(arg, str) or not arg.strip():
        # Auto-pick most recent.
        if _OTRL is not None:
            p = _OTRL.find_most_recent_ledger(
                [otr_audio_dir(), otr_legacy_audio_dir()]
            )
            if p is not None:
                led = _OTRL.load_ledger_safe(p)
                if led is not None:
                    return led, p
        return None, None

    s = arg.strip()
    # Path?
    if s.startswith("{") or s.startswith("["):
        try:
            return _json.loads(s), None
        except _json.JSONDecodeError:
            return None, None
    # Path string.
    try:
        p = Path(s)
    except Exception:  # noqa: BLE001
        return None, None
    if not p.is_file():
        return None, None
    if _OTRL is not None:
        led = _OTRL.load_ledger_safe(p)
        return led, p
    try:
        with open(p, "r", encoding="utf-8") as f:
            return _json.load(f), p
    except Exception:  # noqa: BLE001
        return None, None
```

## SISTER REFERENCE -- `BatchHumoRender._load_ledger_with_path` @ nodes/batch_humo_render.py lines 1747-1869 (production-stable, has been in tree for weeks)

```python
@staticmethod
def _load_ledger_with_path(ledger_arg: str) -> tuple[dict, Path | None]:
    """Accept either:
      - inline JSON string (starts with '{') -- returns (dict, None)
      - path to *_ledger.json -- returns (dict, Path)
      - path to *.mp4 (audio episode); ledger inferred via
        suffix swap (.mp4 -> _ledger.json), since that's the
        convention OTR_SignalLostVideo / EpisodeAssembler write.
        Lets us wire BatchHumoRender's ledger_json input directly
        from SignalLostVideo.video_path -- no separate ledger
        output node required.
      - empty -> auto-pick newest non-pending in the canonical
        audio dirs (BUG-LOCAL-076 fallback chain).
    """
    s = (ledger_arg or "").strip()

    # Auto-pick fallback when input empty
    if not s:
        audio_dirs = [otr_audio_dir(), otr_legacy_audio_dir()]
        cands = []
        for d in audio_dirs:
            if d.exists():
                cands.extend(p for p in d.glob("*_ledger.json") if not p.name.startswith("pending_"))
        if not cands:
            raise RuntimeError("BatchHumoRender: ledger_json empty and auto-pick found no ledger")
        p = max(cands, key=lambda x: x.stat().st_mtime)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), p

    if s.startswith("{"):
        return json.loads(s), None

    p = Path(s)
    # .mp4 path -> swap suffix to _ledger.json (SignalLostVideo
    # convention). BUG-LOCAL-118 hardening 2026-04-30: tries
    # (1) exact match, (2) collapsed-underscore variant,
    # (3) directory scan for newest <1h old fuzzy match.
    if p.suffix.lower() == ".mp4":
        audio_dir = p.parent
        stem = p.stem

        # Tier 1: exact match
        ledger_p = audio_dir / f"{stem}_ledger.json"
        if ledger_p.exists():
            with open(ledger_p, "r", encoding="utf-8") as f:
                return json.load(f), ledger_p

        # Tier 2: collapsed-underscore variant (a__20260430 -> a_20260430)
        collapsed = stem
        while "__" in collapsed:
            collapsed = collapsed.replace("__", "_")
        if collapsed != stem:
            cand = audio_dir / f"{collapsed}_ledger.json"
            if cand.exists():
                log.warning("[BatchHumoRender] BUG-LOCAL-118 underscore-mismatch fallback...")
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f), cand

        # Tier 3: directory scan for fuzzy-match ledger <1h old
        try:
            cands = list(audio_dir.glob("*_ledger.json"))
            cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            stem_norm = collapsed
            for cand in cands[:10]:
                cand_eid = cand.stem
                if cand_eid.endswith("_ledger"):
                    cand_eid = cand_eid[: -len("_ledger")]
                cand_norm = cand_eid
                while "__" in cand_norm:
                    cand_norm = cand_norm.replace("__", "_")
                if cand_norm == stem_norm or cand_norm in stem_norm or stem_norm in cand_norm:
                    age_s = time.time() - cand.stat().st_mtime
                    if age_s > 3600:
                        continue
                    log.warning("[BatchHumoRender] BUG-LOCAL-118 fuzzy fallback: binding to %r", cand.name)
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f), cand
        except Exception as scan_exc:
            log.warning("[BatchHumoRender] BUG-LOCAL-118 directory-scan fallback failed (%s)", scan_exc)

        raise RuntimeError(f"BatchHumoRender: derived ledger from .mp4 not found: {ledger_p}")

    # Plain ledger.json path
    if not p.exists():
        raise RuntimeError(f"BatchHumoRender: ledger path not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), p
```

## NEW (proposed) code -- nodes/batch_ltx_render.py @ commit 5ebc94b

```python
@staticmethod
def _load_ledger(arg: str) -> tuple[dict | None, Path | None]:
    """Accept either:
      - inline JSON string (starts with '{') -- returns (dict, None)
      - path to *_ledger.json -- returns (dict, Path)
      - path to *.mp4 (audio episode); ledger inferred via
        suffix swap (.mp4 -> _ledger.json), since that's the
        convention OTR_SignalLostVideo / EpisodeAssembler write.
      - empty -> auto-pick newest non-pending in the canonical
        audio dirs.

    Returns (ledger_dict_or_None, ledger_path_or_None). Mirrors the
    contract of BatchHumoRender._load_ledger_with_path. BUG-LOCAL-011
    (2026-05-02): originally only handled exact .json paths and failed
    when wired to SignalLostVideo's .mp4 output -- ported the
    multi-tier stem fallback from batch_humo_render.py.
    """
    import json as _json

    s = (arg or "").strip()

    # Layer 0: empty input -> auto-pick newest non-pending ledger
    if not s:
        audio_dirs = [otr_audio_dir(), otr_legacy_audio_dir()]
        cands = []
        for d in audio_dirs:
            if d.exists():
                cands.extend(p for p in d.glob("*_ledger.json") if not p.name.startswith("pending_"))
        if not cands:
            log.warning("[BatchLTXRender] ledger_json empty and auto-pick found no ledger in audio dirs")
            return None, None
        p = max(cands, key=lambda x: x.stat().st_mtime)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return _json.load(f), p
        except Exception as exc:
            log.warning("[BatchLTXRender] auto-pick ledger %s failed to load: %s", p, exc)
            return None, None

    # Layer 1: inline JSON object
    if s.startswith("{") or s.startswith("["):
        try:
            return _json.loads(s), None
        except _json.JSONDecodeError as exc:
            log.warning("[BatchLTXRender] inline JSON parse failed: %s", exc)
            return None, None

    # Layer 2: filesystem path -- could be .mp4 or _ledger.json
    try:
        p = Path(s)
    except Exception:
        return None, None

    # Layer 2a: .mp4 path -> swap suffix to _ledger.json (SignalLostVideo
    # / EpisodeAssembler convention). Same multi-tier fallback chain as
    # BatchHumoRender (BUG-LOCAL-118 hardening): (1) exact match,
    # (2) collapsed-underscore variant, (3) directory scan for newest
    # <1h old fuzzy match.
    if p.suffix.lower() == ".mp4":
        audio_dir = p.parent
        stem = p.stem

        # Tier 1: direct match
        ledger_p = audio_dir / f"{stem}_ledger.json"
        if ledger_p.exists():
            try:
                with open(ledger_p, "r", encoding="utf-8") as f:
                    return _json.load(f), ledger_p
            except Exception as exc:
                log.warning("[BatchLTXRender] tier-1 ledger %s failed to load: %s", ledger_p, exc)

        # Tier 2: underscore-collapse variant
        collapsed = stem
        while "__" in collapsed:
            collapsed = collapsed.replace("__", "_")
        if collapsed != stem:
            cand = audio_dir / f"{collapsed}_ledger.json"
            if cand.exists():
                log.warning("[BatchLTXRender] BUG-LOCAL-118 underscore-mismatch fallback: ...")
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        return _json.load(f), cand
                except Exception as exc:
                    log.warning("[BatchLTXRender] tier-2 ledger %s failed to load: %s", cand, exc)

        # Tier 3: directory scan for fuzzy-match ledger <1h old
        try:
            cands = list(audio_dir.glob("*_ledger.json"))
            cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            stem_norm = collapsed
            for cand in cands[:10]:
                cand_eid = cand.stem
                if cand_eid.endswith("_ledger"):
                    cand_eid = cand_eid[: -len("_ledger")]
                cand_norm = cand_eid
                while "__" in cand_norm:
                    cand_norm = cand_norm.replace("__", "_")
                if cand_norm == stem_norm or cand_norm in stem_norm or stem_norm in cand_norm:
                    age_s = time.time() - cand.stat().st_mtime
                    if age_s > 3600:
                        continue
                    log.warning("[BatchLTXRender] BUG-LOCAL-118 fuzzy fallback: binding to %r ...", cand.name)
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            return _json.load(f), cand
                    except Exception as exc:
                        log.warning("[BatchLTXRender] tier-3 ledger %s failed to load: %s", cand, exc)
        except Exception as scan_exc:
            log.warning("[BatchLTXRender] BUG-LOCAL-118 directory-scan fallback failed (%s)", scan_exc)

        log.warning("[BatchLTXRender] derived ledger from .mp4 not found: %s", ledger_p)
        return None, None

    # Layer 2b: plain _ledger.json path
    if not p.is_file():
        return None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return _json.load(f), p
    except Exception as exc:
        log.warning("[BatchLTXRender] ledger path %s failed to load: %s", p, exc)
        return None, None
```

## Architecture context

- OTR is a ComfyUI plugin generating 1940s-style radio drama episodes; pipeline is LLM (script/director) -> audio (Bark + Kokoro + MusicGen + AudioGen + AudioEnhance + EpisodeAssembler) -> SignalLostVideo (procgen base mp4) -> BatchFluxRender (cast portraits + radio bookend) -> BatchHumoRender (character lip-sync) -> LowVRAMCheckpointLoader + BatchLTXRender (radio motion for non-character roles) -> VideoComposite (832x480 native) -> RTXUpscale (1080p with audio -c:a copy passthrough).
- Hardware: Windows 11, RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud. VRAM ceiling 14.5 GB audio / 15.5 GB video.
- The locked Architecture Truth (2026-05-02) wanted UNETLoader+CLIPLoader+VAELoader for LTX 2B, but the bundled `ltx-video-2b-v0.9.safetensors` from Lightricks is the only artifact, so we use ComfyUI-LTXVideo's `LowVRAMCheckpointLoader` (a CheckpointLoaderSimple subclass with a `dependencies` STRING input that ComfyUI uses to force sequential model load). HuMo's strict teardown clears 16.5 GB before LTX claims VRAM.
- Round-robin consult before this commit (gpt-5.5 + gemini-3.1-pro-preview-customtools) caught: ffmpeg pipe deadlock on Windows (DEVNULL fix), ComfyUI cache desync (link 86 routed through HuMo.report instead of clips_dir), anti-clobber `if out_mp4.exists(): skip` in LTX, dropping `-shortest` from upscale audio mux. None of those flagged the .mp4 -> _ledger.json mismatch -- they all assumed the LTX node's resolver matched HuMo's because the input slot has the same name `ledger_json`.

## Resolver test confirmed (offline, before any GPU work)

| Test | Result |
|---|---|
| .mp4 path -> stem-swap to .json | ledger loaded, 10 lines, 4 humo clips already stamped |
| LTX-eligible roles in ledger | 6 (2 music_open + 2 announcer + 2 music_close) |
| Radio bookend resolution | 537 KB PNG resolved by `_resolve_radio_still_path` |
| Explicit .json path branch | also works |
| Empty input auto-pick | returns None when no non-pending ledgers exist |

## Specific questions

1. **Is this fix correct?** Or is there a sharper alternative we missed? (E.g. just delegate to a shared resolver module since BatchHumoRender + VideoComposite + BatchLTXRender all need the same logic -- DRY refactor before more divergence?)

2. **Did the new resolver lose any behavioural property the old broken one had?** (E.g. the old code accepted JSON arrays starting with `[`, and used `_OTRL.load_ledger_safe` which I think does some additional validation. The new code dropped the `_OTRL` indirection because BatchHumoRender's reference doesn't use it. Are we losing schema validation by going direct to `json.load()`?)

3. **Is the multi-tier fuzzy-fallback (tier 3 directory scan with <1h freshness gate) appropriate for LTX?** It was hardened for BUG-LOCAL-118 (a HuMo-era underscore-mismatch issue). Could it bind LTX to the wrong ledger in any plausible scenario?

4. **Why didn't the prior round-robin catch this?** Both ChatGPT and Gemini reviewed BatchLTXRender's wiring and the JSON before the commit. They flagged 4 real issues (pipe deadlock, cache desync, anti-clobber, -shortest) but missed that the resolver internals diverged from HuMo. What heuristic would catch this class of "node interface matches but internals diverge" bug in the future?

5. **Is there a deeper architectural fix?** Three nodes (HuMo, LTX, VideoComposite) all need to resolve "ledger from a STRING that might be .mp4 or .json". Right now HuMo + VideoComposite share the resolver but LTX has its own copy. A shared `_otr_ledger_resolver.py` module would prevent this. Worth doing now or wait until a fourth node needs it?

6. **Any concern with the new `humo_clips_dir` widget** (added as an explicit DAG sequencing edge from BatchHumoRender.report)? The render code ignores its value -- it's purely there to make ComfyUI evaluate LowVRAMCheckpointLoader after HuMo runs (cache-buster from prior consult). Is "use a widget for sequencing only" an anti-pattern, or fine?

## Format requested

Bullet list per question. If the fix is sound, say so plainly so I can move on. If you'd reject the fix, name the specific failure mode you're worried about. Don't relitigate the architecture choices above (LowVRAMCheckpointLoader vs UNETLoader, .mp4 paths as ledger inputs, etc.) unless you're flagging a concrete runtime risk.


## Relevant subgraph from `workflows/otr_scifi_16gb_full.json` (commit 7c4dfd4)

Only the video-stage nodes are shown -- the upstream LLM/audio chain is omitted because it executed cleanly before this failure.


### Nodes

| ID | Type | Title |
|---|---|---|
| 12 | OTR_SignalLostVideo |  |
| 22 | CheckpointLoaderSimple |  |
| 23 | OTR_BatchFluxRender |  |
| 24 | OTR_UnloadAll |  |
| 25 | SaveImage |  |
| 42 | PathchSageAttentionKJ |  |
| 45 | UNETLoader |  |
| 46 | LoraLoaderModelOnly |  |
| 47 | ModelSamplingSD3 |  |
| 48 | CLIPLoader |  |
| 49 | VAELoader |  |
| 50 | AudioEncoderLoader |  |
| 51 | OTR_BatchHumoRender |  |
| 52 | OTR_VideoComposite |  |
| 54 | LowVRAMCheckpointLoader | LowVRAM Loader (LTX 2B v0.9) |
| 55 | OTR_BatchLTXRender | Batch LTX Render (radio) |
| 56 | OTR_RTXUpscale | RTX VSR Upscale (1080p) |

### Per-node inputs / outputs (link IDs)


**12 `OTR_SignalLostVideo`** -- 
  - in [0] `audio`:`AUDIO` link=15
  - in [1] `script_json`:`STRING` link=16
  - in [2] `production_plan_json`:`STRING` link=17
  - in [3] `news_used`:`STRING` link=18
  - out[0] `video_path`:`STRING` links=[47, 79, 80, 82, 90]

**22 `CheckpointLoaderSimple`** -- 
  - out[0] `MODEL`:`MODEL` links=[42]
  - out[1] `CLIP`:`CLIP` links=[43]
  - out[2] `VAE`:`VAE` links=[44]

**23 `OTR_BatchFluxRender`** -- 
  - in [0] `model`:`MODEL` link=69
  - in [1] `clip`:`CLIP` link=43
  - in [2] `vae`:`VAE` link=44
  - in [3] `script_json`:`STRING` link=41
  - out[0] `images`:`IMAGE` links=[45]
  - out[1] `report`:`STRING` links=None

**24 `OTR_UnloadAll`** -- 
  - in [0] `image`:`IMAGE` link=45
  - out[0] `image`:`IMAGE` links=[46, 83]

**25 `SaveImage`** -- 
  - in [0] `images`:`IMAGE` link=46

**42 `PathchSageAttentionKJ`** -- 
  - in [0] `model`:`MODEL` link=42
  - out[0] `MODEL`:`MODEL` links=[69]

**45 `UNETLoader`** -- 
  - out[0] `MODEL`:`MODEL` links=[72]

**46 `LoraLoaderModelOnly`** -- 
  - in [0] `model`:`MODEL` link=72
  - out[0] `MODEL`:`MODEL` links=[73]

**47 `ModelSamplingSD3`** -- 
  - in [0] `model`:`MODEL` link=73
  - out[0] `MODEL`:`MODEL` links=[74]

**48 `CLIPLoader`** -- 
  - out[0] `CLIP`:`CLIP` links=[75]

**49 `VAELoader`** -- 
  - out[0] `VAE`:`VAE` links=[76]

**50 `AudioEncoderLoader`** -- 
  - out[0] `AUDIO_ENCODER`:`AUDIO_ENCODER` links=[77]

**51 `OTR_BatchHumoRender`** -- 
  - in [0] `model`:`MODEL` link=74
  - in [1] `clip`:`CLIP` link=75
  - in [2] `vae`:`VAE` link=76
  - in [3] `audio_encoder`:`AUDIO_ENCODER` link=77
  - in [4] `audio`:`AUDIO` link=78
  - in [5] `ledger_json`:`STRING` link=79
  - in [6] `portraits_dir`:`STRING` link=None
  - in [7] `flux_done_gate`:`IMAGE` link=83
  - out[0] `clips_dir`:`STRING` links=[91]
  - out[1] `clip_count`:`INT` links=None
  - out[2] `report`:`STRING` links=[86]

**52 `OTR_VideoComposite`** -- 
  - in [0] `procgen_video_path`:`STRING` link=80
  - in [1] `clips_dir`:`STRING` link=92
  - in [2] `ledger_json`:`STRING` link=82
  - out[0] `final_mp4_path`:`STRING` links=[93]
  - out[1] `report`:`STRING` links=None

**54 `LowVRAMCheckpointLoader`** -- LowVRAM Loader (LTX 2B v0.9)
  - in [0] `dependencies`:`*` link=86
  - out[0] `MODEL`:`MODEL` links=[87]
  - out[1] `CLIP`:`CLIP` links=[88]
  - out[2] `VAE`:`VAE` links=[89]

**55 `OTR_BatchLTXRender`** -- Batch LTX Render (radio)
  - in [0] `model`:`MODEL` link=87
  - in [1] `clip`:`CLIP` link=88
  - in [2] `vae`:`VAE` link=89
  - in [3] `ledger_json`:`STRING` link=90
  - in [4] `humo_clips_dir`:`STRING` link=91
  - out[0] `clips_dir`:`STRING` links=[92]
  - out[1] `clip_count`:`INT` links=[]
  - out[2] `report`:`STRING` links=[]

**56 `OTR_RTXUpscale`** -- RTX VSR Upscale (1080p)
  - in [0] `source_mp4_path`:`STRING` link=93
  - out[0] `upscaled_mp4_path`:`STRING` links=[]
  - out[1] `report`:`STRING` links=[]

### Links touching this subgraph

| link_id | src | dst | type |
|---|---|---|---|
| 15 | 7.0 | 12.0 | AUDIO |
| 16 | 53.1 | 12.1 | STRING |
| 17 | 2.0 | 12.2 | STRING |
| 18 | 1.2 | 12.3 | STRING |
| 41 | 21.0 | 23.3 | STRING |
| 42 | 22.0 | 42.0 | MODEL |
| 43 | 22.1 | 23.1 | CLIP |
| 44 | 22.2 | 23.2 | VAE |
| 45 | 23.0 | 24.0 | IMAGE |
| 46 | 24.0 | 25.0 | IMAGE |
| 47 | 12.0 | 20.1 | STRING |
| 69 | 42.0 | 23.0 | MODEL |
| 72 | 45.0 | 46.0 | MODEL |
| 73 | 46.0 | 47.0 | MODEL |
| 74 | 47.0 | 51.0 | MODEL |
| 75 | 48.0 | 51.1 | CLIP |
| 76 | 49.0 | 51.2 | VAE |
| 77 | 50.0 | 51.3 | AUDIO_ENCODER |
| 78 | 7.0 | 51.4 | AUDIO |
| 79 | 12.0 | 51.5 | STRING |
| 80 | 12.0 | 52.0 | STRING |
| 82 | 12.0 | 52.2 | STRING |
| 83 | 24.0 | 51.7 | IMAGE |
| 86 | 51.2 | 54.0 | STRING |
| 87 | 54.0 | 55.0 | MODEL |
| 88 | 54.1 | 55.1 | CLIP |
| 89 | 54.2 | 55.2 | VAE |
| 90 | 12.0 | 55.3 | STRING |
| 91 | 51.0 | 55.4 | STRING |
| 92 | 55.0 | 52.1 | STRING |
| 93 | 52.0 | 56.0 | STRING |

### Critical wiring path that triggered the bug

```
OTR_SignalLostVideo (12).0 (STRING = mp4 path)
  --link 90--> OTR_BatchLTXRender (55).3 (ledger_json STRING)

At runtime, BatchLTXRender._load_ledger received the .mp4 path,
failed to stem-swap to _ledger.json, returned (None, None),
raised RuntimeError('ledger could not be loaded ...').
```

### Other nodes that consume the same SignalLostVideo (12).0 STRING and DO have the stem-swap (work correctly)

- OTR_BatchHumoRender (51).5 (ledger_json) via link 79
- OTR_VideoComposite (52).0 (procgen_video_path) via link 80
- OTR_VideoComposite (52).2 (ledger_json) via link 82
- OTR_BatchLTXRender (55).3 (ledger_json) via link 90 -- **THIS is the failing one**
