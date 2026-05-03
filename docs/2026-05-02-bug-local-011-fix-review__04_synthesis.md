# Synthesis -- 2026-05-02

**Question:** # OTR Sprint 3 BUG-LOCAL-011 fix review -- production workflow attached

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

---

## ChatGPT (gpt-5.5)

- **Bottom line**
  - **Yes, the proposed fix is basically correct as a tactical production fix.** It addresses the observed failure: `OTR_BatchLTXRender.ledger_json` is wired from `OTR_SignalLostVideo.video_path` via workflow link `90`, and that value is an `.mp4` path, not a `*_ledger.json` path.
  - I would **not reject the fix**, and if you need to unblock the production run, this is good enough to move forward.
  - The sharper/safest small improvement would be:
    - keep the `.mp4 -> *_ledger.json` resolver logic,
    - but preserve `_otr_ledger.load_ledger_safe()` when loading resolved ledger paths,
    - and avoid falling through to fuzzy fallback if the exact derived ledger exists but is malformed/unloadable.
  - The deeper maintainability fix is a shared resolver module, but I would do that **after** the production unblock unless you already have tests around all three consumers.

---

## 1. Is this fix correct? Or is there a sharper alternative?

- **Correct for BUG-LOCAL-011: yes.**
  - The production failure was caused by `nodes/batch_ltx_render.py` receiving an `.mp4` path through `workflows/otr_scifi_16gb_full.json` link `90`:
    - `OTR_SignalLostVideo.0 video_path`
    - → `OTR_BatchLTXRender.3 ledger_json`
  - The old `BatchLTXRender._load_ledger()` accepted inline JSON or an existing path, but did **not** do the `.mp4` stem swap.
  - Your new code adds the same major behavior that `BatchHumoRender._load_ledger_with_path()` has at `nodes/batch_humo_render.py` lines `1747-1869`: derive `*_ledger.json` from an `.mp4` input.

- **I would accept this as the production hotfix.**
  - It is the smallest change with the largest payoff.
  - It does not touch VRAM behavior.
  - It does not touch model loading.
  - It does not relitigate `LowVRAMCheckpointLoader`.
  - It fixes the concrete crash at `nodes/batch_ltx_render.py` line `446`, where the node raises because `_load_ledger()` returned `(None, None)`.

- **Sharper small alternative: preserve the old `_OTRL` loader path.**
  - The original code used:

    ```python
    from . import _otr_ledger as _OTRL
    ...
    led = _OTRL.load_ledger_safe(p)
    ```

  - The proposed code switches to direct `json.load()`.
  - I would prefer a helper inside `BatchLTXRender._load_ledger()` like:

    ```python
    def _load_path(p: Path) -> tuple[dict | None, Path | None]:
        if _OTRL is not None:
            led = _OTRL.load_ledger_safe(p)
            return (led, p) if led is not None else (None, None)
        try:
            with open(p, "r", encoding="utf-8") as f:
                led = _json.load(f)
            return (led, p) if isinstance(led, dict) else (None, None)
        except Exception:
            return None, None
    ```

  - Then use `_load_path()` for:
    - exact `.mp4 -> *_ledger.json`,
    - collapsed-underscore candidate,
    - fuzzy candidate,
    - explicit `.json` path,
    - auto-pick path.

- **One important tiny safety tweak: do not fuzzy-fallback after a present exact ledger fails to parse.**
  - Current proposed behavior:
    - if `episode.mp4` maps to `episode_ledger.json`,
    - and `episode_ledger.json` exists but fails to load,
    - the resolver logs and continues to tier 2 / tier 3.
  - That could bind to a different nearby ledger.
  - Safer behavior:
    - if exact path exists but is malformed, stop and fail.
    - malformed exact ledger is not a “not found” case.
  - Same for tier 2: if the collapsed-underscore candidate exists but fails to load, stop and fail rather than scanning fuzzily.

- **Check `time` import.**
  - The new tier 3 code uses `time.time()`.
  - If `time` is not already imported at the top of `nodes/batch_ltx_render.py`, tier 3 will fail with `NameError`, but that error will be caught by the surrounding `except Exception as scan_exc`.
  - So it probably will not crash the run, but tier 3 silently becomes ineffective.
  - Still worth adding explicitly:

    ```python
    import time
    ```

  - Or import locally inside the tier 3 block.

---

## 2. Did the new resolver lose any behavioral property the old one had?

- **It did not lose inline JSON array support.**
  - Old code accepted strings starting with `{` or `[`.
  - New code also accepts `{` or `[`.
  - So that specific behavior is retained.

- **It may have lost `_otr_ledger.load_ledger_safe()` behavior.**
  - This is the main concern.
  - The old code delegated path loading to `_OTRL.load_ledger_safe(p)` when `_otr_ledger` was importable.
  - The new code does direct `json.load()`.
  - I cannot say with certainty what you lost without seeing `nodes/_otr_ledger.py`, but possible losses include:
    - schema validation,
    - pending-ledger rejection,
    - normalizing legacy fields,
    - rejecting partial/corrupt ledgers,
    - logging consistency,
    - future validation improvements automatically inherited by callers.

- **I would restore `_OTRL.load_ledger_safe()` for file-path loads.**
  - This is low risk and preserves the older contract.
  - The `.mp4` branch only needs to resolve the candidate ledger path.
  - Once it has the candidate path, it should ideally load it the same way the old LTX resolver loaded explicit ledger paths.

- **The new empty-input auto-pick is different.**
  - Old behavior:
    - only auto-picked if `_OTRL` imported successfully,
    - used `_OTRL.find_most_recent_ledger([otr_audio_dir(), otr_legacy_audio_dir()])`,
    - then loaded with `_OTRL.load_ledger_safe()`.
  - New behavior:
    - manually globs `*_ledger.json`,
    - excludes names starting with `pending_`,
    - picks max `mtime`,
    - direct `json.load()`.

- **That may be fine, but it is not behavior-identical.**
  - If `_OTRL.find_most_recent_ledger()` has more rules than “newest non-pending `*_ledger.json`,” the new code bypasses them.
  - Since you said empty input returns `None` in your offline test when no ledgers exist, this likely does not affect the production path.
  - Production path is `.mp4` input, not empty input.

- **The proposed code is slightly less type-safe.**
  - Inline JSON beginning with `[` can return a Python `list`, while the annotation says `dict | None`.
  - The old code had the same issue for inline arrays.
  - If downstream assumes `dict`, then allowing `[` is arguably legacy compatibility rather than a desirable behavior.
  - I would leave it alone unless you want to intentionally tighten the contract.

- **The new code can resolve from an `.mp4` path even if the `.mp4` itself does not exist.**
  - It does not check `p.is_file()` before deriving the ledger path.
  - In the real workflow, `OTR_SignalLostVideo` has just produced the `.mp4`, so this should not matter.
  - But for stale/manual paths, this could bind to a ledger even when the referenced video does not exist.
  - Not a blocker, but a possible hardening point.

---

## 3. Is the tier 3 fuzzy fallback appropriate for LTX?

- **Tier 1 is definitely appropriate.**
  - `.mp4` path:

    ```text
    episode.mp4
    ```

    should map to:

    ```text
    episode_ledger.json
    ```

  - This is the direct fix for BUG-LOCAL-011.

- **Tier 2 is probably appropriate.**
  - The collapsed-underscore fallback exists because of prior BUG-LOCAL-118 hardening.
  - If HuMo and VideoComposite already need it, keeping LTX consistent is reasonable.

- **Tier 3 is the only part I am uneasy about.**
  - Directory scan + fuzzy stem matching + `<1h` freshness gate is useful as a rescue path, but it is inherently less deterministic.
  - It depends on:
    - directory contents,
    - file mtimes,
    - wall-clock time,
    - how many recent ledgers exist,
    - substring matching between normalized stems.

- **Could tier 3 bind LTX to the wrong ledger? Yes, plausibly.**
  - Example scenario:
    - two episodes generated within an hour,
    - similar stems,
    - exact ledger missing or malformed for current `.mp4`,
    - fuzzy match sees another recent `*_ledger.json`,
    - substring condition matches:

      ```python
      cand_norm == stem_norm or cand_norm in stem_norm or stem_norm in cand_norm
      ```

  - This is especially risky if many OTR outputs share a common prefix like:

    ```text
    signal_lost_...
    ```

- **For C7-style repeatability discipline, tier 3 is also philosophically suspect.**
  - Your C7 rule is specifically about audio byte identity, and this resolver is downstream video, so it does not directly alter generated audio bytes.
  - But the same determinism principle applies: time-based fuzzy binding can produce different downstream video decisions across runs if directory state changes.
  - I would avoid depending on tier 3 for normal production.

- **My recommendation for LTX:**
  - Keep tier 1.
  - Keep tier 2.
  - Either:
    - disable tier 3 by default for LTX, or
    - keep tier 3 only if it validates that the candidate ledger truly belongs to the `.mp4`.

- **If you keep tier 3, add validation before accepting the candidate.**
  - Ideally verify one of:
    - ledger episode id matches normalized `.mp4` stem,
    - ledger contains the exact procgen/audio/video path,
    - ledger base output stem matches,
    - ledger timestamp/id matches the filename.
  - If the ledger format does not currently carry that information, consider adding it later.

- **Minimum safer tweak:**
  - Do not enter tier 3 if the exact ledger file exists but failed to load.
  - Only fuzzy scan when the expected ledger path is absent.

---

## 4. Why did the prior round-robin miss this?

- **Because the interface looked semantically identical but the implementation was not.**
  - The workflow said `ledger_json`.
  - HuMo had `ledger_json`.
  - LTX had `ledger_json`.
  - VideoComposite had `ledger_json`.
  - All are `STRING`.
  - Reviewers inferred that “same input name” meant “same accepted values.”
  - But `BatchLTXRender._load_ledger()` had diverged internally.

- **The workflow JSON alone did not expose the bug.**
  - Link `90` was visibly:

    ```text
    OTR_SignalLostVideo.video_path -> OTR_BatchLTXRender.ledger_json
    ```

  - But whether that is valid depends on the internals of `BatchLTXRender._load_ledger()`.
  - HuMo and VideoComposite made that edge pattern seem safe because links `79` and `82` worked with the same source.

- **This is a “semantic contract drift” bug.**
  - Same interface label.
  - Same type.
  - Same upstream source.
  - Different hidden parser/resolver behavior.

- **Heuristics that would catch this class next time:**
  - **Semantic twin audit**
    - Grep for every resolver-like method:

      ```text
      _load_ledger
      _load_ledger_with_path
      ledger_json
      ```

    - If multiple nodes accept `ledger_json`, verify they share one helper or have explicit documented differences.
  - **Workflow edge contract lint**
    - Detect edges where a source named `video_path` feeds a destination named `ledger_json`.
    - Require the destination node to declare that it accepts `.mp4` ledger references.
  - **No-GPU resolver parity tests**
    - For every node with `ledger_json`, run the same fixture table:
      - inline JSON object,
      - inline JSON array if legacy-supported,
      - explicit `*_ledger.json`,
      - `.mp4` with exact ledger,
      - `.mp4` with double-underscore mismatch,
      - missing file,
      - malformed exact ledger,
      - multiple fuzzy candidates.
  - **Contract test from real workflow**
    - Parse `workflows/otr_scifi_16gb_full.json`.
    - For every edge from `OTR_SignalLostVideo.0`, test the receiving node’s resolver against a fake `.mp4` path.
  - **Shared resolver requirement**
    - If two nodes have an input named `ledger_json`, they should call the same resolver unless there is a strong reason not to.

- **The offline resolver test you ran is exactly the right kind of test.**
  - I would make that permanent.
  - It is cheap, deterministic, and catches this before any GPU work.

---

## 5. Is there a deeper architectural fix?

- **Yes: a shared resolver module is the right maintainability fix.**
  - You now have at least three consumers of the same semantic input:
    - `nodes/batch_humo_render.py`
    - `nodes/batch_ltx_render.py`
    - `nodes/video_composite.py`
  - That is enough duplication to justify centralizing.

- **Suggested module name:**

  ```text
  nodes/_otr_ledger_resolver.py
  ```

  Or extend:

  ```text
  nodes/_otr_ledger.py
  ```

  if that file is already the canonical ledger utility module.

- **Suggested API shape:**

  ```python
  def resolve_ledger_arg(
      arg: str,
      *,
      caller: str,
      allow_mp4: bool = True,
      allow_inline: bool = True,
      allow_auto_pick: bool = True,
      allow_fuzzy: bool = False,
      require_dict: bool = True,
  ) -> tuple[dict | None, Path | None]:
      ...
  ```

- **Default `allow_fuzzy=False` would be my preference.**
  - Then HuMo can explicitly opt into the BUG-LOCAL-118 fuzzy behavior if you still want it there.
  - LTX can use exact + collapsed-underscore only unless you explicitly enable fuzzy.
  - VideoComposite can choose based on how risky wrong binding would be.

- **The shared resolver should use `_otr_ledger.load_ledger_safe()` if available.**
  - This avoids losing central validation.
  - The resolver should be responsible for path discovery.
  - `_otr_ledger` should remain responsible for safe ledger loading/validation.

- **Worth doing now?**
  - My recommendation:
    - **For immediate production unblock:** keep the current LTX hotfix, with the small safety tweaks above if convenient.
    - **Immediately after the successful run:** refactor to shared resolver with tests.
  - I would not do a broad DRY refactor right before another 40+ minute GPU run unless you already have tests covering HuMo, LTX, and VideoComposite behavior.

- **Why not wait for a fourth node?**
  - Three nodes is already enough.
  - This bug happened precisely because the third node had a divergent copy.
  - Waiting for a fourth node just increases the chance of another hidden contract drift.

---

## 6. Any concern with the new `humo_clips_dir` sequencing widget?

- **Using a linked input purely as a DAG sequencing edge is acceptable in ComfyUI.**
  - It is not elegant, but it is a common practical pattern.
  - In this workflow, sequencing matters because HuMo must fully finish and tear down before LTX claims VRAM.
  - Given the 16 GB VRAM ceiling and your desire to avoid low-level VRAM work, explicit graph ordering is the right kind of fix.

- **One correction/clarification from the provided workflow:**
  - In the subgraph you attached:
    - `BatchHumoRender.report` output `51.2` feeds `LowVRAMCheckpointLoader.dependencies` via link `86`.
    - `BatchHumoRender.clips_dir` output `51.0` feeds `BatchLTXRender.humo_clips_dir` via link `91`.
  - So the `humo_clips_dir` widget is not wired from `report` in the shown JSON; it is wired from `clips_dir`.
  - That is actually cleaner.

- **I would keep it, but document it.**
  - Rename/comment it as a dependency gate if possible:

    ```text
    humo_done_gate
    humo_clips_dir_dependency
    dependencies
    ```

  - Or add a code comment in `nodes/batch_ltx_render.py` explaining:

    ```python
    # Intentionally consumed as a DAG dependency so ComfyUI schedules
    # LTX after HuMo teardown. The value is not used by render logic.
    ```

- **Potential anti-pattern risk: future cleanup.**
  - A future maintainer may see an ignored input and remove it.
  - If they remove it, ComfyUI may again schedule/load LTX too early.
  - So the risk is not runtime behavior now; the risk is maintainability.

- **Consider lightly validating it.**
  - Since `humo_clips_dir` is linked to `BatchHumoRender.clips_dir`, you could cheaply check/log:

    ```python
    if humo_clips_dir and not Path(humo_clips_dir).exists():
        log.warning("[BatchLTXRender] HuMo clips_dir dependency path does not exist: %s", humo_clips_dir)
    ```

  - I would not make this fatal unless LTX truly needs those clips.
  - If it is strictly sequencing-only, do not introduce a new hard failure.

- **The LowVRAM loader `dependencies` input is the more important sequencing edge.**
  - Link `86`:

    ```text
    BatchHumoRender.report -> LowVRAMCheckpointLoader.dependencies
    ```

  - This is the edge that forces the LTX checkpoint loader to wait until HuMo reports completion.
  - Link `91`:

    ```text
    BatchHumoRender.clips_dir -> BatchLTXRender.humo_clips_dir
    ```

  - This additionally keeps the render node ordered behind HuMo.

---

## My recommended final adjustment before moving on

- I would make only these small changes before another production run:

  - **Add/confirm `import time`** in `nodes/batch_ltx_render.py` if tier 3 remains.

  - **Use `_otr_ledger.load_ledger_safe()` when available** instead of direct `json.load()` for file-path ledgers.

  - **Do not fuzzy-fallback if an exact candidate exists but fails to load.**
    - Exact path malformed should fail.
    - It should not silently bind to a nearby recent ledger.

  - **Document `humo_clips_dir` as a sequencing dependency** so it does not get “cleaned up” later.

- With those tweaks, I would call the fix sound and move on.

---

## Gemini (gemini-3.1-pro-preview-customtools)

Here is your second opinion.

### 1. Core Recommendation
**PARTIALLY AGREE** with ChatGPT's assessment that the logic is tactically correct, but I **DISAGREE** with accepting the proposed code as-is; you should reject commit `5ebc94b` because it introduces dangerous exception-swallowing and likely `NameError` crashes.

### 2. Factual Errors in the ChatGPT Answer
*   **Missed `NameError` on `log`:** ChatGPT caught the missing `import time` but completely missed that `log.warning(...)` is used extensively in the proposed code. If `log` is not imported at the top of `batch_ltx_render.py` (e.g., `import logging; log = logging.getLogger(__name__)`), any triggered fallback or empty auto-pick will raise a `NameError` and crash the node anyway.
*   **Understated the severity of the fall-through:** ChatGPT called it a "tiny safety tweak" to not fuzzy-fallback if an exact ledger fails to parse. This is actually a critical flaw. The proposed code wraps `_json.load(f)` in a broad `except Exception as exc:` and continues to the next tier. 

### 3. Important Things That Were Omitted
*   **Windows File Locking:** You are running on a Windows workstation. If `EpisodeAssembler` or another process still has a lock on `episode_ledger.json`, or if an antivirus scan touches it, `open()` will raise a `PermissionError`. Under your proposed code, this `PermissionError` is caught, swallowed, and the code silently falls through to Tier 3, binding LTX to a completely different, older radio episode. This violates the strict determinism required by your architecture.
*   **Schema Migrations in `_OTRL`:** ChatGPT noted you lose validation by dropping `_OTRL.load_ledger_safe()`, but omitted that you likely lose *schema migrations*. If your ledger format evolved over the last few weeks, `load_ledger_safe` is likely injecting default keys that `BatchLTXRender` expects. Bypassing it with a raw `json.load()` is a high risk for downstream `KeyError`s.
*   **ComfyUI Execution Model vs. Exceptions:** In ComfyUI, it is almost always better to let a node crash loudly (`raise RuntimeError`) than to silently guess wrong. A crash halts the queue and preserves the VRAM state for debugging. Silent fuzzy-matching wastes 40+ minutes of GPU time generating the wrong video.

### 4. My Short Recommendation
Reject the proposed code and implement a much tighter, safer hotfix for this production run:

*   **Restore `_OTRL.load_ledger_safe`:** Do not use raw `json.load()`. Keep the original file's import of `_otr_ledger` and use it to load the resolved path.
*   **Implement Tier 1 and Tier 2 ONLY:** Add the `.mp4` -> `.json` exact match (Tier 1) and the collapsed-underscore match (Tier 2). 
*   **Kill Tier 3 (Fuzzy Scan) for LTX:** Time-based fuzzy matching is non-deterministic. If the exact or underscore-collapsed ledger isn't there, raise a `RuntimeError` immediately.
*   **Remove the `try/except` blocks around file loading:** If the file exists but throws `PermissionError` or `JSONDecodeError`, let it crash the ComfyUI run. Do not fall through.
*   **Keep the `humo_clips_dir` widget:** It is a perfectly valid ComfyUI DAG sequencing edge. As long as HuMo's internal teardown is as strict as you say (clearing 16.5 GB), this edge guarantees LTX won't load into VRAM until HuMo is done.

### 5. Uncertainties I Would Want to Verify
*   **Is `log` defined globally in `batch_ltx_render.py`?** If not, your proposed code is a ticking time bomb for `NameError`.
*   **What exactly does `_OTRL.load_ledger_safe()` do?** I am assuming it handles legacy field normalization. If it *only* does a `json.load()`, dropping it is harmless, but I wouldn't bet a 1-hour render on that assumption.
*   **Does `OTR_UnloadAll` (Node 24) need to be duplicated between HuMo and LTX?** You rely on HuMo's strict teardown, but ComfyUI's VRAM manager can be stubborn. If LTX OOMs on the next run, I would verify if an explicit `empty_cache()` or unload node is needed in that specific gap.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### Third Opinion: Breaking the Tie on BUG-LOCAL-011 Fix

---

#### **1. Where ChatGPT and Gemini AGREE**
- **Core Fix Correctness**: Both agree the proposed code addresses the immediate BUG-LOCAL-011 failure (`.mp4` → `_ledger.json` resolution gap).  
  - **Concur**: Yes, but with Gemini’s safety caveats. The fix is *tactically necessary* but *not production-ready* as written.  
  - **Flaw Missed**: Neither addressed whether `BatchLTXRender` *actually needs* the ledger data beyond resolution. If LTX only uses a subset of ledger fields, schema validation in `_OTRL.load_ledger_safe()` might be critical (Gemini’s point about `KeyError` risks).

---

#### **2. Where They DISAGREE**
- **Proposed Code Safety**  
  - **ChatGPT**: Accept with tweaks (preserve `_OTRL`, tighten fallbacks).  
  - **Gemini**: Reject outright due to `NameError` risks and silent failures.  
  - **Third Opinion**: **Side with Gemini**. The proposed code’s broad `except Exception` blocks and missing `log`/`time` imports make it unsafe for production. A minimal, explicit fix is better.  

- **Tier 3 Fuzzy Scan**  
  - **ChatGPT**: Keep with validation.  
  - **Gemini**: Kill it for LTX.  
  - **Third Opinion**: **Agree with Gemini**. Tier 3 introduces non-determinism (violating C7 spirit) and silent fallback risks. LTX should fail fast if exact/collapsed ledger resolution fails.

---

#### **3. Factual Errors in Prior Answers**
- **ChatGPT**:  
  - Missed that `log.warning()` is used extensively in the proposed code without confirming `log` is imported.  
  - Understated the risk of silent failures in fuzzy fallback (Gemini correctly flags this as a critical flaw).  
- **Gemini**:  
  - Correctly identified `NameError` risks for `log` and `time`.  
  - Overstated the certainty about `_OTRL.load_ledger_safe()` behavior (uncertainty is fair here).  

---

#### **4. Important Omissions by Both**
- **VRAM Sequencing Guarantees**:  
  - Both assumed `humo_clips_dir` and `LowVRAMCheckpointLoader.dependencies` ensure HuMo teardown completes before LTX loads.  
  - **Risk**: If ComfyUI’s scheduler bypasses this (e.g., due to node prioritization), LTX could still OOM. No mention of testing this edge.  
- **Schema Validation Impact**:  
  - If `_OTRL.load_ledger_safe()` injects default values or migrates legacy fields, dropping it could break LTX downstream.  
- **Windows File Locking**:  
  - Gemini raised this, but neither proposed a fix (e.g., retry logic for `PermissionError`).

---

#### **5. Prioritized Recommendation**
1. **Reject the proposed `5ebc94b` commit** due to:  
   - Missing `log` import (risk of `NameError`).  
   - Overly broad `except Exception` blocks that hide errors.  
   - Non-deterministic Tier 3 fallback.  

2. **Implement a Minimal Hotfix**:  
   - **Port only Tier 1 + Tier 2** from `BatchHumoRender._load_ledger_with_path` (exact `.mp4` → `_ledger.json` and collapsed-underscore).  
   - **Use `_OTRL.load_ledger_safe()`** for loading resolved paths (preserve schema validation/legacy fixes).  
   - **Remove Tier 3** (fuzzy scan).  

3. **Ensure Imports**:  
   - Add `import logging; log = logging.getLogger(__name__)` and `import time` to `batch_ltx_render.py`.  

4. **Fail Fast on Errors**:  
   - Remove `try/except` around file loading. Let `PermissionError`/`JSONDecodeError` crash the run explicitly.  

5. **Document `humo_clips_dir`**:  
   - Add a comment in `batch_ltx_render.py` explaining it’s a sequencing edge, not used for data.  

6. **Post-Production Refactor**:  
   - Centralize ledger resolution in `_otr_ledger_resolver.py` with configurable fallbacks (e.g., `allow_fuzzy=False` by default).  

---

#### **6. Uncertainties Requiring Verification**
- **What does `_OTRL.load_ledger_safe()` do?**  
  - If it only wraps `json.load()`, dropping it is safe.  
  - If it migrates fields or validates schemas, it’s critical.  
  - **Action**: Review `nodes/_otr_ledger.py` to audit its behavior.  

- **Does `BatchLTXRender` need HuMo’s strict VRAM teardown?**  
  - If LTX OOMs in next run, consider duplicating `OTR_UnloadAll` before LTX loads.  

- **Are there existing ledger schema migrations?**  
  - Check if `load_ledger_safe()` injects defaults (e.g., for older ledgers missing LTX-specific fields).  

---

### **Final Verdict**  
The proposed fix is **not safe for production** in its current form. A minimal, explicit resolver with Tier 1 + Tier 2, using `_OTRL.load_ledger_safe()`, and strict error handling should be implemented instead. Tier 3 fuzzy resolution is too risky for LTX. Prioritize determinism and explicit failures over silent recovery.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
