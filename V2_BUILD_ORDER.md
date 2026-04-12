# V2.0 BUILD ORDER — CLAUDE CODE EXECUTION DOC

```yaml
target: rtx5080_blackwell_16gb
os: windows_11
cuda: "13.8"
torch: ">=2.8"
framework: comfyui
consumer: claude_code
mode: imperative
supersedes: [V2_ANIMATION_SPEC.md, V2_ANIMATION_ACTION_PLAN.md]
```

---

## DOCTRINE

1. Audio = timeline truth. Visual conforms to audio, never inverse.
2. One heavy model resident at a time. Enforced by `MemoryBoundary`.
3. SAFE MODE ships first. EXPERIMENT MODE gated behind numeric validation.
4. Dereference ≠ VRAM release. Every transition passes the boundary.
5. `SceneSegmenter` owns scene *count*. Director is *advisory* for scene *meaning*.
6. All network I/O and prompt assembly complete **before** ComfyUI queue starts.
7. Disk writes never overlap `MemoryBoundary` windows.

---

## FILE TREE (CREATE EXACTLY)

```
v2/
  v2_preview.py                          # CLI entrypoint (existing; extend)
  config/
    rtx5080.yaml
  orchestrator/
    story_orchestrator.py                # existing
    scene_segmenter.py                   # NEW
    production_bus.py                    # NEW
    prompt_builder.py                    # NEW
    director_reconciler.py               # NEW
  nodes/
    memory_boundary.py                   # NEW
    vram_guard.py                        # NEW
    scene_animator.py                    # existing; harden
    signal_lost_video.py                 # NEW
    disk_writer.py                       # NEW (serialized queue)
  assets/
    signal_lost_prerender.mp4            # pre-baked, required
    embeddings/
      manifest.json
  scripts/
    preflight.py
    overnight_run.py
    train_embedding.py                   # out-of-band
  logs/
    vram_peaks.jsonl
    ledger.jsonl
    run_<ts>.json
```

---

## CONFIG: `config/rtx5080.yaml`

```yaml
mode: safe                     # safe | experiment

boundaries:
  llm_to_audio_s: 1.2
  audio_to_visual_s: 1.8
  forge_to_painter_s: 1.8
  sd_to_ltx_s: 2.3
  drift_flush_every_n_scenes: 10
  drift_flush_s: 3.0

vram:
  watermark_gb: 14.3
  peak_fail_gb: 14.5

image:
  model: sd3.5_medium_gguf_q8_0
  loader: ComfyUI-GGUF
  resolution: [1024, 1024]

video:                         # EXPERIMENT only
  model: ltx-2.3-22b-distilled-fp8
  loader: ComfyUI-LTXVideo
  precision: fp8_e4m3fn        # e5m2 → hard-fail
  resolution: [768, 512]
  fps: 25
  duration_s_range: [2.0, 4.0]

audio:
  master_sample_rate: 48000
  master_bit_depth: 24
  resample_method: sox_vhq

scenes:
  segmenter_trigger_lines: 8
  lines_per_scene_when_chunking: 4

reconcile:
  av_duration_policy: hold_last_frame   # hold_last_frame|loop|crossfade_black|timestretch
  av_tolerance_ms: 150
  av_drift_fail_ms: 500

fallback:
  clip: assets/signal_lost_prerender.mp4
  max_per_episode: 2

ffmpeg:
  static_clip_timeout_s: 30
  concat_timeout_s: 120

comfyui_required_args:
  - --highvram
  - --force-fp16
  - --cuda-malloc
```

---

## TASKS (STRICT ORDER)

### T1 — `config/rtx5080.yaml`
**Create** the YAML above verbatim.
**Accept:** `yaml.safe_load()` parses; all keys accessible.

---

### T2 — `nodes/memory_boundary.py`
```python
# Signature (exact)
def memory_boundary(sleep_s: float, label: str) -> dict: ...
```
**Behavior (in order):**
1. `pre_gb = torch.cuda.memory_allocated() / 1e9`
2. `comfy.model_management.unload_all_models()`
3. `comfy.model_management.soft_empty_cache()`
4. `torch.cuda.empty_cache()`
5. `torch.cuda.synchronize()`
6. `time.sleep(sleep_s)`
7. `post_gb = torch.cuda.memory_allocated() / 1e9`
8. Append `{ts, label, pre_gb, post_gb, sleep_s}` to `logs/vram_peaks.jsonl`
9. Return that dict.

**ComfyUI node:** expose with a **required passthrough input** so the scheduler treats it as a hard dependency. Add header comment:
```
# This passthrough enforces ComfyUI execution order.
# Removing it produces non-deterministic OOMs on Blackwell. Do not remove.
```
**Accept:** called with `(2.3, "SD->LTX")` after a SD3.5 load → `post_gb < 3.0` on hardware.

---

### T3 — `nodes/vram_guard.py`
```python
class VRAMWatermarkExceeded(RuntimeError): ...
def vram_guard(threshold_gb: float, label: str) -> None: ...
```
**Behavior:** if `torch.cuda.memory_allocated() / 1e9 > threshold_gb`, raise `VRAMWatermarkExceeded(f"{label}: {allocated:.2f} > {threshold_gb}")`.
**Call sites:** immediately before every SD3.5, Flux, and LTX load.

---

### T4 — `nodes/signal_lost_video.py`
```python
def signal_lost_clip(duration_s: float, out_path: str) -> str: ...
```
**Behavior:** ffmpeg-loop `assets/signal_lost_prerender.mp4` to `duration_s`, write to `out_path`. Never loads any model.
**Preflight:** if asset missing → hard-fail run start.
**Accept:** returns path to a valid mp4 of requested duration ±50 ms.

---

### T5 — `nodes/disk_writer.py`
Single-threaded queue for all mp4 writes. Writes executed only when no `MemoryBoundary` is in progress (shared `threading.Event`).
```python
class DiskWriter:
    def enqueue(self, src_frames_or_path, dest_path, duration_s): ...
    def drain(self, timeout_s: float = 300) -> list[str]: ...
```
**Accept:** boundary and write never overlap; verified by interleaved log timestamps.

---

### T6 — `orchestrator/scene_segmenter.py`
```python
@dataclass
class Scene:
    scene_id: str         # s01, s02, ...
    line_indices: list[int]
    dialogue: list[str]

def segment(lines: list[str]) -> list[Scene]: ...
```
**Rule:** if `len(lines) > 8` → chunk every 4 lines. Else single scene.
**Accept:** deterministic across runs. Scene IDs zero-padded, stable.

---

### T7 — `orchestrator/director_reconciler.py`
```python
def reconcile(director_scenes: list[dict], seg_scenes: list[Scene]) -> list[dict]: ...
```
**Algorithm:** for each Segmenter scene, pick the Director scene with max Jaccard overlap on `line_indices`. Tie → earlier Director scene. Log divergence to `logs/run_<ts>.json`.
**Segmenter count wins.** Always.

---

### T8 — `orchestrator/prompt_builder.py`
Produces fully-materialized prompt matrix before any queue submission.
```python
@dataclass
class ScenePrompt:
    scene_id: str
    anchor_prompt: str
    motion_prompt: str
    motion: Literal["static","low","medium","high"]
    duration_s: float
    character_tokens: list[str]
    director_hint: str | None

def build(scenes: list[Scene], director_json: dict, audio_timings: dict) -> list[ScenePrompt]: ...
```
**Assertions at exit:** every field populated, all strings, no `None` except `director_hint`, no coroutines/futures.

---

### T9 — `orchestrator/production_bus.py`
```python
class ProductionBus:
    def __init__(self, config: dict, mode: Literal["keyframes","animated"]): ...
    def render_scene(self, prompt: ScenePrompt, anchor_path: str, audio_duration_s: float) -> str: ...
    def concat(self, clip_paths: list[str], out_path: str) -> str: ...
```
**Behavior (keyframes mode):** ffmpeg static clip from anchor, length = `audio_duration_s`.
**Behavior (animated mode):** delegate to `SceneAnimator`, then reconcile duration per `reconcile.av_duration_policy`.
**Drift handling:** if `abs(actual - target) > av_drift_fail_ms` → fallback clip, log, increment fallback counter.
**Fallback budget:** if `fallback_count > fallback.max_per_episode` → raise `EpisodeFallbackBudgetExceeded`.

---

### T10 — `nodes/scene_animator.py` (harden existing)
Add entry guards:
1. `if config.mode != "experiment": raise RuntimeError`
2. Validate `precision == "fp8_e4m3fn"` → else hard-fail
3. Validate model string == `ltx-2.3-22b-distilled-fp8` → else hard-fail
4. Validate resolution == `[768, 512]` → else hard-fail
5. Call `memory_boundary(2.3, "SD->LTX")`
6. Call `vram_guard(14.3, "pre-LTX")`
7. Load LTX, render, return clip path

Add `--single-scene=<id>` short-circuit in `v2_preview.py`.

---

### T11 — `scripts/preflight.py`
Checks before any run:
- ComfyUI started with all `comfyui_required_args` → else refuse.
- `assets/signal_lost_prerender.mp4` exists.
- `assets/embeddings/manifest.json` valid; for each entry, file exists and sha256 matches.
- Missing embeddings for referenced tokens → warn, substitute neutral descriptor, continue.
- `nvidia-smi` idle VRAM < 2.5 GB → else warn.
- All referenced models present on disk.

Exit nonzero on any hard-fail.

---

### T12 — `scripts/overnight_run.py`
```bash
python scripts/overnight_run.py --mode=safe --episode=<path> [--single-scene=<id>] [--dry-run]
```
**Order:**
1. `preflight.py`
2. Parse script → Director LLM → Segmenter → Reconciler → PromptBuilder → fully-materialized matrix
3. If `--dry-run`: dump `dry_run_plan.json`, exit 0
4. Audio pass (Bark/Kokoro/SFX/Music) → SceneSequencer → EpisodeAssembler
5. Visual pass per scene: CharacterForge → boundary → ScenePainter → boundary → VisualCompositor → ProductionBus
6. Final concat
7. Write `logs/run_<ts>.json` + append `logs/ledger.jsonl`

**Ledger line:**
```json
{"run_id":"...","ts":"...","git_sha":"...","config_sha":"...","seed":123,
 "mode":"safe","episode_id":"...","scene_count":14,"total_s":512.3,
 "fallback_count":0,"peak_vram_gb":13.1,"completed":true}
```

Exit nonzero if `completed: false`.

---

### T13 — `scripts/train_embedding.py`
Out-of-band. Trains a Textual Inversion `.safetensors` for a character. Updates `assets/embeddings/manifest.json` with `{token, path, sha256, trained_on_model, trained_at}`. **Never called during a run.**

---

## PROMOTION GATES (NUMERIC, NON-NEGOTIABLE)

| Gate | Criteria |
|---|---|
| **A — SAFE stable** | 3 consecutive overnight SAFE runs, 0 fallbacks, no phase peak > 13.5 GB |
| **B — EXPERIMENT single** | 10 consecutive single-scene LTX renders, 0 OOM, every peak < 14.3 GB |
| **C — EXPERIMENT full** | 1 full episode: ≥ 12 scenes, ≥ 8 min runtime, ≤ 1 fallback, 0 `VRAMWatermarkExceeded` |

Do not skip. Do not collapse.

---

## FAILURE MATRIX

| Failure | Trigger | Response |
|---|---|---|
| `VRAMWatermarkExceeded` | pre-load guard | fallback clip, log, continue |
| CUDA OOM mid-render | allocator drift | `memory_boundary(3.0)`, retry once, else fallback |
| `EpisodeFallbackBudgetExceeded` | > 2 fallbacks | abort run, exit nonzero |
| Director JSON malformed | LLM drift | Segmenter-only, generic prompts, log, continue |
| RSS/network stall | upstream | caught in preflight; abort before queue |
| Missing fallback asset | deploy | refuse to start |
| e5m2 precision | config drift | hard-fail at LTX load |
| FFmpeg timeout | zombie/handle lock | kill, fallback, log |
| AV drift > 500 ms | model misbehavior | fallback clip for that scene |

---

## FORBIDDEN IN V2.0

- IP-Adapter (resident cost kills budget)
- Hunyuan co-residency (offline depth-map only allowed)
- LTX 19B variant (face temporal drift)
- FP16 LTX
- Concurrent upscale in render pass
- `subprocess.Popen` without `finally` cleanup
- Lazy prompt assembly (any network I/O after queue start)
- Removing the `MemoryBoundary` passthrough input

---

## DEFINITION OF DONE — V2.0 RC1

- T1–T13 merged
- Gates A and B passed
- `overnight_run.py --mode=safe` produces complete episode unattended
- `overnight_run.py --mode=experiment --single-scene` produces one animated scene unattended
- `logs/vram_peaks.jsonl` shows 0 phases > 14.3 GB across full SAFE run
- `logs/ledger.jsonl` has ≥ 3 `completed: true` SAFE entries

---

**BEGIN AT T1. EXECUTE IN ORDER. DO NOT REORDER.**
