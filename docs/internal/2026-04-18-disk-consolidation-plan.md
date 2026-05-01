# ComfyUI Disk Consolidation Plan — 2026-04-18

**Total ComfyUI footprint: ~505 GB**
**Reclaimable without risk: ~165 GB**
**Reclaimable with your OK: +90 GB**

---

## The picture

| Folder | Size | Notes |
|---|---:|---|
| `models/checkpoints/` | 255 GB | LTX 19B/22B variants alone = 118 GB and won't fit 16 GB VRAM |
| `models/diffusers/` | 104 GB | FLUX (54), Wan2.1 (27), SDXL inpaint (19), LTX cache (3) — all wired |
| `models/huggingface/` | 50 GB | LLM cache; partly duplicated in `~/.cache/huggingface/` |
| `output/` | 38 GB | Old episode renders |
| `models/text_encoders/` | 19 GB | t5xxl + gemma fp4; both used |
| `models/loras/` | 7 GB | keep |
| `.venv/` | 6 GB | keep |
| `custom_nodes/` | 6 GB | mostly `custom_nodes/models/` (5 GB) — orphan |
| `models/controlnet/` | 7 GB | keep |
| `models/WorldMirror-V2/` | 4.7 GB | **retired per memory** |
| `models/pulid/` | 3.2 GB | three copies of the same file |

---

## Bucket 1 — Safe to delete now (~165 GB)

These are clear cases. No active code path uses them and they exceed our 16 GB VRAM ceiling.

### LTX 19B/22B checkpoints — 118 GB
The video stack target is the 8.7 GB **2B v0.9** model (`ltx-video-2b-v0.9.safetensors`). The 19B/22B variants cannot fit on a 16 GB GPU even at fp8:

```
40.31 GB  models/checkpoints/ltx-2-19b-distilled.safetensors
27.50 GB  models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors
25.22 GB  models/checkpoints/ltx-2-19b-distilled-fp8.safetensors
25.22 GB  models/checkpoints/ltx-2-19b-dev-fp8.safetensors
```

### WorldMirror — KEEP (updated 2026-04-19)
**Jeffrey confirmed HY-World 2.0 is active in a different project.** Do NOT delete:

```
4.71 GB  models/WorldMirror-V2/                                             (KEEP)
4.71 GB  C:\Users\jeffr\.cache\huggingface\hub\models--tencent--HunyuanWorld-Mirror\  (KEEP)
```

Earlier memory `project_v17_solid_pivot_hyworld.md` said HY-World was retired — that applies only to the OTR v2 sidecar architecture, not to Jeffrey's other projects.

### Duplicate LLM caches — 23 GB
These exist in BOTH `models/huggingface/hub/` AND `~/.cache/huggingface/hub/`. The `models/huggingface/` copies are the active ones (your custom-node loaders point there). Drop the user-cache duplicates:

```
14.92 GB  ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/
 4.18 GB  ~/.cache/huggingface/hub/models--suno--bark/
 4.18 GB  ~/.cache/huggingface/hub/models--facebook--musicgen-medium/  (subset duplicate)
```

(The full musicgen-medium download is in user cache at 14.98 GB; the smaller 7.49 GB version in `models/huggingface/` is the partial. Decide: keep one, drop the other. Default plan = keep `models/huggingface/` copy, drop `~/.cache/` copy → reclaims 14.98 GB.)

### Old PuLID versions — 2.1 GB
Three identical-size files; only `v0.9.1` is the current one:

```
1.06 GB  models/pulid/pulid_flux_v0.9.0.safetensors
1.06 GB  models/pulid/pulid_flux.safetensors  (untagged — likely same as v0.9.1)
```

### Bucket 1 total: ~165 GB

---

## Bucket 2 — Needs your OK (~90 GB)

### `models/checkpoints/` — unused diffusion bases (18 GB)
Not wired into the OTR pipeline today, but you may want them for one-off experiments outside the main render:

```
13.91 GB  sd3.5_large_fp8_scaled.safetensors    (SD 3.5 — never used in OTR)
 3.97 GB  v1-5-pruned-emaonly.ckpt              (SD 1.5 — pivoted away from)
```

### `output/` — old renders (38 GB)
Old episode MP4s, scratch frames, intermediate WAVs. I can:
- (a) move them to a dated archive folder so they're out of the way but recoverable, or
- (b) delete only files older than N days (you pick N), or
- (c) leave alone.

### `custom_nodes/models/` — orphan model dir (5 GB)
Looks like an old custom-node download path that's no longer the convention. Worth inspecting before deleting — could be Florence-2 or PuLID weights misplaced.

### Stale custom nodes (0.2 GB, but cleans the load list)
- `custom_nodes/ComfyUI-HYRadio/` — pre-OTR HY-World node, retired

### Bucket 3 — keep, don't touch
- All `models/diffusers/` (FLUX, Wan2.1, SDXL inpaint, LTX-Video cache) — actively wired
- `models/text_encoders/` — both t5xxl_fp16 and gemma_3_12B fp4 are loaded
- `models/controlnet/`, `models/loras/`, `models/florence2/` — wired
- `models/huggingface/hub/models--google--gemma-4-E4B-it/` — primary LLM
- `.venv/` — your Python env

---

## What I propose

1. **You review this plan.** No deletes happen until you say go.
2. **You pick the bucket-2 calls** (SD 3.5? old output? custom_nodes/models?).
3. **I write `scripts/automation/consolidate_disk.ps1`** with explicit per-target functions, dry-run mode default, and a `-Confirm` flag to actually delete. You'd run it as:
   ```powershell
   # See exactly what would happen, no changes:
   powershell -File scripts\automation\consolidate_disk.ps1
   # Actually delete after review:
   powershell -File scripts\automation\consolidate_disk.ps1 -Confirm
   ```
4. **I add `HF_HOME` env var fix** to stop the duplicate-cache problem from recurring. One canonical location, future downloads land there.

---

## Why the duplicates happened

`HF_HOME` is unset, so `huggingface_hub` defaults to `~/.cache/huggingface/`. But several of your custom nodes hardcode `models/huggingface/` as their cache. So the same weight gets pulled twice — once when ComfyUI auto-loads, once when a script runs. Fix: set `HF_HOME=C:\Users\jeffr\Documents\ComfyUI\models\huggingface` in user env vars, then nothing duplicates going forward.
