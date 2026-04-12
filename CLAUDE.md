# OldTimeRadio — Project Rules

## Branch & Shipping

- All v2 work on `v2.0-alpha`. Do not touch `main`.
- Only Jeffrey merges to `main` and tags releases.

## Git Push

1. Try push once. If it fails, hand Jeffrey a PowerShell block with `cd` included:
   ```
   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   git add -A && git commit -m "message" && git push
   ```
2. After every push, verify lockstep (`local HEAD == origin HEAD`), no 0-byte files, no BOM, all node classes registered in `__init__.py`, workflow JSONs valid.

## Testing

Run all three before committing:

```bash
# Core + VRAM
pytest tests/test_core.py tests/vram_profile_test.py -v

# Bug Bible regression
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v --pack-dir .

# v2 audio regression (Phase 0+)
pytest tests/v2/test_audio_byte_identical.py -v
```

**VRAM ceiling:** 14.5 GB peak. Never use `force_vram_offload()` between LLM phases — use `_flush_vram_keep_llm()`. Always enforce prompt truncation against `context_cap`. All LLM loaders must do a 1-token warmup pass.

## Bug Log Pipeline

Log every bug immediately in `BUG_LOG.md`:

```markdown
### BUG-LOCAL-NNN: Title
- **Date:** YYYY-MM-DD | **Phase:** 0-6 | **Bible candidate:** yes/no
- **Symptom:** exact error/behavior
- **Cause:** root cause
- **Fix:** what resolved it
- **Verify:** how to confirm
- **Tags:** vram, widget-drift, ffmpeg, subprocess, etc.
```

Mark `[FIXED]` when resolved — don't delete entries. When `Bible candidate: yes` and fix is verified, promote to the survival guide repo (`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`):
1. Add entry to `BUG_BIBLE.yaml` (schema: `id`, `phase`, `area`, `symptom`, `cause`, `fix`, `verify`, `tags`, `legacy_id`)
2. Add regression test to `tests/bug_bible_regression.py`
3. Update `README.md` entry count
4. Run three-file contract test to confirm sync

## Content Standards

Safe for work. No profanity. Good narrative arc (beginning, middle, end). Non-violent.

## References

| Resource | Location |
|----------|----------|
| OTR Repo | https://github.com/jbrick2070/ComfyUI-OldTimeRadio |
| Bug Bible | https://github.com/jbrick2070/comfyui-custom-node-survival-guide |
| v2 Design Spec | `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md` |

---

## v2.0 Constraints

**Audio is king. Full narrative output must never break, shorten, or degrade. If video breaks audio, revert immediately.**

| ID | Rule |
|----|------|
| C1 | **No new inputs** on any v1.5 node (`OTR_BatchBarkGenerator`, `OTR_SceneSequencer`, `OTR_KokoroAnnouncer`, `OTR_AudioEnhance`, `OTR_EpisodeAssembler`, `OTR_MusicGenTheme`, `OTR_BatchAudioGenGenerator`, `OTR_Gemma4ScriptWriter`, `OTR_Gemma4Director`). Shifts `widgets_values` indices, silently corrupts seeds/voices. |
| C2 | **No `CheckpointLoaderSimple`** or stock diffusion nodes in the main graph. OOM on 16 GB. |
| C3 | All visual generation in **subprocesses** via `multiprocessing.get_context("spawn")`. |
| C4 | LTX-2.3 clips **max 10-12 s** (257 frames @ 24fps). Auto-chunk + ffmpeg crossfade. |
| C5 | LTX-2.3 uses `torch.float8_e4m3fn` (Blackwell-native). |
| C6 | IP-Adapter for **environments only**, never characters (Silent Lip Bug). |
| C7 | Audio output **byte-identical** to v1.5 baseline at every gate. |

**Only legal v1.5 modification:** `OTR_SignalLostVideo` (#12) gets one optional `visual_overlay` input (STRING, last slot). Byte-identical when unwired.

**Hardware:** RTX 5080, 16 GB VRAM, Blackwell, single GPU, no cloud.

**Previous attempt failed** (`v2.0-visual-engine`, deleted) by modifying node inputs (widget drift) and loading diffusion in-process (OOM). This sidecar architecture prevents both.
