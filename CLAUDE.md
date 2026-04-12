# OldTimeRadio — Rules of the Road

These are the permanent operating rules for any AI assistant working on this project.
Violating these rules is grounds for immediate rollback.

---

## 1. Branching and Versioning

- **All v2 work lives on `v2.0-alpha`**, never on `main`, until Jeffrey says ship.
- v1.5 is shipped and stable on `main`. Do not modify `main`.
- **No changelogs until v2.0 ships.** Keep the `ROADMAP.md` updated with current status instead.
- **Only Jeffrey ships.** Merge to `main` and tag a release only when Jeffrey personally confirms.

---

## 2. Git Push Protocol

Pushing to GitHub frequently fails. Follow this exact sequence every time:

1. **Try the push once** programmatically (quick, no fuss).
2. **If it fails**, immediately hand Jeffrey a PowerShell block to paste. The block **must include the `cd` command** because he may not be in the right directory:
   ```
   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   git add -A
   git commit -m "your message here"
   git push
   ```
3. **After push (whether automatic or manual), always run the integrity checklist:**
   - Verify local HEAD matches `origin/main` (or the target branch) — lockstep check.
   - Scan for **0-byte files** in the repo.
   - Scan for **BOM signatures** (UTF-8 only, no BOM on Windows).
   - Check for **truncated files** (compare expected line counts).
   - Verify **all node classes are registered** in `__init__.py` `NODE_CLASS_MAPPINGS`.
   - Check for **missing widget mapping blocks** in workflow JSON files.
4. **If any check fails**, fix it and repeat from step 1.
5. **Do not declare done until lockstep is confirmed.**

---

## 3. Regression Testing (Non-Negotiable)

Every code change must pass a regression sweep before being considered complete:

- **Run the full test suite** (`tests/test_core.py`, `tests/vram_profile_test.py`).
- **Widget error testing**: Verify all nodes load without widget errors in ComfyUI. Check that all `INPUT_TYPES` return valid type specs and that workflow JSONs have explicit `{"widget": "string"}` mapping blocks for string inputs.
- **Bug Bible regression**: Cross-reference changes against the Bug Bible:
  - Repo: https://github.com/jbrick2070/comfyui-custom-node-survival-guide
  - File: `BUG_BIBLE.yaml`
  - Key bugs to always check: BUG-010 (0-dialogue hard abort), BUG-12.33 (Oversized prompt pre-fill stall/VRAM spike), widget connection drops, VRAM spills, cache deadlocks.
- **VRAM ceiling**: Peak VRAM must stay at or below 14.5 GB. Run `vram_profile_test.py` to confirm.
- **VRAM & Context Etiquette**:
  - **Never use `force_vram_offload()` between LLM phases** within the same script-writing pass. Use `_flush_vram_keep_llm()` to retain weights while clearing KV cache.
  - **Always enforce Prompt Truncation** against `context_cap` before calling `model.generate()`. If prompt > cap, truncate head.
  - **Warmup Mandatory**: All LLM loaders must perform a 1-token warmup pass to front-load CUDA kernel JIT compilation.
- Create **scratch scripts** (in `scratch/` directory) to help validate edge cases. These are disposable — delete when done.

---

## 4. Bug Log and QA Guide Lifecycle

- **No QA guides or bug logs exist unless there are active bugs.** Zero stale docs.
- If the **same bug appears 3 times**, create a formal bug log file (e.g., `BUG_LOG_v1.5.md`) in the repo root.
- A QA guide may be created alongside the bug log to document testing protocols for that specific bug.
- The bug log exists for **peer review** — Jeffrey may share it with reviewers.
- **Delete both the bug log AND the QA guide** once the bug is resolved and the fix is verified.
- The Bug Bible (`BUG_BIBLE.yaml` in the survival guide repo) is the permanent record. Local QA docs are temporary.

---

## 5. Code Quality Standards

- **Clean code.** Every function, variable, and file should make sense to a reader on first pass.
- **Clean logs.** Log messages should be informative, structured, and greppable. No noise, no spam.
- **Meaningful comments.** Explain the *why*, not the *what*.
- **Perfection is the target.** If something looks half-done, finish it.

---

## 6. Content Standards

Generated stories and narrative content must follow these rules:

- **Safe for work.** No explicit, violent, or disturbing content.
- **No profanity.** Zero tolerance for curse words in code, comments, logs, or generated output.
- **Good narrative structure.** Every story should have a clear beginning, middle, and end with a satisfying arc.
- **Non-violent.** Drama and tension are fine, but no graphic violence.

---

## 7. Security Awareness

- **Assume compromise.** Always verify that the right files are updated and that nothing unexpected has changed.
- **After every push**, visually confirm on GitHub (via web) that the pushed files match what was intended.
- **Check file integrity** — 0-byte files, unexpected modifications, or missing files could indicate a problem.

---

## 8. Reference Links

| Resource | URL |
|----------|-----|
| **OldTimeRadio Repo** | https://github.com/jbrick2070/ComfyUI-OldTimeRadio |
| **Bug Bible (Survival Guide)** | https://github.com/jbrick2070/comfyui-custom-node-survival-guide |
| **Bug Bible YAML** | https://github.com/jbrick2070/comfyui-custom-node-survival-guide/blob/main/BUG_BIBLE.yaml |
| **ROADMAP** | `ROADMAP.md` (in this repo) |

---

## 9. Integrity Checklist (Run After Every Push)

```
[ ] Local HEAD == origin HEAD (lockstep)
[ ] No 0-byte files in repo
[ ] No BOM signatures (UTF-8 only)
[ ] No truncated files
[ ] All node classes registered in __init__.py NODE_CLASS_MAPPINGS
[ ] All workflow JSONs have widget mapping blocks for string inputs
[ ] vram_profile_test.py passes
[ ] Full test suite passes (89+ tests)
```

---

## 10. v2.0 Visual Sidecar — Inviolable Constraints

**Audio is king. The full narrative story output must never break, shorten, or degrade.**

The v2 visual sidecar adds video generation alongside the audio pipeline. The audio pipeline is frozen — it must produce byte-identical output to v1.5 at every step. If adding video breaks audio in any way, revert immediately.

**Design spec:** `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md`

| ID | Rule |
|----|------|
| C1 | **No new inputs** on `OTR_BatchBarkGenerator`, `OTR_SceneSequencer`, `OTR_KokoroAnnouncer`, `OTR_AudioEnhance`, `OTR_EpisodeAssembler`, `OTR_MusicGenTheme`, `OTR_BatchAudioGenGenerator`, `OTR_Gemma4ScriptWriter`, `OTR_Gemma4Director`. Adding inputs shifts `widgets_values` indices and silently corrupts seeds/voices. |
| C2 | **No `CheckpointLoaderSimple`** or stock diffusion nodes in the main graph. They load checkpoints into the ComfyUI process while audio models hold residual VRAM — OOM on 16 GB. |
| C3 | All visual generation runs in **subprocesses** spawned with `multiprocessing.get_context("spawn")`. OS-level VRAM reclaim is the only reliable boundary across PyTorch + Blender. |
| C4 | LTX-2.3 clips capped at **10-12 seconds** (257 frames @ 24fps). Longer shots auto-chunk + ffmpeg crossfade. |
| C5 | LTX-2.3 must use Blackwell-native `torch.float8_e4m3fn`. |
| C6 | IP-Adapter is for **environments only**, never characters with lipsync (causes "Silent Lip Bug"). |
| C7 | Episode audio output must be **byte-identical** to v1.5 baseline at every gate. Full-length episode, full narrative arc, no truncation, no hash changes. |

### Audio Regression Gate

Before committing any v2 change, run:

```bash
pytest tests/v2/test_audio_byte_identical.py
```

If it fails, **revert immediately**. The audio path is non-negotiable.

### The Only Legal v1.5 Node Modification

`OTR_SignalLostVideo` (node #12) may gain **one** new optional input: `visual_overlay` (STRING — path to MP4). It must be the **last** input slot. When unwired, output must be byte-identical to v1.5. No other v1.5 node may be modified.

### Hardware

RTX 5080, 16 GB VRAM, Blackwell, single GPU, no cloud. LTX-2.3 at FP8 uses ~11 GB, leaving 5 GB for audio residuals + OS + ffmpeg.

### What Went Wrong Last Time (v2.0-visual-engine branch)

The previous v2 attempt broke audio output by modifying existing node inputs, causing widget drift that silently corrupted seeds and voices. It also tried to load diffusion models in-process, causing OOM. That branch is preserved as reference but should not be merged. This fresh attempt uses a sidecar architecture that is physically incapable of modifying the audio DAG.