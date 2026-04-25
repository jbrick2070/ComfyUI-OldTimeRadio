# Autonomous Run Status — 2026-04-18

**While you were at the movies.** Audio bus untouched. No merges to `main`. Work on `v2.0-alpha`.

## Short version
- First test episode **shipped**: `output\visual_renders\The Silent Uprising_20260418_191228.mp4` (41.9 MB, 7:13 PM).
- Cleaned up a stale sidecar pair (launcher 37212 + worker 64544) that was burning CPU post-completion, plus a leftover `vram.lock`.
- Found and fixed a latent bug in `scripts/soak_operator.py` that would have broken any HTTP-API submission of the TEST workflow. Logged as **BUG-LOCAL-045**, Bible candidate.
- Re-submitted the workflow through the HTTP API — **queued and completed in 32 s**, `status_str=success`.
- Full regression green: Bug Bible 23/1/2 and OTR suite 160/1 skipped.

## Timeline

**19:02** — Sidecar `vs_8b073fc3f6e4` spawned (PID 37212, backend=video_stack). STATUS.json shows it hit the `wan21_loop real mode: 9 shots, I2V loops from FLUX stills` phase and froze there.

**19:13** — Renderer completed the episode and wrote the final mux to `visual_renders\The Silent Uprising_20260418_191228.mp4`. Audio path looks correct — the in-process renderer finished while the sidecar was still grinding on I2V loops in the background.

**~19:30** — Verified LTX and Wan2.1 shots were running in stub mode per `io\visual_out\vs_8b073fc3f6e4\s01_01\meta.json`: `backend=ltx_motion, mode=stub, reason=ltx_weights_missing:...\models\diffusers\LTX-Video`. Expected — LTX weights install is still Task #72.

**~19:45** — Killed PIDs 64544 and 37212 (sidecar + launcher), deleted `custom_nodes\io\vram.lock` (116 bytes, would have blocked a fresh run).

**~19:55** — First API re-submission failed HTTP 400 with `LoadAudio.validate_inputs() missing 1 required positional argument: 'audio'`. Dumped `outputs\_api_prompt.json` — node 100 had empty `inputs: {}` despite `widgets_values[0]="silent_uprising_test_hq48k.wav"`. Traced to the UI→API converter.

**~20:00** — Root cause: `scripts/soak_operator.py::_WIDGET_PRIMITIVE_TYPES` only allowed `{STRING, INT, FLOAT, BOOLEAN, BOOL}`. ComfyUI's newer unified dropdown schema uses `"COMBO"` — LoadAudio declares its `audio` input as `["COMBO", {options:[...], audio_upload:True}]`. Converter saw COMBO, said "not widget-backed", and silently dropped the value. Added `"COMBO"` to the set. One-line fix.

**~20:05** — Re-submitted via `outputs\submit_test_workflow.py`. `prompt_id=fb49c470-e5ae-4c8e-95c9-16d596008b52`, queue=3. Completed in 32 s. Node 18 (VisualRenderer) was the only real re-execution; 16/17/101/102 came from execution cache.

**~20:10** — Full regression.

## Regression (green)

```
Bug Bible : 23 passed, 1 skipped, 2 xfailed (0.94 s)
OTR suite : 160 passed, 1 skipped (121 s)
  - tests\test_dropdown_guardrails.py  47 passed
  - tests\test_core.py                100 passed
  - tests\test_audio_byte_identical.py 13 passed, 1 skipped
```

AST parse of `scripts/soak_operator.py` clean.

## What changed on disk

| File | Change |
|---|---|
| `scripts/soak_operator.py` | Added `"COMBO"` to `_WIDGET_PRIMITIVE_TYPES` (line 974). Not committed yet — waiting for you to push. |
| `docs/BUG_LOG.md` | New entry BUG-LOCAL-045 at top. Bible candidate: yes. |
| `docs/2026-04-18-autonomous-run-status.md` | This file. |
| `outputs\submit_test_workflow.py` | Standalone API-submit harness. Portable — uses the converter from `soak_operator` plus `urllib`, no framework deps. |
| Killed | sidecar PIDs 64544 + 37212; `custom_nodes\io\vram.lock`. |

## Still pending (unchanged)

- **Task #72** — LTX-Video weights install. Shots are running in `stub` mode. Episode renders still work because renderer falls back to still-as-motion via ffmpeg Ken Burns.
- **Task #60** — Re-launch FLUX + PuLID + Wan2.1 T2V weights download.
- **Task #62** — Verify end-to-end after weights are real.
- **Task #82** — Rename `OTR_Gemma4*` → `OTR_LLM*`.
- **Task #88, #92** — Emoji strip + flatten commit.

## Handoff for the git push

The soak_operator fix is a one-line code change + a doc update. When you're ready:

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git status
git add scripts\soak_operator.py docs\BUG_LOG.md docs\2026-04-18-autonomous-run-status.md
echo BUG-LOCAL-045 soak_operator: accept COMBO widget type in UI to API converter> .git\COMMIT_EDITMSG
git commit -F .git\COMMIT_EDITMSG
git push origin v2.0-alpha
```

Hope the movie was good.
