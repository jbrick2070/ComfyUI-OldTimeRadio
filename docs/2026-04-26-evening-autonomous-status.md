# 2026-04-26 PM autonomous run - status when Jeffrey gets back

## TL;DR

Comfy is dead. Reboot or kill `ComfyUI.exe` from Task Manager to recover.
All my fixes are committed/pushed and will load on next clean start.

## What happened while you were out

1. **Ran the architectural voice-health change** you authorised. Refactored
   `_bark_health_check` into pure `_bark_test_presets` + legacy wrapper +
   new `_bark_health_check_for_cast`. Removed the eager full-catalog
   warmup at top of `direct()`. Wired the lazy cast-only check into the
   cast-merge path right after `_consolidate_similar_cast_rows`. On
   individual preset failure: remap to known-good fallback of same
   gender from `_VOICE_PROFILES`. Net: -95s on first-run-after-boot,
   +25s per queue, individual failures self-heal.
2. **34/34 tests pass.** AST clean. Committed as `b9245da` BUG-LOCAL-072.

3. **Validated the FULL pipeline twice on the OLD code** before things
   went sideways:
   - Run A (your queue): "Times Fault Line" - 4:55 episode, 7 FLUX
     env stills, HuMo `clip_00004_.mp4` shipped. **HuMo Sage-off
     workaround validated for the second run.** STANLEARY/STANLEY
     cast split reproduced exactly as predicted (this is what BUG-071
     fixes).
   - Run B (queued via API): hit a CUDA fault before producing
     anything. See BUG-LOCAL-073.

4. **Comfy server now hung.** At 20:16:27 the orchestrator hit
   `[StoryOrchestrator] model.cpu() during unload failed: CUDA error:
   an illegal memory access was encountered` during the OpenClose
   expansion phase. The python server (PID 66452, 8.4 GB) became a
   CUDA-locked zombie. `taskkill /F` and PowerShell `Stop-Process -Force`
   both reported success but the process stayed alive holding port 8000.
   ComfyUI Desktop's electron wrapper did NOT auto-respawn the child.

## Recovery

**Easy path:** reboot Windows. Cleanest reset.

**Faster path** (if you don't want to reboot):

1. Task Manager - End task on the parent ComfyUI.exe (the one with
   children, was PID 32140).
2. If `Get-Process python | Where-Object Id -eq 66452` still shows it
   alive, run an admin PowerShell `Stop-Process -Id 66452 -Force`.
3. Start ComfyUI from Start Menu.
4. Wait for boot. Look for `[OldTimeRadio] HTTP route registered:
   GET /otr/latest_ledger (with CORS)` in the boot log -- that
   confirms BUG-071/072/073-route fixes loaded.

## What to test on next run

Queue the FULL workflow with the same Mistral-only widgets. Look for:

- **STANLEARY/STANLEY collapse** - the cast should have ONE row for
  Stanley (whichever variant won), not two. If you see both, BUG-071
  didn't catch it; surface the cast with `curl http://127.0.0.1:8000/otr/latest_ledger`.
- **Lazy voice health log line** - should see `[VoiceHealth] Lazy
  cast check on N preset(s)` instead of the legacy `Running 1-second
  Bark health check on English presets` banner. If you see the
  legacy banner, BUG-072 didn't load.
- **Live artifact** - `otr-live-run-tail` in the Cowork sidebar should
  start working. The CORS headers are now on the `/otr/latest_ledger`
  route. If the artifact shows `fetch error: Failed to fetch`, the
  CORS headers didn't land.

## Commits today

```
b24f1ce  BUG-LOCAL-073: log CUDA-locked zombie + unkillable Comfy server
b9245da  BUG-LOCAL-072: lazy voice health check after Director
2a95d47  BUG-LOCAL-071: fuzzy-merge cast dedup + HTTP route + CORS
6458b97  BUG-LOCAL-070 update: scope is FLUX-fp8 + Blackwell, not BatchFluxRender
```

All on `v2.0-alpha`. Pushed to origin.

## Tests added

- `tests/test_cast_fuzzy_consolidate.py` - 23 cases, all pass. Covers
  prefix-overlap (LLOYD/LLOYD KAPOOR), typo divergence (STANLEY/STANLEARY),
  no-over-merge guards (ROBERT FROST vs ROBERT FORD), edge cases.
- `tests/test_director_cast_namespace_merge.py` - existing 11 still pass.

## Open items

- BUG-073 needs a watchdog or pre-cpu() `torch.cuda.synchronize()`
  to surface CUDA faults before they zombify the process. Not shipped
  this session; logged with bible promotion notes.
- ComfyUI Desktop respawn-on-zombie behaviour is broken; worth an
  upstream issue.
- Env-token rescue for low-`[ENV:]` LLM output (would prevent the
  1-FLUX-fallback we saw on Run B earlier today). Not shipped this
  session; noted as future work in BUG-LOG entry for BUG-LOCAL-070.
