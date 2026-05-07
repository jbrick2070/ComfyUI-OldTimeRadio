# Morning-after handoff — BUG-LOCAL-117a sirens_print regression

You went to sleep at ~22:30 on 2026-05-06. Code shipped while you slept:

- `workflows/otr_scifi_16gb_full.json` cut over to LTX 2.3 + distilled LoRA chain + Gemma encoder
- `nodes/batch_ltx_render.py` dual-engine refactor with `OTR_LTX_ENGINE` env var (default `v2_3`)
- `tests/test_core.py` whitelist updated for new `LTXAVTextEncoderLoader` node type
- All static checks green (AST, JSON parse, 185 tests passed)

**What's still pending:** the real-world regression. The unit tests can't tell us whether RES4LYF actually loads inside ComfyUI's runtime, whether the 22B BF16 model + 2 LoRAs + Gemma encoder + ClownSampler chain stays under 14.5 GB VRAM, and whether the audio path still byte-identifies to v1.5. That's a real ComfyUI run.

## What to do this morning

### 1. Boot ComfyUI fresh

Close any running ComfyUI window. Start clean so `OTR_LTX_ENGINE` env var (which defaults to `v2_3` per the new code) is properly read.

If you want to verify or override the engine:
```cmd
set OTR_LTX_ENGINE=v2_3
```

In the startup log you should see a banner like:
```
================================================================
[BatchLTXRender] BUG-LOCAL-117 engine=v2_3
[BatchLTXRender]   model:    LTX 2.3 22B-dev BF16 + distilled LoRA x2 (0.5, 0.2)
[BatchLTXRender]   encoder:  LTXAVTextEncoderLoader -> Gemma FP4 mixed
[BatchLTXRender]   sampler:  ClownSampler_Beta (exponential/res_2s, eta=0.25, bongmath=True)
[BatchLTXRender]   guider:   MultimodalGuider + GuiderParameters (VIDEO cfg=3.0 stg=1.0)
[BatchLTXRender]   decode:   LTXVTiledVAEDecode (LTX-specific tiled)
[BatchLTXRender]   sigmas:   LTX_DISTILLED_SIGMAS (9 vals, float32, CPU)
================================================================
```

The banner only fires when BatchLTXRender actually executes (mid-episode). If you don't want to wait until the LTX phase, queue a small smoke first.

### 2. Load `otr_scifi_16gb_full.json` and queue sirens_print

Use the same episode parameters as the 2026-05-05 verified run that's still our gold-standard reference (the `sirens_print` ep that proved BUG-106 + BUG-108).

If the script-writer / sequencer / Bark / FLUX / HuMo phases are still in the workflow, those should run unchanged (the cutover only touches the LTX section). Total wall time estimate:
- Script + Bark TTS: ~5 min
- FLUX bookend + portraits: ~3 min
- HuMo character lines (2 lines @ ~10-12 min): ~24 min
- **LTX 2.3 non-character lines (3 lines @ ~6-7 min): ~21 min** ← the test
- Compose + upscale + procgen blend: ~5 min

**Total ~58 min.** Was ~36 min on the v0.9 path. Extra ~22 min is the 22B-vs-2B model cost.

### 3. What to watch in the ComfyUI console

**Required signals:**
- The `[BatchLTXRender] BUG-LOCAL-117 engine=v2_3` banner fires — confirms env var resolved correctly.
- For each non-character line: `[BatchLTXRender] <line_id> done: role=<role> length=<frames> dur_s=<sec> -> <line_id>.mp4 (<ms> ms)`. Per-line wall time should be 6-12 min for a typical 5-7s line.
- No `'Linear' object has no attribute 'weight'` (the FP4 Gemma trap from last night's smoke). If this fires, `LTXAVTextEncoderLoader` widgets aren't being passed through correctly.
- No `RuntimeError: ... missing: ['ClownSampler_Beta', ...]`. If this fires, RES4LYF didn't load — check `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\RES4LYF\` exists and ComfyUI saw it on startup.

**Failure modes to watch for:**
- `OutOfMemoryError` on the FIRST line: VRAM ceiling exceeded. Probably the simultaneous Gemma + 22B + 2 LoRAs is too much. Mitigation: in workflow JSON, `bypass` (mode=4) one of the two LoRA nodes and try again — single-LoRA may fit.
- `OutOfMemoryError` on line 3+ but lines 1-2 succeed: per-line GC isn't draining VRAM fast enough. The new code already has aggressive `del + gc.collect() + empty_cache()` per chunk; if it still leaks, escalate to a fuller per-line teardown.
- Line 1 takes >20 minutes: PCIe thrashing on the streaming model. Round-robin (Gemini) flagged this risk for `ClownSampler_Beta` not respecting ComfyUI memory hooks. Open Task Manager → Performance → GPU → check **Shared GPU Memory**. If it's near 32 GB and **Dedicated GPU Memory** keeps swinging high/low, the sampler is thrashing the PCIe bus. Mitigation: roll back to v0_9 engine (`set OTR_LTX_ENGINE=v0_9` plus `git checkout pre-bug-117a-cutover workflows/otr_scifi_16gb_full.json`).
- Per-line tensor shape mismatch crash inside `MultimodalGuider`: this would indicate I got the `GuiderParameters` widget values wrong. Compare the values logged by ComfyUI to the constants in `batch_ltx_render.py` near `LTX_V2_3_VIDEO_CFG`.

### 4. Verify after the run

Three things must hold:

#### Visual
Watch the final `.mp4` in `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\sirens_print\<timestamp>\sirens_print.mp4`.
- Non-character beats (announcer, music, sfx) should show smooth motion (the "subtle zoom in" quality from last night's smoke), NOT static frames or glitching.
- Character lines (HuMo) unchanged — should look exactly like the 2026-05-05 sirens_print baseline.
- Composite seams between LTX clips and HuMo clips should be invisible (matching fps + same canvas dimensions).

#### VRAM peak
Check the run ledger or console for per-line peak VRAM. Should stay under 14.5 GB. If you didn't capture this, add it to the next run via the ledger telemetry path (or watch Task Manager during the LTX phase).

#### C7 audio byte-identity
The audio master mix should be byte-identical to v1.5 baseline. The LTX node never writes audio so this should hold trivially, but verify with:
```powershell
$base = 'C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\sirens_print'
$prev = Get-ChildItem $base -Directory | Where-Object { $_.Name -lt '2026-05-06' } | Sort-Object Name -Descending | Select-Object -First 1
$curr = Get-ChildItem $base -Directory | Where-Object { $_.Name -ge '2026-05-06' } | Sort-Object Name -Descending | Select-Object -First 1
$prevAudio = Join-Path $prev.FullName 'master_mix.wav'
$currAudio = Join-Path $curr.FullName 'master_mix.wav'
if ((Test-Path $prevAudio) -and (Test-Path $currAudio)) {
  $h1 = (Get-FileHash $prevAudio).Hash
  $h2 = (Get-FileHash $currAudio).Hash
  if ($h1 -eq $h2) { Write-Host "C7 OK: audio byte-identical" } else { Write-Host "C7 BREACH: master_mix changed" }
}
```

(Adjust path if `master_mix.wav` lives elsewhere in your tree.)

### 5. If everything green

Mark BUG-LOCAL-117a `[FIXED]` in BUG_LOG.md (currently `[SHIPPED, REGRESSION PENDING]`) and ship the closing commit:

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
echo BUG-LOCAL-117a [FIXED]: sirens_print regression confirms LTX 2.3 in OTR production> .git\COMMIT_EDITMSG
echo.>> .git\COMMIT_EDITMSG
echo Verified visual motion, VRAM under 14.5 GB peak, audio C7 byte-identical.>> .git\COMMIT_EDITMSG
git add docs\BUG_LOG.md
git commit -F .git\COMMIT_EDITMSG
git push origin v2.0-alpha
```

### 6. If something failed

The pre-cutover state is tagged `pre-bug-117a-cutover`. Full rollback:

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git checkout pre-bug-117a-cutover -- workflows/otr_scifi_16gb_full.json nodes/batch_ltx_render.py
set OTR_LTX_ENGINE=v0_9
```

Restart ComfyUI. The legacy v0.9 path will run. Then ping me and we'll diagnose what broke.

Partial rollback (e.g., LoRA loaders OOM but sampler is fine): bypass the LoRAs in the workflow (right-click each → Bypass / mode=4) and re-queue.

## Files changed in the cutover commit

- `workflows/otr_scifi_16gb_full.json` — model widget, +2 LoRA nodes, encoder type swap, link reroute
- `nodes/batch_ltx_render.py` — +194 LOC for v2_3 engine path
- `tests/test_core.py` — whitelist update
- `docs/BUG_LOG.md` — BUG-LOCAL-117a entry
- `docs/2026-05-06-bug-117-ltx23-res4lyf-migration__*` — round-robin transcripts + synthesis (existing, not modified by this commit)

## Outstanding questions for next session

1. PCIe thrashing observation — did Shared GPU Memory stay calm or spike? If spiked, we need a subprocess-isolation pass on `_render_one_line_v2_3_res4lyf` so ClownSampler runs in its own process and can't hold the model patcher hostage.
2. Wall time per line — if it ran much slower than ~6-7 min/line, the model isn't streaming efficiently and we may need to look at `LowVRAMCheckpointLoader` settings or fall back to standard `CheckpointLoaderSimple`.
3. Output quality — was motion as smooth as last night's standalone smoke, or did the chunking + per-line teardown introduce per-clip artifacts? Sample the MAD between consecutive frames as we did for BUG-LOCAL-112 to get a quantitative signal.
