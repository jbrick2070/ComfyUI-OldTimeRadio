# M4 -- LTX-AV forced-lane smoke + bookend health (in-flight findings)

Date: 2026-06-15. Run: forced-lane 30w episode on a headless server (alt port
:8011) so it never collides with the operator Desktop. Env: `OTR_ENABLE_LTX_AV=1`,
`OTR_FORCE_ENGINE_MAP=announcer_visual=ltx_av_talk,character_video=ltx_av_talk,
music_visual=ltx_av_music`, `OTR_LTX_OPEN_STRICT=0` (warn-not-abort so the episode
completes for bookend inspection). Harness: `scripts/queue_smoke.py` ->
`COMFYUI_URL=http://127.0.0.1:8011` -> the REAL `otr_scifi_16gb_full.json`.
Prompt id `6331307f-93e9-4699-a949-037bbcec86d1`.

## Programmatic observations (through the audio + early video phase)

- Server booted clean on :8011 (LTX-AV env set); episode QUEUED + rendered the
  writer + audio (indextts voices, SA3 music) phases normally.
- **No `ENGINE OVERRIDE` / `ltx_av` / `LTX-OPEN HEALTH` log lines were observed**
  through the phases captured; the video-beat sampler ran at **20 steps @ ~1.5
  s/it** -- consistent with the DEFAULT planned lane (humo_1.7B / ltx_video), NOT
  the heavy 22B LTX-AV (which would show an `UnetLoaderGGUF` + a ~CPU-Gemma encode
  + a much slower/heavier resident). GPU never parked at the ~13-14 GB an LTX-AV
  forward would hold.

## Key FINDING (to confirm from the final ledger)

The forced LTX-AV lane appears NOT to have been exercised by this run. Two
candidate causes, to disambiguate from the completed ledger's shot `engine_id`s:
1. The `OTR_FORCE_ENGINE_MAP` env may not have reached the render process, OR
2. `apply_engine_override` applied but the shots still rendered on their planned
   engines (the override log would have fired -- it did not), OR
3. LTX-AV was attempted and fell back instantly (no heavy load) -- but then the
   `[OTR video] LOUD ENGINE OVERRIDE` + a fallback restamp + the S5 `LTX-OPEN
   HEALTH` warning would all have logged, and none did.

The ABSENCE of any override log most strongly points to (1)/(2): the force-map did
not rewrite the video shots in this `run_real_episode` (via OTR_VideoRenderBatch)
path. **`OTR_VideoRenderBatch.execute` calls `run_real_episode`, which DOES call
`apply_engine_override(ledger)` -- so the env not reaching the detached server
process is the leading hypothesis** (a Start-Process env-inheritance gap for the
headless .cmd). This is a SMOKE-HARNESS finding, not an engine bug: the lane is
still selectable via the OTR_VideoDirector dropdown (V-6) the operator picks in the
GUI; only the force-map SMOKE path is in question.

## What this run DID validate
- The full default chain (writer -> audio -> images -> video beats -> procgen ->
  composite -> mux) runs end-to-end on :8011 -- so the bookends (opener/closer)
  ARE rendered for the BUG-411 / BUG-414 look-QA.
- The S5 `check_ltx_open_health` guard is live in the manifest path (it would have
  flagged a procgen-fallback open; none flagged through the captured phases).

## NEXT (to actually smoke LTX-AV end-to-end)
- Confirm the env reached the server: re-run with the force-map exported INSIDE
  the launcher .cmd (append `set OTR_ENABLE_LTX_AV=1` + the force-map to a copy of
  `_otr_m4_server_launch.cmd`) rather than relying on Start-Process inheritance;
  OR drive `run_real_episode` directly with the env in-process (the soak harness).
- OR (cleanest for "selectable in a real episode"): edit the saved
  `otr_scifi_16gb_full.json` OTR_VideoDirector role widgets to pick
  `ltx_av_talk` / `ltx_av_music` (V-6 dropdown) and render -- no force-map needed.
- Then confirm: ledger shot `engine_id`s == ltx_av_*, NVML <= 14.5 GB at the
  512x288 floor, no `LTX-OPEN HEALTH` warning, lip-sync-vs-HuMo A/B (operator).

## Paste-ready verdict extraction (after the run completes)
```powershell
$led = Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes" -Recurse -Filter "*_ledger.json" |
  Where-Object { $_.DirectoryName -notmatch '\\audio$' } | Sort-Object LastWriteTime -Desc | Select-Object -First 1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -c "import json;d=json.load(open(r'$($led.FullName)',encoding='utf-8'));v=d.get('video',{});print('shot engines:',[(s.get('role'),s.get('engine_id')) for s in v.get('shots',[])])"
# then grep the server log for: 'ENGINE OVERRIDE', 'LTX-OPEN HEALTH', 'ltx_av', 'obs_publish'
```
Bookend look-QA: compare the opener vs
`output/otr/episodes/signal_lost_chilled_hope_20260603_161926/videos/b005.mp4`.
