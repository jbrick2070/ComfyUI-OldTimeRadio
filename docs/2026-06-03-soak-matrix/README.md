# 2026-06-03 soak matrix -- stress the PROMISING configs to find holes

Goal: not to re-prove the weak models fail (we know -- gemma-2-2b/E2B/E4B trip the
structured gates), but to stress the configs **most likely to be great** across
length + writer + music, and surface integration holes -- especially the two NEW
things shipped 2026-06-03: **Stable Audio 3 as the music default** and
**gemma-4-12b-it as a writer candidate**.

## Axes
- Writer LLM: `mistralai/Mistral-Nemo-Instruct-2407` (proven) and
  `google/gemma-4-12b-it` (new 12B candidate). NOT the weak small gemmas.
- Music: `stable_audio_3` (new default, needs full-episode validation) and
  `musicgen` (proven fallback / regression baseline).
- Voice = `bark`, announcer = `kokoro` (the only GPU-viable stack; the newer
  voice engines brick the cu130 venv -- see reference_audio_engine_dep_conflicts).
- Length: 30 words (quick) then 100 words.

## How to run (prune-to-audio: writer + audio only, no video, ~5-9 min/combo)
PRE-REQ: gemma-4-12b-it downloaded AND ComfyUI restarted (so SA3 + the new model
are live). ComfyUI up at localhost:8000.

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\_otr_soak_matrix.py 30  docs\2026-06-03-soak-matrix\combos_30.json  m30  > _otr_m30.log 2>&1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\_otr_soak_matrix.py 100 docs\2026-06-03-soak-matrix\combos_100.json m100 > _otr_m100.log 2>&1
```
Each run prunes to the audio sub-DAG (closure of node 7 EpisodeAssembler) so it
costs no video. The driver records PASS/FAIL + the `audio_done` marker + VRAM
peak per combo and writes `_otr_matrix_<tag>.json`. Watch with
`scripts/otr_tail_logs.py --lines 30`.

## What "PASS" means / what holes to look for
- PASS = `audio_done:length_sec=...` (full episode_audio assembled, no halt).
- SA3 holes: does the theme node's `_render_clips` drive eng_stable_audio_3 for
  all 3 cues (opening/closing/interstitial) without a crash? Right SR (44100)?
  VRAM under 14.5 GB with the writer just unloaded?
- gemma-4-12b holes: does it clear the freeze gate (BUG-305 helps abbreviations)
  + the structured passes (news briefs / normalize_length / cast lock)? Slower
  than mistral-nemo but does it COMPLETE? Log real bugs to BUG_LOG.md immediately.

## Stretch (extend the driver -- it patches by widget NAME already)
Add OFAT knobs on the best config (mn + SA3, 100w): cast_voice_policy (node 80
preserve_ledger<->auto_registry), num_characters (node 1: 1 vs 3), lemmy_cameo
(node 1 on/off). Use otr_api.patch_widget_by_name like the existing combos.
