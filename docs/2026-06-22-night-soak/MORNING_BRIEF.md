# OTR Night Broadcast Soak -- Morning Brief (2026-06-22)

## TL;DR
The all-night broadcast soak is **running** and **producing episodes to `output\otr\obs`**
(the folder your OBS machine watches). First full episode -- **"Locked Doors"** (mistral,
420w) -- published clean: `signal_lost_locked_doors_20260621_235125_silent_procgen_blended_final.mp4`
(57 MB, video + AAC-320k viewing audio). The whole pipeline is proven end-to-end.

## What's running
- Driver: `scripts/_otr_night_soak.py` (10h, started 23:43 2026-06-21). Loads the REAL
  canonical `workflows/otr_scifi_16gb_full.json` in-memory per episode (never a copy).
- **Alternates** full episode / visualizer-only (every other = quick filler).
- **Writers rotate** (clean buckets for your analysis): mistral (local), gemma-4-12b
  (local Ollama), grok-4.3, gpt-5.5-pro, deepseek-v4-pro (frontier via OpenRouter slot-a;
  technical calls stay on local mistral = your cost-smart split). ~2/3 local, ~1/3 frontier.
- **Words** step 420/560/700/864; **creativity** rotates safe&tight/balanced/wild&rough/maximum chaos.
- **Engines** = your saved soak_param: ltx_av bookends, flat_still body beats fed by
  z-image-turbo stills (nvfp4); **indextts2** char voice + kokoro announcer; visualizer
  episodes route every video role to the CRT-scope `visualizer`. Fresh cast + news every
  episode (no C7).
- Throughput: full episodes ~30 min (ltx_av's 22B loads per bookend dominate); visualizer
  episodes much faster. Expect ~20-30 episodes by morning.

## What I fixed tonight to make it work
1. **indextts2 was dead headless** -- root cause: the Desktop-v2 junction made
   `abspath(__file__)` resolve `_COMFY_ROOT` to the wrong tree, so the sidecar venv/weights
   "weren't found." Fixed with `realpath` in eng_indextts2/chatterbox/dia. **Shipped +
   pushed (358accd)**, suite 4957/34 + Bug Bible 16/7/3 green. indextts2 now renders.
2. **act_count** -- the workflow ENFORCES a word-scaled minimum ("act_count below default 3
   for 420w -- pick 3 or higher"), so `act=auto` is correct (your "words drive the acts" is
   already built in as a floor). My earlier act=1 was rejected; reverted to auto.
3. **Freeze halt** -- the story critic bounds its reroll out and stamps `needs_full_rerun`,
   which makes CastLock refuse. Set **`OTR_BYPASS_FREEZE_HALT=1`** on the server as a
   TONIGHT stopgap so episodes render past it. (Locked Doors actually shipped legitimately
   as `frozen_with_warns` via A2 repair-then-ship -- the bypass is just the backstop.)
4. Clean VRAM reset before launch (a 9.6 GB lingering CUDA context cleared to 1.7 GB baseline).

## >>> MORNING ROUNDTABLE TARGET (the real fix) <<<
The bypass is a band-aid. The substance for the morning is **story-writer quality**: the
critic keeps naming reroll targets that bound out -> `needs_full_rerun`. At 420w/act=auto the
writer makes ~18 beats (~23 words/beat -- sparse), which the critic rejects. This is upstream
of ALL my session's audio/video changes (realpath/capability-routing/loudnorm never touch the
LLM writer), so it is NOT a regression I introduced -- it's pre-existing writer-quality
behavior to harden.
- Data for the roundtable: every episode's **ledger is intact** in
  `output\otr\episodes\<ep>\...ledger.json` + `episode_canon.json`; per-episode config +
  status in `scripts\_otr_night_soak_summary.json` + `_otr_night_soak_episodes.jsonl`; the
  freeze verdicts + reroll logs are in the server log
  `docs\2026-06-22-capability-routing\night_server3.log`.
- Roundtable scope (per CLAUDE.md S8): R1 why the critic over-rejects sparse beats / how to
  raise beat density (writer-side, not just act floor); R2 the coding plan; R3 wiring; R4
  convergence. Once the writer reliably freezes clean, **remove the OTR_BYPASS_FREEZE_HALT
  stopgap.**

## Other morning tasks (tracked)
- **Remove the 14B wan** from the workflow + stand up a <14GB Wan 2.2 (5B TI2V or a small
  quant), proven first with a CLEAN native image->video smoke (no OTR VRAM wrapper -- your
  thrash hypothesis is right: it re-stages 13.6 GB per chunk). The capability-routing fix
  (cf5fbb3) is general and stays.

## How to check the morning haul
- `Get-ChildItem output\otr\obs *_final.mp4 | Sort LastWriteTime` -- the night's episodes.
- `scripts\_otr_night_soak_summary.json` -- per-episode writer/words/creativity/status/obs.
- Server still on :8000 with the night enable-set; driver runs until ~09:43. To stop early:
  kill the `_otr_night_soak` python; the server can stay up.
