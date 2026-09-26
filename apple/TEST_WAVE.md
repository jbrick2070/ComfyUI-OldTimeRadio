# Test wave -- 2026-09-25

**Operator 2026-09-25: "the only thing is to regress test."** This file is the
whole wave. Part A reviews the 5080's overnight chain. Part B is the 4060
regression on the 8 GB rows. Part C is the suite and the Bug Bible. Each part is
run by a Claude window ON that box; a cloud window cannot touch either machine.

The morning starts in `otr/obs/`, not the editor.

## Rules for every leg

1. **Pass means published.** A leg passes only with all four: the runner prints
   `[canonical-api] RESULT SUCCESS`, it prints `[canonical-api] EPISODE
   episodes/<ep>`, the server log says `obs_publish OK`, and the final `.mp4` is
   in `otr/obs/`. Anything else is a fail.
2. **Five-minute rule.** A leg that has run 5 minutes with no new heartbeat
   (`[canonical-api] t=<N>s prompt_id=... status=...`) is presumed dead: read
   the leg log and the server log before waiting longer. For long legs run
   `scripts/otr_render_watchdog.ps1 -LegLog <leg.log>` (exit 2 = stalled 300 s
   or the server went away; it reports, it does not reset).
3. **The runner.** One leg:
   `python scripts/otr_canonical_api_run.py --comfyui-url http://127.0.0.1:<port> --profile <row> --act-count 1 --timeout 0 --run-label <label>`.
   `--timeout 0` waits for a terminal result. Use `--run-label`, never
   `--title` (`--title` names the episode and the title card). Exit 0 =
   SUCCESS; exit 1 = anything else. A missing weight the row cannot download
   stops in seconds with `PREFLIGHT FAIL` naming the file. `RESULT TIMEOUT`
   followed by "BUT THE RENDER IS STILL ALIVE" means the watcher stopped, not
   the render.
4. **Dry-run first.** `--dry-run` on every row before any GPU time. It
   builds the prompt and does not queue it.
5. **Reset selectively** (CLAUDE.md section 4): kill by CommandLine, never a
   blanket python kill; confirm port empty and VRAM back to desktop baseline
   before booting. **Never on the 5080 while the chain runs.**
6. **Models live under `C:\ComfyUI-Models`** (or `OTR_COMFYUI_MODELS_ROOT`).
   `Test-Path` there before calling any weight missing.
7. **Receipts per leg:** row id, HEAD, prompt_id, episode id, `otr/obs`
   filename, wall time, peak VRAM (`nvidia-smi`), runner exit code, and any
   `[OTR.assets]` fetch or `EXISTING` lines. Keep the leg log and the server log
   tail before any reset. Write the receipts as a new top entry in
   [HANDOFF_LOG](HANDOFF_LOG.md).
8. **A failure is work.** Root-cause it, fix it, push it (CLAUDE.md). A failure
   a live leg produced is admissible to `PROD_BUG_LOG.md`.

## Part A -- 5080 overnight chain (review when it ends)

Chain: `C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_chain.ps1`, 1-act legs
rotating still, video, foley, mime, animatediff on the 16 GB rows, until 08:00
local on 2026-09-25 or the first failure. Server log:
`C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_server.log`. Already published
before handoff: `covenant_ink_20260924_232810` (otr_16gb_still),
`notched_key_20260924_234023` (otr_16gb_video), `black_fog_20260925_011016`
(otr_16gb_foley). The mime leg (prompt `ea876b5c`) was rendering at handoff.

- **A1.** Confirm the chain has ended on its own (08:00, or a first failure in
  its log). Do not stop it early.
- **A2.** Table every prompt the chain queued: prompt_id, row, RESULT, episode,
  `otr/obs` file. Count published episodes against queued prompts.
- **A3.** The AnimateDiff leg is the first live `otr_16gb_animatediff` on the
  per-engine gate (7180701): no `PREFLIGHT FAIL`, and the SD 1.5 checkpoint
  shows as `[OTR.assets] EXISTING` or a verified fetch.
- **A4.** Ghost Half B measurement (owed): collect every
  `[OTR_ShotLock] Ghost Half B: eligible=N submitted=N admitted=N candidates=N`
  line and count `kernel_source` values (`authored_subject`,
  `key_object_in_beat`, `key_object`) across the night.
- **A5.** On a failure: keep the leg log and server log tail first, then
  classify it and fix at the root.
- **A6.** Only after the chain has ended: reset the box if anything else needs
  it. Then the dirty partial red-test patch in that tree (see the handoff log)
  gets finished or dropped on its own; it is not part of this wave.

**Part A outcome (reviewed 2026-09-25 afternoon from `yt_server.log`, 1.7 MB,
last written 08:34; HEAD of the chain's pack was `fe17f426`).**
A1: ended by the clock. Four prompts queued, four `Prompt executed`, no
fifth `got prompt`: the mime leg finished at 08:10, past the 08:00 cutoff,
so the AnimateDiff leg never queued. No `FAILED -- no episode`, no
`RenderError`, no `PREFLIGHT FAIL`, no interrupt.
A2: 4 queued / 4 published, in order: `covenant_ink_20260924_232810`
(otr_16gb_still, 00:10:53), `notched_key_20260924_234023` (otr_16gb_video,
01:32:07), `black_fog_20260925_011016` (otr_16gb_foley, 02:16:02),
`counting_three_20260925_033216` (otr_16gb_mime, 04:48:18 -- a measurement,
not a failure; his judgement whether a one-act mime leg should take that).
Every weight the legs used read `[OTR.assets] EXISTING`.
A3: DONE the same afternoon on the 5080 at `884cacf9` (pack `a553ac3b`
code): one `otr_16gb_animatediff` leg, 1 act, through
`scripts/otr_canonical_api_run.py` (prompt `ec7d0d3b`). The queue-time gate
passed it (ADE classes registered), every weight read `[OTR.assets] EXISTING`
(SD 1.5, v3 motion module, v3 adapter), no `PREFLIGHT FAIL`. RESULT SUCCESS in
00:28:08, 8 clips, published
`deed_summary_20260925_122647__shst__adhv__none__koko__marc__g412__sa3_final.mp4`
(137 MB) to `otr/obs`.
A4: MEASURED on that leg. `[OTR_ShotLock] Ghost Half B: eligible=4
submitted=4 admitted=4 candidates=5`. `kernel_source` over the 8 shots (from
`node_episode_report.json`; it is not in the ledger): `bookend_radio` 4 (the
music and announcer bookends), `key_object_in_beat` 2, `authored_subject` 2,
`key_object` 0. Ghost modes: signal 3, object 3, figure 2. One leg, so a
first reading, not a distribution.
A5: nothing to classify. The five WARNING tracebacks at boot are the known
duplicate-pack / `models` folder scan noise (same on every boot); the three
ERROR tracebacks at the END of the log are `/object_info` probes at 08:33
from the 5080 window hitting the half-moved `ComfyUI-OTR-UpstreamStoryLab`
pack during that morning's custom_nodes cleanup -- not the chain's, and that
pack is gone.
A6: done -- the box was reset per section 4 at 09:23 for the cleanup legs;
the red-test patch landed as `004d07ac`.

## Part B -- 4060 regression, 8 GB rows

Pull first (`git fetch origin main`, `git log --oneline HEAD..origin/main`,
`git pull --rebase origin main`) and record HEAD. The head must include the
AnimateDiff auto-download change (branch `claude/practical-hamilton-ftvd2j`,
merged to main) or B3 cannot prove it. Boot the headless server with the UTF-8
launcher (CLAUDE.md section 5). One leg at a time, in this order -- cheapest
weights first, so an early failure costs the least:

| leg | row | video lane | what it proves | status (4060, fresh start at `1585d38a`, 2026-09-25/26 night) |
|---|---|---|---|---|
| B1 | `otr_8gb_low` | `viz_camera` (no video weights) | Qwen3.5-4B writer at the 6.8 GB ceiling, Kokoro, Stable Audio 3, publish | PASSED -- 10:57, peak 4379 MiB, `foam_drowning_20260926_002211` |
| B2 | `otr_8gb_still` | `still_motion` | Z-Image Turbo stills on 8 GB | PASSED through the GUI on a wiped 2.3.4 install (`c2f14301`, 32:09) |
| B3 | `otr_8gb_animatediff` | `animatediff15_v3_haunted_video` | SD 1.5 checkpoint, v3 motion module and v3 adapter all download at queue time; no `PREFLIGHT FAIL` | PASSED -- WITHOUT the pack it was refused in 7 s leading with "Install ComfyUI-AnimateDiff-Evolved", nothing downloaded (PBUG-20260925-02 live verify); with the pack: 3.9 GB fetched at queue time, 60:07, peak 6041 MiB, `split_ridge_20260925_225242` |
| B4 | `otr_8gb_video` | `ltx_8gb` | LTX 0.9.8 2B plus T5 fetched at queue time | PASSED WITHOUT ComfyUI-LTXVideo -- 16.1 GB fetched, 33:39, peak 7300 MiB, free RAM low 0.42 GB (the T5 reloads per segment -- PBUG-20260926-01, fixed `f011bb79`: one load per beat, 6 of 13 on the 5080's proof leg; the 4060's free-RAM low is OWED on the next B4 run), `broken_horsehair_20260925_235312`. Verdict below. |
| B5 | `otr_8gb_ltx25_foley` | `ltx25_foley_16gb` | the ~25 GB LTX 2.5 stack on 8 GB; measured 707 s per 97-frame clip on this card | refused by the runner preflight at `1585d38a` (pinned Stable Audio 3 base file; fixed `0d44385c`); re-run at `0d44385c` running |
| B6 | `otr_8gb_ltx25_mime` | `ltx25_mime_16gb` | same weights, `EXISTING` | same as B5 |
| B7 | `otr_8gb_ltx25_audio_in` | `ltx25_audio_in_16gb` | same weights, `EXISTING` | same as B5 (first in the re-run). The same row PASSED on the 5080 (1 act, 2:00:29, `stone_key_20260925_230558`) |

**The fresh-start order (operator 2026-09-25: wipe the 4060's OTR install and
models root first).** Test ONE commit the 5080 names, not a moving `main`.
Install the pack; queue `otr_8gb_animatediff` BEFORE installing
AnimateDiff-Evolved and log the refusal verbatim with its wall time -- it must
arrive in seconds, lead with "Install ComfyUI-AnimateDiff-Evolved", and download
nothing (PBUG-20260925-02's live verify). Then install the pack via Manager, B3,
B4, then B1 and B5-B7.

**B4 DECIDED IT (2026-09-26): ComfyUI-LTXVideo is gone** -- `f111293b` removed the
clone, the kornia patch, their checks and tests and the doc mentions, as ruled below.
The original rule, kept for the record: **B4 decides the ComfyUI-LTXVideo dependency.** Measured 2026-09-25: every class
the LTX engines ask for is ComfyUI core, the pack's own registry (77 node ids)
holds none of them, and no OTR module imports its Python. If B4 publishes on a
box WITHOUT the pack, the provisioner's clone, its kornia pad patch and their
checks and tests, and the pack's mentions in DEPENDENCIES and the README come
out in one change. If B4 fails for want of the pack, that is the evidence the
measurement missed something: record which class, and the docs say why instead.

- Each is 1 act, `--timeout 0`. Record peak VRAM and any partial-load or
  offload lines; those are the 8 GB facts the 5080 cannot produce.
- **B3 on a box that already holds the AnimateDiff files** proves the lane and
  shows `EXISTING`; it does not prove the fresh download. Proving that needs a
  models root without those three files, which is the operator's call.
- A fix found here that touches shared code must show the 5080's path
  unchanged, measured (CLAUDE.md section 0B).

## Part C -- suite and Bug Bible (the suite runs at every push; the Bible is owed)

On whichever box is idle, after its GPU legs: the full suite with the Windows
venv (`$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`, backgrounded to a
log because of the 60 s tool ceiling), then the Bug Bible from its own repo
root. `EXPECTED_FAILED_NODEIDS` in `tests/conftest.py` is empty, so every red
prints as NEW. Two are inherited and fail on `d167f0ab` already, before this
wave's code: `tests/test_lane_preflight_matrix.py::test_g2_canvas_truth` and
`::test_g4_admission_honesty` (LTX 2.5 lanes missing from the canvas pins and
the cost rows). The 5080's dirty partial patch is aimed at that class. Explain
every other red.

**Result 2026-09-26 night (5080, HEAD `a2c36590`):** the suite is fully green --
16906 passed, 0 failed. The Bug Bible regression: 11 red, 38 green, and a
Sonnet triage against git history found **0 real OTR regressions**: 4 are
heuristic false positives (a prose match on `unload_all_models` in comments;
`proc.py` is the single spawn owner by design; output nodes route through
`_otr_paths`, which OTR's own ratchet REQUIRES; guarded depth-pinned dirname
fallbacks) and 7 are stale Bible pins on mechanisms OTR ripped on purpose (the
word-fit apparatus `314dd481`, deleted harness scripts `96a6bae8`, a retired
engine id, ShotLock's gate link now at slot 3 with the gate intact). The fixes
belong in the Bible repo (exclude `.claude/` from its scans -- a stale nested
agent worktree doubled four of the file lists -- and update or retire those
pins); that is a cross-pack contract, so it waits for the operator's word.

**Fixed 2026-09-26 on his word** (Bible `85cb0de2`, OTR `267e9f48`): the
regression against OTR now reads **46 passed, 0 failed** (14 skipped, 3
xfailed). The four checks look for a real CALL or the rule's own sanctioned
shape, and a synthetic pack with a bare `unload_all_models`, a four-deep
module-level chain, an unguarded `Popen` and an output node writing beside
its own file still fails all of them. The pins on ripped mechanisms skip with
the reason or follow the live code. ONE pin was right and OTR was wrong:
BUG-12.120 names a test proving the portrait identity seed -- every
`scene_character` still of one character draws that character's portrait
seed -- and the 09-04 rip (`47bf95d6`) had deleted it along with
`resolve_object_seed`, while the mechanism itself stayed live. Restored
against `resolve_seed_and_mode`.

**Final-HEAD receipts 2026-09-26 early morning (5080, HEAD `b545d1f2`, after
Sprint 2's code landed):** full suite 16973 passed, 0 failed (11:43); Bug
Bible 46 passed / 0 failed; a fresh server on ComfyUI 0.37.4 loads all 25
nodes; a 1-act `otr_8gb_low` leg through the headless runner PASSED in
9:49 and published `crack_repair_20260926_045130` -- the canonical with the
Start-here note and the credential node, end to end.


## Owed, not in this wave

- Six-act repeatability on one row.
- Chunked music: reachable only through an authored `scifi_news_pro` music
  row with `target_duration_s` above 22.
- Listening. The operator's ear only.
- Mac and RunPod legs.
