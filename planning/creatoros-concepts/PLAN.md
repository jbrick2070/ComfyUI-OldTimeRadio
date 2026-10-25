# CreatorOS concepts -> OTR: coding plan

Baseline `main` 90413ac3, ComfyUI core 0.42.1, frontend 1.55.18, 2026-10-24. Every seam below was
read in the real files; re-read them at the commit you start from.

**Shipped (2026-10-24), receipts in the commit messages:** S1 rows 1.1 to 1.3 (65f1fa29, 8e5b37b7,
cc85dc49), S2 rows 2.1 and 2.2 (2b268d5f), S3 rows 3.1 to 3.3 in the Studio (74a5bcb8; 3.4 and
3.5 judged not worth building, `S3_STUDIO_PROPOSALS.md`). S4, the broadcast feed, shipped
(26cfc269) and was removed the same day by the operator: OBS and his new OBS plugin watch the one
folder `otr/obs/`, where every episode lands. What is left is below.

**Where the ideas come from.** [thePloggy/CreatorOS-Studio](https://github.com/thePloggy/CreatorOS-Studio),
Apache-2.0, a Node/React studio that drives a local ComfyUI (built for Comfy's #OpenSourceHiggsfield
challenge). No CreatorOS code is copied: every row re-implements an idea in OTR's own Python/JS. If
code is ever copied verbatim it keeps its Apache-2.0 header and a change note, which is compatible
with OTR's MIT licence.

## Scope rules (hold for every row)

1. **No Ollama, no llama.cpp.** Nothing here adds, calls or depends on either. CreatorOS's Ollama
   features (GPU hand-off between Ollama and ComfyUI, prompt rewriter, vision checks, CPU fallback)
   are out.
2. **`otr/obs/` is untouched.** Every finished episode still publishes there (CLAUDE.md 2026-08-17;
   mux ruling 2026-10-15 at `nodes/otr_master_audio_mux.py:2411`). No row moves, hides, sorts or
   withholds an obs file.
3. **No fallbacks.** A value the code cannot read is reported as unknown, with the reason. A
   display-only feature (live view, notification, hint) never changes or gates a render.
4. **Story and music are done** (CLAUDE.md 2026-08-04, 2026-09-17). No row touches prompts or prose.
5. **Wired in the same change.** Grep for a non-test caller of every new helper before pushing.
6. **No new widgets planned.** If a row ends up needing one: append-only, then
   `build_variants.py --all` and the four checks in CLAUDE.md section 0.
7. **Two boxes.** `ListAgents` + `SendMessage` the peer window before touching a file it owns. State
   the 5080 / 4060 blast radius in every commit message (CLAUDE.md 0B).
8. **No mid-run approvals (operator, 2026-10-24).** Nothing in this plan pauses a run to wait for a
   person. Every row runs unattended end to end: the live view, hints and pings only report; the
   environment stamp and the known-good snapshot write themselves; the drift report runs on demand.
   A failure in any of these is reported and never holds the render or the publish.

## What we are not taking, and why

| CreatorOS feature | Why not |
|---|---|
| Setup wizard, model manifest, per-recipe missing node/model check | Already stronger here: `scripts/otr_provision.py` (installs, prints a receipt), `scripts/otr_asset_index.py` (generated "to use X, download Z"), `js/lane_node_packs_core.js` (missing pack named when the workflow opens) |
| Render watchdog, ComfyUI link state | `scripts/otr_render_watchdog.ps1` (heartbeat stall, queue down) |
| Per-asset provenance, replayable recipe | The ledger plus replay bundles (`scripts/otr_freeze_replay_bundle.py`, `scripts/otr_verify_replay.py`) |
| Redo one shot, keep the rest, re-cut | Scene Producer: kept clips, takes, reuse by causal key (`nodes/otr_video_render_batch.py:429` `_producer_clips`, `:663`) |
| ETA learned from past runs | Radio Play Studio already calibrates from saved timings (`contest/radio-play-studio/radio_play_studio/runner.py:536`, `save_timing` at `:560`) |
| Re-attach to a job after a restart | Studio looks for a lost submit in ComfyUI's queue and history (`runner.py:58` `HISTORY_LOOKBACK`, `:762`) |
| Prompt files re-read on every job | Its value is prompt iteration, and story quality is done |
| Ollama GPU hand-off, CPU fallback | Rule 1; a silent CPU fallback also breaks rule 3 |
| Reject = move the file out of the library | Rule 2 |
| Per-recipe wiring sidecars (`.pilotage.json`) | One canonical graph |
| Public-demo owner key, visitor stats, React/mobile shell | Not OTR's shape |

## The plan at a glance

| Sprint | Left | Touches |
|---|---|---|
| **S1 Headless ops** | 1.1 residual: previews on the LTX video walk | measurement only |

## S1 residual

### 1.1 Previews on the LTX video walk

`--preview-file` is shipped (cc85dc49). Measured on `otr_16gb_still` (MOSS-Image): 425 preview frames,
the last 74,135 bytes, with `latent2rgb`. Not yet measured on the LTX video walk, where
`QuietInnerBars` may silence them: a one-act canonical leg (`--act-count 1`) on `otr_16gb_video`
with `--preview-file`, then read `LIVE ... preview_nodes=` in the leg log. Frames arrive: record the
count here and delete this row. None: record that, and delete the row.

Delete each row from this file when it ships. Its receipt goes in the commit message, not here.
