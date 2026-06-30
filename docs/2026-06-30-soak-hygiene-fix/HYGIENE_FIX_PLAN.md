# SOAK HYGIENE FIX -- the "11 failed but rendered" temp-leak (for kibitz)

## What happened (grounded)
The 2026-06-30 overnight combo soak rendered every engine and published a final mp4 to obs, but 11
legs scored `SOAK_FAIL` with: `run leaked otr_* entries into the system temp dir
'%LOCALAPPDATA%\\Temp': ['otr_scopes_pending_<ts>_<ts>.mp4']`. The RENDER is fine; a cleanliness gate
trips.

ROOT CAUSE (grounded):
- `nodes/otr_scene_aware_scopes.py:537` writes its scopes intermediate to
  `os.path.join(tempfile.gettempdir(), f"otr_scopes_{key}_{ts}.mp4")` -- the AMBIENT TEMP dir, and the
  file is NOT deleted after use (it persists).
- The soak's hygiene gate `assert_no_stray_writes` (`scripts/_otr_soak_capstone.py:225-276`) records
  the `otr*` entries in `%LOCALAPPDATA%\Temp` BEFORE the leg and FAILS the leg if any new `otr*` entry
  appears after. It assumes "the in-tree TEMP repoint held" (line 231) -- i.e. that the server was
  booted via `scripts/_otr_soak_server_launch.cmd`, which sets `TEMP=%OTR_TMP%` (the reserved
  `otr/episodes/_shared/tmp` tier). The overnight run used the ALREADY-RUNNING Desktop server (TEMP =
  system Temp), so the scopes file leaked there and tripped the gate.

The OTR tmp authority ALREADY EXISTS and is the intended sink:
- `nodes/_otr_paths.py:254 otr_shared_tmp_dir()` -> `otr/episodes/_shared/tmp` (the OH-3 janitor sweeps it).
- `nodes/_otr_video_engines/_tmp.py:30 otr_engine_tmp_mp4(prefix)` -> uses `_in_tree_tmp_dir()` and
  RAISES rather than silently leak to the system temp dir (the exact pattern to mirror).

## FIX (root cause, no shim)
1. `otr_scene_aware_scopes.py` -- write the scopes intermediate to the OTR-controlled tmp tier via the
   path authority (`_otr_paths.otr_shared_tmp_dir()`), NOT `tempfile.gettempdir()`, AND delete the
   intermediate after the final scopes mp4 is produced (it is a transient). Hygiene-clean REGARDLESS of
   the ambient TEMP env (robust whether or not a launcher repointed TEMP). Fail-soft if the tmp tier
   can't resolve (mirror `otr_engine_tmp_mp4`'s loud behavior; never silently fall back to system temp
   in production).
2. AUDIT the peer `otr_*` system-temp writers that PERSIST and could leak the same way:
   `nodes/video_engine.py:2201` (`otr_video_audio.wav`) and `nodes/scene_sequencer.py:1275` (gettempdir
   wav dir). Route persistent `otr_*` scratch through the path authority. CONFIRM the `mkdtemp(prefix=
   "otr_*")` callers clean up (rmtree) so they never persist: `otr_silent_composite.py:751`
   (otr_assemble_), `rtx_upscale.py:653` (otr_rtx_upscale_), `eng_mesh_stage.py` -- these are deleted in
   a finally today; verify + leave alone if so.

## ACCEPTANCE
- A scopes render leaves NO `otr_*` entry in `%LOCALAPPDATA%\Temp` (the gate's check) regardless of the
  TEMP env; the scopes intermediate lands in `otr/episodes/_shared/tmp` (or is deleted).
- Re-run the previously-failed legs (or a scoped scopes-only smoke) -> the hygiene gate passes -> GREEN.
- Suite + Bug Bible + B7 green; master audio byte-identical (the scopes path never touches the master).
- No workflow-JSON change (pure node-internal temp path).

## OPEN QUESTIONS FOR THE PANEL (ground vs the real code)
1. Does `otr_scene_aware_scopes` already delete its intermediate anywhere (so only the path needs
   changing), or does it persist by design (consumed by node 93 then orphaned)?
2. Are there OTHER persistent `otr_*` system-temp writers the gate would catch on a non-launcher server
   (full audit of `tempfile.gettempdir()` / fixed-name otr_* writes)?
3. Should the fix also make the hygiene gate itself robust (tolerate a launcher-independent run), or is
   routing the scratch to the OTR tier sufficient (preferred -- the gate is correct; the node was wrong)?
4. Any determinism / IS_CHANGED / cache-key implication of moving the scopes temp path?
