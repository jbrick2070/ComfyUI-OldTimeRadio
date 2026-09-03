# NEWBUG sweep -- video-tiers kibitz arc (2026-07-20)

Static-review findings mined from the kibitz big logs (operator request 2026-07-20).
These are STATIC findings, NOT live-proven production failures -> they are NOT PBUGs
and do NOT auto-promote to the Bug Bible. Surfaced here so a cross-check window can
compare against past PBUGs / Bible before any live proof. (feedback_new_bug_problem_statement)

## FINDING 1 (real, low-moderate) -- wan_ti2v discards its measured VRAM peak
**Where:** `nodes/_otr_video_engines/eng_wan_ti2v.py` ~450-485 (render_clip).
**What:** `render_peak = probe.stop()` (VramPeakProbe) is computed and only written to
an INFO log (~483-484). The returned raw dict (~485) is `{"out_path", "frame_count"}`
-- the peak is never threaded into `canonicalize()` / `_clip_from_raw`, so the manifest
`vram_peak_mb` receipt (otr_video_render_batch.py:2821-2824 reads `clip.get("vram_peak_mb")`)
is ALWAYS None for wan_ti2v. render_driver.py:2229-2233 then falls back to a
less-accurate instantaneous post-render read.
**Impact:** the S-E5 recipe receipt / ledger "what VRAM did this beat use?" is blank for
the wan lane. No render break; a telemetry/observability gap.
**Problem statement (pasteable):**
> eng_wan_ti2v.render_clip measures render_peak via VramPeakProbe but returns only
> {out_path, frame_count}; the peak is logged, never placed in the raw dict, so
> canonicalize()->manifest vram_peak_mb is always None for wan_ti2v. Thread render_peak
> into the returned raw (raw["vram_peak_mb"] = render_peak) so _clip_from_raw stamps it,
> matching the eng_ltx_av receipt shape.
**Fix owner:** fold opportunistically -- eng_ltx_8gb will thread its peak (this build);
wan_ti2v's one-line thread can ride the same chunk (SW touch) or a separate tiny fix.

## FINDING 2 (real, low) -- hf_download_driver verifies neither revision nor integrity
**Where:** `scripts/hf_download_driver.py` ~55-105 (exact-file seam ~96-105).
**What:** the driver downloads an exact file but passes no `revision` to hf_hub_download
and does no byte-length / SHA verification, so a wrong-revision or truncated/corrupt
weight is accepted silently.
**Impact:** latent -- a bad download surfaces only later as a runtime load/decode error.
**Problem statement (pasteable):**
> scripts/hf_download_driver.py downloads exact files but passes no revision and verifies
> no checksum/byte length; extend it to accept + pass revision and to verify the
> materialized file size/SHA against expected, failing loud on mismatch.
**Fix owner:** folded into the video-tiers plan sec E (download_ltx_0_9_8.ps1 needs a
pinned revision + integrity check, so the driver extension lands there).

## BORDERLINE (noted, NOT classified active)
- Wan writes a temp mp4 then relies on a best-effort post-render move; a code comment
  (render_driver.py ~2624-2625) admits it "can re-create a stale folder and strand the
  provider clips." The path is guarded; raised only as an [ASSUMPTION] risk for the NEW
  adapter to avoid (eng_ltx_8gb follows the proven pattern + Test-Path verifies the
  final asset). Not an active bug on the current engines.

## Excluded
All other "defect/will-fail/mismatch" hits are critiques of the video-tiers PLAN
(public-id aliasing / resolve_engine_id / presets that do not exist yet) -- not existing
bugs. Historical BUG-LOCAL-*/BUG-070/291 markers in source are pre-existing, not new.
