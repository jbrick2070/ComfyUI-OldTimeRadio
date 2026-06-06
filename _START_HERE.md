# START HERE -- this repo's ACTIVE build = the OTR Open Video Model Platform

**The active build plan lives OUTSIDE this repo** at `C:\Users\jeffr\Documents\otr-video-roundtable\` -- read **`VIDEO_BUILD_HANDOFF.md`** there FIRST (anti-drift entry point), then `_START_HERE.md` / `SUBAGENT_TICKETS.md`.

## The ONE active build
The model-agnostic video platform (Subprojects A=video, B=3D, C=image-gen). First code = the A-Seam deltas (AS-1..AS-5), then Sprint 1 (the per-role A/B/C selector UI + Other-Beats clip-mode + the audio-derived Clip Budget Calculator). See the roundtable `SUBAGENT_TICKETS.md`.

## Do NOT resume any other sprint (the story-spine drift lesson)
Do NOT start / resume / "continue" story-spine, story-pipeline, audio, or any other ROADMAP item -- they are SHIPPED or PARKED. The audio refactor is SHIPPED + the audio ledger is FROZEN (read-only).

## What the in-repo planning docs ARE (NOT a build path)
`docs/`, `ROADMAP.md`, and `SPRINT.md` are SHIPPED/PARKED history and the **spec-of-record for EXISTING code** (live nodes/tests cite them in docstrings) -- keep them, but they are reference, NOT a plan to resume. New work is driven ONLY by the roundtable `VIDEO_BUILD_HANDOFF.md` + `SUBAGENT_TICKETS.md`. Truly-parked stray plans were moved to `docs/_ARCHIVE/` (2026-06-06).

Invariants stay in force: byte-identical master audio + mux-LAST; single resident heavy engine 14.5 GB / 14.0 GB 3D; 100% local/offline; cleanbreak; run Bug Bible + core + dropdown regression after each change.
