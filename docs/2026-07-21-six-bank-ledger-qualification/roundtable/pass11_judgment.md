# Pass 11 judgment -- positioned crossfade timeline repair

## Verdict

The terminal mux is correct. Fix the shared positioned video timeline, do not
widen tolerance and do not disguise body drift as credits duration.

Antigravity remained review-only on exact `gemini-3.5-flash-high` in the clean
detached worktree
`otr-kibitz-livebug12-crossfade-tail-20260721-flash-high`. Sol checked the
review against the real Windows repository and was the sole driver, coder, and
judge.

## Grounded findings

Accepted:

- `build_clip_manifest` treated the full per-shot work sum as final duration;
- POSITION planning ignored overlap ownership and emitted every full request;
- the master probe could grow but not shrink an oversized positioned total;
- timeline QA would call intentional overlap trimming an underrun/mismatch;
- all six banks share this post-audio video tail.

Corrected by Sol:

- `total_episode_dur_s` is already durable post-audio authority carried to the
  manifest, so a new wire is unnecessary;
- the safe interval is `[start, min(requested_end, next_start, timeline_end))`,
  not blindly `next_start - start`, which would stretch short clips across
  genuine gaps;
- upward quantization with `ceil(duration * fps)` covers the last fractional
  audio frame while leaving the terminal three-frame tolerance unchanged;
- the filesystem master is a cross-check/fallback, not a replacement owner.

Discarded:

- `media_archive` is not a special non-crossfade lane;
- `scifi_news_pro` is a content/model lane, not a higher-resolution media tail;
- the relevant planner is not a nonexistent generic `video_engine.py` path;
- the intentional all-15-shot render warning is not a failure;
- the known LTX-open soft-fallback warning is unrelated.

## Implemented contract

- manifest `render_target_frames` records full requested work;
- `timeline_total_frames` / compatible `total_target_frames` record final
  positioned output, derived from durable accepted duration when available;
- stable positioned slots remove duplicate crossfade frames and preserve gaps;
- requested/rendered/visible/trimmed QA fields prevent false underrun failures;
- actual-master reconciliation may shrink positioned manifests but retains the
  old sequential grow-only path;
- no workflow, mux-tolerance, or credits-ownership change was made.

## Verification state

The exact failed durable ledger replays as 2,889 full render-work frames, 2,864
authoritative positioned output frames, and exactly 25 overlap-trimmed frames;
the new visible-frame QA is green. The focused render-driver, composite,
clip-fill, and real-ffmpeg gate passed 96/96. The full project suite passed
8,328 tests with 33 skipped and one expected failure. The standard Bug Bible
gate passed 17 tests with 22 route-local skips and three expected failures; the
isolated BUG-12.69 OTR guard passed. Canonical workflow validation is green at
23 nodes / 58 links with no widget drift, valid link/input ownership, and
unchanged SHA-256
`f9d9c2c3a101ec607c9658456f6e191a164d8214be7b6d560bc68975d0511e9a`.

Git and live six-bank 320-word qualification receipts follow this judgment.
