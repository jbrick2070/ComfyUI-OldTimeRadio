# Registry review request -- the SHORT one. Post this.

**PREPARED, NOT POSTED. Posting is a PUBLIC ACT and is yours alone.**
Version is filled in and live: `2.0.0-alpha.19` published and Pending as of
2026-09-05. Every number below is MEASURED from the real scan record, not
predicted.

Supersedes the two earlier drafts, which are now WRONG to send: they asked for a
review of finding COUNTS and never mentioned the ban. See the note at the
bottom.

---

**Title:** Re-review request: comfyui-old-time-radio `2.0.0-alpha.19` -- both banned
surfaces closed

Hi -- asking for a re-review of `comfyui-old-time-radio` (publisher `fluxus`).
Versions 2.0.0-alpha.13 and .14 were banned with:

> policy-v0.2: RCE (code execution) -- attacker-reachable via unauthenticated
> /prompt (node widget) or no-auth route

**You were right on both counts, and we have closed both.**

**The no-auth route.** `POST /otr/video_render_single` and
`/otr/video_render_soak` were registered unconditionally in alpha.13 -- they
took caller-supplied paths from an unauthenticated body and started a render.
They have been behind an opt-in env flag, default off, since 2026-09-03.

**The node widget.** Five nodes exposed an `ffmpeg` STRING widget whose value
became `argv[0]`. We reproduced it: a widget value beat the operator's own
`OTR_FFMPEG` pin, the ffprobe sibling rule turned it into a second attacker
binary, and our executable allowlist did not stop it because it only checked the
basename. In `2.0.0-alpha.19` each node discards that widget at its execute method,
and the resolvers now return an absolute path or nothing.

We also fixed, from our own review: a `replay_from` widget that accepted a UNC
path (SMB/NTLM coercion), a wildcard CORS header on an unauthenticated GET, and
an unescaped filename interpolated into an ffmpeg filtergraph.

**We also cut the noise, so the report is readable.** alpha.17 scanned 158
findings; alpha.18 scanned **12**, all `info`, zero critical -- by giving each
machine fact one owner in the source rather than asking you to overlook
anything:

| rule | alpha.17 | alpha.18 |
|---|---:|---:|
| `python_environment_manipulation` | 103 | 4 |
| `python_command_injection_risk` | 35 | 3 |
| `python_url_command_execution` | 12 | 0 |
| `python_network_operations` | 5 | 5 |
| `windows_process_manipulation` | 1 | 0 |
| `python_bytecode_manipulation` | 1 | 0 |
| `python_sensitive_file_access` | 1 | 0 |

**What remains is the render path.** This pack encodes video, so it runs ffmpeg
via `subprocess`; PyAV is FFmpeg in-process and our build lacks libass/drawtext,
and OpenCV here cannot write H.264. Zero is not reachable without deleting the
product. `comfyui-video-xy-plot` has four Active versions whose `status_reason`
reads `{"message": "subprocess: ffprobe"}` -- the same class.

Happy to walk through any of it. Is there a per-finding review path we should
use instead of asking version by version?

Thanks.

---

## Notes for the operator (not part of the post)

* ~320 words. Do not send either older draft.
* **The two earlier drafts are actively misleading now.** They never mention the
  BAN -- they ask for a review of finding counts, as though flagging were the
  problem. Asking for a re-review while the thing they banned us for is unfixed
  is the one way to burn the request. They also claim credit for fixing 2
  `critical` findings that were rule `prohibited-string` and were already gone
  as of alpha.16, and they credit a `python_url_command_execution` drop to
  reworded error strings -- that rule fires on `cmd = [...]` argv builders and
  was never touched.
* Do NOT predict scan counts. Read them from
  `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true` after
  the version has actually been scanned.
* The backing detail (per-class finding table, the boundaries list, the two
  files deliberately not migrated) is in `...-READY-v2.md`. Send it only if
  asked.
