# Registry review request for `2.0.0-alpha.20`. POST THIS ONE.

**PREPARED, NOT POSTED. Posting is a PUBLIC ACT and is yours alone.**

Supersedes every earlier draft, all of which are now DO-NOT-SEND: they name
`alpha.19`, and alpha.19 ships an arbitrary file read and nine unconfined write
sinks. Asking for a review of that artifact would have invited a third ban.

**Where:** a new issue at https://github.com/Comfy-Org/registry-backend/issues
(the ~20 "Manual review request" issues filed there since 2026-08-02 have no
maintainer replies, so treat this as a courtesy note, not the mechanism. The
mechanism is the code, and the code is published.)

**Verified before writing this:** the published zip is byte-identical to commit
`3d133ec1` (809/809 files), every fix below is present in the artifact itself,
and alpha.20's own scan record has been read.

---

**Title:** Re-review request: comfyui-old-time-radio `2.0.0-alpha.20` — the two
banned surfaces plus four more classes, closed and published

Hi — asking for a re-review of `comfyui-old-time-radio` (publisher `fluxus`).
Versions 2.0.0-alpha.13 and .14 were banned with:

> policy-v0.2: RCE (code execution) — attacker-reachable via unauthenticated
> /prompt (node widget) or no-auth route

**You were right on both counts, and both have been closed since alpha.19.**
The no-auth render routes were env-gated in alpha.17 and are **deleted outright
in alpha.20** — they read caller-supplied paths from an unauthenticated body and
started a background render, and nothing that ships called them. The node widget
that became `argv[0]` is discarded at each node's execute method, and the ffmpeg
resolvers now return an absolute path or nothing.

**Since then we audited ourselves against your own published verdicts rather
than against the scanner, and alpha.20 closes four more classes:**

- **Write-side path traversal (Rule 11).** Nine sinks where a value declared in
  `INPUT_TYPES` decided a filesystem destination with no containment — including
  the non-obvious ones, where the destination widget is empty and the node
  derives the output path from an *input* path, so confining the output widget
  alone would not have closed it. All nine now resolve both sides and require
  the destination under the ComfyUI output root, the operator's configured
  publication root, ComfyUI's own input directory, or an explicit env allowlist.
- **Arbitrary file read.** A ledger-carried `pool_path` was copied verbatim into
  a directory ComfyUI serves over `/view`, behind nothing but an `isfile` check
  and reachable with no GPU work. The copier now requires the source inside the
  run's own output subtree, a real PNG, past a size floor.
- **UNC/SMB coercion.** Several `IS_CHANGED` fingerprint hooks and one JSON-carried
  path still reached `stat`/`open` on a caller-named host before any guard ran.
  Refused now, in the hooks as well as the execute methods.
- **Information disclosure.** The one remaining unauthenticated route (a
  read-only `GET` of the current episode ledger, no request parameters, no side
  effect) no longer returns an absolute path or raw exception text.

**What remains, and why.** alpha.20 scans as **12 findings, all `info`, zero
critical** — 4 environment reads, 5 network operations, 3 `subprocess` calls.
They are the product: this pack encodes video, so it runs ffmpeg via
`subprocess` (PyAV is FFmpeg in-process and our build lacks libass/drawtext, and
OpenCV here cannot write H.264), it reads its own configuration from environment
variables, and its optional cloud lanes are opt-in and default-off. Every
`subprocess` call goes through one gateway that refuses `shell=True`, refuses a
string argv, and allowlists `argv[0]`; no network host is settable from a node
widget. Zero findings is not reachable without deleting the product.

Happy to walk through any of it, or to point at specific commits. Is there a
per-finding review path we should use instead of asking version by version?

Thanks.

---

## Notes for the operator (not part of the post)

* ~430 words. Post the section between the horizontal rules only.
* **Every number in it was measured**, not predicted: alpha.20's scan record was
  read from
  `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true`
  after it landed, and it is 12 `info` / 0 critical — identical to alpha.19's,
  which is the point: the info findings were never what the ban was about.
* **Do not send any earlier draft.** `...-SHORT.md`, `...-READY-v2.md` and the
  09-03 draft all name a version that ships the file read.
* **`Flagged` is the expected state** for a version with any finding; it is not a
  failure and it is not a reason to change more code. The transition that matters
  is the human `reviewed SAFE (GOAL2 verify-deep, policy-v0.2)`, which is what
  moves a version to `Active`.
* The claim about approved packs carrying findings is checkable if anyone asks:
  of 102 versions approved under policy-v0.2 between 2026-08-15 and 09-01, none
  had a clean scan; 31 carried `python_command_injection_risk`.
