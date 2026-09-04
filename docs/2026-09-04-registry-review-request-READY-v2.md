# Registry manual-review request -- v2 draft, POST-COLLAPSE

**STATUS: PREPARED, NOT POSTED. Posting is a PUBLIC ACT and is the operator's
alone.** This supersedes `docs/2026-09-03-registry-review-request-READY.md`, which
is kept as the alpha.17 record. Do not post that one -- its numbers are now stale
in our favour.

**BEFORE POSTING, THREE THINGS MUST BE FILLED IN, and only the operator can
produce them** (they need a published version):

1. `<VERSION>` -- the first non-alpha version string, after the
   `pyproject.toml` bump. Every `<VERSION>` below.
2. `<SCAN COUNTS>` -- the REAL counts from
   `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true`
   once that version has been scanned. **Do not predict them.** The table below
   carries our measured source-side change, which is a different claim from what
   the scanner reports, and conflating the two is exactly the mistake that would
   make a reviewer distrust the rest.
3. The `0 critical` claim -- re-verify against that scan, do not carry it over.

---

**To:** Comfy-Org registry admins
**Title:** Manual review request: comfyui-old-time-radio `<VERSION>` --
`<SCAN COUNTS>`, no Active version

**Ask:** review and activate `<VERSION>`. No version of this pack has ever been
Active, so `latest_version` resolves to `null` and ComfyUI Manager reports
"Cannot resolve install target" -- the pack is uninstallable through the
registry today.

| | |
|---|---|
| Node id | `comfyui-old-time-radio` |
| Publisher | `fluxus` |
| Version | `<VERSION>` |
| Repo | https://github.com/jbrick2070/ComfyUI-OldTimeRadio |
| Scan | `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true` |

## What changed since alpha.17, and why we are asking now

alpha.17 scanned **158 info findings**. We did not come back asking you to
overlook them. We went and collapsed the ones that could be collapsed, by giving
each machine fact exactly one owner in the source, and we are asking about what
is left.

**This is the measured source-side change, not a prediction of your scanner's
output:**

| what the scanner keys on | alpha.17 | `<VERSION>` | how |
|---|---:|---:|---|
| files that spell `os.environ` (rule fires per FILE) | 103 | **4** under `nodes/` + the root | one owner module, `nodes/_otr_shared/env.py`; 99 files now ask it |
| `subprocess` spawn SITES (rule fires per SITE) | 35 in 20 files | **3 in 2 files** | one owner, `nodes/_otr_shared/proc.py`, with an executable allowlist |
| `kernel32.OpenProcess` | 1 | **0** | deleted; `psutil.pid_exists` was already the primary path |
| `__import__("sys")` | 1 | **0** | it was never dynamic -- the argument was a literal; now a plain import |
| `ffprobe -count_frames` inside error STRINGS | 6 | **0** | reworded to "the frame-count probe"; the argv is unchanged |

Two of the four remaining `os.environ` files are the owner itself and
`prestartup_script.py`, which runs before the pack is a package and has no owner
to import. One further file, `tools/engine_matrix.py`, is a developer tool that
ships only because `.comfyignore` excludes neither `tools/` nor `config/`; it
sets a test-mode flag before putting the repo on `sys.path`, deliberately, so no
adapter reaches for a GPU while it is only reading declarations. Nothing shipped
imports it. The other two are held by contracts we chose not to break for a
scanner count: one is byte-hashed by a voice-qualification record, so editing it
by any byte demotes a voice the operator approved by ear; the other is a leaf
module whose test asserts it imports nothing from the pack, because a pack import
reintroduces an import cycle that once left two subsystems running blind.

Both owners are stdlib-only, read live, and change no value: a caller's default,
cast and precedence stay at the call site. The migration was semantics-neutral by
construction and is covered by AST guards that fail the build if a new
`os.environ` or `subprocess` site appears anywhere under `nodes/`.

## What remains, and why it is irreducible

The remaining findings are `yara_scan`, `severity: info`. Some carry a
`credential-access` admin tag; that tag sits on `os.environ.get` reads of
`HF_TOKEN` and similar, not on any credential the pack transmits. After the
collapse that tag is on **two** files instead of eleven.

**Zero is not reachable for this pack, and we can say exactly why.** This is a
radio-drama renderer: it encodes video. Encoding video means running ffmpeg, and
running ffmpeg means `subprocess`. We checked for compliant replacements and
found none:

- PyAV is FFmpeg in-process -- the same library -- and this build ships no
  `libass`/`drawtext`, so caption burn and the credits roll cannot move to it;
- OpenCV in this environment cannot write H.264;
- torchvision's writer is a deprecated PyAV wrapper.

So the honest position is not "we could reach zero and would rather not". It is
"the remaining findings are the render path, and deleting them deletes the
product".

## Boundaries

- No `eval` / `exec` builtins. (`eval(` greps to `ast.literal_eval` on one config
  value and PyTorch `.eval()`; `exec(` greps to nothing in shipped code.)
- No `os.system`, no `shell=True`, no runtime `pip`, no obfuscation, no bundled
  executables, no telemetry.
- The process owner refuses `shell=True` and a string argv outright, and checks
  `argv[0]` against a named allowlist (ffmpeg, ffprobe, git, nvidia-smi, blender,
  and a sidecar venv's own python) before spawning anything.
- HTTP routes, all in `__init__.py`: `GET /otr/latest_ledger` is read-only.
  `POST /otr/video_render_single` and `/otr/video_render_soak` are a local test
  harness, registered only when `OTR_ENABLE_HTTP_RENDER_ROUTES=1`. A registry
  install exposes only the read-only GET.
- Output stays under ComfyUI's output directory.

## Precedent

`comfyui-video-xy-plot` has four Active versions (1.0.1, 1.0.2, 1.0.4, 1.0.5).
Their `status_reason` shows `NodeVersionStatusFlagged` followed by:

```json
{"message": "subprocess: ffprobe", "by": "dr.lt.data@gmail.com", ...}
```

A pack flagged for shelling out to ffprobe was reviewed and activated, with the
reason recorded on the version. That is the same finding class as our largest
remaining group. We are asking for that review, not for a policy change.

## Unrelated, so it is not misread

Our page shows "No nodes found". That is the `node-pack-extract` pipeline, not
this scan -- it appears to have produced no successful extract for any pack since
2026-04-28 (filed separately as #203). The pack registers 34 nodes on a local
install. Mentioned only so an empty panel is not read as a broken pack.

## Request

Activate `<VERSION>`, or name the findings that need to change and we will ship
the change. We would rather drop a feature than stay uninstallable.

We would also value one answer for the future: is there a node-level or
per-finding review path we should be using instead of asking version by version?
That would cost you less than us returning for each publish.
