PASTE AS A NEW ISSUE: https://github.com/Comfy-Org/registry-backend/issues
Jeffrey posts it. Title first, then everything from "## Manual review request".

Separate from #203 (node extraction / empty NODES panel), which is already filed.

------------------------------------------------------------------------------

**Title:** Manual review request: comfyui-old-time-radio 2.0.0-alpha.17 -- 158 info findings, 0 critical, no Active version

## Manual review request

**Ask:** review and activate `2.0.0-alpha.17`. It scans **0 critical, 0 high, 0
medium, 0 low, 158 info**. No version of this pack is Active, so
`latest_version` is `null` and ComfyUI Manager cannot install it at all.

| | |
|---|---|
| Node | `comfyui-old-time-radio` |
| Publisher | `fluxus` |
| Version | `2.0.0-alpha.17` |
| Version ID | `57edb190-794e-487d-89c1-ad52bab106c1` |
| Status | `NodeVersionStatusFlagged` |
| Source | https://github.com/jbrick2070/ComfyUI-OldTimeRadio (`v2.0-alpha`) |
| Scan | `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true` |

OldTimeRadio is a local radio-drama pipeline: script -> TTS -> mix -> optional
video.

### Track record

alpha.13-15 carried two `critical` findings: our writer node declared an
`auth_token_comfy_org` hidden input beside `api_key_comfy_org` for an opt-in,
default-off lane. Our mistake. Removed in alpha.16, with a regression test that
now refuses that type name in every shipped file. alpha.16 and alpha.17 both
scan 0 critical. alpha.17 additionally puts two POST harness routes behind an
opt-in env flag, off by default.

### The 158 info findings

All `yara_scan`, `severity: info`. Some carry a `credential-access` admin tag;
that tag is on `os.environ.get` reads of `HF_TOKEN` and similar, not on any
credential the pack transmits.

| count | class | what it is |
|---:|---|---|
| 103 | `python_environment_manipulation` | `os.environ.get` on documented `OTR_*` knobs and standard `HF_HOME` / `HF_TOKEN` / `HF_HUB_OFFLINE`; a few process-local writes. Nothing persists outside the process. |
| 35 | `python_command_injection_risk` | `subprocess.run` / `Popen`, argument lists, `shell=False`: ffmpeg/ffprobe for mux, captions, credits; three optional TTS engines in isolated venvs. |
| 12 | `python_url_command_execution` | Same calls. Several are ffmpeg log strings, not calls. |
| 5 | `python_network_operations` | `requests` / `urllib` on opt-in, default-off cloud lanes and an RSS fetcher. Endpoints are constants or env overrides; no widget accepts a URL. |
| 1 | `windows_process_manipulation` | `kernel32.OpenProcess(SYNCHRONIZE, ...)` to test whether a sibling process holding a GPU lease is alive. |
| 1 | `python_sensitive_file_access` | sha256 of a downloaded model file, to verify it. |
| 1 | `python_bytecode_manipulation` | a `sys.modules` lookup. No bytecode is touched. |

### Boundaries

- No `eval` / `exec` builtins. (`eval(` greps to `ast.literal_eval` on one config
  value and PyTorch `.eval()`; `exec(` greps to nothing.)
- No `os.system`, no `shell=True`, no runtime `pip`, no obfuscation, no bundled
  executables, no telemetry.
- HTTP routes, all in `__init__.py`: `GET /otr/latest_ledger` is read-only.
  `POST /otr/video_render_single` and `/otr/video_render_soak` are a local test
  harness, registered only when `OTR_ENABLE_HTTP_RENDER_ROUTES=1`. A registry
  install exposes only the read-only GET.
- Output stays under ComfyUI's output directory.

### Why we are asking instead of fixing it

Zero findings is not reachable for this pack. Measured: **0** of the 158 are in
files `.comfyignore` already excludes; **156 of 158 are inside `nodes/`**, across
**107 of ~270 shipped modules**, at most 7 per file. The largest class is
`os.environ.get` (451 call sites, 101 files). Removing every subprocess and
network finding would still leave 106. Reaching zero means deleting roughly 40%
of the pack.

We checked for compliant replacements and found none: PyAV is FFmpeg in-process
(same library, and this build has no `libass`/`drawtext`, so caption burn cannot
move); OpenCV here cannot write H.264; torchvision's writer is a deprecated PyAV
wrapper.

### Precedent

`comfyui-video-xy-plot` has four Active versions (1.0.1, 1.0.2, 1.0.4, 1.0.5).
Their `status_reason` shows `NodeVersionStatusFlagged` followed by:

```json
{"message": "subprocess: ffprobe", "by": "dr.lt.data@gmail.com", ...}
```

A pack flagged for shelling out to ffprobe was reviewed and activated, reason
recorded on the version. Same finding class as our largest subprocess group. We
are asking for that review, not for a policy change.

### Unrelated, so it is not misread

Our page shows "No nodes found". That is the `node-pack-extract` pipeline, not
this scan -- it appears to have produced no successful extract for any pack since
2026-04-28 (filed separately as #203). The pack registers 34 nodes on a local
install. Mentioned only so an empty panel is not read as a broken pack.

### Request

Activate `2.0.0-alpha.17`, or name the findings that need to change and we will
ship the change. We would rather drop a feature than stay uninstallable.
