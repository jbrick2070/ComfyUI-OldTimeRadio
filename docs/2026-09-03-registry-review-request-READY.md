PASTE THIS AS A NEW ISSUE ON https://github.com/Comfy-Org/registry-backend/issues
Jeffrey posts it; Claude does not. Title line first, then everything after it.

Separate from issue #203 (node extraction) -- that one is about the empty NODES
panel and is already filed. This one is about the pack being uninstallable.

------------------------------------------------------------------------------

**Title:** Manual review request: comfyui-old-time-radio 2.0.0-alpha.17 flagged on info-level YARA findings (env reads, ffmpeg subprocess, opt-in cloud lanes)

## Manual review request

- Registry: https://registry.comfy.org/nodes/comfyui-old-time-radio
- Registry API: https://api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true
- Publisher: `fluxus`
- Version: `2.0.0-alpha.17` (version ID: `57edb190-794e-487d-89c1-ad52bab106c1`)
- Current status: `NodeVersionStatusFlagged`. No version is Active, so `latest_version` resolves
  to null and ComfyUI Manager cannot install the pack at all.
- Repository: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch `v2.0-alpha`)

### What changed since the last flagged versions

2.0.0-alpha.13 through alpha.15 carried two `critical` `prohibited-string` findings: the writer
node declared the `auth_token_comfy_org` hidden input next to `api_key_comfy_org` (the API-nodes
convention) for its opt-in, default-off Comfy Credits lane. That was our mistake. It was removed
in alpha.16 -- the pack declares only `api_key_comfy_org`, and a regression test refuses the
session-bearer type name in every shipped file -- and alpha.16 duly scanned with **0 critical
findings**, down from 2. alpha.17 additionally puts the two POST harness routes described under
Boundaries behind an opt-in environment flag that is off by default. Every remaining finding is
`severity: info`.

### The info findings, grouped

All 158 are `yara_scan`, `severity: info`; there are no `low`, `medium`, `high` or `critical`
findings. Some carry admin tags including `credential-access` -- that tag is attached to
`os.environ.get` reads of `HF_TOKEN` and similar, not to any credential the pack transmits.
OldTimeRadio is a local radio-drama pipeline (script -> TTS -> mix -> optional video), and the
findings are its ordinary machinery (counts read from alpha.17's own scan on 2026-09-03, not carried over):

1. `python_environment_manipulation` (103): `os.environ.get` reads of documented `OTR_*`
   operator knobs and the standard `HF_HOME` / `HF_TOKEN` / `HF_HUB_OFFLINE`, plus a handful of
   process-local writes (`OTR_OUTPUT_DIR`, `HF_HOME`, `OTR_ACTIVE_PROFILE`, one feature flag).
   Nothing persists outside the ComfyUI process.
2. `python_command_injection_risk` (35) and `python_url_command_execution` (12):
   `subprocess.run` / `Popen` with argument lists and `shell=False`, running ffmpeg / ffprobe for
   muxing, captions and the credits roll, and launching three TTS engines (Chatterbox, Dia,
   IndexTTS2) in their own isolated virtualenvs. Several of the 12 are ffmpeg log strings, not
   calls.
3. `python_network_operations` (5): `requests` / `urllib` on the opt-in cloud lanes (OpenRouter
   own-key, Google API, ElevenLabs, Comfy partner media) and the RSS/Atom fetcher for the news
   bank. Every endpoint is a constant or an env override; no node widget accepts a URL or host,
   and nothing is fetched unless the user enables the lane.
4. One each: `windows_process_manipulation` (`kernel32.OpenProcess(SYNCHRONIZE, ...)` to check
   whether a sibling process holding a GPU lease is still alive), `python_sensitive_file_access`
   (sha256 of a downloaded model file to verify it), `python_bytecode_manipulation` (a
   `sys.modules` lookup; no bytecode is touched).

Totals as scanned (alpha.17, 2026-09-03): 103 + 35 + 12 + 5 + 1 + 1 + 1 = 158.

### Boundaries

- No use of the `eval` / `exec` builtins. A grep for `eval(` hits `ast.literal_eval` on one
  config value and PyTorch `.eval()` mode switches on loaded models; `exec(` hits nothing. No
  `os.system`, no `shell=True`, no runtime `pip`, no obfuscation, no bundled executables.
- HTTP routes, all in `__init__.py`: `GET` / `OPTIONS /otr/latest_ledger` returns the newest
  episode ledger JSON so a local dashboard can poll it (read-only). `POST /otr/video_render_single`
  and `POST /otr/video_render_soak` are a hand-built GPU test harness that starts a local render
  from a JSON body; nothing that ships calls them. They are registered only when
  `OTR_ENABLE_HTTP_RENDER_ROUTES=1` is set, so a registry install exposes only the read-only
  ledger GET.
- No telemetry. The only outbound traffic is the opt-in lanes above, to their documented vendors.
- Output is written under ComfyUI's output directory (`otr/episodes`, `otr/obs`).

### Why we are asking rather than fixing it ourselves

We checked whether we could reach a zero-finding scan and we cannot. Of the 158
findings, **zero** are in files our `.comfyignore` already excludes -- the bundle
is as small as it can be. 156 of the 158 are inside `nodes/` itself, spread
across **107 of the pack's ~270 shipped modules**, at most 7 in any one file. The
dominant class is `os.environ.get` on documented operator knobs; the next is
`subprocess` with argument lists and `shell=False` to run ffmpeg. Reaching zero
would mean deleting roughly 40% of the pack's modules, which would not leave a
working node pack.

We are not asking for an exception to the rules. If any of these findings
represents a real risk, we would rather change the code -- please tell us which.

### Request

Please review and activate `2.0.0-alpha.17`. If any finding above needs a code change or a
permission declaration to pass automatically, tell us which and we will ship it. We would rather
drop a feature than keep the pack uninstallable.
