# Manual review request for the Comfy Registry -- draft (2026-09-02)

**File this AFTER `2.0.0-alpha.16` is published and its scan has run.** Before posting:
1. `GET https://api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true`
   and confirm alpha.16 shows ZERO `critical` findings (only `info`). If a critical remains,
   fix that first; a reviewer who sees `credential-access` will not approve.
2. Paste the real version ID from that response where marked.
3. Open it at https://github.com/Comfy-Org/registry-backend/issues/new (no template exists;
   the title shape below is the tracker's convention).

Context for whoever files it: every open "Manual review request" on that tracker (#184-#220)
is unanswered, but admin batch approvals do happen silently (`"Batch approved by admin"` in the
API's `status_reason`), so the issue is the only lever there is. Keep it factual and short.

**DECISION BEFORE FILING (operator).** `__init__.py` registers two unauthenticated POST routes,
`/otr/video_render_single` and `/otr/video_render_soak`: a JSON body picks the engine and the
portrait / audio file paths and a render thread starts. That is the class the registry banned
comfyui-easyuse-anima 1.1.2 for (`policy-v0.2: UNAUTHENTICATED_SIDE_EFFECT`), and the 09-01 ship
audit (registry-flag-04) refuted them only as the CAUSE of the flag, leaving their merit open.
Nothing that ships calls them; they are a hand-built GPU-gate harness. Options, best first:
(a) stop registering them on a registry install (gate behind `OTR_ENABLE_HTTP_RENDER_ROUTES=1`,
default off, so only the read-only ledger GET is registered), then say so in the Boundaries
bullet below; (b) file as-is with the disclosure below and accept the ban risk. Never file
without the disclosure: a reviewer who greps `routes.post` after reading "no routes" is done
with us.

---

**Title:** Manual review request: comfyui-old-time-radio 2.0.0-alpha.16 flagged on info-level YARA findings (env reads, ffmpeg subprocess, opt-in cloud lanes)

## Manual review request

- Registry: https://registry.comfy.org/nodes/comfyui-old-time-radio
- Registry API: https://api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true
- Publisher: `fluxus`
- Version: `2.0.0-alpha.16` (version ID: `<paste from the API>`)
- Current status: `NodeVersionStatusFlagged`. No version is Active, so `latest_version` resolves
  to null and ComfyUI Manager cannot install the pack at all.
- Repository: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch `v2.0-alpha`)

### What changed since the last flagged versions

2.0.0-alpha.13 through alpha.15 carried two `critical` `prohibited-string` findings: the writer
node declared the `auth_token_comfy_org` hidden input next to `api_key_comfy_org` (the API-nodes
convention) for its opt-in, default-off Comfy Credits lane. That was our mistake and it is removed
in alpha.16: the pack declares only `api_key_comfy_org`, and a regression test refuses the
session-bearer type name in every shipped file. The remaining findings are all `severity: info`.

### The info findings, grouped

All are `yara_scan`, `severity: info`. OldTimeRadio is a local radio-drama pipeline (script ->
TTS -> mix -> optional video), and the findings are its ordinary machinery:

1. `python_environment_manipulation` (about 100): `os.environ.get` reads of documented `OTR_*`
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

### Boundaries

- No use of the `eval` / `exec` builtins. A grep for `eval(` hits `ast.literal_eval` on one
  config value and PyTorch `.eval()` mode switches on loaded models; `exec(` hits nothing. No
  `os.system`, no `shell=True`, no runtime `pip`, no obfuscation, no bundled executables.
- HTTP routes, all in `__init__.py`: `GET` / `OPTIONS /otr/latest_ledger` returns the newest
  episode ledger JSON so a local dashboard can poll it (read-only). `POST /otr/video_render_single`
  and `POST /otr/video_render_soak` are a hand-built GPU test harness that starts a local render
  from a JSON body; nothing that ships calls them. (If option (a) above landed, replace this
  sentence with: they are registered only when `OTR_ENABLE_HTTP_RENDER_ROUTES=1` is set, so a
  registry install exposes only the read-only ledger GET.)
- No telemetry. The only outbound traffic is the opt-in lanes above, to their documented vendors.
- Output is written under ComfyUI's output directory (`otr/episodes`, `otr/obs`).

### Request

Please review and activate `2.0.0-alpha.16`. If any finding above needs a code change or a
permission declaration to pass automatically, tell us which and we will ship it. We would rather
drop a feature than keep the pack uninstallable.
