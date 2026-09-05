# Deep-research prompt -- hand this to an external research agent as-is

Everything the agent needs is inline. It needs web access only. Do not give it
filesystem access, and it has no reason to clone a repository or download a
binary -- reading files over HTTP is enough.

---

I need you to determine what makes the ComfyUI Registry's automated security
scanner emit a finding, precisely enough that I can predict for any given line
of Python whether it will flag.

## Background, all of it measured against live systems on 2026-09-05

I publish a ComfyUI custom node pack to registry.comfy.org. Every version I
publish comes back `Flagged`. While no version is `Active`, the pack's
`latest_version` field resolves to `null`, and ComfyUI Manager therefore refuses
to install it at all. I need one version to reach `Active`.

**The promotion logic is public and trivial.** In `Comfy-Org/registry-backend`,
file `services/registry/registry_svc.go`, function `PerformSecurityCheck`:

```go
issues, err := sendScanRequest(s.config.SecretScannerURL, nodeVersion.Edges.StorageFile.FileURL)
if issues == "" {
    // "No security issues found ... Updating to active."
    SetStatus(schema.NodeVersionStatusActive).SetStatusReason("Passed automated checks")
} else {
    // "Security issues found ... Updating to flagged."
}
```

`sendScanRequest` just POSTs `{"url": "<zip url>"}` to `config.SecretScannerURL`
and returns the raw response body. So:

- `issues == ""` is the entire test. A clean scan auto-promotes, no human.
- It is ALL-OR-NOTHING. One finding flags a version exactly as thirteen do.
  Reducing 13 findings to 1 is worth nothing.
- `SecretScannerURL` has no default committed in that repo.

There is also a human path: some packs show
`status_reason = {"message":"Approved by admin","by":"dr.lt.data@gmail.com"}`.
I cannot rely on that -- roughly 20 manual review issues since 2026-08-02 have
had zero maintainer replies.

## What my findings look like

Every finding names the exact YARA string identifier that matched. My 13, all
severity `info`, zero critical:

| file | rule (`issue_type`) | matched patterns |
|---|---|---|
| `prestartup_script.py` | python_environment_manipulation | `$env_read1` `$env_read2` `$env_mod1` |
| `nodes/_otr_writer_heartbeat.py` | python_environment_manipulation | `$env_read2` |
| `nodes/_otr_audio_engines/eng_indextts2.py` | python_environment_manipulation | `$env_read2` `$env_read3` |
| `nodes/_otr_shared/env.py` | python_environment_manipulation | `$env_read1` `$env_read2` `$env_mod1` `$env_mod4` |
| `scripts/_otr_idx_download_weights.py` | python_environment_manipulation | `$env_read2` |
| `nodes/_otr_comfy_backend.py` | python_network_operations | `$http1` |
| `nodes/_otr_feed_fetch.py` | python_network_operations | `$socket1` `$socket3` `$socket_stage_assign` |
| `nodes/_otr_openrouter_backend.py` | python_network_operations | `$http1` |
| `nodes/_otr_google_api/client.py` | python_network_operations | `$http2` |
| `nodes/_otr_shared/cloud_media_invoke.py` | python_network_operations | `$http2` |
| `nodes/_otr_audio_engines/eng_indextts2.py` | python_command_injection_risk | `$subprocess_popen_direct` |
| `nodes/_otr_shared/proc.py` | python_command_injection_risk | `$subprocess_run_direct` |
| `nodes/_otr_shared/proc.py` | python_command_injection_risk | `$subprocess_popen_direct` |

A full finding object, for shape:

```json
{
  "issue_type": "python_environment_manipulation",
  "description": "Detects environment variable manipulation and reading",
  "file_path": "scripts/_otr_idx_download_weights.py",
  "line_number": 70,
  "line_snippet": "env = os.environ.get(\"OTR_INDEXTTS2_DIR\")",
  "admin_tags": ["system-modification"],
  "metadata": {
    "confidence": 90,
    "matched_patterns": ["$env_read2"],
    "yara_matches": [{"filepath": "...", "strings": [{"identifier": "$env_read2", "instances": [...]}]}]
  }
}
```

Confidence is 90 for env and network rules, 95 for the subprocess rule.

## The control that breaks every obvious theory

`rgthree-comfy` version `1.0.2608210019`, created **2026-08-21** (after a rule
change in late August 2026 that started flagging my pack), has
`status_reason` = **`Passed automated checks`** -- it auto-promoted on a CLEAN
scan. I downloaded its published bundle and grepped it. It contains:

- **12 `subprocess` occurrences across 3 files**, including
  `subprocess.Popen(['git', 'add', '.'], ...)` and, at `__build__.py:107`,
  `subprocess.run(cmds, check=True)` where `cmds` is a bare **variable**
- **3 `requests.` / `urlopen(` occurrences** in `py/server/utils_info.py`
- 0 `os.environ`

Note that its three subprocess-bearing files are all root-level and dunder-named:
`__build__.py`, `__commit__.py`, `__update_comfy__.py`.

Older and possibly predating the rule change, but pointing the same way:
`comfyui-videohelpersuite` 1.7.9 (created 2025-12-17) also reads
`Passed automated checks` while shipping **61 `subprocess` and 9 `os.environ`
occurrences** -- a pack whose whole purpose is shelling out to ffmpeg.

**So these are all FALSE:** "any subprocess call flags", "a variable argv flags
where a list literal does not", "any network call flags".

## What I have already ruled out, with the measurement

- **Bundle size is not it.** An earlier version of mine was Active at 814 files
  including 119 dev scripts and 28 `.ps1`/`.bat`/`.cmd` installers; a later one
  was Flagged at 715 files with none of those, and the executable trigger
  surface was identical between them.
- **Dependency weight is not it.** `comfyui_controlnet_aux` declares 24
  dependencies including torch, torchvision and opencv and is Active on
  `Passed automated checks`. I declare 19.
- **Comfy-Org's published standards** (docs.comfy.org/registry/standards)
  prohibit only: `eval`/`exec`, runtime package installation through subprocess,
  and code obfuscation. I do none of the three.
- **`Comfy-Org/cbyrne-custom-nodes-security-scan` is a public repo** and appears
  to be a Comfy-Org security scanner, but my three rule names and twelve
  identifiers appear nowhere in it, so it is probably not the deployed ruleset,
  or is an older version of it.

## What I want you to find out

**Primary: what distinguishes a call site that emits a finding from one that
does not?** State it as a rule I can apply to a line of code.

Angles worth trying:

1. **Find the ruleset.** The identifiers are distinctive: `$env_read1..3`,
   `$env_mod1`, `$env_mod4`, `$http1`, `$http2`, `$socket1`, `$socket3`,
   `$subprocess_popen_direct`, `$subprocess_run_direct`, and especially
   `$socket_stage_assign`, which is hand-named rather than a numbered variant.
   Rule names: `python_environment_manipulation`, `python_network_operations`,
   `python_command_injection_risk`. The description string
   "Detects environment variable manipulation and reading" is also searchable.
   Try GitHub code search, public YARA collections, PyPI packages shipping YARA
   rules for Python source, and security vendor rulesets. If this set derives
   from an open-source parent, finding it gives me everything.
2. **Find the deployed scanner service** behind `config.SecretScannerURL`. Look
   for a default URL in the git history of `Comfy-Org/registry-backend`,
   deployment manifests, Cloud Run configuration, CI workflows, or any newer
   Comfy-Org scanner repo than `cbyrne-custom-nodes-security-scan`.
3. **Differential more bundles.** Registry API, no auth:
   `GET https://api.comfy.org/nodes/<id>/versions?include_status_reason=true`
   finds versions whose reason is exactly `Passed automated checks`;
   `GET https://api.comfy.org/nodes/<id>/versions/<version>` gives `downloadUrl`
   and `createdAt`. Prefer versions created after 2026-08-01. Find a clean
   version whose bundle contains one of my flagged shapes -- each one narrows
   the rule. Read the zips; do not execute anything from them.
4. **Is there a local scan oracle?** Check `Comfy-Org/comfy-cli`: does
   `comfy node publish` or any lint/validate/dry-run subcommand run the same
   scan locally? If so that is the single most valuable answer, because it lets
   me iterate to zero findings without burning version strings.
5. **Establish the clean response contract.** `issues` is the raw response body
   and the test is `issues == ""`. If a clean scan returned `[]` or `{}` instead
   of an empty string, nothing would ever auto-promote -- which contradicts
   rgthree. Confirm what a clean response actually is.

## Rules for your answer

- Ground every claim: name the file, URL, commit, or exact query.
- Label each claim MEASURED, INFERRED, or ASSUMED.
- Do not propose that I remove product capability. The rgthree evidence shows
  packs using subprocess and network scan clean, so capability removal is the
  wrong shape of fix until proven otherwise.
- Do not propose partial finding reduction; one finding flags the same as
  thirteen.
- If you cannot determine the discriminator, say so plainly and give me the
  single cheapest experiment that would settle it.
