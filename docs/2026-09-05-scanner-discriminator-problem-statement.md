# What exactly makes the Comfy Registry scanner emit a finding, and how do we ship a pack that emits none?

**Open-ended research brief. Everything below was MEASURED on 2026-09-05 against
the live API, the public backend source, or real bundles downloaded from
`cdn.comfy.org`. Nothing here is recollection. Refute anything you can check and
find false -- a refutation with a source is worth more than agreement.**

---

## 1. THE GOAL, STATED EXACTLY

Node pack `comfyui-old-time-radio`, publisher `fluxus`, on registry.comfy.org.
Every published version is `Flagged` (two early ones are `Banned`). While no
version is `Active`, the node's `latest_version` resolves to `null`, and ComfyUI
Manager therefore reports "not a CNR node" and **no user can install the pack**.

**We want a published version to reach `Active`.**

## 2. THE PROMOTION LOGIC IS PUBLIC AND TRIVIAL

`Comfy-Org/registry-backend`, `services/registry/registry_svc.go`,
`RegistryService.PerformSecurityCheck` (fetched from `raw.githubusercontent.com`,
main branch):

```go
issues, err := sendScanRequest(s.config.SecretScannerURL, nodeVersion.Edges.StorageFile.FileURL)
if err != nil { ... }

if issues == "" {
    // "No security issues found in node %s@%s. Updating to active."
    nodeVersion.Update().
        SetStatus(schema.NodeVersionStatusActive).
        SetStatusReason("Passed automated checks").Exec(ctx)
    s.discordService.SendSecurityCouncilMessage(...)
} else {
    // "Security issues found in node %s@%s. Updating to flagged."
}
```

And `sendScanRequest` (same file, ~line 1461) is a bare HTTP POST:

```go
requestBody, _ := json.Marshal(ScanRequest{URL: fileURL})
req, _ := http.NewRequest("POST", apiURL, bytes.NewBuffer(requestBody))
req.Header.Set("Content-Type", "application/json")
resp, _ := client.Do(req)
responseBody, _ := io.ReadAll(resp.Body)
if resp.StatusCode != 200 { return "", fmt.Errorf(...) }
return string(responseBody), nil
```

**Three consequences, and they frame the whole problem:**

1. **`issues == ""` is the entire test.** A clean scan auto-promotes with no
   human. There IS a publisher-controlled path to Active.
2. **It is ALL-OR-NOTHING.** One surviving finding flags a version exactly as
   thirteen do. Reducing 13 to 1 is worth nothing. Only zero counts.
3. **The rules are not in this repo.** They live behind
   `config.SecretScannerURL`, declared in `config/config.go` as a bare string
   with no default committed. A GitHub search of `org:Comfy-Org` for "scan"
   returns **zero** repositories. The scanner service appears to be private.

**A second, human path also exists and is observable:** `ComfyUI-Crystools`
1.27.4 has `status_reason` = `{"message":"Approved by admin","by":"dr.lt.data@gmail.com",...}`.
So Active is reachable either by a clean scan or by an admin. We have never had
a reply on ~20 manual-review issues since 2026-08-02, so the clean-scan path is
the one we can actually drive.

## 3. WHAT THE FINDINGS LEAK -- THE RULESET IS PARTIALLY OBSERVABLE

We cannot read the rules, but every finding names the **exact YARA pattern
identifier** that matched, plus a confidence, the file, the line number, and the
matching source line. This is our best window into the ruleset.

Our 13 findings on `2.0.0-alpha.21`, complete:

| file | rule (`issue_type`) | line | conf | matched patterns | admin_tags |
|---|---|---|---|---|---|
| `prestartup_script.py` | python_environment_manipulation | 60 | 90 | `$env_read1` `$env_read2` `$env_mod1` | any-folder-access, system-modification |
| `nodes/_otr_writer_heartbeat.py` | python_environment_manipulation | 61 | 90 | `$env_read2` | credential-access, system-modification |
| `nodes/_otr_audio_engines/eng_indextts2.py` | python_environment_manipulation | 176 | 90 | `$env_read2` `$env_read3` | system-modification |
| `nodes/_otr_shared/env.py` | python_environment_manipulation | 77 | 90 | `$env_read1` `$env_read2` `$env_mod1` `$env_mod4` | system-modification |
| `scripts/_otr_idx_download_weights.py` | python_environment_manipulation | 70 | 90 | `$env_read2` | system-modification |
| `nodes/_otr_comfy_backend.py` | python_network_operations | 384 | 90 | `$http1` | any-network-requests |
| `nodes/_otr_feed_fetch.py` | python_network_operations | 249 | 90 | `$socket1` `$socket3` `$socket_stage_assign` | any-network-requests |
| `nodes/_otr_openrouter_backend.py` | python_network_operations | 1011 | 90 | `$http1` | any-network-requests, credential-access |
| `nodes/_otr_google_api/client.py` | python_network_operations | 191 | 90 | `$http2` | any-network-requests |
| `nodes/_otr_shared/cloud_media_invoke.py` | python_network_operations | 578 | 90 | `$http2` | any-folder-access, any-network-requests |
| `nodes/_otr_audio_engines/eng_indextts2.py` | python_command_injection_risk | 214 | 95 | `$subprocess_popen_direct` | any-code-execute, any-folder-access |
| `nodes/_otr_shared/proc.py` | python_command_injection_risk | 161 | 95 | `$subprocess_run_direct` | any-code-execute |
| `nodes/_otr_shared/proc.py` | python_command_injection_risk | 168 | 95 | `$subprocess_popen_direct` | any-code-execute |

Twelve distinct pattern identifiers across three rules:
* `python_environment_manipulation`: `$env_read1` `$env_read2` `$env_read3` `$env_mod1` `$env_mod4`
* `python_network_operations`: `$http1` `$http2` `$socket1` `$socket3` `$socket_stage_assign`
* `python_command_injection_risk`: `$subprocess_popen_direct` `$subprocess_run_direct`

One finding's full metadata, as a shape example:

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

The `$`-prefixed identifiers and the `yara_matches` key mean this is **YARA**.
The rule names are snake_case and descriptive
(`python_command_injection_risk`), and the string identifiers are numbered
variants of a concept (`$env_read1..3`, `$env_mod1`, `$env_mod4`, `$http1..2`,
`$socket1`, `$socket3`, plus one NAMED string `$socket_stage_assign`). The gaps
in the numbering (`$env_mod2/3`, `$socket2`) are strings that exist in the rule
but did not match us.

## 4. THE CONTROL THAT BREAKS THE OBVIOUS THEORIES

**`rgthree-comfy` version `1.0.2608210019`.** This is the important one:
* `createdAt` = **2026-08-21**, i.e. AFTER the ruleset change that flagged our
  alpha.9 (our alpha.8 was Active, alpha.9 onward Flagged, with an identical
  trigger surface -- so a rule change happened in late August 2026). It was
  therefore scanned by a scanner close to today's.
* `status_reason` = **`Passed automated checks`** -- auto-promoted on a CLEAN
  scan, not admin-approved.
* Its published bundle (`cdn.comfy.org/rgthree/rgthree-comfy/1.0.2608210019/node.zip`,
  4.06 MB, 45 `.py` files), grepped by us, contains:

| pattern | hits |
|---|---|
| `subprocess` | **12**, across 3 files |
| `requests.` / `urlopen(` | **3**, in 1 file |
| `os.environ` | 0 |
| `eval(` / `exec(` / `socket` | 0 |

The subprocess call sites include, verbatim:

```
__build__.py:77    ts_version_result = subprocess.run(["node", "./node_modules/typescript/bin/tsc", "-v"], ...
__build__.py:107   checked = subprocess.run(cmds, check=True)
__commit__.py:44   process = subprocess.Popen(['git', 'add', '.'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
__commit__.py:49   process = subprocess.Popen(['git', 'commit', '-a', '-v', '-m', args.message], ...
__update_comfy__.py:7   from subprocess import Popen, PIPE, STDOUT
```

Note `__build__.py:107` passes a bare VARIABLE `cmds` -- the same shape as our
flagged `proc.py:161`, `subprocess.run(argv, **kwargs)`.

Its network hit is in `py/server/utils_info.py`.

**Therefore all of these are FALSE:**
* "any `subprocess` call produces a finding"
* "a variable argv flags where a list literal does not"
* "any network call produces a finding"

**Corroborating but CONFOUNDED by age:** `comfyui-videohelpersuite` 1.7.9, also
`Passed automated checks`, ships **61 `subprocess` hits and 9 `os.environ` hits**
-- a pack whose entire purpose is shelling out to ffmpeg, scanned clean. But
`createdAt` = 2025-12-17, which likely predates the rule change. Cite rgthree
when the date matters.

## 5. THE LEAD WE CHASED AND COULD NOT CONFIRM

rgthree's three subprocess-bearing files are all root-level and dunder-named:
`__build__.py`, `__commit__.py`, `__update_comfy__.py`. That suggested the
scanner might skip `__*__.py`, or skip files unreachable from the node entry
points.

**We tested it against our own bundle and it does not hold.** Our `__init__.py`
is unflagged, but only because it contains no matching pattern: its sole
`os.environ` occurrence is inside a COMMENT and its only network-ish line is
`from aiohttp import web`. So `__init__.py` being clean is explained by content,
not by a skip rule. The file-selection theory survives for rgthree but is
unproven, and one fact strains it: our `scripts/_otr_idx_download_weights.py` is
imported by nothing and invoked only by a PowerShell installer, and it still
produced a finding.

## 6. WHAT ELSE WE HAVE ALREADY RULED OUT, WITH THE MEASUREMENT

* **Bundle size / file count is not it.** Our alpha.8 was ACTIVE at 814 files
  with 119 files under `scripts/` and 28 `.ps1`/`.bat`/`.cmd` installers.
  alpha.12 was FLAGGED at 715 files with zero of either. The executable trigger
  surface was identical across the two (subprocess 33/33, Popen 9/9, base64 5/5,
  urlopen 2/2). Shrinking the bundle has been tried and changed nothing.
* **Dependency weight is not it.** `comfyui_controlnet_aux` declares 24
  dependencies including torch, torchvision, opencv-python and scikit-image and
  is Active on `Passed automated checks`. We declare 19.
* **The 2026-08 ban was real and is closed.** alpha.13/.14 were Banned by a
  human with: *"policy-v0.2: RCE (code execution) — attacker-reachable via
  unauthenticated /prompt (node widget) or no-auth route; confirmed by
  code-level verify-deep."* Both clauses were true. Seven surfaces were closed
  across `a9e0383e`, `843b79d4`, `79dc9828`, `9d3f56a7`, `b198026a`, `31dc6861`,
  `14c6a6db`; both POST routes are DELETED, not gated; findings went 158 -> 12.
  **The current 13 findings are all `info`, zero critical.** The ban and the
  flag are different mechanisms and must not be conflated.
* **Node extraction is independent of approval.** `TriggerComfyNodesBackfill`
  selects on `ComfyNodeExtractStatusEQ(Pending)` with no status filter. Our node
  index is empty (`/comfy-nodes?node_id=comfyui-old-time-radio` -> `total: 0`;
  the versioned receipt is `null` on alpha.17, .20 and .21) which is a real and
  separate defect, but it does not gate Active.

## 7. WHAT WE ARE ASKING YOU TO FIND OUT

**The core question: what, precisely, distinguishes a call site that emits a
finding from one that does not?** Answer it well enough that we can predict, for
any given line of our code, whether it will flag.

Attack it from any angle that works. Ideas, not a constraint:

1. **Find the ruleset.** The identifiers are distinctive: `$env_read2`,
   `$subprocess_popen_direct`, `$subprocess_run_direct`, `$socket_stage_assign`,
   `$http1`, `$env_mod4`. Rule names: `python_environment_manipulation`,
   `python_network_operations`, `python_command_injection_risk`. Search GitHub
   code search, YARA rule collections, PyPI, security-vendor rulesets and any
   Comfy-Org repo or container image. **`$socket_stage_assign` is the most
   distinctive string here -- it is not a numbered variant and reads like it was
   hand-named.** If this ruleset is derived from an existing open-source YARA
   set, finding the parent gives us the whole thing.
2. **Find the scanner service.** `config.SecretScannerURL`. Look for deployment
   manifests, Terraform, Cloud Run/Scheduler config, docker-compose, CI, or any
   commit in `Comfy-Org/registry-backend` history that ever contained a default
   URL or the service's name. A deleted default in git history would name it.
3. **Differential the bundles empirically.** Download more RECENT versions whose
   `status_reason` is `Passed automated checks` (use
   `GET /nodes/<id>/versions/<ver>` for `downloadUrl` and `createdAt`), grep each
   for the twelve identifiers' likely source shapes, and find a pack that
   CONTAINS a shape and still passed. Each such counter-example narrows the rule.
   Prioritise versions created after 2026-08-01. This is the angle least likely
   to dead-end.
4. **Read Comfy-Org's own published guidance.** docs.comfy.org registry
   standards, the registry docs, blog posts, Discord announcements, the
   `comfy-cli` source (it may run the same scan locally before publish -- if
   `comfy node publish` or a `--dry-run` can scan, that is a fast local oracle
   and would change everything).
5. **Consider that the response may not be a simple boolean.** `issues` is the
   raw response BODY. Establish what a clean response actually looks like --
   empty string, `[]`, `{}`? If the scanner returns `[]` for a clean pack, then
   `issues == ""` would be FALSE and nothing would ever auto-promote, which
   would contradict rgthree. Worth confirming the exact contract.

**Deliverable:** the discriminator, stated as a rule we can apply to a line of
code, with the evidence. If you cannot determine it, say so plainly and give us
the single cheapest experiment that would.

**Constraints on your answer:**
* Ground every claim. Name the file, URL, commit or command.
* Distinguish MEASURED from INFERRED from ASSUMED, explicitly.
* Do not propose removing product capability as the answer -- section 4 shows
  that packs using subprocess and network scan clean, so capability removal is
  the wrong shape of fix until proven otherwise.
* Do not propose partial finding reduction. Section 2 point 2 makes it worthless.
