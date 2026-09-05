# How do we get `comfyui-old-time-radio` from Flagged to Active?

**One question. Every number below was read from the live API today (2026-09-05),
not remembered. Do not accept a claim in this brief that you can check and find
false -- say so instead.**

## The state, measured

Pack `comfyui-old-time-radio`, publisher `fluxus`, published to registry.comfy.org
by `.github/workflows/publish_action.yml` on any push touching `pyproject.toml`.

`GET https://api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true`

| version | status | deps |
|---|---|---|
| 2.0.0-alpha.21 | **Flagged** -- 13 findings, ALL `info`, 0 critical | 19 |
| 2.0.0-alpha.20 | Flagged -- 12 info | 19 |
| 2.0.0-alpha.19 | Flagged -- 12 info | 19 |
| 2.0.0-alpha.18 | Flagged -- 12 info | 19 |
| 2.0.0-alpha.17 | Flagged | 19 |
| 2.0.0-alpha.16 | Flagged | 18 |
| 2.0.0-alpha.15 | Flagged | 18 |
| 2.0.0-alpha.14 | **Banned** | 15 |
| 2.0.0-alpha.13 | **Banned** | 13 |

`GET https://api.comfy.org/nodes/comfyui-old-time-radio` returns
`latest_version: null`. **That is the actual blocker.** A null `latest_version`
is why ComfyUI Manager reports "not a CNR node" / "Cannot resolve install
target", so no user can install the pack at all. It was null while alpha.21 was
Pending and it is still null now that alpha.21 is Flagged.

## The ban, and what we did about it

alpha.13 and alpha.14 were BANNED by a human, `drltdata@comfy.org`, with this
verbatim message:

> policy-v0.2: RCE (code execution) — attacker-reachable via unauthenticated
> /prompt (node widget) or no-auth route; confirmed by code-level verify-deep.

Both clauses were REAL. We closed seven surfaces, each with a commit:

| surface | commit |
|---|---|
| the `ffmpeg` widget reaching `argv[0]` on five nodes | `a9e0383e` |
| UNC/SMB coercion via `replay_from`, `workflow_json_path`, media paths | `a9e0383e` / `843b79d4` |
| the same coercion through `IS_CHANGED`, which runs BEFORE the execute guard | `79dc9828` |
| forged image-cache entries -> arbitrary local FILE READ, served by `/view` | `9d3f56a7` |
| the no-auth route half (`POST /otr/video_render_*`, unconditional in alpha.13) | `b198026a` |
| a pending sweep deleting what it could not READ | `31dc6861` |
| replay import trusting a ledger the manifest never verified | `14c6a6db` |

Both POST routes are DELETED, not gated. Every free-STRING path widget is
confined at the node execute method on the computed destination. The findings
went 158 -> 12 across those versions.

## The 13 `info` findings on alpha.21, in full

| file | issue_type | line | admin_tags |
|---|---|---|---|
| `prestartup_script.py` | python_environment_manipulation | 60 | any-folder-access, system-modification |
| `nodes/_otr_writer_heartbeat.py` | python_environment_manipulation | 61 | credential-access, system-modification |
| `nodes/_otr_audio_engines/eng_indextts2.py` | python_environment_manipulation | 176 | system-modification |
| `nodes/_otr_shared/env.py` | python_environment_manipulation | 77 | system-modification |
| `scripts/_otr_idx_download_weights.py` | python_environment_manipulation | 70 | system-modification |
| `nodes/_otr_comfy_backend.py` | python_network_operations | 384 | any-network-requests |
| `nodes/_otr_feed_fetch.py` | python_network_operations | 249 | any-network-requests |
| `nodes/_otr_openrouter_backend.py` | python_network_operations | 1011 | any-network-requests, credential-access |
| `nodes/_otr_google_api/client.py` | python_network_operations | 191 | any-network-requests |
| `nodes/_otr_shared/cloud_media_invoke.py` | python_network_operations | 578 | any-folder-access, any-network-requests |
| `nodes/_otr_audio_engines/eng_indextts2.py` | python_command_injection_risk | 214 | any-code-execute, any-folder-access |
| `nodes/_otr_shared/proc.py` | python_command_injection_risk | 161 | any-code-execute |
| `nodes/_otr_shared/proc.py` | python_command_injection_risk | 168 | any-code-execute |

The 13th (vs alpha.20's 12) is `scripts/_otr_idx_download_weights.py:70`,
`env = os.environ.get("OTR_INDEXTTS2_DIR")`. That file ships DELIBERATELY: the
IndexTTS2 installer we ship calls it at its weights step, so excluding it left
our own installer pointing at a file that is not there. We traded one `info`
finding for a working installer and we are not reverting that.

Every network finding is an OPT-IN, DEFAULT-OFF cloud lane (OpenRouter, Google
API, Comfy credits, cloud media) plus one RSS feed fetcher. The `proc.py`
findings are the single subprocess gateway that shells out to `ffmpeg`.

## What we already know, so do not re-propose it

1. **A clean scan is NOT the bar.** We sampled the registry's own approvals:
   **0 of 102 policy-v0.2 approvals had a clean scan, and 31 carried the same
   subprocess finding we carry.** Driving 13 -> 0 is therefore not the
   difference between Flagged and Active.
2. **There is no publisher self-service path to Active.** Confirmed by reading
   `Comfy-Org/registry-backend`. Promotion is their Cloud Scheduler cron hitting
   their own `/security-scan` endpoint (`registry.go:938`, versions older than 30
   minutes only). The scanner is a PRIVATE repo. Any finding at all keeps a
   version Flagged until an admin batch-approves it.
3. **Bundle contents were measured and are NOT the cause.** alpha.8 was ACTIVE
   with 814 files, 119 under `scripts/`, 28 `.ps1`/`.bat`/`.cmd` installers.
   alpha.12 was FLAGGED with 715 files, 0 under `scripts/`, 0 installers. The
   executable trigger surface was IDENTICAL across the two (subprocess 33/33,
   Popen 9/9, base64 5/5, urlopen 2/2). Shrinking the bundle has already been
   tried and did nothing.
4. `.comfyignore` already excludes `tests/`, `kibitz-runs/`, `.github/`,
   `.claude/`, `docs/`, `tools/`, `viewer/`, `scripts/*` (with four negations
   for files the shipped adapters resolve at runtime), `assets/`, and
   `workflows/variants/*.md`.
5. **Deleting a VERSION is a soft delete that burns the version string forever;
   deleting the NODE is a hard delete that frees every string.** The DELETE API
   needs the operator's browser Firebase session -- a publish token gets 401.
6. A manual review request is DRAFTED at
   `docs/2026-09-05-registry-review-request-ALPHA20.md` and NOT posted. Posting
   is a public act the operator gates. He has said he is not posting anything
   until he is ready to ship.

## The question

**What is the highest-probability path from Flagged to Active, and what exactly
do we do next?**

Be concrete and rank by expected value. In particular:

- **Is the manual review request actually the only move, or is there a
  mechanism we have not found?** If you can read anything in
  `Comfy-Org/registry-backend`, the docs, the Discord, or another pack's history
  that shows a different promotion path, name it with the source.
- **If the review request IS the move: what makes one get actioned?** We have
  never had a reply on a registry issue. Where should it be filed, addressed to
  whom, and what should it contain and NOT contain? Is there a better venue than
  a GitHub issue on `registry-backend`?
- **Is any of the 13 findings worth removing on its own merits** -- not to game
  the count, but because the code would genuinely be better? Judge each on
  whether removal is a real simplification or scanner-dodging.
- **Would a fresh NODE ID** (hard-delete and republish under a new id, or a new
  id alongside) plausibly land Active, given that alpha.8 under this same id was
  Active and the ban is now attached to this id's history? What is the risk?
- **Is there a version-shape we have not tried** -- e.g. publishing a
  deliberately minimal first version and adding lanes back?
- **What would you check that we have not checked?** Name the specific request,
  file, or repo.

**Disagree with anything above that you can show is wrong.** A refutation with a
source beats agreement.
