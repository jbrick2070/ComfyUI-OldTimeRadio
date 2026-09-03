# Getting ONE Active version on the Comfy Registry without deleting features

**For a fresh reviewer (Cursor).** Read this, then read the registry's own source at
https://github.com/Comfy-Org/registry-backend **with fresh eyes** — do not take this
document's reading of it on trust. Prior sessions have already read parts of it and reached
conclusions that this document's own evidence now contradicts, so re-derive the rules from
the code rather than inheriting ours.

---

## The situation in one paragraph

`comfyui-old-time-radio` (publisher `fluxus`) is a local, offline-first ComfyUI custom-node
pack that generates old-time-radio episodes: a local LLM writes a script into a JSON ledger,
a TTS engine voices it, ffmpeg muxes and captions, and a video engine renders per-beat
clips. It has **4 versions on the registry and every one is `Flagged`. It has never had an
Active version that survived.** Because no version is Active, the node record's
`latest_version` resolves to `null`, and ComfyUI Manager therefore cannot install the pack
at all — it reports "not a CNR node / cannot resolve install target". Downloads: 0.

## The question

**Is there a path to one Active version that does not require deleting working features?**

The constraint is hard and comes from the operator: **do not remove good features to satisfy
a scanner.** Reaching zero findings by deleting the operator-knob system, ffmpeg, or the
opt-in cloud lanes is not a cleanup, it is deleting the product. If the only honest answer is
"there is no path without that", say so plainly — that is a legitimate finding.

---

## Verified facts (re-verifiable; commands included)

All of this was pulled live on 2026-09-03. Re-run anything you doubt:

```bash
curl -s "https://api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true"
curl -s "https://api.comfy.org/nodes/<any-other-pack>"
```

### 1. Our four versions and their findings

| version | status | critical | info |
|---|---|---|---|
| 2.0.0-alpha.16 | Flagged | **0** | 157 |
| 2.0.0-alpha.15 | Flagged | 2 | 156 |
| 2.0.0-alpha.14 | Flagged | 2 | 156 |
| 2.0.0-alpha.13 | Flagged | 2 | 156 |

The 2 criticals were ours and are fixed: the writer declared the session-bearer hidden input
(`auth_token_comfy_org`) beside `api_key_comfy_org`, which the scanner reads as a
`prohibited-string` with the `credential-access` tag. Removed in alpha.16; a regression test
now refuses that type name in every shipped file. **alpha.16 scanned with zero criticals and
was still Flagged.**

alpha.16's 157 findings by `issue_type`, all `severity: info`, all `scanner: yara_scan`:

| issue_type | count | what it actually is |
|---|---|---|
| `python_environment_manipulation` | 102 | `os.environ.get` of documented `OTR_*` knobs, `HF_HOME`, `HF_TOKEN`, `HF_HUB_OFFLINE`; a few process-local writes |
| `python_command_injection_risk` | 35 | `subprocess.run`/`Popen` with **argument lists and `shell=False`** — ffmpeg/ffprobe for mux, captions, credits; three TTS engines in isolated venvs |
| `python_url_command_execution` | 12 | same subprocess family; several are ffmpeg **log strings**, not calls |
| `python_network_operations` | 5 | `requests`/`urllib` on opt-in cloud lanes + the RSS news fetcher. Every endpoint is a constant or env override; **no node widget accepts a URL or host** |
| `windows_process_manipulation` | 1 | `kernel32.OpenProcess(SYNCHRONIZE, …)` to check if a sibling process holding a GPU lease is alive |
| `python_sensitive_file_access` | 1 | sha256 of a downloaded model file, to verify it |
| `python_bytecode_manipulation` | 1 | a `sys.modules` lookup; no bytecode touched |

No `eval(`/`exec(` builtins, no `os.system`, no `shell=True`, no runtime `pip`, no
obfuscation, no bundled executables, no telemetry.

### 2. THE FINDING THAT REFRAMES THE PROBLEM — Flagged is survivable, and normal

| pack | downloads | versions | `latest_version` resolves to |
|---|---|---|---|
| `rgthree-comfy` | 3,923,218 | **55 Active, 2 Flagged** | an **Active** one (skips the Flagged newest) |
| `comfyui-easy-use` | 3,484,164 | **20 Active, 1 Flagged** | an **Active** one |
| `comfyui-old-time-radio` | 0 | **0 Active, 4 Flagged** | **null** |

**So "Flagged" is not the disease.** Two of the most-installed packs in the ecosystem each
carry Flagged versions and install fine, because `latest_version` resolves to their most
recent **Active** version instead. Our problem is narrower and more specific:
**we have never had a surviving Active version, so there is nothing to fall back to.**

Relevant history: we *did* have one. `2.0.0-alpha.8` was Active. A later **node-level delete**
(hard delete, cascades all versions) destroyed its row. Version strings alpha.1–alpha.12 are
burned and unusable — `(node_id, version)` is uniquely indexed server-side and a version
delete is a *soft* delete that permanently burns the string.

### 3. Every Active version sampled scanned CLEAN

`rgthree-comfy@1.0.2608210019`, `comfyui-easy-use@1.3.6`, `comfyui-kjnodes@1.5.0` — all
Active, and all carry `status_reason: "Passed automated checks"` with **no findings array at
all**. Not "findings that were accepted" — no findings.

**This is the central puzzle for you to solve.** Those packs unquestionably read environment
variables and shell out to subprocesses too. Why do they scan clean while we produce 157
findings? Candidate explanations, none verified:

- **The scanner tightened recently.** Note the pattern: rgthree's *two newest* versions are
  Flagged while its 55 older ones are Active; easy-use's *newest* is Flagged while its 20
  older are Active. Ours are all recent. If the ruleset moved, the comparison to their older
  Active versions is invalid and the real question is what the *current* ruleset accepts.
- **What gets scanned differs.** We ship 963 files. `.comfyignore` already strips `tests/`,
  `kibitz-runs/`, `.github/`, `.claude/`, `docs/`, most of `scripts/`. Do they ship far less?
  Is there a path-scoping rule (e.g. only `nodes/`, or only files reachable from
  `NODE_CLASS_MAPPINGS`)?
- **Density or pattern shape matters.** 102 env findings is a lot. Would centralising every
  `os.environ.get` behind one accessor module collapse 102 findings into 1 — and would that
  even help, or does one finding Flag you just as hard as 102?
- **There is a declaration/suppression mechanism we do not know about** — a manifest field, a
  permissions declaration, an allowlist. We have not found one; that is not proof it is absent.

### 4. What actually gets you BANNED (different, and we are not in it)

`was-node-suite-comfyui@1.0.2` is `Banned`, reason verbatim:

> `policy-v0.2: DATA_EXFIL` — Free STRING 'url' widget flows to
> `requests.Request(url=url).prepare()/session.send()` carrying the workflow IMAGE bytes as a
> multipart body. No host allowlist; unauthenticated `/prompt` POST exfiltrates rendered image
> content to an attacker-chosen host.

That is a real data-flow: user-controllable URL → network send of user data, reachable
unauthenticated. The adjacent policy class (`UNAUTHENTICATED_SIDE_EFFECT`, which banned
`comfyui-easyuse-anima 1.1.2`) is why we just gated two POST routes — `__init__.py` registered
`/otr/video_render_single` and `/otr/video_render_soak` unconditionally, each reading
caller-supplied file paths from an unauthenticated JSON body and starting a render thread.
They are now behind `OTR_ENABLE_HTTP_RENDER_ROUTES=1`, default off, shipping in alpha.17.

Also note `was-node-suite@1.0.1` is **Active with 79 findings** — but their `severity` and
`issue_type` are both `null`, i.e. a legacy scan schema. Treat it as evidence the schema
changed, **not** as proof that today's scanner promotes a findings-carrying version.

---

## What has already been done (do not re-propose these)

1. **The critical credential finding — fixed and verified.** alpha.16 = 0 critical.
2. **The unauthenticated POST routes — fixed**, gated behind an off-by-default env flag,
   shipping in alpha.17 (`__init__.py`, test at `tests/test_http_render_route_gate.py`).
3. **A `README.md` string shaped like an HF token** (`hf_` + 38 chars, inside gitleaks'
   `hf_[a-zA-Z0-9]{34,40}` window) — removed in `64d81ca7`. It was the only shipped string
   matching a published secret-detection pattern.
4. **Static dependency list** — the registry does not evaluate setuptools `dynamic`; a pack
   published with `dependencies = []` installs its code with none of its libraries.
5. **A manual review request is drafted** and retargeted at alpha.17:
   `docs/2026-09-02-registry-manual-review-request.md`. Not yet filed. Every open "Manual
   review request" on the Comfy-Org tracker (#184–#220) is unanswered, though silent admin
   batch approvals demonstrably happen (`"Batch approved by admin"` appears in some
   `status_reason` values).

## Known-open, cheap, not yet done

- `viewer/index.html` (16 KB) still ships and calls `/ledger?latest=1`, `/list`,
  `/ledger?path=` — three endpoints the pack does not register. Nothing in `nodes/` loads it.
  Fix is one line in `.comfyignore`. (ship-audit `registry-flag-09`, MEDIUM)

---

## What we want from you

Answer these from the **registry-backend source**, not from our summary:

1. **What exactly gates the `Flagged` → `Active` transition today?** Find the code path. Is
   any finding disqualifying regardless of severity, or is there a severity threshold? Does
   `severity: info` actually block?
2. **Is there any publisher-side lever** — a manifest field, a declaration, a suppression
   file, a re-scan trigger, an appeal endpoint — that moves a Flagged version to Active
   without an admin acting manually? Prior sessions concluded there is none. Verify or refute.
3. **What is scanned?** Whole zip, or scoped paths? If scoped, can `.comfyignore` legitimately
   shrink the scanned surface without removing anything a user needs at runtime?
4. **Would reducing finding COUNT help at all**, or is 1 finding equivalent to 157? This
   decides whether centralising 102 `os.environ.get` calls behind one accessor is worth doing.
   Do not recommend it unless the code says count matters.
5. **Why do rgthree / kjnodes / easy-use scan clean** while we do not, given they use the same
   Python facilities? If it is because the ruleset recently tightened, what changed and when?
6. **Is there a supported way to get a fresh scan** of an existing version, or is a new version
   string the only way? (Each new string is spent permanently.)
7. **Given all of the above — is there a route to one Active version that keeps every current
   feature?** If yes, give the concrete steps. If no, say so, and say what the minimum
   feature-affecting change would be so the operator can decide with real numbers.

## Ground rules for the answer

- **Cite file and line** in `Comfy-Org/registry-backend` for every claim about how the
  registry behaves. This document's own prior conclusions were wrong once already
  ("Active requires zero findings" did not survive contact with the data above).
- **Do not propose removing a feature** as the headline recommendation. If a feature genuinely
  must change, present it as a costed option, not as the plan.
- Distinguish clearly between what the code says, what the API data shows, and what you are
  inferring.
