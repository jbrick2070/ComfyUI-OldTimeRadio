# PROBLEM STATEMENT: twelve YARA matches on `os.environ`, `requests`, `socket`, `subprocess`

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
READ-ONLY. Cite `file:line` and quote what you read. A grep hit is a lead.

## The problem in one paragraph

Our ComfyUI node pack is published to the Comfy Registry and sits **Flagged**.
The scanner is YARA. It emits **twelve `severity=info` findings, and nothing
else** -- no criticals, no remediation text, no other scanner. Every one is a
pattern match on one of four Python constructs:

| construct | matches | our files |
|---|---|---|
| `os.environ` read/write | 4 | `prestartup_script.py:60`; `nodes/_otr_writer_heartbeat.py:61`; `nodes/_otr_audio_engines/eng_indextts2.py:176`; `nodes/_otr_shared/env.py:77` |
| `requests` / `urllib` / `socket` | 5 | `nodes/_otr_comfy_backend.py:384`; `nodes/_otr_feed_fetch.py:249`; `nodes/_otr_openrouter_backend.py:1011`; `nodes/_otr_google_api/client.py:191`; `nodes/_otr_shared/cloud_media_invoke.py:578` |
| `subprocess.run` / `.Popen` | 3 | `nodes/_otr_audio_engines/eng_indextts2.py:214`; `nodes/_otr_shared/proc.py:161`; `nodes/_otr_shared/proc.py:168` |

The YARA patterns named in the findings are literally `$env_read1/2/3`,
`$env_mod1/4`, `$http1`, `$http2`, `$socket1/3`, `$socket_stage_assign`,
`$subprocess_run_direct`, `$subprocess_popen_direct`.

## The question, and it has two halves

**HALF A -- CAN WE REDUCE THE COUNT AT ALL, honestly?**
For each of the twelve sites: is the construct REMOVABLE, REPLACEABLE by
something that does not match the pattern, or IRREDUCIBLE? Be concrete --
"use PyAV" is only an answer if you check whether THIS build's PyAV can do
the specific thing that call does.

Known shape of our code, verify rather than trust:
* **Every spawn funnels through ONE gateway**, `nodes/_otr_shared/proc.py`
  (`run` at :161, `popen` at :168). It refuses `shell=True`, refuses a string
  argv, and allowlists `argv[0]`. Two of the three subprocess findings ARE that
  gateway. So the count is already near-minimal by construction -- unless the
  gateway itself can be written so it does not match `$subprocess_run_direct`.
  **Is that a legitimate engineering change or is it pattern-dodging?** Say so
  either way; we would rather be honest than clever.
* **`nodes/_otr_shared/env.py:77` is the ONE `os.environ[...] =` writer**, same
  single-owner shape. The other three env findings are READS.
* Two of the twelve are inside `nodes/_otr_audio_engines/eng_indextts2.py`,
  which is **BYTE-HASHED** (`_otr_voice_route.RUNTIME_FINGERPRINT_SOURCES`):
  changing one byte demotes a voice route the operator approved by ear. Treat
  those two as immovable and say so.
* The network sites are opt-in cloud lanes plus one RSS fetcher
  (`_otr_feed_fetch.py`, which is https-only, refuses non-public addresses, and
  bounds the read). Are any of them dead weight we could stop shipping?

**HALF B -- SHOULD WE BOTHER?**
We surveyed 578 packs / 3,707 versions. Of the **102 versions approved** under
the same policy between 2026-08-15 and 09-01, **not one had a clean scan**: 31
carried `python_command_injection_risk`, 72 carried network findings, 39 carried
env findings. Approval is a HUMAN verdict (`reviewed SAFE (GOAL2 verify-deep,
policy-v0.2)`); bans name an attacker-reachable CLASS (RCE 203, path traversal
35, unauthenticated side-effect ~18, SSRF/egress ~16, arbitrary file read 10).
Our own two bans were `RCE ... attacker-reachable via unauthenticated /prompt
(node widget) or no-auth route`, and both of those surfaces are now closed.

**So: is chasing the twelve `info` findings the wrong optimisation entirely?**
If yes, say it plainly and tell us what to spend the effort on instead. If no,
show the arithmetic that makes it worth it.

## What we would find most useful

Anything we have not thought of. The reviewer is a human reading code for
attacker-reachable classes, not a linter counting patterns. What makes a pack
like ours READ as safe on a code-level review -- structure, naming, an explicit
capability declaration, a SECURITY.md, narrowing widget types, dropping an
optional lane from the shipped bundle? Ground every suggestion in our files.

Answer terse, evidence first, markdown.
