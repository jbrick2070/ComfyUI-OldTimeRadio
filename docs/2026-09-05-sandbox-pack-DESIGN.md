# The sandbox pack -- a throwaway registry pack that answers scanner questions without burning OTR version strings

**DESIGN, not code. Written 2026-09-05 as the driver anchor for the arc this item
owes (a new capability with more than one defensible shape -- CLAUDE.md 08-17
amendment: YES -> full arc before code).** Nothing here is built. The arc could
not run in the window that wrote this because that window had no shell.

**Reading of the request.** "Sandbox pack" is taken to mean a SEPARATE, minimal
ComfyUI node pack published under the `fluxus` publisher whose only job is to be
scanned, so that every question we currently answer by publishing another
`comfyui-old-time-radio` version -- and burning a string, and waiting on a cron,
and risking a third ban on the real listing -- is answered on a listing nobody
installs. If that reading is wrong, stop here.

---

## 1. Why it is worth a listing

Every fact we know about the registry's reviewer was bought with an OTR version:

| question | what it cost |
|---|---|
| are `os.environ` reads / list-argv `subprocess` / opt-in `requests` really `info`? | alpha.17 -> .18 -> .19, a four-batch migration, three scans |
| does a clean scan promote to Active without a human? | still unknown -- `docs/GO_FORWARD_PLAN.md` item 1.4a says "needs zero findings or the manual review" and has never measured the first half |
| is the "N Nodes" card null because pycairo killed the Linux extract? | alpha.17 shipped a win32 marker to find out; result still `null` (handover item (a)) |
| does `scripts/*` + `!scripts/x.py` re-include in the PUBLISHER's zip? | proven only by downloading the alpha.20 zip after the fact |
| is a `.ps1` in the zip scanned at all? | unknown; the remote-installer pipe in `_otr_indextts2_install.ps1` shipped twice unflagged |
| how long from publish to scan, and is the cron nightly? | inferred from three timestamps |

Each of those is a one-file experiment. On the real pack each one is also a new
version, a new 809-file zip, a new human-review target and -- under the
2026-09-05 sequencing rule, "never bump a version while another version is
Pending" -- a day of latency. On a two-file pack it is a two-file diff.

**What the sandbox CANNOT answer, so nobody expects it to:** anything about OTR's
own history. The row-J control experiment (republish the alpha.8 tree
byte-identical to see whether the ruleset moved) is a question about THAT
listing's bytes and stays on the OTR node id. The sandbox measures RULES, not our
past.

## 2. Shape

* **Its own GitHub repo** (`jbrick2070/ComfyUI-OTR-Sandbox` or similar). Not a
  subfolder of this repo and not a branch: `publish_action.yml` here fires on any
  push touching `pyproject.toml` on `v2.0-alpha`, and a second `pyproject.toml`
  in this tree is a second way to publish the wrong thing. The sandbox copies the
  action verbatim (`Comfy-Org/publish-node-action@main`, keyed on the same
  publisher secret -- the token is per PUBLISHER, not per pack), with its own
  `paths: [pyproject.toml]` trigger so a probe is published by editing one line.
* **`[tool.comfy] PublisherId = "fluxus"`, node id `otr-registry-sandbox`**, and a
  `DisplayName` / `description` that say what it is in the first six words:
  *"Publisher test pack -- installs one no-op node."* The registry is public and
  ComfyUI Manager will list it; the honest label is the whole defence against it
  being read as spam, and it is also the truth.
* **One node, `OTR_SandboxNoop`**, `STRING -> STRING`, identity. It exists so the
  Linux extract container has something to put on the card, which is itself one
  of the probes.
* **Files:** `__init__.py`, `nodes/noop.py`, `pyproject.toml`, `README.md`,
  `.comfyignore`, `.github/workflows/publish.yml`, plus whatever ONE file the
  current probe adds. Nothing else, ever -- a probe that needs a second file is
  two probes.
* **A `PROBES.md` ledger in the sandbox repo**, one row per published version:
  version, the single delta, publish timestamp, scan-landed timestamp, status,
  findings by rule (read from
  `GET /nodes/otr-registry-sandbox/versions?include_status_reason=true`), and
  `/versions/<v>/comfy-nodes` result. The version string IS the probe id; the
  ledger is what turns the strings into knowledge. The findings are also copied
  into `docs/GO_FORWARD_PLAN.md`'s registry section here when they change a
  ruling.

## 3. The probe series, ordered by information per publish

The order matters because each publish waits on the scanner. Front-load the
questions whose answers change what we do next on the REAL pack.

| v | delta (one file, one pattern) | the question | what a result changes |
|---|---|---|---|
| `0.1.0` | nothing -- the no-op node alone | **Does a zero-finding version go Active with no human?** Also: time to scan; does `comfy-nodes` populate for a trivially importable pack? | If yes: zero findings is a self-service route to Active and the OTR collapse has a finish line. If no: only the human review moves anything, and further OTR findings-chasing is pointless -- the post is the mechanism. Either answer is worth the whole exercise. |
| `0.2.0` | `+ os.environ.get("OTR_SANDBOX_KNOB")` in the node | Is one env read `info`, and does one `info` finding flip a version to `Flagged`? | Confirms or refutes the "any finding = Flagged until human" reading that every OTR plan assumes. |
| `0.3.0` | replace with `subprocess.run(["ffmpeg", "-version"], shell=False)` behind `otr_proc`-style checks | Is a list-argv, allowlisted spawn `info` or `critical`? Is it tagged `command_injection` regardless of shape? | Decides whether the proc gateway can ever clear the rule or only annotate it -- i.e. whether the OTR-Lite "ffmpeg-less" idea (GO_FORWARD 4.X) is a registry requirement or a preference. |
| `0.4.0` | replace with an opt-in `urllib.request.urlopen` gated on an env flag, default off | Is default-off network `info`? Is the flag's presence read as `network_operations` or as `environment_manipulation` as well? | Tells us whether the 5 network findings on OTR are reducible by gating (no) or only by deletion. |
| `0.5.0` | `+ scripts/probe_helper.py` (stdlib, no patterns) with `scripts/*` and `!scripts/probe_helper.py` in `.comfyignore` | Does the publisher's zip honour the negation? (We believe yes from alpha.20; this is the two-minute proof that does not need a download of an 809-file zip.) | If no, three shipped TTS workers and the mesh stage are silently missing again on the next OTR publish. |
| `0.6.0` | `+ scripts/probe.ps1` containing only `Write-Host "probe"` | Is a `.ps1` scanned at all? (Then, ONLY if yes and ONLY if the operator rules it: a second `.ps1` with a `Invoke-WebRequest` line, to learn whether the pattern that shipped twice in OTR was ever seen.) | Decides whether PowerShell installers are a review surface or invisible to the reviewer. |
| `0.7.0` | `pycairo>=1.24` as an UNMARKED dependency | Does an sdist-only Linux dependency kill the extract and null the card? | Confirms or kills the pycairo theory for OTR's "N Nodes" without touching OTR. |

**The series can be cut after `0.1.0` and `0.2.0`.** Those two answer the
question the whole registry plan hangs on. `0.3.0`-`0.7.0` are cheap follow-ons
and each one is optional.

**Not in the series, and not to be added:** anything the registry BANNED us for.
No prohibited strings, no `exec`, no request-body-to-filesystem route, no widget
reaching `argv[0]`. The sandbox learns what is `info`; it does not test the
reviewer's patience with what is `critical`. A probe that would embarrass us in
a screenshot is not a probe.

## 4. Rules the sandbox inherits, verbatim

* **Never version-delete** (soft delete burns the string). At the end of the
  series the OPERATOR node-deletes the whole listing from the browser (hard
  delete, strings freed, no litter left on the registry). The publish token
  cannot delete (CLAUDE.md 7A).
* **Never bump while a version is Pending.** One probe in flight at a time.
  This is what makes the series serial and why section 3 is ordered.
* **Every result is READ, never inferred:** `include_status_reason=true`, two or
  three reads minutes apart (eventual consistency), copied into `PROBES.md`
  before the next bump.
* **Publishing is the operator's act.** Editing the sandbox's `pyproject.toml`
  fires a public publish; a coder window prepares the probe on a branch and the
  operator merges the version bump, exactly as with the real pack.
* **A sandbox result changes an OTR ruling only through `GO_FORWARD_PLAN.md`,
  with the probe version cited.** No "the sandbox showed X" in a commit message
  without the row.

## 5. Open design forks -- what the arc is for

1. **Serial single pack vs parallel N packs.** One listing, probes as versions,
   ~1 scan-cadence per answer (a week if the cron is nightly). Or seven listings
   published the same hour, all answers in one cron. Parallel is faster and
   leaves seven node ids under `fluxus` to hard-delete afterwards; it also loses
   the clean "same pack, one delta" comparison. **Driver's lean: serial for
   `0.1.0`/`0.2.0` (the pair must be same-pack to mean anything), then parallel
   for the optional tail if the cron proves slow.**
2. **Does `0.1.0` go out with the honest description, or a plainer one?** The
   description is scanned too (prohibited-string rule). "test", "probe",
   "sandbox" are not prohibited words as far as anything we have read shows, but
   nothing we have read is the rule list. **Lean: honest description; if it
   flags on the description alone, that is itself the first finding and worth
   knowing.**
3. **Where the ledger of results lives.** `PROBES.md` in the sandbox repo (with
   the code it describes) vs a section of this repo's `GO_FORWARD_ARCHIVE.md`
   (where every other receipt lives). **Lean: both -- the row in the sandbox is
   the primary record; the ARCHIVE gets the summary when a result lands, because
   readers of THIS repo will not go looking in another one.**
4. **Whether a local `comfy node publish --dry-run` / comfy-cli pre-scan makes
   `0.5.0` and `0.6.0` unnecessary.** If comfy-cli builds the same zip locally,
   the `.comfyignore` and `.ps1`-presence probes are answered offline and only the
   SERVER-side YARA questions need a publish. The arc should check comfy-cli's
   current CLI before those two rows are kept.
5. **Whether `0.7.0` is ethical to run at all** -- it deliberately publishes a
   version whose install FAILS on Linux. It is a no-op node nobody installs and
   the description says so, but a broken listing is still a broken listing.
   **Lean: run it once, hard-delete promptly, and only if the card question is
   still open after `0.1.0` (a populated card there already tells us the extract
   works for a trivial pack, which narrows the OTR question to "pycairo or
   kokoro's torch pull", still two suspects).**

## 6. DONE WHEN

* The arc has picked a shape for forks 1-5 and the decisions are written here.
* The sandbox repo exists with `0.1.0` prepared on a branch and the operator has
  merged the bump.
* `PROBES.md` carries the `0.1.0` row with a READ status and findings.
* `docs/GO_FORWARD_PLAN.md` item 1.4a's "needs zero findings or the manual
  review" sentence is replaced by whichever half `0.1.0` proved.

## 7. What this is not

* Not OTR-Lite (GO_FORWARD 4.X). Lite is a product; this is an instrument. Lite
  waits for v2 to ship; this can run tomorrow.
* Not a second copy of any OTR code. The moment a probe wants an OTR module, it
  has stopped being a probe.
* Not a way around the human review. If `0.1.0` proves zero findings is not a
  route to Active, the sandbox has done its job by saying so, and the answer for
  OTR is the review request, which is already posted.
