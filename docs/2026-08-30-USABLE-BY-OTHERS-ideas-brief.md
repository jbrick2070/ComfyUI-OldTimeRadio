# Ideas brief — can anyone other than the author actually use OTR?

**This is an IDEATION round, not a review.** There is no plan to critique. The
driver wants approaches, and would rather have one uncomfortable idea than five
safe ones.

**Context:** OTR is going open source. It works — nine published episodes on an
8 GB card, dozens on 16 GB, five 5-act episodes today alone. The question is
whether that survives contact with somebody who is not its author.

Everything below was measured today, 2026-08-30, mostly by trying to stand it up
on a rented GPU and failing repeatedly.

---

## The evidence that a stranger currently cannot install this

**1. The registry entry is FLAGGED with ZERO Active versions.**

    2.0.0-alpha.14   NodeVersionStatusFlagged
    2.0.0-alpha.13   NodeVersionStatusFlagged
    versions: 2      active: 0

`latest_version` resolves to null, so ComfyUI-Manager cannot install OTR by any
route — `@latest` has no target, and the `nightly` git path is refused by
Manager's `security_level` on any network-exposed instance. **This is a total
distribution outage through the standard channel**, not a pod problem. Flagged
does not self-clear the way Pending does.

**2. Even if it were Active, the install would be incomplete.** `.comfyignore`
excludes the entire `scripts/` tree, so a registry install ships the nodes but
NOT `otr_canonical_api_run.py` — the headless runner. A user who installs
"correctly" cannot drive it from the command line at all.

**3. The reason `scripts/` was excluded is contradicted by our own record.** It
was removed to appease the security scanner, but the same file notes:

    alpha.8    ACTIVE    119 files under scripts/, 28 .ps1/.bat/.cmd
    alpha.12   FLAGGED     0 files under scripts/,  0 .ps1/.bat/.cmd

Removing them did not clear the flag. We do not actually know what is flagged.

**4. Eight manual terminal pastes** were needed to get one pod rendering today —
clone, deps, weights, a path correction, ffmpeg check, a pull, and two retries.
An agent with full HTTP access to the machine could not do it.

**5. `scripts/setup_cloud.sh` exists and is unused.** It already detects
`COMFY_ROOT`, `CUSTOM_NODES` and `MODELS_ROOT` properly. The driver hand-derived
those paths by trial and error today without knowing it was there.

**6. Silent wrong-platform behaviour.** `_models_root()` falls back to
`C:\ComfyUI-Models`. On Linux that becomes a literal directory of that name, so
~4 GB of weights download "successfully" somewhere ComfyUI never scans, and the
tool reports OK.

**7. A capability test that lied.** `has_nvenc` checked whether ffmpeg was
COMPILED with nvenc rather than whether nvenc could RUN, so every container
without the NVIDIA encode library died 7-18 minutes into a render with a broken
pipe. It existed in TWO independent copies; fixing one left the other. (Fixed
today.)

**8. Partial installs report success.** `__init__.py` loads each node in its own
try/except by design, printing `Skipped '<name>': <reason>`. A user missing a
dependency gets a working-looking pack with silently absent nodes.

**9. `main` is stale.** It sits thousands of commits behind and advertises
`version = "1.0.0"`. A fresh `git clone` with no branch argument gets v1 code.

**10. Tests run on one machine.** The second developer box cannot execute the
suite at all — no pytest, and the runner path in CLAUDE.md describes the author's
machine. Every "12,474 passed" in this project means "passed on the 5080."

---

## What the driver wants ideas about

1. **What is the shortest honest path from "someone has a GPU" to "they have an
   episode"?** Name it concretely. Assume they will not read a 5,000-character
   README and will give up after two failures.
2. **What do we do about FLAGGED**, given we cannot see the scanner's findings
   and cannot ask it? Is there a way to bisect what it objects to, or a
   distribution channel that routes around the registry entirely?
3. **What is the right first-run experience?** A doctor command, a self-check
   node, a one-shot script, a template — or something none of us have named.
4. **Which of the ten items above actually blocks adoption, and which just look
   bad?** The driver's ranking is untrustworthy: he has never installed this
   software as a stranger and cannot.
5. **What is the cheapest thing that would have caught 6, 7 and 8 before a
   user did?** All three are "works on the author's box, silently wrong
   elsewhere."
6. **Uncomfortable question, answer it honestly:** is a ComfyUI node pack even
   the right distribution shape for something that writes, voices, scores and
   renders a whole episode? What else could it be?

## Constraints that bound any idea

* **100% local, offline-first.** No cloud services, no API keys, no paid
  services in the shipped path.
* **No gates.** A guard that refuses a render is forbidden by standing operator
  rule; an OOM is the only acceptable killer.
* **8 GB is the floor** and must stay supported — it is the machine that proves
  portability.
* One author, no support team.
