# Problem statement — get OTR onto a machine with nobody typing into a terminal

**Driver:** Claude (Cowork, 5080). Written after a full session of failing at
this, so every constraint below is measured, not assumed.

## The problem, in one line

An AI agent can reach a rented ComfyUI pod over HTTP and do almost everything —
but it could not install this pack, and a human had to paste shell commands into
a browser terminal roughly eight times.

## What the operator wants

> *"How do we get the commands you show me to run via AI or automatically in the
> template?"*

The deliverable is a design, not code: **the cheapest reliable path from "a bare
pod exists" to "OTR is installed and rendering", with zero human terminal
input.** Voice/agent-driven and template-driven are both on the table.

---

## MEASURED constraints (all verified today, 2026-08-30, two different pods)

**What an agent CAN do over plain HTTP, no auth, no shell:**

    GET  /system_stats                    which GPU, VRAM, ComfyUI version, argv
    GET  /object_info                     the only real proof a pack loaded
    POST /api/manager/queue/install       install a node pack
    POST /api/manager/queue/install_model install a model
    POST /api/manager/queue/start         run the queue
    POST /api/manager/reboot              restart (returns 502; works anyway)
    POST /prompt                          submit a render

**PROVEN:** ComfyUI-AnimateDiff-Evolved was installed on a pod with **zero
terminal input** — 1036 -> 1181 node classes, 143 `ADE_` registered. The
mechanism works. Payload:

    {"id": "<cnr-id>", "version": "latest", "selected_version": "latest",
     "channel": "default", "mode": "remote"}

`version`, `channel`, `mode` are read as `json_data['...']`, not `.get()` —
omitting any is a 500.

**What BLOCKED the same route for OTR — four independent walls:**

1. **Our registry entry is FLAGGED, and there is no Active version.**

       2.0.0-alpha.14   NodeVersionStatusFlagged
       2.0.0-alpha.13   NodeVersionStatusFlagged
       versions: 2      active: 0

   `@latest` has nothing to resolve to. Flagged does not self-clear the way
   Pending does. **This breaks Manager installs for every user, not just pods.**

2. **The `nightly` (git URL) branch is refused by policy.** Manager routes it
   through `get_risky_level(git_url, pip)` then `is_allowed_security_level(...)`
   and returns `404 "A security error has occurred"` on a network-exposed
   instance. Lowering `security_level` requires editing Manager's config, which
   requires the shell we are trying to avoid — and is a security downgrade on a
   publicly reachable port.

3. **Every other remote door is credential-walled.** JupyterLab returns
   `403 Forbidden` on `/api/terminals`, `/api/contents` and `/api/status`
   without its token; FileBrowser returns 401. RunPod's web terminal is **not**
   on a pod-proxied port at all — `<podId>-19123.proxy.runpod.net` returns 404
   on every path, confirmed on two pods. It exists only inside RunPod's console.

4. **Even when install succeeds, the weights land in the wrong place on Linux.**
   `scripts/otr_fetch_lane_weights.py` resolves its destination through
   `_models_root()`, which defaults to `C:\ComfyUI-Models`. On Linux that
   becomes a literal directory of that name, so ~4 GB downloads "successfully"
   somewhere ComfyUI never scans. This cost a full false diagnosis today — the
   first pod's checkpoint list stayed empty and it was read as a failed fetch.

---

## Candidate designs (the driver's list — the panel should break, merge or add)

**A. Template start command.** Runs as container init, so it never meets
Manager's security policy and does not care that we are Flagged. Idempotent
clone + pip + fetch, then exec the image's normal entrypoint.
*Risk:* it REPLACES the default entrypoint — get that wrong and the pod boots
and serves nothing, silently. Requires knowing the base image's real start.

**B. Fix the registry flag.** Root cause for every user. Turns the whole
install into one POST that already works. Not in our control on our timetable.

**C. Hand the agent a Jupyter token once.** One paste buys a persistent shell
for the pod's whole life. *Risk:* a credential, and per-pod.

**D. Custom Docker image with the pack baked in.** True zero-step.
*Risk:* most work; goes stale unless it clones at boot anyway, which is A.

**E. An OTR-side self-install path** — e.g. something already-installed that
can fetch the rest. Circular for a first install; may help updates.

## Questions for the panel

1. **Is A actually sufficient**, or does the entrypoint-replacement risk make it
   worse than it looks? What is the safe pattern for chaining to an unknown
   image entrypoint?
2. **Is there a door we did not try?** ComfyUI's `/api/userdata` write surface,
   `extra_model_paths.yaml`, RunPod's own REST API, a CNR mirror, anything.
3. **What is the correct fix for constraint 4** — env var at deploy, a Linux
   default in `_models_root()`, or reading ComfyUI's `folder_paths` directly?
   This is a real portability bug independent of the install question.
4. **Ranking:** given Flagged is not on our timetable, what is the ordering that
   gets a stranger from bare pod to rendering episode fastest?
5. **What breaks first** when a non-author tries this?

## Out of scope

Story quality, the visual authored-path bug, PBUG-11, PBUG-20, the prefs guide.
