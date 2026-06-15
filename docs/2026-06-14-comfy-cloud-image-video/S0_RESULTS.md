# S0 Spike — RESULT: BLOCKED (cannot run; missing prerequisite)

**Date:** 2026-06-14 (overnight autonomous window)
**Status:** ⛔ **BLOCKED at the HARD STOP — no adapters written (correct per CODER_KICKOFF.md).**
**Outcome:** the S0 auth spike could not be run, so per the kickoff
("if headless auth can't be made to work, STOP and leave me a note") the entire
cloud-engine build (S1–S5) is **not started**. No `cloud_*` adapter code exists.

---

## Why it's blocked (two independent blockers, both operator-resolvable)

### 1. `OTR_COMFY_API_KEY` is NOT set (the primary blocker)
Checked all three Windows scopes via `[Environment]::GetEnvironmentVariable(...)`:

| scope | `OTR_COMFY_API_KEY` set? |
|---|---|
| User | **No** |
| Machine | **No** |
| Process | **No** |

S0_SPIKE_AND_SPRINT.md §3 Part A step 1 is *"read `OTR_COMFY_API_KEY` from env"* —
with no key there is nothing to authenticate with, and the headless-auth gotcha
(§2: hidden `auth_token_comfy_org`/`api_key_comfy_org` are `None` under `/prompt`)
is exactly the thing the spike must defeat by passing an **explicit** key. No key ⇒
the spike cannot even begin, and the live billed calls (~$0.04 image + ~$0.10 clip)
were not made.

### 2. `comfy_api_nodes` not locatable for even a non-billed signature probe
The S0 Part-A signature probe (`import comfy_api_nodes.util.client`, pin
`sync_op`/`poll_op` signatures + the auth-arg name) does **not** need the key or a
billed call — I tried to run just that to give a head start. But ComfyUI Desktop
bundles its Python core inside the Electron app (the package is not greppable on
disk under `C:\Users\jeffr\AppData\Local\Programs\ComfyUI`, same as `nodes_flux.py`
/ `comfy_extras` earlier), so `comfy_api_nodes` is not importable from the standalone
`.venv` without the server's bootstrap/sys.path. The signature pin therefore has to
happen **inside a running ComfyUI process** (a throwaway debug node via `/prompt`,
exactly as §3 Part B describes) — which also needs the box booted and (for the live
half) the key. I did **not** boot the server unattended for this (box-collision risk
while you're asleep, and Part A's live call needs the key anyway).

**Follow-up (operator: "the Desktop version has the API key baked in"):** I checked.
The key IS present in ComfyUI Desktop, but it is stored **encrypted in the OS keychain**
(Electron `safeStorage` / Windows DPAPI), NOT in a readable file — `config.json`
(213 B) and `user/default/comfy.settings.json` (2444 B) contain **no** api-key /
comfy_org / token field. So the baked-in Desktop key cannot be read out as a plaintext
value to wire into `OTR_COMFY_API_KEY` for a headless OTR run, and decrypting/exfiltrating
an OS-keychain credential is not something I'll do. Also note the §2 gotcha: even running
inside the Desktop server's `/prompt`, the hidden `auth_token_comfy_org` is `None` — the
Desktop's stored key does NOT auto-feed a headless `util.client` call; OTR must pass an
**explicit** key. So the explicit `OTR_COMFY_API_KEY` value is still required from you.

**One-line unblock:** grab the key from comfy.org (Settings → API Keys) and
`setx OTR_COMFY_API_KEY "<key>"`, restart Desktop, re-run the S0 task.

---

## What you need to do to unblock (then the build is ready to go)

1. **Set the key** (User scope, so headless `/prompt` runs inherit it):
   `setx OTR_COMFY_API_KEY "<your comfy.org API key>"` (or set it in the ComfyUI
   Desktop launch environment). Restart ComfyUI Desktop so the server picks it up.
2. (Optional but recommended) set the episode credit ceiling:
   `setx OTR_CLOUD_CREDIT_CEILING "5"` (the planned ~$5 skip-to-floor default).
3. Re-run this window's S0 task. With the key present I will: run the gated spike
   (`OTR_RUN_LIVE_CLOUD_SPIKE=1`), pin the installed build's `util.client`
   signatures + auth-arg name + async-wrapper + output type, prove headless auth in
   the executor thread (Part B), write the real `S0_RESULTS.md` (secrets redacted),
   and only THEN build S1→S5 to the flux_gen1 quality bar.

Everything upstream of the key is ready: the plan (CODER_KICKOFF + S0_SPIKE_AND_SPRINT
+ WIRING_PLAN + the roundtable judgments) is internalized and the invariants are
understood (is_network lease-skip on both layers, cost reserve→commit→release-on-
failure, `assert_usable` = flag + key + `find_spec("comfy_api_nodes")`, torch-free
image path, atomic video write, V-12 cold-import, default-OFF behind `OTR_ENABLE_CLOUD`).
No code was written, so there is nothing to revert.

---

## What I DID ship this window instead (so the night wasn't idle)
- **BUG-LOCAL-411** flux lush-tint restore (FluxGuidance 3.5 + grade/radio still
  tails + bookend seed 4242 + grade on portraits for full flux consistency) —
  green + pushed.
- **BUG-LOCAL-412** LTX motion forensic **and** the restore: the LTX default is now
  the fast 8-step `distilled` + `euler_cfg_pp` recipe (downgraded from the slow
  30-step ksampler) — green + pushed (`21bfe7a`).
- Queued next: a local all-LTX 120-word soak (announcer/music/character) to exercise
  the restored LTX.
