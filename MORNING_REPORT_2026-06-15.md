# Morning Report — 2026-06-15 (overnight autonomous window)

**Branch `v2.0-alpha`, HEAD `1976842` == origin.** Suite **4265/0**, Bug Bible **16/7/3**
green throughout. Four threads; everything code is committed + pushed. One live soak
is still running (see §4).

---

## 1. BUG-411 — flux "lush" tint restored (DONE, pushed)
The 6/5 image pipeline rewrite had dropped the look levers. Restored in 3 green chunks:
- `bd1fbb2` — **FluxGuidance node @ 3.5** wired pos-CLIP→guidance→KSampler in `flux_gen1`
  (env `OTR_FLUX_GUIDANCE`). The biggest factor; global to every flux render.
- `cdc1411` — **cinematic grade + radio broadcast-distress still tails** + **deterministic
  bookend seed 4242** (env `OTR_RADIO_BOOKEND_SEED`).
- `1eb5c78` — **grade tail extended to PORTRAITS** so ALL flux PNGs match (a flux_still beat
  standing in for HuMo now reads with the same 6/5 grade).
- **You do:** restart ComfyUI Desktop, render, A/B the bookend vs
  `output\otr\episodes\signal_lost_melting_glass_pressure_20260605_093330\stills\radio_bookend_*.png`.
  Bump `OTR_FLUX_GUIDANCE` 3.5→4.0 if you want it richer.

## 2. BUG-412 — LTX motion restored to the fast 8-step recipe (DONE, pushed `21bfe7a`)
Forensic diff (5/09 `l001` + 5/28 `b001` good bookends vs current) found the cleanbreak
`70d379b` deleted the proven recipe and the new engine shipped slow/flat defaults. **Per your
call ("make LTX as it was; 30-step too slow → 8-step"), flipped the defaults:**
- `OTR_LTX_SAMPLER` default → **`distilled`** (8-step, FAST). 30-step ksampler is now opt-in
  via `OTR_LTX_SAMPLER=ksampler`.
- `OTR_LTX_SAMPLER_NAME` default → **`euler_cfg_pp`** (the documented dynamic-motion CFG++
  sampler the refactor had left on plain `euler`).
- **Held back (canvas-dependent, needs your eyeball):** I2V strength stays 1.0 — the old 0.75
  is probe-proven *mush* at the current 1472×832 canvas (the good 0.75 clips were 832×480);
  and the boomerang-loop / audio-length restore. Both are easy follow-ups once you confirm the
  new default looks right. Full forensic: `BUG_LOG_2026-06.md` BUG-412.

## 3. Comfy-Cloud engines — BLOCKED at S0, NO adapters written (correct per hard-stop)
The kickoff said run the S0 auth spike first and STOP if headless auth can't work. It can't yet:
- **`OTR_COMFY_API_KEY` is not set** in any scope (User/Machine/Process).
- The key IS in ComfyUI Desktop but stored **encrypted in the OS keychain** (DPAPI), not in any
  readable config file — so it can't be wired into a headless env, and I won't extract an
  encrypted credential.
- **You do (one line):** grab the key from comfy.org → Settings → API Keys, then
  `setx OTR_COMFY_API_KEY "<key>"`, restart Desktop, and re-run the cloud task. The whole plan
  is internalized and ready; the moment the key is present I run the spike and build S1→S5 to
  the flux_gen1 quality bar. Detail: `docs/2026-06-14-comfy-cloud-image-video/S0_RESULTS.md`.

## 4. All-LTX 120-word soak — LAUNCHED + RUNNING (your "for good measure" ask)
Forces `ltx_video` onto **announcer + music + character** beats (sanctioned profile
`role_overrides`, not a raw patch) at 120 words, exercising the restored 8-step euler_cfg_pp
recipe on every beat. Up to 3 episodes / 4.5h, then it stops clean.
- **Headless server:** UP on `:8000` (LTX lane, HuMo OFF). Boots clean; episode 1 was writing +
  voicing healthily when I left it (writer 176w, indextts2 voices, no errors).
- **Where to look in the morning:**
  - verdicts: `scripts/_otr_ltx_allroles_soak_summary.json`
  - server log: `scripts/_otr_ltx_soak_server_20260614_234303.log`
  - episodes: `output/otr/episodes/...` + finals in `output/otr/obs/`
- **What to eyeball:** (a) did the openers/bookends get their MOTION back (vs the flat 30-step
  era); (b) do the **character beats** render on LTX, or do they fall back to a still? — grep the
  server log for `LOUD`/`fallback`. If they fall back, that's the expected next step: `ltx_video`
  needs `character_video` added to its `roles` (it's currently announcer/music/scene only). You
  flagged this ("character LTX beats need adjustment; improve the LTX audio-input versions") —
  the LTX-AV (audio-conditioned) lane is the real path for lip-synced character LTX and is still
  parked.
- **IMPORTANT:** the soak server stays RESIDENT on :8000 after it finishes (no teardown by
  design). **Reset the box** (kill the soak python + the :8000 server, confirm GPU baseline)
  before any other GPU work. To stop the soak early: kill the `_otr_ltx_allroles_soak.py` python
  and the headless `main.py` server.

---

## Quick status table
| thread | status | ref |
|---|---|---|
| flux lush-tint (411) | shipped, pushed | `bd1fbb2`/`cdc1411`/`1eb5c78` |
| LTX 8-step restore (412) | shipped, pushed | `21bfe7a` |
| cloud engines | BLOCKED on `OTR_COMFY_API_KEY` | `S0_RESULTS.md` |
| all-LTX soak | running locally | `_otr_ltx_allroles_soak_summary.json` |

All operator-gated items are visual A/B (flux bookend, LTX motion) + the one-line cloud key.
Nothing is half-merged; every code change is green + pushed.
