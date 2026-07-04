# OTR Soak Runbook -- "a good soak, every time"

Canonical procedure for running an unattended OTR ComfyUI soak that (a) actually
produces episodes in `otr\obs` and (b) surfaces engine/config failures LOUDLY and
EARLY instead of churning them silently all night.

Grounded in the 2026-06-29 failure: a 9.5h soak ran **337 legs, all errored, 0
episodes** because the loop was trusted after only confirming the harness BOOTED
-- never after a leg SHIPPED. Root cause: `OTR_SOAK_ACT_COUNT=3` (the default)
with 80-word episodes -> `InvalidEpisodeBudgetError: act_count=3 exceeds max 1`
on every single leg.

---

## 0. PRIME INVARIANT (the one rule that would have saved last night)

A soak is NOT "running" until **one leg has shipped a real `*_final.mp4` into
`otr\obs` end-to-end**. Booting the server and seeing the harness print its
recipe is NOT verification. Watch the FIRST leg to a literal `obs_publish OK` +
the file on disk in `otr\obs` BEFORE launching the unattended loop.

**Verification gate (mandatory, every soak):**
1. Boot the server (Section 2). Confirm `:8000` is listening.
2. Run ONE leg. Poll the server log until you see, in order:
   `phase 2A budget: act_count=...` (cleared the budget node) ->
   `[OTR voice P-OBS] char_voice` (voicing) ->
   `[Video] ... Encode complete` -> `obs_publish OK`.
3. Confirm a fresh `*_final.mp4` exists in
   `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs`.
4. ONLY THEN launch the overnight loop. If the first leg errors, STOP and
   diagnose -- do not let the loop run.

The harness is error-tolerant by design (each leg is wrapped in try/except and
the loop keeps going), so a config bug fails EVERY leg the same way silently.
The error tolerance is a feature for one-off render failures and a TRAP for a
bad config -- the gate is what closes that trap.

---

## 1. THE EPISODE-BUDGET RULE (last night's actual bug)

Each act needs ~50 words. The validator rejects `act_count * 50 > target_words`:

| `OTR_SOAK_TARGET_WORDS` | max `OTR_SOAK_ACT_COUNT` | use |
|---|---|---|
| 30 - 120  | 1            | per-engine smoke (fast, many legs) |
| 150 - 250 | 2 - 3        | short stories |
| 400 - 700 | 3 - 5        | frontier-LLM stories |
| 1500 - 3120 | up to ~10  | long multi-act |

The harness default is `OTR_SOAK_ACT_COUNT=3`. **Any soak under ~150 words MUST
set `OTR_SOAK_ACT_COUNT=1`.** When in doubt, set act_count = max(1, target_words
// 60) and let the gate (Section 0) catch a mismatch on leg 1.

---

## 2. ENGINE ENABLEMENT (for a soak, the launcher is the switch -- not the dropdown)

In normal use the node-87 dropdown selects the engine. In a SOAK the harness
overrides the policy via `OTR_SOAK_*` env, and the engines still need their
opt-in flags set when the HEADLESS SERVER boots (flags are read once at boot).

- Launcher: `scripts\_otr_soak_server_launch.cmd <logfile>`. Default mode sets
  `OTR_ENABLE_HUMO=1`. It inherits any other `OTR_ENABLE_*` set in the parent
  shell before `Start-Process`. Boot UTF-8 (`PYTHONUTF8=1`) or prestartup
  crashes on an emoji glyph (~13s boot death).
- Default-ON (no flag needed): `visualizer`, `ltx_video` (opt-OUT: `=0` disables).
- Default-OFF (set the flag before boot): `OTR_ENABLE_HUMO`, `OTR_ENABLE_LTX_AV`,
  `OTR_ENABLE_WAN_I2V` (+`OTR_WAN_I2V_CKPT`), `OTR_ENABLE_WAN_TI2V`
  (+`OTR_WAN_TI2V_CKPT` +`OTR_WAN_TI2V_VAE_NAME`), `OTR_ENABLE_STILL_PARALLAX`,
  `OTR_ENABLE_CHARACTER_3D`, `OTR_ENABLE_MESH_STAGE`; images
  `OTR_ENABLE_ZIMAGE`, `OTR_ENABLE_FLUX2_KLEIN` (`flux_gen1` needs none).
- A missing checkpoint still HARD-fails the engine loudly (MISSING_MODEL) even
  with the flag set -- that is correct, not a soak bug.

---

## 3. THE CONFIG KNOBS (`OTR_SOAK_*`, read once at harness start)

- `OTR_SOAK_ANNOUNCER` / `OTR_SOAK_MUSIC` / `OTR_SOAK_BEATS` -- video engine per
  bookend / other-beats role.
- `OTR_SOAK_CHARACTER` / `OTR_SOAK_SCENE` / `OTR_SOAK_BG` -- per-role video
  (empty = inherit other-beats).
- `OTR_SOAK_BEATS_IMG` -- image engine (flux_gen1 / flux2_klein / z_image_turbo).
- `OTR_SOAK_CHAR_VOICE` -- indextts2 (cloned) / kokoro / bark.
- `OTR_SOAK_TARGET_WORDS`, `OTR_SOAK_ACT_COUNT` -- see Section 1.
- One harness invocation = ONE video/image/voice config across MANY writer legs
  (local + frontier rotation, varied creativity/seed). To vary the VIDEO model
  you run the harness again with a new config (or a wrapper loop), each against
  the same already-booted server.

---

## 4. THE COVERAGE MATRIX (the operator's intent)

Run these as successive harness invocations against one booted server, each
gated by Section 0 on its first leg:

1. **Per-video-model smoke** -- 30-120w, `act_count=1`, one config per engine:
   visualizer, ltx_video, humo_14B_169 (character), wan_ti2v, still_parallax
   (the working 3D). Proves "pick X -> X renders".
2. **Frontier stories** -- visualizer video, frontier LLM (the harness rotates
   them), indextts2 voice, 400-700w, `act_count=3`.
3. **Long multi-video** -- ~1500-3120w, higher `act_count`, mixed video engines.
4. **Image variety** -- rotate `OTR_SOAK_BEATS_IMG` across flux_gen1, flux2_klein,
   z_image_turbo over the above.

---

## 5. FAILURE MODES + EARLY CATCHES

- **Budget mismatch** -> `InvalidEpisodeBudgetError` on leg 1 (Section 1).
- **Opt-in engine not enabled** -> `EngineUnusable ... gated_by_flag` (Section 2).
- **Missing checkpoint** -> `MISSING_MODEL` (install/verify on GPU box).
- **VRAM over-subscription** -> a leg that crawls (one 19s line taking minutes):
  3 co-resident heavies (gemma via Ollama + Mistral-Nemo + indextts2/humo)
  exhaust 16GB -> WDDM spill to system RAM. Watch for a leg whose wall time is
  10x its peers; prefer freeing the writer LLM before voice/render.
- **Voice swap** -> `[eng_kokoro] LOUD voice swap`: a non-kokoro bank id (vz_*)
  reached the kokoro slot, or a kokoro voice .pt is missing. Use indextts2 for
  cloned char voices; verify the kokoro pack is installed for the announcer.
- **Stale `pending_*` dirs** -> pytest fixtures + interrupted legs leave them;
  they are NOT shipped episodes. Count `otr\obs\*_final.mp4`, not pending dirs.

---

## 6. MONITORING + THE MORNING REPORT

- Harness writes per-leg JSONL + a SUMMARY json (legs / success / elapsed). A
  healthy soak shows `success` climbing and `outputs` non-empty.
- Quick health check any time: `otr\obs\*_final.mp4` count rising; the soak log's
  `[soak]   -> success (...) outputs=[...]` lines (NOT `-> error ... outputs=[]`).
- A scheduled morning task should read `otr\obs` + `otr\episodes` + the ComfyUI
  log and report what shipped / what errored / which combos failed.

---

## 7. ONE-SHOT CHECKLIST (run top to bottom, every soak)

1. Box clean: `:8000` free, GPU at desktop baseline (selective kill, never a
   blanket `Stop-Process -Name python` -- that severs the MCP pythons).
2. Set every `OTR_ENABLE_*` for the engines this soak will touch (Section 2).
3. Boot server via the .cmd (UTF-8); confirm `:8000` listening + log "Starting server".
4. Set `OTR_SOAK_*` for config #1; set `OTR_SOAK_ACT_COUNT` to fit the words.
5. Launch ONE leg; GATE on a real `otr\obs` ship (Section 0).
6. Only then launch the loop / next configs.
7. Spot-check `success` count + obs file count within the first ~2 legs.
