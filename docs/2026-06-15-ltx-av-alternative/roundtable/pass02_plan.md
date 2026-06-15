# LTX AUDIO-INPUT (A2V) ALTERNATIVE PATH -- CODE-READY (pass02, panel-hardened)

> pass01 (Claude synthesis) -> live panel pass02 (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4, $0.13) + Claude
> grounding. 15 must-fixes folded. Builds on the converged `docs/2026-06-10-ltx-av-lane/LTX_AV_SPRINT_PLAN.md`
> (its line refs are STALE vs HEAD `9633e1e` -- re-ground before coding). Lane A = prod NOW; Lane B = DARK,
> prove-it-or-park.

## DECISIONS (judge)
1. **Lane B is gated behind an M0 GRAPH SPIKE -- probe-or-park.** Do NOT hardcode any recipe yet. The existing
   "distilled v1.1 / 8-step" is the **2B TEXT** recipe in `eng_ltx_video.py`, NOT the LTX-2.3 **A2V** topology.
   The A2V model is 22B-class; fp8/distilled-22B are likely DEAD under 14.5 GB -> M0 ranks GGUF Q3_K_S/Q3_K_M,
   CPU-offload of the Gemma encoder, and block-swap (NVFP4 is CUT -- exceeds 16 GB). **M0 captures, from the
   official LTX-2.3 ComfyUI template + a live `/object_info`: the exact A2V node classes (audio loader,
   conditioning, sampler, and the VIDEO-ONLY decode/separation node), the viable low-VRAM artifact, the VAE
   decode floor at 384x216 / 512x288, and the peak NVML on THIS 5080. OOM/thrash -> PARK, write the finding,
   Lane A stands.**
2. **Sequencing:** No GRAPH / heavy-render code until M0 GO. CPU-only skeleton + wiring + tests are allowed
   before M0.
3. **2 adapters, ONE module-level lazy singleton core** (`_ltx_av_core`, lock-guarded) so talk + music share
   ONE resident model load -- independent cores would load twice and blow VRAM. The core obeys the AS-3
   single-resident lease + BUG-291 reclaim + a below-ceiling NVML check AFTER each clip.
4. **Frozen-audio V-1 (hard gate):** OTR's engine return has NO audio path and the encoded clip has ZERO
   audio streams (ffprobe-asserted, mirrors Wan M7). The video-only decode uses the **M0-captured** node
   class (node-gated fail-closed) -- NOT a preselected `LTXVSeparateAVLatent` (ungrounded). Any M0 research
   probe of LTX's own audio happens OUTSIDE OTR; no OTR milestone hashes LTX-generated audio.
   `test_audio_byte_identical` stays GREEN every milestone.
5. **Boomerang stays OUT of Lane B entirely** -- the audio defines the clip length; mirroring desyncs it.
   Implement NO boomerang code in `eng_ltx_av.py`; do NOT touch `OTR_LTX_LOOP_VIA_REVERSE`,
   `_LOOP_VIA_REVERSE_DEFAULT`, or `eng_ltx_video.py`.

## ARCHITECTURE (A)
ONE file `nodes/_otr_video_engines/eng_ltx_av.py`: module-level lazy `_ltx_av_core` (V-12 lazy heavy imports,
the M0-captured A2V graph, the video-only decode) + two thin `MotionEngineBase` adapters. `ltx_av_talk`
(roles announcer_visual+character_video; family `audio_driven_face`; req text_prompt+audio_ref+init_image;
fallback humo->humo_1.7B->latentsync->still_kenburns) and `ltx_av_music` (role music_visual; family NEW
`audio_conditioned_video`; req text_prompt+audio_ref; fallback ltx_video->still_kenburns). Dark
(`default_roles ()`), one flag `OTR_ENABLE_LTX_AV`, `@register` unconditional. ISOLATION_IN_PROCESS + the
cu130 freeze-identical STOP rule (-> SIDECAR if deps shift). Heavy forward in the EXECUTOR THREAD. Additive
only -- ltx_video/humo/latentsync UNTOUCHED.

**`assert_usable` ordered gate** (EngineUsabilityReason values ONLY, no new reason): 1 flag (GATED_BY_FLAG);
2 Sage (BUG-070); 3 NVML REQUIRED -> fail CLOSED (heaviest lane); 4 node gate -- every required class (the
M0-captured set) resolves in NODE_CLASS_MAPPINGS via LAZY read -> MISSING_MODEL naming the classes; 5 weights
-- resolved REALPATH exists + size >= per-artifact floors, message names the artifact; 6 av_dims on
request_template.canvas (None tolerated), violations re-raised as EngineUnusable (no raw ValueError).

## WIRING (B)
- **schemas.py:** `FAMILIES += "audio_conditioned_video"` + a `FAMILY_REQUIRED_INPUTS` entry; role_compat
  supplies `audio_ref` for music_visual. (Registry metadata alone is NOT enough -- breaks role/schema tests.)
- **Registry:** `@register` + CAPABILITIES row (vram_class heavy, vram_estimate from the M0 probe).
- **Prod JSON audit FIRST:** locate the `OTR_VideoDirector` per-role dropdown structure in
  `otr_scifi_16gb_full.json`; CONFIRM options are registry/roles-driven (so flag-off stays dropdown-VISIBLE,
  degrades only at render). If options are static arrays -> add the option + re-validate
  (OTR_WorkflowValidator + link/widget audit) in the SAME commit (CLAUDE.md); if dynamic -> NO JSON edit.
  NO new widgets (V-11).
- **Force-map (M3 smoke):** explicit `announcer_visual=ltx_av_talk,character_video=ltx_av_talk,music_visual=ltx_av_music`
  (NOT a `*` wildcard).
- **Audio input:** reuse the per-beat frozen-master slice; the slice fed to the model is padded/trimmed to
  the **8n+1** duration BEFORE generation (not post-hoc in canonicalize -- else lip-sync drifts); stage under
  a shot/seed name; never re-encode the master.
- **Fallback re-ground:** verify humo / humo_1.7B / latentsync / still_kenburns exist in the registry + the
  `_otr_shared/fallback.py` resolver (CW-7); update the chain tests. Re-ground all stale sprint-plan line refs
  against HEAD `9633e1e` (`render_driver.py` etc.) before any delta.

## BUGS/RISKS (C)
- **VRAM (dominant):** 22B busts 16 GB; M0 ranks GGUF-Q3 / offload / block-swap; the singleton core + AS-3
  lease + reclaim hold; NVML REQUIRED fail-closed; Stage-2 upscale OUT for v1 (reuse OTR composite/upscale).
- **VAE decode floor + 8n+1:** mirror `eng_ltx_video`'s `_ltx_frame_length` floor/raise + the SAME 8n+1
  snap-DOWN (`((n-1)//8)*8+1`) -- do NOT diverge Lane A vs B frame math; M0 probes the actual A2V VAE floor.
- **OOM recovery:** explicit teardown (forced `reclaim_idle_models` before release) + a restart rule in the
  engine docstring/error so a wedged clip can't hold VRAM.
- determinism seed-keyed; cold-import + AST no-heavy-import test (V-12); UTF-8/SFW.

## POLISH (D) / GRADUATION
Stage-2 upscaler deferred to v2. **Graduation bar (M4):** character LIP-SYNC vs HuMo -- operator A/B on
IDENTICAL audio/still/seed -- PLUS N short clips with no OOM. Promote dark->selectable ONLY if it clearly
beats HuMo at acceptable wall/VRAM. A MINIMAL motion-sanity check only for announcer/music (the full
optical-flow gate is unnecessary there -- ltx_video already covers motion). README "what each engine gives"
ONLY after M4 measures Lane B (no lip-sync claims before).

## TICKETS (each chunk: suite + Bug Bible + audio-byte-identical green)
- **M0 GRAPH SPIKE (probe-or-park, NO engine code):** official template + `/object_info` -> capture A2V node
  classes (audio loader / conditioning / sampler / video-only decode), pick the low-VRAM artifact, probe the
  VAE floor + peak NVML at 384x216 & 512x288 / batch1 / 4-6 s / Gemma offloaded. Write the GRAPH SPEC. OOM ->
  PARK.
- **M1:** `eng_ltx_av.py` skeleton -- singleton core + 2 adapters + the ordered assert_usable + schemas.py
  family + CAPABILITIES row + CPU/cold-import/AST tests. (CPU only; uses the M0 GRAPH SPEC node names.)
- **M2:** frozen-audio V-1 -- video-only decode (M0 class) + the no-audio-path / zero-audio-stream ffprobe
  test + byte-identical green; the 8n+1-padded audio-slice conditioning input.
- **M3:** wiring -- schemas/role_compat/fallback re-grounded; prod-JSON audit-then-(maybe)-edit; the explicit
  force-map smoke.
- **M4:** graduation soak -- lip-sync-vs-HuMo A/B + N-clip no-OOM. Promote or park.

## CUTS (panel consensus)
NVFP4 path (exceeds 16 GB); audio-reactive ledger->prompt verbs (separate Lane-A ticket, not needed to
prove/park Lane B, risks the prod look); slice-cache `mtime_ns+size` + storm-line counts + announcer-portrait
ledger alias (separate defects, out of Lane-B scope); the optical-flow/framediff HARD gate for
announcer/music (keep only the lip-sync-vs-HuMo gate + a motion sanity).

## INVARIANTS
test_audio_byte_identical GREEN (V-1); single heavy resident proven on 16 GB with margin; 100% local;
determinism; LOUD fallbacks; UTF-8 no BOM; SFW; additive-only; no new static widgets (V-11); JSON changes in
`otr_scifi_16gb_full.json` same commit + re-validate.
