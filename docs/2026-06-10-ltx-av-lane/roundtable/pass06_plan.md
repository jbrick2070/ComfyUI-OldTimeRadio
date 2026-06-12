# LTX-AV lane -- sprint plan after pass06 (all but pre-mortem/finishing LOCKED)

> Campaign docs/2026-06-10-ltx-av-lane/. Judgments pass01-06. Remaining:
> pre-mortem, finishing. NO production code this window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video; I2V lip-sync from the FLUX still + per-beat audio) and
`ltx_av_music` (music_visual; audio-reactive scene motion). All shipped
engines untouched.

## ARCHITECTURE (pass01) -- summary

eng_ltx_av.py: shared core + 2 thin MotionEngineBase adapters.
ltx_av_talk (announcer_visual, character_video; audio_driven_face;
required text_prompt+audio_ref+init_image; fallback humo -> humo_1.7B ->
latentsync -> still_kenburns; degrade aspect change = LOUD policy).
ltx_av_music (music_visual; audio_conditioned_video NEW; required
text_prompt+audio_ref; fallback ltx_video -> still_kenburns). Both dark;
ONE flag OTR_ENABLE_LTX_AV (usability; registration is import-time);
IN_PROCESS; fps 25; Sage gate; AS-3 lease; BUG-291 reclaim; V-12;
executor-thread. assert_usable: flag -> Sage -> node gate (MISSING_MODEL
naming the class; six-reason enum pinned) -> weights (message names the
exact artifact: encoder vs transformer vs projection vs VAE) -> av_dims
on template (None tolerated). STOP rule: pip-freeze sandwich + nodes
resolve, else SIDECAR_REQUIRED + STOP. Config envs: OTR_LTX_AV_CKPT,
OTR_LTX_AV_TEXT_ENCODER (humo pattern).

## I/O CONTRACTS (pass02) -- summary

_ref_path(request.audio_ref) tolerant; asset_refs["init_image"];
base_clip_ref ignored; talk fails closed pre-render on missing inputs.
Core ffmpeg-normalizes audio to s16le/44.1k/mono (episode tmp).
T = timing.target_frame_count; next_8n1 snap-UP (legacy :281 snaps DOWN);
render = min(next_8n1(T), LTX_AV_MAX_FRAMES [M0; init 497]); trim to T /
pad-by-last-frame, LOUD "[ltx_av] pad-tail rendered=<n> target=<T>" >2s.
Graph ends at video VAEDecode -> IMAGE batch ->
encode_frames_to_silent_mp4 (-an); joint-AV strip + fake-AV test;
CanonicalClip silent/bt709/yuv420p/fps25/int frames + engine_id/family
stamps; ffprobe zero audio streams. canvas = request.canvas; av_dims
RAISES w/ nearest-valid (#347 doc-error note). init image: in-graph
ImageScale+crop COVER+center-crop; pad+outpaint = M0 cell.

## PROMPTS (pass03) -- summary

Adapter-thin (text_prompt/negative_prompt only; AST-tested no brief
imports). Driver: ltx_av_music JOINS the :418 tuple verbatim (radio
override honored); ltx_av_talk SIBLING branch, NO radio override,
fallback = character_description-or-"a 1940s radio announcer" + "head
and shoulders at a period microphone" (no speech verbs) via
finish_visual_prompt(240, style_tail=False) + no-text clause; forbidden:
dialogue/stage directions/vocatives/captions. NEGATIVE =
_LTX_DEFAULT_NEGATIVE verbatim (extension only on M0 inert evidence,
music-only). No music tail v1 (M0 cell). Cap 240.

## WIRING (pass04) -- summary

Universal master-slice already feeds line-backed beats; ONE delta:
line-less shots slice from SHOT synthetic timing iff engine_id ==
ltx_av_music. Hash/seed safety proven by tests (ShotLock-stamped hash,
_seed_from_hash). _render_one passes request_template=request (TypeError
guard). ENGINE_FAMILY += both; SYNTH_FALLBACKS += both (belt-and-braces).
Flag-off = RENDER-TIME degrade (ShotLock never asserts; registry
docstring corrected). Force map role-guarded (LOUD-ignore), never
bypasses asserts; M4 smoke env documented. Announcer portrait alias
(ltx_av_talk-gated; object id VERIFY). Identity: post-restamp engine_id,
trail keeps origin; greps = format_swap_log + manifest engine_id +
pad-tail marker. No group pruning (no providers).

## TESTING (pass05) -- summary

tests/test_av_dims.py (pure unit incl. snap-up + cap cases);
tests/test_video_ltx_av.py (mirrors test_video_humo.py: registration/
dark/role-fit/schema/order/extraction/canonicalize/identity/fake-AV/
pad-marker/AST/cold-import/ascii/chains/SYNTH membership; node gate via
mocked NODE_CLASS_MAPPINGS; request_template=None tolerated);
tests/test_ltx_av_driver_wiring.py (dark-lane GOLDEN FIXTURES; flag-off
degrade; force guard; alias; synthetic slice gating; template
pass-through; ENGINE_FAMILY; canvas; prompt gates). Fallout: retry
-taxonomy sweep gains both chains; exact enumerations -> membership;
b7 auto-covers (loop var `imp` if edited). Forgot-it matrix: every edit
-> named failing test. Byte-identical: CPU structural; DEDICATED
forced-lane master-hash = M4 GPU. Pytest = no network/CUDA/weights/
forwards. M0 sheet checked in; parser test post-M0; LTX_AV_MAX_FRAMES
== sheet (drift test). Bug Bible: BUG-070/291 pins; new dims row at
ship (Three-File Contract).

## HARDWARE (LOCKED pass06)

JUDGE-VERIFIED sizes (HF API 2026-06-10): Kijai 22B distilled-1.1
fp8_scaled 23.5 GiB; QuantStack distilled GGUF Q2_K 11.6 / Q3_K_S 13.0 /
Q3_K_M 13.7 / Q4_K_S 15.6 / Q4_K_M 16.5 / Q5_K_S 17.3 / Q5_K_M 18.1 /
Q6_K 19.6 / Q8_0 23.7 GiB; gemma_3_12B_it_fp8_scaled encoder 13.2 GB;
audio VAE 365 MB; video VAE 1.45 GB; taeltx2_3 23.5 MB; text_projection
2.3 GB; dynamic LoRA 2.7 GB; NVFP4 dev 21.7 GB.

- DEAD-ENDS: full residency under 14500 MB NVML is DEAD for Q4_K_S/
  Q4_K_M/Q5+/fp8_scaled -- offload/block-swap rows only. Full-resident
  candidates: Q3_K_S (~1.5 GiB headroom) and Q3_K_M (BORDERLINE).
  Total NVML decides, never file size.
- M0 TABLE columns: lane / artifact / file GiB / aux in phase / encoder
  placement / offload setting / NVML idle-preload-peak-sustained-post
  (MiB, probe_used_mb + mid-render assert_vram_within_ceiling) / wall
  (1472x832 x ~6s) / frames / quality vs 2B (A/B) / PASS-WARN-FAIL /
  notes. Rows: Q3_K_S resident, Q3_K_M resident, Q4_K_M offloaded,
  L1 fp8 block-swap; optional taeltx-vs-full-VAE cell; FLUX-ordering
  verification row (lease released + below-ceiling before video).
- PASS BARS: NVML peak+sustained <= 14500 MB; wall <= 10 min/clip
  PASS, 10-15 WARN (ship-able opt-in, documented), > 15 FAIL (parked).
  Episode (~3 clips/30w): <= 30 min PASS; > 45 min dead. Quality >= 2B.
- ENCODER PHASING (inside ONE lease; never release between phases):
  acquire AS-3 -> text encode -> reclaim_idle_models("ltx_av
  text-encode phase") [_soft_free insufficient] -> load transformer ->
  sample -> decode -> teardown reclaim -> release +
  wait_until_below_mb(14500). M0 measures GPU-encode-then-reclaim AND
  CPU-offloaded encode; v1 default = the passing mode (prefer GPU
  encode); GGUF Q3 encoder = contingency row only.
- SYSTEM RAM: sheet records RAM, pagefile, peak commit + working set
  per lane; paging -> wall blowup (wall gate catches); pre-M0: RAM >=
  32 GB required for any block-swap row; disk-free check before pulls.
- L3 NVFP4: CUT from M0 (dev-only steps, 21.7 GB, #11864); stretch
  column for later; never gates the decision.
- Two-stage: base-only v1 CONFIRMED. FLUX: sequential, verified by an
  M0 row, no co-residency.
- download_ltx_2_3.ps1 disk note bumped >= 24 GiB; GGUF/encoder pulls
  reuse the cache+symlink pattern. ComfyUI-GGUF pack presence = M0
  inventory row (Manager-installed custom pack, not pip; pip-freeze
  sandwich still binds; if it can't hold, L2 is blocked and
  Q3-resident via the native loader path or L1 is the lane).

## Additive touch list (consolidated)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py
- NEW  nodes/_otr_shared/av_dims.py
- NEW  tests/test_av_dims.py, tests/test_video_ltx_av.py,
       tests/test_ltx_av_driver_wiring.py (+ fixtures/ltx_av_dark/)
- EDIT nodes/_otr_video_engines/__init__.py; schemas.py; registry.py
       (docstrings); nodes/_otr_shared/role_compat.py;
       render_driver.py (deltas a-g + SYNTH_FALLBACKS)
- EDIT retry-taxonomy sweep + any exact-enumeration tests found
- EDIT scripts/download_ltx_2_3.ps1 (disk note) + sibling pull lines
       for GGUF/encoder as needed
- NEW  docs/2026-06-10-ltx-av-lane/M0_RESULTS.md (sheet)
- Docs/tracker + Bug Bible row at ship.

## Claims ledger / misreads -- see pass04/05 judgments (cumulative)

## Milestones

- M0 PROBE (GPU evening, AFTER the acceptance-test window; never
  concurrent with it): disk+pack inventory; node presence (Desktop +
  headless); pip-freeze sandwich; scratch IA2V render w/ real slice;
  output-audio hash probe; the M0 TABLE (above); prompt cells; P1
  matrix -> LIPSYNC | STYLIZED | INERT; INERT everywhere closes the
  lane with a finding.
- M1 ADAPTERS (CPU): eng_ltx_av dark + av_dims + schemas/role_compat/
  __init__/registry + driver deltas + 3 test files + goldens + fallout;
  suite + Bug Bible green.
- M2 GRAPH + LANE: winning-lane graph; pre-flight; lease+phasing;
  silent encode; trim/pad; max-frames pinned to sheet (drift test).
- M3 WIRING PROOF: slot asserts; flag-off degrade; force guard; alias;
  identity/manifest tests.
- M4 GATES: full suite + Bug Bible + byte-identical + DEDICATED
  forced-lane master-hash + live 30-word smoke (flag ON + force map);
  greps (swap-log, manifest engine_id, pad-tail, NVML <= 14.5); obs
  playable AAC only.
- M5 LOOK-QA + DOCS + Bug Bible row.

## Appendix: cut lanes

Yvann-Nodes (p01). New prompt/negative envs (p03). ASPECT_CHANGE kind +
group-prune wiring (p04). New usability reason / GPU pytest / framework
(p05). NVFP4-in-M0 (p06).

## Open questions (assigned)

- pass07 PRE-MORTEM: rank + mitigate the kill-list -- OOM mid-episode;
  fallback STORM (flag on, weights absent -> every beat degrades);
  model RELOAD THRASH between consecutive same-engine clips (does
  free_after_use force a 13-16 GiB reload per clip? batch ordering?);
  partial/corrupt downloads + broken symlinks; Comfy module-cache
  staleness (restart discipline Desktop vs headless); cancel mid-sample
  (zombie VRAM, lease stuck, executor-thread state); slice-cache key
  (frozen-master mtime); captions/credits/timeline interplay with
  pad-tail clips; Desktop node lag (#13194/#13308); golden-fixture rot;
  GPU contention with the ACTIVE acceptance render (hard schedule
  rule); NVML unavailable (nvml_available False -- ceiling unenforced?).
- pass08 FINISHING: convergence verdict + coder-window tickets
  (CW-LTX-AV-1..n) + M0 operator checklist + tracker row.
